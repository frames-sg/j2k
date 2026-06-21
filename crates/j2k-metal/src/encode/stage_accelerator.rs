// SPDX-License-Identifier: Apache-2.0

#[cfg(target_os = "macos")]
use crate::compute;
use j2k::adapter::encode_stage;
#[cfg(target_os = "macos")]
use j2k::{EncodeBackendPreference, J2kLosslessEncodeOptions};
#[cfg(target_os = "macos")]
use j2k_core::PixelFormat;

#[cfg(target_os = "macos")]
use super::{
    borrow_padded_metal_buffer_from_bytes, encode_resident_ht_tile_body_with_cpu_packetization,
    lossless_options_for_resident_htj2k_tile_job, should_use_resident_htj2k_host_tile_for_auto,
    MetalEncodeInputStaging, MetalLosslessEncodeTile,
};

/// Encode-stage accelerator for JPEG 2000 Metal work.
///
/// The type is wired into the public J2K encode-stage interface and reports
/// dispatches for each required encode stage.
#[derive(Debug, Clone)]
pub struct MetalEncodeStageAccelerator {
    dispatch_stages: MetalEncodeDispatchStages,
    parallel_cpu_code_block_fallback: bool,
    auto_host_output_force_cpu_fallback: bool,
    deinterleave_attempts: usize,
    forward_rct_attempts: usize,
    forward_dwt53_attempts: usize,
    forward_dwt97_attempts: usize,
    tier1_code_block_attempts: usize,
    ht_code_block_attempts: usize,
    packetization_attempts: usize,
    deinterleave_dispatches: usize,
    forward_rct_dispatches: usize,
    forward_dwt53_dispatches: usize,
    forward_dwt97_dispatches: usize,
    tier1_code_block_dispatches: usize,
    ht_code_block_dispatches: usize,
    packetization_dispatches: usize,
}

impl Default for MetalEncodeStageAccelerator {
    fn default() -> Self {
        Self {
            dispatch_stages: MetalEncodeDispatchStages::ALL,
            parallel_cpu_code_block_fallback: false,
            auto_host_output_force_cpu_fallback: false,
            deinterleave_attempts: 0,
            forward_rct_attempts: 0,
            forward_dwt53_attempts: 0,
            forward_dwt97_attempts: 0,
            tier1_code_block_attempts: 0,
            ht_code_block_attempts: 0,
            packetization_attempts: 0,
            deinterleave_dispatches: 0,
            forward_rct_dispatches: 0,
            forward_dwt53_dispatches: 0,
            forward_dwt97_dispatches: 0,
            tier1_code_block_dispatches: 0,
            ht_code_block_dispatches: 0,
            packetization_dispatches: 0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MetalEncodeDispatchStages(u8);

impl MetalEncodeDispatchStages {
    const DEINTERLEAVE: Self = Self(1 << 0);
    const FORWARD_RCT: Self = Self(1 << 1);
    const FORWARD_DWT53: Self = Self(1 << 2);
    const FORWARD_DWT97: Self = Self(1 << 3);
    const TIER1_CODE_BLOCK: Self = Self(1 << 4);
    const HT_CODE_BLOCK: Self = Self(1 << 5);
    const PACKETIZATION: Self = Self(1 << 6);
    const AUTO_HOST_OUTPUT: Self = Self(0);
    const ALL: Self = Self(
        Self::DEINTERLEAVE.0
            | Self::FORWARD_RCT.0
            | Self::FORWARD_DWT53.0
            | Self::FORWARD_DWT97.0
            | Self::TIER1_CODE_BLOCK.0
            | Self::HT_CODE_BLOCK.0
            | Self::PACKETIZATION.0,
    );

    fn contains(self, stage: Self) -> bool {
        self.0 & stage.0 != 0
    }

    fn without(self, stage: Self) -> Self {
        Self(self.0 & !stage.0)
    }
}

impl MetalEncodeStageAccelerator {
    /// Create an accelerator that leaves forward RCT on the CPU path.
    pub fn with_cpu_forward_rct() -> Self {
        Self {
            dispatch_stages: MetalEncodeDispatchStages::ALL
                .without(MetalEncodeDispatchStages::FORWARD_RCT),
            ..Self::default()
        }
    }

    /// Create the conservative automatic accelerator for host codestream output.
    pub fn for_auto_host_output() -> Self {
        Self {
            dispatch_stages: MetalEncodeDispatchStages::AUTO_HOST_OUTPUT,
            parallel_cpu_code_block_fallback: true,
            ..Self::default()
        }
    }

    /// Create an accelerator that only attempts the HT code-block stage on Metal.
    pub fn for_ht_code_block_encode() -> Self {
        Self {
            dispatch_stages: MetalEncodeDispatchStages::HT_CODE_BLOCK,
            parallel_cpu_code_block_fallback: true,
            ..Self::default()
        }
    }

    #[cfg(all(test, target_os = "macos"))]
    pub(super) fn for_forward_dwt97_encode() -> Self {
        Self {
            dispatch_stages: MetalEncodeDispatchStages::FORWARD_DWT97,
            ..Self::default()
        }
    }

    #[cfg(target_os = "macos")]
    pub(super) fn for_host_output(options: J2kLosslessEncodeOptions) -> Self {
        if options.backend == EncodeBackendPreference::Auto {
            Self::for_auto_host_output()
        } else {
            Self::with_cpu_forward_rct()
        }
    }

    /// Number of deinterleave stage attempts.
    pub fn deinterleave_attempts(&self) -> usize {
        self.deinterleave_attempts
    }

    /// Number of forward RCT stage attempts.
    pub fn forward_rct_attempts(&self) -> usize {
        self.forward_rct_attempts
    }

    /// Number of forward 5/3 DWT stage attempts.
    pub fn forward_dwt53_attempts(&self) -> usize {
        self.forward_dwt53_attempts
    }

    /// Number of forward 9/7 DWT stage attempts.
    pub fn forward_dwt97_attempts(&self) -> usize {
        self.forward_dwt97_attempts
    }

    /// Number of classic Tier-1 code-block encode attempts.
    pub fn tier1_code_block_attempts(&self) -> usize {
        self.tier1_code_block_attempts
    }

    /// Number of HT code-block encode attempts.
    pub fn ht_code_block_attempts(&self) -> usize {
        self.ht_code_block_attempts
    }

    /// Number of packetization stage attempts.
    pub fn packetization_attempts(&self) -> usize {
        self.packetization_attempts
    }

    /// Number of deinterleave Metal dispatches.
    pub fn deinterleave_dispatches(&self) -> usize {
        self.deinterleave_dispatches
    }

    /// Number of forward RCT Metal dispatches.
    pub fn forward_rct_dispatches(&self) -> usize {
        self.forward_rct_dispatches
    }

    /// Number of forward 5/3 DWT Metal dispatches.
    pub fn forward_dwt53_dispatches(&self) -> usize {
        self.forward_dwt53_dispatches
    }

    /// Number of forward 9/7 DWT Metal dispatches.
    pub fn forward_dwt97_dispatches(&self) -> usize {
        self.forward_dwt97_dispatches
    }

    /// Number of classic Tier-1 Metal dispatches.
    pub fn tier1_code_block_dispatches(&self) -> usize {
        self.tier1_code_block_dispatches
    }

    /// Number of HT code-block Metal dispatches.
    pub fn ht_code_block_dispatches(&self) -> usize {
        self.ht_code_block_dispatches
    }

    /// Number of packetization Metal dispatches.
    pub fn packetization_dispatches(&self) -> usize {
        self.packetization_dispatches
    }
}

#[cfg(target_os = "macos")]
fn metal_dispatch_result(
    result: &Result<(), crate::Error>,
    message: &'static str,
) -> Result<bool, &'static str> {
    match result {
        Ok(()) => Ok(true),
        Err(crate::Error::MetalUnavailable) => Ok(false),
        Err(_) => Err(message),
    }
}

#[cfg(target_os = "macos")]
pub(super) fn metal_dispatch_option<T>(
    result: Result<T, crate::Error>,
    message: &'static str,
) -> Result<Option<T>, &'static str> {
    match result {
        Ok(value) => Ok(Some(value)),
        Err(crate::Error::MetalUnavailable) => Ok(None),
        Err(_) => Err(message),
    }
}

impl encode_stage::J2kEncodeStageAccelerator for MetalEncodeStageAccelerator {
    fn dispatch_report(&self) -> encode_stage::J2kEncodeDispatchReport {
        encode_stage::J2kEncodeDispatchReport {
            deinterleave: self.deinterleave_dispatches,
            forward_rct: self.forward_rct_dispatches,
            forward_ict: 0,
            forward_dwt53: self.forward_dwt53_dispatches,
            forward_dwt97: self.forward_dwt97_dispatches,
            quantize_subband: 0,
            tier1_code_block: self.tier1_code_block_dispatches,
            ht_code_block: self.ht_code_block_dispatches,
            packetization: self.packetization_dispatches,
        }
    }

    fn prefer_parallel_cpu_code_block_fallback(&self) -> bool {
        self.parallel_cpu_code_block_fallback
    }

    fn encode_deinterleave(
        &mut self,
        job: encode_stage::J2kDeinterleaveToF32Job<'_>,
    ) -> core::result::Result<Option<Vec<Vec<f32>>>, &'static str> {
        self.deinterleave_attempts = self.deinterleave_attempts.saturating_add(1);
        if !self
            .dispatch_stages
            .contains(MetalEncodeDispatchStages::DEINTERLEAVE)
        {
            let _ = job;
            return Ok(None);
        }
        #[cfg(target_os = "macos")]
        {
            match compute::encode_deinterleave_to_f32(job) {
                Ok(Some(components)) => {
                    self.deinterleave_dispatches = self.deinterleave_dispatches.saturating_add(1);
                    Ok(Some(components))
                }
                Ok(None) | Err(crate::Error::MetalUnavailable) => Ok(None),
                Err(crate::Error::UnsupportedMetalRequest { .. }) => {
                    Err("Metal deinterleave encode shape is unsupported")
                }
                Err(_) => Err("Metal deinterleave encode kernel failed"),
            }
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = job;
            Ok(None)
        }
    }

    fn encode_forward_rct(
        &mut self,
        job: encode_stage::J2kForwardRctJob<'_>,
    ) -> core::result::Result<bool, &'static str> {
        self.forward_rct_attempts = self.forward_rct_attempts.saturating_add(1);
        if !self
            .dispatch_stages
            .contains(MetalEncodeDispatchStages::FORWARD_RCT)
        {
            let _ = job;
            return Ok(false);
        }
        #[cfg(target_os = "macos")]
        {
            let result = compute::encode_forward_rct(job.plane0, job.plane1, job.plane2);
            let dispatched =
                metal_dispatch_result(&result, "Metal forward RCT encode kernel failed")?;
            if dispatched {
                self.forward_rct_dispatches = self.forward_rct_dispatches.saturating_add(1);
            }
            Ok(dispatched)
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = job;
            Ok(false)
        }
    }

    fn encode_forward_dwt53(
        &mut self,
        job: encode_stage::J2kForwardDwt53Job<'_>,
    ) -> core::result::Result<Option<encode_stage::J2kForwardDwt53Output>, &'static str> {
        self.forward_dwt53_attempts = self.forward_dwt53_attempts.saturating_add(1);
        if job.num_levels == 0 {
            return Ok(None);
        }
        if self.auto_host_output_force_cpu_fallback {
            let _ = job;
            return Ok(None);
        }
        if !self
            .dispatch_stages
            .contains(MetalEncodeDispatchStages::FORWARD_DWT53)
        {
            let _ = job;
            return Ok(None);
        }
        #[cfg(target_os = "macos")]
        {
            let output = metal_dispatch_option(
                compute::encode_forward_dwt53(job.samples, job.width, job.height, job.num_levels),
                "Metal forward 5/3 DWT encode kernel failed",
            )?;
            if output.is_some() {
                self.forward_dwt53_dispatches = self.forward_dwt53_dispatches.saturating_add(1);
            }
            Ok(output)
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = job;
            Ok(None)
        }
    }

    fn encode_forward_dwt97(
        &mut self,
        job: encode_stage::J2kForwardDwt97Job<'_>,
    ) -> core::result::Result<Option<encode_stage::J2kForwardDwt97Output>, &'static str> {
        self.forward_dwt97_attempts = self.forward_dwt97_attempts.saturating_add(1);
        if job.num_levels == 0 || (job.width < 2 && job.height < 2) {
            return Ok(None);
        }
        if self.auto_host_output_force_cpu_fallback {
            let _ = job;
            return Ok(None);
        }
        if !self
            .dispatch_stages
            .contains(MetalEncodeDispatchStages::FORWARD_DWT97)
        {
            let _ = job;
            return Ok(None);
        }
        #[cfg(target_os = "macos")]
        {
            let output = metal_dispatch_option(
                compute::encode_forward_dwt97(job.samples, job.width, job.height, job.num_levels),
                "Metal forward 9/7 DWT encode kernel failed",
            )?;
            if output.is_some() {
                self.forward_dwt97_dispatches = self.forward_dwt97_dispatches.saturating_add(1);
            }
            Ok(output)
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = job;
            Ok(None)
        }
    }

    fn encode_tier1_code_block(
        &mut self,
        job: encode_stage::J2kTier1CodeBlockEncodeJob<'_>,
    ) -> core::result::Result<Option<encode_stage::EncodedJ2kCodeBlock>, &'static str> {
        self.tier1_code_block_attempts = self.tier1_code_block_attempts.saturating_add(1);
        if !self
            .dispatch_stages
            .contains(MetalEncodeDispatchStages::TIER1_CODE_BLOCK)
        {
            let _ = job;
            return Ok(None);
        }
        #[cfg(target_os = "macos")]
        {
            let encoded = metal_dispatch_option(
                compute::encode_classic_tier1_code_block(job),
                "Metal classic Tier-1 encode kernel failed",
            )?;
            if encoded.is_some() {
                self.tier1_code_block_dispatches =
                    self.tier1_code_block_dispatches.saturating_add(1);
            }
            Ok(encoded)
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = job;
            Ok(None)
        }
    }

    fn encode_tier1_code_blocks(
        &mut self,
        jobs: &[encode_stage::J2kTier1CodeBlockEncodeJob<'_>],
    ) -> core::result::Result<Option<Vec<encode_stage::EncodedJ2kCodeBlock>>, &'static str> {
        self.tier1_code_block_attempts = self.tier1_code_block_attempts.saturating_add(jobs.len());
        if !self
            .dispatch_stages
            .contains(MetalEncodeDispatchStages::TIER1_CODE_BLOCK)
        {
            let _ = jobs;
            return Ok(None);
        }
        #[cfg(target_os = "macos")]
        {
            let encoded = metal_dispatch_option(
                compute::encode_classic_tier1_code_blocks(jobs),
                "Metal classic Tier-1 encode batch kernel failed",
            )?;
            if encoded.is_some() && !jobs.is_empty() {
                self.tier1_code_block_dispatches =
                    self.tier1_code_block_dispatches.saturating_add(1);
            }
            Ok(encoded)
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = jobs;
            Ok(None)
        }
    }

    fn encode_ht_code_block(
        &mut self,
        job: encode_stage::J2kHtCodeBlockEncodeJob<'_>,
    ) -> core::result::Result<Option<encode_stage::EncodedHtJ2kCodeBlock>, &'static str> {
        self.ht_code_block_attempts = self.ht_code_block_attempts.saturating_add(1);
        if !self
            .dispatch_stages
            .contains(MetalEncodeDispatchStages::HT_CODE_BLOCK)
            || self.auto_host_output_force_cpu_fallback
        {
            let _ = job;
            return Ok(None);
        }
        #[cfg(target_os = "macos")]
        {
            let encoded = metal_dispatch_option(
                compute::encode_ht_cleanup_code_block(job),
                "Metal HTJ2K code-block encode kernel failed",
            )?;
            if encoded.is_some() {
                self.ht_code_block_dispatches = self.ht_code_block_dispatches.saturating_add(1);
            }
            Ok(encoded)
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = job;
            Ok(None)
        }
    }

    fn encode_ht_code_blocks(
        &mut self,
        jobs: &[encode_stage::J2kHtCodeBlockEncodeJob<'_>],
    ) -> core::result::Result<Option<Vec<encode_stage::EncodedHtJ2kCodeBlock>>, &'static str> {
        self.ht_code_block_attempts = self.ht_code_block_attempts.saturating_add(jobs.len());
        if !self
            .dispatch_stages
            .contains(MetalEncodeDispatchStages::HT_CODE_BLOCK)
            || self.auto_host_output_force_cpu_fallback
        {
            let _ = jobs;
            return Ok(None);
        }
        #[cfg(target_os = "macos")]
        {
            let encoded = metal_dispatch_option(
                compute::encode_ht_cleanup_code_blocks(jobs),
                "Metal HTJ2K code-block encode batch kernel failed",
            )?;
            if encoded.is_some() && !jobs.is_empty() {
                self.ht_code_block_dispatches = self.ht_code_block_dispatches.saturating_add(1);
            }
            Ok(encoded)
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = jobs;
            Ok(None)
        }
    }

    fn encode_htj2k_tile(
        &mut self,
        job: encode_stage::J2kHtj2kTileEncodeJob<'_>,
    ) -> core::result::Result<Option<Vec<u8>>, &'static str> {
        #[cfg(target_os = "macos")]
        {
            if self.dispatch_stages != MetalEncodeDispatchStages::AUTO_HOST_OUTPUT {
                let _ = job;
                return Ok(None);
            }
            let native_job = job;
            self.auto_host_output_force_cpu_fallback = false;
            let Some(options) = lossless_options_for_resident_htj2k_tile_job(native_job) else {
                return Ok(None);
            };
            if !should_use_resident_htj2k_host_tile_for_auto(native_job) {
                self.auto_host_output_force_cpu_fallback = true;
                return Ok(None);
            }
            let session = match crate::MetalBackendSession::system_default() {
                Ok(session) => session,
                Err(crate::Error::MetalUnavailable) => return Ok(None),
                Err(_) => return Err("Metal HTJ2K hybrid tile encode session creation failed"),
            };
            let source_buffer = match borrow_padded_metal_buffer_from_bytes(&session, job.pixels) {
                Ok(buffer) => buffer,
                Err(crate::Error::MetalUnavailable) => return Ok(None),
                Err(_) => return Err("Metal HTJ2K hybrid tile input buffer creation failed"),
            };
            let pitch_bytes = (job.width as usize)
                .checked_mul(usize::from(job.num_components))
                .ok_or("Metal HTJ2K hybrid tile pitch overflow")?;
            let tile = MetalLosslessEncodeTile {
                buffer: &source_buffer,
                byte_offset: 0,
                width: job.width,
                height: job.height,
                pitch_bytes,
                output_width: job.width,
                output_height: job.height,
                format: PixelFormat::Rgb8,
            };
            let encoded = match encode_resident_ht_tile_body_with_cpu_packetization(
                tile,
                options,
                &session,
                MetalEncodeInputStaging::AlreadyPaddedContiguous,
                job.code_block_width,
                job.code_block_height,
            ) {
                Ok(Some(encoded)) => encoded,
                Ok(None) | Err(crate::Error::MetalUnavailable) => return Ok(None),
                Err(_) => return Err("Metal resident HTJ2K hybrid tile encode failed"),
            };

            self.forward_rct_attempts = self.forward_rct_attempts.saturating_add(1);
            if encoded.used_fused_rct {
                self.forward_rct_dispatches = self.forward_rct_dispatches.saturating_add(1);
            }
            if encoded.num_decomposition_levels > 0 {
                let component_count = usize::from(job.num_components);
                self.forward_dwt53_attempts =
                    self.forward_dwt53_attempts.saturating_add(component_count);
                self.forward_dwt53_dispatches = self
                    .forward_dwt53_dispatches
                    .saturating_add(encoded.forward_dwt53_dispatches);
            }
            self.ht_code_block_attempts = self
                .ht_code_block_attempts
                .saturating_add(encoded.code_block_count);
            self.ht_code_block_dispatches = self
                .ht_code_block_dispatches
                .saturating_add(encoded.ht_code_block_dispatches);
            Ok(Some(encoded.tile_data))
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = job;
            Ok(None)
        }
    }

    fn encode_packetization(
        &mut self,
        job: encode_stage::J2kPacketizationEncodeJob<'_>,
    ) -> core::result::Result<Option<Vec<u8>>, &'static str> {
        self.packetization_attempts = self.packetization_attempts.saturating_add(1);
        self.auto_host_output_force_cpu_fallback = false;
        if !self
            .dispatch_stages
            .contains(MetalEncodeDispatchStages::PACKETIZATION)
        {
            let _ = job;
            return Ok(None);
        }
        #[cfg(target_os = "macos")]
        {
            let native_job = job;
            let encoded = metal_dispatch_option(
                compute::encode_tier2_packetization(native_job),
                "Metal Tier-2 packetization encode kernel failed",
            )?;
            if encoded.is_some() {
                self.packetization_dispatches = self.packetization_dispatches.saturating_add(1);
            }
            Ok(encoded)
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = job;
            Ok(None)
        }
    }
}
