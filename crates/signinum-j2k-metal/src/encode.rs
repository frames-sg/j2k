// SPDX-License-Identifier: Apache-2.0

#[cfg(target_os = "macos")]
use crate::compute;
#[cfg(target_os = "macos")]
use metal::Buffer;
#[cfg(all(test, target_os = "macos"))]
use rayon::prelude::*;
use signinum_core::DeviceSubmission;
#[cfg(target_os = "macos")]
use signinum_core::{BackendKind, DeviceSurface, PixelFormat};
use signinum_j2k::native_bridge::{
    dwt53_output_from_native, encoded_ht_from_native, encoded_j2k_from_native, ht_job_to_native,
    htj2k_tile_job_to_native, packet_descriptor_to_native, packet_progression_to_native,
    packet_resolutions_to_native, tier1_job_to_native,
};
#[cfg(target_os = "macos")]
use signinum_j2k::{
    EncodeBackendPreference, J2kBlockCodingMode, J2kEncodeValidation, J2kProgressionOrder,
    ReversibleTransform,
};
use signinum_j2k::{EncodedJ2k, J2kLosslessEncodeOptions, J2kLosslessSamples};
#[cfg(target_os = "macos")]
use signinum_j2k_native::{
    EncodeOptions, EncodeProgressionOrder, J2kPacketizationBlockCodingMode,
    J2kPacketizationCodeBlock, J2kPacketizationPacketDescriptor, J2kPacketizationProgressionOrder,
    J2kPacketizationResolution, J2kPacketizationSubband, J2kSubBandType,
};
use signinum_j2k_native::{
    EncodedHtJ2kCodeBlock, J2kEncodeStageAccelerator, J2kHtj2kTileEncodeJob,
    J2kPacketizationEncodeJob,
};
#[cfg(all(test, target_os = "macos"))]
use std::cell::Cell;
#[cfg(all(test, target_os = "macos"))]
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use std::time::Duration;
#[cfg(target_os = "macos")]
use std::time::Instant;

/// Encode-stage accelerator for JPEG 2000 Metal work.
///
/// The type is wired into the public J2K encode-stage interface and reports
/// dispatches for each required encode stage.
#[derive(Debug, Clone)]
pub struct MetalEncodeStageAccelerator {
    dispatch_stages: MetalEncodeDispatchStages,
    parallel_cpu_code_block_fallback: bool,
    auto_host_output_force_cpu_fallback: bool,
    forward_rct_attempts: usize,
    forward_dwt53_attempts: usize,
    tier1_code_block_attempts: usize,
    ht_code_block_attempts: usize,
    packetization_attempts: usize,
    forward_rct_dispatches: usize,
    forward_dwt53_dispatches: usize,
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
            forward_rct_attempts: 0,
            forward_dwt53_attempts: 0,
            tier1_code_block_attempts: 0,
            ht_code_block_attempts: 0,
            packetization_attempts: 0,
            forward_rct_dispatches: 0,
            forward_dwt53_dispatches: 0,
            tier1_code_block_dispatches: 0,
            ht_code_block_dispatches: 0,
            packetization_dispatches: 0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MetalEncodeDispatchStages(u8);

impl MetalEncodeDispatchStages {
    const FORWARD_RCT: Self = Self(1 << 0);
    const FORWARD_DWT53: Self = Self(1 << 1);
    const TIER1_CODE_BLOCK: Self = Self(1 << 2);
    const HT_CODE_BLOCK: Self = Self(1 << 3);
    const PACKETIZATION: Self = Self(1 << 4);
    const AUTO_HOST_OUTPUT: Self = Self(Self::FORWARD_DWT53.0 | Self::HT_CODE_BLOCK.0);
    const ALL: Self = Self(
        Self::FORWARD_RCT.0
            | Self::FORWARD_DWT53.0
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

    #[cfg(target_os = "macos")]
    fn for_host_output(options: J2kLosslessEncodeOptions) -> Self {
        if options.backend == EncodeBackendPreference::Auto {
            Self::for_auto_host_output()
        } else {
            Self::with_cpu_forward_rct()
        }
    }

    /// Number of forward RCT stage attempts.
    pub fn forward_rct_attempts(&self) -> usize {
        self.forward_rct_attempts
    }

    /// Number of forward 5/3 DWT stage attempts.
    pub fn forward_dwt53_attempts(&self) -> usize {
        self.forward_dwt53_attempts
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

    /// Number of forward RCT Metal dispatches.
    pub fn forward_rct_dispatches(&self) -> usize {
        self.forward_rct_dispatches
    }

    /// Number of forward 5/3 DWT Metal dispatches.
    pub fn forward_dwt53_dispatches(&self) -> usize {
        self.forward_dwt53_dispatches
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
fn metal_dispatch_option<T>(
    result: Result<T, crate::Error>,
    message: &'static str,
) -> Result<Option<T>, &'static str> {
    match result {
        Ok(value) => Ok(Some(value)),
        Err(crate::Error::MetalUnavailable) => Ok(None),
        Err(_) => Err(message),
    }
}

impl signinum_j2k::J2kEncodeStageAccelerator for MetalEncodeStageAccelerator {
    fn dispatch_report(&self) -> signinum_j2k::J2kEncodeDispatchReport {
        signinum_j2k::J2kEncodeDispatchReport {
            deinterleave: 0,
            forward_rct: self.forward_rct_dispatches,
            forward_ict: 0,
            forward_dwt53: self.forward_dwt53_dispatches,
            forward_dwt97: 0,
            quantize_subband: 0,
            tier1_code_block: self.tier1_code_block_dispatches,
            ht_code_block: self.ht_code_block_dispatches,
            packetization: self.packetization_dispatches,
        }
    }

    fn prefer_parallel_cpu_code_block_fallback(&self) -> bool {
        self.parallel_cpu_code_block_fallback
    }

    fn encode_forward_rct(
        &mut self,
        job: signinum_j2k::J2kForwardRctJob<'_>,
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
        job: signinum_j2k::J2kForwardDwt53Job<'_>,
    ) -> core::result::Result<Option<signinum_j2k::J2kForwardDwt53Output>, &'static str> {
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
            )?
            .map(dwt53_output_from_native);
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

    fn encode_tier1_code_block(
        &mut self,
        job: signinum_j2k::J2kTier1CodeBlockEncodeJob<'_>,
    ) -> core::result::Result<Option<signinum_j2k::EncodedJ2kCodeBlock>, &'static str> {
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
                compute::encode_classic_tier1_code_block(tier1_job_to_native(job)),
                "Metal classic Tier-1 encode kernel failed",
            )?
            .map(encoded_j2k_from_native);
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
        jobs: &[signinum_j2k::J2kTier1CodeBlockEncodeJob<'_>],
    ) -> core::result::Result<Option<Vec<signinum_j2k::EncodedJ2kCodeBlock>>, &'static str> {
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
            let native_jobs = jobs
                .iter()
                .copied()
                .map(tier1_job_to_native)
                .collect::<Vec<_>>();
            let encoded = metal_dispatch_option(
                compute::encode_classic_tier1_code_blocks(&native_jobs),
                "Metal classic Tier-1 encode batch kernel failed",
            )?
            .map(|blocks| blocks.into_iter().map(encoded_j2k_from_native).collect());
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
        job: signinum_j2k::J2kHtCodeBlockEncodeJob<'_>,
    ) -> core::result::Result<Option<signinum_j2k::EncodedHtJ2kCodeBlock>, &'static str> {
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
                compute::encode_ht_cleanup_code_block(ht_job_to_native(job)),
                "Metal HTJ2K code-block encode kernel failed",
            )?
            .map(encoded_ht_from_native);
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
        jobs: &[signinum_j2k::J2kHtCodeBlockEncodeJob<'_>],
    ) -> core::result::Result<Option<Vec<signinum_j2k::EncodedHtJ2kCodeBlock>>, &'static str> {
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
            let native_jobs = jobs
                .iter()
                .copied()
                .map(ht_job_to_native)
                .collect::<Vec<_>>();
            let encoded = metal_dispatch_option(
                compute::encode_ht_cleanup_code_blocks(&native_jobs),
                "Metal HTJ2K code-block encode batch kernel failed",
            )?
            .map(|blocks| blocks.into_iter().map(encoded_ht_from_native).collect());
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
        job: signinum_j2k::J2kHtj2kTileEncodeJob<'_>,
    ) -> core::result::Result<Option<Vec<u8>>, &'static str> {
        #[cfg(target_os = "macos")]
        {
            if self.dispatch_stages != MetalEncodeDispatchStages::AUTO_HOST_OUTPUT {
                let _ = job;
                return Ok(None);
            }
            let native_job = htj2k_tile_job_to_native(job);
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
        job: signinum_j2k::J2kPacketizationEncodeJob<'_>,
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
            let packet_descriptors = job
                .packet_descriptors
                .iter()
                .copied()
                .map(packet_descriptor_to_native)
                .collect::<Vec<_>>();
            let resolutions = packet_resolutions_to_native(job.resolutions);
            let native_job = J2kPacketizationEncodeJob {
                resolution_count: job.resolution_count,
                num_layers: job.num_layers,
                num_components: job.num_components,
                code_block_count: job.code_block_count,
                progression_order: packet_progression_to_native(job.progression_order),
                packet_descriptors: &packet_descriptors,
                resolutions: &resolutions,
            };
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

#[cfg(target_os = "macos")]
#[derive(Debug, Clone, Copy)]
/// Metal buffer and layout metadata for one lossless J2K encode tile.
pub struct MetalLosslessEncodeTile<'a> {
    /// Source Metal buffer containing Gray or RGB pixels.
    pub buffer: &'a Buffer,
    /// Byte offset of the first source pixel in `buffer`.
    pub byte_offset: usize,
    /// Width of the valid input region in pixels.
    pub width: u32,
    /// Height of the valid input region in pixels.
    pub height: u32,
    /// Number of bytes between consecutive input rows.
    pub pitch_bytes: usize,
    /// Encoded image width in pixels.
    pub output_width: u32,
    /// Encoded image height in pixels.
    pub output_height: u32,
    /// Pixel format of the source buffer.
    pub format: PixelFormat,
}

#[cfg(not(target_os = "macos"))]
#[derive(Debug, Clone, Copy)]
/// Placeholder lossless encode tile type for non-macOS builds.
pub struct MetalLosslessEncodeTile<'a> {
    _private: core::marker::PhantomData<&'a ()>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Residency decisions used by a lossless Metal encode.
pub struct MetalLosslessEncodeResidency {
    /// Whether coefficient preparation ran on Metal.
    pub coefficient_prep_used: bool,
    /// Whether packetization ran on Metal.
    pub packetization_used: bool,
    /// Whether codestream assembly stayed resident on Metal.
    pub codestream_assembly_used: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Lossless Metal encode output with host codestream bytes and timings.
///
/// API note: this diagnostic report is constructed by this crate. It is not
/// `#[non_exhaustive]`, but adapter releases may add diagnostic fields as the
/// resident encode path gains more profiling detail.
pub struct MetalLosslessEncodeOutcome {
    /// Encoded J2K codestream.
    pub encoded: EncodedJ2k,
    /// Whether the input buffer had to be copied or padded.
    pub input_copy_used: bool,
    /// Residency decisions for the encode stages.
    pub resident: MetalLosslessEncodeResidency,
    /// Time spent copying or padding the input.
    pub input_copy_duration: Duration,
    /// End-to-end encode duration for this tile.
    pub encode_duration: Duration,
    /// GPU-only duration when timestamp data is available.
    pub gpu_duration: Option<Duration>,
    /// Time spent validating the encoded output.
    pub validation_duration: Duration,
    /// Time spent materializing buffer-backed codestream bytes into host bytes.
    pub host_readback_duration: Duration,
}

#[cfg(target_os = "macos")]
/// JPEG 2000 codestream bytes owned by a Metal buffer.
///
/// The buffer is CPU-readable for the current padded resident encode API, so
/// callers can stream `codestream_bytes()` into file or network writers without
/// first materializing an owned `Vec<u8>`.
pub struct MetalEncodedJ2k {
    /// Metal buffer containing the codestream bytes.
    pub codestream_buffer: Buffer,
    /// Byte offset of the first codestream byte in `codestream_buffer`.
    pub byte_offset: usize,
    /// Number of valid codestream bytes.
    pub byte_len: usize,
    /// Allocated codestream capacity in bytes.
    pub capacity: usize,
    /// Encoded image width in pixels.
    pub width: u32,
    /// Encoded image height in pixels.
    pub height: u32,
    /// Number of encoded components.
    pub components: u8,
    /// Component bit depth.
    pub bit_depth: u8,
    /// Whether components are signed.
    pub signed: bool,
}

#[cfg(target_os = "macos")]
impl MetalEncodedJ2k {
    /// Borrow the finished codestream bytes from the backing Metal buffer.
    pub fn codestream_bytes(&self) -> Result<&[u8], crate::Error> {
        let end = self.byte_offset.checked_add(self.byte_len).ok_or_else(|| {
            crate::Error::MetalKernel {
                message: "J2K Metal codestream byte range overflow".to_string(),
            }
        })?;
        let buffer_len = usize::try_from(self.codestream_buffer.length()).map_err(|_| {
            crate::Error::MetalKernel {
                message: "J2K Metal codestream buffer length exceeds usize".to_string(),
            }
        })?;
        if end > buffer_len {
            return Err(crate::Error::MetalKernel {
                message: "J2K Metal codestream byte range exceeds buffer length".to_string(),
            });
        }
        let ptr = self.codestream_buffer.contents().cast::<u8>();
        if ptr.is_null() {
            return Err(crate::Error::MetalKernel {
                message: "J2K Metal codestream buffer is not CPU-readable".to_string(),
            });
        }
        Ok(unsafe { core::slice::from_raw_parts(ptr.add(self.byte_offset), self.byte_len) })
    }

    /// Materialize the buffer-backed codestream into the compatibility `Vec` API shape.
    pub fn to_encoded_j2k(&self) -> Result<EncodedJ2k, crate::Error> {
        let (encoded, _host_readback_duration) = self.to_encoded_j2k_with_readback_duration()?;
        Ok(encoded)
    }

    fn to_encoded_j2k_with_readback_duration(
        &self,
    ) -> Result<(EncodedJ2k, Duration), crate::Error> {
        let readback_started = Instant::now();
        let codestream = self.codestream_bytes()?.to_vec();
        let host_readback_duration = readback_started.elapsed();
        Ok((
            EncodedJ2k {
                codestream,
                backend: BackendKind::Metal,
                dispatch_report: signinum_j2k::J2kEncodeDispatchReport::default(),
                width: self.width,
                height: self.height,
                components: self.components,
                bit_depth: self.bit_depth,
                signed: self.signed,
            },
            host_readback_duration,
        ))
    }
}

#[cfg(not(target_os = "macos"))]
/// Placeholder Metal codestream type for non-macOS builds.
pub struct MetalEncodedJ2k {
    _private: (),
}

/// Metal lossless encode report for buffer-backed codestream output.
pub struct MetalLosslessBufferEncodeOutcome {
    /// Encoded codestream stored in a Metal buffer.
    pub encoded: MetalEncodedJ2k,
    /// Whether the input buffer had to be copied or padded.
    pub input_copy_used: bool,
    /// Residency decisions for the encode stages.
    pub resident: MetalLosslessEncodeResidency,
    /// Time spent copying or padding the input.
    pub input_copy_duration: Duration,
    /// End-to-end encode duration for this tile.
    pub encode_duration: Duration,
    /// GPU-only duration when timestamp data is available.
    pub gpu_duration: Option<Duration>,
    /// Time spent validating the encoded output.
    pub validation_duration: Duration,
}

/// Tuning knobs for resident Metal lossless J2K/HTJ2K tile batch encode.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MetalLosslessEncodeConfig {
    /// Requested maximum number of tiles submitted concurrently.
    ///
    /// `None` uses the crate default and still clamps by the memory budget.
    pub gpu_encode_inflight_tiles: Option<usize>,
    /// Resident encode memory budget in bytes.
    ///
    /// `None` uses `min(10 GiB, hw_memsize * 0.40)` when host memory can be
    /// discovered.
    pub gpu_encode_memory_budget_bytes: Option<usize>,
}

/// Optional resident Metal encode stage timings.
///
/// API note: this diagnostic report is constructed by this crate. It is not
/// `#[non_exhaustive]`, but adapter releases may add diagnostic fields as the
/// resident encode path gains more profiling detail.
///
/// Unless a field explicitly says otherwise, timing fields are host-side
/// `Instant` buckets for RCA and should not be read as exact GPU execution
/// elapsed time.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MetalLosslessEncodeStageStats {
    /// Time spent planning the resident encode batch.
    pub plan_duration: Duration,
    /// Time spent preparing and submitting Metal work.
    pub prepare_submit_duration: Duration,
    /// Host-side wall time spent preparing resident encode coefficients.
    pub coefficient_prep_duration: Duration,
    /// Reserved for future finer-grained deinterleave plus RCT profiling.
    ///
    /// Current resident prep timing is reported in `coefficient_prep_duration`.
    pub deinterleave_rct_duration: Duration,
    /// Reserved for future finer-grained forward 5/3 DWT profiling.
    ///
    /// Current resident prep timing is reported in `coefficient_prep_duration`.
    pub dwt53_duration: Duration,
    /// Reserved for future finer-grained coefficient extraction profiling.
    ///
    /// Current resident prep timing is reported in `coefficient_prep_duration`.
    pub coefficient_extract_duration: Duration,
    /// Time spent building HT lookup tables.
    pub ht_table_build_duration: Duration,
    /// Time spent allocating HT output buffers.
    pub ht_buffer_allocation_duration: Duration,
    /// Host-side Metal command encoding time for HT resident command buffers.
    ///
    /// This is the sum of the split command-encode buckets below and is not GPU
    /// kernel execution elapsed time.
    pub ht_command_encode_duration: Duration,
    /// Host-side Metal command encoding time for HT code-block dispatch setup.
    pub ht_block_encode_duration: Duration,
    /// CPU-side setup time for classic Tier-1 batch jobs and buffers.
    pub classic_tier1_setup_duration: Duration,
    /// Host-side Metal command encoding time for classic code-block dispatch setup.
    pub classic_block_encode_duration: Duration,
    /// Host-side CPU time spent packing compact classic Tier-1 tokens.
    ///
    /// This is populated only when
    /// `SIGNINUM_J2K_METAL_PROFILE_CLASSIC_TIER1_TOKEN_PACK=1` is enabled.
    pub classic_tier1_token_pack_duration: Duration,
    /// CPU-side packet metadata planning time for classic resident batches.
    pub classic_packet_plan_duration: Duration,
    /// CPU-side packet/codestream buffer setup time for classic resident batches.
    pub classic_packet_buffer_setup_duration: Duration,
    /// Host-side time spent committing split classic resident command buffers.
    pub classic_command_buffer_commit_duration: Duration,
    /// Host-side wall time spent harvesting completed resident batch results.
    pub result_harvest_duration: Duration,
    /// Host-side time spent copying shared status buffers into CPU-owned status arrays.
    pub result_status_copy_duration: Duration,
    /// Host-side time spent returning private buffers to the resident buffer pool.
    pub result_private_recycle_duration: Duration,
    /// Host-side time spent returning shared buffers to the resident buffer pool.
    pub result_shared_recycle_duration: Duration,
    /// Host-side time spent validating per-tile status and building codestream handles.
    pub result_codestream_collect_duration: Duration,
    /// Host-side Metal command encoding time for packet block metadata dispatch setup.
    pub packet_block_prep_duration: Duration,
    /// Host-side Metal command encoding time for packet body dispatch setup.
    pub packetization_duration: Duration,
    /// Host-side Metal command encoding time for codestream assembly dispatch setup.
    pub codestream_assembly_duration: Duration,
    /// GPU time spent preparing resident coefficient buffers.
    ///
    /// This includes the resident input deinterleave/RCT, DWT, and coefficient
    /// extraction command buffer when stage profiling is enabled.
    pub coefficient_prep_gpu_duration: Duration,
    /// GPU time spent deinterleaving resident input planes and applying RCT.
    ///
    /// This is populated only when resident coefficient-prep split profiling is enabled.
    pub coefficient_deinterleave_rct_gpu_duration: Duration,
    /// GPU time spent running resident forward DWT 5/3 coefficient prep.
    ///
    /// This is populated only when resident coefficient-prep split profiling is enabled.
    pub coefficient_dwt53_gpu_duration: Duration,
    /// GPU time spent in resident forward DWT 5/3 vertical passes.
    ///
    /// This is populated only when resident coefficient-prep split profiling is enabled.
    pub coefficient_dwt53_vertical_gpu_duration: Duration,
    /// GPU time spent in resident forward DWT 5/3 horizontal passes.
    ///
    /// This is populated only when resident coefficient-prep split profiling is enabled.
    pub coefficient_dwt53_horizontal_gpu_duration: Duration,
    /// GPU time spent extracting resident code-block coefficients.
    ///
    /// This is populated only when resident coefficient-prep split profiling is enabled.
    pub coefficient_extract_gpu_duration: Duration,
    /// GPU time spent copying per-tile coefficient buffers into a batch buffer.
    ///
    /// This is populated only when resident split-command profiling is enabled.
    pub coefficient_copy_gpu_duration: Duration,
    /// Elapsed GPU timestamp window across the resident encode command buffers.
    ///
    /// This is `max(GPUEndTime) - min(GPUStartTime)` for the command buffers
    /// retained by the batch. It is a wall-window companion to summed GPU busy
    /// rows and should not be added to per-stage GPU durations.
    pub gpu_elapsed_wall_duration: Duration,
    /// GPU time spent in the classic Tier-1 code-block encode command.
    ///
    /// This is populated only when classic split-command profiling is enabled.
    pub classic_block_gpu_duration: Duration,
    /// GPU time spent in the profile-only classic Tier-1 density probe.
    ///
    /// This is populated only when classic split-command profiling and
    /// `SIGNINUM_J2K_METAL_PROFILE_CLASSIC_TIER1_DENSITY=1` are enabled.
    pub classic_tier1_density_gpu_duration: Duration,
    /// GPU time spent in the profile-only classic Tier-1 raw bypass packing probe.
    ///
    /// This is populated only when classic split-command profiling and
    /// `SIGNINUM_J2K_METAL_PROFILE_CLASSIC_TIER1_RAW_PACK=1` are enabled.
    pub classic_tier1_raw_pack_gpu_duration: Duration,
    /// GPU time spent in the profile-only classic Tier-1 MQ arithmetic packing probe.
    ///
    /// This is populated only when classic split-command profiling and
    /// `SIGNINUM_J2K_METAL_PROFILE_CLASSIC_TIER1_ARITHMETIC_PACK=1` are enabled.
    pub classic_tier1_arithmetic_pack_gpu_duration: Duration,
    /// GPU time spent in the profile-only classic Tier-1 ordered symbol-plan probe.
    ///
    /// This is populated only when classic split-command profiling and
    /// `SIGNINUM_J2K_METAL_PROFILE_CLASSIC_TIER1_SYMBOL_PLAN=1` are enabled.
    pub classic_tier1_symbol_plan_gpu_duration: Duration,
    /// GPU time spent in the profile-only classic Tier-1 pass-plan probe.
    ///
    /// This is populated only when classic split-command profiling and
    /// `SIGNINUM_J2K_METAL_PROFILE_CLASSIC_TIER1_PASS_PLAN=1` are enabled.
    pub classic_tier1_pass_plan_gpu_duration: Duration,
    /// GPU time spent in the profile-only classic Tier-1 compact token-emitter probe.
    ///
    /// This is populated only when classic split-command profiling and
    /// `SIGNINUM_J2K_METAL_PROFILE_CLASSIC_TIER1_TOKEN_EMIT=1` are enabled.
    pub classic_tier1_token_emit_gpu_duration: Duration,
    /// GPU time spent in the profile-only classic Tier-1 split MQ/raw token-emitter probe.
    ///
    /// This is populated only when classic split-command profiling and
    /// `SIGNINUM_J2K_METAL_PROFILE_CLASSIC_TIER1_SPLIT_TOKEN_EMIT=1` are enabled.
    pub classic_tier1_split_token_emit_gpu_duration: Duration,
    /// GPU time spent packing compact classic Tier-1 tokens into resident payloads.
    ///
    /// This is populated when the gated classic GPU token-pack route is enabled.
    pub classic_tier1_token_pack_gpu_duration: Duration,
    /// GPU time spent in the HT Tier-1 code-block encode command.
    ///
    /// This is populated only when HT split-command profiling is enabled.
    pub ht_block_gpu_duration: Duration,
    /// GPU time spent preparing packet-block metadata from HT Tier-1 status.
    ///
    /// This is populated only when HT split-command profiling is enabled.
    pub packet_block_prep_gpu_duration: Duration,
    /// GPU time spent in HTJ2K packetization.
    ///
    /// This is populated only when HT split-command profiling is enabled.
    pub packetization_gpu_duration: Duration,
    /// GPU time spent copying packet payload bytes after header packetization.
    ///
    /// This is populated only when HT split-command profiling is enabled.
    pub packet_payload_copy_gpu_duration: Duration,
    /// GPU time spent assembling the HTJ2K codestream buffer.
    ///
    /// This is populated only when HT split-command profiling is enabled.
    pub codestream_assembly_gpu_duration: Duration,
    /// GPU time spent copying packet payload bytes into final codestream buffers.
    ///
    /// This is populated only when HT split-command profiling is enabled.
    pub codestream_payload_copy_gpu_duration: Duration,
    /// Total Tier-1 output capacity, in bytes, across resident code blocks.
    pub tier1_output_capacity_total: usize,
    /// Maximum Tier-1 output capacity, in bytes, for any resident code block.
    pub max_tier1_output_capacity: usize,
    /// Actual Tier-1 output bytes written across resident code blocks.
    pub tier1_output_used_bytes_total: usize,
    /// Maximum actual Tier-1 output bytes written by any resident code block.
    pub max_tier1_output_used_bytes: usize,
    /// Total Tier-1 segment metadata capacity across resident code blocks.
    pub tier1_segment_capacity_total: usize,
    /// Maximum Tier-1 segment metadata capacity for any resident code block.
    pub max_tier1_segment_capacity_per_block: usize,
    /// Actual Tier-1 coding passes emitted across resident code blocks.
    pub tier1_coding_pass_count_total: usize,
    /// Maximum actual Tier-1 coding passes emitted by any resident code block.
    pub max_tier1_coding_passes_per_block: usize,
    /// Estimated classic MQ/arithmetic coding passes across resident code blocks.
    ///
    /// For HTJ2K Tier-1 this remains zero.
    pub tier1_arithmetic_pass_count_total: usize,
    /// Estimated classic raw bypass coding passes across resident code blocks.
    ///
    /// For HTJ2K Tier-1 this remains zero.
    pub tier1_raw_pass_count_total: usize,
    /// Estimated classic cleanup passes across resident code blocks.
    ///
    /// For HTJ2K Tier-1 this remains zero.
    pub tier1_cleanup_pass_count_total: usize,
    /// Estimated classic significance propagation passes across resident code blocks.
    ///
    /// For HTJ2K Tier-1 this remains zero.
    pub tier1_sigprop_pass_count_total: usize,
    /// Estimated classic magnitude refinement passes across resident code blocks.
    ///
    /// For HTJ2K Tier-1 this remains zero.
    pub tier1_magref_pass_count_total: usize,
    /// Estimated classic MQ/arithmetic cleanup passes across resident code blocks.
    ///
    /// For HTJ2K Tier-1 this remains zero.
    pub tier1_arithmetic_cleanup_pass_count_total: usize,
    /// Estimated classic MQ/arithmetic significance propagation passes.
    ///
    /// For HTJ2K Tier-1 this remains zero.
    pub tier1_arithmetic_sigprop_pass_count_total: usize,
    /// Estimated classic MQ/arithmetic magnitude refinement passes.
    ///
    /// For HTJ2K Tier-1 this remains zero.
    pub tier1_arithmetic_magref_pass_count_total: usize,
    /// Estimated classic raw bypass significance propagation passes.
    ///
    /// For HTJ2K Tier-1 this remains zero.
    pub tier1_raw_sigprop_pass_count_total: usize,
    /// Estimated classic raw bypass magnitude refinement passes.
    ///
    /// For HTJ2K Tier-1 this remains zero.
    pub tier1_raw_magref_pass_count_total: usize,
    /// Estimated full coefficient visits made by classic Tier-1 pass scans.
    ///
    /// This is derived from actual emitted pass counts and code-block areas.
    /// For HTJ2K Tier-1 this remains zero.
    pub tier1_full_scan_coeff_visit_count_total: usize,
    /// Estimated full coefficient visits made by MQ/arithmetic pass scans.
    ///
    /// For HTJ2K Tier-1 this remains zero.
    pub tier1_arithmetic_scan_coeff_visit_count_total: usize,
    /// Estimated full coefficient visits made by raw bypass pass scans.
    ///
    /// For HTJ2K Tier-1 this remains zero.
    pub tier1_raw_scan_coeff_visit_count_total: usize,
    /// Estimated full coefficient visits made by cleanup pass scans.
    ///
    /// For HTJ2K Tier-1 this remains zero.
    pub tier1_cleanup_scan_coeff_visit_count_total: usize,
    /// Estimated full coefficient visits made by significance propagation scans.
    ///
    /// For HTJ2K Tier-1 this remains zero.
    pub tier1_sigprop_scan_coeff_visit_count_total: usize,
    /// Estimated full coefficient visits made by magnitude refinement scans.
    ///
    /// For HTJ2K Tier-1 this remains zero.
    pub tier1_magref_scan_coeff_visit_count_total: usize,
    /// Maximum estimated full coefficient scan visits for any classic block.
    ///
    /// For HTJ2K Tier-1 this remains zero.
    pub max_tier1_full_scan_coeff_visits_per_block: usize,
    /// Profile-only count of classic significance propagation candidates.
    ///
    /// This is populated only when classic Tier-1 density profiling is enabled.
    pub tier1_sigprop_active_candidate_count_total: usize,
    /// Profile-only count of coefficients that become significant in sigprop.
    ///
    /// This is populated only when classic Tier-1 density profiling is enabled.
    pub tier1_sigprop_new_significant_count_total: usize,
    /// Profile-only count of classic magnitude refinement candidates.
    ///
    /// This is populated only when classic Tier-1 density profiling is enabled.
    pub tier1_magref_active_candidate_count_total: usize,
    /// Profile-only count of arithmetic-coded significance propagation candidates.
    pub tier1_arithmetic_sigprop_active_candidate_count_total: usize,
    /// Profile-only count of coefficients that become significant in arithmetic sigprop.
    pub tier1_arithmetic_sigprop_new_significant_count_total: usize,
    /// Profile-only count of raw bypass significance propagation candidates.
    pub tier1_raw_sigprop_active_candidate_count_total: usize,
    /// Profile-only count of coefficients that become significant in raw sigprop.
    pub tier1_raw_sigprop_new_significant_count_total: usize,
    /// Profile-only count of arithmetic-coded magnitude refinement candidates.
    pub tier1_arithmetic_magref_active_candidate_count_total: usize,
    /// Profile-only count of raw bypass magnitude refinement candidates.
    pub tier1_raw_magref_active_candidate_count_total: usize,
    /// Profile-only count of cleanup-pass coefficient candidates.
    ///
    /// This excludes coefficients represented only by cleanup RLC stripes.
    pub tier1_cleanup_active_candidate_count_total: usize,
    /// Profile-only count of coefficients that become significant in cleanup.
    ///
    /// This includes significance discovered through cleanup RLC.
    pub tier1_cleanup_new_significant_count_total: usize,
    /// Profile-only count of cleanup stripes encoded by the RLC path.
    pub tier1_cleanup_rlc_stripe_count_total: usize,
    /// Profile-only count of cleanup RLC stripes with no significant coefficient.
    pub tier1_cleanup_rlc_zero_stripe_count_total: usize,
    /// Profile-only exact MQ symbol count from the ordered symbol-plan probe.
    pub tier1_symbol_plan_mq_symbol_count_total: usize,
    /// Profile-only exact raw bypass bit count from the ordered symbol-plan probe.
    pub tier1_symbol_plan_raw_bit_count_total: usize,
    /// Maximum MQ symbols emitted by any block in the ordered symbol-plan probe.
    pub max_tier1_symbol_plan_mq_symbols_per_block: usize,
    /// Maximum raw bypass bits emitted by any block in the ordered symbol-plan probe.
    pub max_tier1_symbol_plan_raw_bits_per_block: usize,
    /// Estimated compact token bytes needed for all blocks in the symbol-plan probe.
    pub tier1_symbol_plan_packed_token_bytes_total: usize,
    /// Maximum estimated compact token bytes needed by any one block.
    pub max_tier1_symbol_plan_packed_token_bytes_per_block: usize,
    /// Profile-only exact cleanup MQ symbol count from the ordered symbol-plan probe.
    pub tier1_symbol_plan_cleanup_mq_symbol_count_total: usize,
    /// Profile-only exact sigprop MQ symbol count from the ordered symbol-plan probe.
    pub tier1_symbol_plan_sigprop_mq_symbol_count_total: usize,
    /// Profile-only exact magref MQ symbol count from the ordered symbol-plan probe.
    pub tier1_symbol_plan_magref_mq_symbol_count_total: usize,
    /// Profile-only exact raw sigprop bit count from the ordered symbol-plan probe.
    pub tier1_symbol_plan_raw_sigprop_bit_count_total: usize,
    /// Profile-only exact raw magref bit count from the ordered symbol-plan probe.
    pub tier1_symbol_plan_raw_magref_bit_count_total: usize,
    /// Profile-only cleanup sign-symbol count from the ordered symbol-plan probe.
    pub tier1_symbol_plan_cleanup_sign_symbol_count_total: usize,
    /// Profile-only sigprop sign-symbol count from the ordered symbol-plan probe.
    pub tier1_symbol_plan_sigprop_sign_symbol_count_total: usize,
    /// XOR of per-block order-sensitive MQ symbol hashes from the symbol-plan probe.
    pub tier1_symbol_plan_mq_symbol_hash_xor: usize,
    /// XOR of per-block order-sensitive raw bit hashes from the symbol-plan probe.
    pub tier1_symbol_plan_raw_bit_hash_xor: usize,
    /// Profile-only MQ symbols counted by coding-pass index.
    pub tier1_pass_plan_mq_symbol_count_total: usize,
    /// Profile-only raw bypass bits counted by coding-pass index.
    pub tier1_pass_plan_raw_bit_count_total: usize,
    /// Count of block-local coding passes that emit at least one MQ symbol.
    pub tier1_pass_plan_nonempty_mq_pass_count_total: usize,
    /// Count of block-local coding passes that emit at least one raw bypass bit.
    pub tier1_pass_plan_nonempty_raw_pass_count_total: usize,
    /// Maximum MQ symbols emitted by any single block-local coding pass.
    pub max_tier1_pass_plan_mq_symbols_per_pass: usize,
    /// Maximum raw bypass bits emitted by any single block-local coding pass.
    pub max_tier1_pass_plan_raw_bits_per_pass: usize,
    /// Exact MQ symbol count from the compact token-emitter probe or gated GPU token-pack route.
    pub tier1_token_emit_mq_symbol_count_total: usize,
    /// Exact raw bypass bit count from the compact token-emitter probe or gated GPU token-pack route.
    pub tier1_token_emit_raw_bit_count_total: usize,
    /// Compact token bytes emitted by the token-emitter probe or gated GPU token-pack route.
    pub tier1_token_emit_token_bytes_total: usize,
    /// Maximum compact token bytes emitted by any one block.
    pub max_tier1_token_emit_token_bytes_per_block: usize,
    /// Segment records emitted by the token-emitter probe or gated GPU token-pack route.
    pub tier1_token_emit_segment_count_total: usize,
    /// Maximum token-emitter segment records for any one block.
    pub max_tier1_token_emit_segments_per_block: usize,
    /// XOR of per-block order-sensitive MQ symbol hashes from token emission.
    pub tier1_token_emit_mq_symbol_hash_xor: usize,
    /// XOR of per-block order-sensitive raw bit hashes from token emission.
    pub tier1_token_emit_raw_bit_hash_xor: usize,
    /// Total bytes produced by packing emitted Tier-1 tokens.
    pub tier1_token_pack_output_bytes_total: usize,
    /// Maximum token-pack output bytes for any one block.
    pub max_tier1_token_pack_output_bytes_per_block: usize,
    /// Resident Tier-1 code blocks that emitted at least one coding pass.
    pub tier1_nonzero_block_count_total: usize,
    /// Resident Tier-1 code blocks that emitted no coding passes.
    pub tier1_zero_block_count_total: usize,
    /// Missing most-significant bitplanes across resident Tier-1 code blocks.
    pub tier1_missing_bitplane_count_total: usize,
    /// Maximum missing most-significant bitplanes for any resident code block.
    pub max_tier1_missing_bitplanes_per_block: usize,
    /// Classic Tier-1 segment records emitted across resident code blocks.
    ///
    /// This remains zero for HTJ2K Tier-1, which does not use classic segment
    /// records.
    pub tier1_segment_count_total: usize,
    /// Maximum classic Tier-1 segment records emitted by any resident code block.
    pub max_tier1_segments_per_block: usize,
    /// Total host-planned packet payload-copy job slots across resident chunks.
    pub packet_payload_copy_job_capacity_total: usize,
    /// Maximum packet payload-copy job slots needed by any tile in the batch.
    pub max_packet_payload_copy_jobs_per_tile: usize,
    /// Actual packet payload-copy jobs emitted by packetization across resident chunks.
    pub packet_payload_copy_job_count_total: usize,
    /// Maximum actual packet payload-copy jobs emitted by any tile in the batch.
    pub max_packet_payload_copy_jobs_used_per_tile: usize,
    /// Actual packet payload-copy bytes emitted by packetization across resident chunks.
    pub packet_payload_copy_bytes_total: usize,
    /// Maximum actual packet payload-copy bytes emitted by any tile in the batch.
    pub max_packet_payload_copy_bytes_per_tile: usize,
    /// Packet payload-copy jobs at or below one copy-kernel stripe.
    pub packet_payload_copy_small_job_count_total: usize,
    /// Packet payload-copy jobs above one stripe and at or below 512 bytes.
    pub packet_payload_copy_medium_job_count_total: usize,
    /// Packet payload-copy jobs above 512 bytes.
    pub packet_payload_copy_large_job_count_total: usize,
    /// Packet payload-copy stripes launched by the copy kernel.
    pub packet_payload_copy_launched_stripe_count_total: usize,
    /// Packet payload-copy stripes that correspond to emitted copy jobs.
    pub packet_payload_copy_active_stripe_count_total: usize,
    /// Total packet output capacity, in bytes, across resident chunks.
    pub packet_output_capacity_total: usize,
    /// Maximum packet output capacity, in bytes, for any tile in the batch.
    pub max_packet_output_capacity: usize,
    /// Actual packet output bytes written by packetization across resident chunks.
    pub packet_output_used_bytes_total: usize,
    /// Maximum actual packet output bytes written by any tile in the batch.
    pub max_packet_output_used_bytes: usize,
    /// Codestream payload-copy bytes, in bytes, across resident chunks.
    pub codestream_payload_copy_bytes_total: usize,
    /// Codestream payload-copy threads launched by the copy kernel.
    pub codestream_payload_copy_launched_thread_count_total: usize,
    /// Estimated codestream payload-copy threads with in-range bytes to copy.
    pub codestream_payload_copy_active_thread_count_total: usize,
    /// Time spent waiting for codestream buffers.
    pub codestream_wait_duration: Duration,
    /// Alias of `codestream_wait_duration` using RCA naming.
    ///
    /// Do not sum this with `codestream_wait_duration` as an independent bucket.
    pub sync_wait_duration: Duration,
    /// Time spent materializing buffer-backed codestream bytes into host bytes.
    ///
    /// Current batch stats paths may leave this at zero. Host byte
    /// materialization timing is surfaced on `MetalLosslessEncodeOutcome` where
    /// applicable; this stage-stats bucket is reserved for stats-bearing
    /// host-output paths.
    pub host_readback_duration: Duration,
    /// Number of resident encode chunks.
    pub chunk_count: usize,
    /// Number of encoded tiles.
    pub tile_count: usize,
    /// Number of encoded code blocks.
    pub code_block_count: usize,
}

impl MetalLosslessEncodeStageStats {
    /// Return whether any non-zero timing was recorded.
    pub fn has_timings(&self) -> bool {
        self.plan_duration > Duration::ZERO
            || self.prepare_submit_duration > Duration::ZERO
            || self.coefficient_prep_duration > Duration::ZERO
            || self.deinterleave_rct_duration > Duration::ZERO
            || self.dwt53_duration > Duration::ZERO
            || self.coefficient_extract_duration > Duration::ZERO
            || self.ht_table_build_duration > Duration::ZERO
            || self.ht_buffer_allocation_duration > Duration::ZERO
            || self.ht_command_encode_duration > Duration::ZERO
            || self.ht_block_encode_duration > Duration::ZERO
            || self.classic_tier1_setup_duration > Duration::ZERO
            || self.classic_block_encode_duration > Duration::ZERO
            || self.classic_tier1_token_pack_duration > Duration::ZERO
            || self.classic_packet_plan_duration > Duration::ZERO
            || self.classic_packet_buffer_setup_duration > Duration::ZERO
            || self.classic_command_buffer_commit_duration > Duration::ZERO
            || self.result_harvest_duration > Duration::ZERO
            || self.result_status_copy_duration > Duration::ZERO
            || self.result_private_recycle_duration > Duration::ZERO
            || self.result_shared_recycle_duration > Duration::ZERO
            || self.result_codestream_collect_duration > Duration::ZERO
            || self.packet_block_prep_duration > Duration::ZERO
            || self.packetization_duration > Duration::ZERO
            || self.codestream_assembly_duration > Duration::ZERO
            || self.coefficient_prep_gpu_duration > Duration::ZERO
            || self.coefficient_deinterleave_rct_gpu_duration > Duration::ZERO
            || self.coefficient_dwt53_gpu_duration > Duration::ZERO
            || self.coefficient_dwt53_vertical_gpu_duration > Duration::ZERO
            || self.coefficient_dwt53_horizontal_gpu_duration > Duration::ZERO
            || self.coefficient_extract_gpu_duration > Duration::ZERO
            || self.coefficient_copy_gpu_duration > Duration::ZERO
            || self.gpu_elapsed_wall_duration > Duration::ZERO
            || self.classic_block_gpu_duration > Duration::ZERO
            || self.classic_tier1_density_gpu_duration > Duration::ZERO
            || self.classic_tier1_raw_pack_gpu_duration > Duration::ZERO
            || self.classic_tier1_arithmetic_pack_gpu_duration > Duration::ZERO
            || self.classic_tier1_symbol_plan_gpu_duration > Duration::ZERO
            || self.classic_tier1_pass_plan_gpu_duration > Duration::ZERO
            || self.classic_tier1_token_emit_gpu_duration > Duration::ZERO
            || self.classic_tier1_split_token_emit_gpu_duration > Duration::ZERO
            || self.classic_tier1_token_pack_gpu_duration > Duration::ZERO
            || self.ht_block_gpu_duration > Duration::ZERO
            || self.packet_block_prep_gpu_duration > Duration::ZERO
            || self.packetization_gpu_duration > Duration::ZERO
            || self.packet_payload_copy_gpu_duration > Duration::ZERO
            || self.codestream_assembly_gpu_duration > Duration::ZERO
            || self.codestream_payload_copy_gpu_duration > Duration::ZERO
            || self.codestream_wait_duration > Duration::ZERO
            || self.sync_wait_duration > Duration::ZERO
            || self.host_readback_duration > Duration::ZERO
    }

    /// Accumulate another stage-stats value using saturating duration and counter additions.
    pub fn add_assign(&mut self, other: Self) {
        self.plan_duration = self.plan_duration.saturating_add(other.plan_duration);
        self.prepare_submit_duration = self
            .prepare_submit_duration
            .saturating_add(other.prepare_submit_duration);
        self.coefficient_prep_duration = self
            .coefficient_prep_duration
            .saturating_add(other.coefficient_prep_duration);
        self.deinterleave_rct_duration = self
            .deinterleave_rct_duration
            .saturating_add(other.deinterleave_rct_duration);
        self.dwt53_duration = self.dwt53_duration.saturating_add(other.dwt53_duration);
        self.coefficient_extract_duration = self
            .coefficient_extract_duration
            .saturating_add(other.coefficient_extract_duration);
        self.ht_table_build_duration = self
            .ht_table_build_duration
            .saturating_add(other.ht_table_build_duration);
        self.ht_buffer_allocation_duration = self
            .ht_buffer_allocation_duration
            .saturating_add(other.ht_buffer_allocation_duration);
        self.ht_command_encode_duration = self
            .ht_command_encode_duration
            .saturating_add(other.ht_command_encode_duration);
        self.ht_block_encode_duration = self
            .ht_block_encode_duration
            .saturating_add(other.ht_block_encode_duration);
        self.classic_tier1_setup_duration = self
            .classic_tier1_setup_duration
            .saturating_add(other.classic_tier1_setup_duration);
        self.classic_block_encode_duration = self
            .classic_block_encode_duration
            .saturating_add(other.classic_block_encode_duration);
        self.classic_tier1_token_pack_duration = self
            .classic_tier1_token_pack_duration
            .saturating_add(other.classic_tier1_token_pack_duration);
        self.classic_packet_plan_duration = self
            .classic_packet_plan_duration
            .saturating_add(other.classic_packet_plan_duration);
        self.classic_packet_buffer_setup_duration = self
            .classic_packet_buffer_setup_duration
            .saturating_add(other.classic_packet_buffer_setup_duration);
        self.classic_command_buffer_commit_duration = self
            .classic_command_buffer_commit_duration
            .saturating_add(other.classic_command_buffer_commit_duration);
        self.result_harvest_duration = self
            .result_harvest_duration
            .saturating_add(other.result_harvest_duration);
        self.result_status_copy_duration = self
            .result_status_copy_duration
            .saturating_add(other.result_status_copy_duration);
        self.result_private_recycle_duration = self
            .result_private_recycle_duration
            .saturating_add(other.result_private_recycle_duration);
        self.result_shared_recycle_duration = self
            .result_shared_recycle_duration
            .saturating_add(other.result_shared_recycle_duration);
        self.result_codestream_collect_duration = self
            .result_codestream_collect_duration
            .saturating_add(other.result_codestream_collect_duration);
        self.packet_block_prep_duration = self
            .packet_block_prep_duration
            .saturating_add(other.packet_block_prep_duration);
        self.packetization_duration = self
            .packetization_duration
            .saturating_add(other.packetization_duration);
        self.codestream_assembly_duration = self
            .codestream_assembly_duration
            .saturating_add(other.codestream_assembly_duration);
        self.coefficient_prep_gpu_duration = self
            .coefficient_prep_gpu_duration
            .saturating_add(other.coefficient_prep_gpu_duration);
        self.coefficient_deinterleave_rct_gpu_duration = self
            .coefficient_deinterleave_rct_gpu_duration
            .saturating_add(other.coefficient_deinterleave_rct_gpu_duration);
        self.coefficient_dwt53_gpu_duration = self
            .coefficient_dwt53_gpu_duration
            .saturating_add(other.coefficient_dwt53_gpu_duration);
        self.coefficient_dwt53_vertical_gpu_duration = self
            .coefficient_dwt53_vertical_gpu_duration
            .saturating_add(other.coefficient_dwt53_vertical_gpu_duration);
        self.coefficient_dwt53_horizontal_gpu_duration = self
            .coefficient_dwt53_horizontal_gpu_duration
            .saturating_add(other.coefficient_dwt53_horizontal_gpu_duration);
        self.coefficient_extract_gpu_duration = self
            .coefficient_extract_gpu_duration
            .saturating_add(other.coefficient_extract_gpu_duration);
        self.coefficient_copy_gpu_duration = self
            .coefficient_copy_gpu_duration
            .saturating_add(other.coefficient_copy_gpu_duration);
        self.gpu_elapsed_wall_duration = self
            .gpu_elapsed_wall_duration
            .saturating_add(other.gpu_elapsed_wall_duration);
        self.classic_block_gpu_duration = self
            .classic_block_gpu_duration
            .saturating_add(other.classic_block_gpu_duration);
        self.classic_tier1_density_gpu_duration = self
            .classic_tier1_density_gpu_duration
            .saturating_add(other.classic_tier1_density_gpu_duration);
        self.classic_tier1_raw_pack_gpu_duration = self
            .classic_tier1_raw_pack_gpu_duration
            .saturating_add(other.classic_tier1_raw_pack_gpu_duration);
        self.classic_tier1_arithmetic_pack_gpu_duration = self
            .classic_tier1_arithmetic_pack_gpu_duration
            .saturating_add(other.classic_tier1_arithmetic_pack_gpu_duration);
        self.classic_tier1_symbol_plan_gpu_duration = self
            .classic_tier1_symbol_plan_gpu_duration
            .saturating_add(other.classic_tier1_symbol_plan_gpu_duration);
        self.classic_tier1_pass_plan_gpu_duration = self
            .classic_tier1_pass_plan_gpu_duration
            .saturating_add(other.classic_tier1_pass_plan_gpu_duration);
        self.classic_tier1_token_emit_gpu_duration = self
            .classic_tier1_token_emit_gpu_duration
            .saturating_add(other.classic_tier1_token_emit_gpu_duration);
        self.classic_tier1_split_token_emit_gpu_duration = self
            .classic_tier1_split_token_emit_gpu_duration
            .saturating_add(other.classic_tier1_split_token_emit_gpu_duration);
        self.classic_tier1_token_pack_gpu_duration = self
            .classic_tier1_token_pack_gpu_duration
            .saturating_add(other.classic_tier1_token_pack_gpu_duration);
        self.ht_block_gpu_duration = self
            .ht_block_gpu_duration
            .saturating_add(other.ht_block_gpu_duration);
        self.packet_block_prep_gpu_duration = self
            .packet_block_prep_gpu_duration
            .saturating_add(other.packet_block_prep_gpu_duration);
        self.packetization_gpu_duration = self
            .packetization_gpu_duration
            .saturating_add(other.packetization_gpu_duration);
        self.packet_payload_copy_gpu_duration = self
            .packet_payload_copy_gpu_duration
            .saturating_add(other.packet_payload_copy_gpu_duration);
        self.codestream_assembly_gpu_duration = self
            .codestream_assembly_gpu_duration
            .saturating_add(other.codestream_assembly_gpu_duration);
        self.codestream_payload_copy_gpu_duration = self
            .codestream_payload_copy_gpu_duration
            .saturating_add(other.codestream_payload_copy_gpu_duration);
        self.tier1_output_capacity_total = self
            .tier1_output_capacity_total
            .saturating_add(other.tier1_output_capacity_total);
        self.max_tier1_output_capacity = self
            .max_tier1_output_capacity
            .max(other.max_tier1_output_capacity);
        self.tier1_output_used_bytes_total = self
            .tier1_output_used_bytes_total
            .saturating_add(other.tier1_output_used_bytes_total);
        self.max_tier1_output_used_bytes = self
            .max_tier1_output_used_bytes
            .max(other.max_tier1_output_used_bytes);
        self.tier1_segment_capacity_total = self
            .tier1_segment_capacity_total
            .saturating_add(other.tier1_segment_capacity_total);
        self.max_tier1_segment_capacity_per_block = self
            .max_tier1_segment_capacity_per_block
            .max(other.max_tier1_segment_capacity_per_block);
        self.tier1_coding_pass_count_total = self
            .tier1_coding_pass_count_total
            .saturating_add(other.tier1_coding_pass_count_total);
        self.max_tier1_coding_passes_per_block = self
            .max_tier1_coding_passes_per_block
            .max(other.max_tier1_coding_passes_per_block);
        self.tier1_arithmetic_pass_count_total = self
            .tier1_arithmetic_pass_count_total
            .saturating_add(other.tier1_arithmetic_pass_count_total);
        self.tier1_raw_pass_count_total = self
            .tier1_raw_pass_count_total
            .saturating_add(other.tier1_raw_pass_count_total);
        self.tier1_cleanup_pass_count_total = self
            .tier1_cleanup_pass_count_total
            .saturating_add(other.tier1_cleanup_pass_count_total);
        self.tier1_sigprop_pass_count_total = self
            .tier1_sigprop_pass_count_total
            .saturating_add(other.tier1_sigprop_pass_count_total);
        self.tier1_magref_pass_count_total = self
            .tier1_magref_pass_count_total
            .saturating_add(other.tier1_magref_pass_count_total);
        self.tier1_arithmetic_cleanup_pass_count_total = self
            .tier1_arithmetic_cleanup_pass_count_total
            .saturating_add(other.tier1_arithmetic_cleanup_pass_count_total);
        self.tier1_arithmetic_sigprop_pass_count_total = self
            .tier1_arithmetic_sigprop_pass_count_total
            .saturating_add(other.tier1_arithmetic_sigprop_pass_count_total);
        self.tier1_arithmetic_magref_pass_count_total = self
            .tier1_arithmetic_magref_pass_count_total
            .saturating_add(other.tier1_arithmetic_magref_pass_count_total);
        self.tier1_raw_sigprop_pass_count_total = self
            .tier1_raw_sigprop_pass_count_total
            .saturating_add(other.tier1_raw_sigprop_pass_count_total);
        self.tier1_raw_magref_pass_count_total = self
            .tier1_raw_magref_pass_count_total
            .saturating_add(other.tier1_raw_magref_pass_count_total);
        self.tier1_full_scan_coeff_visit_count_total = self
            .tier1_full_scan_coeff_visit_count_total
            .saturating_add(other.tier1_full_scan_coeff_visit_count_total);
        self.tier1_arithmetic_scan_coeff_visit_count_total = self
            .tier1_arithmetic_scan_coeff_visit_count_total
            .saturating_add(other.tier1_arithmetic_scan_coeff_visit_count_total);
        self.tier1_raw_scan_coeff_visit_count_total = self
            .tier1_raw_scan_coeff_visit_count_total
            .saturating_add(other.tier1_raw_scan_coeff_visit_count_total);
        self.tier1_cleanup_scan_coeff_visit_count_total = self
            .tier1_cleanup_scan_coeff_visit_count_total
            .saturating_add(other.tier1_cleanup_scan_coeff_visit_count_total);
        self.tier1_sigprop_scan_coeff_visit_count_total = self
            .tier1_sigprop_scan_coeff_visit_count_total
            .saturating_add(other.tier1_sigprop_scan_coeff_visit_count_total);
        self.tier1_magref_scan_coeff_visit_count_total = self
            .tier1_magref_scan_coeff_visit_count_total
            .saturating_add(other.tier1_magref_scan_coeff_visit_count_total);
        self.max_tier1_full_scan_coeff_visits_per_block = self
            .max_tier1_full_scan_coeff_visits_per_block
            .max(other.max_tier1_full_scan_coeff_visits_per_block);
        self.tier1_sigprop_active_candidate_count_total = self
            .tier1_sigprop_active_candidate_count_total
            .saturating_add(other.tier1_sigprop_active_candidate_count_total);
        self.tier1_sigprop_new_significant_count_total = self
            .tier1_sigprop_new_significant_count_total
            .saturating_add(other.tier1_sigprop_new_significant_count_total);
        self.tier1_magref_active_candidate_count_total = self
            .tier1_magref_active_candidate_count_total
            .saturating_add(other.tier1_magref_active_candidate_count_total);
        self.tier1_arithmetic_sigprop_active_candidate_count_total = self
            .tier1_arithmetic_sigprop_active_candidate_count_total
            .saturating_add(other.tier1_arithmetic_sigprop_active_candidate_count_total);
        self.tier1_arithmetic_sigprop_new_significant_count_total = self
            .tier1_arithmetic_sigprop_new_significant_count_total
            .saturating_add(other.tier1_arithmetic_sigprop_new_significant_count_total);
        self.tier1_raw_sigprop_active_candidate_count_total = self
            .tier1_raw_sigprop_active_candidate_count_total
            .saturating_add(other.tier1_raw_sigprop_active_candidate_count_total);
        self.tier1_raw_sigprop_new_significant_count_total = self
            .tier1_raw_sigprop_new_significant_count_total
            .saturating_add(other.tier1_raw_sigprop_new_significant_count_total);
        self.tier1_arithmetic_magref_active_candidate_count_total = self
            .tier1_arithmetic_magref_active_candidate_count_total
            .saturating_add(other.tier1_arithmetic_magref_active_candidate_count_total);
        self.tier1_raw_magref_active_candidate_count_total = self
            .tier1_raw_magref_active_candidate_count_total
            .saturating_add(other.tier1_raw_magref_active_candidate_count_total);
        self.tier1_cleanup_active_candidate_count_total = self
            .tier1_cleanup_active_candidate_count_total
            .saturating_add(other.tier1_cleanup_active_candidate_count_total);
        self.tier1_cleanup_new_significant_count_total = self
            .tier1_cleanup_new_significant_count_total
            .saturating_add(other.tier1_cleanup_new_significant_count_total);
        self.tier1_cleanup_rlc_stripe_count_total = self
            .tier1_cleanup_rlc_stripe_count_total
            .saturating_add(other.tier1_cleanup_rlc_stripe_count_total);
        self.tier1_cleanup_rlc_zero_stripe_count_total = self
            .tier1_cleanup_rlc_zero_stripe_count_total
            .saturating_add(other.tier1_cleanup_rlc_zero_stripe_count_total);
        self.tier1_symbol_plan_mq_symbol_count_total = self
            .tier1_symbol_plan_mq_symbol_count_total
            .saturating_add(other.tier1_symbol_plan_mq_symbol_count_total);
        self.tier1_symbol_plan_raw_bit_count_total = self
            .tier1_symbol_plan_raw_bit_count_total
            .saturating_add(other.tier1_symbol_plan_raw_bit_count_total);
        self.max_tier1_symbol_plan_mq_symbols_per_block = self
            .max_tier1_symbol_plan_mq_symbols_per_block
            .max(other.max_tier1_symbol_plan_mq_symbols_per_block);
        self.max_tier1_symbol_plan_raw_bits_per_block = self
            .max_tier1_symbol_plan_raw_bits_per_block
            .max(other.max_tier1_symbol_plan_raw_bits_per_block);
        self.tier1_symbol_plan_packed_token_bytes_total = self
            .tier1_symbol_plan_packed_token_bytes_total
            .saturating_add(other.tier1_symbol_plan_packed_token_bytes_total);
        self.max_tier1_symbol_plan_packed_token_bytes_per_block = self
            .max_tier1_symbol_plan_packed_token_bytes_per_block
            .max(other.max_tier1_symbol_plan_packed_token_bytes_per_block);
        self.tier1_symbol_plan_cleanup_mq_symbol_count_total = self
            .tier1_symbol_plan_cleanup_mq_symbol_count_total
            .saturating_add(other.tier1_symbol_plan_cleanup_mq_symbol_count_total);
        self.tier1_symbol_plan_sigprop_mq_symbol_count_total = self
            .tier1_symbol_plan_sigprop_mq_symbol_count_total
            .saturating_add(other.tier1_symbol_plan_sigprop_mq_symbol_count_total);
        self.tier1_symbol_plan_magref_mq_symbol_count_total = self
            .tier1_symbol_plan_magref_mq_symbol_count_total
            .saturating_add(other.tier1_symbol_plan_magref_mq_symbol_count_total);
        self.tier1_symbol_plan_raw_sigprop_bit_count_total = self
            .tier1_symbol_plan_raw_sigprop_bit_count_total
            .saturating_add(other.tier1_symbol_plan_raw_sigprop_bit_count_total);
        self.tier1_symbol_plan_raw_magref_bit_count_total = self
            .tier1_symbol_plan_raw_magref_bit_count_total
            .saturating_add(other.tier1_symbol_plan_raw_magref_bit_count_total);
        self.tier1_symbol_plan_cleanup_sign_symbol_count_total = self
            .tier1_symbol_plan_cleanup_sign_symbol_count_total
            .saturating_add(other.tier1_symbol_plan_cleanup_sign_symbol_count_total);
        self.tier1_symbol_plan_sigprop_sign_symbol_count_total = self
            .tier1_symbol_plan_sigprop_sign_symbol_count_total
            .saturating_add(other.tier1_symbol_plan_sigprop_sign_symbol_count_total);
        self.tier1_symbol_plan_mq_symbol_hash_xor ^= other.tier1_symbol_plan_mq_symbol_hash_xor;
        self.tier1_symbol_plan_raw_bit_hash_xor ^= other.tier1_symbol_plan_raw_bit_hash_xor;
        self.tier1_pass_plan_mq_symbol_count_total = self
            .tier1_pass_plan_mq_symbol_count_total
            .saturating_add(other.tier1_pass_plan_mq_symbol_count_total);
        self.tier1_pass_plan_raw_bit_count_total = self
            .tier1_pass_plan_raw_bit_count_total
            .saturating_add(other.tier1_pass_plan_raw_bit_count_total);
        self.tier1_pass_plan_nonempty_mq_pass_count_total = self
            .tier1_pass_plan_nonempty_mq_pass_count_total
            .saturating_add(other.tier1_pass_plan_nonempty_mq_pass_count_total);
        self.tier1_pass_plan_nonempty_raw_pass_count_total = self
            .tier1_pass_plan_nonempty_raw_pass_count_total
            .saturating_add(other.tier1_pass_plan_nonempty_raw_pass_count_total);
        self.max_tier1_pass_plan_mq_symbols_per_pass = self
            .max_tier1_pass_plan_mq_symbols_per_pass
            .max(other.max_tier1_pass_plan_mq_symbols_per_pass);
        self.max_tier1_pass_plan_raw_bits_per_pass = self
            .max_tier1_pass_plan_raw_bits_per_pass
            .max(other.max_tier1_pass_plan_raw_bits_per_pass);
        self.tier1_token_emit_mq_symbol_count_total = self
            .tier1_token_emit_mq_symbol_count_total
            .saturating_add(other.tier1_token_emit_mq_symbol_count_total);
        self.tier1_token_emit_raw_bit_count_total = self
            .tier1_token_emit_raw_bit_count_total
            .saturating_add(other.tier1_token_emit_raw_bit_count_total);
        self.tier1_token_emit_token_bytes_total = self
            .tier1_token_emit_token_bytes_total
            .saturating_add(other.tier1_token_emit_token_bytes_total);
        self.max_tier1_token_emit_token_bytes_per_block = self
            .max_tier1_token_emit_token_bytes_per_block
            .max(other.max_tier1_token_emit_token_bytes_per_block);
        self.tier1_token_emit_segment_count_total = self
            .tier1_token_emit_segment_count_total
            .saturating_add(other.tier1_token_emit_segment_count_total);
        self.max_tier1_token_emit_segments_per_block = self
            .max_tier1_token_emit_segments_per_block
            .max(other.max_tier1_token_emit_segments_per_block);
        self.tier1_token_emit_mq_symbol_hash_xor ^= other.tier1_token_emit_mq_symbol_hash_xor;
        self.tier1_token_emit_raw_bit_hash_xor ^= other.tier1_token_emit_raw_bit_hash_xor;
        self.tier1_token_pack_output_bytes_total = self
            .tier1_token_pack_output_bytes_total
            .saturating_add(other.tier1_token_pack_output_bytes_total);
        self.max_tier1_token_pack_output_bytes_per_block = self
            .max_tier1_token_pack_output_bytes_per_block
            .max(other.max_tier1_token_pack_output_bytes_per_block);
        self.tier1_nonzero_block_count_total = self
            .tier1_nonzero_block_count_total
            .saturating_add(other.tier1_nonzero_block_count_total);
        self.tier1_zero_block_count_total = self
            .tier1_zero_block_count_total
            .saturating_add(other.tier1_zero_block_count_total);
        self.tier1_missing_bitplane_count_total = self
            .tier1_missing_bitplane_count_total
            .saturating_add(other.tier1_missing_bitplane_count_total);
        self.max_tier1_missing_bitplanes_per_block = self
            .max_tier1_missing_bitplanes_per_block
            .max(other.max_tier1_missing_bitplanes_per_block);
        self.tier1_segment_count_total = self
            .tier1_segment_count_total
            .saturating_add(other.tier1_segment_count_total);
        self.max_tier1_segments_per_block = self
            .max_tier1_segments_per_block
            .max(other.max_tier1_segments_per_block);
        self.packet_payload_copy_job_capacity_total = self
            .packet_payload_copy_job_capacity_total
            .saturating_add(other.packet_payload_copy_job_capacity_total);
        self.max_packet_payload_copy_jobs_per_tile = self
            .max_packet_payload_copy_jobs_per_tile
            .max(other.max_packet_payload_copy_jobs_per_tile);
        self.packet_payload_copy_job_count_total = self
            .packet_payload_copy_job_count_total
            .saturating_add(other.packet_payload_copy_job_count_total);
        self.max_packet_payload_copy_jobs_used_per_tile = self
            .max_packet_payload_copy_jobs_used_per_tile
            .max(other.max_packet_payload_copy_jobs_used_per_tile);
        self.packet_payload_copy_bytes_total = self
            .packet_payload_copy_bytes_total
            .saturating_add(other.packet_payload_copy_bytes_total);
        self.max_packet_payload_copy_bytes_per_tile = self
            .max_packet_payload_copy_bytes_per_tile
            .max(other.max_packet_payload_copy_bytes_per_tile);
        self.packet_payload_copy_small_job_count_total = self
            .packet_payload_copy_small_job_count_total
            .saturating_add(other.packet_payload_copy_small_job_count_total);
        self.packet_payload_copy_medium_job_count_total = self
            .packet_payload_copy_medium_job_count_total
            .saturating_add(other.packet_payload_copy_medium_job_count_total);
        self.packet_payload_copy_large_job_count_total = self
            .packet_payload_copy_large_job_count_total
            .saturating_add(other.packet_payload_copy_large_job_count_total);
        self.packet_payload_copy_launched_stripe_count_total = self
            .packet_payload_copy_launched_stripe_count_total
            .saturating_add(other.packet_payload_copy_launched_stripe_count_total);
        self.packet_payload_copy_active_stripe_count_total = self
            .packet_payload_copy_active_stripe_count_total
            .saturating_add(other.packet_payload_copy_active_stripe_count_total);
        self.packet_output_capacity_total = self
            .packet_output_capacity_total
            .saturating_add(other.packet_output_capacity_total);
        self.max_packet_output_capacity = self
            .max_packet_output_capacity
            .max(other.max_packet_output_capacity);
        self.packet_output_used_bytes_total = self
            .packet_output_used_bytes_total
            .saturating_add(other.packet_output_used_bytes_total);
        self.max_packet_output_used_bytes = self
            .max_packet_output_used_bytes
            .max(other.max_packet_output_used_bytes);
        self.codestream_payload_copy_bytes_total = self
            .codestream_payload_copy_bytes_total
            .saturating_add(other.codestream_payload_copy_bytes_total);
        self.codestream_payload_copy_launched_thread_count_total = self
            .codestream_payload_copy_launched_thread_count_total
            .saturating_add(other.codestream_payload_copy_launched_thread_count_total);
        self.codestream_payload_copy_active_thread_count_total = self
            .codestream_payload_copy_active_thread_count_total
            .saturating_add(other.codestream_payload_copy_active_thread_count_total);
        self.codestream_wait_duration = self
            .codestream_wait_duration
            .saturating_add(other.codestream_wait_duration);
        self.sync_wait_duration = self
            .sync_wait_duration
            .saturating_add(other.sync_wait_duration);
        self.host_readback_duration = self
            .host_readback_duration
            .saturating_add(other.host_readback_duration);
        self.chunk_count = self.chunk_count.saturating_add(other.chunk_count);
        self.tile_count = self.tile_count.saturating_add(other.tile_count);
        self.code_block_count = self.code_block_count.saturating_add(other.code_block_count);
    }
}

#[cfg(target_os = "macos")]
impl From<compute::J2kResidentEncodeStageStats> for MetalLosslessEncodeStageStats {
    fn from(stats: compute::J2kResidentEncodeStageStats) -> Self {
        Self {
            coefficient_prep_duration: stats.coefficient_prep_duration,
            deinterleave_rct_duration: stats.deinterleave_rct_duration,
            dwt53_duration: stats.dwt53_duration,
            coefficient_extract_duration: stats.coefficient_extract_duration,
            ht_table_build_duration: stats.ht_table_build_duration,
            ht_buffer_allocation_duration: stats.ht_buffer_allocation_duration,
            ht_command_encode_duration: stats.ht_command_encode_duration,
            ht_block_encode_duration: stats.ht_block_encode_duration,
            classic_tier1_setup_duration: stats.classic_tier1_setup_duration,
            classic_block_encode_duration: stats.classic_block_encode_duration,
            classic_tier1_token_pack_duration: stats.classic_tier1_token_pack_duration,
            classic_packet_plan_duration: stats.classic_packet_plan_duration,
            classic_packet_buffer_setup_duration: stats.classic_packet_buffer_setup_duration,
            classic_command_buffer_commit_duration: stats.classic_command_buffer_commit_duration,
            result_harvest_duration: stats.result_harvest_duration,
            result_status_copy_duration: stats.result_status_copy_duration,
            result_private_recycle_duration: stats.result_private_recycle_duration,
            result_shared_recycle_duration: stats.result_shared_recycle_duration,
            result_codestream_collect_duration: stats.result_codestream_collect_duration,
            packet_block_prep_duration: stats.packet_block_prep_duration,
            packetization_duration: stats.packetization_duration,
            codestream_assembly_duration: stats.codestream_assembly_duration,
            coefficient_prep_gpu_duration: stats.coefficient_prep_gpu_duration,
            coefficient_deinterleave_rct_gpu_duration: stats
                .coefficient_deinterleave_rct_gpu_duration,
            coefficient_dwt53_gpu_duration: stats.coefficient_dwt53_gpu_duration,
            coefficient_dwt53_vertical_gpu_duration: stats.coefficient_dwt53_vertical_gpu_duration,
            coefficient_dwt53_horizontal_gpu_duration: stats
                .coefficient_dwt53_horizontal_gpu_duration,
            coefficient_extract_gpu_duration: stats.coefficient_extract_gpu_duration,
            coefficient_copy_gpu_duration: stats.coefficient_copy_gpu_duration,
            gpu_elapsed_wall_duration: stats.gpu_elapsed_wall_duration,
            classic_block_gpu_duration: stats.classic_block_gpu_duration,
            classic_tier1_density_gpu_duration: stats.classic_tier1_density_gpu_duration,
            classic_tier1_raw_pack_gpu_duration: stats.classic_tier1_raw_pack_gpu_duration,
            classic_tier1_arithmetic_pack_gpu_duration: stats
                .classic_tier1_arithmetic_pack_gpu_duration,
            classic_tier1_symbol_plan_gpu_duration: stats.classic_tier1_symbol_plan_gpu_duration,
            classic_tier1_pass_plan_gpu_duration: stats.classic_tier1_pass_plan_gpu_duration,
            classic_tier1_token_emit_gpu_duration: stats.classic_tier1_token_emit_gpu_duration,
            classic_tier1_split_token_emit_gpu_duration: stats
                .classic_tier1_split_token_emit_gpu_duration,
            classic_tier1_token_pack_gpu_duration: stats.classic_tier1_token_pack_gpu_duration,
            ht_block_gpu_duration: stats.ht_block_gpu_duration,
            packet_block_prep_gpu_duration: stats.packet_block_prep_gpu_duration,
            packetization_gpu_duration: stats.packetization_gpu_duration,
            packet_payload_copy_gpu_duration: stats.packet_payload_copy_gpu_duration,
            codestream_assembly_gpu_duration: stats.codestream_assembly_gpu_duration,
            codestream_payload_copy_gpu_duration: stats.codestream_payload_copy_gpu_duration,
            tier1_output_capacity_total: stats.tier1_output_capacity_total,
            max_tier1_output_capacity: stats.max_tier1_output_capacity,
            tier1_output_used_bytes_total: stats.tier1_output_used_bytes_total,
            max_tier1_output_used_bytes: stats.max_tier1_output_used_bytes,
            tier1_segment_capacity_total: stats.tier1_segment_capacity_total,
            max_tier1_segment_capacity_per_block: stats.max_tier1_segment_capacity_per_block,
            tier1_coding_pass_count_total: stats.tier1_coding_pass_count_total,
            max_tier1_coding_passes_per_block: stats.max_tier1_coding_passes_per_block,
            tier1_arithmetic_pass_count_total: stats.tier1_arithmetic_pass_count_total,
            tier1_raw_pass_count_total: stats.tier1_raw_pass_count_total,
            tier1_cleanup_pass_count_total: stats.tier1_cleanup_pass_count_total,
            tier1_sigprop_pass_count_total: stats.tier1_sigprop_pass_count_total,
            tier1_magref_pass_count_total: stats.tier1_magref_pass_count_total,
            tier1_arithmetic_cleanup_pass_count_total: stats
                .tier1_arithmetic_cleanup_pass_count_total,
            tier1_arithmetic_sigprop_pass_count_total: stats
                .tier1_arithmetic_sigprop_pass_count_total,
            tier1_arithmetic_magref_pass_count_total: stats
                .tier1_arithmetic_magref_pass_count_total,
            tier1_raw_sigprop_pass_count_total: stats.tier1_raw_sigprop_pass_count_total,
            tier1_raw_magref_pass_count_total: stats.tier1_raw_magref_pass_count_total,
            tier1_full_scan_coeff_visit_count_total: stats.tier1_full_scan_coeff_visit_count_total,
            tier1_arithmetic_scan_coeff_visit_count_total: stats
                .tier1_arithmetic_scan_coeff_visit_count_total,
            tier1_raw_scan_coeff_visit_count_total: stats.tier1_raw_scan_coeff_visit_count_total,
            tier1_cleanup_scan_coeff_visit_count_total: stats
                .tier1_cleanup_scan_coeff_visit_count_total,
            tier1_sigprop_scan_coeff_visit_count_total: stats
                .tier1_sigprop_scan_coeff_visit_count_total,
            tier1_magref_scan_coeff_visit_count_total: stats
                .tier1_magref_scan_coeff_visit_count_total,
            max_tier1_full_scan_coeff_visits_per_block: stats
                .max_tier1_full_scan_coeff_visits_per_block,
            tier1_sigprop_active_candidate_count_total: stats
                .tier1_sigprop_active_candidate_count_total,
            tier1_sigprop_new_significant_count_total: stats
                .tier1_sigprop_new_significant_count_total,
            tier1_magref_active_candidate_count_total: stats
                .tier1_magref_active_candidate_count_total,
            tier1_arithmetic_sigprop_active_candidate_count_total: stats
                .tier1_arithmetic_sigprop_active_candidate_count_total,
            tier1_arithmetic_sigprop_new_significant_count_total: stats
                .tier1_arithmetic_sigprop_new_significant_count_total,
            tier1_raw_sigprop_active_candidate_count_total: stats
                .tier1_raw_sigprop_active_candidate_count_total,
            tier1_raw_sigprop_new_significant_count_total: stats
                .tier1_raw_sigprop_new_significant_count_total,
            tier1_arithmetic_magref_active_candidate_count_total: stats
                .tier1_arithmetic_magref_active_candidate_count_total,
            tier1_raw_magref_active_candidate_count_total: stats
                .tier1_raw_magref_active_candidate_count_total,
            tier1_cleanup_active_candidate_count_total: stats
                .tier1_cleanup_active_candidate_count_total,
            tier1_cleanup_new_significant_count_total: stats
                .tier1_cleanup_new_significant_count_total,
            tier1_cleanup_rlc_stripe_count_total: stats.tier1_cleanup_rlc_stripe_count_total,
            tier1_cleanup_rlc_zero_stripe_count_total: stats
                .tier1_cleanup_rlc_zero_stripe_count_total,
            tier1_symbol_plan_mq_symbol_count_total: stats.tier1_symbol_plan_mq_symbol_count_total,
            tier1_symbol_plan_raw_bit_count_total: stats.tier1_symbol_plan_raw_bit_count_total,
            max_tier1_symbol_plan_mq_symbols_per_block: stats
                .max_tier1_symbol_plan_mq_symbols_per_block,
            max_tier1_symbol_plan_raw_bits_per_block: stats
                .max_tier1_symbol_plan_raw_bits_per_block,
            tier1_symbol_plan_packed_token_bytes_total: stats
                .tier1_symbol_plan_packed_token_bytes_total,
            max_tier1_symbol_plan_packed_token_bytes_per_block: stats
                .max_tier1_symbol_plan_packed_token_bytes_per_block,
            tier1_symbol_plan_cleanup_mq_symbol_count_total: stats
                .tier1_symbol_plan_cleanup_mq_symbol_count_total,
            tier1_symbol_plan_sigprop_mq_symbol_count_total: stats
                .tier1_symbol_plan_sigprop_mq_symbol_count_total,
            tier1_symbol_plan_magref_mq_symbol_count_total: stats
                .tier1_symbol_plan_magref_mq_symbol_count_total,
            tier1_symbol_plan_raw_sigprop_bit_count_total: stats
                .tier1_symbol_plan_raw_sigprop_bit_count_total,
            tier1_symbol_plan_raw_magref_bit_count_total: stats
                .tier1_symbol_plan_raw_magref_bit_count_total,
            tier1_symbol_plan_cleanup_sign_symbol_count_total: stats
                .tier1_symbol_plan_cleanup_sign_symbol_count_total,
            tier1_symbol_plan_sigprop_sign_symbol_count_total: stats
                .tier1_symbol_plan_sigprop_sign_symbol_count_total,
            tier1_symbol_plan_mq_symbol_hash_xor: stats.tier1_symbol_plan_mq_symbol_hash_xor,
            tier1_symbol_plan_raw_bit_hash_xor: stats.tier1_symbol_plan_raw_bit_hash_xor,
            tier1_pass_plan_mq_symbol_count_total: stats.tier1_pass_plan_mq_symbol_count_total,
            tier1_pass_plan_raw_bit_count_total: stats.tier1_pass_plan_raw_bit_count_total,
            tier1_pass_plan_nonempty_mq_pass_count_total: stats
                .tier1_pass_plan_nonempty_mq_pass_count_total,
            tier1_pass_plan_nonempty_raw_pass_count_total: stats
                .tier1_pass_plan_nonempty_raw_pass_count_total,
            max_tier1_pass_plan_mq_symbols_per_pass: stats.max_tier1_pass_plan_mq_symbols_per_pass,
            max_tier1_pass_plan_raw_bits_per_pass: stats.max_tier1_pass_plan_raw_bits_per_pass,
            tier1_token_emit_mq_symbol_count_total: stats.tier1_token_emit_mq_symbol_count_total,
            tier1_token_emit_raw_bit_count_total: stats.tier1_token_emit_raw_bit_count_total,
            tier1_token_emit_token_bytes_total: stats.tier1_token_emit_token_bytes_total,
            max_tier1_token_emit_token_bytes_per_block: stats
                .max_tier1_token_emit_token_bytes_per_block,
            tier1_token_emit_segment_count_total: stats.tier1_token_emit_segment_count_total,
            max_tier1_token_emit_segments_per_block: stats.max_tier1_token_emit_segments_per_block,
            tier1_token_emit_mq_symbol_hash_xor: stats.tier1_token_emit_mq_symbol_hash_xor,
            tier1_token_emit_raw_bit_hash_xor: stats.tier1_token_emit_raw_bit_hash_xor,
            tier1_token_pack_output_bytes_total: stats.tier1_token_pack_output_bytes_total,
            max_tier1_token_pack_output_bytes_per_block: stats
                .max_tier1_token_pack_output_bytes_per_block,
            tier1_nonzero_block_count_total: stats.tier1_nonzero_block_count_total,
            tier1_zero_block_count_total: stats.tier1_zero_block_count_total,
            tier1_missing_bitplane_count_total: stats.tier1_missing_bitplane_count_total,
            max_tier1_missing_bitplanes_per_block: stats.max_tier1_missing_bitplanes_per_block,
            tier1_segment_count_total: stats.tier1_segment_count_total,
            max_tier1_segments_per_block: stats.max_tier1_segments_per_block,
            packet_payload_copy_job_capacity_total: stats.packet_payload_copy_job_capacity_total,
            max_packet_payload_copy_jobs_per_tile: stats.max_packet_payload_copy_jobs_per_tile,
            packet_payload_copy_job_count_total: stats.packet_payload_copy_job_count_total,
            max_packet_payload_copy_jobs_used_per_tile: stats
                .max_packet_payload_copy_jobs_used_per_tile,
            packet_payload_copy_bytes_total: stats.packet_payload_copy_bytes_total,
            max_packet_payload_copy_bytes_per_tile: stats.max_packet_payload_copy_bytes_per_tile,
            packet_payload_copy_small_job_count_total: stats
                .packet_payload_copy_small_job_count_total,
            packet_payload_copy_medium_job_count_total: stats
                .packet_payload_copy_medium_job_count_total,
            packet_payload_copy_large_job_count_total: stats
                .packet_payload_copy_large_job_count_total,
            packet_payload_copy_launched_stripe_count_total: stats
                .packet_payload_copy_launched_stripe_count_total,
            packet_payload_copy_active_stripe_count_total: stats
                .packet_payload_copy_active_stripe_count_total,
            packet_output_capacity_total: stats.packet_output_capacity_total,
            max_packet_output_capacity: stats.max_packet_output_capacity,
            packet_output_used_bytes_total: stats.packet_output_used_bytes_total,
            max_packet_output_used_bytes: stats.max_packet_output_used_bytes,
            codestream_payload_copy_bytes_total: stats.codestream_payload_copy_bytes_total,
            codestream_payload_copy_launched_thread_count_total: stats
                .codestream_payload_copy_launched_thread_count_total,
            codestream_payload_copy_active_thread_count_total: stats
                .codestream_payload_copy_active_thread_count_total,
            code_block_count: stats.code_block_count,
            ..Self::default()
        }
    }
}

#[cfg(any(target_os = "macos", test))]
fn add_resident_prep_duration(
    stats: &mut MetalLosslessEncodeBatchStats,
    duration: Duration,
    profile_stages: bool,
) {
    if !profile_stages {
        return;
    }
    stats.stage_stats.coefficient_prep_duration = stats
        .stage_stats
        .coefficient_prep_duration
        .saturating_add(duration);
}

#[cfg(any(target_os = "macos", test))]
fn add_resident_prep_wall_duration(
    stats: &mut MetalLosslessEncodeBatchStats,
    wall_duration: Duration,
    profile_stages: bool,
) {
    add_resident_prep_duration(stats, wall_duration, profile_stages);
}

/// Resolved resident Metal lossless J2K/HTJ2K tile batch encode metrics.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MetalLosslessEncodeBatchStats {
    /// Caller-requested maximum number of in-flight tiles.
    pub configured_inflight_tiles: Option<usize>,
    /// Effective maximum number of in-flight tiles after clamping.
    pub effective_inflight_tiles: usize,
    /// Caller-requested resident encode memory budget in bytes.
    pub configured_memory_budget_bytes: Option<usize>,
    /// Effective resident encode memory budget in bytes.
    pub effective_memory_budget_bytes: usize,
    /// Estimated peak resident memory required per tile.
    pub estimated_peak_bytes_per_tile: usize,
    /// Maximum observed in-flight tiles during the batch.
    pub max_observed_inflight_tiles: usize,
    /// End-to-end wall time for the batch encode.
    pub encode_wall_duration: Duration,
    /// Resident encode stage timing summary.
    pub stage_stats: MetalLosslessEncodeStageStats,
}

/// Resident Metal lossless J2K/HTJ2K tile batch output and batch-level metrics.
pub struct MetalLosslessBufferEncodeBatchOutcome {
    /// Per-tile buffer-backed encode outcomes.
    pub outcomes: Vec<MetalLosslessBufferEncodeOutcome>,
    /// Batch-level resident encode metrics.
    pub stats: MetalLosslessEncodeBatchStats,
}

#[cfg(target_os = "macos")]
/// Submitted single-tile Metal encode that resolves to host codestream bytes.
pub struct SubmittedJ2kLosslessMetalEncode {
    inner: SubmittedJ2kLosslessMetalEncodeBatch,
}

#[cfg(target_os = "macos")]
/// Submitted multi-tile Metal encode that resolves to host codestream bytes.
pub struct SubmittedJ2kLosslessMetalEncodeBatch {
    state: SubmittedJ2kLosslessMetalEncodeBatchState,
}

#[cfg(target_os = "macos")]
/// Submitted multi-tile Metal encode that resolves to Metal-backed codestreams.
pub struct SubmittedJ2kLosslessMetalBufferEncodeBatch {
    state: SubmittedJ2kLosslessMetalBufferEncodeBatchState,
}

#[cfg(target_os = "macos")]
enum SubmittedJ2kLosslessMetalEncodeBatchState {
    Ready(Vec<EncodedJ2k>),
    Deferred {
        tiles: Vec<OwnedMetalLosslessEncodeTile>,
        options: J2kLosslessEncodeOptions,
        session: crate::MetalBackendSession,
        staging: MetalEncodeInputStaging,
        config: MetalLosslessEncodeConfig,
    },
}

#[cfg(target_os = "macos")]
enum SubmittedJ2kLosslessMetalBufferEncodeBatchState {
    Resident(Box<SubmittedResidentLosslessMetalBufferEncodeBatch>),
    Deferred {
        tiles: Vec<OwnedMetalLosslessEncodeTile>,
        options: J2kLosslessEncodeOptions,
        session: crate::MetalBackendSession,
        staging: MetalEncodeInputStaging,
    },
}

#[cfg(target_os = "macos")]
struct OwnedMetalLosslessEncodeTile {
    buffer: Buffer,
    byte_offset: usize,
    width: u32,
    height: u32,
    pitch_bytes: usize,
    output_width: u32,
    output_height: u32,
    format: PixelFormat,
}

#[cfg(target_os = "macos")]
impl OwnedMetalLosslessEncodeTile {
    fn from_tile(tile: MetalLosslessEncodeTile<'_>) -> Self {
        Self {
            buffer: tile.buffer.to_owned(),
            byte_offset: tile.byte_offset,
            width: tile.width,
            height: tile.height,
            pitch_bytes: tile.pitch_bytes,
            output_width: tile.output_width,
            output_height: tile.output_height,
            format: tile.format,
        }
    }

    fn as_tile(&self) -> MetalLosslessEncodeTile<'_> {
        MetalLosslessEncodeTile {
            buffer: &self.buffer,
            byte_offset: self.byte_offset,
            width: self.width,
            height: self.height,
            pitch_bytes: self.pitch_bytes,
            output_width: self.output_width,
            output_height: self.output_height,
            format: self.format,
        }
    }
}

#[cfg(not(target_os = "macos"))]
/// Placeholder submitted single-tile encode for non-macOS builds.
pub struct SubmittedJ2kLosslessMetalEncode {
    _private: (),
}

#[cfg(not(target_os = "macos"))]
/// Placeholder submitted multi-tile encode for non-macOS builds.
pub struct SubmittedJ2kLosslessMetalEncodeBatch {
    _private: (),
}

#[cfg(not(target_os = "macos"))]
/// Placeholder submitted Metal-buffer encode for non-macOS builds.
pub struct SubmittedJ2kLosslessMetalBufferEncodeBatch {
    _private: (),
}

#[cfg(target_os = "macos")]
impl DeviceSubmission for SubmittedJ2kLosslessMetalEncode {
    type Output = EncodedJ2k;
    type Error = crate::Error;

    fn wait(self) -> Result<Self::Output, Self::Error> {
        let mut encoded = self.inner.wait()?;
        if encoded.len() != 1 {
            return Err(crate::Error::MetalKernel {
                message: "submitted J2K Metal single encode produced an unexpected batch length"
                    .to_string(),
            });
        }
        Ok(encoded.remove(0))
    }
}

#[cfg(target_os = "macos")]
impl DeviceSubmission for SubmittedJ2kLosslessMetalEncodeBatch {
    type Output = Vec<EncodedJ2k>;
    type Error = crate::Error;

    fn wait(self) -> Result<Self::Output, Self::Error> {
        match self.state {
            SubmittedJ2kLosslessMetalEncodeBatchState::Ready(encoded) => Ok(encoded),
            SubmittedJ2kLosslessMetalEncodeBatchState::Deferred {
                tiles,
                options,
                session,
                staging,
                config,
            } => {
                encode_lossless_owned_tiles_with_report(&tiles, options, &session, staging, config)
                    .map(|outcomes| {
                        outcomes
                            .into_iter()
                            .map(|outcome| outcome.encoded)
                            .collect()
                    })
            }
        }
    }
}

#[cfg(target_os = "macos")]
impl DeviceSubmission for SubmittedJ2kLosslessMetalBufferEncodeBatch {
    type Output = MetalLosslessBufferEncodeBatchOutcome;
    type Error = crate::Error;

    fn wait(self) -> Result<Self::Output, Self::Error> {
        match self.state {
            SubmittedJ2kLosslessMetalBufferEncodeBatchState::Resident(submitted) => {
                wait_submitted_resident_lossless_buffer_encode_batch(*submitted)
            }
            SubmittedJ2kLosslessMetalBufferEncodeBatchState::Deferred {
                tiles,
                options,
                session,
                staging,
            } => encode_owned_lossless_tiles_to_metal_buffer_fallback_batch(
                &tiles, options, &session, staging,
            ),
        }
    }
}

#[cfg(not(target_os = "macos"))]
impl DeviceSubmission for SubmittedJ2kLosslessMetalEncode {
    type Output = EncodedJ2k;
    type Error = crate::Error;

    fn wait(self) -> Result<Self::Output, Self::Error> {
        let _ = self;
        Err(crate::Error::MetalUnavailable)
    }
}

#[cfg(not(target_os = "macos"))]
impl DeviceSubmission for SubmittedJ2kLosslessMetalEncodeBatch {
    type Output = Vec<EncodedJ2k>;
    type Error = crate::Error;

    fn wait(self) -> Result<Self::Output, Self::Error> {
        let _ = self;
        Err(crate::Error::MetalUnavailable)
    }
}

#[cfg(not(target_os = "macos"))]
impl DeviceSubmission for SubmittedJ2kLosslessMetalBufferEncodeBatch {
    type Output = MetalLosslessBufferEncodeBatchOutcome;
    type Error = crate::Error;

    fn wait(self) -> Result<Self::Output, Self::Error> {
        let _ = self;
        Err(crate::Error::MetalUnavailable)
    }
}

#[cfg(target_os = "macos")]
/// Encode one Metal-resident tile into host codestream bytes.
pub fn encode_lossless_from_metal_buffer(
    tile: MetalLosslessEncodeTile<'_>,
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<EncodedJ2k, crate::Error> {
    submit_lossless_from_metal_buffer(tile, options, session)?.wait()
}

#[cfg(target_os = "macos")]
/// Encode one Metal-resident tile into a Metal-backed codestream buffer.
pub fn encode_lossless_from_metal_buffer_to_metal(
    tile: MetalLosslessEncodeTile<'_>,
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<MetalEncodedJ2k, crate::Error> {
    Ok(encode_lossless_from_metal_buffer_to_metal_with_report(tile, options, session)?.encoded)
}

#[cfg(target_os = "macos")]
/// Submit one Metal-resident tile encode for later host-byte collection.
pub fn submit_lossless_from_metal_buffer(
    tile: MetalLosslessEncodeTile<'_>,
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<SubmittedJ2kLosslessMetalEncode, crate::Error> {
    let inner = submit_lossless_from_metal_buffers(&[tile], options, session)?;
    Ok(SubmittedJ2kLosslessMetalEncode { inner })
}

#[cfg(target_os = "macos")]
/// Encode one Metal-resident tile and return a host-byte timing report.
pub fn encode_lossless_from_metal_buffer_with_report(
    tile: MetalLosslessEncodeTile<'_>,
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<MetalLosslessEncodeOutcome, crate::Error> {
    let mut accelerator = MetalEncodeStageAccelerator::for_host_output(*options);
    encode_lossless_tile_with_report(
        tile,
        *options,
        session,
        MetalEncodeInputStaging::CopyAndPad,
        &mut accelerator,
    )
}

#[cfg(target_os = "macos")]
/// Encode one Metal-resident tile into a Metal buffer with timing data.
pub fn encode_lossless_from_metal_buffer_to_metal_with_report(
    tile: MetalLosslessEncodeTile<'_>,
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<MetalLosslessBufferEncodeOutcome, crate::Error> {
    let mut outcomes =
        encode_lossless_from_metal_buffers_to_metal_with_report(&[tile], options, session)?;
    if outcomes.len() != 1 {
        return Err(crate::Error::MetalKernel {
            message: "J2K Metal single buffer encode produced an unexpected batch length"
                .to_string(),
        });
    }
    Ok(outcomes.remove(0))
}

#[cfg(target_os = "macos")]
/// Encode one already padded Metal-resident tile into host codestream bytes.
pub fn encode_lossless_from_padded_metal_buffer(
    tile: MetalLosslessEncodeTile<'_>,
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<EncodedJ2k, crate::Error> {
    submit_lossless_from_padded_metal_buffer(tile, options, session)?.wait()
}

#[cfg(target_os = "macos")]
/// Encode one already padded Metal-resident tile into a Metal-backed codestream.
pub fn encode_lossless_from_padded_metal_buffer_to_metal(
    tile: MetalLosslessEncodeTile<'_>,
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<MetalEncodedJ2k, crate::Error> {
    Ok(
        encode_lossless_from_padded_metal_buffer_to_metal_with_report(tile, options, session)?
            .encoded,
    )
}

#[cfg(target_os = "macos")]
/// Submit one already padded Metal-resident tile encode for host-byte collection.
pub fn submit_lossless_from_padded_metal_buffer(
    tile: MetalLosslessEncodeTile<'_>,
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<SubmittedJ2kLosslessMetalEncode, crate::Error> {
    let inner = submit_lossless_from_padded_metal_buffers(&[tile], options, session)?;
    Ok(SubmittedJ2kLosslessMetalEncode { inner })
}

#[cfg(target_os = "macos")]
/// Encode one already padded tile and return a host-byte timing report.
pub fn encode_lossless_from_padded_metal_buffer_with_report(
    tile: MetalLosslessEncodeTile<'_>,
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<MetalLosslessEncodeOutcome, crate::Error> {
    let mut accelerator = MetalEncodeStageAccelerator::for_host_output(*options);
    encode_lossless_tile_with_report(
        tile,
        *options,
        session,
        MetalEncodeInputStaging::AlreadyPaddedContiguous,
        &mut accelerator,
    )
}

#[cfg(target_os = "macos")]
/// Encode one already padded tile into a Metal buffer with timing data.
pub fn encode_lossless_from_padded_metal_buffer_to_metal_with_report(
    tile: MetalLosslessEncodeTile<'_>,
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<MetalLosslessBufferEncodeOutcome, crate::Error> {
    let mut outcomes =
        encode_lossless_from_padded_metal_buffers_to_metal_with_report(&[tile], options, session)?;
    if outcomes.len() != 1 {
        return Err(crate::Error::MetalKernel {
            message: "J2K Metal single buffer encode produced an unexpected batch length"
                .to_string(),
        });
    }
    Ok(outcomes.remove(0))
}

#[cfg(target_os = "macos")]
/// Encode multiple Metal-resident tiles into host codestream bytes.
pub fn encode_lossless_from_metal_buffers(
    tiles: &[MetalLosslessEncodeTile<'_>],
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<Vec<EncodedJ2k>, crate::Error> {
    submit_lossless_from_metal_buffers(tiles, options, session)?.wait()
}

#[cfg(target_os = "macos")]
/// Encode multiple Metal-resident tiles into Metal-backed codestream buffers.
pub fn encode_lossless_from_metal_buffers_to_metal(
    tiles: &[MetalLosslessEncodeTile<'_>],
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<Vec<MetalEncodedJ2k>, crate::Error> {
    Ok(
        encode_lossless_from_metal_buffers_to_metal_with_report(tiles, options, session)?
            .into_iter()
            .map(|outcome| outcome.encoded)
            .collect(),
    )
}

#[cfg(target_os = "macos")]
/// Submit multiple Metal-resident tile encodes for later host-byte collection.
pub fn submit_lossless_from_metal_buffers(
    tiles: &[MetalLosslessEncodeTile<'_>],
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<SubmittedJ2kLosslessMetalEncodeBatch, crate::Error> {
    submit_lossless_from_metal_buffers_with_config(
        tiles,
        options,
        session,
        MetalLosslessEncodeConfig::default(),
    )
}

#[cfg(target_os = "macos")]
/// Submit multiple Metal-resident tile encodes with explicit batch tuning.
pub fn submit_lossless_from_metal_buffers_with_config(
    tiles: &[MetalLosslessEncodeTile<'_>],
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
    config: MetalLosslessEncodeConfig,
) -> Result<SubmittedJ2kLosslessMetalEncodeBatch, crate::Error> {
    submit_lossless_tiles(
        tiles,
        *options,
        session,
        MetalEncodeInputStaging::CopyAndPad,
        config,
    )
}

#[cfg(target_os = "macos")]
/// Encode multiple Metal-resident tiles and return host-byte timing reports.
pub fn encode_lossless_from_metal_buffers_with_report(
    tiles: &[MetalLosslessEncodeTile<'_>],
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<Vec<MetalLosslessEncodeOutcome>, crate::Error> {
    encode_lossless_tiles_with_report(
        tiles,
        *options,
        session,
        MetalEncodeInputStaging::CopyAndPad,
    )
}

#[cfg(target_os = "macos")]
/// Encode multiple Metal-resident tiles into Metal buffers with timing data.
pub fn encode_lossless_from_metal_buffers_to_metal_with_report(
    tiles: &[MetalLosslessEncodeTile<'_>],
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<Vec<MetalLosslessBufferEncodeOutcome>, crate::Error> {
    Ok(encode_lossless_from_metal_buffers_to_metal_batch(
        tiles,
        options,
        session,
        MetalLosslessEncodeConfig::default(),
    )?
    .outcomes)
}

#[cfg(target_os = "macos")]
/// Submit multiple tile encodes that resolve to Metal-backed codestream buffers.
pub fn submit_lossless_from_metal_buffers_to_metal_batch(
    tiles: &[MetalLosslessEncodeTile<'_>],
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
    config: MetalLosslessEncodeConfig,
) -> Result<SubmittedJ2kLosslessMetalBufferEncodeBatch, crate::Error> {
    submit_lossless_tiles_to_metal_buffer_batch(
        tiles,
        *options,
        session,
        MetalEncodeInputStaging::CopyAndPad,
        config,
    )
}

#[cfg(target_os = "macos")]
/// Encode multiple tiles into Metal-backed codestream buffers with batch stats.
pub fn encode_lossless_from_metal_buffers_to_metal_batch(
    tiles: &[MetalLosslessEncodeTile<'_>],
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
    config: MetalLosslessEncodeConfig,
) -> Result<MetalLosslessBufferEncodeBatchOutcome, crate::Error> {
    submit_lossless_from_metal_buffers_to_metal_batch(tiles, options, session, config)?.wait()
}

#[cfg(target_os = "macos")]
/// Encode multiple already padded Metal-resident tiles into host codestream bytes.
pub fn encode_lossless_from_padded_metal_buffers(
    tiles: &[MetalLosslessEncodeTile<'_>],
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<Vec<EncodedJ2k>, crate::Error> {
    submit_lossless_from_padded_metal_buffers(tiles, options, session)?.wait()
}

#[cfg(target_os = "macos")]
/// Encode multiple already padded tiles into Metal-backed codestream buffers.
pub fn encode_lossless_from_padded_metal_buffers_to_metal(
    tiles: &[MetalLosslessEncodeTile<'_>],
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<Vec<MetalEncodedJ2k>, crate::Error> {
    Ok(
        encode_lossless_from_padded_metal_buffers_to_metal_with_report(tiles, options, session)?
            .into_iter()
            .map(|outcome| outcome.encoded)
            .collect(),
    )
}

#[cfg(target_os = "macos")]
/// Submit already padded tile encodes for later host-byte collection.
pub fn submit_lossless_from_padded_metal_buffers(
    tiles: &[MetalLosslessEncodeTile<'_>],
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<SubmittedJ2kLosslessMetalEncodeBatch, crate::Error> {
    submit_lossless_from_padded_metal_buffers_with_config(
        tiles,
        options,
        session,
        MetalLosslessEncodeConfig::default(),
    )
}

#[cfg(target_os = "macos")]
/// Submit already padded tile encodes with explicit batch tuning.
pub fn submit_lossless_from_padded_metal_buffers_with_config(
    tiles: &[MetalLosslessEncodeTile<'_>],
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
    config: MetalLosslessEncodeConfig,
) -> Result<SubmittedJ2kLosslessMetalEncodeBatch, crate::Error> {
    submit_lossless_tiles(
        tiles,
        *options,
        session,
        MetalEncodeInputStaging::AlreadyPaddedContiguous,
        config,
    )
}

#[cfg(target_os = "macos")]
/// Encode multiple already padded tiles and return host-byte timing reports.
pub fn encode_lossless_from_padded_metal_buffers_with_report(
    tiles: &[MetalLosslessEncodeTile<'_>],
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<Vec<MetalLosslessEncodeOutcome>, crate::Error> {
    encode_lossless_tiles_with_report(
        tiles,
        *options,
        session,
        MetalEncodeInputStaging::AlreadyPaddedContiguous,
    )
}

#[cfg(target_os = "macos")]
/// Encode multiple already padded tiles into Metal buffers with timing data.
pub fn encode_lossless_from_padded_metal_buffers_to_metal_with_report(
    tiles: &[MetalLosslessEncodeTile<'_>],
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<Vec<MetalLosslessBufferEncodeOutcome>, crate::Error> {
    Ok(encode_lossless_from_padded_metal_buffers_to_metal_batch(
        tiles,
        options,
        session,
        MetalLosslessEncodeConfig::default(),
    )?
    .outcomes)
}

#[cfg(target_os = "macos")]
/// Submit already padded tile encodes that resolve to Metal-backed codestreams.
pub fn submit_lossless_from_padded_metal_buffers_to_metal_batch(
    tiles: &[MetalLosslessEncodeTile<'_>],
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
    config: MetalLosslessEncodeConfig,
) -> Result<SubmittedJ2kLosslessMetalBufferEncodeBatch, crate::Error> {
    submit_lossless_tiles_to_metal_buffer_batch(
        tiles,
        *options,
        session,
        MetalEncodeInputStaging::AlreadyPaddedContiguous,
        config,
    )
}

#[cfg(target_os = "macos")]
/// Encode already padded tiles into Metal-backed codestreams with batch stats.
pub fn encode_lossless_from_padded_metal_buffers_to_metal_batch(
    tiles: &[MetalLosslessEncodeTile<'_>],
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
    config: MetalLosslessEncodeConfig,
) -> Result<MetalLosslessBufferEncodeBatchOutcome, crate::Error> {
    submit_lossless_from_padded_metal_buffers_to_metal_batch(tiles, options, session, config)?
        .wait()
}

#[cfg(target_os = "macos")]
fn host_outcome_from_buffer_outcome(
    outcome: MetalLosslessBufferEncodeOutcome,
) -> Result<MetalLosslessEncodeOutcome, crate::Error> {
    let (encoded, host_readback_duration) =
        outcome.encoded.to_encoded_j2k_with_readback_duration()?;
    Ok(MetalLosslessEncodeOutcome {
        encoded,
        input_copy_used: outcome.input_copy_used,
        resident: outcome.resident,
        input_copy_duration: outcome.input_copy_duration,
        encode_duration: outcome.encode_duration,
        gpu_duration: outcome.gpu_duration,
        validation_duration: outcome.validation_duration,
        host_readback_duration,
    })
}

#[cfg(target_os = "macos")]
fn encode_lossless_tiles_with_report(
    tiles: &[MetalLosslessEncodeTile<'_>],
    options: J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
    staging: MetalEncodeInputStaging,
) -> Result<Vec<MetalLosslessEncodeOutcome>, crate::Error> {
    if should_try_resident_lossless_host_encode(options) {
        let batch = try_encode_resident_lossless_tiles_to_metal_buffer_batch(
            tiles,
            options,
            session,
            staging,
            MetalLosslessEncodeConfig::default(),
        )?;
        if let Some(outcomes) = batch {
            return outcomes
                .outcomes
                .into_iter()
                .map(host_outcome_from_buffer_outcome)
                .collect();
        }
    }

    let mut accelerator = MetalEncodeStageAccelerator::for_host_output(options);
    tiles
        .iter()
        .map(|&tile| {
            encode_lossless_tile_with_report(tile, options, session, staging, &mut accelerator)
        })
        .collect()
}

#[cfg(target_os = "macos")]
fn encode_lossless_owned_tiles_with_report(
    tiles: &[OwnedMetalLosslessEncodeTile],
    options: J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
    staging: MetalEncodeInputStaging,
    config: MetalLosslessEncodeConfig,
) -> Result<Vec<MetalLosslessEncodeOutcome>, crate::Error> {
    let borrowed = tiles
        .iter()
        .map(OwnedMetalLosslessEncodeTile::as_tile)
        .collect::<Vec<_>>();
    if should_try_resident_lossless_host_encode(options) {
        let batch = try_encode_resident_lossless_tiles_to_metal_buffer_batch(
            &borrowed, options, session, staging, config,
        )?;
        if let Some(outcomes) = batch {
            return outcomes
                .outcomes
                .into_iter()
                .map(host_outcome_from_buffer_outcome)
                .collect();
        }
    }

    let mut accelerator = MetalEncodeStageAccelerator::for_host_output(options);
    borrowed
        .iter()
        .map(|&tile| {
            encode_lossless_tile_with_report(tile, options, session, staging, &mut accelerator)
        })
        .collect()
}

#[cfg(target_os = "macos")]
fn submit_lossless_tiles_to_metal_buffer_batch(
    tiles: &[MetalLosslessEncodeTile<'_>],
    options: J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
    staging: MetalEncodeInputStaging,
    config: MetalLosslessEncodeConfig,
) -> Result<SubmittedJ2kLosslessMetalBufferEncodeBatch, crate::Error> {
    if options.backend != EncodeBackendPreference::CpuOnly {
        if let Some(submitted) = try_submit_resident_lossless_tiles_to_metal_buffer_batch(
            tiles, options, session, staging, config,
        )? {
            return Ok(SubmittedJ2kLosslessMetalBufferEncodeBatch {
                state: SubmittedJ2kLosslessMetalBufferEncodeBatchState::Resident(Box::new(
                    submitted,
                )),
            });
        }
    }

    let mut owned = Vec::with_capacity(tiles.len());
    for &tile in tiles {
        validate_metal_encode_tile(tile)?;
        if matches!(staging, MetalEncodeInputStaging::AlreadyPaddedContiguous) {
            lossless_sample_shape(tile.format)?;
            validate_padded_contiguous_metal_encode_tile(tile, tile.format.bytes_per_pixel())?;
        }
        owned.push(OwnedMetalLosslessEncodeTile::from_tile(tile));
    }
    Ok(SubmittedJ2kLosslessMetalBufferEncodeBatch {
        state: SubmittedJ2kLosslessMetalBufferEncodeBatchState::Deferred {
            tiles: owned,
            options,
            session: session.clone(),
            staging,
        },
    })
}

#[cfg(target_os = "macos")]
fn encode_owned_lossless_tiles_to_metal_buffer_fallback_batch(
    tiles: &[OwnedMetalLosslessEncodeTile],
    options: J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
    staging: MetalEncodeInputStaging,
) -> Result<MetalLosslessBufferEncodeBatchOutcome, crate::Error> {
    let mut outcomes = Vec::with_capacity(tiles.len());
    for tile in tiles {
        outcomes.push(encode_lossless_tile_to_metal_buffer_with_report(
            tile.as_tile(),
            options,
            session,
            staging,
        )?);
    }
    Ok(MetalLosslessBufferEncodeBatchOutcome {
        outcomes,
        stats: MetalLosslessEncodeBatchStats::default(),
    })
}

#[cfg(target_os = "macos")]
fn try_submit_resident_lossless_tiles_to_metal_buffer_batch(
    tiles: &[MetalLosslessEncodeTile<'_>],
    options: J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
    staging: MetalEncodeInputStaging,
    config: MetalLosslessEncodeConfig,
) -> Result<Option<SubmittedResidentLosslessMetalBufferEncodeBatch>, crate::Error> {
    let profile_stages = compute::metal_profile_stages_enabled();
    if tiles.is_empty() {
        return Ok(Some(SubmittedResidentLosslessMetalBufferEncodeBatch {
            options,
            session: session.clone(),
            stats: resolve_lossless_encode_config(0, 1, config)?,
            encode_started: Instant::now(),
            tiles: Vec::new(),
            staging,
            kind: SubmittedResidentLosslessMetalBufferEncodeBatchKind::Empty,
        }));
    }

    let plan_started = profile_stages.then(Instant::now);
    let mut planned = Vec::with_capacity(tiles.len());
    for (index, &tile) in tiles.iter().enumerate() {
        let Some(item) = plan_resident_lossless_buffer_encode(index, tile, options, staging)?
        else {
            return Ok(None);
        };
        planned.push(item);
    }
    let estimated_peak_bytes_per_tile = planned
        .iter()
        .map(PlannedResidentLosslessBufferEncode::estimated_peak_bytes)
        .max()
        .unwrap_or(1);
    let classic_resident_mode = planned
        .iter()
        .all(|planned| planned.metadata.plan.block_coding_mode == J2kBlockCodingMode::Classic);
    let ht_resident_mode = planned.iter().all(|planned| {
        planned.metadata.plan.block_coding_mode == J2kBlockCodingMode::HighThroughput
    });
    if !(classic_resident_mode || ht_resident_mode) {
        return Ok(None);
    }
    let resolved_config =
        resident_lossless_encode_config_for_mode(config, classic_resident_mode, tiles.len());
    let mut stats = resolve_lossless_encode_config(
        tiles.len(),
        estimated_peak_bytes_per_tile,
        resolved_config,
    )?;
    if let Some(started) = plan_started {
        stats.stage_stats.plan_duration = started.elapsed();
    }
    let encode_started = Instant::now();
    let kind = submit_planned_resident_lossless_tiles(
        planned,
        session,
        stats.effective_inflight_tiles,
        &mut stats,
    )?;
    let tiles = tiles
        .iter()
        .map(|&tile| OwnedMetalLosslessEncodeTile::from_tile(tile))
        .collect();
    Ok(Some(SubmittedResidentLosslessMetalBufferEncodeBatch {
        options,
        session: session.clone(),
        stats,
        encode_started,
        tiles,
        staging,
        kind,
    }))
}

#[cfg(target_os = "macos")]
fn try_encode_resident_lossless_tiles_to_metal_buffer_batch(
    tiles: &[MetalLosslessEncodeTile<'_>],
    options: J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
    staging: MetalEncodeInputStaging,
    config: MetalLosslessEncodeConfig,
) -> Result<Option<MetalLosslessBufferEncodeBatchOutcome>, crate::Error> {
    let Some(submitted) = try_submit_resident_lossless_tiles_to_metal_buffer_batch(
        tiles, options, session, staging, config,
    )?
    else {
        return Ok(None);
    };
    wait_submitted_resident_lossless_buffer_encode_batch(submitted).map(Some)
}

#[cfg(any(test, target_os = "macos"))]
const GPU_ENCODE_DEFAULT_INFLIGHT_TILES: usize = 512;
#[cfg(any(test, target_os = "macos"))]
const CLASSIC_GPU_ENCODE_SMALL_BATCH_INFLIGHT_TILES: usize = 16;
#[cfg(any(test, target_os = "macos"))]
const CLASSIC_GPU_ENCODE_LARGE_BATCH_INFLIGHT_TILES: usize = 64;
#[cfg(any(test, target_os = "macos"))]
const CLASSIC_GPU_ENCODE_VERY_LARGE_BATCH_MIN_TILES: usize = 64;
#[cfg(any(test, target_os = "macos"))]
const CLASSIC_GPU_ENCODE_VERY_LARGE_BATCH_INFLIGHT_TILES: usize = 128;
#[cfg(any(test, target_os = "macos"))]
const HTJ2K_GPU_ENCODE_MEDIUM_BATCH_TILES: usize = 64;
#[cfg(any(test, target_os = "macos"))]
const HTJ2K_GPU_ENCODE_MEDIUM_BATCH_INFLIGHT_TILES: usize = 32;
#[cfg(any(test, target_os = "macos"))]
const HTJ2K_GPU_ENCODE_LARGE_BATCH_MIN_TILES: usize = 64;
#[cfg(any(test, target_os = "macos"))]
const HTJ2K_GPU_ENCODE_LARGE_BATCH_INFLIGHT_TILES: usize = 64;
#[cfg(any(test, target_os = "macos"))]
const GPU_ENCODE_FALLBACK_HW_MEM_BYTES: usize = 8 * 1024 * 1024 * 1024;
#[cfg(any(test, target_os = "macos"))]
const GPU_ENCODE_MAX_DEFAULT_MEMORY_BUDGET_BYTES: usize = 10 * 1024 * 1024 * 1024;
#[cfg(any(test, target_os = "macos"))]
const GPU_ENCODE_MEMORY_BUDGET_PERCENT: usize = 40;
#[cfg(any(test, target_os = "macos"))]
const RESIDENT_HT_DEFAULT_CHUNK_CODE_BLOCKS: usize = 131_072;

#[cfg(any(test, target_os = "macos"))]
fn default_gpu_encode_memory_budget_bytes_for_hw_mem(hw_memsize: usize) -> usize {
    hw_memsize
        .saturating_mul(GPU_ENCODE_MEMORY_BUDGET_PERCENT)
        .checked_div(100)
        .unwrap_or(0)
        .clamp(1, GPU_ENCODE_MAX_DEFAULT_MEMORY_BUDGET_BYTES)
}

#[cfg(any(test, target_os = "macos"))]
fn default_gpu_encode_memory_budget_bytes() -> usize {
    let hw_memsize = host_memory_bytes().unwrap_or(GPU_ENCODE_FALLBACK_HW_MEM_BYTES);
    default_gpu_encode_memory_budget_bytes_for_hw_mem(hw_memsize)
}

#[cfg(any(test, target_os = "macos"))]
fn resident_lossless_encode_config_for_mode(
    config: MetalLosslessEncodeConfig,
    classic_resident_mode: bool,
    tile_count: usize,
) -> MetalLosslessEncodeConfig {
    if config.gpu_encode_inflight_tiles.is_some() {
        return config;
    }
    if classic_resident_mode {
        let classic_inflight_tiles = if tile_count <= CLASSIC_GPU_ENCODE_SMALL_BATCH_INFLIGHT_TILES
        {
            CLASSIC_GPU_ENCODE_SMALL_BATCH_INFLIGHT_TILES
        } else if tile_count <= CLASSIC_GPU_ENCODE_VERY_LARGE_BATCH_MIN_TILES {
            CLASSIC_GPU_ENCODE_LARGE_BATCH_INFLIGHT_TILES
        } else {
            CLASSIC_GPU_ENCODE_VERY_LARGE_BATCH_INFLIGHT_TILES
        };
        MetalLosslessEncodeConfig {
            gpu_encode_inflight_tiles: Some(classic_inflight_tiles),
            ..config
        }
    } else if tile_count == HTJ2K_GPU_ENCODE_MEDIUM_BATCH_TILES {
        MetalLosslessEncodeConfig {
            gpu_encode_inflight_tiles: Some(HTJ2K_GPU_ENCODE_MEDIUM_BATCH_INFLIGHT_TILES),
            ..config
        }
    } else if tile_count > HTJ2K_GPU_ENCODE_LARGE_BATCH_MIN_TILES {
        MetalLosslessEncodeConfig {
            gpu_encode_inflight_tiles: Some(HTJ2K_GPU_ENCODE_LARGE_BATCH_INFLIGHT_TILES),
            ..config
        }
    } else {
        config
    }
}

#[cfg(target_os = "macos")]
fn host_memory_bytes() -> Option<usize> {
    let mut value = 0u64;
    let mut len = core::mem::size_of::<u64>();
    let name = b"hw.memsize\0";
    let rc = unsafe {
        libc::sysctlbyname(
            name.as_ptr().cast(),
            (&raw mut value).cast(),
            &raw mut len,
            core::ptr::null_mut(),
            0,
        )
    };
    (rc == 0 && len == core::mem::size_of::<u64>())
        .then(|| usize::try_from(value).ok())
        .flatten()
}

#[cfg(all(test, not(target_os = "macos")))]
fn host_memory_bytes() -> Option<usize> {
    None
}

#[cfg(any(test, target_os = "macos"))]
fn resolve_lossless_encode_config(
    tile_count: usize,
    estimated_peak_bytes_per_tile: usize,
    config: MetalLosslessEncodeConfig,
) -> Result<MetalLosslessEncodeBatchStats, crate::Error> {
    if config.gpu_encode_inflight_tiles == Some(0) {
        return Err(crate::Error::UnsupportedMetalRequest {
            reason: "J2K Metal encode in-flight tile cap must be greater than zero",
        });
    }
    if config.gpu_encode_memory_budget_bytes == Some(0) {
        return Err(crate::Error::UnsupportedMetalRequest {
            reason: "J2K Metal encode memory budget must be greater than zero",
        });
    }

    let effective_memory_budget_bytes = config
        .gpu_encode_memory_budget_bytes
        .unwrap_or_else(default_gpu_encode_memory_budget_bytes)
        .max(1);
    let estimated_peak_bytes_per_tile = estimated_peak_bytes_per_tile.max(1);
    let memory_limited_tiles =
        (effective_memory_budget_bytes / estimated_peak_bytes_per_tile).max(1);
    let configured_or_default = config
        .gpu_encode_inflight_tiles
        .unwrap_or(GPU_ENCODE_DEFAULT_INFLIGHT_TILES);
    let effective_inflight_tiles = configured_or_default
        .min(memory_limited_tiles)
        .min(tile_count.max(1))
        .max(1);

    Ok(MetalLosslessEncodeBatchStats {
        configured_inflight_tiles: config.gpu_encode_inflight_tiles,
        effective_inflight_tiles,
        configured_memory_budget_bytes: config.gpu_encode_memory_budget_bytes,
        effective_memory_budget_bytes,
        estimated_peak_bytes_per_tile,
        max_observed_inflight_tiles: 0,
        encode_wall_duration: Duration::ZERO,
        stage_stats: MetalLosslessEncodeStageStats::default(),
    })
}

#[cfg(test)]
fn resolve_lossless_encode_config_for_test(
    tile_count: usize,
    estimated_peak_bytes_per_tile: usize,
    config: MetalLosslessEncodeConfig,
) -> Result<MetalLosslessEncodeBatchStats, crate::Error> {
    resolve_lossless_encode_config(tile_count, estimated_peak_bytes_per_tile, config)
}

#[cfg(target_os = "macos")]
fn checked_add_bytes(lhs: usize, rhs: usize) -> usize {
    lhs.saturating_add(rhs)
}

#[cfg(target_os = "macos")]
fn checked_mul_bytes(lhs: usize, rhs: usize) -> usize {
    lhs.saturating_mul(rhs)
}

#[cfg(any(test, target_os = "macos"))]
fn resident_lossless_code_block_chunk_cap(code_block_counts: &[usize]) -> usize {
    code_block_counts
        .iter()
        .copied()
        .max()
        .unwrap_or(1)
        .max(RESIDENT_HT_DEFAULT_CHUNK_CODE_BLOCKS)
}

#[cfg(any(test, target_os = "macos"))]
fn resident_lossless_chunk_ranges_from_code_blocks(
    code_block_counts: &[usize],
    max_tiles: usize,
    max_code_blocks: usize,
) -> Vec<std::ops::Range<usize>> {
    if code_block_counts.is_empty() {
        return Vec::new();
    }
    let max_tiles = max_tiles.max(1);
    let max_code_blocks = max_code_blocks.max(1);
    let mut ranges = Vec::new();
    let mut start = 0usize;
    while start < code_block_counts.len() {
        let mut end = start;
        let mut chunk_code_blocks = 0usize;
        while end < code_block_counts.len() && end - start < max_tiles {
            let next_code_blocks = code_block_counts[end].max(1);
            let would_exceed_code_blocks =
                end > start && chunk_code_blocks.saturating_add(next_code_blocks) > max_code_blocks;
            if would_exceed_code_blocks {
                break;
            }
            chunk_code_blocks = chunk_code_blocks.saturating_add(next_code_blocks);
            end += 1;
        }
        if end == start {
            end += 1;
        }
        ranges.push(start..end);
        start = end;
    }
    ranges
}

#[cfg(test)]
fn resident_lossless_chunk_ranges_for_test(
    code_block_counts: &[usize],
    max_tiles: usize,
    max_code_blocks: usize,
) -> Vec<std::ops::Range<usize>> {
    resident_lossless_chunk_ranges_from_code_blocks(code_block_counts, max_tiles, max_code_blocks)
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
struct LosslessSubbandPlan {
    num_cbs_x: u32,
    num_cbs_y: u32,
    code_block_start: usize,
    code_block_count: usize,
}

#[cfg(target_os = "macos")]
#[derive(Clone)]
struct LosslessResolutionPlan {
    subbands: Vec<LosslessSubbandPlan>,
}

#[cfg(target_os = "macos")]
struct LosslessDeviceEncodePlan {
    components: u8,
    bit_depth: u8,
    block_coding_mode: J2kBlockCodingMode,
    num_decomposition_levels: u8,
    use_mct: bool,
    guard_bits: u8,
    code_block_width_exp: u8,
    code_block_height_exp: u8,
    code_blocks: Vec<compute::J2kLosslessDeviceCodeBlock>,
    resolutions: Vec<LosslessResolutionPlan>,
    progression_order: EncodeProgressionOrder,
    write_tlm: bool,
}

#[cfg(target_os = "macos")]
struct ResidentLosslessBufferEncodeMetadata {
    tile: OwnedMetalLosslessEncodeTile,
    components: u8,
    bit_depth: u8,
    bytes_per_pixel: usize,
    plan: LosslessDeviceEncodePlan,
    packet_descriptors: Vec<J2kPacketizationPacketDescriptor>,
    packetization_resolutions: Vec<compute::J2kResidentPacketizationResolution>,
}

#[cfg(target_os = "macos")]
struct PreparedResidentLosslessBufferEncode {
    metadata: ResidentLosslessBufferEncodeMetadata,
    prepared: compute::J2kPreparedLosslessDeviceCodeBlocks,
}

#[cfg(target_os = "macos")]
struct PlannedResidentLosslessBufferEncode {
    index: usize,
    metadata: ResidentLosslessBufferEncodeMetadata,
    coefficient_count: usize,
    bytes_per_sample: u8,
    estimated_peak_bytes: usize,
    #[cfg(test)]
    failure_injection_index: Option<usize>,
}

#[cfg(target_os = "macos")]
impl PlannedResidentLosslessBufferEncode {
    fn estimated_peak_bytes(&self) -> usize {
        self.estimated_peak_bytes
    }
}

#[cfg(target_os = "macos")]
struct SubmittedResidentLosslessMetalBufferEncodeBatch {
    options: J2kLosslessEncodeOptions,
    session: crate::MetalBackendSession,
    stats: MetalLosslessEncodeBatchStats,
    encode_started: Instant,
    tiles: Vec<OwnedMetalLosslessEncodeTile>,
    staging: MetalEncodeInputStaging,
    kind: SubmittedResidentLosslessMetalBufferEncodeBatchKind,
}

#[cfg(target_os = "macos")]
enum SubmittedResidentLosslessMetalBufferEncodeBatchKind {
    Empty,
    Chunks(Vec<SubmittedResidentLosslessMetalBufferEncodeChunk>),
}

#[cfg(target_os = "macos")]
struct SubmittedResidentLosslessMetalBufferEncodeChunk {
    metadatas: Vec<ResidentLosslessBufferEncodeMetadata>,
    prepare_durations: Vec<Duration>,
    pending: compute::J2kPendingResidentLosslessCodestreamBatch,
    batch_started: Instant,
}

#[cfg(target_os = "macos")]
struct FinishedResidentLosslessBufferEncode {
    metadata: ResidentLosslessBufferEncodeMetadata,
    encoded: MetalEncodedJ2k,
    encode_duration: Duration,
    gpu_duration: Option<Duration>,
}

#[cfg(target_os = "macos")]
fn lossless_device_encode_levels(width: u32, height: u32, options: J2kLosslessEncodeOptions) -> u8 {
    const MIN_LOSSLESS_DWT_DIMENSION: u32 = 64;
    let levels = if matches!(
        options.progression,
        J2kProgressionOrder::Rpcl | J2kProgressionOrder::Pcrl | J2kProgressionOrder::Cprl
    ) {
        let mut levels = 0u8;
        let mut w = width;
        let mut h = height;
        let max_levels = if width.min(height) <= 1 {
            0
        } else {
            width.min(height).ilog2() as u8
        };
        while w.min(h) > MIN_LOSSLESS_DWT_DIMENSION && levels < max_levels {
            w = w.div_ceil(2);
            h = h.div_ceil(2);
            levels = levels.saturating_add(1);
        }
        levels
    } else {
        u8::from(width.min(height) >= MIN_LOSSLESS_DWT_DIMENSION)
    };

    options
        .max_decomposition_levels
        .map_or(levels, |requested| {
            let max_levels = if width.min(height) <= 1 {
                0
            } else {
                width.min(height).ilog2() as u8
            };
            requested.min(max_levels)
        })
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
struct LosslessDwtLevelPlan {
    low_width: u32,
    low_height: u32,
    high_width: u32,
    high_height: u32,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
struct LosslessSubbandInput {
    component: u32,
    subband_x: u32,
    subband_y: u32,
    width: u32,
    height: u32,
    sub_band_type: J2kSubBandType,
    total_bitplanes: u8,
}

#[cfg(target_os = "macos")]
const RESIDENT_CLASSIC_CODE_BLOCK_EDGE: u32 = 32;

#[cfg(target_os = "macos")]
fn lossless_code_block_exp(edge: u32, axis: &str) -> Result<u8, crate::Error> {
    if edge < 4 || !edge.is_power_of_two() {
        return Err(crate::Error::MetalKernel {
            message: format!(
                "J2K Metal resident encode {axis} code-block edge must be a power of two >= 4"
            ),
        });
    }
    let exp = edge
        .trailing_zeros()
        .checked_sub(2)
        .ok_or_else(|| crate::Error::MetalKernel {
            message: format!("J2K Metal resident encode {axis} code-block exponent underflow"),
        })?;
    if exp > 8 {
        return Err(crate::Error::MetalKernel {
            message: format!(
                "J2K Metal resident encode {axis} code-block edge exceeds JPEG 2000 COD range"
            ),
        });
    }
    u8::try_from(exp).map_err(|_| crate::Error::MetalKernel {
        message: format!("J2K Metal resident encode {axis} code-block exponent exceeds u8"),
    })
}

#[cfg(target_os = "macos")]
fn push_lossless_subband_plan(
    resolution: &mut LosslessResolutionPlan,
    code_blocks: &mut Vec<compute::J2kLosslessDeviceCodeBlock>,
    coefficient_offset: &mut u32,
    code_block_width: u32,
    code_block_height: u32,
    subband: LosslessSubbandInput,
) -> Result<(), crate::Error> {
    if subband.width == 0 || subband.height == 0 {
        resolution.subbands.push(LosslessSubbandPlan {
            num_cbs_x: 0,
            num_cbs_y: 0,
            code_block_start: code_blocks.len(),
            code_block_count: 0,
        });
        return Ok(());
    }
    let cb_width = code_block_width;
    let cb_height = code_block_height;
    let num_cbs_x = subband.width.div_ceil(cb_width);
    let num_cbs_y = subband.height.div_ceil(cb_height);
    let code_block_start = code_blocks.len();
    for cby in 0..num_cbs_y {
        for cbx in 0..num_cbs_x {
            let block_x = cbx * cb_width;
            let block_y = cby * cb_height;
            let block_width = (block_x + cb_width).min(subband.width) - block_x;
            let block_height = (block_y + cb_height).min(subband.height) - block_y;
            let coeff_count =
                block_width
                    .checked_mul(block_height)
                    .ok_or_else(|| crate::Error::MetalKernel {
                        message: "J2K Metal resident encode code-block size overflow".to_string(),
                    })?;
            code_blocks.push(compute::J2kLosslessDeviceCodeBlock {
                coefficient_offset: *coefficient_offset,
                component: subband.component,
                subband_x: subband.subband_x,
                subband_y: subband.subband_y,
                block_x,
                block_y,
                width: block_width,
                height: block_height,
                sub_band_type: subband.sub_band_type,
                total_bitplanes: subband.total_bitplanes,
            });
            *coefficient_offset = coefficient_offset.checked_add(coeff_count).ok_or_else(|| {
                crate::Error::MetalKernel {
                    message: "J2K Metal resident encode coefficient offset overflow".to_string(),
                }
            })?;
        }
    }
    resolution.subbands.push(LosslessSubbandPlan {
        num_cbs_x,
        num_cbs_y,
        code_block_start,
        code_block_count: code_blocks.len() - code_block_start,
    });
    Ok(())
}

#[cfg(target_os = "macos")]
fn lossless_dwt_level_plans(
    width: u32,
    height: u32,
    num_decomposition_levels: u8,
) -> Vec<LosslessDwtLevelPlan> {
    let mut levels = Vec::with_capacity(usize::from(num_decomposition_levels));
    let mut current_width = width;
    let mut current_height = height;
    for _ in 0..num_decomposition_levels {
        let low_width = current_width.div_ceil(2);
        let low_height = current_height.div_ceil(2);
        let high_width = current_width / 2;
        let high_height = current_height / 2;
        levels.push(LosslessDwtLevelPlan {
            low_width,
            low_height,
            high_width,
            high_height,
        });
        current_width = low_width;
        current_height = low_height;
    }
    levels
}

#[cfg(target_os = "macos")]
fn lossless_device_encode_plan(
    width: u32,
    height: u32,
    components: u8,
    bit_depth: u8,
    options: J2kLosslessEncodeOptions,
    code_block_width: u32,
    code_block_height: u32,
) -> Result<Option<LosslessDeviceEncodePlan>, crate::Error> {
    if !matches!(
        options.block_coding_mode,
        J2kBlockCodingMode::Classic | J2kBlockCodingMode::HighThroughput
    ) {
        return Ok(None);
    }
    if code_block_width == 0 || code_block_height == 0 {
        return Err(crate::Error::MetalKernel {
            message: "J2K Metal resident encode code-block dimensions must be non-zero".to_string(),
        });
    }
    let code_block_width_exp = lossless_code_block_exp(code_block_width, "width")?;
    let code_block_height_exp = lossless_code_block_exp(code_block_height, "height")?;
    let num_decomposition_levels = lossless_device_encode_levels(width, height, options);
    let progression_order = match options.progression {
        J2kProgressionOrder::Lrcp => EncodeProgressionOrder::Lrcp,
        J2kProgressionOrder::Rlcp => EncodeProgressionOrder::Rlcp,
        J2kProgressionOrder::Rpcl => EncodeProgressionOrder::Rpcl,
        J2kProgressionOrder::Pcrl => EncodeProgressionOrder::Pcrl,
        J2kProgressionOrder::Cprl => EncodeProgressionOrder::Cprl,
    };
    let use_mct = components >= 3;
    let guard_bits: u8 = if use_mct { 2 } else { 1 };
    let mut code_blocks = Vec::new();
    let mut coefficient_offset = 0u32;
    let mut component_resolutions = Vec::<Vec<LosslessResolutionPlan>>::new();
    for component in 0..components {
        let mut component_packets = Vec::new();
        let dwt_levels = lossless_dwt_level_plans(width, height, num_decomposition_levels);
        let mut base_packet = LosslessResolutionPlan {
            subbands: Vec::new(),
        };
        if num_decomposition_levels == 0 {
            push_lossless_subband_plan(
                &mut base_packet,
                &mut code_blocks,
                &mut coefficient_offset,
                code_block_width,
                code_block_height,
                LosslessSubbandInput {
                    component: u32::from(component),
                    subband_x: 0,
                    subband_y: 0,
                    width,
                    height,
                    sub_band_type: J2kSubBandType::LowLow,
                    total_bitplanes: guard_bits.saturating_add(bit_depth).saturating_sub(1),
                },
            )?;
            component_packets.push(base_packet);
        } else {
            let final_ll = dwt_levels
                .last()
                .expect("nonzero DWT level count has a final LL level");
            push_lossless_subband_plan(
                &mut base_packet,
                &mut code_blocks,
                &mut coefficient_offset,
                code_block_width,
                code_block_height,
                LosslessSubbandInput {
                    component: u32::from(component),
                    subband_x: 0,
                    subband_y: 0,
                    width: final_ll.low_width,
                    height: final_ll.low_height,
                    sub_band_type: J2kSubBandType::LowLow,
                    total_bitplanes: guard_bits.saturating_add(bit_depth).saturating_sub(1),
                },
            )?;
            component_packets.push(base_packet);

            for level in dwt_levels.iter().rev().copied() {
                let mut detail_packet = LosslessResolutionPlan {
                    subbands: Vec::new(),
                };
                push_lossless_subband_plan(
                    &mut detail_packet,
                    &mut code_blocks,
                    &mut coefficient_offset,
                    code_block_width,
                    code_block_height,
                    LosslessSubbandInput {
                        component: u32::from(component),
                        subband_x: level.low_width,
                        subband_y: 0,
                        width: level.high_width,
                        height: level.low_height,
                        sub_band_type: J2kSubBandType::HighLow,
                        total_bitplanes: guard_bits.saturating_add(bit_depth),
                    },
                )?;
                push_lossless_subband_plan(
                    &mut detail_packet,
                    &mut code_blocks,
                    &mut coefficient_offset,
                    code_block_width,
                    code_block_height,
                    LosslessSubbandInput {
                        component: u32::from(component),
                        subband_x: 0,
                        subband_y: level.low_height,
                        width: level.low_width,
                        height: level.high_height,
                        sub_band_type: J2kSubBandType::LowHigh,
                        total_bitplanes: guard_bits.saturating_add(bit_depth),
                    },
                )?;
                push_lossless_subband_plan(
                    &mut detail_packet,
                    &mut code_blocks,
                    &mut coefficient_offset,
                    code_block_width,
                    code_block_height,
                    LosslessSubbandInput {
                        component: u32::from(component),
                        subband_x: level.low_width,
                        subband_y: level.low_height,
                        width: level.high_width,
                        height: level.high_height,
                        sub_band_type: J2kSubBandType::HighHigh,
                        total_bitplanes: guard_bits.saturating_add(bit_depth).saturating_add(1),
                    },
                )?;
                component_packets.push(detail_packet);
            }
        }
        component_resolutions.push(component_packets);
    }

    let resolution_count = component_resolutions.first().map_or(0usize, Vec::len);
    let mut resolutions =
        Vec::with_capacity(resolution_count.saturating_mul(usize::from(components)));
    for resolution in 0..resolution_count {
        for component in &component_resolutions {
            resolutions.push(component[resolution].clone());
        }
    }

    Ok(Some(LosslessDeviceEncodePlan {
        components,
        bit_depth,
        block_coding_mode: options.block_coding_mode,
        num_decomposition_levels,
        use_mct,
        guard_bits,
        code_block_width_exp,
        code_block_height_exp,
        code_blocks,
        resolutions,
        progression_order,
        write_tlm: options.progression == J2kProgressionOrder::Rpcl,
    }))
}

#[cfg(target_os = "macos")]
#[derive(Debug, Clone, Copy)]
enum MetalEncodeInputStaging {
    CopyAndPad,
    AlreadyPaddedContiguous,
}

#[cfg(target_os = "macos")]
fn submit_lossless_tiles(
    tiles: &[MetalLosslessEncodeTile<'_>],
    options: J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
    staging: MetalEncodeInputStaging,
    config: MetalLosslessEncodeConfig,
) -> Result<SubmittedJ2kLosslessMetalEncodeBatch, crate::Error> {
    if matches!(staging, MetalEncodeInputStaging::AlreadyPaddedContiguous)
        && should_try_resident_lossless_host_encode(options)
    {
        let mut ready = Vec::with_capacity(tiles.len());
        let mut all_ready = true;
        for &tile in tiles {
            validate_metal_encode_tile(tile)?;
            lossless_sample_shape(tile.format)?;
            validate_padded_contiguous_metal_encode_tile(tile, tile.format.bytes_per_pixel())?;
            if let Some(outcome) = try_encode_lossless_tile_device_resident_with_report(
                tile, options, session, staging,
            )? {
                ready.push(outcome.encoded);
            } else {
                all_ready = false;
                break;
            }
        }
        if all_ready {
            return Ok(SubmittedJ2kLosslessMetalEncodeBatch {
                state: SubmittedJ2kLosslessMetalEncodeBatchState::Ready(ready),
            });
        }
        if options.backend == EncodeBackendPreference::RequireDevice {
            return Err(crate::Error::UnsupportedMetalRequest {
                reason: "J2K Metal resident encode requires classic padded contiguous Gray/RGB lossless input with at most one DWT level",
            });
        }
    }

    let mut owned = Vec::with_capacity(tiles.len());
    for &tile in tiles {
        validate_metal_encode_tile(tile)?;
        if matches!(staging, MetalEncodeInputStaging::AlreadyPaddedContiguous) {
            lossless_sample_shape(tile.format)?;
            validate_padded_contiguous_metal_encode_tile(tile, tile.format.bytes_per_pixel())?;
        }
        owned.push(OwnedMetalLosslessEncodeTile::from_tile(tile));
    }
    Ok(SubmittedJ2kLosslessMetalEncodeBatch {
        state: SubmittedJ2kLosslessMetalEncodeBatchState::Deferred {
            tiles: owned,
            options,
            session: session.clone(),
            staging,
            config,
        },
    })
}

#[cfg(target_os = "macos")]
fn should_try_resident_lossless_host_encode(options: J2kLosslessEncodeOptions) -> bool {
    options.backend == EncodeBackendPreference::RequireDevice
}

#[cfg(target_os = "macos")]
fn host_output_encode_options(mut options: J2kLosslessEncodeOptions) -> J2kLosslessEncodeOptions {
    options.validation = J2kEncodeValidation::External;
    options
}

#[cfg(target_os = "macos")]
const AUTO_HTJ2K_HOST_RESIDENT_MIN_PIXELS: usize = 1024 * 1024;

#[cfg(target_os = "macos")]
fn lossless_progression_from_packetization_order(
    order: J2kPacketizationProgressionOrder,
) -> J2kProgressionOrder {
    match order {
        J2kPacketizationProgressionOrder::Lrcp => J2kProgressionOrder::Lrcp,
        J2kPacketizationProgressionOrder::Rlcp => J2kProgressionOrder::Rlcp,
        J2kPacketizationProgressionOrder::Rpcl => J2kProgressionOrder::Rpcl,
        J2kPacketizationProgressionOrder::Pcrl => J2kProgressionOrder::Pcrl,
        J2kPacketizationProgressionOrder::Cprl => J2kProgressionOrder::Cprl,
    }
}

#[cfg(target_os = "macos")]
fn lossless_options_for_resident_htj2k_tile_job(
    job: J2kHtj2kTileEncodeJob<'_>,
) -> Option<J2kLosslessEncodeOptions> {
    if job.num_components != 3
        || job.bit_depth != 8
        || job.signed
        || !job.reversible
        || !job.use_mct
        || job.guard_bits != 2
        || job.code_block_width != 64
        || job.code_block_height != 64
    {
        return None;
    }
    if job.component_sampling.len() != usize::from(job.num_components)
        || job
            .component_sampling
            .iter()
            .any(|&(x_sampling, y_sampling)| x_sampling != 1 || y_sampling != 1)
    {
        return None;
    }
    let expected_len = (job.width as usize)
        .checked_mul(job.height as usize)?
        .checked_mul(usize::from(job.num_components))?;
    if expected_len != job.pixels.len() {
        return None;
    }
    Some(J2kLosslessEncodeOptions::new(
        EncodeBackendPreference::Auto,
        J2kBlockCodingMode::HighThroughput,
        lossless_progression_from_packetization_order(job.progression_order),
        Some(job.num_decomposition_levels),
        ReversibleTransform::Rct53,
        J2kEncodeValidation::External,
    ))
}

#[cfg(target_os = "macos")]
fn should_use_resident_htj2k_host_tile_for_auto(job: J2kHtj2kTileEncodeJob<'_>) -> bool {
    (job.width as usize).saturating_mul(job.height as usize) >= AUTO_HTJ2K_HOST_RESIDENT_MIN_PIXELS
}

#[cfg(target_os = "macos")]
fn borrow_padded_metal_buffer_from_bytes(
    session: &crate::MetalBackendSession,
    bytes: &[u8],
) -> Result<Buffer, crate::Error> {
    if bytes.is_empty() {
        return Err(crate::Error::MetalKernel {
            message: "J2K Metal hybrid encode input is empty".to_string(),
        });
    }
    Ok(session.device().new_buffer_with_bytes_no_copy(
        bytes.as_ptr().cast(),
        bytes.len() as u64,
        metal::MTLResourceOptions::StorageModeShared,
        None,
    ))
}

#[cfg(target_os = "macos")]
fn packet_descriptors_for_lossless_device_order(
    packet_count: usize,
    num_components: u8,
    progression_order: EncodeProgressionOrder,
) -> Result<Vec<J2kPacketizationPacketDescriptor>, crate::Error> {
    let component_count = usize::from(num_components).max(1);
    let mut descriptors = (0..packet_count)
        .map(|packet_index| {
            Ok(J2kPacketizationPacketDescriptor {
                packet_index: u32::try_from(packet_index).map_err(|_| {
                    crate::Error::MetalKernel {
                        message: "J2K Metal resident encode packet index exceeds u32".to_string(),
                    }
                })?,
                state_index: u32::try_from(packet_index).map_err(|_| {
                    crate::Error::MetalKernel {
                        message: "J2K Metal resident encode packet state index exceeds u32"
                            .to_string(),
                    }
                })?,
                layer: 0,
                resolution: u32::try_from(packet_index / component_count).map_err(|_| {
                    crate::Error::MetalKernel {
                        message: "J2K Metal resident encode packet resolution exceeds u32"
                            .to_string(),
                    }
                })?,
                component: u8::try_from(packet_index % component_count).map_err(|_| {
                    crate::Error::MetalKernel {
                        message: "J2K Metal resident encode packet component exceeds u8"
                            .to_string(),
                    }
                })?,
                precinct: 0,
            })
        })
        .collect::<Result<Vec<_>, crate::Error>>()?;
    sort_lossless_device_packet_descriptors(&mut descriptors, progression_order);
    Ok(descriptors)
}

#[cfg(target_os = "macos")]
fn sort_lossless_device_packet_descriptors(
    descriptors: &mut [J2kPacketizationPacketDescriptor],
    progression_order: EncodeProgressionOrder,
) {
    match progression_order {
        EncodeProgressionOrder::Lrcp => descriptors.sort_by_key(|descriptor| {
            (
                descriptor.layer,
                descriptor.resolution,
                descriptor.component,
                descriptor.precinct,
            )
        }),
        EncodeProgressionOrder::Rlcp => descriptors.sort_by_key(|descriptor| {
            (
                descriptor.resolution,
                descriptor.layer,
                descriptor.component,
                descriptor.precinct,
            )
        }),
        EncodeProgressionOrder::Rpcl => descriptors.sort_by_key(|descriptor| {
            (
                descriptor.resolution,
                descriptor.precinct,
                descriptor.component,
                descriptor.layer,
            )
        }),
        EncodeProgressionOrder::Pcrl => descriptors.sort_by_key(|descriptor| {
            (
                descriptor.precinct,
                descriptor.component,
                descriptor.resolution,
                descriptor.layer,
            )
        }),
        EncodeProgressionOrder::Cprl => descriptors.sort_by_key(|descriptor| {
            (
                descriptor.component,
                descriptor.precinct,
                descriptor.resolution,
                descriptor.layer,
            )
        }),
    }
}

#[cfg(target_os = "macos")]
fn resident_packetization_resolutions_from_lossless_device_plan(
    plan: &LosslessDeviceEncodePlan,
) -> Result<Vec<compute::J2kResidentPacketizationResolution>, crate::Error> {
    plan.resolutions
        .iter()
        .map(|resolution| {
            let subbands = resolution
                .subbands
                .iter()
                .map(|subband| {
                    let code_block_end = subband
                        .code_block_start
                        .checked_add(subband.code_block_count)
                        .ok_or_else(|| crate::Error::MetalKernel {
                            message: "J2K Metal resident encode code-block range overflow"
                                .to_string(),
                        })?;
                    if code_block_end > plan.code_blocks.len() {
                        return Err(crate::Error::MetalKernel {
                            message: "J2K Metal resident encode code-block range out of bounds"
                                .to_string(),
                        });
                    }
                    Ok(compute::J2kResidentPacketizationSubband {
                        code_block_start: u32::try_from(subband.code_block_start).map_err(
                            |_| crate::Error::MetalKernel {
                                message: "J2K Metal resident encode code-block offset exceeds u32"
                                    .to_string(),
                            },
                        )?,
                        code_block_count: u32::try_from(subband.code_block_count).map_err(
                            |_| crate::Error::MetalKernel {
                                message: "J2K Metal resident encode code-block count exceeds u32"
                                    .to_string(),
                            },
                        )?,
                        num_cbs_x: subband.num_cbs_x,
                        num_cbs_y: subband.num_cbs_y,
                    })
                })
                .collect::<Result<Vec<_>, crate::Error>>()?;
            Ok(compute::J2kResidentPacketizationResolution { subbands })
        })
        .collect()
}

#[cfg(target_os = "macos")]
fn packetization_progression_order(
    order: EncodeProgressionOrder,
) -> J2kPacketizationProgressionOrder {
    match order {
        EncodeProgressionOrder::Lrcp => J2kPacketizationProgressionOrder::Lrcp,
        EncodeProgressionOrder::Rlcp => J2kPacketizationProgressionOrder::Rlcp,
        EncodeProgressionOrder::Rpcl => J2kPacketizationProgressionOrder::Rpcl,
        EncodeProgressionOrder::Pcrl => J2kPacketizationProgressionOrder::Pcrl,
        EncodeProgressionOrder::Cprl => J2kPacketizationProgressionOrder::Cprl,
    }
}

#[cfg(target_os = "macos")]
fn cpu_packetization_resolutions_from_lossless_device_plan<'a>(
    plan: &LosslessDeviceEncodePlan,
    encoded_blocks: &'a [EncodedHtJ2kCodeBlock],
) -> Result<Vec<J2kPacketizationResolution<'a>>, crate::Error> {
    if encoded_blocks.len() != plan.code_blocks.len() {
        return Err(crate::Error::MetalKernel {
            message: "J2K Metal resident hybrid HT block count mismatch".to_string(),
        });
    }
    plan.resolutions
        .iter()
        .map(|resolution| {
            let subbands = resolution
                .subbands
                .iter()
                .map(|subband| {
                    let code_block_end = subband
                        .code_block_start
                        .checked_add(subband.code_block_count)
                        .ok_or_else(|| crate::Error::MetalKernel {
                            message: "J2K Metal resident hybrid code-block range overflow"
                                .to_string(),
                        })?;
                    let encoded = encoded_blocks
                        .get(subband.code_block_start..code_block_end)
                        .ok_or_else(|| crate::Error::MetalKernel {
                            message: "J2K Metal resident hybrid code-block range out of bounds"
                                .to_string(),
                        })?;
                    let code_blocks = encoded
                        .iter()
                        .map(|block| J2kPacketizationCodeBlock {
                            data: block.data.as_slice(),
                            ht_cleanup_length: block.cleanup_length,
                            ht_refinement_length: block.refinement_length,
                            num_coding_passes: block.num_coding_passes,
                            num_zero_bitplanes: block.num_zero_bitplanes,
                            previously_included: false,
                            l_block: 3,
                            block_coding_mode: J2kPacketizationBlockCodingMode::HighThroughput,
                        })
                        .collect();
                    Ok(J2kPacketizationSubband {
                        code_blocks,
                        num_cbs_x: subband.num_cbs_x,
                        num_cbs_y: subband.num_cbs_y,
                    })
                })
                .collect::<Result<Vec<_>, crate::Error>>()?;
            Ok(J2kPacketizationResolution { subbands })
        })
        .collect()
}

#[cfg(target_os = "macos")]
struct ResidentHybridHtTileBody {
    tile_data: Vec<u8>,
    components: u8,
    bit_depth: u8,
    bytes_per_pixel: usize,
    code_block_count: usize,
    code_block_width_exp: u8,
    code_block_height_exp: u8,
    num_decomposition_levels: u8,
    used_fused_rct: bool,
    guard_bits: u8,
    progression_order: EncodeProgressionOrder,
    write_tlm: bool,
    forward_dwt53_dispatches: usize,
    ht_code_block_dispatches: usize,
}

#[cfg(target_os = "macos")]
fn encode_resident_ht_tile_body_with_cpu_packetization(
    tile: MetalLosslessEncodeTile<'_>,
    options: J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
    staging: MetalEncodeInputStaging,
    code_block_width: u32,
    code_block_height: u32,
) -> Result<Option<ResidentHybridHtTileBody>, crate::Error> {
    if !should_try_resident_lossless_ht_cpu_packetization(tile, options, staging) {
        return Ok(None);
    }
    validate_metal_encode_tile(tile)?;
    let (components, bit_depth) = lossless_sample_shape(tile.format)?;
    let bytes_per_pixel = tile.format.bytes_per_pixel();
    let bytes_per_sample =
        u8::try_from(tile.format.bytes_per_sample()).map_err(|_| crate::Error::MetalKernel {
            message: "J2K Metal resident hybrid bytes per sample exceeds u8".to_string(),
        })?;
    validate_padded_contiguous_metal_encode_tile(tile, bytes_per_pixel)?;
    let Some(plan) = lossless_device_encode_plan(
        tile.output_width,
        tile.output_height,
        components,
        bit_depth,
        options,
        code_block_width,
        code_block_height,
    )?
    else {
        return Ok(None);
    };
    if plan.block_coding_mode != J2kBlockCodingMode::HighThroughput {
        return Ok(None);
    }

    let coefficient_count = lossless_device_coefficient_count(&plan.code_blocks)?;
    let prepared = compute::prepare_lossless_device_code_blocks(
        session,
        compute::J2kLosslessDevicePrepareJob {
            input: tile.buffer,
            input_byte_offset: tile.byte_offset,
            input_width: tile.width,
            input_height: tile.height,
            input_pitch_bytes: tile.pitch_bytes,
            output_width: tile.output_width,
            output_height: tile.output_height,
            components,
            bytes_per_sample,
            bit_depth,
            num_decomposition_levels: plan.num_decomposition_levels,
            coefficient_count,
        },
        plan.code_blocks.clone(),
    )?;
    let resident_tier1 =
        compute::encode_ht_prepared_device_code_blocks_resident(session, prepared)?;
    let encoded_blocks = compute::read_resident_ht_tier1_code_blocks_for_cpu_packetization(
        session,
        &resident_tier1,
    )?;
    let packetization_resolutions =
        cpu_packetization_resolutions_from_lossless_device_plan(&plan, &encoded_blocks)?;
    let packet_descriptors = packet_descriptors_for_lossless_device_order(
        plan.resolutions.len(),
        plan.components,
        plan.progression_order,
    )?;
    let packetization_job = J2kPacketizationEncodeJob {
        resolution_count: u32::try_from(plan.resolutions.len()).map_err(|_| {
            crate::Error::MetalKernel {
                message: "J2K Metal resident hybrid resolution count exceeds u32".to_string(),
            }
        })?,
        num_layers: 1,
        num_components: plan.components,
        code_block_count: u32::try_from(plan.code_blocks.len()).map_err(|_| {
            crate::Error::MetalKernel {
                message: "J2K Metal resident hybrid code-block count exceeds u32".to_string(),
            }
        })?,
        progression_order: packetization_progression_order(plan.progression_order),
        packet_descriptors: &packet_descriptors,
        resolutions: &packetization_resolutions,
    };
    let tile_data = signinum_j2k_native::encode_j2k_packetization_scalar(packetization_job)
        .map_err(|reason| crate::Error::MetalKernel {
            message: format!("J2K Metal resident hybrid CPU packetization failed: {reason}"),
        })?;

    Ok(Some(ResidentHybridHtTileBody {
        tile_data,
        components,
        bit_depth,
        bytes_per_pixel,
        code_block_count: plan.code_blocks.len(),
        code_block_width_exp: plan.code_block_width_exp,
        code_block_height_exp: plan.code_block_height_exp,
        num_decomposition_levels: plan.num_decomposition_levels,
        used_fused_rct: plan.use_mct && tile.format == PixelFormat::Rgb8,
        guard_bits: plan.guard_bits,
        progression_order: plan.progression_order,
        write_tlm: plan.write_tlm,
        forward_dwt53_dispatches: if plan.num_decomposition_levels > 0 {
            usize::from(plan.components)
        } else {
            0
        },
        ht_code_block_dispatches: usize::from(!plan.code_blocks.is_empty()),
    }))
}

#[cfg(target_os = "macos")]
#[derive(Debug, Default)]
struct PrepacketizedHtj2kTileAccelerator {
    tile_data: Option<Vec<u8>>,
}

#[cfg(target_os = "macos")]
impl J2kEncodeStageAccelerator for PrepacketizedHtj2kTileAccelerator {
    fn encode_htj2k_tile(
        &mut self,
        _job: J2kHtj2kTileEncodeJob<'_>,
    ) -> core::result::Result<Option<Vec<u8>>, &'static str> {
        Ok(self.tile_data.take())
    }
}

#[cfg(target_os = "macos")]
fn lossless_device_coefficient_count(
    code_blocks: &[compute::J2kLosslessDeviceCodeBlock],
) -> Result<usize, crate::Error> {
    let mut count = 0usize;
    for block in code_blocks {
        let offset =
            usize::try_from(block.coefficient_offset).map_err(|_| crate::Error::MetalKernel {
                message: "J2K Metal resident encode coefficient offset exceeds usize".to_string(),
            })?;
        let block_count = (block.width as usize)
            .checked_mul(block.height as usize)
            .ok_or_else(|| crate::Error::MetalKernel {
                message: "J2K Metal resident encode coefficient count overflow".to_string(),
            })?;
        count = count.max(offset.checked_add(block_count).ok_or_else(|| {
            crate::Error::MetalKernel {
                message: "J2K Metal resident encode coefficient count overflow".to_string(),
            }
        })?);
    }
    Ok(count)
}

#[cfg(target_os = "macos")]
fn plan_resident_lossless_buffer_encode(
    index: usize,
    tile: MetalLosslessEncodeTile<'_>,
    options: J2kLosslessEncodeOptions,
    staging: MetalEncodeInputStaging,
) -> Result<Option<PlannedResidentLosslessBufferEncode>, crate::Error> {
    validate_metal_encode_tile(tile)?;
    if options.backend == EncodeBackendPreference::CpuOnly {
        return Ok(None);
    }
    let (components, bit_depth) = lossless_sample_shape(tile.format)?;
    let bytes_per_pixel = tile.format.bytes_per_pixel();
    let bytes_per_sample =
        u8::try_from(tile.format.bytes_per_sample()).map_err(|_| crate::Error::MetalKernel {
            message: "J2K Metal resident encode bytes per sample exceeds u8".to_string(),
        })?;
    if matches!(staging, MetalEncodeInputStaging::AlreadyPaddedContiguous) {
        validate_padded_contiguous_metal_encode_tile(tile, bytes_per_pixel)?;
    }
    let Some(plan) = lossless_device_encode_plan(
        tile.output_width,
        tile.output_height,
        components,
        bit_depth,
        options,
        RESIDENT_CLASSIC_CODE_BLOCK_EDGE,
        RESIDENT_CLASSIC_CODE_BLOCK_EDGE,
    )?
    else {
        return Ok(None);
    };
    let coefficient_count = lossless_device_coefficient_count(&plan.code_blocks)?;
    let packetization_resolutions =
        resident_packetization_resolutions_from_lossless_device_plan(&plan)?;
    let packet_descriptors = packet_descriptors_for_lossless_device_order(
        plan.resolutions.len(),
        plan.components,
        plan.progression_order,
    )?;
    let metadata = ResidentLosslessBufferEncodeMetadata {
        tile: OwnedMetalLosslessEncodeTile::from_tile(tile),
        components,
        bit_depth,
        bytes_per_pixel,
        plan,
        packet_descriptors,
        packetization_resolutions,
    };
    let estimated_peak_bytes =
        estimate_resident_lossless_encode_peak_bytes(&metadata, coefficient_count, staging);
    Ok(Some(PlannedResidentLosslessBufferEncode {
        index,
        metadata,
        coefficient_count,
        bytes_per_sample,
        estimated_peak_bytes,
        #[cfg(test)]
        failure_injection_index: test_resident_encode_failure_index(),
    }))
}

#[cfg(target_os = "macos")]
fn estimate_resident_lossless_encode_peak_bytes(
    metadata: &ResidentLosslessBufferEncodeMetadata,
    coefficient_count: usize,
    staging: MetalEncodeInputStaging,
) -> usize {
    let pixels = checked_mul_bytes(
        metadata.tile.output_width as usize,
        metadata.tile.output_height as usize,
    )
    .max(1);
    let plane_bytes = checked_mul_bytes(pixels, core::mem::size_of::<f32>());
    let code_block_count = metadata.plan.code_blocks.len().max(1);
    let packet_count = metadata
        .packet_descriptors
        .len()
        .max(metadata.plan.resolutions.len())
        .max(1);
    let input_bytes = checked_mul_bytes(
        checked_mul_bytes(metadata.tile.width as usize, metadata.tile.height as usize),
        metadata.bytes_per_pixel,
    );
    let staged_input_bytes = if matches!(staging, MetalEncodeInputStaging::CopyAndPad) {
        checked_mul_bytes(pixels, metadata.bytes_per_pixel)
    } else {
        0
    };
    let coefficient_bytes =
        checked_mul_bytes(coefficient_count.max(1), core::mem::size_of::<i32>());
    let plane_buffers = checked_mul_bytes(3, plane_bytes);
    let scratch_buffers = checked_mul_bytes(usize::from(metadata.components), plane_bytes);
    let code_block_tables = checked_mul_bytes(code_block_count, 256);
    let tier1_output = estimated_tier1_output_bytes(&metadata.plan);
    let packet_header = checked_add_bytes(checked_mul_bytes(code_block_count, 256), 4096);
    let packet_output = checked_add_bytes(
        checked_add_bytes(tier1_output, checked_mul_bytes(packet_header, packet_count)),
        1024,
    );
    let codestream_capacity = checked_add_bytes(
        packet_output,
        checked_add_bytes(4096, checked_mul_bytes(pixels, metadata.bytes_per_pixel)),
    );
    let validation_bytes = checked_mul_bytes(pixels, metadata.bytes_per_pixel).saturating_mul(
        usize::from(metadata.plan.write_tlm || metadata.plan.use_mct || metadata.components > 0),
    );

    [
        input_bytes / 4,
        staged_input_bytes,
        plane_buffers,
        scratch_buffers,
        coefficient_bytes,
        code_block_tables,
        tier1_output,
        packet_output,
        codestream_capacity,
        validation_bytes,
        4 * 1024 * 1024,
    ]
    .into_iter()
    .fold(0usize, checked_add_bytes)
}

#[cfg(target_os = "macos")]
fn estimated_tier1_output_bytes(plan: &LosslessDeviceEncodePlan) -> usize {
    fn estimated_ht_output_capacity(width: usize, height: usize) -> usize {
        const HT_MAX_SAMPLES: usize = 16_384;
        const HT_MEL_SIZE: usize = 192;
        const HT_VLC_SIZE: usize = 3072 - HT_MEL_SIZE;
        const HT_MS_SIZE: usize = (HT_MAX_SAMPLES * 16).div_ceil(15);
        const HT_MS_BYTES_PER_SAMPLE_FLOOR: usize = 5;

        let samples = checked_mul_bytes(width, height).min(HT_MAX_SAMPLES);
        let scaled_ms = checked_mul_bytes(HT_MS_SIZE, samples)
            .div_ceil(HT_MAX_SAMPLES)
            .max(1);
        let ms_floor = checked_mul_bytes(samples, HT_MS_BYTES_PER_SAMPLE_FLOOR);
        let ms_size = scaled_ms.max(ms_floor).min(HT_MS_SIZE);
        let fixed_entropy = checked_add_bytes(HT_MEL_SIZE, HT_VLC_SIZE);
        checked_add_bytes(ms_size, fixed_entropy)
    }

    plan.code_blocks
        .iter()
        .map(|block| match plan.block_coding_mode {
            J2kBlockCodingMode::HighThroughput => {
                estimated_ht_output_capacity(block.width as usize, block.height as usize)
            }
            J2kBlockCodingMode::Classic => {
                let samples = checked_mul_bytes(block.width as usize, block.height as usize);
                checked_add_bytes(
                    checked_mul_bytes(samples, usize::from(block.total_bitplanes).max(1)),
                    4097,
                )
                .max(4097)
            }
        })
        .fold(0usize, checked_add_bytes)
        .max(1)
}

#[cfg(target_os = "macos")]
fn resident_codestream_assembly_job_for_metadata(
    metadata: &ResidentLosslessBufferEncodeMetadata,
) -> compute::J2kLosslessCodestreamAssemblyJob {
    compute::J2kLosslessCodestreamAssemblyJob {
        width: metadata.tile.output_width,
        height: metadata.tile.output_height,
        num_components: metadata.plan.components,
        bit_depth: metadata.plan.bit_depth,
        signed: false,
        num_decomposition_levels: metadata.plan.num_decomposition_levels,
        use_mct: metadata.plan.use_mct,
        guard_bits: metadata.plan.guard_bits,
        code_block_width_exp: metadata.plan.code_block_width_exp,
        code_block_height_exp: metadata.plan.code_block_height_exp,
        progression_order: metadata.plan.progression_order,
        write_tlm: metadata.plan.write_tlm,
        block_coding_mode: match metadata.plan.block_coding_mode {
            J2kBlockCodingMode::Classic => compute::J2kLosslessCodestreamBlockCodingMode::Classic,
            J2kBlockCodingMode::HighThroughput => {
                compute::J2kLosslessCodestreamBlockCodingMode::HighThroughput
            }
        },
    }
}

#[cfg(target_os = "macos")]
fn resident_classic_batch_encode_should_retry_conservative(error: &crate::Error) -> bool {
    let crate::Error::MetalKernel { message } = error else {
        return false;
    };

    message.contains("classic Tier-1 Metal encode kernel failure (detail=4)")
        || message.contains("classic Tier-1 Metal encode kernel failure (detail=5)")
        || message.contains("packetization Metal encode kernel failure (detail=3)")
        || message.contains("packetization Metal encode kernel failure (detail=4)")
        || message.contains("packetization Metal encode kernel failure (detail=5)")
        || message.contains("packetization Metal encode kernel failure (detail=7, tier1_detail=4)")
        || message.contains("packetization Metal encode kernel failure (detail=7, tier1_detail=5)")
        || message
            .contains("J2K batched codestream assembly Metal encode kernel failure (detail=2)")
        || message
            .contains("J2K batched codestream assembly Metal encode kernel failure (detail=3)")
}

#[cfg(target_os = "macos")]
fn resident_ht_batch_encode_should_retry_conservative(error: &crate::Error) -> bool {
    let crate::Error::MetalKernel { message } = error else {
        return false;
    };

    message.contains("packetization Metal encode kernel failure (detail=3)")
        || message.contains("packetization Metal encode kernel failure (detail=4)")
        || message.contains("packetization Metal encode kernel failure (detail=5)")
        || message
            .contains("HTJ2K batched codestream assembly Metal encode kernel failure (detail=2)")
        || message
            .contains("HTJ2K batched codestream assembly Metal encode kernel failure (detail=3)")
}

#[cfg(target_os = "macos")]
fn wait_submitted_resident_lossless_buffer_encode_batch(
    mut submitted: SubmittedResidentLosslessMetalBufferEncodeBatch,
) -> Result<MetalLosslessBufferEncodeBatchOutcome, crate::Error> {
    let result = wait_submitted_resident_lossless_buffer_encode_batch_once(&mut submitted);
    match result {
        Ok(outcome) => Ok(outcome),
        Err(err) => {
            if submitted.options.block_coding_mode == J2kBlockCodingMode::Classic
                && !submitted.tiles.is_empty()
                && resident_classic_batch_encode_should_retry_conservative(&err)
            {
                return encode_owned_lossless_tiles_to_metal_buffer_fallback_batch(
                    &submitted.tiles,
                    submitted.options,
                    &submitted.session,
                    submitted.staging,
                )
                .map_err(|retry_err| crate::Error::MetalKernel {
                    message: format!(
                        "J2K Metal resident classic batch conservative retry failed after tight resident capacity failure ({err}); retry error: {retry_err}"
                    ),
                });
            }
            if submitted.options.block_coding_mode == J2kBlockCodingMode::HighThroughput
                && !submitted.tiles.is_empty()
                && resident_ht_batch_encode_should_retry_conservative(&err)
            {
                return encode_owned_lossless_tiles_to_metal_buffer_fallback_batch(
                    &submitted.tiles,
                    submitted.options,
                    &submitted.session,
                    submitted.staging,
                )
                .map_err(|retry_err| crate::Error::MetalKernel {
                    message: format!(
                        "J2K Metal resident HT batch conservative retry failed after tight packet capacity failure ({err}); retry error: {retry_err}"
                    ),
                });
            }
            Err(err)
        }
    }
}

#[cfg(target_os = "macos")]
fn wait_submitted_resident_lossless_buffer_encode_batch_once(
    submitted: &mut SubmittedResidentLosslessMetalBufferEncodeBatch,
) -> Result<MetalLosslessBufferEncodeBatchOutcome, crate::Error> {
    let mut outcomes = Vec::new();
    match std::mem::replace(
        &mut submitted.kind,
        SubmittedResidentLosslessMetalBufferEncodeBatchKind::Empty,
    ) {
        SubmittedResidentLosslessMetalBufferEncodeBatchKind::Empty => {}
        SubmittedResidentLosslessMetalBufferEncodeBatchKind::Chunks(chunks) => {
            outcomes.reserve(chunks.iter().map(|chunk| chunk.metadatas.len()).sum());
            if submitted.options.validation == J2kEncodeValidation::External
                && submitted.options.block_coding_mode == J2kBlockCodingMode::HighThroughput
                && chunks.len() > 1
            {
                let wait_started = compute::metal_profile_stages_enabled().then(Instant::now);
                let mut chunk_metadatas = Vec::with_capacity(chunks.len());
                let mut pending_batches = Vec::with_capacity(chunks.len());
                for chunk in chunks {
                    chunk_metadatas.push((
                        chunk.metadatas,
                        chunk.prepare_durations,
                        chunk.batch_started,
                    ));
                    pending_batches.push(chunk.pending);
                }
                let batches = compute::wait_resident_lossless_codestream_batches(pending_batches)?;
                if let Some(started) = wait_started {
                    let elapsed = started.elapsed();
                    submitted.stats.stage_stats.codestream_wait_duration = submitted
                        .stats
                        .stage_stats
                        .codestream_wait_duration
                        .saturating_add(elapsed);
                    submitted.stats.stage_stats.sync_wait_duration = submitted
                        .stats
                        .stage_stats
                        .sync_wait_duration
                        .saturating_add(elapsed);
                }
                for ((metadatas, prepare_durations, batch_started), batch) in
                    chunk_metadatas.into_iter().zip(batches)
                {
                    if compute::metal_profile_stages_enabled() {
                        submitted
                            .stats
                            .stage_stats
                            .add_assign(MetalLosslessEncodeStageStats::from(batch.stage_stats));
                    }
                    let codestreams = batch.codestreams;
                    let batch_duration = duration_share(batch_started.elapsed(), codestreams.len());
                    for ((metadata, prepare_duration), codestream) in metadatas
                        .into_iter()
                        .zip(prepare_durations)
                        .zip(codestreams)
                    {
                        let finished = finished_resident_lossless_buffer_encode(
                            metadata,
                            codestream,
                            prepare_duration.saturating_add(batch_duration),
                        );
                        outcomes.push(validate_finished_resident_lossless_buffer_encode(
                            finished,
                            submitted.options,
                            &submitted.session,
                        )?);
                    }
                }
            } else {
                for chunk in chunks {
                    let wait_started = compute::metal_profile_stages_enabled().then(Instant::now);
                    let batch = compute::wait_resident_lossless_codestream_batch(chunk.pending)?;
                    if let Some(started) = wait_started {
                        let elapsed = started.elapsed();
                        submitted.stats.stage_stats.codestream_wait_duration = submitted
                            .stats
                            .stage_stats
                            .codestream_wait_duration
                            .saturating_add(elapsed);
                        submitted.stats.stage_stats.sync_wait_duration = submitted
                            .stats
                            .stage_stats
                            .sync_wait_duration
                            .saturating_add(elapsed);
                        submitted
                            .stats
                            .stage_stats
                            .add_assign(MetalLosslessEncodeStageStats::from(batch.stage_stats));
                    }
                    let codestreams = batch.codestreams;
                    let batch_duration =
                        duration_share(chunk.batch_started.elapsed(), codestreams.len());
                    for ((metadata, prepare_duration), codestream) in chunk
                        .metadatas
                        .into_iter()
                        .zip(chunk.prepare_durations)
                        .zip(codestreams)
                    {
                        let finished = finished_resident_lossless_buffer_encode(
                            metadata,
                            codestream,
                            prepare_duration.saturating_add(batch_duration),
                        );
                        outcomes.push(validate_finished_resident_lossless_buffer_encode(
                            finished,
                            submitted.options,
                            &submitted.session,
                        )?);
                    }
                }
            }
        }
    }
    submitted.stats.encode_wall_duration = submitted.encode_started.elapsed();
    Ok(MetalLosslessBufferEncodeBatchOutcome {
        outcomes,
        stats: submitted.stats,
    })
}

#[cfg(target_os = "macos")]
fn finished_resident_lossless_buffer_encode(
    metadata: ResidentLosslessBufferEncodeMetadata,
    codestream: compute::J2kResidentLosslessCodestream,
    encode_duration: Duration,
) -> FinishedResidentLosslessBufferEncode {
    let encoded = MetalEncodedJ2k {
        codestream_buffer: codestream.buffer,
        byte_offset: codestream.byte_offset,
        byte_len: codestream.byte_len,
        capacity: codestream.capacity,
        width: metadata.tile.output_width,
        height: metadata.tile.output_height,
        components: metadata.components,
        bit_depth: metadata.bit_depth,
        signed: false,
    };

    FinishedResidentLosslessBufferEncode {
        metadata,
        encoded,
        encode_duration,
        gpu_duration: codestream.gpu_duration,
    }
}

#[cfg(target_os = "macos")]
fn validate_finished_resident_lossless_buffer_encode(
    finished: FinishedResidentLosslessBufferEncode,
    options: J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<MetalLosslessBufferEncodeOutcome, crate::Error> {
    let FinishedResidentLosslessBufferEncode {
        metadata,
        encoded,
        encode_duration,
        gpu_duration,
    } = finished;

    let validation_duration = if options.validation == J2kEncodeValidation::CpuRoundTrip {
        let validation_started = Instant::now();
        let tile = metadata.tile.as_tile();
        if tile.width == tile.output_width
            && tile.height == tile.output_height
            && tile.pitch_bytes == tile.output_width as usize * metadata.bytes_per_pixel
        {
            validate_lossless_roundtrip_on_metal_tile_with_session(
                tile,
                encoded.codestream_bytes()?,
                session,
            )?;
        } else {
            validate_lossless_roundtrip_on_metal_region_with_session(
                tile,
                tile.output_width,
                tile.output_height,
                metadata.bytes_per_pixel,
                encoded.codestream_bytes()?,
                session,
            )?;
        }
        validation_started.elapsed()
    } else {
        Duration::ZERO
    };

    Ok(MetalLosslessBufferEncodeOutcome {
        encoded,
        input_copy_used: false,
        resident: MetalLosslessEncodeResidency {
            coefficient_prep_used: true,
            packetization_used: true,
            codestream_assembly_used: true,
        },
        input_copy_duration: Duration::ZERO,
        encode_duration,
        gpu_duration,
        validation_duration,
    })
}

#[cfg(target_os = "macos")]
#[cfg(test)]
struct InflightLimitedOrderedItems<T> {
    items: Vec<T>,
    max_observed_inflight_items: usize,
}

#[cfg(target_os = "macos")]
#[cfg(test)]
fn collect_inflight_limited_ordered<T, O, F>(
    items: Vec<T>,
    inflight_items: usize,
    f: F,
) -> Result<InflightLimitedOrderedItems<O>, crate::Error>
where
    T: Send,
    O: Send,
    F: Fn(usize, T) -> Result<O, crate::Error> + Sync,
{
    if items.is_empty() {
        return Ok(InflightLimitedOrderedItems {
            items: Vec::new(),
            max_observed_inflight_items: 0,
        });
    }

    let active = Arc::new(AtomicUsize::new(0));
    let observed = Arc::new(AtomicUsize::new(0));
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(inflight_items.max(1))
        .build()
        .map_err(|err| crate::Error::MetalKernel {
            message: format!("J2K Metal encode worker pool initialization failed: {err}"),
        })?;

    let active_for_tasks = Arc::clone(&active);
    let observed_for_tasks = Arc::clone(&observed);
    let results = pool.install(|| {
        items
            .into_par_iter()
            .enumerate()
            .map(|(index, item)| {
                let _guard = ActiveTileGuard::new(&active_for_tasks, &observed_for_tasks);
                f(index, item)
            })
            .collect::<Vec<_>>()
    });

    let max_observed_inflight_items = observed.load(Ordering::Relaxed);
    let mut ordered = Vec::with_capacity(results.len());
    let mut first_error = None;
    for result in results {
        match result {
            Ok(item) if first_error.is_none() => ordered.push(item),
            Ok(_) => {}
            Err(err) => {
                if first_error.is_none() {
                    first_error = Some(err);
                }
            }
        }
    }

    if let Some(err) = first_error {
        return Err(err);
    }

    Ok(InflightLimitedOrderedItems {
        items: ordered,
        max_observed_inflight_items,
    })
}

#[cfg(target_os = "macos")]
fn submit_planned_resident_lossless_tiles(
    planned: Vec<PlannedResidentLosslessBufferEncode>,
    session: &crate::MetalBackendSession,
    inflight_tiles: usize,
    stats: &mut MetalLosslessEncodeBatchStats,
) -> Result<SubmittedResidentLosslessMetalBufferEncodeBatchKind, crate::Error> {
    if planned.is_empty() {
        return Ok(SubmittedResidentLosslessMetalBufferEncodeBatchKind::Empty);
    }
    if planned.iter().all(|planned| {
        planned.metadata.plan.block_coding_mode == J2kBlockCodingMode::HighThroughput
    }) {
        return submit_planned_resident_ht_lossless_tiles_batch(
            planned,
            session,
            inflight_tiles,
            stats,
        );
    }
    if planned
        .iter()
        .all(|planned| planned.metadata.plan.block_coding_mode == J2kBlockCodingMode::Classic)
    {
        return submit_planned_resident_classic_lossless_tiles_batch(
            planned,
            session,
            inflight_tiles,
            stats,
        );
    }
    Ok(SubmittedResidentLosslessMetalBufferEncodeBatchKind::Empty)
}

#[cfg(target_os = "macos")]
struct PreparedResidentLosslessBatchItem {
    prepared: PreparedResidentLosslessBufferEncode,
    prepare_duration: Duration,
}

#[cfg(target_os = "macos")]
fn prepare_planned_resident_ht_lossless_tiles_batch(
    planned: Vec<PlannedResidentLosslessBufferEncode>,
    session: &crate::MetalBackendSession,
) -> Result<Vec<PreparedResidentLosslessBatchItem>, crate::Error> {
    struct HtBatchPlanInfo {
        index: usize,
        coefficient_count: usize,
        bytes_per_sample: u8,
        code_blocks: Vec<compute::J2kLosslessDeviceCodeBlock>,
    }

    let started = Instant::now();
    let mut metadatas = Vec::with_capacity(planned.len());
    let mut plan_infos = Vec::with_capacity(planned.len());
    for planned in planned {
        #[cfg(test)]
        if planned.failure_injection_index == Some(planned.index) {
            return Err(crate::Error::MetalKernel {
                message: format!(
                    "injected J2K Metal resident encode failure at tile {}",
                    planned.index
                ),
            });
        }

        plan_infos.push(HtBatchPlanInfo {
            index: planned.index,
            coefficient_count: planned.coefficient_count,
            bytes_per_sample: planned.bytes_per_sample,
            code_blocks: planned.metadata.plan.code_blocks.clone(),
        });
        metadatas.push(planned.metadata);
    }

    let mut batch_items = Vec::with_capacity(metadatas.len());
    for (metadata, plan_info) in metadatas.iter().zip(plan_infos) {
        let tile = metadata.tile.as_tile();
        batch_items.push(compute::J2kLosslessDeviceBatchPrepareItem {
            tile_index: plan_info.index,
            job: compute::J2kLosslessDevicePrepareJob {
                input: tile.buffer,
                input_byte_offset: tile.byte_offset,
                input_width: tile.width,
                input_height: tile.height,
                input_pitch_bytes: tile.pitch_bytes,
                output_width: tile.output_width,
                output_height: tile.output_height,
                components: metadata.components,
                bytes_per_sample: plan_info.bytes_per_sample,
                bit_depth: metadata.bit_depth,
                num_decomposition_levels: metadata.plan.num_decomposition_levels,
                coefficient_count: plan_info.coefficient_count,
            },
            code_blocks: plan_info.code_blocks,
        });
    }

    let prepared = compute::prepare_lossless_device_code_blocks_batch(session, batch_items)?;
    let prepare_duration = duration_share(started.elapsed(), prepared.len());
    Ok(metadatas
        .into_iter()
        .zip(prepared)
        .map(|(metadata, prepared)| PreparedResidentLosslessBatchItem {
            prepared: PreparedResidentLosslessBufferEncode { metadata, prepared },
            prepare_duration,
        })
        .collect())
}

#[cfg(target_os = "macos")]
fn submit_planned_resident_ht_lossless_tiles_batch(
    mut planned: Vec<PlannedResidentLosslessBufferEncode>,
    session: &crate::MetalBackendSession,
    inflight_tiles: usize,
    stats: &mut MetalLosslessEncodeBatchStats,
) -> Result<SubmittedResidentLosslessMetalBufferEncodeBatchKind, crate::Error> {
    let planned_len = planned.len();
    let profile_stages = compute::metal_profile_stages_enabled();
    let code_block_counts = planned
        .iter()
        .map(|planned| planned.metadata.plan.code_blocks.len())
        .collect::<Vec<_>>();
    let chunk_ranges = resident_lossless_chunk_ranges_from_code_blocks(
        &code_block_counts,
        inflight_tiles,
        resident_lossless_code_block_chunk_cap(&code_block_counts),
    );
    if profile_stages {
        stats.stage_stats.chunk_count = stats
            .stage_stats
            .chunk_count
            .saturating_add(chunk_ranges.len());
        stats.stage_stats.tile_count = stats.stage_stats.tile_count.saturating_add(planned_len);
    }
    stats.max_observed_inflight_tiles = stats.max_observed_inflight_tiles.max(
        chunk_ranges
            .iter()
            .map(std::ops::Range::len)
            .max()
            .unwrap_or(0),
    );

    let mut chunks = Vec::with_capacity(chunk_ranges.len());
    for range in chunk_ranges {
        let take = range.len();
        let chunk_planned = planned.drain(..take).collect::<Vec<_>>();
        let prepare_submit_started = profile_stages.then(Instant::now);
        let prep_wall_started = profile_stages.then(Instant::now);
        let prepared = prepare_planned_resident_ht_lossless_tiles_batch(chunk_planned, session)
            .map_err(|err| crate::Error::MetalKernel {
                message: format!("J2K Metal resident HT batch encode failed: {err}"),
            })?;
        if let Some(started) = prep_wall_started {
            add_resident_prep_wall_duration(stats, started.elapsed(), profile_stages);
        }

        let mut metadatas = Vec::with_capacity(prepared.len());
        let mut prepare_durations = Vec::with_capacity(prepared.len());
        let mut batch_items = Vec::with_capacity(prepared.len());
        for item in prepared {
            let PreparedResidentLosslessBatchItem {
                prepared,
                prepare_duration,
            } = item;
            let metadata = prepared.metadata;
            let codestream = resident_codestream_assembly_job_for_metadata(&metadata);
            batch_items.push(compute::J2kResidentHtBatchEncodeItem {
                prepared: prepared.prepared,
                resolution_count: u32::try_from(metadata.plan.resolutions.len()).map_err(|_| {
                    crate::Error::MetalKernel {
                        message: "J2K Metal resident encode resolution count exceeds u32"
                            .to_string(),
                    }
                })?,
                num_layers: 1,
                num_components: metadata.plan.components,
                code_block_count: u32::try_from(metadata.plan.code_blocks.len()).map_err(|_| {
                    crate::Error::MetalKernel {
                        message: "J2K Metal resident encode code-block count exceeds u32"
                            .to_string(),
                    }
                })?,
                packet_descriptors: metadata.packet_descriptors.clone(),
                resolutions: metadata.packetization_resolutions.clone(),
                codestream,
            });
            prepare_durations.push(prepare_duration);
            metadatas.push(metadata);
        }

        let batch_started = Instant::now();
        let pending = compute::submit_lossless_codestream_buffers_from_prepared_ht_batch(
            session,
            batch_items,
            compute::ht_packet_output_capacity_mode_from_env(),
        )?;
        if let Some(started) = prepare_submit_started {
            stats.stage_stats.prepare_submit_duration = stats
                .stage_stats
                .prepare_submit_duration
                .saturating_add(started.elapsed());
        }
        chunks.push(SubmittedResidentLosslessMetalBufferEncodeChunk {
            metadatas,
            prepare_durations,
            pending,
            batch_started,
        });
    }

    if !planned.is_empty() {
        return Err(crate::Error::MetalKernel {
            message: "J2K Metal resident HT batch chunking left unsubmitted tiles".to_string(),
        });
    }

    if chunks.is_empty() && planned_len > 0 {
        return Err(crate::Error::MetalKernel {
            message: "J2K Metal resident HT batch chunking produced no chunks".to_string(),
        });
    }

    Ok(SubmittedResidentLosslessMetalBufferEncodeBatchKind::Chunks(
        chunks,
    ))
}

#[cfg(target_os = "macos")]
fn submit_planned_resident_classic_lossless_tiles_batch(
    mut planned: Vec<PlannedResidentLosslessBufferEncode>,
    session: &crate::MetalBackendSession,
    inflight_tiles: usize,
    stats: &mut MetalLosslessEncodeBatchStats,
) -> Result<SubmittedResidentLosslessMetalBufferEncodeBatchKind, crate::Error> {
    let planned_len = planned.len();
    let profile_stages = compute::metal_profile_stages_enabled();

    let batch_limit = inflight_tiles.max(1);
    if profile_stages {
        let chunk_count = planned_len.div_ceil(batch_limit);
        stats.stage_stats.chunk_count = stats.stage_stats.chunk_count.saturating_add(chunk_count);
        stats.stage_stats.tile_count = stats.stage_stats.tile_count.saturating_add(planned_len);
    }
    let mut chunks = Vec::new();
    while !planned.is_empty() {
        let take = planned.len().min(batch_limit);
        stats.max_observed_inflight_tiles = stats.max_observed_inflight_tiles.max(take);
        let chunk_planned = planned.drain(..take).collect::<Vec<_>>();
        let prep_wall_started = profile_stages.then(Instant::now);
        let prepared =
            prepare_planned_resident_classic_lossless_tiles_batch(chunk_planned, session).map_err(
                |err| crate::Error::MetalKernel {
                    message: format!("J2K Metal resident classic batch encode failed: {err}"),
                },
            )?;
        if let Some(started) = prep_wall_started {
            add_resident_prep_wall_duration(stats, started.elapsed(), profile_stages);
        }

        let prepared_count = prepared.len();
        let mut chunk_metadatas = Vec::with_capacity(prepared_count);
        let mut chunk_prepare_durations = Vec::with_capacity(prepared_count);
        let mut chunk_items = Vec::with_capacity(prepared_count);
        for item in prepared {
            let PreparedResidentLosslessBatchItem {
                prepared,
                prepare_duration,
            } = item;
            let metadata = prepared.metadata;
            let codestream = resident_codestream_assembly_job_for_metadata(&metadata);
            chunk_items.push(compute::J2kResidentClassicBatchEncodeItem {
                prepared: prepared.prepared,
                resolution_count: u32::try_from(metadata.plan.resolutions.len()).map_err(|_| {
                    crate::Error::MetalKernel {
                        message: "J2K Metal resident encode resolution count exceeds u32"
                            .to_string(),
                    }
                })?,
                num_layers: 1,
                num_components: metadata.plan.components,
                code_block_count: u32::try_from(metadata.plan.code_blocks.len()).map_err(|_| {
                    crate::Error::MetalKernel {
                        message: "J2K Metal resident encode code-block count exceeds u32"
                            .to_string(),
                    }
                })?,
                packet_descriptors: metadata.packet_descriptors.clone(),
                resolutions: metadata.packetization_resolutions.clone(),
                codestream,
            });
            chunk_prepare_durations.push(prepare_duration);
            chunk_metadatas.push(metadata);
        }
        let batch_started = Instant::now();
        let prepare_submit_started = profile_stages.then(Instant::now);
        let pending = compute::submit_lossless_codestream_buffers_from_prepared_classic_batch(
            session,
            chunk_items,
            compute::J2kClassicEncodeOutputCapacityMode::Tight,
        )?;
        if let Some(started) = prepare_submit_started {
            stats.stage_stats.prepare_submit_duration = stats
                .stage_stats
                .prepare_submit_duration
                .saturating_add(started.elapsed());
        }
        chunks.push(SubmittedResidentLosslessMetalBufferEncodeChunk {
            metadatas: chunk_metadatas,
            prepare_durations: chunk_prepare_durations,
            pending,
            batch_started,
        });
    }
    Ok(SubmittedResidentLosslessMetalBufferEncodeBatchKind::Chunks(
        chunks,
    ))
}

#[cfg(target_os = "macos")]
fn prepare_planned_resident_classic_lossless_tiles_batch(
    planned: Vec<PlannedResidentLosslessBufferEncode>,
    session: &crate::MetalBackendSession,
) -> Result<Vec<PreparedResidentLosslessBatchItem>, crate::Error> {
    struct ClassicBatchPlanInfo {
        index: usize,
        coefficient_count: usize,
        bytes_per_sample: u8,
        code_blocks: Vec<compute::J2kLosslessDeviceCodeBlock>,
    }

    let started = Instant::now();
    let mut metadatas = Vec::with_capacity(planned.len());
    let mut plan_infos = Vec::with_capacity(planned.len());
    for planned in planned {
        #[cfg(test)]
        if planned.failure_injection_index == Some(planned.index) {
            return Err(crate::Error::MetalKernel {
                message: format!(
                    "injected J2K Metal resident encode failure at tile {}",
                    planned.index
                ),
            });
        }

        plan_infos.push(ClassicBatchPlanInfo {
            index: planned.index,
            coefficient_count: planned.coefficient_count,
            bytes_per_sample: planned.bytes_per_sample,
            code_blocks: planned.metadata.plan.code_blocks.clone(),
        });
        metadatas.push(planned.metadata);
    }

    let mut batch_items = Vec::with_capacity(metadatas.len());
    for (metadata, plan_info) in metadatas.iter().zip(plan_infos) {
        let tile = metadata.tile.as_tile();
        batch_items.push(compute::J2kLosslessDeviceBatchPrepareItem {
            tile_index: plan_info.index,
            job: compute::J2kLosslessDevicePrepareJob {
                input: tile.buffer,
                input_byte_offset: tile.byte_offset,
                input_width: tile.width,
                input_height: tile.height,
                input_pitch_bytes: tile.pitch_bytes,
                output_width: tile.output_width,
                output_height: tile.output_height,
                components: metadata.components,
                bytes_per_sample: plan_info.bytes_per_sample,
                bit_depth: metadata.bit_depth,
                num_decomposition_levels: metadata.plan.num_decomposition_levels,
                coefficient_count: plan_info.coefficient_count,
            },
            code_blocks: plan_info.code_blocks,
        });
    }

    let prepared = compute::prepare_lossless_device_code_blocks_batch(session, batch_items)?;
    let prepare_duration = duration_share(started.elapsed(), prepared.len());
    Ok(metadatas
        .into_iter()
        .zip(prepared)
        .map(|(metadata, prepared)| PreparedResidentLosslessBatchItem {
            prepared: PreparedResidentLosslessBufferEncode { metadata, prepared },
            prepare_duration,
        })
        .collect())
}

#[cfg(target_os = "macos")]
fn duration_share(duration: Duration, count: usize) -> Duration {
    if count == 0 {
        return Duration::ZERO;
    }
    let nanos = duration.as_nanos() / count as u128;
    Duration::from_nanos(nanos.min(u128::from(u64::MAX)) as u64)
}

#[cfg(target_os = "macos")]
#[cfg(test)]
struct ActiveTileGuard<'a> {
    active: &'a AtomicUsize,
}

#[cfg(target_os = "macos")]
#[cfg(test)]
impl<'a> ActiveTileGuard<'a> {
    fn new(active: &'a AtomicUsize, observed: &AtomicUsize) -> Self {
        let now = active.fetch_add(1, Ordering::AcqRel).saturating_add(1);
        let mut current = observed.load(Ordering::Relaxed);
        while now > current {
            match observed.compare_exchange(current, now, Ordering::AcqRel, Ordering::Relaxed) {
                Ok(_) => break,
                Err(next) => current = next,
            }
        }
        Self { active }
    }
}

#[cfg(target_os = "macos")]
#[cfg(test)]
impl Drop for ActiveTileGuard<'_> {
    fn drop(&mut self) {
        self.active.fetch_sub(1, Ordering::AcqRel);
    }
}

#[cfg(all(test, target_os = "macos"))]
thread_local! {
    static TEST_RESIDENT_ENCODE_FAILURE_INDEX: Cell<Option<usize>> = const { Cell::new(None) };
}

#[cfg(all(test, target_os = "macos"))]
fn set_test_resident_encode_failure_index(index: Option<usize>) {
    TEST_RESIDENT_ENCODE_FAILURE_INDEX.set(index);
}

#[cfg(all(test, target_os = "macos"))]
fn test_resident_encode_failure_index() -> Option<usize> {
    TEST_RESIDENT_ENCODE_FAILURE_INDEX.get()
}

#[cfg(target_os = "macos")]
fn validate_lossless_roundtrip_on_metal_tile_with_session(
    tile: MetalLosslessEncodeTile<'_>,
    codestream: &[u8],
    session: &crate::MetalBackendSession,
) -> Result<(), crate::Error> {
    let mut decoder = crate::J2kDecoder::new(codestream)?;
    let surface = decoder.decode_to_device_with_session(tile.format, session)?;
    if surface.dimensions() != (tile.output_width, tile.output_height) {
        return Err(crate::Error::MetalKernel {
            message: format!(
                "J2K Metal resident validation geometry mismatch: expected {}x{}, got {}x{}",
                tile.output_width,
                tile.output_height,
                surface.dimensions().0,
                surface.dimensions().1
            ),
        });
    }
    if surface.pixel_format() != tile.format {
        return Err(crate::Error::MetalKernel {
            message: format!(
                "J2K Metal resident validation format mismatch: expected {:?}, got {:?}",
                tile.format,
                surface.pixel_format()
            ),
        });
    }
    let expected_pitch = tile.output_width as usize * tile.format.bytes_per_pixel();
    if surface.pitch_bytes() != expected_pitch || tile.pitch_bytes != expected_pitch {
        return Err(crate::Error::MetalKernel {
            message: "J2K Metal resident validation requires contiguous source and decoded rows"
                .to_string(),
        });
    }
    let byte_len = expected_pitch
        .checked_mul(tile.output_height as usize)
        .ok_or_else(|| crate::Error::MetalKernel {
            message: "J2K Metal resident validation byte length overflow".to_string(),
        })?;
    let (decoded_buffer, decoded_offset) =
        surface
            .metal_buffer()
            .ok_or(crate::Error::UnsupportedMetalRequest {
                reason: "J2K Metal resident validation decode did not return a Metal buffer",
            })?;
    compute::validate_metal_buffers_match(
        tile.buffer,
        tile.byte_offset,
        decoded_buffer,
        decoded_offset,
        byte_len,
        session,
    )
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn validate_lossless_roundtrip_on_metal_region_with_session(
    source: MetalLosslessEncodeTile<'_>,
    output_width: u32,
    output_height: u32,
    bytes_per_pixel: usize,
    codestream: &[u8],
    session: &crate::MetalBackendSession,
) -> Result<(), crate::Error> {
    let staged_buffer = compute::copy_interleaved_padded_to_shared_buffer(
        source.buffer,
        source.byte_offset,
        source.width,
        source.height,
        source.pitch_bytes,
        output_width,
        output_height,
        bytes_per_pixel,
        session,
    )?;
    let staged_tile = MetalLosslessEncodeTile {
        buffer: &staged_buffer,
        byte_offset: 0,
        width: output_width,
        height: output_height,
        pitch_bytes: output_width as usize * bytes_per_pixel,
        output_width,
        output_height,
        format: source.format,
    };
    validate_lossless_roundtrip_on_metal_tile_with_session(staged_tile, codestream, session)
}

#[cfg(target_os = "macos")]
fn should_try_resident_lossless_ht_cpu_packetization(
    tile: MetalLosslessEncodeTile<'_>,
    options: J2kLosslessEncodeOptions,
    staging: MetalEncodeInputStaging,
) -> bool {
    options.backend == EncodeBackendPreference::Auto
        && options.block_coding_mode == J2kBlockCodingMode::HighThroughput
        && options.reversible_transform == ReversibleTransform::Rct53
        && matches!(staging, MetalEncodeInputStaging::AlreadyPaddedContiguous)
        && tile.format == PixelFormat::Rgb8
}

#[cfg(target_os = "macos")]
fn encode_cpu_codestream_from_prepacketized_ht_tile(
    tile_body: ResidentHybridHtTileBody,
    tile: MetalLosslessEncodeTile<'_>,
) -> Result<EncodedJ2k, crate::Error> {
    let dummy_len = checked_mul_bytes(
        checked_mul_bytes(tile.output_width as usize, tile.output_height as usize),
        tile_body.bytes_per_pixel,
    );
    let dummy = vec![0u8; dummy_len];
    let samples = J2kLosslessSamples::new(
        &dummy,
        tile.output_width,
        tile.output_height,
        tile_body.components,
        tile_body.bit_depth,
        false,
    )
    .map_err(crate::Error::Decode)?;
    let mut wrapper = PrepacketizedHtj2kTileAccelerator {
        tile_data: Some(tile_body.tile_data),
    };
    let native_options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: tile_body.num_decomposition_levels,
        code_block_width_exp: tile_body.code_block_width_exp,
        code_block_height_exp: tile_body.code_block_height_exp,
        guard_bits: tile_body.guard_bits,
        use_ht_block_coding: true,
        progression_order: tile_body.progression_order,
        write_tlm: tile_body.write_tlm,
        use_mct: tile_body.used_fused_rct,
        validate_high_throughput_codestream: false,
        ..EncodeOptions::default()
    };
    let codestream = signinum_j2k_native::encode_with_accelerator(
        samples.data,
        samples.width,
        samples.height,
        samples.components,
        samples.bit_depth,
        samples.signed,
        &native_options,
        &mut wrapper,
    )
    .map_err(|err| {
        crate::Error::Decode(signinum_j2k::J2kError::Backend(format!(
            "JPEG 2000 lossless encode failed: {err}"
        )))
    })?;
    Ok(EncodedJ2k {
        codestream,
        backend: BackendKind::Cpu,
        dispatch_report: signinum_j2k::J2kEncodeDispatchReport::default(),
        width: samples.width,
        height: samples.height,
        components: samples.components,
        bit_depth: samples.bit_depth,
        signed: samples.signed,
    })
}

#[cfg(target_os = "macos")]
fn try_encode_lossless_tile_resident_ht_cpu_packetization_with_report(
    tile: MetalLosslessEncodeTile<'_>,
    options: J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
    staging: MetalEncodeInputStaging,
) -> Result<Option<MetalLosslessEncodeOutcome>, crate::Error> {
    let encode_started = Instant::now();
    let Some(tile_body) = encode_resident_ht_tile_body_with_cpu_packetization(
        tile,
        options,
        session,
        staging,
        RESIDENT_CLASSIC_CODE_BLOCK_EDGE,
        RESIDENT_CLASSIC_CODE_BLOCK_EDGE,
    )?
    else {
        return Ok(None);
    };
    let encoded = encode_cpu_codestream_from_prepacketized_ht_tile(tile_body, tile)?;
    let encode_duration = encode_started.elapsed();
    let validation_duration = if options.validation == J2kEncodeValidation::CpuRoundTrip {
        let validation_started = Instant::now();
        validate_lossless_roundtrip_on_metal_tile_with_session(
            tile,
            encoded.codestream.as_slice(),
            session,
        )?;
        validation_started.elapsed()
    } else {
        Duration::ZERO
    };

    Ok(Some(MetalLosslessEncodeOutcome {
        encoded,
        input_copy_used: false,
        resident: MetalLosslessEncodeResidency {
            coefficient_prep_used: true,
            packetization_used: false,
            codestream_assembly_used: false,
        },
        input_copy_duration: Duration::ZERO,
        encode_duration,
        gpu_duration: None,
        validation_duration,
        host_readback_duration: Duration::ZERO,
    }))
}

#[cfg(target_os = "macos")]
fn try_encode_lossless_tile_device_resident_with_report(
    tile: MetalLosslessEncodeTile<'_>,
    options: J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
    staging: MetalEncodeInputStaging,
) -> Result<Option<MetalLosslessEncodeOutcome>, crate::Error> {
    let Some(outcome) = try_encode_lossless_tile_device_resident_to_metal_buffer_with_report(
        tile, options, session, staging,
    )?
    else {
        return Ok(None);
    };
    host_outcome_from_buffer_outcome(outcome).map(Some)
}

#[cfg(target_os = "macos")]
fn try_encode_lossless_tile_device_resident_to_metal_buffer_with_report(
    tile: MetalLosslessEncodeTile<'_>,
    options: J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
    staging: MetalEncodeInputStaging,
) -> Result<Option<MetalLosslessBufferEncodeOutcome>, crate::Error> {
    if options.backend == EncodeBackendPreference::CpuOnly {
        return Ok(None);
    }
    let (components, bit_depth) = lossless_sample_shape(tile.format)?;
    let bytes_per_pixel = tile.format.bytes_per_pixel();
    let bytes_per_sample =
        u8::try_from(tile.format.bytes_per_sample()).map_err(|_| crate::Error::MetalKernel {
            message: "J2K Metal resident encode bytes per sample exceeds u8".to_string(),
        })?;
    if matches!(staging, MetalEncodeInputStaging::AlreadyPaddedContiguous) {
        validate_padded_contiguous_metal_encode_tile(tile, bytes_per_pixel)?;
    }
    let Some(plan) = lossless_device_encode_plan(
        tile.output_width,
        tile.output_height,
        components,
        bit_depth,
        options,
        RESIDENT_CLASSIC_CODE_BLOCK_EDGE,
        RESIDENT_CLASSIC_CODE_BLOCK_EDGE,
    )?
    else {
        return Ok(None);
    };

    let encode_started = Instant::now();
    let coefficient_count = lossless_device_coefficient_count(&plan.code_blocks)?;
    let prepared = compute::prepare_lossless_device_code_blocks(
        session,
        compute::J2kLosslessDevicePrepareJob {
            input: tile.buffer,
            input_byte_offset: tile.byte_offset,
            input_width: tile.width,
            input_height: tile.height,
            input_pitch_bytes: tile.pitch_bytes,
            output_width: tile.output_width,
            output_height: tile.output_height,
            components,
            bytes_per_sample,
            bit_depth,
            num_decomposition_levels: plan.num_decomposition_levels,
            coefficient_count,
        },
        plan.code_blocks.clone(),
    )?;
    let packetization_resolutions =
        resident_packetization_resolutions_from_lossless_device_plan(&plan)?;
    let packet_descriptors = packet_descriptors_for_lossless_device_order(
        plan.resolutions.len(),
        plan.components,
        plan.progression_order,
    )?;
    let packetization_job = compute::J2kResidentPacketizationEncodeJob {
        resolution_count: u32::try_from(plan.resolutions.len()).map_err(|_| {
            crate::Error::MetalKernel {
                message: "J2K Metal resident encode resolution count exceeds u32".to_string(),
            }
        })?,
        num_layers: 1,
        num_components: plan.components,
        code_block_count: u32::try_from(plan.code_blocks.len()).map_err(|_| {
            crate::Error::MetalKernel {
                message: "J2K Metal resident encode code-block count exceeds u32".to_string(),
            }
        })?,
        packet_descriptors: &packet_descriptors,
        resolutions: &packetization_resolutions,
    };
    let assembly_job = compute::J2kLosslessCodestreamAssemblyJob {
        width: tile.output_width,
        height: tile.output_height,
        num_components: plan.components,
        bit_depth: plan.bit_depth,
        signed: false,
        num_decomposition_levels: plan.num_decomposition_levels,
        use_mct: plan.use_mct,
        guard_bits: plan.guard_bits,
        code_block_width_exp: plan.code_block_width_exp,
        code_block_height_exp: plan.code_block_height_exp,
        progression_order: plan.progression_order,
        write_tlm: plan.write_tlm,
        block_coding_mode: match plan.block_coding_mode {
            J2kBlockCodingMode::Classic => compute::J2kLosslessCodestreamBlockCodingMode::Classic,
            J2kBlockCodingMode::HighThroughput => {
                compute::J2kLosslessCodestreamBlockCodingMode::HighThroughput
            }
        },
    };
    let codestream = match plan.block_coding_mode {
        J2kBlockCodingMode::Classic => {
            let resident_tier1 =
                compute::encode_classic_tier1_prepared_device_code_blocks_resident(
                    session, prepared,
                )?;
            compute::encode_lossless_codestream_buffer_from_resident_classic_tier1(
                session,
                &resident_tier1,
                packetization_job,
                assembly_job,
            )?
        }
        J2kBlockCodingMode::HighThroughput => {
            let resident_tier1 =
                compute::encode_ht_prepared_device_code_blocks_resident(session, prepared)?;
            compute::encode_lossless_codestream_buffer_from_resident_ht_tier1(
                session,
                &resident_tier1,
                packetization_job,
                assembly_job,
            )?
        }
    };
    let encode_duration = encode_started.elapsed();

    let encoded = MetalEncodedJ2k {
        codestream_buffer: codestream.buffer,
        byte_offset: codestream.byte_offset,
        byte_len: codestream.byte_len,
        capacity: codestream.capacity,
        width: tile.output_width,
        height: tile.output_height,
        components,
        bit_depth,
        signed: false,
    };

    let validation_duration = if options.validation == J2kEncodeValidation::CpuRoundTrip {
        let validation_started = Instant::now();
        if matches!(staging, MetalEncodeInputStaging::AlreadyPaddedContiguous) {
            validate_lossless_roundtrip_on_metal_tile_with_session(
                tile,
                encoded.codestream_bytes()?,
                session,
            )?;
        } else {
            validate_lossless_roundtrip_on_metal_region_with_session(
                tile,
                tile.output_width,
                tile.output_height,
                bytes_per_pixel,
                encoded.codestream_bytes()?,
                session,
            )?;
        }
        validation_started.elapsed()
    } else {
        Duration::ZERO
    };

    Ok(Some(MetalLosslessBufferEncodeOutcome {
        encoded,
        input_copy_used: false,
        resident: MetalLosslessEncodeResidency {
            coefficient_prep_used: true,
            packetization_used: true,
            codestream_assembly_used: true,
        },
        input_copy_duration: Duration::ZERO,
        encode_duration,
        gpu_duration: codestream.gpu_duration,
        validation_duration,
    }))
}

#[cfg(target_os = "macos")]
fn encode_lossless_tile_to_metal_buffer_with_report(
    tile: MetalLosslessEncodeTile<'_>,
    options: J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
    staging: MetalEncodeInputStaging,
) -> Result<MetalLosslessBufferEncodeOutcome, crate::Error> {
    validate_metal_encode_tile(tile)?;
    lossless_sample_shape(tile.format)?;
    if options.backend == EncodeBackendPreference::CpuOnly {
        return Err(crate::Error::UnsupportedMetalRequest {
            reason: "J2K Metal buffer output encode requires a device backend",
        });
    }
    let bytes_per_pixel = tile.format.bytes_per_pixel();
    if matches!(staging, MetalEncodeInputStaging::AlreadyPaddedContiguous) {
        validate_padded_contiguous_metal_encode_tile(tile, bytes_per_pixel)?;
    }
    if let Some(outcome) = try_encode_lossless_tile_device_resident_to_metal_buffer_with_report(
        tile, options, session, staging,
    )? {
        return Ok(outcome);
    }
    Err(crate::Error::UnsupportedMetalRequest {
        reason: "J2K Metal buffer output encode requires classic padded contiguous Gray/RGB lossless input with at most one DWT level",
    })
}

#[cfg(target_os = "macos")]
fn encode_lossless_tile_with_report(
    tile: MetalLosslessEncodeTile<'_>,
    options: J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
    staging: MetalEncodeInputStaging,
    accelerator: &mut MetalEncodeStageAccelerator,
) -> Result<MetalLosslessEncodeOutcome, crate::Error> {
    validate_metal_encode_tile(tile)?;
    let (components, bit_depth) = lossless_sample_shape(tile.format)?;
    let bytes_per_pixel = tile.format.bytes_per_pixel();
    if let Some(outcome) = try_encode_lossless_tile_resident_ht_cpu_packetization_with_report(
        tile, options, session, staging,
    )? {
        return Ok(outcome);
    }
    if should_try_resident_lossless_host_encode(options) {
        if let Some(outcome) =
            try_encode_lossless_tile_device_resident_with_report(tile, options, session, staging)?
        {
            return Ok(outcome);
        }
    }
    if matches!(staging, MetalEncodeInputStaging::AlreadyPaddedContiguous)
        && options.backend == EncodeBackendPreference::RequireDevice
    {
        return Err(crate::Error::UnsupportedMetalRequest {
            reason: "J2K Metal resident encode requires classic padded contiguous Gray/RGB lossless input with at most one DWT level",
        });
    }
    let mut input_copy_used = false;
    let mut input_copy_duration = Duration::ZERO;
    let mut staged_buffer = None;
    let mut source_byte_offset = tile.byte_offset;
    if matches!(staging, MetalEncodeInputStaging::AlreadyPaddedContiguous) {
        validate_padded_contiguous_metal_encode_tile(tile, bytes_per_pixel)?;
        if tile.buffer.contents().is_null() {
            let copy_started = Instant::now();
            staged_buffer = Some(compute::copy_interleaved_padded_to_shared_buffer(
                tile.buffer,
                tile.byte_offset,
                tile.width,
                tile.height,
                tile.pitch_bytes,
                tile.output_width,
                tile.output_height,
                bytes_per_pixel,
                session,
            )?);
            input_copy_duration = copy_started.elapsed();
            input_copy_used = true;
            source_byte_offset = 0;
        }
    } else {
        let copy_started = Instant::now();
        staged_buffer = Some(compute::copy_interleaved_padded_to_shared_buffer(
            tile.buffer,
            tile.byte_offset,
            tile.width,
            tile.height,
            tile.pitch_bytes,
            tile.output_width,
            tile.output_height,
            bytes_per_pixel,
            session,
        )?);
        input_copy_duration = copy_started.elapsed();
        input_copy_used = true;
        source_byte_offset = 0;
    }
    let buffer = staged_buffer.as_ref().unwrap_or(tile.buffer);
    let len = tile.output_width as usize * tile.output_height as usize * bytes_per_pixel;
    let ptr = buffer.contents().cast::<u8>();
    if ptr.is_null() {
        return Err(crate::Error::UnsupportedMetalRequest {
            reason: "J2K Metal encode input buffer is not host-visible",
        });
    }
    let data = unsafe { core::slice::from_raw_parts(ptr.add(source_byte_offset), len) };
    let samples = J2kLosslessSamples::new(
        data,
        tile.output_width,
        tile.output_height,
        components,
        bit_depth,
        false,
    )
    .map_err(crate::Error::Decode)?;

    let encode_options = host_output_encode_options(options);
    let encode_started = Instant::now();
    let encoded = signinum_j2k::encode_j2k_lossless_with_accelerator(
        samples,
        &encode_options,
        BackendKind::Metal,
        accelerator,
    )
    .map_err(crate::Error::Decode)?;
    let encode_duration = encode_started.elapsed();
    let validation_duration = if options.validation == J2kEncodeValidation::CpuRoundTrip {
        let validation_started = Instant::now();
        validate_lossless_roundtrip_on_metal_with_session(samples, &encoded.codestream, session)?;
        validation_started.elapsed()
    } else {
        Duration::ZERO
    };
    Ok(MetalLosslessEncodeOutcome {
        encoded,
        input_copy_used,
        resident: MetalLosslessEncodeResidency {
            coefficient_prep_used: false,
            packetization_used: false,
            codestream_assembly_used: false,
        },
        input_copy_duration,
        encode_duration,
        gpu_duration: None,
        validation_duration,
        host_readback_duration: Duration::ZERO,
    })
}

#[cfg(not(target_os = "macos"))]
/// Return `Error::MetalUnavailable` for single-tile Metal encode on non-macOS.
pub fn encode_lossless_from_metal_buffer(
    tile: MetalLosslessEncodeTile<'_>,
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<EncodedJ2k, crate::Error> {
    submit_lossless_from_metal_buffer(tile, options, session)?.wait()
}

#[cfg(not(target_os = "macos"))]
/// Return `Error::MetalUnavailable` for Metal-buffer output on non-macOS.
pub fn encode_lossless_from_metal_buffer_to_metal(
    tile: MetalLosslessEncodeTile<'_>,
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<MetalEncodedJ2k, crate::Error> {
    let _ = (tile, options, session);
    Err(crate::Error::MetalUnavailable)
}

#[cfg(not(target_os = "macos"))]
/// Return `Error::MetalUnavailable` for submitted Metal encode on non-macOS.
pub fn submit_lossless_from_metal_buffer(
    tile: MetalLosslessEncodeTile<'_>,
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<SubmittedJ2kLosslessMetalEncode, crate::Error> {
    let _ = (tile, options, session);
    Err(crate::Error::MetalUnavailable)
}

#[cfg(not(target_os = "macos"))]
/// Return `Error::MetalUnavailable` for reported Metal encode on non-macOS.
pub fn encode_lossless_from_metal_buffer_with_report(
    tile: MetalLosslessEncodeTile<'_>,
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<MetalLosslessEncodeOutcome, crate::Error> {
    let _ = (tile, options, session);
    Err(crate::Error::MetalUnavailable)
}

#[cfg(not(target_os = "macos"))]
/// Return `Error::MetalUnavailable` for reported Metal-buffer output on non-macOS.
pub fn encode_lossless_from_metal_buffer_to_metal_with_report(
    tile: MetalLosslessEncodeTile<'_>,
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<MetalLosslessBufferEncodeOutcome, crate::Error> {
    let _ = (tile, options, session);
    Err(crate::Error::MetalUnavailable)
}

#[cfg(not(target_os = "macos"))]
/// Return `Error::MetalUnavailable` for padded single-tile encode on non-macOS.
pub fn encode_lossless_from_padded_metal_buffer(
    tile: MetalLosslessEncodeTile<'_>,
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<EncodedJ2k, crate::Error> {
    submit_lossless_from_padded_metal_buffer(tile, options, session)?.wait()
}

#[cfg(not(target_os = "macos"))]
/// Return `Error::MetalUnavailable` for padded Metal-buffer output on non-macOS.
pub fn encode_lossless_from_padded_metal_buffer_to_metal(
    tile: MetalLosslessEncodeTile<'_>,
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<MetalEncodedJ2k, crate::Error> {
    let _ = (tile, options, session);
    Err(crate::Error::MetalUnavailable)
}

#[cfg(not(target_os = "macos"))]
/// Return `Error::MetalUnavailable` for submitted padded encode on non-macOS.
pub fn submit_lossless_from_padded_metal_buffer(
    tile: MetalLosslessEncodeTile<'_>,
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<SubmittedJ2kLosslessMetalEncode, crate::Error> {
    let _ = (tile, options, session);
    Err(crate::Error::MetalUnavailable)
}

#[cfg(not(target_os = "macos"))]
/// Return `Error::MetalUnavailable` for reported padded encode on non-macOS.
pub fn encode_lossless_from_padded_metal_buffer_with_report(
    tile: MetalLosslessEncodeTile<'_>,
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<MetalLosslessEncodeOutcome, crate::Error> {
    let _ = (tile, options, session);
    Err(crate::Error::MetalUnavailable)
}

#[cfg(not(target_os = "macos"))]
/// Return `Error::MetalUnavailable` for reported padded Metal-buffer output on non-macOS.
pub fn encode_lossless_from_padded_metal_buffer_to_metal_with_report(
    tile: MetalLosslessEncodeTile<'_>,
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<MetalLosslessBufferEncodeOutcome, crate::Error> {
    let _ = (tile, options, session);
    Err(crate::Error::MetalUnavailable)
}

#[cfg(not(target_os = "macos"))]
/// Return `Error::MetalUnavailable` for multi-tile Metal encode on non-macOS.
pub fn encode_lossless_from_metal_buffers(
    tiles: &[MetalLosslessEncodeTile<'_>],
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<Vec<EncodedJ2k>, crate::Error> {
    submit_lossless_from_metal_buffers(tiles, options, session)?.wait()
}

#[cfg(not(target_os = "macos"))]
/// Return `Error::MetalUnavailable` for multi-tile Metal-buffer output on non-macOS.
pub fn encode_lossless_from_metal_buffers_to_metal(
    tiles: &[MetalLosslessEncodeTile<'_>],
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<Vec<MetalEncodedJ2k>, crate::Error> {
    let _ = (tiles, options, session);
    Err(crate::Error::MetalUnavailable)
}

#[cfg(not(target_os = "macos"))]
/// Return `Error::MetalUnavailable` for submitted multi-tile encode on non-macOS.
pub fn submit_lossless_from_metal_buffers(
    tiles: &[MetalLosslessEncodeTile<'_>],
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<SubmittedJ2kLosslessMetalEncodeBatch, crate::Error> {
    let _ = (tiles, options, session);
    Err(crate::Error::MetalUnavailable)
}

#[cfg(not(target_os = "macos"))]
/// Return `Error::MetalUnavailable` for configured multi-tile encode on non-macOS.
pub fn submit_lossless_from_metal_buffers_with_config(
    tiles: &[MetalLosslessEncodeTile<'_>],
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
    config: MetalLosslessEncodeConfig,
) -> Result<SubmittedJ2kLosslessMetalEncodeBatch, crate::Error> {
    let _ = (tiles, options, session, config);
    Err(crate::Error::MetalUnavailable)
}

#[cfg(not(target_os = "macos"))]
/// Return `Error::MetalUnavailable` for reported multi-tile encode on non-macOS.
pub fn encode_lossless_from_metal_buffers_with_report(
    tiles: &[MetalLosslessEncodeTile<'_>],
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<Vec<MetalLosslessEncodeOutcome>, crate::Error> {
    let _ = (tiles, options, session);
    Err(crate::Error::MetalUnavailable)
}

#[cfg(not(target_os = "macos"))]
/// Return `Error::MetalUnavailable` for reported multi-tile Metal-buffer output on non-macOS.
pub fn encode_lossless_from_metal_buffers_to_metal_with_report(
    tiles: &[MetalLosslessEncodeTile<'_>],
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<Vec<MetalLosslessBufferEncodeOutcome>, crate::Error> {
    let _ = (tiles, options, session);
    Err(crate::Error::MetalUnavailable)
}

#[cfg(not(target_os = "macos"))]
/// Return `Error::MetalUnavailable` for Metal-buffer batch output on non-macOS.
pub fn encode_lossless_from_metal_buffers_to_metal_batch(
    tiles: &[MetalLosslessEncodeTile<'_>],
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
    config: MetalLosslessEncodeConfig,
) -> Result<MetalLosslessBufferEncodeBatchOutcome, crate::Error> {
    let _ = (tiles, options, session, config);
    Err(crate::Error::MetalUnavailable)
}

#[cfg(not(target_os = "macos"))]
/// Return `Error::MetalUnavailable` for submitted Metal-buffer batch output on non-macOS.
pub fn submit_lossless_from_metal_buffers_to_metal_batch(
    tiles: &[MetalLosslessEncodeTile<'_>],
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
    config: MetalLosslessEncodeConfig,
) -> Result<SubmittedJ2kLosslessMetalBufferEncodeBatch, crate::Error> {
    let _ = (tiles, options, session, config);
    Err(crate::Error::MetalUnavailable)
}

#[cfg(not(target_os = "macos"))]
/// Return `Error::MetalUnavailable` for padded multi-tile encode on non-macOS.
pub fn encode_lossless_from_padded_metal_buffers(
    tiles: &[MetalLosslessEncodeTile<'_>],
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<Vec<EncodedJ2k>, crate::Error> {
    submit_lossless_from_padded_metal_buffers(tiles, options, session)?.wait()
}

#[cfg(not(target_os = "macos"))]
/// Return `Error::MetalUnavailable` for padded multi-tile Metal-buffer output on non-macOS.
pub fn encode_lossless_from_padded_metal_buffers_to_metal(
    tiles: &[MetalLosslessEncodeTile<'_>],
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<Vec<MetalEncodedJ2k>, crate::Error> {
    let _ = (tiles, options, session);
    Err(crate::Error::MetalUnavailable)
}

#[cfg(not(target_os = "macos"))]
/// Return `Error::MetalUnavailable` for submitted padded multi-tile encode on non-macOS.
pub fn submit_lossless_from_padded_metal_buffers(
    tiles: &[MetalLosslessEncodeTile<'_>],
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<SubmittedJ2kLosslessMetalEncodeBatch, crate::Error> {
    let _ = (tiles, options, session);
    Err(crate::Error::MetalUnavailable)
}

#[cfg(not(target_os = "macos"))]
/// Return `Error::MetalUnavailable` for configured padded multi-tile encode on non-macOS.
pub fn submit_lossless_from_padded_metal_buffers_with_config(
    tiles: &[MetalLosslessEncodeTile<'_>],
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
    config: MetalLosslessEncodeConfig,
) -> Result<SubmittedJ2kLosslessMetalEncodeBatch, crate::Error> {
    let _ = (tiles, options, session, config);
    Err(crate::Error::MetalUnavailable)
}

#[cfg(not(target_os = "macos"))]
/// Return `Error::MetalUnavailable` for reported padded multi-tile encode on non-macOS.
pub fn encode_lossless_from_padded_metal_buffers_with_report(
    tiles: &[MetalLosslessEncodeTile<'_>],
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<Vec<MetalLosslessEncodeOutcome>, crate::Error> {
    let _ = (tiles, options, session);
    Err(crate::Error::MetalUnavailable)
}

#[cfg(not(target_os = "macos"))]
/// Return `Error::MetalUnavailable` for padded Metal-buffer batch output on non-macOS.
pub fn encode_lossless_from_padded_metal_buffers_to_metal_batch(
    tiles: &[MetalLosslessEncodeTile<'_>],
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
    config: MetalLosslessEncodeConfig,
) -> Result<MetalLosslessBufferEncodeBatchOutcome, crate::Error> {
    let _ = (tiles, options, session, config);
    Err(crate::Error::MetalUnavailable)
}

#[cfg(not(target_os = "macos"))]
/// Return `Error::MetalUnavailable` for submitted padded Metal-buffer batch output on non-macOS.
pub fn submit_lossless_from_padded_metal_buffers_to_metal_batch(
    tiles: &[MetalLosslessEncodeTile<'_>],
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
    config: MetalLosslessEncodeConfig,
) -> Result<SubmittedJ2kLosslessMetalBufferEncodeBatch, crate::Error> {
    let _ = (tiles, options, session, config);
    Err(crate::Error::MetalUnavailable)
}

#[cfg(not(target_os = "macos"))]
/// Return `Error::MetalUnavailable` for reported padded Metal-buffer output on non-macOS.
pub fn encode_lossless_from_padded_metal_buffers_to_metal_with_report(
    tiles: &[MetalLosslessEncodeTile<'_>],
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<Vec<MetalLosslessBufferEncodeOutcome>, crate::Error> {
    let _ = (tiles, options, session);
    Err(crate::Error::MetalUnavailable)
}

#[cfg(target_os = "macos")]
/// Validate a lossless codestream by decoding it through the default Metal session.
pub fn validate_lossless_roundtrip_on_metal(
    samples: J2kLosslessSamples<'_>,
    codestream: &[u8],
) -> Result<(), crate::Error> {
    let session = crate::MetalBackendSession::system_default()?;
    validate_lossless_roundtrip_on_metal_with_session(samples, codestream, &session)
}

#[cfg(not(target_os = "macos"))]
/// Return `Error::MetalUnavailable` for Metal roundtrip validation on non-macOS.
pub fn validate_lossless_roundtrip_on_metal(
    samples: J2kLosslessSamples<'_>,
    codestream: &[u8],
) -> Result<(), crate::Error> {
    let _ = (samples, codestream);
    Err(crate::Error::MetalUnavailable)
}

#[cfg(target_os = "macos")]
/// Validate a lossless codestream by decoding it through a provided Metal session.
pub fn validate_lossless_roundtrip_on_metal_with_session(
    samples: J2kLosslessSamples<'_>,
    codestream: &[u8],
    session: &crate::MetalBackendSession,
) -> Result<(), crate::Error> {
    let fmt = validation_pixel_format(samples)?;
    let mut decoder = crate::J2kDecoder::new(codestream)?;
    let surface = decoder.decode_to_device_with_session(fmt, session)?;

    if surface.dimensions() != (samples.width, samples.height) {
        return Err(crate::Error::MetalKernel {
            message: format!(
                "J2K Metal validation geometry mismatch: expected {}x{}, got {}x{}",
                samples.width,
                samples.height,
                surface.dimensions().0,
                surface.dimensions().1
            ),
        });
    }
    if surface.pixel_format() != fmt {
        return Err(crate::Error::MetalKernel {
            message: format!(
                "J2K Metal validation format mismatch: expected {:?}, got {:?}",
                fmt,
                surface.pixel_format()
            ),
        });
    }
    let expected_pitch = samples.width as usize * fmt.bytes_per_pixel();
    if surface.pitch_bytes() != expected_pitch {
        return Err(crate::Error::MetalKernel {
            message: format!(
                "J2K Metal validation pitch mismatch: expected {expected_pitch}, got {}",
                surface.pitch_bytes()
            ),
        });
    }
    if surface.byte_len() != samples.data.len() {
        return Err(crate::Error::MetalKernel {
            message: format!(
                "J2K Metal validation length mismatch: expected {} bytes, got {} bytes",
                samples.data.len(),
                surface.byte_len()
            ),
        });
    }

    let (buffer, byte_offset) =
        surface
            .metal_buffer()
            .ok_or(crate::Error::UnsupportedMetalRequest {
                reason: "J2K Metal validation decode did not return a Metal buffer",
            })?;
    compute::validate_metal_buffer_matches_bytes(samples.data, buffer, byte_offset, session)
}

#[cfg(not(target_os = "macos"))]
/// Return `Error::MetalUnavailable` for session validation on non-macOS.
pub fn validate_lossless_roundtrip_on_metal_with_session(
    samples: J2kLosslessSamples<'_>,
    codestream: &[u8],
    session: &crate::MetalBackendSession,
) -> Result<(), crate::Error> {
    let _ = (samples, codestream, session);
    Err(crate::Error::MetalUnavailable)
}

#[cfg(target_os = "macos")]
fn validation_pixel_format(samples: J2kLosslessSamples<'_>) -> Result<PixelFormat, crate::Error> {
    match (samples.components, samples.bit_depth) {
        (1, 1..=8) => Ok(PixelFormat::Gray8),
        (3, 1..=8) => Ok(PixelFormat::Rgb8),
        (1, 9..=16) => Ok(PixelFormat::Gray16),
        (3, 9..=16) => Ok(PixelFormat::Rgb16),
        _ => Err(crate::Error::UnsupportedMetalRequest {
            reason: "J2K Metal validation supports only grayscale or RGB samples up to 16 bits",
        }),
    }
}

#[cfg(target_os = "macos")]
fn lossless_sample_shape(format: PixelFormat) -> Result<(u8, u8), crate::Error> {
    match format {
        PixelFormat::Gray8 => Ok((1, 8)),
        PixelFormat::Rgb8 => Ok((3, 8)),
        PixelFormat::Gray16 => Ok((1, 16)),
        PixelFormat::Rgb16 => Ok((3, 16)),
        PixelFormat::Rgba8 | PixelFormat::Rgba16 => Err(crate::Error::UnsupportedMetalRequest {
            reason: "J2K Metal encode from RGBA tiles requires explicit alpha handling",
        }),
        _ => Err(crate::Error::UnsupportedMetalRequest {
            reason: "J2K Metal encode received an unknown pixel format",
        }),
    }
}

#[cfg(target_os = "macos")]
fn validate_metal_encode_tile(tile: MetalLosslessEncodeTile<'_>) -> Result<(), crate::Error> {
    if tile.width == 0 || tile.height == 0 || tile.output_width == 0 || tile.output_height == 0 {
        return Err(crate::Error::MetalKernel {
            message: "J2K Metal encode tile dimensions must be nonzero".to_string(),
        });
    }
    if tile.width > tile.output_width || tile.height > tile.output_height {
        return Err(crate::Error::MetalKernel {
            message: "J2K Metal encode input tile exceeds output tile dimensions".to_string(),
        });
    }
    let row_bytes = tile
        .width
        .checked_mul(tile.format.bytes_per_pixel() as u32)
        .ok_or_else(|| crate::Error::MetalKernel {
            message: "J2K Metal encode row byte count overflow".to_string(),
        })? as usize;
    if tile.pitch_bytes < row_bytes {
        return Err(crate::Error::MetalKernel {
            message: "J2K Metal encode tile pitch is shorter than one row".to_string(),
        });
    }
    let required_end = tile
        .byte_offset
        .checked_add(
            tile.pitch_bytes
                .checked_mul(tile.height.saturating_sub(1) as usize)
                .and_then(|prefix| prefix.checked_add(row_bytes))
                .ok_or_else(|| crate::Error::MetalKernel {
                    message: "J2K Metal encode input byte range overflow".to_string(),
                })?,
        )
        .ok_or_else(|| crate::Error::MetalKernel {
            message: "J2K Metal encode input byte range overflow".to_string(),
        })?;
    let buffer_len =
        usize::try_from(tile.buffer.length()).map_err(|_| crate::Error::MetalKernel {
            message: "J2K Metal encode buffer length exceeds usize".to_string(),
        })?;
    if required_end > buffer_len {
        return Err(crate::Error::MetalKernel {
            message: format!(
                "J2K Metal encode input byte range exceeds buffer length: need {required_end}, buffer has {buffer_len}"
            ),
        });
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn validate_padded_contiguous_metal_encode_tile(
    tile: MetalLosslessEncodeTile<'_>,
    bytes_per_pixel: usize,
) -> Result<(), crate::Error> {
    if tile.width != tile.output_width || tile.height != tile.output_height {
        return Err(crate::Error::MetalKernel {
            message:
                "J2K Metal no-copy encode requires input dimensions to match output dimensions"
                    .to_string(),
        });
    }
    let expected_pitch = (tile.output_width as usize)
        .checked_mul(bytes_per_pixel)
        .ok_or_else(|| crate::Error::MetalKernel {
            message: "J2K Metal no-copy encode pitch overflow".to_string(),
        })?;
    if tile.pitch_bytes != expected_pitch {
        return Err(crate::Error::MetalKernel {
            message: format!(
                "J2K Metal no-copy encode requires contiguous rows: expected pitch {expected_pitch}, got {}",
                tile.pitch_bytes
            ),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::MetalEncodeStageAccelerator;
    #[cfg(target_os = "macos")]
    use crate::compute;
    #[cfg(target_os = "macos")]
    use metal::foreign_types::ForeignType;
    #[cfg(target_os = "macos")]
    use metal::Buffer;
    use signinum_core::DeviceSubmission;
    #[cfg(target_os = "macos")]
    use signinum_core::{BackendKind, PixelFormat};
    use signinum_j2k::{
        encode_j2k_lossless_with_accelerator, EncodeBackendPreference, EncodedJ2k,
        J2kEncodeStageAccelerator, J2kForwardDwt53Job, J2kForwardRctJob, J2kLosslessEncodeOptions,
        J2kLosslessSamples,
    };
    #[cfg(target_os = "macos")]
    use signinum_j2k::{
        encode_j2k_lossy_with_accelerator, J2kBlockCodingMode, J2kEncodeValidation,
        J2kLossyEncodeOptions, J2kLossySamples, J2kProgressionOrder,
    };
    #[cfg(target_os = "macos")]
    use signinum_j2k_native::{forward_dwt53_reference, J2kCodeBlockStyle};
    use signinum_j2k_native::{DecodeSettings, Image};
    use std::time::Duration;

    #[cfg(target_os = "macos")]
    macro_rules! lossless_options {
        ($($field:ident: $value:expr),+ $(,)?) => {{
            let mut options = J2kLosslessEncodeOptions::default();
            $(options.$field = $value;)+
            options
        }};
    }

    #[cfg(target_os = "macos")]
    fn private_buffer_with_bytes(session: &crate::MetalBackendSession, bytes: &[u8]) -> Buffer {
        let upload = session.device().new_buffer_with_data(
            bytes.as_ptr().cast(),
            bytes.len() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let private = session.device().new_buffer(
            bytes.len() as u64,
            metal::MTLResourceOptions::StorageModePrivate,
        );
        let queue = session.device().new_command_queue();
        let command_buffer = queue.new_command_buffer();
        let blit = command_buffer.new_blit_command_encoder();
        blit.copy_from_buffer(&upload, 0, &private, 0, bytes.len() as u64);
        blit.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        private
    }

    #[cfg(target_os = "macos")]
    fn overwrite_private_buffer_with_bytes(
        session: &crate::MetalBackendSession,
        dst: &Buffer,
        bytes: &[u8],
    ) {
        let upload = session.device().new_buffer_with_data(
            bytes.as_ptr().cast(),
            bytes.len() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let queue = session.device().new_command_queue();
        let command_buffer = queue.new_command_buffer();
        let blit = command_buffer.new_blit_command_encoder();
        blit.copy_from_buffer(&upload, 0, dst, 0, bytes.len() as u64);
        blit.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
    }

    #[cfg(target_os = "macos")]
    fn assert_decoded_bytes_match(actual: &[u8], expected: &[u8]) {
        if actual == expected {
            return;
        }
        let mismatch = actual
            .iter()
            .zip(expected.iter())
            .position(|(actual, expected)| actual != expected)
            .unwrap_or_else(|| actual.len().min(expected.len()));
        let actual_value = actual.get(mismatch).copied();
        let expected_value = expected.get(mismatch).copied();
        panic!(
            "decoded bytes mismatch at byte {mismatch}: actual={actual_value:?} expected={expected_value:?} actual_len={} expected_len={}",
            actual.len(),
            expected.len()
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn inflight_limited_runner_starts_next_item_before_slow_peer_finishes() {
        use std::sync::{Arc, Condvar, Mutex};
        use std::time::Duration;

        #[derive(Default)]
        struct Probe {
            third_item_started: bool,
        }

        let probe = Arc::new((Mutex::new(Probe::default()), Condvar::new()));
        let task_probe = Arc::clone(&probe);

        let outcomes = super::collect_inflight_limited_ordered(vec![0usize, 1, 2], 2, move |_, item| {
            match item {
                0 => Ok(item),
                1 => {
                    let (lock, cvar) = &*task_probe;
                    let state = lock.lock().expect("probe mutex");
                    let (state, _timeout) = cvar
                        .wait_timeout_while(state, Duration::from_millis(250), |state| {
                            !state.third_item_started
                        })
                        .expect("probe wait");
                    if !state.third_item_started {
                        return Err(crate::Error::MetalKernel {
                            message:
                                "runner waited for the whole in-flight chunk before scheduling more work"
                                    .to_string(),
                        });
                    }
                    Ok(item)
                }
                2 => {
                    let (lock, cvar) = &*task_probe;
                    let mut state = lock.lock().expect("probe mutex");
                    state.third_item_started = true;
                    cvar.notify_all();
                    Ok(item)
                }
                _ => unreachable!("unexpected test item"),
            }
        })
        .expect("in-flight runner should slide past a slow peer");

        assert_eq!(outcomes.items, vec![0, 1, 2]);
        assert!(outcomes.max_observed_inflight_items <= 2);
        assert!(outcomes.max_observed_inflight_items > 0);
    }

    #[test]
    fn submitted_lossless_metal_encode_public_api_is_available() {
        fn assert_single_submission<
            S: DeviceSubmission<Output = EncodedJ2k, Error = crate::Error>,
        >() {
        }
        fn assert_batch_submission<
            S: DeviceSubmission<Output = Vec<EncodedJ2k>, Error = crate::Error>,
        >() {
        }
        fn assert_submit_single_fn(
            _submit: for<'tile, 'options, 'session> fn(
                super::MetalLosslessEncodeTile<'tile>,
                &'options J2kLosslessEncodeOptions,
                &'session crate::MetalBackendSession,
            ) -> Result<
                crate::SubmittedJ2kLosslessMetalEncode,
                crate::Error,
            >,
        ) {
        }
        fn assert_submit_batch_fn(
            _submit: for<'slice, 'tile, 'options, 'session> fn(
                &'slice [super::MetalLosslessEncodeTile<'tile>],
                &'options J2kLosslessEncodeOptions,
                &'session crate::MetalBackendSession,
            ) -> Result<
                crate::SubmittedJ2kLosslessMetalEncodeBatch,
                crate::Error,
            >,
        ) {
        }

        assert_single_submission::<crate::SubmittedJ2kLosslessMetalEncode>();
        assert_batch_submission::<crate::SubmittedJ2kLosslessMetalEncodeBatch>();
        assert_submit_single_fn(crate::submit_lossless_from_metal_buffer);
        assert_submit_single_fn(crate::submit_lossless_from_padded_metal_buffer);
        assert_submit_batch_fn(crate::submit_lossless_from_metal_buffers);
        assert_submit_batch_fn(crate::submit_lossless_from_padded_metal_buffers);
    }

    #[test]
    fn submitted_lossless_metal_buffer_encode_public_api_is_available() {
        fn assert_buffer_batch_submission<
            S: DeviceSubmission<
                Output = super::MetalLosslessBufferEncodeBatchOutcome,
                Error = crate::Error,
            >,
        >() {
        }
        fn assert_submit_buffer_batch_fn(
            _submit: for<'slice, 'tile, 'options, 'session> fn(
                &'slice [super::MetalLosslessEncodeTile<'tile>],
                &'options J2kLosslessEncodeOptions,
                &'session crate::MetalBackendSession,
                super::MetalLosslessEncodeConfig,
            ) -> Result<
                crate::SubmittedJ2kLosslessMetalBufferEncodeBatch,
                crate::Error,
            >,
        ) {
        }

        assert_buffer_batch_submission::<crate::SubmittedJ2kLosslessMetalBufferEncodeBatch>();
        assert_submit_buffer_batch_fn(crate::submit_lossless_from_metal_buffers_to_metal_batch);
        assert_submit_buffer_batch_fn(
            crate::submit_lossless_from_padded_metal_buffers_to_metal_batch,
        );
    }

    #[test]
    fn resident_lossless_stage_stats_default_to_zero() {
        let stats = super::MetalLosslessEncodeBatchStats::default();

        assert_eq!(
            stats.stage_stats,
            super::MetalLosslessEncodeStageStats::default()
        );
        assert_eq!(stats.stage_stats.coefficient_prep_duration, Duration::ZERO);
        assert_eq!(stats.stage_stats.deinterleave_rct_duration, Duration::ZERO);
        assert_eq!(stats.stage_stats.dwt53_duration, Duration::ZERO);
        assert_eq!(
            stats.stage_stats.coefficient_extract_duration,
            Duration::ZERO
        );
        assert_eq!(stats.stage_stats.ht_block_encode_duration, Duration::ZERO);
        assert_eq!(
            stats.stage_stats.classic_tier1_setup_duration,
            Duration::ZERO
        );
        assert_eq!(
            stats.stage_stats.classic_block_encode_duration,
            Duration::ZERO
        );
        assert_eq!(
            stats.stage_stats.classic_packet_plan_duration,
            Duration::ZERO
        );
        assert_eq!(
            stats.stage_stats.classic_packet_buffer_setup_duration,
            Duration::ZERO
        );
        assert_eq!(
            stats.stage_stats.classic_command_buffer_commit_duration,
            Duration::ZERO
        );
        assert_eq!(stats.stage_stats.packet_block_prep_duration, Duration::ZERO);
        assert_eq!(stats.stage_stats.packetization_duration, Duration::ZERO);
        assert_eq!(
            stats.stage_stats.packet_payload_copy_gpu_duration,
            Duration::ZERO
        );
        assert_eq!(stats.stage_stats.gpu_elapsed_wall_duration, Duration::ZERO);
        assert_eq!(stats.stage_stats.classic_block_gpu_duration, Duration::ZERO);
        assert_eq!(
            stats.stage_stats.classic_tier1_density_gpu_duration,
            Duration::ZERO
        );
        assert_eq!(
            stats.stage_stats.codestream_assembly_duration,
            Duration::ZERO
        );
        assert_eq!(
            stats.stage_stats.codestream_payload_copy_gpu_duration,
            Duration::ZERO
        );
        assert_eq!(stats.stage_stats.tier1_output_capacity_total, 0);
        assert_eq!(stats.stage_stats.max_tier1_output_capacity, 0);
        assert_eq!(stats.stage_stats.tier1_output_used_bytes_total, 0);
        assert_eq!(stats.stage_stats.max_tier1_output_used_bytes, 0);
        assert_eq!(stats.stage_stats.tier1_coding_pass_count_total, 0);
        assert_eq!(stats.stage_stats.max_tier1_coding_passes_per_block, 0);
        assert_eq!(stats.stage_stats.tier1_arithmetic_pass_count_total, 0);
        assert_eq!(stats.stage_stats.tier1_raw_pass_count_total, 0);
        assert_eq!(stats.stage_stats.tier1_cleanup_pass_count_total, 0);
        assert_eq!(stats.stage_stats.tier1_sigprop_pass_count_total, 0);
        assert_eq!(stats.stage_stats.tier1_magref_pass_count_total, 0);
        assert_eq!(
            stats.stage_stats.tier1_arithmetic_cleanup_pass_count_total,
            0
        );
        assert_eq!(
            stats.stage_stats.tier1_arithmetic_sigprop_pass_count_total,
            0
        );
        assert_eq!(
            stats.stage_stats.tier1_arithmetic_magref_pass_count_total,
            0
        );
        assert_eq!(stats.stage_stats.tier1_raw_sigprop_pass_count_total, 0);
        assert_eq!(stats.stage_stats.tier1_raw_magref_pass_count_total, 0);
        assert_eq!(stats.stage_stats.tier1_full_scan_coeff_visit_count_total, 0);
        assert_eq!(
            stats
                .stage_stats
                .tier1_arithmetic_scan_coeff_visit_count_total,
            0
        );
        assert_eq!(stats.stage_stats.tier1_raw_scan_coeff_visit_count_total, 0);
        assert_eq!(
            stats.stage_stats.tier1_cleanup_scan_coeff_visit_count_total,
            0
        );
        assert_eq!(
            stats.stage_stats.tier1_sigprop_scan_coeff_visit_count_total,
            0
        );
        assert_eq!(
            stats.stage_stats.tier1_magref_scan_coeff_visit_count_total,
            0
        );
        assert_eq!(
            stats.stage_stats.max_tier1_full_scan_coeff_visits_per_block,
            0
        );
        assert_eq!(
            stats.stage_stats.tier1_sigprop_active_candidate_count_total,
            0
        );
        assert_eq!(
            stats.stage_stats.tier1_sigprop_new_significant_count_total,
            0
        );
        assert_eq!(
            stats.stage_stats.tier1_magref_active_candidate_count_total,
            0
        );
        assert_eq!(
            stats
                .stage_stats
                .tier1_arithmetic_sigprop_active_candidate_count_total,
            0
        );
        assert_eq!(
            stats
                .stage_stats
                .tier1_arithmetic_sigprop_new_significant_count_total,
            0
        );
        assert_eq!(
            stats
                .stage_stats
                .tier1_raw_sigprop_active_candidate_count_total,
            0
        );
        assert_eq!(
            stats
                .stage_stats
                .tier1_raw_sigprop_new_significant_count_total,
            0
        );
        assert_eq!(
            stats
                .stage_stats
                .tier1_arithmetic_magref_active_candidate_count_total,
            0
        );
        assert_eq!(
            stats
                .stage_stats
                .tier1_raw_magref_active_candidate_count_total,
            0
        );
        assert_eq!(
            stats.stage_stats.tier1_cleanup_active_candidate_count_total,
            0
        );
        assert_eq!(
            stats.stage_stats.tier1_cleanup_new_significant_count_total,
            0
        );
        assert_eq!(stats.stage_stats.tier1_cleanup_rlc_stripe_count_total, 0);
        assert_eq!(
            stats.stage_stats.tier1_cleanup_rlc_zero_stripe_count_total,
            0
        );
        assert_eq!(stats.stage_stats.tier1_nonzero_block_count_total, 0);
        assert_eq!(stats.stage_stats.tier1_zero_block_count_total, 0);
        assert_eq!(stats.stage_stats.tier1_missing_bitplane_count_total, 0);
        assert_eq!(stats.stage_stats.max_tier1_missing_bitplanes_per_block, 0);
        assert_eq!(stats.stage_stats.tier1_segment_count_total, 0);
        assert_eq!(stats.stage_stats.max_tier1_segments_per_block, 0);
        assert_eq!(stats.stage_stats.packet_payload_copy_job_capacity_total, 0);
        assert_eq!(stats.stage_stats.max_packet_payload_copy_jobs_per_tile, 0);
        assert_eq!(stats.stage_stats.packet_payload_copy_job_count_total, 0);
        assert_eq!(
            stats.stage_stats.max_packet_payload_copy_jobs_used_per_tile,
            0
        );
        assert_eq!(stats.stage_stats.packet_payload_copy_bytes_total, 0);
        assert_eq!(stats.stage_stats.max_packet_payload_copy_bytes_per_tile, 0);
        assert_eq!(
            stats.stage_stats.packet_payload_copy_small_job_count_total,
            0
        );
        assert_eq!(
            stats.stage_stats.packet_payload_copy_medium_job_count_total,
            0
        );
        assert_eq!(
            stats.stage_stats.packet_payload_copy_large_job_count_total,
            0
        );
        assert_eq!(stats.stage_stats.packet_output_capacity_total, 0);
        assert_eq!(stats.stage_stats.max_packet_output_capacity, 0);
        assert_eq!(stats.stage_stats.packet_output_used_bytes_total, 0);
        assert_eq!(stats.stage_stats.max_packet_output_used_bytes, 0);
        assert_eq!(stats.stage_stats.sync_wait_duration, Duration::ZERO);
        assert_eq!(stats.stage_stats.host_readback_duration, Duration::ZERO);
        assert!(!stats.stage_stats.has_timings());
    }

    #[test]
    fn resident_lossless_stage_stats_add_assign_saturates() {
        let mut stats = super::MetalLosslessEncodeStageStats {
            plan_duration: Duration::MAX,
            tile_count: usize::MAX,
            ..super::MetalLosslessEncodeStageStats::default()
        };

        stats.add_assign(super::MetalLosslessEncodeStageStats {
            plan_duration: Duration::from_micros(1),
            prepare_submit_duration: Duration::from_micros(2),
            classic_tier1_setup_duration: Duration::from_micros(4),
            classic_block_encode_duration: Duration::from_micros(5),
            classic_tier1_token_pack_duration: Duration::from_micros(9),
            classic_packet_plan_duration: Duration::from_micros(6),
            classic_packet_buffer_setup_duration: Duration::from_micros(7),
            classic_command_buffer_commit_duration: Duration::from_micros(8),
            packet_payload_copy_job_capacity_total: 11,
            max_packet_payload_copy_jobs_per_tile: 5,
            packet_payload_copy_job_count_total: 13,
            max_packet_payload_copy_jobs_used_per_tile: 6,
            packet_payload_copy_bytes_total: 23,
            max_packet_payload_copy_bytes_per_tile: 12,
            packet_payload_copy_small_job_count_total: 2,
            packet_payload_copy_medium_job_count_total: 3,
            packet_payload_copy_large_job_count_total: 4,
            tier1_output_capacity_total: 17,
            max_tier1_output_capacity: 9,
            tier1_output_used_bytes_total: 19,
            max_tier1_output_used_bytes: 10,
            tier1_segment_capacity_total: 25,
            max_tier1_segment_capacity_per_block: 11,
            tier1_coding_pass_count_total: 31,
            max_tier1_coding_passes_per_block: 8,
            tier1_arithmetic_pass_count_total: 21,
            tier1_raw_pass_count_total: 10,
            tier1_cleanup_pass_count_total: 11,
            tier1_sigprop_pass_count_total: 10,
            tier1_magref_pass_count_total: 10,
            tier1_arithmetic_cleanup_pass_count_total: 11,
            tier1_arithmetic_sigprop_pass_count_total: 6,
            tier1_arithmetic_magref_pass_count_total: 4,
            tier1_raw_sigprop_pass_count_total: 4,
            tier1_raw_magref_pass_count_total: 6,
            tier1_full_scan_coeff_visit_count_total: 31_744,
            tier1_arithmetic_scan_coeff_visit_count_total: 21_504,
            tier1_raw_scan_coeff_visit_count_total: 10_240,
            tier1_cleanup_scan_coeff_visit_count_total: 11_264,
            tier1_sigprop_scan_coeff_visit_count_total: 10_240,
            tier1_magref_scan_coeff_visit_count_total: 10_240,
            max_tier1_full_scan_coeff_visits_per_block: 8_192,
            tier1_sigprop_active_candidate_count_total: 101,
            tier1_sigprop_new_significant_count_total: 37,
            tier1_magref_active_candidate_count_total: 203,
            tier1_arithmetic_sigprop_active_candidate_count_total: 61,
            tier1_arithmetic_sigprop_new_significant_count_total: 23,
            tier1_raw_sigprop_active_candidate_count_total: 40,
            tier1_raw_sigprop_new_significant_count_total: 14,
            tier1_arithmetic_magref_active_candidate_count_total: 123,
            tier1_raw_magref_active_candidate_count_total: 80,
            tier1_cleanup_active_candidate_count_total: 307,
            tier1_cleanup_new_significant_count_total: 41,
            tier1_cleanup_rlc_stripe_count_total: 53,
            tier1_cleanup_rlc_zero_stripe_count_total: 47,
            tier1_token_pack_output_bytes_total: 29,
            max_tier1_token_pack_output_bytes_per_block: 15,
            tier1_nonzero_block_count_total: 2,
            tier1_zero_block_count_total: 1,
            tier1_missing_bitplane_count_total: 5,
            max_tier1_missing_bitplanes_per_block: 4,
            tier1_segment_count_total: 7,
            max_tier1_segments_per_block: 3,
            packet_output_capacity_total: 17,
            max_packet_output_capacity: 9,
            packet_output_used_bytes_total: 19,
            max_packet_output_used_bytes: 10,
            tile_count: 1,
            code_block_count: 3,
            ..super::MetalLosslessEncodeStageStats::default()
        });

        assert_eq!(stats.plan_duration, Duration::MAX);
        assert_eq!(stats.prepare_submit_duration, Duration::from_micros(2));
        assert_eq!(stats.classic_tier1_setup_duration, Duration::from_micros(4));
        assert_eq!(
            stats.classic_block_encode_duration,
            Duration::from_micros(5)
        );
        assert_eq!(
            stats.classic_tier1_token_pack_duration,
            Duration::from_micros(9)
        );
        assert_eq!(stats.classic_packet_plan_duration, Duration::from_micros(6));
        assert_eq!(
            stats.classic_packet_buffer_setup_duration,
            Duration::from_micros(7)
        );
        assert_eq!(
            stats.classic_command_buffer_commit_duration,
            Duration::from_micros(8)
        );
        assert_eq!(stats.tile_count, usize::MAX);
        assert_eq!(stats.code_block_count, 3);
        assert_eq!(stats.packet_payload_copy_job_capacity_total, 11);
        assert_eq!(stats.max_packet_payload_copy_jobs_per_tile, 5);
        assert_eq!(stats.packet_payload_copy_job_count_total, 13);
        assert_eq!(stats.max_packet_payload_copy_jobs_used_per_tile, 6);
        assert_eq!(stats.packet_payload_copy_bytes_total, 23);
        assert_eq!(stats.max_packet_payload_copy_bytes_per_tile, 12);
        assert_eq!(stats.packet_payload_copy_small_job_count_total, 2);
        assert_eq!(stats.packet_payload_copy_medium_job_count_total, 3);
        assert_eq!(stats.packet_payload_copy_large_job_count_total, 4);
        assert_eq!(stats.tier1_output_capacity_total, 17);
        assert_eq!(stats.max_tier1_output_capacity, 9);
        assert_eq!(stats.tier1_output_used_bytes_total, 19);
        assert_eq!(stats.max_tier1_output_used_bytes, 10);
        assert_eq!(stats.tier1_segment_capacity_total, 25);
        assert_eq!(stats.max_tier1_segment_capacity_per_block, 11);
        assert_eq!(stats.tier1_coding_pass_count_total, 31);
        assert_eq!(stats.max_tier1_coding_passes_per_block, 8);
        assert_eq!(stats.tier1_arithmetic_pass_count_total, 21);
        assert_eq!(stats.tier1_raw_pass_count_total, 10);
        assert_eq!(stats.tier1_cleanup_pass_count_total, 11);
        assert_eq!(stats.tier1_sigprop_pass_count_total, 10);
        assert_eq!(stats.tier1_magref_pass_count_total, 10);
        assert_eq!(stats.tier1_arithmetic_cleanup_pass_count_total, 11);
        assert_eq!(stats.tier1_arithmetic_sigprop_pass_count_total, 6);
        assert_eq!(stats.tier1_arithmetic_magref_pass_count_total, 4);
        assert_eq!(stats.tier1_raw_sigprop_pass_count_total, 4);
        assert_eq!(stats.tier1_raw_magref_pass_count_total, 6);
        assert_eq!(stats.tier1_full_scan_coeff_visit_count_total, 31_744);
        assert_eq!(stats.tier1_arithmetic_scan_coeff_visit_count_total, 21_504);
        assert_eq!(stats.tier1_raw_scan_coeff_visit_count_total, 10_240);
        assert_eq!(stats.tier1_cleanup_scan_coeff_visit_count_total, 11_264);
        assert_eq!(stats.tier1_sigprop_scan_coeff_visit_count_total, 10_240);
        assert_eq!(stats.tier1_magref_scan_coeff_visit_count_total, 10_240);
        assert_eq!(stats.max_tier1_full_scan_coeff_visits_per_block, 8_192);
        assert_eq!(stats.tier1_sigprop_active_candidate_count_total, 101);
        assert_eq!(stats.tier1_sigprop_new_significant_count_total, 37);
        assert_eq!(stats.tier1_magref_active_candidate_count_total, 203);
        assert_eq!(
            stats.tier1_arithmetic_sigprop_active_candidate_count_total,
            61
        );
        assert_eq!(
            stats.tier1_arithmetic_sigprop_new_significant_count_total,
            23
        );
        assert_eq!(stats.tier1_raw_sigprop_active_candidate_count_total, 40);
        assert_eq!(stats.tier1_raw_sigprop_new_significant_count_total, 14);
        assert_eq!(
            stats.tier1_arithmetic_magref_active_candidate_count_total,
            123
        );
        assert_eq!(stats.tier1_raw_magref_active_candidate_count_total, 80);
        assert_eq!(stats.tier1_cleanup_active_candidate_count_total, 307);
        assert_eq!(stats.tier1_cleanup_new_significant_count_total, 41);
        assert_eq!(stats.tier1_cleanup_rlc_stripe_count_total, 53);
        assert_eq!(stats.tier1_cleanup_rlc_zero_stripe_count_total, 47);
        assert_eq!(stats.tier1_token_pack_output_bytes_total, 29);
        assert_eq!(stats.max_tier1_token_pack_output_bytes_per_block, 15);
        assert_eq!(stats.tier1_nonzero_block_count_total, 2);
        assert_eq!(stats.tier1_zero_block_count_total, 1);
        assert_eq!(stats.tier1_missing_bitplane_count_total, 5);
        assert_eq!(stats.max_tier1_missing_bitplanes_per_block, 4);
        assert_eq!(stats.tier1_segment_count_total, 7);
        assert_eq!(stats.max_tier1_segments_per_block, 3);
        assert_eq!(stats.packet_output_capacity_total, 17);
        assert_eq!(stats.max_packet_output_capacity, 9);
        assert_eq!(stats.packet_output_used_bytes_total, 19);
        assert_eq!(stats.max_packet_output_used_bytes, 10);
    }

    #[test]
    fn resident_lossless_stage_stats_accumulates_split_gpu_durations() {
        let mut stats = super::MetalLosslessEncodeStageStats {
            ht_block_gpu_duration: Duration::from_micros(2),
            ..super::MetalLosslessEncodeStageStats::default()
        };

        stats.add_assign(super::MetalLosslessEncodeStageStats {
            coefficient_prep_gpu_duration: Duration::from_micros(23),
            coefficient_deinterleave_rct_gpu_duration: Duration::from_micros(4),
            coefficient_dwt53_gpu_duration: Duration::from_micros(6),
            coefficient_dwt53_vertical_gpu_duration: Duration::from_micros(2),
            coefficient_dwt53_horizontal_gpu_duration: Duration::from_micros(4),
            coefficient_extract_gpu_duration: Duration::from_micros(8),
            coefficient_copy_gpu_duration: Duration::from_micros(1),
            gpu_elapsed_wall_duration: Duration::from_micros(29),
            classic_block_gpu_duration: Duration::from_micros(19),
            classic_tier1_density_gpu_duration: Duration::from_micros(31),
            classic_tier1_raw_pack_gpu_duration: Duration::from_micros(37),
            classic_tier1_arithmetic_pack_gpu_duration: Duration::from_micros(39),
            classic_tier1_symbol_plan_gpu_duration: Duration::from_micros(41),
            classic_tier1_token_emit_gpu_duration: Duration::from_micros(43),
            classic_tier1_split_token_emit_gpu_duration: Duration::from_micros(45),
            classic_tier1_token_pack_gpu_duration: Duration::from_micros(47),
            ht_block_gpu_duration: Duration::from_micros(3),
            packet_block_prep_gpu_duration: Duration::from_micros(5),
            packetization_gpu_duration: Duration::from_micros(7),
            packet_payload_copy_gpu_duration: Duration::from_micros(11),
            codestream_assembly_gpu_duration: Duration::from_micros(13),
            codestream_payload_copy_gpu_duration: Duration::from_micros(17),
            ..super::MetalLosslessEncodeStageStats::default()
        });

        assert_eq!(
            stats.coefficient_prep_gpu_duration,
            Duration::from_micros(23)
        );
        assert_eq!(
            stats.coefficient_deinterleave_rct_gpu_duration,
            Duration::from_micros(4)
        );
        assert_eq!(
            stats.coefficient_dwt53_gpu_duration,
            Duration::from_micros(6)
        );
        assert_eq!(
            stats.coefficient_dwt53_vertical_gpu_duration,
            Duration::from_micros(2)
        );
        assert_eq!(
            stats.coefficient_dwt53_horizontal_gpu_duration,
            Duration::from_micros(4)
        );
        assert_eq!(
            stats.coefficient_extract_gpu_duration,
            Duration::from_micros(8)
        );
        assert_eq!(
            stats.coefficient_copy_gpu_duration,
            Duration::from_micros(1)
        );
        assert_eq!(stats.gpu_elapsed_wall_duration, Duration::from_micros(29));
        assert_eq!(stats.classic_block_gpu_duration, Duration::from_micros(19));
        assert_eq!(
            stats.classic_tier1_density_gpu_duration,
            Duration::from_micros(31)
        );
        assert_eq!(
            stats.classic_tier1_raw_pack_gpu_duration,
            Duration::from_micros(37)
        );
        assert_eq!(
            stats.classic_tier1_arithmetic_pack_gpu_duration,
            Duration::from_micros(39)
        );
        assert_eq!(
            stats.classic_tier1_symbol_plan_gpu_duration,
            Duration::from_micros(41)
        );
        assert_eq!(
            stats.classic_tier1_token_emit_gpu_duration,
            Duration::from_micros(43)
        );
        assert_eq!(
            stats.classic_tier1_split_token_emit_gpu_duration,
            Duration::from_micros(45)
        );
        assert_eq!(
            stats.classic_tier1_token_pack_gpu_duration,
            Duration::from_micros(47)
        );
        assert_eq!(stats.ht_block_gpu_duration, Duration::from_micros(5));
        assert_eq!(
            stats.packet_block_prep_gpu_duration,
            Duration::from_micros(5)
        );
        assert_eq!(stats.packetization_gpu_duration, Duration::from_micros(7));
        assert_eq!(
            stats.packet_payload_copy_gpu_duration,
            Duration::from_micros(11)
        );
        assert_eq!(
            stats.codestream_assembly_gpu_duration,
            Duration::from_micros(13)
        );
        assert_eq!(
            stats.codestream_payload_copy_gpu_duration,
            Duration::from_micros(17)
        );
        assert!(stats.has_timings());
    }

    #[test]
    fn resident_lossless_prep_duration_only_records_when_profiled() {
        let mut stats = super::MetalLosslessEncodeBatchStats::default();

        super::add_resident_prep_duration(&mut stats, Duration::from_micros(7), false);
        assert_eq!(stats.stage_stats.coefficient_prep_duration, Duration::ZERO);
        assert!(!stats.stage_stats.has_timings());

        super::add_resident_prep_duration(&mut stats, Duration::from_micros(7), true);
        assert_eq!(
            stats.stage_stats.coefficient_prep_duration,
            Duration::from_micros(7)
        );
        assert!(stats.stage_stats.has_timings());
    }

    #[test]
    fn resident_lossless_prep_duration_uses_wall_time_not_per_tile_sum() {
        let mut stats = super::MetalLosslessEncodeBatchStats::default();
        let wall_duration = Duration::from_micros(11);
        let per_tile_sum = Duration::from_micros(9).saturating_add(Duration::from_micros(10));
        assert_ne!(wall_duration, per_tile_sum);

        super::add_resident_prep_wall_duration(&mut stats, wall_duration, true);

        assert_eq!(stats.stage_stats.coefficient_prep_duration, wall_duration);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn resident_classic_peak_estimate_matches_tight_batch_capacity() {
        let plan = super::LosslessDeviceEncodePlan {
            components: 1,
            bit_depth: 8,
            block_coding_mode: J2kBlockCodingMode::Classic,
            num_decomposition_levels: 0,
            use_mct: false,
            guard_bits: 2,
            code_block_width_exp: 4,
            code_block_height_exp: 4,
            code_blocks: vec![compute::J2kLosslessDeviceCodeBlock {
                coefficient_offset: 0,
                component: 0,
                subband_x: 0,
                subband_y: 0,
                block_x: 0,
                block_y: 0,
                width: 64,
                height: 64,
                sub_band_type: signinum_j2k_native::J2kSubBandType::LowLow,
                total_bitplanes: 11,
            }],
            resolutions: Vec::new(),
            progression_order: signinum_j2k_native::EncodeProgressionOrder::Lrcp,
            write_tlm: false,
        };

        assert_eq!(
            super::estimated_tier1_output_bytes(&plan),
            64 * 64 * 11 + 4097
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn resident_classic_batch_retry_covers_tight_capacity_failures() {
        let tight_tier1_error = crate::Error::MetalKernel {
            message: "packetization Metal encode kernel failure (detail=7, tier1_detail=4)"
                .to_string(),
        };
        assert!(super::resident_classic_batch_encode_should_retry_conservative(&tight_tier1_error));

        let tight_tier1_finish_error = crate::Error::MetalKernel {
            message: "classic Tier-1 Metal encode kernel failure (detail=5)".to_string(),
        };
        assert!(
            super::resident_classic_batch_encode_should_retry_conservative(
                &tight_tier1_finish_error
            )
        );

        let packet_error = crate::Error::MetalKernel {
            message: "packetization Metal encode kernel failure (detail=5)".to_string(),
        };
        assert!(super::resident_classic_batch_encode_should_retry_conservative(&packet_error));

        let codestream_error = crate::Error::MetalKernel {
            message: "J2K batched codestream assembly Metal encode kernel failure (detail=2)"
                .to_string(),
        };
        assert!(super::resident_classic_batch_encode_should_retry_conservative(&codestream_error));

        let unrelated_error = crate::Error::MetalKernel {
            message: "packetization Metal encode kernel failure (detail=8)".to_string(),
        };
        assert!(!super::resident_classic_batch_encode_should_retry_conservative(&unrelated_error));
    }

    #[test]
    fn resident_lossless_ht_command_duration_matches_split_buckets() {
        let stats = super::MetalLosslessEncodeStageStats {
            ht_command_encode_duration: Duration::from_micros(2)
                .saturating_add(Duration::from_micros(3))
                .saturating_add(Duration::from_micros(5))
                .saturating_add(Duration::from_micros(7)),
            ht_block_encode_duration: Duration::from_micros(2),
            packet_block_prep_duration: Duration::from_micros(3),
            packetization_duration: Duration::from_micros(5),
            codestream_assembly_duration: Duration::from_micros(7),
            ..super::MetalLosslessEncodeStageStats::default()
        };

        assert_eq!(
            stats.ht_command_encode_duration,
            stats
                .ht_block_encode_duration
                .saturating_add(stats.packet_block_prep_duration)
                .saturating_add(stats.packetization_duration)
                .saturating_add(stats.codestream_assembly_duration)
        );
    }

    #[test]
    fn lossless_encode_outcome_exposes_host_readback_duration() {
        let outcome = super::MetalLosslessEncodeOutcome {
            encoded: EncodedJ2k {
                codestream: Vec::new(),
                backend: signinum_core::BackendKind::Metal,
                dispatch_report: signinum_j2k::J2kEncodeDispatchReport::default(),
                width: 0,
                height: 0,
                components: 1,
                bit_depth: 8,
                signed: false,
            },
            input_copy_used: false,
            resident: super::MetalLosslessEncodeResidency {
                coefficient_prep_used: false,
                packetization_used: false,
                codestream_assembly_used: false,
            },
            input_copy_duration: Duration::ZERO,
            encode_duration: Duration::ZERO,
            gpu_duration: None,
            validation_duration: Duration::ZERO,
            host_readback_duration: Duration::from_micros(3),
        };

        assert_eq!(outcome.host_readback_duration, Duration::from_micros(3));
    }

    #[test]
    fn resident_lossless_chunk_ranges_respect_inflight_and_code_block_caps() {
        assert_eq!(
            super::resident_lossless_chunk_ranges_for_test(&[32, 32, 32, 32, 32], 3, 96),
            vec![0..3, 3..5]
        );
        assert_eq!(
            super::resident_lossless_chunk_ranges_for_test(&[80, 80, 10], 8, 96),
            vec![0..1, 1..3]
        );
    }

    #[test]
    fn resident_lossless_default_code_block_cap_allows_large_wsi_chunks() {
        let code_blocks = vec![192usize; 600];
        let cap = super::resident_lossless_code_block_chunk_cap(&code_blocks);

        assert_eq!(
            super::resident_lossless_chunk_ranges_for_test(&code_blocks, 512, cap),
            vec![0..512, 512..600]
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_dispatch_option_treats_unavailable_as_no_dispatch() {
        let result: Result<Option<u8>, &'static str> =
            super::metal_dispatch_option(Err(crate::Error::MetalUnavailable), "kernel failed");

        assert_eq!(result, Ok(None));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_dispatch_option_preserves_kernel_errors() {
        let result: Result<Option<u8>, &'static str> = super::metal_dispatch_option(
            Err(crate::Error::MetalKernel {
                message: "bad status".to_string(),
            }),
            "kernel failed",
        );

        assert_eq!(result, Err("kernel failed"));
    }

    #[test]
    fn metal_encode_stage_accelerator_preserves_cpu_codestream_validity() {
        let pixels: Vec<u8> = (0..8 * 8 * 3).map(|i| (i & 0xFF) as u8).collect();
        let samples =
            J2kLosslessSamples::new(&pixels, 8, 8, 3, 8, false).expect("valid RGB samples");
        let options = J2kLosslessEncodeOptions::default()
            .with_backend(EncodeBackendPreference::PreferDevice)
            .with_max_decomposition_levels(Some(1));
        let mut accelerator = MetalEncodeStageAccelerator::default();

        let encoded = encode_j2k_lossless_with_accelerator(
            samples,
            &options,
            signinum_core::BackendKind::Metal,
            &mut accelerator,
        )
        .expect("encode with metal stage accelerator");
        let decoded = Image::new(&encoded.codestream, &DecodeSettings::default())
            .expect("codestream parses")
            .decode_native()
            .expect("codestream decodes");

        assert_eq!(decoded.width, 8);
        assert_eq!(decoded.height, 8);
        assert_eq!(decoded.num_components, 3);
        assert_eq!(decoded.bit_depth, 8);
        assert_eq!(accelerator.forward_rct_attempts(), 1);
        assert_eq!(accelerator.forward_dwt53_attempts(), 3);
        assert!(accelerator.tier1_code_block_attempts() > 0);
        assert_eq!(accelerator.packetization_attempts(), 1);
    }

    #[test]
    fn metal_encode_stage_accelerator_can_leave_forward_rct_on_cpu() {
        let mut plane0 = vec![0.0, 64.0, 128.0, 255.0];
        let mut plane1 = vec![3.0, 67.0, 131.0, 252.0];
        let mut plane2 = vec![7.0, 71.0, 135.0, 248.0];
        let original = (plane0.clone(), plane1.clone(), plane2.clone());
        let mut accelerator = MetalEncodeStageAccelerator::with_cpu_forward_rct();

        let dispatched = accelerator
            .encode_forward_rct(J2kForwardRctJob {
                plane0: &mut plane0,
                plane1: &mut plane1,
                plane2: &mut plane2,
            })
            .expect("CPU RCT fallback should be selectable");

        assert!(!dispatched);
        assert_eq!(accelerator.forward_rct_attempts(), 1);
        assert_eq!(accelerator.forward_rct_dispatches(), 0);
        assert_eq!((plane0, plane1, plane2), original);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_forward_rct_dispatch_round_trips_rgb8_lossless_tile() {
        let pixels: Vec<u8> = (0..7 * 5 * 3).map(|i| ((i * 17) & 0xFF) as u8).collect();
        let samples =
            J2kLosslessSamples::new(&pixels, 7, 5, 3, 8, false).expect("valid RGB samples");
        let options = J2kLosslessEncodeOptions::default()
            .with_backend(EncodeBackendPreference::RequireDevice)
            .with_max_decomposition_levels(Some(0));
        let mut accelerator = MetalEncodeStageAccelerator::default();

        let encoded = encode_j2k_lossless_with_accelerator(
            samples,
            &options,
            BackendKind::Metal,
            &mut accelerator,
        )
        .expect("encode with metal forward RCT");
        let decoded = Image::new(&encoded.codestream, &DecodeSettings::default())
            .expect("codestream parses")
            .decode_native()
            .expect("codestream decodes");

        assert_eq!(decoded.data, pixels);
        assert_eq!(accelerator.forward_rct_attempts(), 1);
        assert_eq!(accelerator.forward_rct_dispatches(), 1);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_validation_decodes_and_compares_lossless_codestream_on_device() {
        let pixels: Vec<u8> = (0..16 * 16 * 3).map(|i| ((i * 29) & 0xFF) as u8).collect();
        let samples = J2kLosslessSamples::new(&pixels, 16, 16, 3, 8, false).unwrap();
        let encoded = signinum_j2k::encode_j2k_lossless(
            samples,
            &lossless_options! {
                backend: EncodeBackendPreference::CpuOnly,
            },
        )
        .expect("lossless encode");

        super::validate_lossless_roundtrip_on_metal(samples, &encoded.codestream)
            .expect("Metal lossless validation");
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_buffer_lossless_encode_pads_edge_tile_on_device() {
        let pixels: Vec<u8> = (0..7 * 5 * 3).map(|i| ((i * 19) & 0xFF) as u8).collect();
        let device = metal::Device::system_default().expect("Metal device");
        let session = crate::MetalBackendSession::new(device);
        let buffer = session.device().new_buffer_with_data(
            pixels.as_ptr().cast(),
            pixels.len() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let encoded = super::encode_lossless_from_metal_buffer(
            super::MetalLosslessEncodeTile {
                buffer: &buffer,
                byte_offset: 0,
                width: 7,
                height: 5,
                pitch_bytes: 7 * 3,
                output_width: 8,
                output_height: 8,
                format: PixelFormat::Rgb8,
            },
            &lossless_options! {
                backend: EncodeBackendPreference::RequireDevice,
            },
            &session,
        )
        .expect("Metal buffer lossless encode");

        assert_eq!(encoded.backend, BackendKind::Metal);
        let decoded = Image::new(&encoded.codestream, &DecodeSettings::default())
            .expect("codestream parses")
            .decode_native()
            .expect("codestream decodes");
        assert_eq!(decoded.width, 8);
        assert_eq!(decoded.height, 8);
        for y in 0..8usize {
            for x in 0..8usize {
                let dst = (y * 8 + x) * 3;
                if x < 7 && y < 5 {
                    let src = (y * 7 + x) * 3;
                    assert_eq!(&decoded.data[dst..dst + 3], &pixels[src..src + 3]);
                } else {
                    assert_eq!(&decoded.data[dst..dst + 3], &[0, 0, 0]);
                }
            }
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn submitted_metal_buffer_lossless_encode_wait_round_trips() {
        let pixels: Vec<u8> = (0..7 * 5 * 3).map(|i| ((i * 19) & 0xFF) as u8).collect();
        let session = crate::MetalBackendSession::system_default().expect("Metal session");
        let buffer = session.device().new_buffer_with_data(
            pixels.as_ptr().cast(),
            pixels.len() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let submitted: crate::SubmittedJ2kLosslessMetalEncode =
            crate::submit_lossless_from_metal_buffer(
                super::MetalLosslessEncodeTile {
                    buffer: &buffer,
                    byte_offset: 0,
                    width: 7,
                    height: 5,
                    pitch_bytes: 7 * 3,
                    output_width: 8,
                    output_height: 8,
                    format: PixelFormat::Rgb8,
                },
                &lossless_options! {
                    backend: EncodeBackendPreference::RequireDevice,
                },
                &session,
            )
            .expect("submit Metal buffer lossless encode");
        let encoded = submitted.wait().expect("wait Metal buffer lossless encode");

        assert_eq!(encoded.backend, BackendKind::Metal);
        let decoded = Image::new(&encoded.codestream, &DecodeSettings::default())
            .expect("codestream parses")
            .decode_native()
            .expect("codestream decodes");
        assert_eq!(decoded.width, 8);
        assert_eq!(decoded.height, 8);
        for y in 0..8usize {
            for x in 0..8usize {
                let dst = (y * 8 + x) * 3;
                if x < 7 && y < 5 {
                    let src = (y * 7 + x) * 3;
                    assert_eq!(&decoded.data[dst..dst + 3], &pixels[src..src + 3]);
                } else {
                    assert_eq!(&decoded.data[dst..dst + 3], &[0, 0, 0]);
                }
            }
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_buffer_lossless_encode_accepts_padded_contiguous_input_without_copy() {
        let pixels: Vec<u8> = (0..8 * 8 * 3).map(|i| ((i * 31) & 0xFF) as u8).collect();
        let session = crate::MetalBackendSession::system_default().expect("Metal session");
        let buffer = session.device().new_buffer_with_data(
            pixels.as_ptr().cast(),
            pixels.len() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let encoded = super::encode_lossless_from_padded_metal_buffer_with_report(
            super::MetalLosslessEncodeTile {
                buffer: &buffer,
                byte_offset: 0,
                width: 8,
                height: 8,
                pitch_bytes: 8 * 3,
                output_width: 8,
                output_height: 8,
                format: PixelFormat::Rgb8,
            },
            &lossless_options! {
                backend: EncodeBackendPreference::RequireDevice,
            },
            &session,
        )
        .expect("Metal padded buffer lossless encode");

        assert_eq!(encoded.encoded.backend, BackendKind::Metal);
        assert!(!encoded.input_copy_used);
        assert_eq!(encoded.input_copy_duration, std::time::Duration::ZERO);
        let decoded = Image::new(&encoded.encoded.codestream, &DecodeSettings::default())
            .expect("codestream parses")
            .decode_native()
            .expect("codestream decodes");
        assert_eq!(decoded.width, 8);
        assert_eq!(decoded.height, 8);
        assert_eq!(decoded.data, pixels);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_padded_private_rgb8_encode_uses_resident_coefficient_prep() {
        let pixels: Vec<u8> = (0..8 * 8 * 3).map(|i| ((i * 31) & 0xFF) as u8).collect();
        let session = crate::MetalBackendSession::system_default().expect("Metal session");
        let buffer = private_buffer_with_bytes(&session, &pixels);

        let encoded = super::encode_lossless_from_padded_metal_buffer_with_report(
            super::MetalLosslessEncodeTile {
                buffer: &buffer,
                byte_offset: 0,
                width: 8,
                height: 8,
                pitch_bytes: 8 * 3,
                output_width: 8,
                output_height: 8,
                format: PixelFormat::Rgb8,
            },
            &lossless_options! {
                backend: EncodeBackendPreference::RequireDevice,
            },
            &session,
        )
        .expect("Metal private padded buffer lossless encode");

        assert_eq!(encoded.encoded.backend, BackendKind::Metal);
        assert!(!encoded.input_copy_used);
        assert!(encoded.resident.coefficient_prep_used);
        assert!(encoded.resident.packetization_used);
        assert!(encoded.resident.codestream_assembly_used);
        let decoded = Image::new(&encoded.encoded.codestream, &DecodeSettings::default())
            .expect("codestream parses")
            .decode_native()
            .expect("codestream decodes");
        assert_decoded_bytes_match(&decoded.data, &pixels);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn auto_host_output_encode_options_preserve_auto_for_hybrid_path() {
        let routed = super::host_output_encode_options(lossless_options! {
            backend: EncodeBackendPreference::Auto,
            validation: J2kEncodeValidation::CpuRoundTrip,
        });

        assert_eq!(routed.backend, EncodeBackendPreference::Auto);
        assert_eq!(routed.validation, J2kEncodeValidation::External);

        let prefer_device = super::host_output_encode_options(lossless_options! {
            backend: EncodeBackendPreference::PreferDevice,
            validation: J2kEncodeValidation::CpuRoundTrip,
        });
        assert_eq!(prefer_device.backend, EncodeBackendPreference::PreferDevice);
        assert_eq!(prefer_device.validation, J2kEncodeValidation::External);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn auto_host_output_accelerator_uses_metal_dwt_with_cpu_block_fallback() {
        let pixels: Vec<u8> = (0..64 * 64).map(|i| ((i * 17) & 0xff) as u8).collect();
        let samples =
            J2kLosslessSamples::new(&pixels, 64, 64, 1, 8, false).expect("valid gray samples");
        let options = lossless_options! {
            backend: EncodeBackendPreference::Auto,
            validation: J2kEncodeValidation::External,
        };
        let mut accelerator = MetalEncodeStageAccelerator::for_auto_host_output();

        let encoded = encode_j2k_lossless_with_accelerator(
            samples,
            &options,
            BackendKind::Metal,
            &mut accelerator,
        )
        .expect("hybrid host-output encode");

        assert_eq!(encoded.backend, BackendKind::Cpu);
        assert_eq!(accelerator.forward_dwt53_dispatches(), 1);
        assert_eq!(accelerator.tier1_code_block_dispatches(), 0);
        assert_eq!(accelerator.packetization_dispatches(), 0);
        assert!(accelerator.prefer_parallel_cpu_code_block_fallback());
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn auto_htj2k_small_host_output_stays_cpu_below_resident_gate() {
        let mut pixels = Vec::with_capacity(64 * 64 * 3);
        for y in 0..64u32 {
            for x in 0..64u32 {
                pixels.push(((x * 3 + y * 5) & 0xff) as u8);
                pixels.push(((x * 7 + y * 11) & 0xff) as u8);
                pixels.push(((x * 13 + y * 17) & 0xff) as u8);
            }
        }
        let samples =
            J2kLosslessSamples::new(&pixels, 64, 64, 3, 8, false).expect("valid RGB samples");
        let options = lossless_options! {
            backend: EncodeBackendPreference::Auto,
            block_coding_mode: J2kBlockCodingMode::HighThroughput,
            validation: J2kEncodeValidation::External,
        };
        let mut accelerator = MetalEncodeStageAccelerator::for_auto_host_output();

        let encoded = encode_j2k_lossless_with_accelerator(
            samples,
            &options,
            BackendKind::Metal,
            &mut accelerator,
        )
        .expect("hybrid HTJ2K host-output encode");
        let decoded = Image::new(&encoded.codestream, &DecodeSettings::default())
            .expect("codestream parses")
            .decode_native()
            .expect("codestream decodes");

        assert_eq!(decoded.data, pixels);
        assert_eq!(encoded.backend, BackendKind::Cpu);
        assert_eq!(accelerator.forward_rct_dispatches(), 0);
        assert_eq!(accelerator.forward_dwt53_dispatches(), 0);
        assert_eq!(accelerator.ht_code_block_dispatches(), 0);
        assert_eq!(accelerator.packetization_dispatches(), 0);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn auto_htj2k_large_host_output_uses_resident_metal_rct_dwt_and_ht_with_cpu_packetization() {
        let width = 1024u32;
        let height = 1024u32;
        let mut pixels = Vec::with_capacity(width as usize * height as usize * 3);
        for y in 0..height {
            for x in 0..width {
                pixels.push(((x * 3 + y * 5) & 0xff) as u8);
                pixels.push(((x * 7 + y * 11) & 0xff) as u8);
                pixels.push(((x * 13 + y * 17) & 0xff) as u8);
            }
        }
        let samples = J2kLosslessSamples::new(&pixels, width, height, 3, 8, false)
            .expect("valid RGB samples");
        let options = lossless_options! {
            backend: EncodeBackendPreference::Auto,
            block_coding_mode: J2kBlockCodingMode::HighThroughput,
            validation: J2kEncodeValidation::External,
        };
        let mut accelerator = MetalEncodeStageAccelerator::for_auto_host_output();

        let encoded = encode_j2k_lossless_with_accelerator(
            samples,
            &options,
            BackendKind::Metal,
            &mut accelerator,
        )
        .expect("hybrid HTJ2K host-output encode");
        let decoded = Image::new(&encoded.codestream, &DecodeSettings::default())
            .expect("codestream parses")
            .decode_native()
            .expect("codestream decodes");

        assert_eq!(decoded.data, pixels);
        assert_eq!(encoded.backend, BackendKind::Cpu);
        assert!(accelerator.forward_rct_dispatches() > 0);
        assert_eq!(accelerator.forward_dwt53_dispatches(), 3);
        assert!(accelerator.ht_code_block_dispatches() > 0);
        assert_eq!(accelerator.packetization_dispatches(), 0);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn auto_htj2k_padded_rgb8_uses_fused_metal_rct_with_cpu_packetization() {
        let mut pixels = Vec::with_capacity(64 * 64 * 3);
        for y in 0..64u32 {
            for x in 0..64u32 {
                pixels.push(((x * 19 + y * 3) & 0xff) as u8);
                pixels.push(((x * 5 + y * 23) & 0xff) as u8);
                pixels.push(((x * 11 + y * 13) & 0xff) as u8);
            }
        }
        let session = crate::MetalBackendSession::system_default().expect("Metal session");
        let buffer = private_buffer_with_bytes(&session, &pixels);
        compute::reset_lossless_deinterleave_rct_fused_dispatches_for_test();

        let encoded = super::encode_lossless_from_padded_metal_buffer_with_report(
            super::MetalLosslessEncodeTile {
                buffer: &buffer,
                byte_offset: 0,
                width: 64,
                height: 64,
                pitch_bytes: 64 * 3,
                output_width: 64,
                output_height: 64,
                format: PixelFormat::Rgb8,
            },
            &lossless_options! {
                backend: EncodeBackendPreference::Auto,
                block_coding_mode: J2kBlockCodingMode::HighThroughput,
                validation: J2kEncodeValidation::External,
            },
            &session,
        )
        .expect("Auto HTJ2K resident hybrid encode");
        let decoded = Image::new(&encoded.encoded.codestream, &DecodeSettings::default())
            .expect("codestream parses")
            .decode_native()
            .expect("codestream decodes");

        assert_eq!(decoded.data, pixels);
        assert_eq!(encoded.encoded.backend, BackendKind::Cpu);
        assert!(encoded.resident.coefficient_prep_used);
        assert!(!encoded.resident.packetization_used);
        assert!(!encoded.resident.codestream_assembly_used);
        assert!(
            compute::lossless_deinterleave_rct_fused_dispatches_for_test() > 0,
            "Auto HTJ2K resident hybrid should use fused deinterleave + RCT"
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_padded_private_rgb8_auto_host_encode_routes_away_from_resident_prep() {
        let pixels: Vec<u8> = (0..8 * 8 * 3).map(|i| ((i * 43) & 0xFF) as u8).collect();
        let session = crate::MetalBackendSession::system_default().expect("Metal session");
        let buffer = private_buffer_with_bytes(&session, &pixels);

        let encoded = super::encode_lossless_from_padded_metal_buffer_with_report(
            super::MetalLosslessEncodeTile {
                buffer: &buffer,
                byte_offset: 0,
                width: 8,
                height: 8,
                pitch_bytes: 8 * 3,
                output_width: 8,
                output_height: 8,
                format: PixelFormat::Rgb8,
            },
            &lossless_options! {
                backend: EncodeBackendPreference::Auto,
                validation: J2kEncodeValidation::External,
            },
            &session,
        )
        .expect("Auto host-output encode should avoid resident prep and still succeed");

        assert_eq!(encoded.encoded.backend, BackendKind::Cpu);
        assert!(!encoded.resident.coefficient_prep_used);
        assert!(!encoded.resident.packetization_used);
        assert!(!encoded.resident.codestream_assembly_used);
        let decoded = Image::new(&encoded.encoded.codestream, &DecodeSettings::default())
            .expect("codestream parses")
            .decode_native()
            .expect("codestream decodes");
        assert_decoded_bytes_match(&decoded.data, &pixels);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_padded_private_rgb8_encode_to_metal_buffer_exposes_finished_bytes() {
        let pixels: Vec<u8> = (0..8 * 8 * 3).map(|i| ((i * 37) & 0xFF) as u8).collect();
        let session = crate::MetalBackendSession::system_default().expect("Metal session");
        let buffer = private_buffer_with_bytes(&session, &pixels);

        let encoded = super::encode_lossless_from_padded_metal_buffer_to_metal_with_report(
            super::MetalLosslessEncodeTile {
                buffer: &buffer,
                byte_offset: 0,
                width: 8,
                height: 8,
                pitch_bytes: 8 * 3,
                output_width: 8,
                output_height: 8,
                format: PixelFormat::Rgb8,
            },
            &lossless_options! {
                backend: EncodeBackendPreference::RequireDevice,
            },
            &session,
        )
        .expect("Metal private padded buffer lossless encode to Metal buffer");

        assert!(!encoded.input_copy_used);
        assert!(encoded.resident.coefficient_prep_used);
        assert!(encoded.resident.packetization_used);
        assert!(encoded.resident.codestream_assembly_used);
        if let Some(duration) = encoded.gpu_duration {
            assert!(duration > Duration::ZERO);
        }
        assert_eq!(encoded.encoded.byte_offset, 0);
        assert!(encoded.encoded.byte_len > 0);
        assert!(encoded.encoded.capacity >= encoded.encoded.byte_len);
        let codestream = encoded
            .encoded
            .codestream_bytes()
            .expect("Metal codestream bytes are CPU-readable");
        assert!(codestream.starts_with(&[0xFF, 0x4F]));
        let decoded = Image::new(codestream, &DecodeSettings::default())
            .expect("codestream parses")
            .decode_native()
            .expect("codestream decodes");
        assert_eq!(decoded.data, pixels);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_edge_private_rgb8_encode_to_metal_buffer_pads_and_stays_resident() {
        let pixels: Vec<u8> = (0..7 * 5 * 3).map(|i| ((i * 41) & 0xFF) as u8).collect();
        let session = crate::MetalBackendSession::system_default().expect("Metal session");
        let buffer = private_buffer_with_bytes(&session, &pixels);

        let encoded = super::encode_lossless_from_metal_buffer_to_metal_with_report(
            super::MetalLosslessEncodeTile {
                buffer: &buffer,
                byte_offset: 0,
                width: 7,
                height: 5,
                pitch_bytes: 7 * 3,
                output_width: 8,
                output_height: 8,
                format: PixelFormat::Rgb8,
            },
            &lossless_options! {
                backend: EncodeBackendPreference::RequireDevice,
            },
            &session,
        )
        .expect("Metal private edge buffer lossless encode to Metal buffer");

        assert!(!encoded.input_copy_used);
        assert!(encoded.resident.coefficient_prep_used);
        assert!(encoded.resident.packetization_used);
        assert!(encoded.resident.codestream_assembly_used);
        let codestream = encoded
            .encoded
            .codestream_bytes()
            .expect("Metal codestream bytes are CPU-readable");
        assert!(codestream.starts_with(&[0xFF, 0x4F]));
        let decoded = Image::new(codestream, &DecodeSettings::default())
            .expect("codestream parses")
            .decode_native()
            .expect("codestream decodes");
        assert_eq!(decoded.width, 8);
        assert_eq!(decoded.height, 8);
        for y in 0..8usize {
            for x in 0..8usize {
                let dst = (y * 8 + x) * 3;
                if x < 7 && y < 5 {
                    let src = (y * 7 + x) * 3;
                    assert_eq!(&decoded.data[dst..dst + 3], &pixels[src..src + 3]);
                } else {
                    assert_eq!(&decoded.data[dst..dst + 3], &[0, 0, 0]);
                }
            }
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn submitted_private_padded_rgb8_encode_snapshots_before_wait() {
        let pixels: Vec<u8> = (0..8 * 8 * 3).map(|i| ((i * 31) & 0xFF) as u8).collect();
        let replacement = vec![0u8; pixels.len()];
        let session = crate::MetalBackendSession::system_default().expect("Metal session");
        let buffer = private_buffer_with_bytes(&session, &pixels);

        let submitted = super::submit_lossless_from_padded_metal_buffer(
            super::MetalLosslessEncodeTile {
                buffer: &buffer,
                byte_offset: 0,
                width: 8,
                height: 8,
                pitch_bytes: 8 * 3,
                output_width: 8,
                output_height: 8,
                format: PixelFormat::Rgb8,
            },
            &lossless_options! {
                backend: EncodeBackendPreference::RequireDevice,
            },
            &session,
        )
        .expect("submit Metal private padded RGB8 encode");
        overwrite_private_buffer_with_bytes(&session, &buffer, &replacement);

        let encoded = submitted.wait().expect("wait submitted encode");
        let decoded = Image::new(&encoded.codestream, &DecodeSettings::default())
            .expect("codestream parses")
            .decode_native()
            .expect("codestream decodes");
        assert_eq!(decoded.data, pixels);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_padded_private_gray8_dwt_encode_uses_resident_coefficient_prep() {
        let mut pixels = Vec::with_capacity(128 * 128);
        for y in 0..128u32 {
            for x in 0..128u32 {
                pixels.push(((x * 7 + y * 11 + (x ^ y)) & 0xFF) as u8);
            }
        }
        let session = crate::MetalBackendSession::system_default().expect("Metal session");
        let buffer = private_buffer_with_bytes(&session, &pixels);

        let encoded = super::encode_lossless_from_padded_metal_buffer_with_report(
            super::MetalLosslessEncodeTile {
                buffer: &buffer,
                byte_offset: 0,
                width: 128,
                height: 128,
                pitch_bytes: 128,
                output_width: 128,
                output_height: 128,
                format: PixelFormat::Gray8,
            },
            &lossless_options! {
                backend: EncodeBackendPreference::RequireDevice,
            },
            &session,
        )
        .expect("Metal private padded DWT buffer lossless encode");

        assert_eq!(encoded.encoded.backend, BackendKind::Metal);
        assert!(!encoded.input_copy_used);
        assert!(encoded.resident.coefficient_prep_used);
        assert!(encoded.resident.packetization_used);
        assert!(encoded.resident.codestream_assembly_used);
        let decoded = Image::new(&encoded.encoded.codestream, &DecodeSettings::default())
            .expect("codestream parses")
            .decode_native()
            .expect("codestream decodes");
        assert_decoded_bytes_match(&decoded.data, &pixels);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_padded_private_rgb8_dwt_encode_uses_resident_coefficient_prep() {
        let mut pixels = Vec::with_capacity(128 * 128 * 3);
        for y in 0..128u32 {
            for x in 0..128u32 {
                pixels.push(((x * 3 + y * 5) & 0xFF) as u8);
                pixels.push(((x * 7 + y * 11) & 0xFF) as u8);
                pixels.push(((x * 13 + y * 17) & 0xFF) as u8);
            }
        }
        let session = crate::MetalBackendSession::system_default().expect("Metal session");
        let buffer = private_buffer_with_bytes(&session, &pixels);

        let encoded = super::encode_lossless_from_padded_metal_buffer_with_report(
            super::MetalLosslessEncodeTile {
                buffer: &buffer,
                byte_offset: 0,
                width: 128,
                height: 128,
                pitch_bytes: 128 * 3,
                output_width: 128,
                output_height: 128,
                format: PixelFormat::Rgb8,
            },
            &lossless_options! {
                backend: EncodeBackendPreference::RequireDevice,
            },
            &session,
        )
        .expect("Metal private padded RGB8 DWT buffer lossless encode");

        assert_eq!(encoded.encoded.backend, BackendKind::Metal);
        assert!(encoded.resident.coefficient_prep_used);
        assert!(encoded.resident.packetization_used);
        assert!(encoded.resident.codestream_assembly_used);
        let decoded = Image::new(&encoded.encoded.codestream, &DecodeSettings::default())
            .expect("codestream parses")
            .decode_native()
            .expect("codestream decodes");
        assert_decoded_bytes_match(&decoded.data, &pixels);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_padded_private_gray8_dwt_resident_codestream_decodes_natively() {
        let mut pixels = Vec::with_capacity(128 * 128);
        for y in 0..128u32 {
            for x in 0..128u32 {
                pixels.push(((x * 7 + y * 11 + (x ^ y)) & 0xFF) as u8);
            }
        }
        let session = crate::MetalBackendSession::system_default().expect("Metal session");
        let buffer = private_buffer_with_bytes(&session, &pixels);

        let encoded = super::encode_lossless_from_padded_metal_buffer_with_report(
            super::MetalLosslessEncodeTile {
                buffer: &buffer,
                byte_offset: 0,
                width: 128,
                height: 128,
                pitch_bytes: 128,
                output_width: 128,
                output_height: 128,
                format: PixelFormat::Gray8,
            },
            &lossless_options! {
                backend: EncodeBackendPreference::RequireDevice,
                validation: J2kEncodeValidation::External,
            },
            &session,
        )
        .expect("Metal private padded DWT buffer lossless encode");

        let decoded = Image::new(&encoded.encoded.codestream, &DecodeSettings::default())
            .expect("codestream parses")
            .decode_native()
            .expect("codestream decodes");
        assert_decoded_bytes_match(&decoded.data, &pixels);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_padded_private_rgb8_dwt_resident_codestream_decodes_natively() {
        let mut pixels = Vec::with_capacity(128 * 128 * 3);
        for y in 0..128u32 {
            for x in 0..128u32 {
                pixels.push(((x * 3 + y * 5) & 0xFF) as u8);
                pixels.push(((x * 7 + y * 11) & 0xFF) as u8);
                pixels.push(((x * 13 + y * 17) & 0xFF) as u8);
            }
        }
        let session = crate::MetalBackendSession::system_default().expect("Metal session");
        let buffer = private_buffer_with_bytes(&session, &pixels);

        let encoded = super::encode_lossless_from_padded_metal_buffer_with_report(
            super::MetalLosslessEncodeTile {
                buffer: &buffer,
                byte_offset: 0,
                width: 128,
                height: 128,
                pitch_bytes: 128 * 3,
                output_width: 128,
                output_height: 128,
                format: PixelFormat::Rgb8,
            },
            &lossless_options! {
                backend: EncodeBackendPreference::RequireDevice,
                validation: J2kEncodeValidation::External,
            },
            &session,
        )
        .expect("Metal private padded RGB8 DWT buffer lossless encode");

        let decoded = Image::new(&encoded.encoded.codestream, &DecodeSettings::default())
            .expect("codestream parses")
            .decode_native()
            .expect("codestream decodes");
        assert_decoded_bytes_match(&decoded.data, &pixels);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_padded_private_gray8_rpcl_encode_uses_resident_coefficient_prep() {
        let mut pixels = Vec::with_capacity(128 * 128);
        for y in 0..128u32 {
            for x in 0..128u32 {
                pixels.push(((x * 5 + y * 9 + (x ^ y)) & 0xFF) as u8);
            }
        }
        let session = crate::MetalBackendSession::system_default().expect("Metal session");
        let buffer = private_buffer_with_bytes(&session, &pixels);

        let encoded = super::encode_lossless_from_padded_metal_buffer_with_report(
            super::MetalLosslessEncodeTile {
                buffer: &buffer,
                byte_offset: 0,
                width: 128,
                height: 128,
                pitch_bytes: 128,
                output_width: 128,
                output_height: 128,
                format: PixelFormat::Gray8,
            },
            &lossless_options! {
                backend: EncodeBackendPreference::RequireDevice,
                progression: J2kProgressionOrder::Rpcl,
            },
            &session,
        )
        .expect("Metal private padded RPCL buffer lossless encode");

        assert_eq!(encoded.encoded.backend, BackendKind::Metal);
        assert!(encoded.resident.coefficient_prep_used);
        assert!(encoded.resident.packetization_used);
        assert!(encoded.resident.codestream_assembly_used);
        let decoded = Image::new(&encoded.encoded.codestream, &DecodeSettings::default())
            .expect("codestream parses")
            .decode_native()
            .expect("codestream decodes");
        assert_eq!(decoded.data, pixels);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_padded_private_gray16_encode_uses_resident_coefficient_prep() {
        let mut pixels = Vec::with_capacity(8 * 8 * 2);
        for idx in 0..64u16 {
            let value = idx.wrapping_mul(997).wrapping_add(123);
            pixels.extend_from_slice(&value.to_le_bytes());
        }
        let session = crate::MetalBackendSession::system_default().expect("Metal session");
        let buffer = private_buffer_with_bytes(&session, &pixels);

        let encoded = super::encode_lossless_from_padded_metal_buffer_with_report(
            super::MetalLosslessEncodeTile {
                buffer: &buffer,
                byte_offset: 0,
                width: 8,
                height: 8,
                pitch_bytes: 8 * 2,
                output_width: 8,
                output_height: 8,
                format: PixelFormat::Gray16,
            },
            &lossless_options! {
                backend: EncodeBackendPreference::RequireDevice,
            },
            &session,
        )
        .expect("Metal private padded Gray16 buffer lossless encode");

        assert_eq!(encoded.encoded.backend, BackendKind::Metal);
        assert!(!encoded.input_copy_used);
        assert!(encoded.resident.coefficient_prep_used);
        assert!(encoded.resident.packetization_used);
        assert!(encoded.resident.codestream_assembly_used);
        let decoded = Image::new(&encoded.encoded.codestream, &DecodeSettings::default())
            .expect("codestream parses")
            .decode_native()
            .expect("codestream decodes");
        assert_eq!(decoded.data, pixels);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_padded_private_ht_encode_to_metal_buffer_stays_resident() {
        let pixels: Vec<u8> = (0..8 * 8).map(|i| ((i * 31) & 0xFF) as u8).collect();
        let session = crate::MetalBackendSession::system_default().expect("Metal session");
        let buffer = private_buffer_with_bytes(&session, &pixels);

        let encoded = super::encode_lossless_from_padded_metal_buffer_to_metal_with_report(
            super::MetalLosslessEncodeTile {
                buffer: &buffer,
                byte_offset: 0,
                width: 8,
                height: 8,
                pitch_bytes: 8,
                output_width: 8,
                output_height: 8,
                format: PixelFormat::Gray8,
            },
            &lossless_options! {
                backend: EncodeBackendPreference::RequireDevice,
                block_coding_mode: J2kBlockCodingMode::HighThroughput,
            },
            &session,
        )
        .expect("Metal private padded HTJ2K buffer lossless encode");

        assert!(!encoded.input_copy_used);
        assert!(encoded.resident.coefficient_prep_used);
        assert!(encoded.resident.packetization_used);
        assert!(encoded.resident.codestream_assembly_used);
        let codestream = encoded
            .encoded
            .codestream_bytes()
            .expect("Metal codestream bytes are CPU-readable");
        assert!(codestream.windows(2).any(|window| window == [0xFF, 0x50]));
        let cod_marker = codestream
            .windows(2)
            .position(|window| window == [0xFF, 0x52])
            .expect("COD marker");
        assert_eq!(codestream[cod_marker + 12], 0x40);
        let decoded = Image::new(codestream, &DecodeSettings::default())
            .expect("codestream parses")
            .decode_native()
            .expect("codestream decodes");
        assert_eq!(decoded.data, pixels);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_padded_private_rgb8_ht_rpcl_512_encode_preserves_three_dwt_levels_and_stays_resident()
    {
        let pixels: Vec<u8> = (0..512 * 512 * 3)
            .map(|idx| ((idx * 47 + idx / 17) & 0xFF) as u8)
            .collect();
        let session = crate::MetalBackendSession::system_default().expect("Metal session");
        let buffer = private_buffer_with_bytes(&session, &pixels);

        let encoded = super::encode_lossless_from_padded_metal_buffer_to_metal_with_report(
            super::MetalLosslessEncodeTile {
                buffer: &buffer,
                byte_offset: 0,
                width: 512,
                height: 512,
                pitch_bytes: 512 * 3,
                output_width: 512,
                output_height: 512,
                format: PixelFormat::Rgb8,
            },
            &lossless_options! {
                backend: EncodeBackendPreference::RequireDevice,
                block_coding_mode: J2kBlockCodingMode::HighThroughput,
                progression: J2kProgressionOrder::Rpcl,
            },
            &session,
        )
        .expect("Metal private padded HTJ2K RPCL 512 buffer lossless encode");

        assert!(!encoded.input_copy_used);
        assert!(encoded.resident.coefficient_prep_used);
        assert!(encoded.resident.packetization_used);
        assert!(encoded.resident.codestream_assembly_used);
        let codestream = encoded
            .encoded
            .codestream_bytes()
            .expect("Metal codestream bytes are CPU-readable");
        let cod_marker = codestream
            .windows(2)
            .position(|window| window == [0xFF, 0x52])
            .expect("COD marker");
        assert_eq!(codestream[cod_marker + 5], 0x02);
        assert_eq!(codestream[cod_marker + 9], 3);
        assert_eq!(codestream[cod_marker + 12], 0x40);
        let decoded = Image::new(codestream, &DecodeSettings::default())
            .expect("codestream parses")
            .decode_native()
            .expect("codestream decodes");
        assert_eq!(decoded.data, pixels);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_rgb8_ht_batch_uses_fused_deinterleave_rct_kernel() {
        const WIDTH: usize = 32;
        const HEIGHT: usize = 32;
        let first: Vec<u8> = (0..WIDTH * HEIGHT * 3)
            .map(|idx| ((idx * 29 + idx / 7) & 0xFF) as u8)
            .collect();
        let second: Vec<u8> = (0..WIDTH * HEIGHT * 3)
            .map(|idx| 255u8.wrapping_sub(((idx * 13 + idx / 5) & 0xFF) as u8))
            .collect();
        let session = crate::MetalBackendSession::system_default().expect("Metal session");
        let first_buffer = private_buffer_with_bytes(&session, &first);
        let second_buffer = private_buffer_with_bytes(&session, &second);
        let tiles = [
            super::MetalLosslessEncodeTile {
                buffer: &first_buffer,
                byte_offset: 0,
                width: WIDTH as u32,
                height: HEIGHT as u32,
                pitch_bytes: WIDTH * 3,
                output_width: WIDTH as u32,
                output_height: HEIGHT as u32,
                format: PixelFormat::Rgb8,
            },
            super::MetalLosslessEncodeTile {
                buffer: &second_buffer,
                byte_offset: 0,
                width: WIDTH as u32,
                height: HEIGHT as u32,
                pitch_bytes: WIDTH * 3,
                output_width: WIDTH as u32,
                output_height: HEIGHT as u32,
                format: PixelFormat::Rgb8,
            },
        ];
        let options = lossless_options! {
            backend: EncodeBackendPreference::RequireDevice,
            block_coding_mode: J2kBlockCodingMode::HighThroughput,
            validation: J2kEncodeValidation::External,
        };

        compute::reset_lossless_deinterleave_rct_fused_dispatches_for_test();
        let encoded = super::encode_lossless_from_padded_metal_buffers_to_metal_with_report(
            &tiles, &options, &session,
        )
        .expect("Metal RGB8 HTJ2K batch encode");

        assert_eq!(encoded.len(), 2);
        assert!(
            compute::lossless_deinterleave_rct_fused_dispatches_for_test() > 0,
            "RGB8 resident lossless encode should fuse deinterleave and RCT"
        );
        for (frame, expected) in encoded.iter().zip([first, second]) {
            let codestream = frame
                .encoded
                .codestream_bytes()
                .expect("Metal codestream bytes are CPU-readable");
            let decoded = Image::new(codestream, &DecodeSettings::default())
                .expect("codestream parses")
                .decode_native()
                .expect("codestream decodes");
            assert_eq!(decoded.data, expected);
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_buffer_lossless_batch_encodes_padded_contiguous_inputs() {
        let first: Vec<u8> = (0..8 * 8 * 3).map(|i| ((i * 7) & 0xFF) as u8).collect();
        let second: Vec<u8> = (0..8 * 8 * 3)
            .map(|i| ((i * 13 + 5) & 0xFF) as u8)
            .collect();
        let session = crate::MetalBackendSession::system_default().expect("Metal session");
        let first_buffer = session.device().new_buffer_with_data(
            first.as_ptr().cast(),
            first.len() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let second_buffer = session.device().new_buffer_with_data(
            second.as_ptr().cast(),
            second.len() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let tiles = [
            super::MetalLosslessEncodeTile {
                buffer: &first_buffer,
                byte_offset: 0,
                width: 8,
                height: 8,
                pitch_bytes: 8 * 3,
                output_width: 8,
                output_height: 8,
                format: PixelFormat::Rgb8,
            },
            super::MetalLosslessEncodeTile {
                buffer: &second_buffer,
                byte_offset: 0,
                width: 8,
                height: 8,
                pitch_bytes: 8 * 3,
                output_width: 8,
                output_height: 8,
                format: PixelFormat::Rgb8,
            },
        ];

        let encoded = super::encode_lossless_from_padded_metal_buffers_with_report(
            &tiles,
            &lossless_options! {
                backend: EncodeBackendPreference::RequireDevice,
            },
            &session,
        )
        .expect("Metal padded buffer batch lossless encode");

        assert_eq!(encoded.len(), 2);
        for (frame, expected) in encoded.iter().zip([first, second]) {
            assert_eq!(frame.encoded.backend, BackendKind::Metal);
            assert!(!frame.input_copy_used);
            let decoded = Image::new(&frame.encoded.codestream, &DecodeSettings::default())
                .expect("codestream parses")
                .decode_native()
                .expect("codestream decodes");
            assert_eq!(decoded.data, expected);
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_padded_private_batch_encode_to_metal_buffers_exposes_per_frame_bytes() {
        let first: Vec<u8> = (0..8 * 8 * 3).map(|i| ((i * 17) & 0xFF) as u8).collect();
        let second: Vec<u8> = (0..8 * 8 * 3)
            .map(|i| 255u8.wrapping_sub(((i * 23) & 0xFF) as u8))
            .collect();
        let session = crate::MetalBackendSession::system_default().expect("Metal session");
        let first_buffer = private_buffer_with_bytes(&session, &first);
        let second_buffer = private_buffer_with_bytes(&session, &second);
        let tiles = [
            super::MetalLosslessEncodeTile {
                buffer: &first_buffer,
                byte_offset: 0,
                width: 8,
                height: 8,
                pitch_bytes: 8 * 3,
                output_width: 8,
                output_height: 8,
                format: PixelFormat::Rgb8,
            },
            super::MetalLosslessEncodeTile {
                buffer: &second_buffer,
                byte_offset: 0,
                width: 8,
                height: 8,
                pitch_bytes: 8 * 3,
                output_width: 8,
                output_height: 8,
                format: PixelFormat::Rgb8,
            },
        ];

        let encoded = super::encode_lossless_from_padded_metal_buffers_to_metal_with_report(
            &tiles,
            &lossless_options! {
                backend: EncodeBackendPreference::RequireDevice,
            },
            &session,
        )
        .expect("Metal padded buffer batch lossless encode to Metal buffers");

        assert_eq!(encoded.len(), 2);
        assert_eq!(
            encoded[0].encoded.codestream_buffer.as_ptr(),
            encoded[1].encoded.codestream_buffer.as_ptr(),
            "classic J2K resident batch encode should assemble codestreams into one shared batch buffer"
        );
        assert_eq!(encoded[0].encoded.byte_offset, 0);
        assert!(
            encoded[1].encoded.byte_offset > 0,
            "second classic J2K batch codestream should be a nonzero slice into the shared batch buffer"
        );
        for (frame, expected) in encoded.iter().zip([first, second]) {
            assert!(!frame.input_copy_used);
            assert!(frame.resident.coefficient_prep_used);
            assert!(frame.resident.packetization_used);
            assert!(frame.resident.codestream_assembly_used);
            let codestream = frame
                .encoded
                .codestream_bytes()
                .expect("Metal codestream bytes are CPU-readable");
            let decoded = Image::new(codestream, &DecodeSettings::default())
                .expect("codestream parses")
                .decode_native()
                .expect("codestream decodes");
            assert_eq!(decoded.data, expected);
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_padded_private_batch_dwt_encode_to_metal_buffers_round_trips() {
        let first: Vec<u8> = (0..128 * 128 * 3)
            .map(|i| ((i * 17 + i / 3) & 0xFF) as u8)
            .collect();
        let second: Vec<u8> = (0..128 * 128 * 3)
            .map(|i| 255u8.wrapping_sub(((i * 23 + i / 5) & 0xFF) as u8))
            .collect();
        let session = crate::MetalBackendSession::system_default().expect("Metal session");
        let first_buffer = private_buffer_with_bytes(&session, &first);
        let second_buffer = private_buffer_with_bytes(&session, &second);
        let tiles = [
            super::MetalLosslessEncodeTile {
                buffer: &first_buffer,
                byte_offset: 0,
                width: 128,
                height: 128,
                pitch_bytes: 128 * 3,
                output_width: 128,
                output_height: 128,
                format: PixelFormat::Rgb8,
            },
            super::MetalLosslessEncodeTile {
                buffer: &second_buffer,
                byte_offset: 0,
                width: 128,
                height: 128,
                pitch_bytes: 128 * 3,
                output_width: 128,
                output_height: 128,
                format: PixelFormat::Rgb8,
            },
        ];

        let encoded = super::encode_lossless_from_padded_metal_buffers_to_metal_with_report(
            &tiles,
            &lossless_options! {
                backend: EncodeBackendPreference::RequireDevice,
                validation: J2kEncodeValidation::External,
            },
            &session,
        )
        .expect("Metal padded DWT buffer batch lossless encode to Metal buffers");

        assert_eq!(encoded.len(), 2);
        for (frame, expected) in encoded.iter().zip([first, second]) {
            assert!(!frame.input_copy_used);
            assert!(frame.resident.coefficient_prep_used);
            assert!(frame.resident.packetization_used);
            assert!(frame.resident.codestream_assembly_used);
            let codestream = frame
                .encoded
                .codestream_bytes()
                .expect("Metal codestream bytes are CPU-readable");
            let decoded = Image::new(codestream, &DecodeSettings::default())
                .expect("codestream parses")
                .decode_native()
                .expect("codestream decodes");
            assert_decoded_bytes_match(&decoded.data, &expected);
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_edge_private_batch_encode_to_metal_buffers_stays_resident() {
        let first: Vec<u8> = (0..7 * 5 * 3).map(|i| ((i * 17) & 0xFF) as u8).collect();
        let second: Vec<u8> = (0..6 * 8 * 3)
            .map(|i| 255u8.wrapping_sub(((i * 19) & 0xFF) as u8))
            .collect();
        let session = crate::MetalBackendSession::system_default().expect("Metal session");
        let first_buffer = private_buffer_with_bytes(&session, &first);
        let second_buffer = private_buffer_with_bytes(&session, &second);
        compute::reset_ht_batch_coefficient_copy_blits_for_test();
        let tiles = [
            super::MetalLosslessEncodeTile {
                buffer: &first_buffer,
                byte_offset: 0,
                width: 7,
                height: 5,
                pitch_bytes: 7 * 3,
                output_width: 8,
                output_height: 8,
                format: PixelFormat::Rgb8,
            },
            super::MetalLosslessEncodeTile {
                buffer: &second_buffer,
                byte_offset: 0,
                width: 6,
                height: 8,
                pitch_bytes: 6 * 3,
                output_width: 8,
                output_height: 8,
                format: PixelFormat::Rgb8,
            },
        ];

        let encoded = super::encode_lossless_from_metal_buffers_to_metal_with_report(
            &tiles,
            &lossless_options! {
                backend: EncodeBackendPreference::RequireDevice,
            },
            &session,
        )
        .expect("Metal edge buffer batch lossless encode to Metal buffers");

        assert_eq!(encoded.len(), 2);
        for frame in &encoded {
            assert!(!frame.input_copy_used);
            assert!(frame.resident.coefficient_prep_used);
            assert!(frame.resident.packetization_used);
            assert!(frame.resident.codestream_assembly_used);
        }

        for (frame, (expected, width, height)) in encoded
            .iter()
            .zip([(first, 7usize, 5usize), (second, 6usize, 8usize)])
        {
            let codestream = frame
                .encoded
                .codestream_bytes()
                .expect("Metal codestream bytes are CPU-readable");
            let decoded = Image::new(codestream, &DecodeSettings::default())
                .expect("codestream parses")
                .decode_native()
                .expect("codestream decodes");
            for y in 0..8usize {
                for x in 0..8usize {
                    let dst = (y * 8 + x) * 3;
                    if x < width && y < height {
                        let src = (y * width + x) * 3;
                        assert_eq!(&decoded.data[dst..dst + 3], &expected[src..src + 3]);
                    } else {
                        assert_eq!(&decoded.data[dst..dst + 3], &[0, 0, 0]);
                    }
                }
            }
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_ht_private_batch_encode_to_metal_buffers_stays_resident() {
        let first: Vec<u8> = (0..8 * 8).map(|i| ((i * 11) & 0xFF) as u8).collect();
        let second: Vec<u8> = (0..8 * 8)
            .map(|i| 255u8.wrapping_sub(((i * 13) & 0xFF) as u8))
            .collect();
        let session = crate::MetalBackendSession::system_default().expect("Metal session");
        let first_buffer = private_buffer_with_bytes(&session, &first);
        let second_buffer = private_buffer_with_bytes(&session, &second);
        let tiles = [
            super::MetalLosslessEncodeTile {
                buffer: &first_buffer,
                byte_offset: 0,
                width: 8,
                height: 8,
                pitch_bytes: 8,
                output_width: 8,
                output_height: 8,
                format: PixelFormat::Gray8,
            },
            super::MetalLosslessEncodeTile {
                buffer: &second_buffer,
                byte_offset: 0,
                width: 8,
                height: 8,
                pitch_bytes: 8,
                output_width: 8,
                output_height: 8,
                format: PixelFormat::Gray8,
            },
        ];

        compute::reset_resident_gpu_timestamp_queries_for_test();
        let encoded = super::encode_lossless_from_padded_metal_buffers_to_metal_with_report(
            &tiles,
            &lossless_options! {
                backend: EncodeBackendPreference::RequireDevice,
                block_coding_mode: J2kBlockCodingMode::HighThroughput,
            },
            &session,
        )
        .expect("Metal HTJ2K batch lossless encode to Metal buffers");

        assert_eq!(encoded.len(), 2);
        assert_eq!(
            compute::ht_batch_coefficient_copy_blits_for_test(),
            0,
            "HTJ2K resident batch prep should write directly into the batch coefficient buffer"
        );
        assert_eq!(
            compute::resident_gpu_timestamp_queries_for_test(),
            7,
            "HTJ2K resident batch should query each unique retained command buffer timestamp once"
        );
        assert_eq!(
            encoded[0].encoded.codestream_buffer.as_ptr(),
            encoded[1].encoded.codestream_buffer.as_ptr(),
            "HTJ2K resident batch encode should assemble codestreams into one shared batch buffer"
        );
        assert_eq!(encoded[0].encoded.byte_offset, 0);
        assert!(
            encoded[1].encoded.byte_offset > 0,
            "second HTJ2K batch codestream should be a nonzero slice into the shared batch buffer"
        );
        for (frame, expected) in encoded.iter().zip([first, second]) {
            assert!(!frame.input_copy_used);
            assert!(frame.resident.coefficient_prep_used);
            assert!(frame.resident.packetization_used);
            assert!(frame.resident.codestream_assembly_used);
            let codestream = frame
                .encoded
                .codestream_bytes()
                .expect("Metal codestream bytes are CPU-readable");
            assert!(codestream.windows(2).any(|window| window == [0xFF, 0x50]));
            let decoded = Image::new(codestream, &DecodeSettings::default())
                .expect("codestream parses")
                .decode_native()
                .expect("codestream decodes");
            assert_eq!(decoded.data, expected);
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_ht_private_batch_encode_reuses_private_arenas_between_batches() {
        const WIDTH: usize = 37;
        const HEIGHT: usize = 41;
        let first: Vec<u8> = (0..WIDTH * HEIGHT)
            .map(|i| ((i * 7 + 3) & 0xFF) as u8)
            .collect();
        let second: Vec<u8> = (0..WIDTH * HEIGHT)
            .map(|i| 255u8.wrapping_sub(((i * 5 + 11) & 0xFF) as u8))
            .collect();
        let device = metal::Device::system_default().expect("Metal device");
        let session = crate::MetalBackendSession::new(device.clone());
        let first_buffer = private_buffer_with_bytes(&session, &first);
        let second_buffer = private_buffer_with_bytes(&session, &second);
        let tiles = [
            super::MetalLosslessEncodeTile {
                buffer: &first_buffer,
                byte_offset: 0,
                width: WIDTH as u32,
                height: HEIGHT as u32,
                pitch_bytes: WIDTH,
                output_width: WIDTH as u32,
                output_height: HEIGHT as u32,
                format: PixelFormat::Gray8,
            },
            super::MetalLosslessEncodeTile {
                buffer: &second_buffer,
                byte_offset: 0,
                width: WIDTH as u32,
                height: HEIGHT as u32,
                pitch_bytes: WIDTH,
                output_width: WIDTH as u32,
                output_height: HEIGHT as u32,
                format: PixelFormat::Gray8,
            },
        ];
        let options = lossless_options! {
            backend: EncodeBackendPreference::RequireDevice,
            block_coding_mode: J2kBlockCodingMode::HighThroughput,
            validation: J2kEncodeValidation::External,
        };

        compute::with_isolated_runtime_for_device_for_test(&device, || {
            compute::reset_private_buffer_pool_misses_for_test();
            super::encode_lossless_from_padded_metal_buffers_to_metal_with_report(
                &tiles, &options, &session,
            )?;
            let first_misses = compute::private_buffer_pool_misses_for_test();
            assert!(
                first_misses > 0,
                "first unique HTJ2K batch should populate reusable private arenas"
            );

            compute::reset_private_buffer_pool_misses_for_test();
            let encoded = super::encode_lossless_from_padded_metal_buffers_to_metal_with_report(
                &tiles, &options, &session,
            )?;

            assert_eq!(
                compute::private_buffer_pool_misses_for_test(),
                0,
                "second same-shape HTJ2K batch should reuse private arenas"
            );
            assert_eq!(encoded.len(), 2);
            Ok(())
        })
        .expect("isolated HTJ2K Metal runtime");
    }

    #[test]
    fn default_gpu_encode_memory_budget_uses_forty_percent_capped_at_ten_gib() {
        const GIB: usize = 1024 * 1024 * 1024;

        assert_eq!(
            super::default_gpu_encode_memory_budget_bytes_for_hw_mem(8 * GIB),
            8 * GIB * 40 / 100
        );
        assert_eq!(
            super::default_gpu_encode_memory_budget_bytes_for_hw_mem(16 * GIB),
            16 * GIB * 40 / 100
        );
        assert_eq!(
            super::default_gpu_encode_memory_budget_bytes_for_hw_mem(24 * GIB),
            24 * GIB * 40 / 100
        );
        assert_eq!(
            super::default_gpu_encode_memory_budget_bytes_for_hw_mem(64 * GIB),
            10 * GIB
        );
    }

    #[test]
    fn gpu_encode_inflight_resolution_clamps_requested_tiles_by_memory_budget() {
        let stats = super::resolve_lossless_encode_config_for_test(
            100,
            1_000,
            super::MetalLosslessEncodeConfig {
                gpu_encode_inflight_tiles: Some(32),
                gpu_encode_memory_budget_bytes: Some(4_500),
            },
        )
        .expect("resolved config");

        assert_eq!(stats.configured_inflight_tiles, Some(32));
        assert_eq!(stats.effective_inflight_tiles, 4);
        assert_eq!(stats.configured_memory_budget_bytes, Some(4_500));
        assert_eq!(stats.effective_memory_budget_bytes, 4_500);
        assert_eq!(stats.estimated_peak_bytes_per_tile, 1_000);
    }

    #[test]
    fn gpu_encode_default_inflight_uses_large_wsi_batch_when_memory_allows() {
        let stats = super::resolve_lossless_encode_config_for_test(
            600,
            1_000,
            super::MetalLosslessEncodeConfig {
                gpu_encode_inflight_tiles: None,
                gpu_encode_memory_budget_bytes: Some(1_000_000),
            },
        )
        .expect("resolved config");

        assert_eq!(stats.configured_inflight_tiles, None);
        assert_eq!(stats.effective_inflight_tiles, 512);
    }

    #[test]
    fn resident_classic_encode_default_inflight_uses_profiled_cap() {
        let config = super::resident_lossless_encode_config_for_mode(
            super::MetalLosslessEncodeConfig {
                gpu_encode_inflight_tiles: None,
                gpu_encode_memory_budget_bytes: Some(1_000_000),
            },
            true,
            16,
        );

        assert_eq!(config.gpu_encode_inflight_tiles, Some(16));
        assert_eq!(config.gpu_encode_memory_budget_bytes, Some(1_000_000));
    }

    #[test]
    fn resident_classic_encode_default_inflight_uses_large_batch_cap() {
        let config = super::resident_lossless_encode_config_for_mode(
            super::MetalLosslessEncodeConfig {
                gpu_encode_inflight_tiles: None,
                gpu_encode_memory_budget_bytes: Some(1_000_000),
            },
            true,
            64,
        );

        assert_eq!(config.gpu_encode_inflight_tiles, Some(64));
        assert_eq!(config.gpu_encode_memory_budget_bytes, Some(1_000_000));
    }

    #[test]
    fn resident_classic_encode_default_inflight_uses_very_large_batch_cap() {
        let config = super::resident_lossless_encode_config_for_mode(
            super::MetalLosslessEncodeConfig {
                gpu_encode_inflight_tiles: None,
                gpu_encode_memory_budget_bytes: Some(1_000_000),
            },
            true,
            128,
        );

        assert_eq!(config.gpu_encode_inflight_tiles, Some(128));
        assert_eq!(config.gpu_encode_memory_budget_bytes, Some(1_000_000));
    }

    #[test]
    fn resident_htj2k_encode_medium_batch_default_inflight_uses_profiled_cap() {
        let config = super::resident_lossless_encode_config_for_mode(
            super::MetalLosslessEncodeConfig {
                gpu_encode_inflight_tiles: None,
                gpu_encode_memory_budget_bytes: Some(1_000_000),
            },
            false,
            64,
        );

        assert_eq!(config.gpu_encode_inflight_tiles, Some(32));
        assert_eq!(config.gpu_encode_memory_budget_bytes, Some(1_000_000));
    }

    #[test]
    fn resident_htj2k_encode_large_batch_default_inflight_uses_profiled_cap() {
        let config = super::resident_lossless_encode_config_for_mode(
            super::MetalLosslessEncodeConfig {
                gpu_encode_inflight_tiles: None,
                gpu_encode_memory_budget_bytes: Some(1_000_000),
            },
            false,
            128,
        );

        assert_eq!(config.gpu_encode_inflight_tiles, Some(64));
        assert_eq!(config.gpu_encode_memory_budget_bytes, Some(1_000_000));
    }

    #[test]
    fn gpu_encode_inflight_resolution_rejects_zero_overrides() {
        let err = super::resolve_lossless_encode_config_for_test(
            4,
            1_000,
            super::MetalLosslessEncodeConfig {
                gpu_encode_inflight_tiles: Some(0),
                gpu_encode_memory_budget_bytes: Some(4_000),
            },
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("in-flight"),
            "unexpected error: {err}"
        );

        let err = super::resolve_lossless_encode_config_for_test(
            4,
            1_000,
            super::MetalLosslessEncodeConfig {
                gpu_encode_inflight_tiles: Some(2),
                gpu_encode_memory_budget_bytes: Some(0),
            },
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("memory budget"),
            "unexpected error: {err}"
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_ht_batch_encode_preserves_order_and_matches_inflight_one() {
        let inputs = [
            (0..8 * 8)
                .map(|i| ((i * 11 + 3) & 0xFF) as u8)
                .collect::<Vec<_>>(),
            (0..8 * 8)
                .map(|i| ((i * 13 + 5) & 0xFF) as u8)
                .collect::<Vec<_>>(),
            (0..8 * 8)
                .map(|i| ((i * 17 + 7) & 0xFF) as u8)
                .collect::<Vec<_>>(),
            (0..8 * 8)
                .map(|i| ((i * 19 + 9) & 0xFF) as u8)
                .collect::<Vec<_>>(),
        ];
        let session = crate::MetalBackendSession::system_default().expect("Metal session");
        let buffers = inputs
            .iter()
            .map(|bytes| private_buffer_with_bytes(&session, bytes))
            .collect::<Vec<_>>();
        let tiles = buffers
            .iter()
            .map(|buffer| super::MetalLosslessEncodeTile {
                buffer,
                byte_offset: 0,
                width: 8,
                height: 8,
                pitch_bytes: 8,
                output_width: 8,
                output_height: 8,
                format: PixelFormat::Gray8,
            })
            .collect::<Vec<_>>();
        let options = lossless_options! {
            backend: EncodeBackendPreference::RequireDevice,
            block_coding_mode: J2kBlockCodingMode::HighThroughput,
            validation: J2kEncodeValidation::External,
        };

        compute::reset_resident_codestream_command_buffer_waits_for_test();
        let serial = super::encode_lossless_from_padded_metal_buffers_to_metal_batch(
            &tiles,
            &options,
            &session,
            super::MetalLosslessEncodeConfig {
                gpu_encode_inflight_tiles: Some(1),
                gpu_encode_memory_budget_bytes: Some(1024 * 1024 * 1024),
            },
        )
        .expect("serial Metal HTJ2K batch");
        assert_eq!(
            compute::resident_codestream_command_buffer_waits_for_test(),
            1,
            "multi-chunk HT batch should wait once before harvesting completed chunks"
        );

        let cpu_validated_options = lossless_options! {
            backend: EncodeBackendPreference::RequireDevice,
            block_coding_mode: J2kBlockCodingMode::HighThroughput,
            validation: J2kEncodeValidation::CpuRoundTrip,
        };
        compute::reset_resident_codestream_command_buffer_waits_for_test();
        let cpu_validated = super::encode_lossless_from_padded_metal_buffers_to_metal_batch(
            &tiles,
            &cpu_validated_options,
            &session,
            super::MetalLosslessEncodeConfig {
                gpu_encode_inflight_tiles: Some(1),
                gpu_encode_memory_budget_bytes: Some(1024 * 1024 * 1024),
            },
        )
        .expect("CPU-validated Metal HTJ2K batch");
        assert_eq!(cpu_validated.outcomes.len(), inputs.len());
        assert_eq!(
            compute::resident_codestream_command_buffer_waits_for_test(),
            inputs.len(),
            "CPU roundtrip validation should keep per-chunk waits to preserve overlap"
        );

        let parallel = super::encode_lossless_from_padded_metal_buffers_to_metal_batch(
            &tiles,
            &options,
            &session,
            super::MetalLosslessEncodeConfig {
                gpu_encode_inflight_tiles: Some(2),
                gpu_encode_memory_budget_bytes: Some(1024 * 1024 * 1024),
            },
        )
        .expect("parallel Metal HTJ2K batch");
        let repeated_parallel = super::encode_lossless_from_padded_metal_buffers_to_metal_batch(
            &tiles,
            &options,
            &session,
            super::MetalLosslessEncodeConfig {
                gpu_encode_inflight_tiles: Some(2),
                gpu_encode_memory_budget_bytes: Some(1024 * 1024 * 1024),
            },
        )
        .expect("repeated parallel Metal HTJ2K batch");

        assert_eq!(serial.outcomes.len(), inputs.len());
        assert_eq!(parallel.outcomes.len(), inputs.len());
        assert_eq!(parallel.stats.effective_inflight_tiles, 2);
        assert!(parallel.stats.max_observed_inflight_tiles <= 2);
        assert!(parallel.stats.max_observed_inflight_tiles > 0);
        for (((serial_outcome, parallel_outcome), repeated_outcome), expected) in serial
            .outcomes
            .iter()
            .zip(parallel.outcomes.iter())
            .zip(repeated_parallel.outcomes.iter())
            .zip(inputs.iter())
        {
            let serial_bytes = serial_outcome
                .encoded
                .codestream_bytes()
                .expect("serial codestream");
            let parallel_bytes = parallel_outcome
                .encoded
                .codestream_bytes()
                .expect("parallel codestream");
            let repeated_bytes = repeated_outcome
                .encoded
                .codestream_bytes()
                .expect("repeated parallel codestream");
            assert_eq!(parallel_bytes, serial_bytes);
            assert_eq!(repeated_bytes, serial_bytes);

            let decoded = Image::new(parallel_bytes, &DecodeSettings::default())
                .expect("codestream parses")
                .decode_native()
                .expect("codestream decodes");
            assert_eq!(&decoded.data, expected);
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_parallel_batch_returns_indexed_injected_failure() {
        let first: Vec<u8> = (0..8 * 8).map(|i| ((i * 3) & 0xFF) as u8).collect();
        let second: Vec<u8> = (0..8 * 8).map(|i| ((i * 5) & 0xFF) as u8).collect();
        let session = crate::MetalBackendSession::system_default().expect("Metal session");
        let first_buffer = private_buffer_with_bytes(&session, &first);
        let second_buffer = private_buffer_with_bytes(&session, &second);
        let tiles = [
            super::MetalLosslessEncodeTile {
                buffer: &first_buffer,
                byte_offset: 0,
                width: 8,
                height: 8,
                pitch_bytes: 8,
                output_width: 8,
                output_height: 8,
                format: PixelFormat::Gray8,
            },
            super::MetalLosslessEncodeTile {
                buffer: &second_buffer,
                byte_offset: 0,
                width: 8,
                height: 8,
                pitch_bytes: 8,
                output_width: 8,
                output_height: 8,
                format: PixelFormat::Gray8,
            },
        ];
        let options = lossless_options! {
            backend: EncodeBackendPreference::RequireDevice,
            block_coding_mode: J2kBlockCodingMode::HighThroughput,
            validation: J2kEncodeValidation::External,
        };

        super::set_test_resident_encode_failure_index(Some(1));
        let Err(err) = super::encode_lossless_from_padded_metal_buffers_to_metal_batch(
            &tiles,
            &options,
            &session,
            super::MetalLosslessEncodeConfig {
                gpu_encode_inflight_tiles: Some(2),
                gpu_encode_memory_budget_bytes: Some(1024 * 1024 * 1024),
            },
        ) else {
            panic!("injected failure should fail the batch");
        };
        super::set_test_resident_encode_failure_index(None);

        assert!(
            err.to_string().contains("tile 1"),
            "unexpected error: {err}"
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_forward_dwt53_dispatch_round_trips_gray8_lossless_tile() {
        let pixels: Vec<u8> = (0..64 * 64).map(|i| ((i * 5) & 0xFF) as u8).collect();
        let samples =
            J2kLosslessSamples::new(&pixels, 64, 64, 1, 8, false).expect("valid gray samples");
        let options = J2kLosslessEncodeOptions::default()
            .with_backend(EncodeBackendPreference::RequireDevice)
            .with_max_decomposition_levels(Some(1));
        let mut accelerator = MetalEncodeStageAccelerator::default();

        let encoded = encode_j2k_lossless_with_accelerator(
            samples,
            &options,
            BackendKind::Metal,
            &mut accelerator,
        )
        .expect("encode with metal forward DWT 5/3");
        let decoded = Image::new(&encoded.codestream, &DecodeSettings::default())
            .expect("codestream parses")
            .decode_native()
            .expect("codestream decodes");

        assert_eq!(decoded.data, pixels);
        assert_eq!(accelerator.forward_dwt53_attempts(), 1);
        assert_eq!(accelerator.forward_dwt53_dispatches(), 1);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_lossless_facade_dispatches_rct_and_dwt_for_wsi_sized_rgb_tile() {
        let mut pixels = Vec::with_capacity(128 * 128 * 3);
        for y in 0..128u32 {
            for x in 0..128u32 {
                pixels.push(((x * 3 + y * 5) & 0xFF) as u8);
                pixels.push(((x * 7 + y * 11) & 0xFF) as u8);
                pixels.push(((x * 13 + y * 17) & 0xFF) as u8);
            }
        }
        let samples =
            J2kLosslessSamples::new(&pixels, 128, 128, 3, 8, false).expect("valid RGB samples");
        let mut accelerator = MetalEncodeStageAccelerator::default();

        let encoded = encode_j2k_lossless_with_accelerator(
            samples,
            &lossless_options! {
                backend: EncodeBackendPreference::PreferDevice,
            },
            BackendKind::Metal,
            &mut accelerator,
        )
        .expect("Metal-accelerated lossless encode");

        assert_eq!(encoded.backend, BackendKind::Metal);
        assert_eq!(accelerator.forward_rct_dispatches(), 1);
        assert_eq!(accelerator.forward_dwt53_dispatches(), 3);
        assert!(accelerator.tier1_code_block_attempts() > 0);
        assert_eq!(accelerator.packetization_attempts(), 1);
        assert!(accelerator.tier1_code_block_dispatches() > 0);
        assert_eq!(accelerator.packetization_dispatches(), 1);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_classic_tier1_uses_one_batched_dispatch_for_multiple_code_blocks() {
        let pixels: Vec<u8> = (0..256 * 256)
            .map(|idx| ((idx * 17 + 3) & 0xFF) as u8)
            .collect();
        let samples =
            J2kLosslessSamples::new(&pixels, 256, 256, 1, 8, false).expect("valid gray samples");
        let options = J2kLosslessEncodeOptions::default()
            .with_backend(EncodeBackendPreference::RequireDevice)
            .with_max_decomposition_levels(Some(0));
        let mut accelerator = MetalEncodeStageAccelerator::default();

        let encoded = encode_j2k_lossless_with_accelerator(
            samples,
            &options,
            BackendKind::Metal,
            &mut accelerator,
        )
        .expect("encode with batched Metal classic Tier-1");
        let decoded = Image::new(&encoded.codestream, &DecodeSettings::default())
            .expect("codestream parses")
            .decode_native()
            .expect("codestream decodes");

        assert_eq!(decoded.data, pixels);
        assert!(accelerator.tier1_code_block_attempts() > 1);
        assert_eq!(accelerator.tier1_code_block_dispatches(), 1);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_classic_resident_uses_mq_byte_split_gpu_token_pack_by_default() {
        let _profile_guard = compute::force_metal_profile_stages_for_test(true);
        compute::reset_classic_gpu_token_pack_dispatches_for_test();
        compute::reset_classic_split_mq_byte_gpu_token_pack_dispatches_for_test();
        let first: Vec<u8> = (0..256 * 256)
            .map(|idx| {
                let x = idx % 256;
                let y = idx / 256;
                ((x + y * 5) & 0xFF) as u8
            })
            .collect();
        let second: Vec<u8> = (0..256 * 256)
            .map(|idx| {
                let x = idx % 256;
                let y = idx / 256;
                ((x * 3 + y * 7) & 0xFF) as u8
            })
            .collect();
        let session = crate::MetalBackendSession::system_default().expect("Metal session");
        let first_buffer = private_buffer_with_bytes(&session, &first);
        let second_buffer = private_buffer_with_bytes(&session, &second);
        let tiles = [
            super::MetalLosslessEncodeTile {
                buffer: &first_buffer,
                byte_offset: 0,
                width: 256,
                height: 256,
                pitch_bytes: 256,
                output_width: 256,
                output_height: 256,
                format: PixelFormat::Gray8,
            },
            super::MetalLosslessEncodeTile {
                buffer: &second_buffer,
                byte_offset: 0,
                width: 256,
                height: 256,
                pitch_bytes: 256,
                output_width: 256,
                output_height: 256,
                format: PixelFormat::Gray8,
            },
        ];

        let encoded = super::encode_lossless_from_padded_metal_buffers_to_metal_batch(
            &tiles,
            &lossless_options! {
                backend: EncodeBackendPreference::RequireDevice,
                block_coding_mode: J2kBlockCodingMode::Classic,
                validation: J2kEncodeValidation::External,
            },
            &session,
            super::MetalLosslessEncodeConfig {
                gpu_encode_inflight_tiles: Some(2),
                gpu_encode_memory_budget_bytes: Some(1024 * 1024 * 1024),
            },
        )
        .expect("resident batch encode with default MQ-byte GPU token-pack Classic Tier-1");
        assert_eq!(encoded.outcomes.len(), 2);
        for (outcome, expected) in encoded.outcomes.iter().zip([&first, &second]) {
            let codestream = outcome
                .encoded
                .to_encoded_j2k()
                .expect("codestream readback");
            let decoded = Image::new(&codestream.codestream, &DecodeSettings::default())
                .expect("codestream parses")
                .decode_native()
                .expect("codestream decodes");

            assert_eq!(codestream.backend, BackendKind::Metal);
            assert_eq!(&decoded.data, expected);
        }
        assert!(
            compute::classic_gpu_token_pack_dispatches_for_test() > 0,
            "default Classic GPU token-pack route was not dispatched"
        );
        assert!(
            compute::classic_split_mq_byte_gpu_token_pack_dispatches_for_test() > 0,
            "default Classic GPU token-pack route did not use MQ-byte split token emit"
        );
        assert_eq!(
            encoded
                .stats
                .stage_stats
                .tier1_token_pack_output_bytes_total,
            encoded.stats.stage_stats.tier1_output_used_bytes_total,
            "default Classic GPU token-pack route should attribute Tier-1 output bytes to token pack"
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_classic_resident_gpu_token_pack_route_round_trips() {
        let _guard = compute::force_classic_gpu_token_pack_route_for_test(true);
        let _profile_guard = compute::force_metal_profile_stages_for_test(true);
        compute::reset_classic_gpu_token_pack_dispatches_for_test();
        let first: Vec<u8> = (0..256 * 256)
            .map(|idx| {
                let x = idx % 256;
                let y = idx / 256;
                ((x + y * 3) & 0xFF) as u8
            })
            .collect();
        let second: Vec<u8> = (0..256 * 256)
            .map(|idx| {
                let x = idx % 256;
                let y = idx / 256;
                ((x * 2 + y) & 0xFF) as u8
            })
            .collect();
        let session = crate::MetalBackendSession::system_default().expect("Metal session");
        let first_buffer = private_buffer_with_bytes(&session, &first);
        let second_buffer = private_buffer_with_bytes(&session, &second);
        let tiles = [
            super::MetalLosslessEncodeTile {
                buffer: &first_buffer,
                byte_offset: 0,
                width: 256,
                height: 256,
                pitch_bytes: 256,
                output_width: 256,
                output_height: 256,
                format: PixelFormat::Gray8,
            },
            super::MetalLosslessEncodeTile {
                buffer: &second_buffer,
                byte_offset: 0,
                width: 256,
                height: 256,
                pitch_bytes: 256,
                output_width: 256,
                output_height: 256,
                format: PixelFormat::Gray8,
            },
        ];

        let encoded = super::encode_lossless_from_padded_metal_buffers_to_metal_batch(
            &tiles,
            &lossless_options! {
                backend: EncodeBackendPreference::RequireDevice,
                block_coding_mode: J2kBlockCodingMode::Classic,
                validation: J2kEncodeValidation::External,
            },
            &session,
            super::MetalLosslessEncodeConfig {
                gpu_encode_inflight_tiles: Some(2),
                gpu_encode_memory_budget_bytes: Some(1024 * 1024 * 1024),
            },
        )
        .expect("resident batch encode with gated GPU token-pack Classic Tier-1");
        assert_eq!(encoded.outcomes.len(), 2);
        for (outcome, expected) in encoded.outcomes.iter().zip([&first, &second]) {
            let codestream = outcome
                .encoded
                .to_encoded_j2k()
                .expect("codestream readback");
            let decoded = Image::new(&codestream.codestream, &DecodeSettings::default())
                .expect("codestream parses")
                .decode_native()
                .expect("codestream decodes");

            assert_eq!(codestream.backend, BackendKind::Metal);
            assert_eq!(&decoded.data, expected);
        }
        assert!(
            compute::classic_gpu_token_pack_dispatches_for_test() > 0,
            "gated Classic GPU token-pack route was not dispatched"
        );
        assert!(
            encoded.stats.stage_stats.tier1_token_emit_token_bytes_total > 0,
            "gated Classic GPU token-pack route did not expose token-emitter byte counters"
        );
        assert!(
            encoded
                .stats
                .stage_stats
                .tier1_token_emit_segment_count_total
                > 0,
            "gated Classic GPU token-pack route did not expose token segment counters"
        );
        assert_eq!(
            encoded
                .stats
                .stage_stats
                .tier1_token_pack_output_bytes_total,
            encoded.stats.stage_stats.tier1_output_used_bytes_total,
            "gated Classic GPU token-pack route should attribute Tier-1 output bytes to token pack"
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_htj2k_uses_one_batched_dispatch_for_multiple_code_blocks() {
        let pixels: Vec<u8> = (0..256 * 256)
            .map(|idx| ((idx * 23 + 9) & 0xFF) as u8)
            .collect();
        let samples =
            J2kLosslessSamples::new(&pixels, 256, 256, 1, 8, false).expect("valid gray samples");
        let mut accelerator = MetalEncodeStageAccelerator::default();

        let encoded = encode_j2k_lossless_with_accelerator(
            samples,
            &lossless_options! {
                backend: EncodeBackendPreference::RequireDevice,
                block_coding_mode: J2kBlockCodingMode::HighThroughput,
            },
            BackendKind::Metal,
            &mut accelerator,
        )
        .expect("Metal-accelerated HTJ2K lossless encode");

        assert_eq!(encoded.backend, BackendKind::Metal);
        assert!(accelerator.ht_code_block_attempts() > 1);
        assert_eq!(accelerator.ht_code_block_dispatches(), 1);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_htj2k_lossless_facade_dispatches_ht_code_blocks_and_packetization() {
        let pixels: Vec<u8> = (0..64).map(|value| ((value * 13) & 0xFF) as u8).collect();
        let samples =
            J2kLosslessSamples::new(&pixels, 8, 8, 1, 8, false).expect("valid gray samples");
        let mut accelerator = MetalEncodeStageAccelerator::default();

        let encoded = encode_j2k_lossless_with_accelerator(
            samples,
            &lossless_options! {
                backend: EncodeBackendPreference::RequireDevice,
                block_coding_mode: J2kBlockCodingMode::HighThroughput,
            },
            BackendKind::Metal,
            &mut accelerator,
        )
        .expect("Metal-accelerated HTJ2K lossless encode");

        assert_eq!(encoded.backend, BackendKind::Metal);
        assert!(accelerator.ht_code_block_attempts() > 0);
        assert!(accelerator.ht_code_block_dispatches() > 0);
        assert_eq!(accelerator.packetization_attempts(), 1);
        assert_eq!(accelerator.packetization_dispatches(), 1);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_htj2k_lossy_facade_require_device_dispatches_supported_stages() {
        let pixels: Vec<u8> = (0..16 * 16)
            .map(|idx| ((idx * 17 + idx / 3) & 0xFF) as u8)
            .collect();
        let samples =
            J2kLossySamples::new(&pixels, 16, 16, 1, 8, false).expect("valid gray samples");
        let mut accelerator = MetalEncodeStageAccelerator::default();

        let encoded = encode_j2k_lossy_with_accelerator(
            samples,
            &J2kLossyEncodeOptions::default()
                .with_backend(EncodeBackendPreference::RequireDevice)
                .with_block_coding_mode(J2kBlockCodingMode::HighThroughput)
                .with_max_decomposition_levels(Some(0))
                .with_validation(J2kEncodeValidation::CpuRoundTrip),
            BackendKind::Metal,
            &mut accelerator,
        )
        .expect("Metal-accelerated HTJ2K lossy encode");

        assert_eq!(encoded.backend, BackendKind::Metal);
        assert!(accelerator.ht_code_block_attempts() > 0);
        assert!(accelerator.ht_code_block_dispatches() > 0);
        assert_eq!(accelerator.packetization_attempts(), 1);
        assert_eq!(accelerator.packetization_dispatches(), 1);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_classic_tier1_kernel_matches_scalar_oracle() {
        let coeffs: Vec<i32> = (0..64)
            .map(|idx| {
                let value = ((idx * 37 + 11) & 0x1ff) - 255;
                if idx % 5 == 0 {
                    0
                } else {
                    value
                }
            })
            .collect();
        let style = J2kCodeBlockStyle {
            selective_arithmetic_coding_bypass: false,
            reset_context_probabilities: false,
            termination_on_each_pass: false,
            vertically_causal_context: false,
            segmentation_symbols: false,
        };
        let job = signinum_j2k_native::J2kTier1CodeBlockEncodeJob {
            coefficients: &coeffs,
            width: 8,
            height: 8,
            sub_band_type: signinum_j2k_native::J2kSubBandType::HighHigh,
            total_bitplanes: 9,
            style,
        };

        let gpu = compute::encode_classic_tier1_code_block(job).expect("Metal classic encode");
        let cpu = signinum_j2k_native::encode_j2k_code_block_scalar_with_style(
            &coeffs,
            8,
            8,
            signinum_j2k_native::J2kSubBandType::HighHigh,
            9,
            style,
        )
        .expect("scalar classic encode");

        assert_eq!(gpu.data, cpu.data);
        assert_eq!(gpu.segments.len(), cpu.segments.len());
        for (gpu_segment, cpu_segment) in gpu.segments.iter().zip(cpu.segments.iter()) {
            assert_eq!(gpu_segment.data_offset, cpu_segment.data_offset);
            assert_eq!(gpu_segment.data_length, cpu_segment.data_length);
            assert_eq!(gpu_segment.start_coding_pass, cpu_segment.start_coding_pass);
            assert_eq!(gpu_segment.end_coding_pass, cpu_segment.end_coding_pass);
            assert_eq!(gpu_segment.use_arithmetic, cpu_segment.use_arithmetic);
        }
        assert_eq!(gpu.number_of_coding_passes, cpu.number_of_coding_passes);
        assert_eq!(gpu.missing_bit_planes, cpu.missing_bit_planes);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_classic_tier1_kernel_matches_scalar_for_terminated_passes() {
        let coeffs: Vec<i32> = (0..64)
            .map(|idx| {
                let value = ((idx * 43 + 5) & 0x3ff) - 511;
                if idx % 6 == 0 {
                    0
                } else {
                    value
                }
            })
            .collect();
        let style = J2kCodeBlockStyle {
            selective_arithmetic_coding_bypass: false,
            reset_context_probabilities: true,
            termination_on_each_pass: true,
            vertically_causal_context: false,
            segmentation_symbols: true,
        };
        let job = signinum_j2k_native::J2kTier1CodeBlockEncodeJob {
            coefficients: &coeffs,
            width: 8,
            height: 8,
            sub_band_type: signinum_j2k_native::J2kSubBandType::LowHigh,
            total_bitplanes: 10,
            style,
        };

        let gpu =
            compute::encode_classic_tier1_code_block(job).expect("Metal classic terminated encode");
        let cpu = signinum_j2k_native::encode_j2k_code_block_scalar_with_style(
            &coeffs,
            8,
            8,
            signinum_j2k_native::J2kSubBandType::LowHigh,
            10,
            style,
        )
        .expect("scalar classic terminated encode");

        assert_eq!(gpu.data, cpu.data);
        assert_eq!(gpu.segments.len(), cpu.segments.len());
        for (gpu_segment, cpu_segment) in gpu.segments.iter().zip(cpu.segments.iter()) {
            assert_eq!(gpu_segment.data_offset, cpu_segment.data_offset);
            assert_eq!(gpu_segment.data_length, cpu_segment.data_length);
            assert_eq!(gpu_segment.start_coding_pass, cpu_segment.start_coding_pass);
            assert_eq!(gpu_segment.end_coding_pass, cpu_segment.end_coding_pass);
            assert_eq!(gpu_segment.use_arithmetic, cpu_segment.use_arithmetic);
        }
        assert_eq!(gpu.number_of_coding_passes, cpu.number_of_coding_passes);
        assert_eq!(gpu.missing_bit_planes, cpu.missing_bit_planes);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_classic_tier1_kernel_matches_scalar_for_selective_bypass() {
        let coeffs: Vec<i32> = (0..64)
            .map(|idx| {
                let value = ((idx * 61 + 29) & 0x7ff) - 1023;
                if idx % 4 == 0 {
                    0
                } else {
                    value
                }
            })
            .collect();
        let style = J2kCodeBlockStyle {
            selective_arithmetic_coding_bypass: true,
            reset_context_probabilities: false,
            termination_on_each_pass: false,
            vertically_causal_context: false,
            segmentation_symbols: false,
        };
        let job = signinum_j2k_native::J2kTier1CodeBlockEncodeJob {
            coefficients: &coeffs,
            width: 8,
            height: 8,
            sub_band_type: signinum_j2k_native::J2kSubBandType::HighLow,
            total_bitplanes: 11,
            style,
        };

        let gpu =
            compute::encode_classic_tier1_code_block(job).expect("Metal classic bypass encode");
        let cpu = signinum_j2k_native::encode_j2k_code_block_scalar_with_style(
            &coeffs,
            8,
            8,
            signinum_j2k_native::J2kSubBandType::HighLow,
            11,
            style,
        )
        .expect("scalar classic bypass encode");

        assert_eq!(gpu.data, cpu.data);
        assert_eq!(gpu.segments.len(), cpu.segments.len());
        for (gpu_segment, cpu_segment) in gpu.segments.iter().zip(cpu.segments.iter()) {
            assert_eq!(gpu_segment.data_offset, cpu_segment.data_offset);
            assert_eq!(gpu_segment.data_length, cpu_segment.data_length);
            assert_eq!(gpu_segment.start_coding_pass, cpu_segment.start_coding_pass);
            assert_eq!(gpu_segment.end_coding_pass, cpu_segment.end_coding_pass);
            assert_eq!(gpu_segment.use_arithmetic, cpu_segment.use_arithmetic);
        }
        assert_eq!(gpu.number_of_coding_passes, cpu.number_of_coding_passes);
        assert_eq!(gpu.missing_bit_planes, cpu.missing_bit_planes);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_classic_tier1_batched_bypass_u16_32_matches_scalar() {
        let coeffs: Vec<i32> = (0..32 * 32)
            .map(|idx| {
                let value = ((idx * 97 + idx / 3 + 19) & 0x7ff) - 1023;
                if idx % 11 == 0 || idx % 17 == 0 {
                    0
                } else {
                    value
                }
            })
            .collect();
        let style = J2kCodeBlockStyle {
            selective_arithmetic_coding_bypass: true,
            reset_context_probabilities: false,
            termination_on_each_pass: false,
            vertically_causal_context: false,
            segmentation_symbols: false,
        };
        let job = signinum_j2k_native::J2kTier1CodeBlockEncodeJob {
            coefficients: &coeffs,
            width: 32,
            height: 32,
            sub_band_type: signinum_j2k_native::J2kSubBandType::HighHigh,
            total_bitplanes: 11,
            style,
        };

        let gpu = compute::encode_classic_tier1_code_blocks(&[job])
            .expect("batched Metal classic bypass_u16_32 encode")
            .pop()
            .expect("one encoded codeblock");
        let cpu = signinum_j2k_native::encode_j2k_code_block_scalar_with_style(
            &coeffs,
            32,
            32,
            signinum_j2k_native::J2kSubBandType::HighHigh,
            11,
            style,
        )
        .expect("scalar classic bypass encode");

        assert_eq!(gpu.data, cpu.data);
        assert_eq!(gpu.segments.len(), cpu.segments.len());
        for (gpu_segment, cpu_segment) in gpu.segments.iter().zip(cpu.segments.iter()) {
            assert_eq!(gpu_segment.data_offset, cpu_segment.data_offset);
            assert_eq!(gpu_segment.data_length, cpu_segment.data_length);
            assert_eq!(gpu_segment.start_coding_pass, cpu_segment.start_coding_pass);
            assert_eq!(gpu_segment.end_coding_pass, cpu_segment.end_coding_pass);
            assert_eq!(gpu_segment.use_arithmetic, cpu_segment.use_arithmetic);
        }
        assert_eq!(gpu.number_of_coding_passes, cpu.number_of_coding_passes);
        assert_eq!(gpu.missing_bit_planes, cpu.missing_bit_planes);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_classic_tier1_token_routes_match_scalar_bytes() {
        let first_coeffs: Vec<i32> = (0..32 * 32)
            .map(|idx| {
                let value = ((idx * 37 + idx / 5 + 31) & 0xff) - 127;
                if idx % 5 == 0 || idx % 11 == 0 {
                    0
                } else {
                    value
                }
            })
            .collect();
        let second_coeffs: Vec<i32> = (0..17 * 29)
            .map(|idx| {
                let value = ((idx * 73 + idx / 7 + 11) & 0xff) - 127;
                if idx % 7 == 0 || idx % 23 == 0 {
                    0
                } else {
                    value
                }
            })
            .collect();
        let style = J2kCodeBlockStyle {
            selective_arithmetic_coding_bypass: true,
            reset_context_probabilities: false,
            termination_on_each_pass: false,
            vertically_causal_context: false,
            segmentation_symbols: false,
        };
        let jobs = [
            signinum_j2k_native::J2kTier1CodeBlockEncodeJob {
                coefficients: &first_coeffs,
                width: 32,
                height: 32,
                sub_band_type: signinum_j2k_native::J2kSubBandType::HighHigh,
                total_bitplanes: 8,
                style,
            },
            signinum_j2k_native::J2kTier1CodeBlockEncodeJob {
                coefficients: &second_coeffs,
                width: 17,
                height: 29,
                sub_band_type: signinum_j2k_native::J2kSubBandType::LowLow,
                total_bitplanes: 8,
                style,
            },
        ];

        let gpu_packed =
            compute::encode_classic_tier1_code_blocks_via_gpu_token_pack_for_test(&jobs)
                .expect("Metal classic GPU token-pack encode");
        let cpu_packed =
            compute::encode_classic_tier1_code_blocks_via_ordered_tokens_cpu_pack_for_test(&jobs)
                .expect("Metal classic ordered-token CPU-pack encode");
        let split_packed =
            compute::encode_classic_tier1_code_blocks_via_split_mq_raw_tokens_cpu_pack_for_test(
                &jobs,
            )
            .expect("Metal classic split MQ/raw token CPU-pack encode");
        let split_gpu_packed =
            compute::encode_classic_tier1_code_blocks_via_split_mq_raw_tokens_gpu_pack_for_test(
                &jobs,
            )
            .expect("Metal classic split MQ/raw token GPU-pack encode");
        let mq_byte_split_gpu_packed = compute::encode_classic_tier1_code_blocks_via_split_mq_byte_raw_tokens_gpu_pack_for_test(
            &jobs,
        )
        .expect("Metal classic split MQ-byte/raw-bit token GPU-pack encode");

        assert_eq!(gpu_packed.len(), jobs.len());
        assert_eq!(cpu_packed.len(), jobs.len());
        assert_eq!(split_packed.len(), jobs.len());
        assert_eq!(split_gpu_packed.len(), jobs.len());
        assert_eq!(mq_byte_split_gpu_packed.len(), jobs.len());
        for (
            (
                (((gpu_block, cpu_packed_block), split_packed_block), split_gpu_packed_block),
                mq_byte_split_gpu_packed_block,
            ),
            job,
            coeffs,
        ) in gpu_packed
            .iter()
            .zip(cpu_packed.iter())
            .zip(split_packed.iter())
            .zip(split_gpu_packed.iter())
            .zip(mq_byte_split_gpu_packed.iter())
            .zip(jobs.iter())
            .zip([&first_coeffs, &second_coeffs])
            .map(|((blocks, job), coeffs)| (blocks, job, coeffs))
        {
            let cpu = signinum_j2k_native::encode_j2k_code_block_scalar_with_style(
                coeffs,
                job.width,
                job.height,
                job.sub_band_type,
                job.total_bitplanes,
                style,
            )
            .expect("scalar classic bypass encode");

            assert_eq!(gpu_block.data, cpu.data);
            assert_eq!(gpu_block.segments, cpu.segments);
            assert_eq!(
                gpu_block.number_of_coding_passes,
                cpu.number_of_coding_passes
            );
            assert_eq!(gpu_block.missing_bit_planes, cpu.missing_bit_planes);
            assert_eq!(cpu_packed_block.data, cpu.data);
            assert_eq!(cpu_packed_block.segments, cpu.segments);
            assert_eq!(
                cpu_packed_block.number_of_coding_passes,
                cpu.number_of_coding_passes
            );
            assert_eq!(cpu_packed_block.missing_bit_planes, cpu.missing_bit_planes);
            assert_eq!(split_packed_block.data, cpu.data);
            assert_eq!(split_packed_block.segments, cpu.segments);
            assert_eq!(
                split_packed_block.number_of_coding_passes,
                cpu.number_of_coding_passes
            );
            assert_eq!(
                split_packed_block.missing_bit_planes,
                cpu.missing_bit_planes
            );
            assert_eq!(split_gpu_packed_block.data, cpu.data);
            assert_eq!(split_gpu_packed_block.segments, cpu.segments);
            assert_eq!(
                split_gpu_packed_block.number_of_coding_passes,
                cpu.number_of_coding_passes
            );
            assert_eq!(
                split_gpu_packed_block.missing_bit_planes,
                cpu.missing_bit_planes
            );
            assert_eq!(mq_byte_split_gpu_packed_block.data, cpu.data);
            assert_eq!(mq_byte_split_gpu_packed_block.segments, cpu.segments);
            assert_eq!(
                mq_byte_split_gpu_packed_block.number_of_coding_passes,
                cpu.number_of_coding_passes
            );
            assert_eq!(
                mq_byte_split_gpu_packed_block.missing_bit_planes,
                cpu.missing_bit_planes
            );
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_htj2k_cleanup_kernel_matches_scalar_oracle() {
        let coeffs: Vec<i32> = (0..64)
            .map(|idx| {
                let value = ((idx * 19 + 7) & 0xff) - 127;
                if idx % 7 == 0 {
                    0
                } else {
                    value
                }
            })
            .collect();
        let job = signinum_j2k_native::J2kHtCodeBlockEncodeJob {
            coefficients: &coeffs,
            width: 8,
            height: 8,
            total_bitplanes: 8,
            target_coding_passes: 1,
        };

        let gpu = compute::encode_ht_cleanup_code_block(job).expect("Metal HT encode");
        let cpu = signinum_j2k_native::encode_ht_code_block_scalar(&coeffs, 8, 8, 8)
            .expect("scalar HT encode");

        assert_eq!(gpu.data, cpu.data);
        assert_eq!(gpu.num_coding_passes, cpu.num_coding_passes);
        assert_eq!(gpu.num_zero_bitplanes, cpu.num_zero_bitplanes);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_tier2_packetization_kernel_matches_scalar_oracle() {
        let block0 = [0x12, 0x34, 0x56, 0x78];
        let block1 = [0x9a, 0xbc];
        let code_blocks = vec![
            signinum_j2k_native::J2kPacketizationCodeBlock {
                data: &block0,
                ht_cleanup_length: 0,
                ht_refinement_length: 0,
                num_coding_passes: 1,
                num_zero_bitplanes: 2,
                previously_included: false,
                l_block: 3,
                block_coding_mode: signinum_j2k_native::J2kPacketizationBlockCodingMode::Classic,
            },
            signinum_j2k_native::J2kPacketizationCodeBlock {
                data: &block1,
                ht_cleanup_length: u32::try_from(block1.len()).expect("test payload fits u32"),
                ht_refinement_length: 0,
                num_coding_passes: 1,
                num_zero_bitplanes: 1,
                previously_included: false,
                l_block: 3,
                block_coding_mode:
                    signinum_j2k_native::J2kPacketizationBlockCodingMode::HighThroughput,
            },
        ];
        let subband = signinum_j2k_native::J2kPacketizationSubband {
            code_blocks,
            num_cbs_x: 2,
            num_cbs_y: 1,
        };
        let resolution = signinum_j2k_native::J2kPacketizationResolution {
            subbands: vec![subband],
        };
        let resolutions = [resolution];
        let packet_descriptors = [signinum_j2k_native::J2kPacketizationPacketDescriptor {
            packet_index: 0,
            state_index: 0,
            layer: 0,
            resolution: 0,
            component: 0,
            precinct: 0,
        }];
        let job = signinum_j2k_native::J2kPacketizationEncodeJob {
            resolution_count: 1,
            num_layers: 1,
            num_components: 1,
            code_block_count: 2,
            progression_order: signinum_j2k_native::J2kPacketizationProgressionOrder::Lrcp,
            packet_descriptors: &packet_descriptors,
            resolutions: &resolutions,
        };

        let gpu = compute::encode_tier2_packetization(job).expect("Metal packet encode");
        let cpu = signinum_j2k_native::encode_j2k_packetization_scalar(job)
            .expect("scalar packet encode");

        assert_eq!(gpu, cpu);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_tier2_packetization_reuses_descriptor_state_across_layers() {
        let block0 = vec![0x11];
        let block1 = vec![0x22];
        let first = signinum_j2k_native::J2kPacketizationResolution {
            subbands: vec![signinum_j2k_native::J2kPacketizationSubband {
                code_blocks: vec![signinum_j2k_native::J2kPacketizationCodeBlock {
                    data: &block0,
                    ht_cleanup_length: 0,
                    ht_refinement_length: 0,
                    num_coding_passes: 1,
                    num_zero_bitplanes: 0,
                    previously_included: false,
                    l_block: 3,
                    block_coding_mode:
                        signinum_j2k_native::J2kPacketizationBlockCodingMode::Classic,
                }],
                num_cbs_x: 1,
                num_cbs_y: 1,
            }],
        };
        let second = signinum_j2k_native::J2kPacketizationResolution {
            subbands: vec![signinum_j2k_native::J2kPacketizationSubband {
                code_blocks: vec![signinum_j2k_native::J2kPacketizationCodeBlock {
                    data: &block1,
                    ht_cleanup_length: 0,
                    ht_refinement_length: 0,
                    num_coding_passes: 1,
                    num_zero_bitplanes: 0,
                    previously_included: false,
                    l_block: 3,
                    block_coding_mode:
                        signinum_j2k_native::J2kPacketizationBlockCodingMode::Classic,
                }],
                num_cbs_x: 1,
                num_cbs_y: 1,
            }],
        };
        let resolutions = [first, second];
        let packet_descriptors = [
            signinum_j2k_native::J2kPacketizationPacketDescriptor {
                packet_index: 0,
                state_index: 0,
                layer: 0,
                resolution: 0,
                component: 0,
                precinct: 0,
            },
            signinum_j2k_native::J2kPacketizationPacketDescriptor {
                packet_index: 1,
                state_index: 0,
                layer: 1,
                resolution: 0,
                component: 0,
                precinct: 0,
            },
        ];
        let job = signinum_j2k_native::J2kPacketizationEncodeJob {
            resolution_count: 2,
            num_layers: 2,
            num_components: 1,
            code_block_count: 2,
            progression_order: signinum_j2k_native::J2kPacketizationProgressionOrder::Rpcl,
            packet_descriptors: &packet_descriptors,
            resolutions: &resolutions,
        };

        let gpu = compute::encode_tier2_packetization(job).expect("Metal packet encode");
        let cpu = signinum_j2k_native::encode_j2k_packetization_scalar(job)
            .expect("scalar packet encode");

        assert_eq!(gpu, cpu);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_tier2_packetization_honors_explicit_descriptor_order() {
        let block0 = vec![0xA0];
        let block1 = vec![0xB0];
        let first = signinum_j2k_native::J2kPacketizationResolution {
            subbands: vec![signinum_j2k_native::J2kPacketizationSubband {
                code_blocks: vec![signinum_j2k_native::J2kPacketizationCodeBlock {
                    data: &block0,
                    ht_cleanup_length: 0,
                    ht_refinement_length: 0,
                    num_coding_passes: 1,
                    num_zero_bitplanes: 0,
                    previously_included: false,
                    l_block: 3,
                    block_coding_mode:
                        signinum_j2k_native::J2kPacketizationBlockCodingMode::Classic,
                }],
                num_cbs_x: 1,
                num_cbs_y: 1,
            }],
        };
        let second = signinum_j2k_native::J2kPacketizationResolution {
            subbands: vec![signinum_j2k_native::J2kPacketizationSubband {
                code_blocks: vec![signinum_j2k_native::J2kPacketizationCodeBlock {
                    data: &block1,
                    ht_cleanup_length: 0,
                    ht_refinement_length: 0,
                    num_coding_passes: 1,
                    num_zero_bitplanes: 0,
                    previously_included: false,
                    l_block: 3,
                    block_coding_mode:
                        signinum_j2k_native::J2kPacketizationBlockCodingMode::Classic,
                }],
                num_cbs_x: 1,
                num_cbs_y: 1,
            }],
        };
        let resolutions = [first, second];
        let packet_descriptors = [
            signinum_j2k_native::J2kPacketizationPacketDescriptor {
                packet_index: 1,
                state_index: 1,
                layer: 0,
                resolution: 1,
                component: 0,
                precinct: 0,
            },
            signinum_j2k_native::J2kPacketizationPacketDescriptor {
                packet_index: 0,
                state_index: 0,
                layer: 0,
                resolution: 0,
                component: 0,
                precinct: 0,
            },
        ];
        let job = signinum_j2k_native::J2kPacketizationEncodeJob {
            resolution_count: 2,
            num_layers: 1,
            num_components: 1,
            code_block_count: 2,
            progression_order: signinum_j2k_native::J2kPacketizationProgressionOrder::Rpcl,
            packet_descriptors: &packet_descriptors,
            resolutions: &resolutions,
        };

        let gpu = compute::encode_tier2_packetization(job).expect("Metal packet encode");
        let cpu = signinum_j2k_native::encode_j2k_packetization_scalar(job)
            .expect("scalar packet encode");

        assert_eq!(gpu, cpu);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_forward_dwt53_handles_single_sample_edge_dimensions() {
        for (width, height) in [(1, 8), (8, 1)] {
            let samples: Vec<f32> = (0..width * height)
                .map(|i| {
                    f32::from(
                        u8::try_from((i * 11 + width * 3 + height * 5) & 0xFF)
                            .expect("masked sample fits in u8"),
                    ) - 128.0
                })
                .collect();
            let mut accelerator = MetalEncodeStageAccelerator::default();

            let output = accelerator
                .encode_forward_dwt53(J2kForwardDwt53Job {
                    samples: &samples,
                    width,
                    height,
                    num_levels: 1,
                })
                .expect("metal DWT 5/3 stage")
                .expect("metal DWT 5/3 dispatch");

            assert_eq!(output.ll_width, width.div_ceil(2));
            assert_eq!(output.ll_height, height.div_ceil(2));
            assert_eq!(output.levels.len(), 1);
            assert_eq!(accelerator.forward_dwt53_attempts(), 1);
            assert_eq!(accelerator.forward_dwt53_dispatches(), 1);
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_forward_dwt53_matches_reference_for_fractional_stage_samples() {
        fn assert_slice_near(actual: &[f32], expected: &[f32], label: &str) {
            assert_eq!(actual.len(), expected.len(), "{label} length mismatch");
            for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
                assert!(
                    (actual - expected).abs() <= 0.0001,
                    "{label}[{index}] mismatch: actual={actual}, expected={expected}"
                );
            }
        }

        let width = 8;
        let height = 8;
        let samples = (0..width * height)
            .map(|idx| f32::from(u16::try_from(idx).expect("test index fits u16")) * 0.5 - 15.25)
            .collect::<Vec<_>>();
        let expected = forward_dwt53_reference(&samples, width, height, 1);
        let mut accelerator = MetalEncodeStageAccelerator::default();

        let actual = accelerator
            .encode_forward_dwt53(J2kForwardDwt53Job {
                samples: &samples,
                width,
                height,
                num_levels: 1,
            })
            .expect("metal DWT 5/3 stage")
            .expect("metal DWT 5/3 dispatch");

        assert_eq!(actual.ll_width, expected.ll_width);
        assert_eq!(actual.ll_height, expected.ll_height);
        assert_slice_near(&actual.ll, &expected.ll, "LL");
        assert_eq!(actual.levels.len(), expected.levels.len());
        for (index, (actual, expected)) in actual.levels.iter().zip(&expected.levels).enumerate() {
            assert_eq!(actual.width, expected.width, "level {index} width");
            assert_eq!(actual.height, expected.height, "level {index} height");
            assert_eq!(
                actual.low_width, expected.low_width,
                "level {index} low width"
            );
            assert_eq!(
                actual.low_height, expected.low_height,
                "level {index} low height"
            );
            assert_eq!(
                actual.high_width, expected.high_width,
                "level {index} high width"
            );
            assert_eq!(
                actual.high_height, expected.high_height,
                "level {index} high height"
            );
            assert_slice_near(&actual.hl, &expected.hl, "HL");
            assert_slice_near(&actual.lh, &expected.lh, "LH");
            assert_slice_near(&actual.hh, &expected.hh, "HH");
        }
    }
}
