// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(target_os = "macos")]
use crate::engine as compute;
#[cfg(target_os = "macos")]
use j2k::J2kEncodeStageError;
#[cfg(target_os = "macos")]
use j2k::{EncodeBackendPreference, J2kLosslessEncodeOptions};
use j2k::{
    EncodedHtJ2kCodeBlock, EncodedJ2kCodeBlock, J2kDeinterleaveMctToF32Job,
    J2kDeinterleaveToF32Job, J2kEncodeContext, J2kEncodeDispatchReport, J2kEncodeStageAccelerator,
    J2kEncodeStageResult, J2kForwardDwt53Job, J2kForwardDwt53Output, J2kForwardDwt97Job,
    J2kForwardDwt97Output, J2kForwardIctJob, J2kForwardRctJob, J2kHtCodeBlockEncodeJob,
    J2kHtj2kTileEncodeJob, J2kPacketizationEncodeJob, J2kQuantizeSubbandJob,
    J2kTier1CodeBlockEncodeJob,
};
#[cfg(target_os = "macos")]
use j2k_core::PixelFormat;

#[cfg(target_os = "macos")]
use super::{
    copy_padded_metal_buffer_from_bytes, encode_resident_ht_tile_body_with_cpu_packetization,
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
    route_profile: MetalEncodeRouteProfile,
    parallel_cpu_code_block_fallback: bool,
    auto_host_output_force_cpu_fallback: bool,
    host_output_stages_enabled: bool,
    combined_input_mct_evidence: CombinedInputMctEvidence,
    ht_tile_required_magnitude_bound: Option<u8>,
    deinterleave_attempts: usize,
    combined_input_mct_attempts: usize,
    forward_rct_attempts: usize,
    forward_ict_attempts: usize,
    forward_dwt53_attempts: usize,
    forward_dwt97_attempts: usize,
    quantize_subband_attempts: usize,
    tier1_code_block_attempts: usize,
    ht_code_block_attempts: usize,
    packetization_attempts: usize,
    deinterleave_dispatches: usize,
    #[cfg(target_os = "macos")]
    combined_input_mct_dispatches: usize,
    forward_rct_dispatches: usize,
    forward_ict_dispatches: usize,
    forward_dwt53_dispatches: usize,
    forward_dwt97_dispatches: usize,
    quantize_subband_dispatches: usize,
    tier1_code_block_dispatches: usize,
    ht_code_block_dispatches: usize,
    packetization_dispatches: usize,
}

impl Default for MetalEncodeStageAccelerator {
    fn default() -> Self {
        Self {
            dispatch_stages: MetalEncodeDispatchStages::ALL,
            route_profile: MetalEncodeRouteProfile::Explicit,
            parallel_cpu_code_block_fallback: false,
            auto_host_output_force_cpu_fallback: false,
            host_output_stages_enabled: false,
            combined_input_mct_evidence: CombinedInputMctEvidence::Eligible,
            ht_tile_required_magnitude_bound: None,
            deinterleave_attempts: 0,
            combined_input_mct_attempts: 0,
            forward_rct_attempts: 0,
            forward_ict_attempts: 0,
            forward_dwt53_attempts: 0,
            forward_dwt97_attempts: 0,
            quantize_subband_attempts: 0,
            tier1_code_block_attempts: 0,
            ht_code_block_attempts: 0,
            packetization_attempts: 0,
            deinterleave_dispatches: 0,
            #[cfg(target_os = "macos")]
            combined_input_mct_dispatches: 0,
            forward_rct_dispatches: 0,
            forward_ict_dispatches: 0,
            forward_dwt53_dispatches: 0,
            forward_dwt97_dispatches: 0,
            quantize_subband_dispatches: 0,
            tier1_code_block_dispatches: 0,
            ht_code_block_dispatches: 0,
            packetization_dispatches: 0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CombinedInputMctEvidence {
    Eligible,
    OutsideMeasuredGeometry,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MetalEncodeRouteProfile {
    Explicit,
    AutoHostOutput,
    HostOutputEvidence,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MetalEncodeDispatchStages(u16);

impl MetalEncodeDispatchStages {
    const DEINTERLEAVE: Self = Self(1 << 0);
    const FORWARD_RCT: Self = Self(1 << 1);
    const FORWARD_ICT: Self = Self(1 << 2);
    const FORWARD_DWT53: Self = Self(1 << 3);
    const FORWARD_DWT97: Self = Self(1 << 4);
    const QUANTIZE_SUBBAND: Self = Self(1 << 5);
    const TIER1_CODE_BLOCK: Self = Self(1 << 6);
    const HT_CODE_BLOCK: Self = Self(1 << 7);
    const PACKETIZATION: Self = Self(1 << 8);
    const AUTO_HOST_OUTPUT_STAGE_DISPATCHES: Self = Self(
        Self::DEINTERLEAVE.0
            | Self::FORWARD_RCT.0
            | Self::FORWARD_ICT.0
            | Self::FORWARD_DWT53.0
            | Self::FORWARD_DWT97.0
            | Self::QUANTIZE_SUBBAND.0,
    );
    const ALL: Self = Self(
        Self::DEINTERLEAVE.0
            | Self::FORWARD_RCT.0
            | Self::FORWARD_ICT.0
            | Self::FORWARD_DWT53.0
            | Self::FORWARD_DWT97.0
            | Self::QUANTIZE_SUBBAND.0
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
            dispatch_stages: MetalEncodeDispatchStages::AUTO_HOST_OUTPUT_STAGE_DISPATCHES,
            route_profile: MetalEncodeRouteProfile::AutoHostOutput,
            parallel_cpu_code_block_fallback: true,
            ..Self::default()
        }
    }

    /// Create the host-output hybrid route without applying the Auto size gate.
    ///
    /// This is intended for reproducible route benchmarks and adapter-IUT
    /// conformance evidence. For supported lossless HT inputs it runs Metal
    /// coefficient preparation and HT Tier-1 with CPU packetization, matching
    /// the production host-output route without applying its size gate. It does
    /// not change the public Auto policy.
    #[must_use]
    #[doc(hidden)]
    pub fn for_host_output_benchmark() -> Self {
        Self {
            dispatch_stages: MetalEncodeDispatchStages::AUTO_HOST_OUTPUT_STAGE_DISPATCHES,
            route_profile: MetalEncodeRouteProfile::HostOutputEvidence,
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

    /// Number of deinterleave stage attempts observed by crate-local diagnostics.
    #[cfg(test)]
    pub(crate) fn deinterleave_attempts(&self) -> usize {
        self.deinterleave_attempts
    }

    /// Number of combined input/MCT attempts observed by crate-local diagnostics.
    #[cfg(test)]
    pub(crate) fn combined_input_mct_attempts(&self) -> usize {
        self.combined_input_mct_attempts
    }

    /// Number of forward RCT stage attempts observed by crate-local diagnostics.
    #[cfg(test)]
    pub(crate) fn forward_rct_attempts(&self) -> usize {
        self.forward_rct_attempts
    }

    /// Number of forward ICT stage attempts observed by crate-local diagnostics.
    #[cfg(all(test, target_os = "macos"))]
    pub(crate) fn forward_ict_attempts(&self) -> usize {
        self.forward_ict_attempts
    }

    /// Number of forward 5/3 DWT stage attempts observed by crate-local diagnostics.
    #[cfg(test)]
    pub(crate) fn forward_dwt53_attempts(&self) -> usize {
        self.forward_dwt53_attempts
    }

    /// Number of forward 9/7 DWT stage attempts observed by crate-local diagnostics.
    #[cfg(all(test, target_os = "macos"))]
    pub(crate) fn forward_dwt97_attempts(&self) -> usize {
        self.forward_dwt97_attempts
    }

    /// Number of sub-band quantization stage attempts observed by crate-local diagnostics.
    #[cfg(all(test, target_os = "macos"))]
    pub(crate) fn quantize_subband_attempts(&self) -> usize {
        self.quantize_subband_attempts
    }

    /// Number of classic Tier-1 code-block encode attempts observed by crate-local diagnostics.
    #[cfg(test)]
    pub(crate) fn tier1_code_block_attempts(&self) -> usize {
        self.tier1_code_block_attempts
    }

    /// Number of HT code-block encode attempts observed by crate-local diagnostics.
    #[cfg(all(test, target_os = "macos"))]
    pub(crate) fn ht_code_block_attempts(&self) -> usize {
        self.ht_code_block_attempts
    }

    /// Number of packetization stage attempts observed by crate-local diagnostics.
    #[cfg(test)]
    pub(crate) fn packetization_attempts(&self) -> usize {
        self.packetization_attempts
    }

    /// Number of deinterleave Metal dispatches observed by crate-local diagnostics.
    #[cfg(test)]
    pub(crate) fn deinterleave_dispatches(&self) -> usize {
        self.deinterleave_dispatches
    }

    /// Number of combined input/MCT Metal dispatches observed by crate-local diagnostics.
    #[cfg(all(test, target_os = "macos"))]
    pub(crate) fn combined_input_mct_dispatches(&self) -> usize {
        self.combined_input_mct_dispatches
    }

    /// Number of forward RCT Metal dispatches observed by crate-local diagnostics.
    #[cfg(test)]
    pub(crate) fn forward_rct_dispatches(&self) -> usize {
        self.forward_rct_dispatches
    }

    /// Number of forward ICT Metal dispatches observed by crate-local diagnostics.
    #[cfg(all(test, target_os = "macos"))]
    pub(crate) fn forward_ict_dispatches(&self) -> usize {
        self.forward_ict_dispatches
    }

    /// Number of forward 5/3 DWT Metal dispatches observed by crate-local diagnostics.
    #[cfg(all(test, target_os = "macos"))]
    pub(crate) fn forward_dwt53_dispatches(&self) -> usize {
        self.forward_dwt53_dispatches
    }

    /// Number of forward 9/7 DWT Metal dispatches observed by crate-local diagnostics.
    #[cfg(all(test, target_os = "macos"))]
    pub(crate) fn forward_dwt97_dispatches(&self) -> usize {
        self.forward_dwt97_dispatches
    }

    /// Number of sub-band quantization Metal dispatches observed by crate-local diagnostics.
    #[cfg(all(test, target_os = "macos"))]
    pub(crate) fn quantize_subband_dispatches(&self) -> usize {
        self.quantize_subband_dispatches
    }

    /// Number of classic Tier-1 Metal dispatches observed by crate-local diagnostics.
    #[cfg(all(test, target_os = "macos"))]
    pub(crate) fn tier1_code_block_dispatches(&self) -> usize {
        self.tier1_code_block_dispatches
    }

    /// Number of HT code-block Metal dispatches observed by crate-local diagnostics.
    #[cfg(all(test, target_os = "macos"))]
    pub(crate) fn ht_code_block_dispatches(&self) -> usize {
        self.ht_code_block_dispatches
    }

    /// Number of packetization Metal dispatches observed by crate-local diagnostics.
    #[cfg(all(test, target_os = "macos"))]
    pub(crate) fn packetization_dispatches(&self) -> usize {
        self.packetization_dispatches
    }

    fn host_output_stage_supported(&self) -> bool {
        self.route_profile == MetalEncodeRouteProfile::Explicit || self.host_output_stages_enabled
    }
}

pub(super) fn auto_host_output_should_dispatch(context: J2kEncodeContext) -> bool {
    if context.reversible || context.bit_depth != 8 || context.signed {
        return false;
    }
    match context.num_components {
        3 => crate::generated::promotion::auto_lossy_rgb8_encode_qualifies(context.num_pixels),
        _ => false,
    }
}

pub(super) fn host_output_evidence_should_dispatch(context: J2kEncodeContext) -> bool {
    matches!(context.num_components, 1..=4)
}

#[cfg(target_os = "macos")]
fn metal_dispatch_result(
    result: Result<(), crate::Error>,
    operation: &'static str,
) -> J2kEncodeStageResult<bool> {
    match result {
        Ok(()) => Ok(true),
        Err(crate::Error::MetalUnavailable) => Ok(false),
        Err(source) => Err(J2kEncodeStageError::backend("metal", operation, source)),
    }
}

#[cfg(target_os = "macos")]
pub(super) fn metal_dispatch_option<T>(
    result: Result<T, crate::Error>,
    operation: &'static str,
) -> J2kEncodeStageResult<Option<T>> {
    match result {
        Ok(value) => Ok(Some(value)),
        Err(crate::Error::MetalUnavailable) => Ok(None),
        Err(source) => Err(J2kEncodeStageError::backend("metal", operation, source)),
    }
}

#[doc(hidden)]
impl J2kEncodeStageAccelerator for MetalEncodeStageAccelerator {
    fn begin_encode(&mut self, context: J2kEncodeContext) -> J2kEncodeStageResult<()> {
        self.auto_host_output_force_cpu_fallback = false;
        self.ht_tile_required_magnitude_bound = None;
        self.host_output_stages_enabled = match self.route_profile {
            MetalEncodeRouteProfile::Explicit => true,
            MetalEncodeRouteProfile::AutoHostOutput => auto_host_output_should_dispatch(context),
            MetalEncodeRouteProfile::HostOutputEvidence => {
                host_output_evidence_should_dispatch(context)
            }
        };
        self.combined_input_mct_evidence = if context.num_pixels == 512 * 512
            && context.num_components == 3
            && context.bit_depth == 8
            && !context.signed
        {
            CombinedInputMctEvidence::Eligible
        } else {
            CombinedInputMctEvidence::OutsideMeasuredGeometry
        };
        Ok(())
    }

    fn dispatch_report(&self) -> J2kEncodeDispatchReport {
        J2kEncodeDispatchReport {
            deinterleave: self.deinterleave_dispatches,
            forward_rct: self.forward_rct_dispatches,
            forward_ict: self.forward_ict_dispatches,
            forward_dwt53: self.forward_dwt53_dispatches,
            forward_dwt97: self.forward_dwt97_dispatches,
            quantize_subband: self.quantize_subband_dispatches,
            tier1_code_block: self.tier1_code_block_dispatches,
            ht_code_block: self.ht_code_block_dispatches,
            packetization: self.packetization_dispatches,
        }
    }

    fn prefer_parallel_cpu_code_block_fallback(&self) -> bool {
        self.parallel_cpu_code_block_fallback
    }

    fn ht_tile_required_magnitude_bound(&self) -> Option<u8> {
        self.ht_tile_required_magnitude_bound
    }

    fn encode_deinterleave(
        &mut self,
        job: J2kDeinterleaveToF32Job<'_>,
    ) -> J2kEncodeStageResult<Option<Vec<Vec<f32>>>> {
        self.deinterleave_attempts = self.deinterleave_attempts.saturating_add(1);
        if !self
            .dispatch_stages
            .contains(MetalEncodeDispatchStages::DEINTERLEAVE)
            || self.auto_host_output_force_cpu_fallback
            || !self.host_output_stage_supported()
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
                Err(crate::Error::UnsupportedMetalRequest { reason }) => {
                    Err(J2kEncodeStageError::unsupported(reason))
                }
                Err(source) => Err(J2kEncodeStageError::backend(
                    "metal",
                    "deinterleave encode",
                    source,
                )),
            }
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = job;
            Ok(None)
        }
    }

    fn encode_deinterleave_mct(
        &mut self,
        job: J2kDeinterleaveMctToF32Job<'_>,
    ) -> J2kEncodeStageResult<Option<Vec<Vec<f32>>>> {
        self.combined_input_mct_attempts = self.combined_input_mct_attempts.saturating_add(1);
        let transform_enabled = if job.reversible {
            self.dispatch_stages
                .contains(MetalEncodeDispatchStages::FORWARD_RCT)
        } else {
            self.dispatch_stages
                .contains(MetalEncodeDispatchStages::FORWARD_ICT)
        };
        #[cfg(target_os = "macos")]
        let fused_input_mct_disabled = crate::profile_env::fused_input_mct_disabled();
        #[cfg(not(target_os = "macos"))]
        let fused_input_mct_disabled = false;
        if !self
            .dispatch_stages
            .contains(MetalEncodeDispatchStages::DEINTERLEAVE)
            || !transform_enabled
            || self.auto_host_output_force_cpu_fallback
            || !self.host_output_stage_supported()
            || self.combined_input_mct_evidence != CombinedInputMctEvidence::Eligible
            || fused_input_mct_disabled
        {
            let _ = job;
            return Ok(None);
        }
        #[cfg(target_os = "macos")]
        {
            match compute::encode_deinterleave_mct_to_f32(job) {
                Ok(components) => {
                    self.combined_input_mct_dispatches =
                        self.combined_input_mct_dispatches.saturating_add(1);
                    self.deinterleave_dispatches = self.deinterleave_dispatches.saturating_add(1);
                    if job.reversible {
                        self.forward_rct_dispatches = self.forward_rct_dispatches.saturating_add(1);
                    } else {
                        self.forward_ict_dispatches = self.forward_ict_dispatches.saturating_add(1);
                    }
                    Ok(Some(components))
                }
                Err(crate::Error::MetalUnavailable) => Ok(None),
                Err(crate::Error::UnsupportedMetalRequest { reason }) => {
                    Err(J2kEncodeStageError::unsupported(reason))
                }
                Err(source) => Err(J2kEncodeStageError::backend(
                    "metal",
                    "combined deinterleave and MCT encode",
                    source,
                )),
            }
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = job;
            Ok(None)
        }
    }

    fn encode_forward_rct(&mut self, job: J2kForwardRctJob<'_>) -> J2kEncodeStageResult<bool> {
        self.forward_rct_attempts = self.forward_rct_attempts.saturating_add(1);
        if !self
            .dispatch_stages
            .contains(MetalEncodeDispatchStages::FORWARD_RCT)
            || self.auto_host_output_force_cpu_fallback
            || !self.host_output_stage_supported()
        {
            let _ = job;
            return Ok(false);
        }
        #[cfg(target_os = "macos")]
        {
            let result = compute::encode_forward_rct(job.plane0, job.plane1, job.plane2);
            let dispatched = metal_dispatch_result(result, "forward RCT encode")?;
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

    fn encode_forward_ict(&mut self, job: J2kForwardIctJob<'_>) -> J2kEncodeStageResult<bool> {
        self.forward_ict_attempts = self.forward_ict_attempts.saturating_add(1);
        if !self
            .dispatch_stages
            .contains(MetalEncodeDispatchStages::FORWARD_ICT)
            || self.auto_host_output_force_cpu_fallback
            || !self.host_output_stage_supported()
        {
            let _ = job;
            return Ok(false);
        }
        #[cfg(target_os = "macos")]
        {
            match compute::encode_forward_ict(job.plane0, job.plane1, job.plane2) {
                Ok(()) => {
                    self.forward_ict_dispatches = self.forward_ict_dispatches.saturating_add(1);
                    Ok(true)
                }
                Err(crate::Error::MetalUnavailable) => Ok(false),
                Err(crate::Error::UnsupportedMetalRequest { reason }) => {
                    Err(J2kEncodeStageError::unsupported(reason))
                }
                Err(source) => Err(J2kEncodeStageError::backend(
                    "metal",
                    "forward ICT encode",
                    source,
                )),
            }
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = job;
            Ok(false)
        }
    }

    fn encode_forward_dwt53(
        &mut self,
        job: J2kForwardDwt53Job<'_>,
    ) -> J2kEncodeStageResult<Option<J2kForwardDwt53Output>> {
        self.forward_dwt53_attempts = self.forward_dwt53_attempts.saturating_add(1);
        if job.num_levels == 0 {
            return Ok(None);
        }
        if self.auto_host_output_force_cpu_fallback {
            let _ = job;
            return Ok(None);
        }
        if !self.host_output_stage_supported() {
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
                "forward 5/3 DWT encode",
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
        job: J2kForwardDwt97Job<'_>,
    ) -> J2kEncodeStageResult<Option<J2kForwardDwt97Output>> {
        self.forward_dwt97_attempts = self.forward_dwt97_attempts.saturating_add(1);
        if job.num_levels == 0 || (job.width < 2 && job.height < 2) {
            return Ok(None);
        }
        if self.auto_host_output_force_cpu_fallback {
            let _ = job;
            return Ok(None);
        }
        if !self.host_output_stage_supported() {
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
                "forward 9/7 DWT encode",
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

    fn encode_quantize_subband(
        &mut self,
        job: J2kQuantizeSubbandJob<'_>,
    ) -> J2kEncodeStageResult<Option<Vec<i32>>> {
        self.quantize_subband_attempts = self.quantize_subband_attempts.saturating_add(1);
        if !self
            .dispatch_stages
            .contains(MetalEncodeDispatchStages::QUANTIZE_SUBBAND)
            || self.auto_host_output_force_cpu_fallback
            || !self.host_output_stage_supported()
        {
            let _ = job;
            return Ok(None);
        }
        #[cfg(target_os = "macos")]
        {
            match compute::encode_quantize_subband(job) {
                Ok(coefficients) => {
                    if !coefficients.is_empty() {
                        self.quantize_subband_dispatches =
                            self.quantize_subband_dispatches.saturating_add(1);
                    }
                    Ok(Some(coefficients))
                }
                Err(crate::Error::MetalUnavailable) => Ok(None),
                Err(crate::Error::UnsupportedMetalRequest { reason }) => {
                    Err(J2kEncodeStageError::unsupported(reason))
                }
                Err(source) => Err(J2kEncodeStageError::backend(
                    "metal",
                    "quantize subband encode",
                    source,
                )),
            }
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = job;
            Ok(None)
        }
    }

    fn encode_tier1_code_block(
        &mut self,
        job: J2kTier1CodeBlockEncodeJob<'_>,
    ) -> J2kEncodeStageResult<Option<EncodedJ2kCodeBlock>> {
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
                "classic Tier-1 encode",
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
        jobs: &[J2kTier1CodeBlockEncodeJob<'_>],
    ) -> J2kEncodeStageResult<Option<Vec<EncodedJ2kCodeBlock>>> {
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
                "classic Tier-1 batch encode",
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
        job: J2kHtCodeBlockEncodeJob<'_>,
    ) -> J2kEncodeStageResult<Option<EncodedHtJ2kCodeBlock>> {
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
                "HTJ2K code-block encode",
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
        jobs: &[J2kHtCodeBlockEncodeJob<'_>],
    ) -> J2kEncodeStageResult<Option<Vec<EncodedHtJ2kCodeBlock>>> {
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
                "HTJ2K code-block batch encode",
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
        job: J2kHtj2kTileEncodeJob<'_>,
    ) -> J2kEncodeStageResult<Option<Vec<u8>>> {
        #[cfg(target_os = "macos")]
        {
            if !matches!(
                self.route_profile,
                MetalEncodeRouteProfile::AutoHostOutput
                    | MetalEncodeRouteProfile::HostOutputEvidence
            ) {
                let _ = job;
                return Ok(None);
            }
            self.auto_host_output_force_cpu_fallback = false;
            self.ht_tile_required_magnitude_bound = None;
            let Some(options) = lossless_options_for_resident_htj2k_tile_job(job) else {
                return Ok(None);
            };
            if self.route_profile == MetalEncodeRouteProfile::AutoHostOutput
                && !should_use_resident_htj2k_host_tile_for_auto(job)
            {
                self.auto_host_output_force_cpu_fallback = true;
                return Ok(None);
            }
            let format = match job.num_components {
                1 => PixelFormat::Gray8,
                3 => PixelFormat::Rgb8,
                _ => return Ok(None),
            };
            let Some(session) = metal_dispatch_option(
                crate::MetalBackendSession::system_default(),
                "HTJ2K hybrid session creation",
            )?
            else {
                return Ok(None);
            };
            let Some(source_buffer) = metal_dispatch_option(
                copy_padded_metal_buffer_from_bytes(&session, job.pixels),
                "HTJ2K hybrid input upload",
            )?
            else {
                return Ok(None);
            };
            let pitch_bytes = (job.width as usize)
                .checked_mul(usize::from(job.num_components))
                .ok_or_else(|| {
                    J2kEncodeStageError::arithmetic_overflow("Metal HTJ2K hybrid tile pitch")
                })?;
            let tile = MetalLosslessEncodeTile {
                buffer: &source_buffer,
                byte_offset: 0,
                width: job.width,
                height: job.height,
                pitch_bytes,
                output_width: job.width,
                output_height: job.height,
                format,
            };
            let Some(Some(encoded)) = metal_dispatch_option(
                encode_resident_ht_tile_body_with_cpu_packetization(
                    tile,
                    options,
                    &session,
                    MetalEncodeInputStaging::AlreadyPaddedContiguous,
                    job.code_block_width,
                    job.code_block_height,
                ),
                "resident HTJ2K hybrid tile encode",
            )?
            else {
                return Ok(None);
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
            self.ht_tile_required_magnitude_bound = Some(encoded.required_ht_magnitude_bound);
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
        job: J2kPacketizationEncodeJob<'_>,
    ) -> J2kEncodeStageResult<Option<Vec<u8>>> {
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
                "Tier-2 packetization encode",
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
