// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k::{
    EncodedHtJ2kCodeBlock, EncodedJ2kCodeBlock, J2kDeinterleaveToF32Job, J2kEncodeDispatchReport,
    J2kEncodeStageAccelerator, J2kForwardDwt53Job, J2kForwardDwt53Output, J2kForwardDwt97Job,
    J2kForwardDwt97Output, J2kForwardIctJob, J2kForwardRctJob, J2kHtCodeBlockEncodeJob,
    J2kHtSubbandEncodeJob, J2kHtj2kTileEncodeJob, J2kPacketizationEncodeJob, J2kQuantizeSubbandJob,
    J2kTier1CodeBlockEncodeJob,
};
#[cfg(feature = "cuda-runtime")]
use j2k_cuda_runtime::{CudaContext, CudaError, CudaHtj2kEncodeResources, CudaJ2kQuantizeJob};
#[cfg(feature = "cuda-runtime")]
use std::sync::Arc;

use crate::profile;

#[cfg(feature = "cuda-runtime")]
use super::htj2k::{
    cuda_encode_ht_code_block, cuda_encode_ht_code_blocks, cuda_encode_ht_subband,
    cuda_encode_htj2k_tile_body, cuda_htj2k_encode_tables, encoded_ht_code_blocks_from_cuda,
};
use super::packetization::flatten_cuda_htj2k_packetization_job;
#[cfg(feature = "cuda-runtime")]
use super::packetization::{
    cuda_packetization_blocks, cuda_packetization_packets, cuda_packetization_subbands,
    cuda_packetization_tag_nodes, cuda_packetization_tag_states,
};
#[cfg(feature = "cuda-runtime")]
use super::{cuda_component_count_u8, cuda_dwt53_output_to_j2k, cuda_dwt97_output_to_j2k};

macro_rules! emit_cuda_encode_route {
    ($(($key:expr, $value:expr)),+ $(,)?) => {{
        if j2k_profile::gpu_route_profile_enabled() {
            let fields = [$(j2k_profile::ProfileField::label($key, $value)),+];
            j2k_profile::emit_gpu_route_fields("j2k", "cuda", &fields);
        }
    }};
}

/// CUDA implementation of selected JPEG 2000 encode stages.
#[derive(Debug, Default, Clone)]
#[allow(clippy::struct_excessive_bools)]
pub struct CudaEncodeStageAccelerator {
    #[cfg(feature = "cuda-runtime")]
    context: Option<CudaContext>,
    #[cfg(feature = "cuda-runtime")]
    encode_resources: Option<Arc<CudaHtj2kEncodeResources>>,
    #[cfg_attr(not(feature = "cuda-runtime"), allow(dead_code))]
    collect_profile: bool,
    deinterleave_attempts: usize,
    forward_rct_attempts: usize,
    forward_ict_attempts: usize,
    forward_dwt53_attempts: usize,
    forward_dwt97_attempts: usize,
    htj2k_tile_attempts: usize,
    quantize_subband_attempts: usize,
    ht_subband_attempts: usize,
    tier1_code_block_attempts: usize,
    ht_code_block_attempts: usize,
    packetization_attempts: usize,
    prefer_cpu_forward_rct: bool,
    prefer_cpu_ht_subband: bool,
    prefer_cpu_quantize_subband: bool,
    prefer_cpu_packetization: bool,
    deinterleave_dispatches: usize,
    forward_rct_dispatches: usize,
    forward_ict_dispatches: usize,
    forward_dwt53_dispatches: usize,
    forward_dwt97_dispatches: usize,
    #[cfg_attr(not(feature = "cuda-runtime"), allow(dead_code))]
    htj2k_tile_dispatches: usize,
    quantize_subband_dispatches: usize,
    #[cfg_attr(not(feature = "cuda-runtime"), allow(dead_code))]
    ht_subband_dispatches: usize,
    tier1_code_block_dispatches: usize,
    ht_code_block_dispatches: usize,
    packetization_dispatches: usize,
    deinterleave_us: u128,
    mct_us: u128,
    dwt_us: u128,
    quantize_us: u128,
    ht_encode_us: u128,
    packetize_us: u128,
}

impl CudaEncodeStageAccelerator {
    /// Create an encode-stage accelerator with optional CUDA stage timing collection.
    #[must_use]
    #[doc(hidden)]
    pub fn with_profile_collection(collect_profile: bool) -> Self {
        Self {
            collect_profile,
            ..Self::default()
        }
    }

    /// Create the measured Auto route for host-output HTJ2K encode.
    ///
    /// CUDA keeps the DWT and HT code-block stages, while forward RCT and
    /// Tier-2 packetization stay on the CPU for the current host-pixel path.
    #[must_use]
    pub fn for_auto_host_output() -> Self {
        Self::default()
            .prefer_cpu_forward_rct(true)
            .prefer_cpu_packetization(true)
    }

    /// Prefer scalar CPU forward RCT while keeping later CUDA stages enabled.
    #[must_use]
    pub fn prefer_cpu_forward_rct(mut self, prefer_cpu_forward_rct: bool) -> Self {
        self.prefer_cpu_forward_rct = prefer_cpu_forward_rct;
        self
    }

    /// Prefer scalar CPU Tier-2 packetization while keeping CUDA Tier-1/HT block coding enabled.
    ///
    /// This is useful for batches of many small tiles where launching a CUDA
    /// packetization kernel and copying several tiny descriptor buffers per tile
    /// costs more than forming the packet body on the host.
    #[must_use]
    pub fn prefer_cpu_packetization(mut self, prefer_cpu_packetization: bool) -> Self {
        self.prefer_cpu_packetization = prefer_cpu_packetization;
        self
    }

    /// Prefer host sub-band quantization while keeping batched CUDA HT code-block encode enabled.
    ///
    /// This avoids launching one CUDA quantize/subband path for every prepared
    /// subband in multi-resolution precomputed transcode outputs, where the
    /// many tiny launches cost more than CPU quantization.
    #[must_use]
    pub fn prefer_cpu_ht_subband(mut self, prefer_cpu_ht_subband: bool) -> Self {
        self.prefer_cpu_ht_subband = prefer_cpu_ht_subband;
        self
    }

    /// Prefer host sub-band quantization while keeping CUDA HT code-block encode enabled.
    ///
    /// Multi-resolution transcode workloads can contain thousands of small
    /// subbands; for those, CPU quantization plus one batched HT code-block
    /// encode per tile is currently faster than launching CUDA quantization for
    /// every subband.
    #[must_use]
    pub fn prefer_cpu_quantize_subband(mut self, prefer_cpu_quantize_subband: bool) -> Self {
        self.prefer_cpu_quantize_subband = prefer_cpu_quantize_subband;
        self
    }

    /// Return cumulative CUDA encode stage timings collected by this accelerator.
    #[must_use]
    pub const fn collected_stage_timings(&self) -> CudaEncodeStageTimings {
        CudaEncodeStageTimings {
            deinterleave_us: self.deinterleave_us,
            mct_us: self.mct_us,
            dwt_us: self.dwt_us,
            quantize_us: self.quantize_us,
            ht_encode_us: self.ht_encode_us,
            packetize_us: self.packetize_us,
        }
    }

    /// Clear cumulative CUDA encode stage timings without changing dispatch counters.
    pub fn reset_collected_stage_timings(&mut self) {
        self.deinterleave_us = 0;
        self.mct_us = 0;
        self.dwt_us = 0;
        self.quantize_us = 0;
        self.ht_encode_us = 0;
        self.packetize_us = 0;
    }

    #[cfg(feature = "cuda-runtime")]
    fn cuda_context(&mut self) -> core::result::Result<Option<CudaContext>, &'static str> {
        if self.context.is_none() {
            match CudaContext::system_default() {
                Ok(context) => self.context = Some(context),
                Err(_) if cuda_runtime_required() => return Err("CUDA encode stage unavailable"),
                Err(_) => return Ok(None),
            }
        }
        Ok(self.context.clone())
    }

    #[cfg(feature = "cuda-runtime")]
    fn cuda_encode_resources(
        &mut self,
        context: &CudaContext,
    ) -> core::result::Result<Arc<CudaHtj2kEncodeResources>, &'static str> {
        if self.encode_resources.is_none() {
            let resources = context
                .upload_htj2k_encode_resources(cuda_htj2k_encode_tables())
                .map_err(|_| "CUDA HTJ2K encode resource upload failed")?;
            self.encode_resources = Some(Arc::new(resources));
        }
        self.encode_resources
            .clone()
            .ok_or("CUDA HTJ2K encode resources unavailable")
    }

    pub(super) fn encode_profile_report(
        &self,
        encoded: &j2k::EncodedJ2k,
        input_bytes: usize,
        total_us: u128,
    ) -> profile::CudaHtj2kEncodeProfileReport {
        profile::CudaHtj2kEncodeProfileReport {
            deinterleave_us: self.deinterleave_us,
            mct_us: self.mct_us,
            dwt_us: self.dwt_us,
            quantize_us: self.quantize_us,
            ht_encode_us: self.ht_encode_us,
            packetize_us: self.packetize_us,
            total_us,
            input_bytes,
            codestream_bytes: encoded.codestream.len(),
            block_count: self.ht_code_block_attempts,
            dispatch_count: self.dispatch_report().total(),
            backend: encoded.backend,
        }
    }

    /// Number of forward RCT attempts observed by crate-local diagnostics.
    #[cfg(test)]
    pub(crate) fn forward_rct_attempts(&self) -> usize {
        self.forward_rct_attempts
    }

    /// Number of forward ICT attempts observed by crate-local diagnostics.
    #[cfg(test)]
    pub(crate) fn forward_ict_attempts(&self) -> usize {
        self.forward_ict_attempts
    }

    /// Number of forward 5/3 DWT attempts observed by crate-local diagnostics.
    #[cfg(test)]
    pub(crate) fn forward_dwt53_attempts(&self) -> usize {
        self.forward_dwt53_attempts
    }

    /// Number of forward 9/7 DWT attempts observed by crate-local diagnostics.
    #[cfg(test)]
    pub(crate) fn forward_dwt97_attempts(&self) -> usize {
        self.forward_dwt97_attempts
    }

    /// Number of resident HTJ2K tile-body attempts observed by crate-local diagnostics.
    #[cfg(test)]
    pub(crate) fn htj2k_tile_attempts(&self) -> usize {
        self.htj2k_tile_attempts
    }

    /// Number of sub-band quantization attempts observed by crate-local diagnostics.
    #[cfg(test)]
    pub(crate) fn quantize_subband_attempts(&self) -> usize {
        self.quantize_subband_attempts
    }

    /// Number of classic Tier-1 code-block attempts observed by crate-local diagnostics.
    #[cfg(test)]
    pub(crate) fn tier1_code_block_attempts(&self) -> usize {
        self.tier1_code_block_attempts
    }

    /// Number of HT code-block attempts observed by crate-local diagnostics.
    #[cfg(test)]
    pub(crate) fn ht_code_block_attempts(&self) -> usize {
        self.ht_code_block_attempts
    }

    /// Number of HT sub-band attempts observed by crate-local diagnostics.
    #[cfg(test)]
    pub(crate) fn ht_subband_attempts(&self) -> usize {
        self.ht_subband_attempts
    }

    /// Number of packetization attempts observed by crate-local diagnostics.
    #[cfg(test)]
    pub(crate) fn packetization_attempts(&self) -> usize {
        self.packetization_attempts
    }

    /// Number of deinterleave CUDA dispatches observed by crate-local diagnostics.
    #[cfg(test)]
    pub(crate) fn deinterleave_dispatches(&self) -> usize {
        self.deinterleave_dispatches
    }

    /// Number of forward RCT CUDA dispatches observed by crate-local diagnostics.
    #[cfg(test)]
    pub(crate) fn forward_rct_dispatches(&self) -> usize {
        self.forward_rct_dispatches
    }

    /// Number of forward ICT CUDA dispatches observed by crate-local diagnostics.
    #[cfg(test)]
    pub(crate) fn forward_ict_dispatches(&self) -> usize {
        self.forward_ict_dispatches
    }

    /// Number of forward 5/3 DWT CUDA dispatches observed by crate-local diagnostics.
    #[cfg(test)]
    pub(crate) fn forward_dwt53_dispatches(&self) -> usize {
        self.forward_dwt53_dispatches
    }

    /// Number of forward 9/7 DWT CUDA dispatches observed by crate-local diagnostics.
    #[cfg(test)]
    pub(crate) fn forward_dwt97_dispatches(&self) -> usize {
        self.forward_dwt97_dispatches
    }

    /// Number of resident HTJ2K tile-body CUDA dispatches observed by crate-local diagnostics.
    #[cfg(test)]
    pub(crate) fn htj2k_tile_dispatches(&self) -> usize {
        self.htj2k_tile_dispatches
    }

    /// Number of sub-band quantization CUDA dispatches observed by crate-local diagnostics.
    #[cfg(test)]
    pub(crate) fn quantize_subband_dispatches(&self) -> usize {
        self.quantize_subband_dispatches
    }

    /// Number of HT code-block CUDA dispatches observed by crate-local diagnostics.
    #[cfg(test)]
    pub(crate) fn ht_code_block_dispatches(&self) -> usize {
        self.ht_code_block_dispatches
    }

    /// Number of HT sub-band CUDA dispatches observed by crate-local diagnostics.
    #[cfg(test)]
    pub(crate) fn ht_subband_dispatches(&self) -> usize {
        self.ht_subband_dispatches
    }

    /// Number of packetization CUDA dispatches observed by crate-local diagnostics.
    #[cfg(test)]
    pub(crate) fn packetization_dispatches(&self) -> usize {
        self.packetization_dispatches
    }
}

#[cfg(feature = "cuda-runtime")]
fn cuda_runtime_required() -> bool {
    std::env::var_os("J2K_REQUIRE_CUDA_RUNTIME").is_some()
}

#[cfg(feature = "cuda-runtime")]
pub(super) fn time_cuda_stage<T>(
    name: &'static str,
    context: &CudaContext,
    collect_profile: bool,
    work: impl FnOnce() -> core::result::Result<T, CudaError>,
) -> core::result::Result<(T, u128), CudaError> {
    if collect_profile {
        context.time_default_stream_named_us(name, work)
    } else {
        context
            .with_nvtx_range(name, work)
            .map(|output| (output, 0))
    }
}

/// Cumulative CUDA encode-stage timings collected by `CudaEncodeStageAccelerator`.
#[allow(clippy::struct_field_names)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CudaEncodeStageTimings {
    /// Pixel deinterleave and level-shift CUDA stage time.
    pub deinterleave_us: u128,
    /// Forward MCT CUDA stage time.
    pub mct_us: u128,
    /// Forward DWT CUDA stage time.
    pub dwt_us: u128,
    /// Quantization CUDA stage time.
    pub quantize_us: u128,
    /// HT code-block encode CUDA stage time.
    pub ht_encode_us: u128,
    /// HTJ2K packetization CUDA stage time.
    pub packetize_us: u128,
}

impl CudaEncodeStageTimings {
    /// Return field-wise saturating timing sums.
    #[must_use]
    pub const fn saturating_add(self, other: Self) -> Self {
        Self {
            deinterleave_us: self.deinterleave_us.saturating_add(other.deinterleave_us),
            mct_us: self.mct_us.saturating_add(other.mct_us),
            dwt_us: self.dwt_us.saturating_add(other.dwt_us),
            quantize_us: self.quantize_us.saturating_add(other.quantize_us),
            ht_encode_us: self.ht_encode_us.saturating_add(other.ht_encode_us),
            packetize_us: self.packetize_us.saturating_add(other.packetize_us),
        }
    }

    /// Total collected CUDA encode-stage time.
    #[must_use]
    pub const fn total_us(self) -> u128 {
        self.deinterleave_us
            .saturating_add(self.mct_us)
            .saturating_add(self.dwt_us)
            .saturating_add(self.quantize_us)
            .saturating_add(self.ht_encode_us)
            .saturating_add(self.packetize_us)
    }
}

fn ht_subband_code_block_count(
    job: J2kHtSubbandEncodeJob<'_>,
) -> core::result::Result<usize, &'static str> {
    if job.code_block_width == 0 || job.code_block_height == 0 {
        return Err("CUDA HTJ2K subband encode job has invalid code-block dimensions");
    }
    let num_cbs_x = job.width.div_ceil(job.code_block_width);
    let num_cbs_y = job.height.div_ceil(job.code_block_height);
    (num_cbs_x as usize)
        .checked_mul(num_cbs_y as usize)
        .ok_or("CUDA HTJ2K subband code-block count overflow")
}

#[doc(hidden)]
impl J2kEncodeStageAccelerator for CudaEncodeStageAccelerator {
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

    fn encode_deinterleave(
        &mut self,
        job: J2kDeinterleaveToF32Job<'_>,
    ) -> core::result::Result<Option<Vec<Vec<f32>>>, &'static str> {
        self.deinterleave_attempts = self.deinterleave_attempts.saturating_add(1);
        #[cfg(feature = "cuda-runtime")]
        if let Some(context) = self.cuda_context()? {
            let num_components = cuda_component_count_u8(
                job.num_components,
                "CUDA deinterleave encode supports at most 255 components",
            )?;
            let (output, elapsed_us) = time_cuda_stage(
                "j2k.j2k.cuda.encode.deinterleave",
                &context,
                self.collect_profile,
                || {
                    context.j2k_deinterleave_to_f32(
                        job.pixels,
                        job.num_pixels,
                        num_components,
                        job.bit_depth,
                        job.signed,
                    )
                },
            )
            .map_err(|_| "CUDA deinterleave encode kernel failed")?;
            let dispatches = output.execution().kernel_dispatches();
            self.deinterleave_dispatches = self.deinterleave_dispatches.saturating_add(dispatches);
            self.deinterleave_us = self.deinterleave_us.saturating_add(elapsed_us);
            emit_cuda_encode_route!(
                ("op", "encode_deinterleave"),
                ("decision", "cuda_dispatch"),
                ("pixels", job.num_pixels),
                ("components", job.num_components),
                ("dispatches", dispatches),
            );
            return Ok(Some(output.into_components()));
        }
        #[cfg(not(feature = "cuda-runtime"))]
        let _ = job;
        emit_cuda_encode_route!(
            ("op", "encode_deinterleave"),
            ("decision", "cpu_fallback"),
            ("reason", "cuda_unavailable"),
        );
        Ok(None)
    }

    fn encode_forward_rct(
        &mut self,
        job: J2kForwardRctJob<'_>,
    ) -> core::result::Result<bool, &'static str> {
        self.forward_rct_attempts = self.forward_rct_attempts.saturating_add(1);
        if self.prefer_cpu_forward_rct {
            emit_cuda_encode_route!(
                ("op", "encode_forward_rct"),
                ("decision", "cpu_fallback"),
                ("reason", "prefer_cpu_forward_rct"),
            );
            let _ = job;
            return Ok(false);
        }
        #[cfg(feature = "cuda-runtime")]
        if let Some(context) = self.cuda_context()? {
            let (execution, elapsed_us) = time_cuda_stage(
                "j2k.j2k.cuda.encode.rct",
                &context,
                self.collect_profile,
                || context.j2k_forward_rct(job.plane0, job.plane1, job.plane2),
            )
            .map_err(|_| "CUDA forward RCT encode kernel failed")?;
            self.forward_rct_dispatches = self
                .forward_rct_dispatches
                .saturating_add(execution.kernel_dispatches());
            self.mct_us = self.mct_us.saturating_add(elapsed_us);
            emit_cuda_encode_route!(
                ("op", "encode_forward_rct"),
                ("decision", "cuda_dispatch"),
                ("dispatches", 1),
            );
            return Ok(true);
        }
        #[cfg(not(feature = "cuda-runtime"))]
        let _ = job;
        emit_cuda_encode_route!(
            ("op", "encode_forward_rct"),
            ("decision", "cpu_fallback"),
            ("reason", "cuda_unavailable"),
        );
        Ok(false)
    }

    fn encode_forward_ict(
        &mut self,
        job: J2kForwardIctJob<'_>,
    ) -> core::result::Result<bool, &'static str> {
        self.forward_ict_attempts = self.forward_ict_attempts.saturating_add(1);
        #[cfg(feature = "cuda-runtime")]
        if let Some(context) = self.cuda_context()? {
            let (execution, elapsed_us) = time_cuda_stage(
                "j2k.j2k.cuda.encode.ict",
                &context,
                self.collect_profile,
                || context.j2k_forward_ict(job.plane0, job.plane1, job.plane2),
            )
            .map_err(|_| "CUDA forward ICT encode kernel failed")?;
            self.forward_ict_dispatches = self
                .forward_ict_dispatches
                .saturating_add(execution.kernel_dispatches());
            self.mct_us = self.mct_us.saturating_add(elapsed_us);
            emit_cuda_encode_route!(
                ("op", "encode_forward_ict"),
                ("decision", "cuda_dispatch"),
                ("dispatches", 1),
            );
            return Ok(true);
        }
        #[cfg(not(feature = "cuda-runtime"))]
        let _ = job;
        emit_cuda_encode_route!(
            ("op", "encode_forward_ict"),
            ("decision", "cpu_fallback"),
            ("reason", "cuda_unavailable"),
        );
        Ok(false)
    }

    fn encode_forward_dwt53(
        &mut self,
        job: J2kForwardDwt53Job<'_>,
    ) -> core::result::Result<Option<J2kForwardDwt53Output>, &'static str> {
        self.forward_dwt53_attempts = self.forward_dwt53_attempts.saturating_add(1);
        if job.num_levels == 0 {
            emit_cuda_encode_route!(
                ("op", "encode_forward_dwt53"),
                ("decision", "cpu_fallback"),
                ("reason", "zero_levels"),
            );
            return Ok(None);
        }
        #[cfg(feature = "cuda-runtime")]
        if let Some(context) = self.cuda_context()? {
            let (output, elapsed_us) = time_cuda_stage(
                "j2k.j2k.cuda.encode.dwt53",
                &context,
                self.collect_profile,
                || context.j2k_forward_dwt53(job.samples, job.width, job.height, job.num_levels),
            )
            .map_err(|_| "CUDA forward 5/3 DWT encode kernel failed")?;
            let dispatches = output.execution().kernel_dispatches();
            self.forward_dwt53_dispatches =
                self.forward_dwt53_dispatches.saturating_add(dispatches);
            self.dwt_us = self.dwt_us.saturating_add(elapsed_us);
            emit_cuda_encode_route!(
                ("op", "encode_forward_dwt53"),
                ("decision", "cuda_dispatch"),
                ("width", job.width),
                ("height", job.height),
                ("levels", job.num_levels),
                ("dispatches", dispatches),
            );
            return Ok(Some(cuda_dwt53_output_to_j2k(&output)?));
        }
        #[cfg(not(feature = "cuda-runtime"))]
        let _ = job;
        emit_cuda_encode_route!(
            ("op", "encode_forward_dwt53"),
            ("decision", "cpu_fallback"),
            ("reason", "cuda_unavailable"),
        );
        Ok(None)
    }

    fn encode_forward_dwt97(
        &mut self,
        job: J2kForwardDwt97Job<'_>,
    ) -> core::result::Result<Option<J2kForwardDwt97Output>, &'static str> {
        self.forward_dwt97_attempts = self.forward_dwt97_attempts.saturating_add(1);
        if job.num_levels == 0 {
            emit_cuda_encode_route!(
                ("op", "encode_forward_dwt97"),
                ("decision", "cpu_fallback"),
                ("reason", "zero_levels"),
            );
            return Ok(None);
        }
        #[cfg(feature = "cuda-runtime")]
        if let Some(context) = self.cuda_context()? {
            let (output, elapsed_us) = time_cuda_stage(
                "j2k.j2k.cuda.encode.dwt97",
                &context,
                self.collect_profile,
                || context.j2k_forward_dwt97(job.samples, job.width, job.height, job.num_levels),
            )
            .map_err(|_| "CUDA forward 9/7 DWT encode kernel failed")?;
            let dispatches = output.execution().kernel_dispatches();
            self.forward_dwt97_dispatches =
                self.forward_dwt97_dispatches.saturating_add(dispatches);
            self.dwt_us = self.dwt_us.saturating_add(elapsed_us);
            emit_cuda_encode_route!(
                ("op", "encode_forward_dwt97"),
                ("decision", "cuda_dispatch"),
                ("width", job.width),
                ("height", job.height),
                ("levels", job.num_levels),
                ("dispatches", dispatches),
            );
            return Ok(Some(cuda_dwt97_output_to_j2k(&output)?));
        }
        #[cfg(not(feature = "cuda-runtime"))]
        let _ = job;
        emit_cuda_encode_route!(
            ("op", "encode_forward_dwt97"),
            ("decision", "cpu_fallback"),
            ("reason", "cuda_unavailable"),
        );
        Ok(None)
    }

    fn encode_quantize_subband(
        &mut self,
        job: J2kQuantizeSubbandJob<'_>,
    ) -> core::result::Result<Option<Vec<i32>>, &'static str> {
        self.quantize_subband_attempts = self.quantize_subband_attempts.saturating_add(1);
        if self.prefer_cpu_quantize_subband {
            emit_cuda_encode_route!(
                ("op", "encode_quantize_subband"),
                ("decision", "cpu_fallback"),
                ("reason", "prefer_cpu_quantize_subband"),
            );
            let _ = job;
            return Ok(None);
        }
        #[cfg(feature = "cuda-runtime")]
        if let Some(context) = self.cuda_context()? {
            let (output, elapsed_us) = time_cuda_stage(
                "j2k.j2k.cuda.encode.quantize",
                &context,
                self.collect_profile,
                || {
                    context.j2k_quantize_subband(
                        job.coefficients,
                        CudaJ2kQuantizeJob {
                            step_exponent: job.step_exponent,
                            step_mantissa: job.step_mantissa,
                            range_bits: job.range_bits,
                            reversible: job.reversible,
                        },
                    )
                },
            )
            .map_err(|_| "CUDA quantize subband encode kernel failed")?;
            let dispatches = output.execution().kernel_dispatches();
            self.quantize_subband_dispatches =
                self.quantize_subband_dispatches.saturating_add(dispatches);
            self.quantize_us = self.quantize_us.saturating_add(elapsed_us);
            emit_cuda_encode_route!(
                ("op", "encode_quantize_subband"),
                ("decision", "cuda_dispatch"),
                ("samples", job.coefficients.len()),
                ("dispatches", dispatches),
            );
            return Ok(Some(output.coefficients().to_vec()));
        }
        #[cfg(not(feature = "cuda-runtime"))]
        let _ = job;
        emit_cuda_encode_route!(
            ("op", "encode_quantize_subband"),
            ("decision", "cpu_fallback"),
            ("reason", "cuda_unavailable"),
        );
        Ok(None)
    }

    fn encode_tier1_code_block(
        &mut self,
        _job: J2kTier1CodeBlockEncodeJob<'_>,
    ) -> core::result::Result<Option<EncodedJ2kCodeBlock>, &'static str> {
        self.tier1_code_block_attempts = self.tier1_code_block_attempts.saturating_add(1);
        emit_cuda_encode_route!(
            ("op", "encode_tier1_code_block"),
            ("decision", "cpu_fallback"),
            ("reason", "unsupported_stage"),
        );
        Ok(None)
    }

    fn encode_ht_code_block(
        &mut self,
        job: J2kHtCodeBlockEncodeJob<'_>,
    ) -> core::result::Result<Option<EncodedHtJ2kCodeBlock>, &'static str> {
        self.ht_code_block_attempts = self.ht_code_block_attempts.saturating_add(1);
        #[cfg(feature = "cuda-runtime")]
        if let Some(context) = self.cuda_context()? {
            let resources = self.cuda_encode_resources(&context)?;
            let encoded = cuda_encode_ht_code_block(&context, resources.as_ref(), job)?;
            let dispatches = encoded.execution().kernel_dispatches();
            let ht_encode_us = encoded.stage_timings().ht_encode_us;
            let mut outputs = encoded_ht_code_blocks_from_cuda(&encoded);
            let output = outputs
                .pop()
                .ok_or("CUDA HTJ2K code-block encode returned no output")?;
            self.ht_code_block_dispatches =
                self.ht_code_block_dispatches.saturating_add(dispatches);
            if self.collect_profile {
                self.ht_encode_us = self.ht_encode_us.saturating_add(ht_encode_us);
            }
            emit_cuda_encode_route!(
                ("op", "encode_ht_code_block"),
                ("decision", "cuda_dispatch"),
                ("width", job.width),
                ("height", job.height),
                ("dispatches", dispatches),
            );
            return Ok(Some(output));
        }
        #[cfg(not(feature = "cuda-runtime"))]
        let _ = job;
        emit_cuda_encode_route!(
            ("op", "encode_ht_code_block"),
            ("decision", "cpu_fallback"),
            ("reason", "unsupported_stage"),
        );
        Ok(None)
    }

    fn encode_ht_code_blocks(
        &mut self,
        jobs: &[J2kHtCodeBlockEncodeJob<'_>],
    ) -> core::result::Result<Option<Vec<EncodedHtJ2kCodeBlock>>, &'static str> {
        self.ht_code_block_attempts = self.ht_code_block_attempts.saturating_add(jobs.len());
        #[cfg(feature = "cuda-runtime")]
        if let Some(context) = self.cuda_context()? {
            let resources = self.cuda_encode_resources(&context)?;
            let encoded = cuda_encode_ht_code_blocks(&context, resources.as_ref(), jobs)?;
            let dispatches = encoded.execution().kernel_dispatches();
            let ht_encode_us = encoded.stage_timings().ht_encode_us;
            let outputs = encoded_ht_code_blocks_from_cuda(&encoded);
            self.ht_code_block_dispatches =
                self.ht_code_block_dispatches.saturating_add(dispatches);
            if self.collect_profile {
                self.ht_encode_us = self.ht_encode_us.saturating_add(ht_encode_us);
            }
            emit_cuda_encode_route!(
                ("op", "encode_ht_code_blocks"),
                ("decision", "cuda_dispatch"),
                ("jobs", jobs.len()),
                ("dispatches", dispatches),
            );
            return Ok(Some(outputs));
        }
        #[cfg(not(feature = "cuda-runtime"))]
        let _ = jobs;
        emit_cuda_encode_route!(
            ("op", "encode_ht_code_blocks"),
            ("decision", "cpu_fallback"),
            ("reason", "cuda_unavailable"),
        );
        Ok(None)
    }

    fn encode_htj2k_tile(
        &mut self,
        job: J2kHtj2kTileEncodeJob<'_>,
    ) -> core::result::Result<Option<Vec<u8>>, &'static str> {
        self.htj2k_tile_attempts = self.htj2k_tile_attempts.saturating_add(1);
        if self.prefer_cpu_forward_rct || self.prefer_cpu_packetization {
            emit_cuda_encode_route!(
                ("op", "encode_htj2k_tile"),
                ("decision", "cpu_fallback"),
                ("reason", "prefer_stage_hybrid"),
            );
            let _ = job;
            return Ok(None);
        }
        #[cfg(feature = "cuda-runtime")]
        if let Some(context) = self.cuda_context()? {
            let resources = self.cuda_encode_resources(&context)?;
            let Some(encoded) = cuda_encode_htj2k_tile_body(
                &context,
                resources.as_ref(),
                job,
                self.collect_profile,
            )?
            else {
                return Ok(None);
            };
            self.htj2k_tile_dispatches = self.htj2k_tile_dispatches.saturating_add(1);
            self.deinterleave_attempts = self.deinterleave_attempts.saturating_add(1);
            self.deinterleave_dispatches = self
                .deinterleave_dispatches
                .saturating_add(encoded.deinterleave_dispatches);
            if job.use_mct {
                if job.reversible {
                    self.forward_rct_attempts = self.forward_rct_attempts.saturating_add(1);
                } else {
                    self.forward_ict_attempts = self.forward_ict_attempts.saturating_add(1);
                }
            }
            self.forward_rct_dispatches = self
                .forward_rct_dispatches
                .saturating_add(encoded.forward_rct_dispatches);
            self.forward_ict_dispatches = self
                .forward_ict_dispatches
                .saturating_add(encoded.forward_ict_dispatches);
            if job.num_decomposition_levels > 0 {
                if job.reversible {
                    self.forward_dwt53_attempts = self
                        .forward_dwt53_attempts
                        .saturating_add(usize::from(job.num_components));
                } else {
                    self.forward_dwt97_attempts = self
                        .forward_dwt97_attempts
                        .saturating_add(usize::from(job.num_components));
                }
            }
            self.forward_dwt53_dispatches = self
                .forward_dwt53_dispatches
                .saturating_add(encoded.forward_dwt53_dispatches);
            self.forward_dwt97_dispatches = self
                .forward_dwt97_dispatches
                .saturating_add(encoded.forward_dwt97_dispatches);
            self.quantize_subband_attempts = self
                .quantize_subband_attempts
                .saturating_add(encoded.quantize_jobs);
            self.quantize_subband_dispatches = self
                .quantize_subband_dispatches
                .saturating_add(encoded.quantize_dispatches);
            self.ht_code_block_attempts = self
                .ht_code_block_attempts
                .saturating_add(encoded.ht_code_block_jobs);
            self.ht_code_block_dispatches = self
                .ht_code_block_dispatches
                .saturating_add(encoded.ht_code_block_dispatches);
            self.packetization_attempts = self.packetization_attempts.saturating_add(1);
            self.packetization_dispatches = self
                .packetization_dispatches
                .saturating_add(encoded.packetization_dispatches);
            if self.collect_profile {
                self.deinterleave_us = self
                    .deinterleave_us
                    .saturating_add(encoded.timings.deinterleave_us);
                self.mct_us = self.mct_us.saturating_add(encoded.timings.mct_us);
                self.dwt_us = self.dwt_us.saturating_add(encoded.timings.dwt_us);
                self.quantize_us = self.quantize_us.saturating_add(encoded.timings.quantize_us);
                self.ht_encode_us = self
                    .ht_encode_us
                    .saturating_add(encoded.timings.ht_encode_us);
                self.packetize_us = self
                    .packetize_us
                    .saturating_add(encoded.timings.packetize_us);
            }
            emit_cuda_encode_route!(
                ("op", "encode_htj2k_tile"),
                ("decision", "cuda_dispatch"),
                ("components", job.num_components),
                ("blocks", encoded.ht_code_block_jobs),
            );
            return Ok(Some(encoded.tile_data));
        }
        #[cfg(not(feature = "cuda-runtime"))]
        let _ = job;
        emit_cuda_encode_route!(
            ("op", "encode_htj2k_tile"),
            ("decision", "cpu_fallback"),
            ("reason", "cuda_unavailable"),
        );
        Ok(None)
    }

    fn encode_ht_subband(
        &mut self,
        job: J2kHtSubbandEncodeJob<'_>,
    ) -> core::result::Result<Option<Vec<EncodedHtJ2kCodeBlock>>, &'static str> {
        let code_block_count = ht_subband_code_block_count(job)?;
        self.ht_subband_attempts = self.ht_subband_attempts.saturating_add(1);
        self.quantize_subband_attempts = self.quantize_subband_attempts.saturating_add(1);
        self.ht_code_block_attempts = self.ht_code_block_attempts.saturating_add(code_block_count);
        if self.prefer_cpu_ht_subband {
            emit_cuda_encode_route!(
                ("op", "encode_ht_subband"),
                ("decision", "cpu_fallback"),
                ("reason", "prefer_cpu_ht_subband"),
            );
            return Ok(None);
        }
        #[cfg(feature = "cuda-runtime")]
        if let Some(context) = self.cuda_context()? {
            let resources = self.cuda_encode_resources(&context)?;
            let encoded =
                cuda_encode_ht_subband(&context, resources.as_ref(), job, self.collect_profile)?;
            let quantize_dispatches = encoded.quantize_dispatches;
            let encode_dispatches = encoded.encode.execution().kernel_dispatches();
            let outputs = encoded_ht_code_blocks_from_cuda(&encoded.encode);
            self.ht_subband_dispatches = self.ht_subband_dispatches.saturating_add(1);
            self.quantize_subband_dispatches = self
                .quantize_subband_dispatches
                .saturating_add(quantize_dispatches);
            self.ht_code_block_dispatches = self
                .ht_code_block_dispatches
                .saturating_add(encode_dispatches);
            if self.collect_profile {
                self.quantize_us = self.quantize_us.saturating_add(encoded.timings.quantize_us);
                self.ht_encode_us = self
                    .ht_encode_us
                    .saturating_add(encoded.timings.ht_encode_us);
            }
            emit_cuda_encode_route!(
                ("op", "encode_ht_subband"),
                ("decision", "cuda_dispatch"),
                ("width", job.width),
                ("height", job.height),
                ("blocks", code_block_count),
                ("quantize_dispatches", quantize_dispatches),
                ("encode_dispatches", encode_dispatches),
            );
            return Ok(Some(outputs));
        }
        #[cfg(not(feature = "cuda-runtime"))]
        let _ = job;
        emit_cuda_encode_route!(
            ("op", "encode_ht_subband"),
            ("decision", "cpu_fallback"),
            ("reason", "cuda_unavailable"),
        );
        Ok(None)
    }

    fn encode_packetization(
        &mut self,
        job: J2kPacketizationEncodeJob<'_>,
    ) -> core::result::Result<Option<Vec<u8>>, &'static str> {
        self.packetization_attempts = self.packetization_attempts.saturating_add(1);
        if self.prefer_cpu_packetization {
            emit_cuda_encode_route!(
                ("op", "encode_packetization"),
                ("decision", "cpu_fallback"),
                ("reason", "prefer_cpu_packetization"),
            );
            let _ = job;
            return Ok(None);
        }
        let plan = match flatten_cuda_htj2k_packetization_job(job) {
            Ok(plan) => plan,
            Err(reason) => {
                emit_cuda_encode_route!(
                    ("op", "encode_packetization"),
                    ("decision", "cpu_fallback"),
                    ("reason", reason),
                );
                return Ok(None);
            }
        };
        #[cfg(feature = "cuda-runtime")]
        if let Some(context) = self.cuda_context()? {
            let packets = cuda_packetization_packets(&plan);
            let subbands = cuda_packetization_subbands(&plan);
            let blocks = cuda_packetization_blocks(&plan);
            let tag_states = cuda_packetization_tag_states(&plan);
            let tag_nodes = cuda_packetization_tag_nodes(&plan);
            let packetized = context
                .packetize_htj2k_cleanup_packets_with_tag_state(
                    &plan.payload,
                    &packets,
                    &subbands,
                    &blocks,
                    &tag_states,
                    &tag_nodes,
                )
                .map_err(|_| "CUDA HTJ2K packetization kernel failed")?;
            let dispatches = packetized.execution().kernel_dispatches();
            let packetize_us = packetized.stage_timings().packetize_us;
            self.packetization_dispatches =
                self.packetization_dispatches.saturating_add(dispatches);
            if self.collect_profile {
                self.packetize_us = self.packetize_us.saturating_add(packetize_us);
            }
            emit_cuda_encode_route!(
                ("op", "encode_packetization"),
                ("decision", "cuda_dispatch"),
                ("packets", packets.len()),
                ("dispatches", dispatches),
            );
            return Ok(Some(packetized.data().to_vec()));
        }
        #[cfg(not(feature = "cuda-runtime"))]
        let _ = plan;
        emit_cuda_encode_route!(
            ("op", "encode_packetization"),
            ("decision", "cpu_fallback"),
            ("reason", "unsupported_stage"),
        );
        Ok(None)
    }
}
