// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_core::BackendKind;

use crate::SurfaceResidency;

mod emit;
mod trace;

/// Detailed route-overhead timings for strict CUDA HTJ2K decode.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
#[doc(hidden)]
pub struct CudaHtj2kDecodeProfileDetail {
    /// End-to-end profiled decode wall time.
    pub wall_total_us: u128,
    /// Sum of the reported decode stage timings.
    pub stage_sum_us: u128,
    /// CUDA table/resource upload time.
    pub table_upload_us: u128,
    /// CUDA compressed payload/resource upload time.
    ///
    /// This includes mixed resource upload calls that contain compressed
    /// payload bytes plus decode metadata. Metadata-only job upload is not
    /// split out until the CUDA runtime exposes separate timings.
    pub payload_upload_us: u128,
    /// CUDA decode job upload time, reserved as zero until split runtime timings exist.
    pub job_upload_us: u128,
    /// CUDA status download time, reserved as zero until split runtime timings exist.
    pub status_d2h_us: u128,
    /// CUDA output download time, reserved as zero until split runtime timings exist.
    pub output_d2h_us: u128,
    /// HT cleanup/refinement CUDA dispatch count.
    pub ht_dispatch_count: usize,
    /// Dequantization CUDA dispatch count.
    pub dequant_dispatch_count: usize,
    /// Inverse DWT CUDA dispatch count.
    pub idwt_dispatch_count: usize,
    /// Inverse MCT CUDA dispatch count.
    pub mct_dispatch_count: usize,
    /// Store/format conversion CUDA dispatch count.
    pub store_dispatch_count: usize,
}

impl CudaHtj2kDecodeProfileDetail {
    /// Create an empty decode profile detail.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

/// Structured stage timings for a strict CUDA HTJ2K operation.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
#[doc(hidden)]
pub struct CudaHtj2kProfileReport {
    /// CPU marker/box parse time.
    pub parse_us: u128,
    /// Native direct-plan construction time.
    pub plan_us: u128,
    /// Flat CUDA plan construction time.
    pub flatten_us: u128,
    /// Host-to-device upload time for payload and metadata.
    pub h2d_us: u128,
    /// HT cleanup kernel time.
    pub ht_cleanup_us: u128,
    /// HT refinement kernel time.
    pub ht_refine_us: u128,
    /// Dequantization kernel time.
    pub dequant_us: u128,
    /// Inverse DWT kernel time.
    pub idwt_us: u128,
    /// Inverse MCT kernel time.
    pub mct_us: u128,
    /// Store/format conversion kernel time.
    pub store_us: u128,
    /// Sum of measured decode stages.
    ///
    /// End-to-end wall time is reported in `detail.wall_total_us`.
    pub total_us: u128,
    /// Number of HTJ2K code blocks in the flat plan.
    pub block_count: usize,
    /// Number of compressed payload bytes uploaded to CUDA.
    pub payload_bytes: usize,
    /// Number of CUDA kernel dispatches.
    pub dispatch_count: usize,
    /// Surface residency represented by this profile.
    pub residency: SurfaceResidency,
    /// Detailed route-overhead profile for RCA.
    pub detail: CudaHtj2kDecodeProfileDetail,
}

impl CudaHtj2kProfileReport {
    /// Create an empty decode profile report.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Emit the report using `J2K_PROFILE_STAGES`, when enabled.
    #[cfg(feature = "cuda-runtime")]
    pub(crate) fn emit(&self, path: &str) {
        emit::emit_htj2k_profile_row(path, self);
        trace::export_trace_if_requested(path, self);
    }
}

/// Structured stage timings for a strict CUDA HTJ2K encode operation.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
#[doc(hidden)]
pub struct CudaHtj2kEncodeProfileReport {
    /// Pixel deinterleave and level-shift CUDA stage time.
    pub deinterleave_us: u128,
    /// Forward MCT CUDA stage time.
    pub mct_us: u128,
    /// Forward DWT CUDA stage time.
    pub dwt_us: u128,
    /// Quantization CUDA stage time.
    pub quantize_us: u128,
    /// HTJ2K cleanup code-block encode CUDA stage time.
    pub ht_encode_us: u128,
    /// HTJ2K packetization CUDA stage time.
    pub packetize_us: u128,
    /// Total wall time for the measured encode call.
    pub total_us: u128,
    /// Input pixel byte count.
    pub input_bytes: usize,
    /// Output codestream byte count.
    pub codestream_bytes: usize,
    /// Number of HTJ2K code blocks encoded.
    pub block_count: usize,
    /// Number of CUDA kernel dispatches.
    pub dispatch_count: usize,
    /// Backend that satisfied the encode request.
    pub backend: BackendKind,
}

impl Default for CudaHtj2kEncodeProfileReport {
    fn default() -> Self {
        Self {
            deinterleave_us: 0,
            mct_us: 0,
            dwt_us: 0,
            quantize_us: 0,
            ht_encode_us: 0,
            packetize_us: 0,
            total_us: 0,
            input_bytes: 0,
            codestream_bytes: 0,
            block_count: 0,
            dispatch_count: 0,
            backend: BackendKind::Cpu,
        }
    }
}

impl CudaHtj2kEncodeProfileReport {
    /// Create an empty encode profile report.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Emit the report using `J2K_PROFILE_STAGES`, when enabled.
    pub(crate) fn emit(&self, path: &str) {
        emit::emit_htj2k_encode_profile_row(path, self);
        trace::export_encode_trace_if_requested(path, self);
    }
}

#[cfg(feature = "cuda-runtime")]
pub(crate) use emit::{add_payload_resource_upload_us, finalize_decode_total_us};
pub(crate) use emit::{emit_optional_gpu_route_fields, profile_stages_enabled};
#[cfg(feature = "cuda-runtime")]
pub(crate) use j2k_profile::ProfileInstant;
pub(crate) use j2k_profile::{elapsed_us, profile_now};

#[cfg(test)]
mod tests;
