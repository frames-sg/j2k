// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::memory::CudaDeviceBuffer;
use j2k_codec_math::classic::{
    MQ_QE_VALUES, PACKED_MQ_TRANSITION_VALUES, PACKED_SIGN_CONTEXT_LOOKUP, ZERO_CTX_HH_LOOKUP,
    ZERO_CTX_HL_LOOKUP, ZERO_CTX_LL_LH_LOOKUP,
};

/// One classic JPEG 2000 Tier-1 code-block decode job.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CudaClassicCodeBlockJob {
    /// Byte offset of this code-block in the shared compressed payload.
    pub payload_offset: u64,
    /// Byte length of this code-block payload.
    pub payload_len: u32,
    /// First segment in the target's segment slice.
    pub segment_start: u32,
    /// Number of contiguous segment records.
    pub segment_count: u32,
    /// Code-block width.
    pub width: u32,
    /// Code-block height.
    pub height: u32,
    /// Output row stride in f32 coefficients.
    pub output_stride: u32,
    /// First output coefficient.
    pub output_offset: u32,
    /// Missing most-significant bitplanes.
    pub missing_bitplanes: u32,
    /// Total bitplanes in the sub-band.
    pub total_bitplanes: u32,
    /// Number of coding passes present.
    pub number_of_coding_passes: u32,
    /// JPEG 2000 sub-band tag: LL=0, HL=1, LH=2, HH=3.
    pub sub_band_type: u32,
    /// JPEG 2000 code-block style bits.
    pub style_flags: u32,
    /// Whether malformed entropy data is rejected.
    pub strict: bool,
    /// Fused coefficient dequantization multiplier.
    pub dequantization_step: f32,
}

/// One bounded classic Tier-1 pass segment.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaClassicSegment {
    /// Byte offset relative to the owning code-block payload.
    pub data_offset: u32,
    /// Segment byte length.
    pub data_length: u32,
    /// Inclusive first coding pass.
    pub start_coding_pass: u32,
    /// Exclusive final coding pass.
    pub end_coding_pass: u32,
    /// True for MQ arithmetic coding; false for raw bypass coding.
    pub use_arithmetic: bool,
}

/// One device coefficient target and its classic Tier-1 work.
#[doc(hidden)]
#[derive(Clone, Copy, Debug)]
pub struct CudaClassicDecodeTarget<'a> {
    /// Device-resident f32 coefficient plane.
    pub coefficients: &'a CudaDeviceBuffer,
    /// Code-block jobs writing this plane.
    pub jobs: &'a [CudaClassicCodeBlockJob],
    /// Segment records referenced by the jobs.
    pub segments: &'a [CudaClassicSegment],
    /// Number of f32 words in the coefficient plane.
    pub output_words: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct CudaClassicKernelJob {
    pub(crate) output_ptr: u64,
    pub(crate) coded_offset: u32,
    pub(crate) coded_len: u32,
    pub(crate) segment_offset: u32,
    pub(crate) segment_count: u32,
    pub(crate) scratch_offset: u32,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) output_stride: u32,
    pub(crate) output_offset: u32,
    pub(crate) missing_msbs: u32,
    pub(crate) total_bitplanes: u32,
    pub(crate) number_of_coding_passes: u32,
    pub(crate) sub_band_type: u32,
    pub(crate) style_flags: u32,
    pub(crate) strict: u32,
    pub(crate) dequantization_step: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct CudaClassicKernelSegment {
    pub(crate) data_offset: u32,
    pub(crate) data_length: u32,
    pub(crate) start_coding_pass: u32,
    pub(crate) end_coding_pass: u32,
    pub(crate) use_arithmetic: u32,
}

/// Status returned for one classic Tier-1 code-block.
#[doc(hidden)]
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CudaClassicStatus {
    /// Zero on success.
    pub code: u32,
    /// Kernel-defined failure detail.
    pub detail: u32,
    pub(crate) reserved0: u32,
    pub(crate) reserved1: u32,
}

/// Timings for one resident classic Tier-1 decode dispatch.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CudaClassicDecodeStageTimings {
    /// Host-observed jobs and segments upload time, in microseconds.
    pub job_upload_us: u128,
    /// Host-observed lookup-table upload time, in microseconds.
    pub table_upload_us: u128,
    /// Classic Tier-1 CUDA kernel time, in microseconds.
    pub kernel_us: u128,
    /// Host-observed status download time, in microseconds.
    pub status_d2h_us: u128,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct CudaClassicKernelTables {
    pub(crate) mq_qe: [u32; 47],
    pub(crate) mq_transitions: [u32; 47],
    pub(crate) sign_contexts: [u16; 256],
    pub(crate) zero_contexts_ll_lh: [u8; 256],
    pub(crate) zero_contexts_hl: [u8; 256],
    pub(crate) zero_contexts_hh: [u8; 256],
}

pub(super) const CLASSIC_KERNEL_TABLES: CudaClassicKernelTables = CudaClassicKernelTables {
    mq_qe: MQ_QE_VALUES,
    mq_transitions: PACKED_MQ_TRANSITION_VALUES,
    sign_contexts: PACKED_SIGN_CONTEXT_LOOKUP,
    zero_contexts_ll_lh: ZERO_CTX_LL_LH_LOOKUP,
    zero_contexts_hl: ZERO_CTX_HL_LOOKUP,
    zero_contexts_hh: ZERO_CTX_HH_LOOKUP,
};
