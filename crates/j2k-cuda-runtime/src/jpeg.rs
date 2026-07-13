// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(test)]
mod abi_tests;
mod decode;
#[cfg(feature = "cuda-oxide-jpeg-decode")]
mod decode_launch;
#[cfg(feature = "cuda-oxide-jpeg-decode")]
mod decode_workspace;
mod diagnostics;
#[cfg(feature = "cuda-oxide-jpeg-decode")]
mod diagnostics_allocation;
#[cfg(feature = "cuda-oxide-jpeg-decode")]
mod diagnostics_execution;
mod encode;
#[cfg(any(feature = "cuda-oxide-jpeg-encode", test))]
mod encode_allocation;
#[cfg(feature = "cuda-oxide-jpeg-encode")]
mod encode_batch;
#[cfg(feature = "cuda-oxide-jpeg-encode")]
mod encode_launch;
mod encode_validation;
mod types;
mod validation;

pub(crate) use self::types::{
    CudaJpeg420Params, CudaJpegBaselineEncodeStatus, CudaJpegDecodeStatus,
    CudaJpegEntropyChunkParams,
};
pub use self::types::{
    CudaJpegBaselineEncodeFormat, CudaJpegBaselineEncodeHuffmanTable, CudaJpegBaselineEncodeParams,
    CudaJpegBaselineEntropyEncodeBatchJob, CudaJpegBaselineEntropyEncodeJob,
    CudaJpegChunkedEntropyConfig, CudaJpegChunkedEntropyPlan, CudaJpegChunkedEntropyReport,
    CudaJpegEntropyCheckpoint, CudaJpegEntropyOverflowState, CudaJpegEntropySyncState,
    CudaJpegHuffmanTable, CudaJpegRgb8DecodePlan, CudaJpegRgb8Sampling,
};
#[cfg(feature = "cuda-oxide-jpeg-decode")]
pub(crate) use self::{
    types::CudaJpegRgb8ValidatedPlan,
    validation::{
        jpeg_rgb8_kernel, validate_jpeg_entropy_chunk_plan, validate_jpeg_rgb8_plan,
        validate_jpeg_rgb8_plan_with_pitch,
    },
};

const _: [(); 32] = [(); core::mem::size_of::<CudaJpeg420Params>()];
const _: [(); 32] = [(); core::mem::size_of::<CudaJpegEntropyChunkParams>()];

#[cfg_attr(
    all(not(feature = "cuda-oxide-jpeg-decode"), not(test)),
    expect(
        dead_code,
        reason = "overflow accounting is used only by the JPEG decode path"
    )
)]
pub(crate) fn jpeg_entropy_overflow_count(subsequence_count: usize) -> usize {
    subsequence_count.saturating_sub(1)
}

#[cfg(test)]
mod structure_tests;
