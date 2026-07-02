// SPDX-License-Identifier: MIT OR Apache-2.0

//! CUDA codec engine and Driver API runtime used by J2K CUDA adapter crates.

#![deny(unsafe_op_in_unsafe_fn)]
#![deny(missing_docs)]
#![warn(unreachable_pub)]

macro_rules! cuda_kernel_params {
    ($($arg:ident),+ $(,)?) => {
        [$(cuda_kernel_param(&mut $arg)),+]
    };
}

macro_rules! impl_cuda_htj2k_encoded_status_accessors {
    () => {
        /// HTJ2K cleanup segment length in bytes.
        pub fn cleanup_length(&self) -> u32 {
            htj2k_encoded_cleanup_length(self.status)
        }

        /// HTJ2K refinement segment length in bytes.
        pub fn refinement_length(&self) -> u32 {
            htj2k_encoded_refinement_length(self.status)
        }

        /// Number of coding passes in the encoded payload.
        pub fn num_coding_passes(&self) -> u8 {
            htj2k_encoded_num_coding_passes(self.status)
        }

        /// Number of missing most-significant bitplanes.
        pub fn num_zero_bitplanes(&self) -> u8 {
            htj2k_encoded_num_zero_bitplanes(self.status)
        }

        /// Kernel status row downloaded after dispatch.
        pub fn status(&self) -> CudaHtj2kEncodeStatus {
            self.status
        }

        /// CUDA execution counters for the encode dispatch.
        pub fn execution(&self) -> CudaExecutionStats {
            self.execution
        }

        /// CUDA event timings for the encode dispatch.
        pub fn stage_timings(&self) -> CudaHtj2kEncodeStageTimings {
            self.stage_timings
        }
    };
}

mod build_flags;
mod bytes;
mod context;
mod driver;
mod error;
mod execution;
mod htj2k_decode;
mod htj2k_encode;
mod htj2k_packetize;
mod j2k_decode;
mod j2k_encode;
mod jpeg;
mod kernels;
mod memory;
#[cfg(test)]
mod tests;
mod transcode;

pub use build_flags::transcode_kernels_built;
pub use context::{
    CudaContext, CudaHtj2kCompactEncodedCodeBlock, CudaHtj2kCompactEncodedCodeBlocks,
    CudaKernelModule, CudaKernelName,
};
pub use error::CudaError;
pub use execution::{
    CudaEvent, CudaExecutionStats, CudaKernelBatchOutput, CudaKernelContiguousBatchOutput,
    CudaKernelOutput, CudaPooledKernelOutput, CudaQueuedExecution, CudaStream,
};
pub use htj2k_decode::{
    CudaHtj2kCleanupTarget, CudaHtj2kCodeBlockJob, CudaHtj2kDecodeOutput, CudaHtj2kDecodeResources,
    CudaHtj2kDecodeStageTimings, CudaHtj2kDecodeTableResources, CudaHtj2kDecodeTables,
    CudaHtj2kDequantizeTarget, CudaHtj2kStatus, CudaPooledHtj2kDecodeOutput,
    CudaQueuedHtj2kCleanup,
};
pub use htj2k_encode::{
    CudaHtj2kEncodeCodeBlockJob, CudaHtj2kEncodeCodeBlockRegionJob, CudaHtj2kEncodeResidentTarget,
    CudaHtj2kEncodeResources, CudaHtj2kEncodeStageTimings, CudaHtj2kEncodeStatus,
    CudaHtj2kEncodeTables, CudaHtj2kEncodedCodeBlock, CudaHtj2kEncodedCodeBlocks,
};
pub use htj2k_packetize::{
    CudaHtj2kPacketizationBlock, CudaHtj2kPacketizationPacket, CudaHtj2kPacketizationStageTimings,
    CudaHtj2kPacketizationStatus, CudaHtj2kPacketizationSubband,
    CudaHtj2kPacketizationSubbandTagState, CudaHtj2kPacketizationTagNodeState,
    CudaHtj2kPacketizedTile,
};
pub use j2k_decode::{
    CudaJ2kIdwtJob, CudaJ2kIdwtTarget, CudaJ2kInverseMctJob, CudaJ2kRect, CudaJ2kStoreGray16Job,
    CudaJ2kStoreGray8Job, CudaJ2kStoreRgb16Job, CudaJ2kStoreRgb16MctJob, CudaJ2kStoreRgb8Job,
    CudaJ2kStoreRgb8MctJob, CudaJ2kStoreRgb8MctTarget, CudaJ2kStridedInterleavedPixels,
};
pub use j2k_encode::{
    CudaDwt53LevelShape, CudaDwt53Output, CudaDwt97BatchStageTimings, CudaDwt97Output,
    CudaJ2kDeinterleavedComponents, CudaJ2kQuantizeJob, CudaJ2kQuantizeSubbandRegionJob,
    CudaJ2kQuantizedSubband, CudaJ2kResidentComponents, CudaJ2kResidentQuantizedSubband,
    CudaResidentDwt53Output, CudaResidentDwt97Output,
};
pub use jpeg::{
    CudaJpeg420Rgb8DecodePlan, CudaJpegBaselineEncodeFormat, CudaJpegBaselineEncodeHuffmanTable,
    CudaJpegBaselineEncodeParams, CudaJpegBaselineEntropyEncodeBatchJob,
    CudaJpegBaselineEntropyEncodeJob, CudaJpegChunkedEntropyConfig, CudaJpegChunkedEntropyPlan,
    CudaJpegChunkedEntropyReport, CudaJpegEntropyCheckpoint, CudaJpegEntropyOverflowState,
    CudaJpegEntropySyncState, CudaJpegHuffmanTable, CudaJpegRgb8DecodePlan, CudaJpegRgb8Sampling,
};
pub use memory::{
    CudaBufferPool, CudaBufferPoolTakeTrace, CudaDeviceBuffer, CudaDeviceBufferRange,
    CudaDeviceBufferView, CudaDeviceBufferViewMut, CudaPinnedHostBuffer, CudaPooledDeviceBuffer,
};
pub use transcode::{
    CudaHtj2k97CodeblockBands, CudaHtj2k97DeviceCodeblockBands, CudaHtj2k97QuantizeParams,
    CudaTranscodeDwt97Bands, CudaTranscodeReversible53Bands,
};

#[cfg(test)]
pub(crate) use bytes::{
    f32_slice_as_bytes, f32_slice_as_bytes_mut, htj2k_cleanup_multi_jobs_as_bytes,
    i32_slice_as_bytes, i32_slice_as_bytes_mut,
};
#[cfg(test)]
pub(crate) use context::HTJ2K_UVLC_ENCODE_TABLE_BYTES;
#[cfg(test)]
pub(crate) use htj2k_decode::{
    htj2k_decode_multi_cleanup_dequant_kernel_for_jobs, htj2k_decode_multi_kernel_for_jobs,
    CudaHtj2kCleanupMultiKernelJob, HTJ2K_STATUS_OK, HTJ2K_STATUS_UNSUPPORTED,
};
#[cfg(test)]
pub(crate) use htj2k_encode::{
    htj2k_encode_compact_jobs, CudaHtj2kEncodeCompactJob, CudaHtj2kEncodeKernelJob,
    HTJ2K_ENCODE_OUTPUT_CAPACITY,
};
#[cfg(test)]
pub(crate) use j2k_decode::{
    checked_f32_words_byte_len, format_idwt_batch_trace_row, idwt_batch_kernel_mode,
    idwt_batch_trace_row, idwt_batch_uses_cooperative_53, CudaJ2kIdwtBatchKernelMode,
    CudaJ2kIdwtMultiKernelJob,
};
#[cfg(test)]
pub(crate) use jpeg::jpeg_entropy_overflow_count;
#[cfg(test)]
pub(crate) use memory::{copy_pooled_bytes_to_vec_uninit, pool_fit_buffer_index_by_len};
#[cfg(test)]
pub(crate) use transcode::{should_use_pinned_pooled_i16_upload, validate_dct_block_grid};
