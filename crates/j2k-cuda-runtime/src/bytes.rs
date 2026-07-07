use crate::{
    error::CudaError,
    htj2k_decode::{
        CudaHtj2kCleanupMultiKernelJob, CudaHtj2kCodeBlockKernelJob, CudaHtj2kDequantizeKernelJob,
        CudaHtj2kStatus,
    },
    htj2k_encode::{
        CudaHtj2kEncodeCompactJob, CudaHtj2kEncodeKernelJob, CudaHtj2kEncodeMultiInputKernelJob,
        CudaHtj2kEncodeStatus,
    },
    htj2k_packetize::{
        CudaHtj2kPacketizationBlock, CudaHtj2kPacketizationKernelPacket,
        CudaHtj2kPacketizationStatus, CudaHtj2kPacketizationSubband,
        CudaHtj2kPacketizationSubbandTagState, CudaHtj2kPacketizationTagNodeState,
    },
    j2k_decode::{
        CudaJ2kIdwtJob, CudaJ2kIdwtMultiKernelJob, CudaJ2kInverseMctJob, CudaJ2kStoreGray16Job,
        CudaJ2kStoreGray8Job, CudaJ2kStoreRgb16Job, CudaJ2kStoreRgb16MctJob, CudaJ2kStoreRgb8Job,
        CudaJ2kStoreRgb8MctBatchJob,
    },
    jpeg::{
        CudaJpegBaselineEncodeHuffmanTable, CudaJpegBaselineEncodeParams,
        CudaJpegBaselineEncodeStatus, CudaJpegDecodeStatus, CudaJpegEntropyCheckpoint,
        CudaJpegEntropyOverflowState, CudaJpegEntropySyncState, CudaJpegHuffmanTable,
    },
};
use j2k_core::accelerator::GpuAbi;

macro_rules! impl_cuda_gpu_abi {
    ($($ty:ty),+ $(,)?) => {
        $(
            // SAFETY: These repr(C) structs are copied byte-for-byte to CUDA kernels with matching ABI tests.
            unsafe impl GpuAbi for $ty {
                const NAME: &'static str = stringify!($ty);
            }
        )+
    };
}

impl_cuda_gpu_abi! {
    CudaJpegHuffmanTable,
    CudaJpegEntropyCheckpoint,
    CudaJpegDecodeStatus,
    CudaJpegEntropySyncState,
    CudaJpegEntropyOverflowState,
    CudaJpegBaselineEncodeParams,
    CudaJpegBaselineEncodeHuffmanTable,
    CudaJpegBaselineEncodeStatus,
    CudaHtj2kEncodeStatus,
    CudaHtj2kEncodeKernelJob,
    CudaHtj2kEncodeMultiInputKernelJob,
    CudaHtj2kEncodeCompactJob,
    CudaHtj2kPacketizationKernelPacket,
    CudaHtj2kPacketizationSubband,
    CudaHtj2kPacketizationBlock,
    CudaHtj2kPacketizationSubbandTagState,
    CudaHtj2kPacketizationTagNodeState,
    CudaHtj2kPacketizationStatus,
    CudaHtj2kCodeBlockKernelJob,
    CudaHtj2kCleanupMultiKernelJob,
    CudaHtj2kDequantizeKernelJob,
    CudaHtj2kStatus,
    CudaJ2kIdwtJob,
    CudaJ2kIdwtMultiKernelJob,
    CudaJ2kStoreGray8Job,
    CudaJ2kStoreGray16Job,
    CudaJ2kInverseMctJob,
    CudaJ2kStoreRgb8Job,
    CudaJ2kStoreRgb16Job,
    CudaJ2kStoreRgb8MctBatchJob,
    CudaJ2kStoreRgb16MctJob,
}

macro_rules! gpu_ref_bytes {
    ($($(#[$attr:meta])* $name:ident: $ty:ty;)+) => {
        $(
            $(#[$attr])*
            pub(crate) fn $name(value: &$ty) -> &[u8] {
                <$ty as GpuAbi>::as_bytes(value)
            }
        )+
    };
}

macro_rules! gpu_slice_bytes {
    ($($(#[$attr:meta])* $name:ident: $ty:ty;)+) => {
        $(
            $(#[$attr])*
            pub(crate) fn $name(values: &[$ty]) -> &[u8] {
                <$ty as GpuAbi>::slice_as_bytes(values)
            }
        )+
    };
}

macro_rules! gpu_slice_bytes_mut {
    ($($(#[$attr:meta])* $name:ident: $ty:ty;)+) => {
        $(
            $(#[$attr])*
            pub(crate) fn $name(values: &mut [$ty]) -> &mut [u8] {
                <$ty as GpuAbi>::slice_as_bytes_mut(values)
            }
        )+
    };
}

gpu_ref_bytes! {
    #[cfg_attr(not(feature = "cuda-oxide-jpeg-decode"), allow(dead_code))]
    cuda_jpeg_huffman_table_as_bytes: CudaJpegHuffmanTable;
    #[cfg_attr(not(feature = "cuda-oxide-jpeg-encode"), allow(dead_code))]
    cuda_jpeg_baseline_encode_huffman_table_as_bytes: CudaJpegBaselineEncodeHuffmanTable;
    store_gray8_job_as_bytes: CudaJ2kStoreGray8Job;
    store_gray16_job_as_bytes: CudaJ2kStoreGray16Job;
    inverse_mct_job_as_bytes: CudaJ2kInverseMctJob;
    store_rgb8_job_as_bytes: CudaJ2kStoreRgb8Job;
    store_rgb16_job_as_bytes: CudaJ2kStoreRgb16Job;
    store_rgb16_mct_job_as_bytes: CudaJ2kStoreRgb16MctJob;
    idwt_job_as_bytes: CudaJ2kIdwtJob;
}

gpu_slice_bytes! {
    f32_slice_as_bytes: f32;
    i16_slice_as_bytes: i16;
    i32_slice_as_bytes: i32;
    u16_slice_as_bytes: u16;
    #[cfg_attr(not(feature = "cuda-oxide-jpeg-decode"), allow(dead_code))]
    cuda_jpeg_entropy_checkpoints_as_bytes: CudaJpegEntropyCheckpoint;
    #[cfg_attr(not(feature = "cuda-oxide-jpeg-decode"), allow(dead_code))]
    cuda_jpeg_decode_statuses_as_bytes: CudaJpegDecodeStatus;
    #[cfg_attr(not(feature = "cuda-oxide-jpeg-decode"), allow(dead_code))]
    cuda_jpeg_entropy_sync_states_as_bytes: CudaJpegEntropySyncState;
    #[cfg_attr(not(feature = "cuda-oxide-jpeg-decode"), allow(dead_code))]
    cuda_jpeg_entropy_overflow_states_as_bytes: CudaJpegEntropyOverflowState;
    #[cfg_attr(not(feature = "cuda-oxide-jpeg-encode"), allow(dead_code))]
    cuda_jpeg_baseline_encode_params_as_bytes: CudaJpegBaselineEncodeParams;
    #[cfg_attr(not(feature = "cuda-oxide-jpeg-encode"), allow(dead_code))]
    cuda_jpeg_baseline_encode_statuses_as_bytes: CudaJpegBaselineEncodeStatus;
    store_rgb8_mct_batch_jobs_as_bytes: CudaJ2kStoreRgb8MctBatchJob;
    htj2k_encode_jobs_as_bytes: CudaHtj2kEncodeKernelJob;
    htj2k_encode_multi_input_jobs_as_bytes: CudaHtj2kEncodeMultiInputKernelJob;
    htj2k_encode_compact_jobs_as_bytes: CudaHtj2kEncodeCompactJob;
    htj2k_packetization_packets_as_bytes: CudaHtj2kPacketizationKernelPacket;
    htj2k_packetization_subbands_as_bytes: CudaHtj2kPacketizationSubband;
    htj2k_packetization_blocks_as_bytes: CudaHtj2kPacketizationBlock;
    htj2k_packetization_subband_tag_states_as_bytes: CudaHtj2kPacketizationSubbandTagState;
    htj2k_packetization_tag_nodes_as_bytes: CudaHtj2kPacketizationTagNodeState;
    htj2k_packetization_statuses_as_bytes: CudaHtj2kPacketizationStatus;
    htj2k_jobs_as_bytes: CudaHtj2kCodeBlockKernelJob;
    htj2k_cleanup_multi_jobs_as_bytes: CudaHtj2kCleanupMultiKernelJob;
    htj2k_dequantize_jobs_as_bytes: CudaHtj2kDequantizeKernelJob;
    idwt_multi_jobs_as_bytes: CudaJ2kIdwtMultiKernelJob;
}

gpu_slice_bytes_mut! {
    f32_slice_as_bytes_mut: f32;
    i32_slice_as_bytes_mut: i32;
    #[cfg_attr(not(feature = "cuda-oxide-jpeg-decode"), allow(dead_code))]
    cuda_jpeg_decode_statuses_as_bytes_mut: CudaJpegDecodeStatus;
    #[cfg_attr(not(feature = "cuda-oxide-jpeg-decode"), allow(dead_code))]
    cuda_jpeg_entropy_sync_states_as_bytes_mut: CudaJpegEntropySyncState;
    #[cfg_attr(not(feature = "cuda-oxide-jpeg-decode"), allow(dead_code))]
    cuda_jpeg_entropy_overflow_states_as_bytes_mut: CudaJpegEntropyOverflowState;
    #[cfg_attr(not(feature = "cuda-oxide-jpeg-encode"), allow(dead_code))]
    cuda_jpeg_baseline_encode_statuses_as_bytes_mut: CudaJpegBaselineEncodeStatus;
    htj2k_encode_statuses_as_bytes_mut: CudaHtj2kEncodeStatus;
    htj2k_packetization_statuses_as_bytes_mut: CudaHtj2kPacketizationStatus;
    htj2k_statuses_as_bytes_mut: CudaHtj2kStatus;
}

pub(crate) fn htj2k_encode_statuses_byte_len(count: usize) -> Result<usize, CudaError> {
    count
        .checked_mul(std::mem::size_of::<CudaHtj2kEncodeStatus>())
        .ok_or(CudaError::LengthTooLarge { len: count })
}

pub(crate) fn htj2k_statuses_byte_len(count: usize) -> Result<usize, CudaError> {
    count
        .checked_mul(std::mem::size_of::<CudaHtj2kStatus>())
        .ok_or(CudaError::LengthTooLarge { len: count })
}
