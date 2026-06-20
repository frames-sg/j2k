use crate::{
    error::CudaError,
    htj2k_decode::{
        CudaHtj2kCleanupMultiKernelJob, CudaHtj2kCodeBlockKernelJob, CudaHtj2kDequantizeKernelJob,
        CudaHtj2kStatus,
    },
    htj2k_encode::{
        CudaHtj2kEncodeCompactJob, CudaHtj2kEncodeKernelJob, CudaHtj2kEncodeMultiInputKernelJob,
        CudaHtj2kEncodeParams, CudaHtj2kEncodeStatus,
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
        CudaJpegDecodeStatus, CudaJpegEntropyCheckpoint, CudaJpegEntropyOverflowState,
        CudaJpegEntropySyncState, CudaJpegHuffmanTable,
    },
};
use j2k_core::GpuAbi;

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
    CudaHtj2kEncodeParams,
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

pub(crate) fn f32_slice_as_bytes(samples: &[f32]) -> &[u8] {
    <f32 as GpuAbi>::slice_as_bytes(samples)
}

pub(crate) fn f32_slice_as_bytes_mut(samples: &mut [f32]) -> &mut [u8] {
    <f32 as GpuAbi>::slice_as_bytes_mut(samples)
}

pub(crate) fn i16_slice_as_bytes(samples: &[i16]) -> &[u8] {
    <i16 as GpuAbi>::slice_as_bytes(samples)
}

pub(crate) fn i32_slice_as_bytes(samples: &[i32]) -> &[u8] {
    <i32 as GpuAbi>::slice_as_bytes(samples)
}

pub(crate) fn i32_slice_as_bytes_mut(samples: &mut [i32]) -> &mut [u8] {
    <i32 as GpuAbi>::slice_as_bytes_mut(samples)
}

pub(crate) fn u16_slice_as_bytes(samples: &[u16]) -> &[u8] {
    <u16 as GpuAbi>::slice_as_bytes(samples)
}

#[cfg_attr(not(j2k_cuda_jpeg_decode_ptx_built), allow(dead_code))]
pub(crate) fn cuda_jpeg_huffman_table_as_bytes(table: &CudaJpegHuffmanTable) -> &[u8] {
    <CudaJpegHuffmanTable as GpuAbi>::as_bytes(table)
}

#[cfg_attr(not(j2k_cuda_jpeg_decode_ptx_built), allow(dead_code))]
pub(crate) fn cuda_jpeg_entropy_checkpoints_as_bytes(
    checkpoints: &[CudaJpegEntropyCheckpoint],
) -> &[u8] {
    <CudaJpegEntropyCheckpoint as GpuAbi>::slice_as_bytes(checkpoints)
}

#[cfg_attr(not(j2k_cuda_jpeg_decode_ptx_built), allow(dead_code))]
pub(crate) fn cuda_jpeg_decode_statuses_as_bytes(statuses: &[CudaJpegDecodeStatus]) -> &[u8] {
    <CudaJpegDecodeStatus as GpuAbi>::slice_as_bytes(statuses)
}

#[cfg_attr(not(j2k_cuda_jpeg_decode_ptx_built), allow(dead_code))]
pub(crate) fn cuda_jpeg_decode_statuses_as_bytes_mut(
    statuses: &mut [CudaJpegDecodeStatus],
) -> &mut [u8] {
    <CudaJpegDecodeStatus as GpuAbi>::slice_as_bytes_mut(statuses)
}

#[cfg_attr(not(j2k_cuda_jpeg_decode_ptx_built), allow(dead_code))]
pub(crate) fn cuda_jpeg_entropy_sync_states_as_bytes(states: &[CudaJpegEntropySyncState]) -> &[u8] {
    <CudaJpegEntropySyncState as GpuAbi>::slice_as_bytes(states)
}

#[cfg_attr(not(j2k_cuda_jpeg_decode_ptx_built), allow(dead_code))]
pub(crate) fn cuda_jpeg_entropy_sync_states_as_bytes_mut(
    states: &mut [CudaJpegEntropySyncState],
) -> &mut [u8] {
    <CudaJpegEntropySyncState as GpuAbi>::slice_as_bytes_mut(states)
}

#[cfg_attr(not(j2k_cuda_jpeg_decode_ptx_built), allow(dead_code))]
pub(crate) fn cuda_jpeg_entropy_overflow_states_as_bytes(
    states: &[CudaJpegEntropyOverflowState],
) -> &[u8] {
    <CudaJpegEntropyOverflowState as GpuAbi>::slice_as_bytes(states)
}

#[cfg_attr(not(j2k_cuda_jpeg_decode_ptx_built), allow(dead_code))]
pub(crate) fn cuda_jpeg_entropy_overflow_states_as_bytes_mut(
    states: &mut [CudaJpegEntropyOverflowState],
) -> &mut [u8] {
    <CudaJpegEntropyOverflowState as GpuAbi>::slice_as_bytes_mut(states)
}

pub(crate) fn store_gray8_job_as_bytes(job: &CudaJ2kStoreGray8Job) -> &[u8] {
    <CudaJ2kStoreGray8Job as GpuAbi>::as_bytes(job)
}

pub(crate) fn store_gray16_job_as_bytes(job: &CudaJ2kStoreGray16Job) -> &[u8] {
    <CudaJ2kStoreGray16Job as GpuAbi>::as_bytes(job)
}

pub(crate) fn inverse_mct_job_as_bytes(job: &CudaJ2kInverseMctJob) -> &[u8] {
    <CudaJ2kInverseMctJob as GpuAbi>::as_bytes(job)
}

pub(crate) fn store_rgb8_job_as_bytes(job: &CudaJ2kStoreRgb8Job) -> &[u8] {
    <CudaJ2kStoreRgb8Job as GpuAbi>::as_bytes(job)
}

pub(crate) fn store_rgb16_job_as_bytes(job: &CudaJ2kStoreRgb16Job) -> &[u8] {
    <CudaJ2kStoreRgb16Job as GpuAbi>::as_bytes(job)
}

pub(crate) fn store_rgb8_mct_batch_jobs_as_bytes(jobs: &[CudaJ2kStoreRgb8MctBatchJob]) -> &[u8] {
    <CudaJ2kStoreRgb8MctBatchJob as GpuAbi>::slice_as_bytes(jobs)
}

pub(crate) fn store_rgb16_mct_job_as_bytes(job: &CudaJ2kStoreRgb16MctJob) -> &[u8] {
    <CudaJ2kStoreRgb16MctJob as GpuAbi>::as_bytes(job)
}

pub(crate) fn htj2k_encode_params_as_bytes(params: &CudaHtj2kEncodeParams) -> &[u8] {
    <CudaHtj2kEncodeParams as GpuAbi>::as_bytes(params)
}

pub(crate) fn htj2k_encode_status_as_bytes(status: &CudaHtj2kEncodeStatus) -> &[u8] {
    <CudaHtj2kEncodeStatus as GpuAbi>::as_bytes(status)
}

pub(crate) fn htj2k_encode_status_as_bytes_mut(status: &mut CudaHtj2kEncodeStatus) -> &mut [u8] {
    <CudaHtj2kEncodeStatus as GpuAbi>::slice_as_bytes_mut(std::slice::from_mut(status))
}

pub(crate) fn htj2k_encode_statuses_byte_len(count: usize) -> Result<usize, CudaError> {
    count
        .checked_mul(std::mem::size_of::<CudaHtj2kEncodeStatus>())
        .ok_or(CudaError::LengthTooLarge { len: count })
}

pub(crate) fn htj2k_encode_jobs_as_bytes(jobs: &[CudaHtj2kEncodeKernelJob]) -> &[u8] {
    <CudaHtj2kEncodeKernelJob as GpuAbi>::slice_as_bytes(jobs)
}

pub(crate) fn htj2k_encode_multi_input_jobs_as_bytes(
    jobs: &[CudaHtj2kEncodeMultiInputKernelJob],
) -> &[u8] {
    <CudaHtj2kEncodeMultiInputKernelJob as GpuAbi>::slice_as_bytes(jobs)
}

pub(crate) fn htj2k_encode_compact_jobs_as_bytes(jobs: &[CudaHtj2kEncodeCompactJob]) -> &[u8] {
    <CudaHtj2kEncodeCompactJob as GpuAbi>::slice_as_bytes(jobs)
}

pub(crate) fn htj2k_encode_statuses_as_bytes_mut(
    statuses: &mut [CudaHtj2kEncodeStatus],
) -> &mut [u8] {
    <CudaHtj2kEncodeStatus as GpuAbi>::slice_as_bytes_mut(statuses)
}

pub(crate) fn htj2k_packetization_packets_as_bytes(
    packets: &[CudaHtj2kPacketizationKernelPacket],
) -> &[u8] {
    <CudaHtj2kPacketizationKernelPacket as GpuAbi>::slice_as_bytes(packets)
}

pub(crate) fn htj2k_packetization_subbands_as_bytes(
    subbands: &[CudaHtj2kPacketizationSubband],
) -> &[u8] {
    <CudaHtj2kPacketizationSubband as GpuAbi>::slice_as_bytes(subbands)
}

pub(crate) fn htj2k_packetization_blocks_as_bytes(blocks: &[CudaHtj2kPacketizationBlock]) -> &[u8] {
    <CudaHtj2kPacketizationBlock as GpuAbi>::slice_as_bytes(blocks)
}

pub(crate) fn htj2k_packetization_subband_tag_states_as_bytes(
    states: &[CudaHtj2kPacketizationSubbandTagState],
) -> &[u8] {
    <CudaHtj2kPacketizationSubbandTagState as GpuAbi>::slice_as_bytes(states)
}

pub(crate) fn htj2k_packetization_tag_nodes_as_bytes(
    nodes: &[CudaHtj2kPacketizationTagNodeState],
) -> &[u8] {
    <CudaHtj2kPacketizationTagNodeState as GpuAbi>::slice_as_bytes(nodes)
}

pub(crate) fn htj2k_packetization_statuses_as_bytes(
    statuses: &[CudaHtj2kPacketizationStatus],
) -> &[u8] {
    <CudaHtj2kPacketizationStatus as GpuAbi>::slice_as_bytes(statuses)
}

pub(crate) fn htj2k_packetization_statuses_as_bytes_mut(
    statuses: &mut [CudaHtj2kPacketizationStatus],
) -> &mut [u8] {
    <CudaHtj2kPacketizationStatus as GpuAbi>::slice_as_bytes_mut(statuses)
}

pub(crate) fn htj2k_jobs_as_bytes(jobs: &[CudaHtj2kCodeBlockKernelJob]) -> &[u8] {
    <CudaHtj2kCodeBlockKernelJob as GpuAbi>::slice_as_bytes(jobs)
}

pub(crate) fn htj2k_cleanup_multi_jobs_as_bytes(jobs: &[CudaHtj2kCleanupMultiKernelJob]) -> &[u8] {
    <CudaHtj2kCleanupMultiKernelJob as GpuAbi>::slice_as_bytes(jobs)
}

pub(crate) fn htj2k_dequantize_jobs_as_bytes(jobs: &[CudaHtj2kDequantizeKernelJob]) -> &[u8] {
    <CudaHtj2kDequantizeKernelJob as GpuAbi>::slice_as_bytes(jobs)
}

pub(crate) fn htj2k_statuses_byte_len(count: usize) -> Result<usize, CudaError> {
    count
        .checked_mul(std::mem::size_of::<CudaHtj2kStatus>())
        .ok_or(CudaError::LengthTooLarge { len: count })
}

pub(crate) fn htj2k_statuses_as_bytes_mut(statuses: &mut [CudaHtj2kStatus]) -> &mut [u8] {
    <CudaHtj2kStatus as GpuAbi>::slice_as_bytes_mut(statuses)
}

pub(crate) fn idwt_job_as_bytes(job: &CudaJ2kIdwtJob) -> &[u8] {
    <CudaJ2kIdwtJob as GpuAbi>::as_bytes(job)
}

pub(crate) fn idwt_multi_jobs_as_bytes(jobs: &[CudaJ2kIdwtMultiKernelJob]) -> &[u8] {
    <CudaJ2kIdwtMultiKernelJob as GpuAbi>::slice_as_bytes(jobs)
}
