// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::error::CudaError;
use crate::htj2k_decode::{
    CudaHtj2kCleanupMultiKernelJob, CudaHtj2kCodeBlockKernelJob, CudaHtj2kDequantizeKernelJob,
    CudaHtj2kStatus,
};
use crate::htj2k_encode::{
    CudaHtj2kEncodeCompactJob, CudaHtj2kEncodeKernelJob, CudaHtj2kEncodeMultiInputKernelJob,
    CudaHtj2kEncodeStatus,
};
use crate::htj2k_packetize::{
    CudaHtj2kPacketizationBlock, CudaHtj2kPacketizationKernelPacket, CudaHtj2kPacketizationStatus,
    CudaHtj2kPacketizationSubband, CudaHtj2kPacketizationSubbandTagState,
    CudaHtj2kPacketizationTagNodeState,
};
use j2k_core::accelerator::GpuAbi;
use std::mem::{offset_of, size_of};

mod htj2k_encode_abi;
mod j2k_abi;

macro_rules! prove_gpu_abi_layout {
    ($ty:ty, $offset:expr;) => {
        let _: [(); size_of::<$ty>()] = [(); $offset];
    };
    ($ty:ty, $offset:expr; $field:ident: $field_ty:ty $(, $rest:ident: $rest_ty:ty)*) => {
        let _: [(); offset_of!($ty, $field)] = [(); $offset];
        prove_gpu_abi_layout!($ty, $offset + size_of::<$field_ty>(); $($rest: $rest_ty),*);
    };
}

macro_rules! impl_gpu_abi {
    ($($ty:ty { $first:ident: $first_ty:ty $(, $field:ident: $field_ty:ty)* $(,)? }),+ $(,)?) => {
        $(
            const _: () = {
                fn assert_field_types(value: &$ty) {
                    let _: &$first_ty = &value.$first;
                    $(let _: &$field_ty = &value.$field;)*
                }
                let _ = assert_field_types;
                prove_gpu_abi_layout!($ty, 0; $first: $first_ty $(, $field: $field_ty)*);
            };

            // SAFETY: compile-time offsets prove a padding-free repr(C)
            // representation composed solely of initialized numeric fields.
            unsafe impl GpuAbi for $ty {
                const NAME: &'static str = stringify!($ty);
            }
        )+
    };
}

impl_gpu_abi! {
    CudaHtj2kCodeBlockKernelJob {
        coded_offset: u32,
        width: u32,
        height: u32,
        coded_len: u32,
        cleanup_length: u32,
        refinement_length: u32,
        missing_msbs: u32,
        num_bitplanes: u32,
        reconstruction: u32,
        number_of_coding_passes: u32,
        output_stride: u32,
        output_offset: u32,
        dequantization_step: f32,
        stripe_causal: u32,
    },
    CudaHtj2kCleanupMultiKernelJob {
        output_ptr: u64,
        coded_offset: u32,
        width: u32,
        height: u32,
        coded_len: u32,
        cleanup_length: u32,
        refinement_length: u32,
        missing_msbs: u32,
        num_bitplanes: u32,
        number_of_coding_passes: u32,
        output_stride: u32,
        output_offset: u32,
        dequantization_step: f32,
        stripe_causal: u32,
        reconstruction: u32,
    },
    CudaHtj2kDequantizeKernelJob {
        output_ptr: u64,
        width: u32,
        height: u32,
        output_stride: u32,
        output_offset: u32,
        num_bitplanes: u32,
        reconstruction: u32,
        dequantization_step: f32,
        reserved_tail: u32,
    },
    CudaHtj2kStatus {
        code: u32,
        detail: u32,
        reserved0: u32,
        reserved1: u32,
    },
}

pub(crate) fn u16_slice_as_bytes(values: &[u16]) -> &[u8] {
    <u16 as GpuAbi>::slice_as_bytes(values)
}

pub(crate) fn f32_slice_as_bytes(values: &[f32]) -> &[u8] {
    <f32 as GpuAbi>::slice_as_bytes(values)
}

pub(crate) fn f32_slice_as_bytes_mut(values: &mut [f32]) -> &mut [u8] {
    <f32 as GpuAbi>::slice_as_bytes_mut(values)
}

pub(crate) fn i32_slice_as_bytes(values: &[i32]) -> &[u8] {
    <i32 as GpuAbi>::slice_as_bytes(values)
}

pub(crate) fn i32_slice_as_bytes_mut(values: &mut [i32]) -> &mut [u8] {
    <i32 as GpuAbi>::slice_as_bytes_mut(values)
}

pub(crate) fn htj2k_encode_statuses_as_bytes_mut(
    values: &mut [CudaHtj2kEncodeStatus],
) -> &mut [u8] {
    <CudaHtj2kEncodeStatus as GpuAbi>::slice_as_bytes_mut(values)
}

pub(crate) fn htj2k_packetization_statuses_as_bytes_mut(
    values: &mut [CudaHtj2kPacketizationStatus],
) -> &mut [u8] {
    <CudaHtj2kPacketizationStatus as GpuAbi>::slice_as_bytes_mut(values)
}

pub(crate) fn htj2k_encode_statuses_byte_len(count: usize) -> Result<usize, CudaError> {
    count
        .checked_mul(size_of::<CudaHtj2kEncodeStatus>())
        .ok_or(CudaError::LengthTooLarge { len: count })
}

macro_rules! j2k_ref_bytes {
    ($($name:ident: $ty:ty;)+) => {
        $(pub(crate) fn $name(value: &$ty) -> &[u8] {
            <$ty as GpuAbi>::as_bytes(value)
        })+
    };
}

macro_rules! j2k_slice_bytes {
    ($($name:ident: $ty:ty;)+) => {
        $(pub(crate) fn $name(values: &[$ty]) -> &[u8] {
            <$ty as GpuAbi>::slice_as_bytes(values)
        })+
    };
}

j2k_slice_bytes! {
    htj2k_encode_jobs_as_bytes: CudaHtj2kEncodeKernelJob;
    htj2k_encode_multi_input_jobs_as_bytes: CudaHtj2kEncodeMultiInputKernelJob;
    htj2k_encode_compact_jobs_as_bytes: CudaHtj2kEncodeCompactJob;
    htj2k_packetization_packets_as_bytes: CudaHtj2kPacketizationKernelPacket;
    htj2k_packetization_subbands_as_bytes: CudaHtj2kPacketizationSubband;
    htj2k_packetization_blocks_as_bytes: CudaHtj2kPacketizationBlock;
    htj2k_packetization_subband_tag_states_as_bytes: CudaHtj2kPacketizationSubbandTagState;
    htj2k_packetization_tag_nodes_as_bytes: CudaHtj2kPacketizationTagNodeState;
    htj2k_packetization_statuses_as_bytes: CudaHtj2kPacketizationStatus;
}

j2k_ref_bytes! {
    store_gray8_job_as_bytes: crate::j2k_decode::CudaJ2kStoreGray8Job;
    store_gray16_job_as_bytes: crate::j2k_decode::CudaJ2kStoreGray16Job;
    inverse_mct_job_as_bytes: crate::j2k_decode::CudaJ2kInverseMctJob;
    store_rgb8_job_as_bytes: crate::j2k_decode::CudaJ2kStoreRgb8Job;
    store_rgb16_job_as_bytes: crate::j2k_decode::CudaJ2kStoreRgb16Job;
    store_rgb16_mct_job_as_bytes: crate::j2k_decode::CudaJ2kStoreRgb16MctJob;
    idwt_job_as_bytes: crate::j2k_decode::CudaJ2kIdwtJob;
}

j2k_slice_bytes! {
    store_rgb8_mct_batch_jobs_as_bytes: crate::j2k_decode::CudaJ2kStoreRgb8MctBatchJob;
    store_rgb_native_batch_jobs_as_bytes: crate::j2k_decode::CudaJ2kStoreRgbNativeBatchJob;
    store_rgba_native_batch_jobs_as_bytes: crate::j2k_decode::CudaJ2kStoreRgbaNativeBatchJob;
    store_gray8_batch_jobs_as_bytes: crate::j2k_decode::CudaJ2kStoreGray8BatchJob;
    store_gray16_batch_jobs_as_bytes: crate::j2k_decode::CudaJ2kStoreGray16BatchJob;
    store_grayi16_batch_jobs_as_bytes: crate::j2k_decode::CudaJ2kStoreGrayI16BatchJob;
    idwt_multi_jobs_as_bytes: crate::j2k_decode::CudaJ2kIdwtMultiKernelJob;
}

pub(crate) fn htj2k_jobs_as_bytes(values: &[CudaHtj2kCodeBlockKernelJob]) -> &[u8] {
    <CudaHtj2kCodeBlockKernelJob as GpuAbi>::slice_as_bytes(values)
}

pub(crate) fn htj2k_cleanup_multi_jobs_as_bytes(
    values: &[CudaHtj2kCleanupMultiKernelJob],
) -> &[u8] {
    <CudaHtj2kCleanupMultiKernelJob as GpuAbi>::slice_as_bytes(values)
}

pub(crate) fn htj2k_dequantize_jobs_as_bytes(values: &[CudaHtj2kDequantizeKernelJob]) -> &[u8] {
    <CudaHtj2kDequantizeKernelJob as GpuAbi>::slice_as_bytes(values)
}

pub(crate) fn htj2k_statuses_as_bytes_mut(values: &mut [CudaHtj2kStatus]) -> &mut [u8] {
    <CudaHtj2kStatus as GpuAbi>::slice_as_bytes_mut(values)
}

pub(crate) fn htj2k_statuses_byte_len(count: usize) -> Result<usize, CudaError> {
    count
        .checked_mul(size_of::<CudaHtj2kStatus>())
        .ok_or(CudaError::LengthTooLarge { len: count })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn htj2k_cuda_abi_sizes_offsets_and_tail_bytes_are_stable() {
        assert_eq!(size_of::<CudaHtj2kCodeBlockKernelJob>(), 56);
        assert_eq!(offset_of!(CudaHtj2kCodeBlockKernelJob, reconstruction), 32);
        assert_eq!(size_of::<CudaHtj2kCleanupMultiKernelJob>(), 64);
        assert_eq!(
            offset_of!(CudaHtj2kCleanupMultiKernelJob, reconstruction),
            60
        );
        assert_eq!(size_of::<CudaHtj2kDequantizeKernelJob>(), 40);
        assert_eq!(offset_of!(CudaHtj2kDequantizeKernelJob, reconstruction), 28);
        assert_eq!(offset_of!(CudaHtj2kDequantizeKernelJob, reserved_tail), 36);

        let jobs = [CudaHtj2kDequantizeKernelJob {
            output_ptr: 1,
            width: 2,
            height: 3,
            output_stride: 4,
            output_offset: 5,
            num_bitplanes: 6,
            reconstruction: 0,
            dequantization_step: 1.0,
            reserved_tail: 0x8877_6655,
        }];
        let job_bytes = <CudaHtj2kDequantizeKernelJob as GpuAbi>::slice_as_bytes(&jobs);
        assert_eq!(job_bytes.len(), 40);
        assert_eq!(&job_bytes[36..40], &0x8877_6655u32.to_ne_bytes());
    }
}
