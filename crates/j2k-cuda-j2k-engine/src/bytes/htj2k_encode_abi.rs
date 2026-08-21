// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    htj2k_encode::{
        CudaHtj2kEncodeCompactJob, CudaHtj2kEncodeKernelJob, CudaHtj2kEncodeMultiInputKernelJob,
        CudaHtj2kEncodeStatus,
    },
    htj2k_packetize::{
        CudaHtj2kPacketizationBlock, CudaHtj2kPacketizationKernelPacket,
        CudaHtj2kPacketizationStatus, CudaHtj2kPacketizationSubband,
        CudaHtj2kPacketizationSubbandTagState, CudaHtj2kPacketizationTagNodeState,
    },
};
use j2k_core::accelerator::GpuAbi;
use std::mem::{offset_of, size_of};

/// Prove that each CUDA ABI struct consists only of the declared fields with
/// no internal or tail padding before permitting safe whole-object byte views.
macro_rules! prove_cuda_gpu_abi_layout {
    ($ty:ty, $offset:expr;) => {
        let _: [(); size_of::<$ty>()] = [(); $offset];
    };
    (
        $ty:ty,
        $offset:expr;
        $field:ident: $field_ty:ty
        $(, $remaining_field:ident: $remaining_field_ty:ty)*
    ) => {
        let _: [(); offset_of!($ty, $field)] = [(); $offset];
        prove_cuda_gpu_abi_layout!(
            $ty,
            $offset + size_of::<$field_ty>();
            $($remaining_field: $remaining_field_ty),*
        );
    };
}

macro_rules! impl_cuda_gpu_abi {
    ($(
        $ty:ty {
            $first_field:ident: $first_field_ty:ty
            $(, $field:ident: $field_ty:ty)*
            $(,)?
        }
    ),+ $(,)?) => {
        $(
            const _: () = {
                fn assert_field_types(value: &$ty) {
                    let _: &$first_field_ty = &value.$first_field;
                    $(let _: &$field_ty = &value.$field;)*
                }
                let _ = assert_field_types;

                prove_cuda_gpu_abi_layout!(
                    $ty,
                    0;
                    $first_field: $first_field_ty
                    $(, $field: $field_ty)*
                );
            };

            // SAFETY: The compile-time assertions above prove that the repr(C)
            // object representation is exactly the listed numeric/array fields,
            // without uninitialized padding. Every listed field accepts every
            // possible bit pattern, and constructors initialize reserved fields.
            unsafe impl GpuAbi for $ty {
                const NAME: &'static str = stringify!($ty);
            }
        )+
    };
}

impl_cuda_gpu_abi! {
    CudaHtj2kEncodeStatus {
        code: u32,
        detail: u32,
        data_len: u32,
        number_of_coding_passes: u32,
        missing_bit_planes: u32,
        reserved0: u32,
        reserved1: u32,
        reserved2: u32,
    },
    CudaHtj2kEncodeKernelJob {
        coefficient_offset: u32,
        coefficient_stride: u32,
        width: u32,
        height: u32,
        total_bitplanes: u32,
        output_offset: u32,
        output_capacity: u32,
        target_coding_passes: u32,
    },
    CudaHtj2kEncodeMultiInputKernelJob {
        coefficient_ptr: u64,
        coefficient_offset: u32,
        coefficient_stride: u32,
        width: u32,
        height: u32,
        total_bitplanes: u32,
        output_offset: u32,
        output_capacity: u32,
        target_coding_passes: u32,
    },
    CudaHtj2kEncodeCompactJob {
        source_offset: u32,
        compact_offset: u32,
        data_len: u32,
        reserved: u32,
    },
    CudaHtj2kPacketizationKernelPacket {
        block_start: u32,
        block_count: u32,
        subband_start: u32,
        subband_count: u32,
        output_offset: u32,
        output_capacity: u32,
        layer: u32,
    },
    CudaHtj2kPacketizationSubband {
        block_start: u32,
        block_count: u32,
        num_cbs_x: u32,
        num_cbs_y: u32,
    },
    CudaHtj2kPacketizationBlock {
        data_offset: u32,
        data_len: u32,
        cleanup_length: u32,
        refinement_length: u32,
        num_coding_passes: u32,
        num_zero_bitplanes: u32,
        l_block: u32,
        previously_included: u32,
        inclusion_layer: u32,
    },
    CudaHtj2kPacketizationSubbandTagState {
        inclusion_node_start: u32,
        zero_bitplane_node_start: u32,
        node_count: u32,
        reserved0: u32,
    },
    CudaHtj2kPacketizationTagNodeState {
        current: u32,
        known: u32,
    },
    CudaHtj2kPacketizationStatus {
        code: u32,
        detail: u32,
        output_len: u32,
        reserved0: u32,
    },
}

#[cfg(test)]
mod tests;
