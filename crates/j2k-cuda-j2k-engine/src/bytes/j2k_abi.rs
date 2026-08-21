// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::j2k_decode::{
    CudaJ2kIdwtJob, CudaJ2kIdwtMultiKernelJob, CudaJ2kInverseMctJob, CudaJ2kRect,
    CudaJ2kStoreGray16BatchJob, CudaJ2kStoreGray16Job, CudaJ2kStoreGray8BatchJob,
    CudaJ2kStoreGray8Job, CudaJ2kStoreGrayI16BatchJob, CudaJ2kStoreRgb16Job,
    CudaJ2kStoreRgb16MctJob, CudaJ2kStoreRgb8Job, CudaJ2kStoreRgb8MctBatchJob,
    CudaJ2kStoreRgb8MctJob,
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
    CudaJ2kRect {
        x0: u32,
        y0: u32,
        x1: u32,
        y1: u32,
    },
    CudaJ2kIdwtJob {
        rect: CudaJ2kRect,
        ll_rect: CudaJ2kRect,
        hl_rect: CudaJ2kRect,
        lh_rect: CudaJ2kRect,
        hh_rect: CudaJ2kRect,
        irreversible97: u32,
    },
    CudaJ2kIdwtMultiKernelJob {
        ll_ptr: u64,
        hl_ptr: u64,
        lh_ptr: u64,
        hh_ptr: u64,
        output_ptr: u64,
        job: CudaJ2kIdwtJob,
        reserved_tail: u32,
    },
    CudaJ2kStoreGray8Job {
        input_width: u32,
        source_x: u32,
        source_y: u32,
        copy_width: u32,
        copy_height: u32,
        output_width: u32,
        output_height: u32,
        output_x: u32,
        output_y: u32,
        addend: f32,
        bit_depth: u32,
    },
    CudaJ2kStoreGray16Job {
        input_width: u32,
        source_x: u32,
        source_y: u32,
        copy_width: u32,
        copy_height: u32,
        output_width: u32,
        output_height: u32,
        output_x: u32,
        output_y: u32,
        addend: f32,
        bit_depth: u32,
    },
    CudaJ2kStoreGray8BatchJob {
        input_ptr: u64,
        output_ptr: u64,
        job: CudaJ2kStoreGray8Job,
        reserved_tail: u32,
    },
    CudaJ2kStoreGray16BatchJob {
        input_ptr: u64,
        output_ptr: u64,
        job: CudaJ2kStoreGray16Job,
        reserved_tail: u32,
    },
    CudaJ2kStoreGrayI16BatchJob {
        input_ptr: u64,
        output_ptr: u64,
        job: CudaJ2kStoreGray16Job,
        reserved_tail: u32,
    },
    CudaJ2kInverseMctJob {
        len: u32,
        irreversible97: u32,
        addend0: f32,
        addend1: f32,
        addend2: f32,
    },
    CudaJ2kStoreRgb8Job {
        input_width0: u32,
        input_width1: u32,
        input_width2: u32,
        source_x0: u32,
        source_y0: u32,
        source_x1: u32,
        source_y1: u32,
        source_x2: u32,
        source_y2: u32,
        copy_width: u32,
        copy_height: u32,
        output_width: u32,
        output_height: u32,
        output_x: u32,
        output_y: u32,
        addend0: f32,
        addend1: f32,
        addend2: f32,
        bit_depth0: u32,
        bit_depth1: u32,
        bit_depth2: u32,
        rgba: u32,
    },
    CudaJ2kStoreRgb16Job {
        input_width0: u32,
        input_width1: u32,
        input_width2: u32,
        source_x0: u32,
        source_y0: u32,
        source_x1: u32,
        source_y1: u32,
        source_x2: u32,
        source_y2: u32,
        copy_width: u32,
        copy_height: u32,
        output_width: u32,
        output_height: u32,
        output_x: u32,
        output_y: u32,
        addend0: f32,
        addend1: f32,
        addend2: f32,
        bit_depth0: u32,
        bit_depth1: u32,
        bit_depth2: u32,
        rgba: u32,
    },
    CudaJ2kStoreRgb8MctJob {
        store: CudaJ2kStoreRgb8Job,
        irreversible97: u32,
    },
    CudaJ2kStoreRgb8MctBatchJob {
        plane0_ptr: u64,
        plane1_ptr: u64,
        plane2_ptr: u64,
        output_ptr: u64,
        job: CudaJ2kStoreRgb8MctJob,
        reserved_tail: u32,
    },
    CudaJ2kStoreRgb16MctJob {
        store: CudaJ2kStoreRgb16Job,
        irreversible97: u32,
    },
}

mod native_store;
