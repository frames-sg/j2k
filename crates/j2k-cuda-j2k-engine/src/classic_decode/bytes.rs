// SPDX-License-Identifier: MIT OR Apache-2.0

use super::abi::{
    CudaClassicKernelJob, CudaClassicKernelSegment, CudaClassicKernelTables, CudaClassicStatus,
};
use j2k_core::accelerator::GpuAbi;
use std::mem::{offset_of, size_of};

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
    CudaClassicKernelJob {
        output_ptr: u64,
        coded_offset: u32,
        coded_len: u32,
        segment_offset: u32,
        segment_count: u32,
        scratch_offset: u32,
        width: u32,
        height: u32,
        output_stride: u32,
        output_offset: u32,
        missing_msbs: u32,
        total_bitplanes: u32,
        number_of_coding_passes: u32,
        sub_band_type: u32,
        style_flags: u32,
        strict: u32,
        irreversible_midpoint: u32,
        dequantization_step: f32,
        roi_shift: u32,
    },
    CudaClassicKernelSegment {
        data_offset: u32,
        data_length: u32,
        start_coding_pass: u32,
        end_coding_pass: u32,
        use_arithmetic: u32,
    },
    CudaClassicKernelTables {
        mq_qe: [u32; 47],
        mq_transitions: [u32; 47],
        sign_contexts: [u16; 256],
        zero_contexts_ll_lh: [u8; 256],
        zero_contexts_hl: [u8; 256],
        zero_contexts_hh: [u8; 256],
    },
    CudaClassicStatus {
        code: u32,
        detail: u32,
        reserved0: u32,
        reserved1: u32,
    },
}

pub(super) fn classic_tables_as_bytes(value: &CudaClassicKernelTables) -> &[u8] {
    <CudaClassicKernelTables as GpuAbi>::as_bytes(value)
}

pub(super) fn classic_jobs_as_bytes(values: &[CudaClassicKernelJob]) -> &[u8] {
    <CudaClassicKernelJob as GpuAbi>::slice_as_bytes(values)
}

pub(super) fn classic_segments_as_bytes(values: &[CudaClassicKernelSegment]) -> &[u8] {
    <CudaClassicKernelSegment as GpuAbi>::slice_as_bytes(values)
}

pub(super) fn classic_statuses_as_bytes_mut(values: &mut [CudaClassicStatus]) -> &mut [u8] {
    <CudaClassicStatus as GpuAbi>::slice_as_bytes_mut(values)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classic_cuda_abi_sizes_and_offsets_match_the_device_contract() {
        assert_eq!(size_of::<CudaClassicKernelJob>(), 80);
        assert_eq!(offset_of!(CudaClassicKernelJob, dequantization_step), 72);
        assert_eq!(offset_of!(CudaClassicKernelJob, roi_shift), 76);
        assert_eq!(size_of::<CudaClassicKernelSegment>(), 20);
        assert_eq!(size_of::<CudaClassicKernelTables>(), 1_656);
        assert_eq!(offset_of!(CudaClassicKernelTables, mq_transitions), 188);
        assert_eq!(offset_of!(CudaClassicKernelTables, sign_contexts), 376);
        assert_eq!(
            offset_of!(CudaClassicKernelTables, zero_contexts_ll_lh),
            888
        );
        assert_eq!(offset_of!(CudaClassicKernelTables, zero_contexts_hl), 1_144);
        assert_eq!(offset_of!(CudaClassicKernelTables, zero_contexts_hh), 1_400);
        assert_eq!(size_of::<CudaClassicStatus>(), 16);
    }
}
