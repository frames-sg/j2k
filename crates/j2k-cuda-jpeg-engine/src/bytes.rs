// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_core::accelerator::GpuAbi;
use std::mem::{offset_of, size_of};

use crate::jpeg::{
    CudaJpegBaselineEncodeHuffmanTable, CudaJpegBaselineEncodeParams, CudaJpegBaselineEncodeStatus,
    CudaJpegDecodeStatus, CudaJpegEntropyCheckpoint, CudaJpegEntropyOverflowState,
    CudaJpegEntropySyncState, CudaJpegHuffmanTable,
};

macro_rules! prove_cuda_gpu_abi_layout {
    ($ty:ty, $offset:expr;) => {
        let _: [(); size_of::<$ty>()] = [(); $offset];
    };
    ($ty:ty, $offset:expr; $field:ident: $field_ty:ty $(, $remaining:ident: $remaining_ty:ty)*) => {
        let _: [(); offset_of!($ty, $field)] = [(); $offset];
        prove_cuda_gpu_abi_layout!(
            $ty,
            $offset + size_of::<$field_ty>();
            $($remaining: $remaining_ty),*
        );
    };
}

macro_rules! impl_cuda_gpu_abi {
    ($($ty:ty { $first:ident: $first_ty:ty $(, $field:ident: $field_ty:ty)* $(,)? }),+ $(,)?) => {
        $(
            const _: () = {
                fn assert_field_types(value: &$ty) {
                    let _: &$first_ty = &value.$first;
                    $(let _: &$field_ty = &value.$field;)*
                }
                let _ = assert_field_types;
                prove_cuda_gpu_abi_layout!(
                    $ty,
                    0;
                    $first: $first_ty
                    $(, $field: $field_ty)*
                );
            };

            // SAFETY: the compile-time assertions prove the repr(C) object is
            // exactly the listed numeric/array fields without padding.
            unsafe impl GpuAbi for $ty {
                const NAME: &'static str = stringify!($ty);
            }
        )+
    };
}

impl_cuda_gpu_abi! {
    CudaJpegHuffmanTable {
        max_code: [i32; 17],
        val_offset: [i32; 17],
        values: [u8; 256],
        values_len: u32,
    },
    CudaJpegEntropyCheckpoint {
        mcu_index: u32,
        entropy_pos: u32,
        bit_acc: u64,
        bit_count: u32,
        y_prev_dc: i32,
        cb_prev_dc: i32,
        cr_prev_dc: i32,
        reserved: u32,
        reserved_tail: u32,
    },
    CudaJpegDecodeStatus {
        code: u32,
        detail: u32,
        position: u32,
        reserved: u32,
    },
    CudaJpegEntropySyncState {
        code: u32,
        start_bit: u32,
        end_bit: u32,
        bit_pos: u32,
        symbol_count: u32,
        block_phase: u32,
        zigzag_index: u32,
        reserved: u32,
    },
    CudaJpegEntropyOverflowState {
        code: u32,
        from_subsequence: u32,
        to_subsequence: u32,
        overflow_bits: u32,
        synchronized: u32,
        reserved: [u32; 3],
    },
    CudaJpegBaselineEncodeParams {
        input_offset_bytes: u32,
        input_width: u32,
        input_height: u32,
        output_width: u32,
        output_height: u32,
        pitch_bytes: u32,
        mcus_per_row: u32,
        mcu_rows: u32,
        restart_interval_mcus: u32,
        format: u32,
        components: u32,
        max_h: u32,
        max_v: u32,
        h0: u32,
        v0: u32,
        h1: u32,
        v1: u32,
        h2: u32,
        v2: u32,
        entropy_offset_bytes: u32,
        entropy_capacity: u32,
    },
    CudaJpegBaselineEncodeHuffmanTable {
        codes: [u16; 256],
        lens: [u8; 256],
    },
    CudaJpegBaselineEncodeStatus {
        code: u32,
        entropy_len: u32,
        detail: u32,
        reserved: u32,
    },
}

macro_rules! gpu_ref_bytes {
    ($($name:ident: $ty:ty;)+) => {
        $(
            pub(crate) fn $name(value: &$ty) -> &[u8] {
                <$ty as GpuAbi>::as_bytes(value)
            }
        )+
    };
}

macro_rules! gpu_slice_bytes {
    ($($name:ident: $ty:ty;)+) => {
        $(
            pub(crate) fn $name(values: &[$ty]) -> &[u8] {
                <$ty as GpuAbi>::slice_as_bytes(values)
            }
        )+
    };
}

macro_rules! gpu_slice_bytes_mut {
    ($($name:ident: $ty:ty;)+) => {
        $(
            pub(crate) fn $name(values: &mut [$ty]) -> &mut [u8] {
                <$ty as GpuAbi>::slice_as_bytes_mut(values)
            }
        )+
    };
}

gpu_ref_bytes! {
    cuda_jpeg_huffman_table_as_bytes: CudaJpegHuffmanTable;
    cuda_jpeg_baseline_encode_huffman_table_as_bytes: CudaJpegBaselineEncodeHuffmanTable;
}

gpu_slice_bytes! {
    u16_slice_as_bytes: u16;
    cuda_jpeg_entropy_checkpoints_as_bytes: CudaJpegEntropyCheckpoint;
    cuda_jpeg_decode_statuses_as_bytes: CudaJpegDecodeStatus;
    cuda_jpeg_entropy_sync_states_as_bytes: CudaJpegEntropySyncState;
    cuda_jpeg_entropy_overflow_states_as_bytes: CudaJpegEntropyOverflowState;
    cuda_jpeg_baseline_encode_params_as_bytes: CudaJpegBaselineEncodeParams;
    cuda_jpeg_baseline_encode_statuses_as_bytes: CudaJpegBaselineEncodeStatus;
}

gpu_slice_bytes_mut! {
    cuda_jpeg_decode_statuses_as_bytes_mut: CudaJpegDecodeStatus;
    cuda_jpeg_entropy_sync_states_as_bytes_mut: CudaJpegEntropySyncState;
    cuda_jpeg_entropy_overflow_states_as_bytes_mut: CudaJpegEntropyOverflowState;
    cuda_jpeg_baseline_encode_statuses_as_bytes_mut: CudaJpegBaselineEncodeStatus;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_checkpoint_tail_is_part_of_the_safe_byte_view() {
        let checkpoint = CudaJpegEntropyCheckpoint {
            mcu_index: 1,
            entropy_pos: 2,
            bit_acc: 3,
            bit_count: 4,
            y_prev_dc: 5,
            cb_prev_dc: 6,
            cr_prev_dc: 7,
            reserved: 8,
            reserved_tail: 0x4433_2211,
        };
        let bytes = <CudaJpegEntropyCheckpoint as GpuAbi>::as_bytes(&checkpoint);
        assert_eq!(bytes.len(), 40);
        assert_eq!(&bytes[36..40], &0x4433_2211u32.to_ne_bytes());
    }
}
