// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    classic_decode::{
        CudaClassicKernelJob, CudaClassicKernelSegment, CudaClassicKernelTables, CudaClassicStatus,
    },
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
        CudaJ2kIdwtJob, CudaJ2kIdwtMultiKernelJob, CudaJ2kInverseMctJob, CudaJ2kRect,
        CudaJ2kStoreGray16BatchJob, CudaJ2kStoreGray16Job, CudaJ2kStoreGray8BatchJob,
        CudaJ2kStoreGray8Job, CudaJ2kStoreGrayI16BatchJob, CudaJ2kStoreRgb16Job,
        CudaJ2kStoreRgb16MctJob, CudaJ2kStoreRgb8Job, CudaJ2kStoreRgb8MctBatchJob,
        CudaJ2kStoreRgb8MctJob,
    },
    jpeg::{
        CudaJpegBaselineEncodeHuffmanTable, CudaJpegBaselineEncodeParams,
        CudaJpegBaselineEncodeStatus, CudaJpegDecodeStatus, CudaJpegEntropyCheckpoint,
        CudaJpegEntropyOverflowState, CudaJpegEntropySyncState, CudaJpegHuffmanTable,
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
    CudaHtj2kCodeBlockKernelJob {
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
        reserved_tail: u32,
    },
    CudaHtj2kDequantizeKernelJob {
        output_ptr: u64,
        width: u32,
        height: u32,
        output_stride: u32,
        output_offset: u32,
        num_bitplanes: u32,
        reserved: u32,
        dequantization_step: f32,
        reserved_tail: u32,
    },
    CudaHtj2kStatus {
        code: u32,
        detail: u32,
        reserved0: u32,
        reserved1: u32,
    },
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
        dequantization_step: f32,
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

#[cfg(test)]
mod tests;
