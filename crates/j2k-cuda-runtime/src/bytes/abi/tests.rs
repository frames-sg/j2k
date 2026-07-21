// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use crate::j2k_decode::{
    CudaJ2kStoreRgbNativeBatchJob, CudaJ2kStoreRgbNativeJob, CudaJ2kStoreRgbaNativeBatchJob,
    CudaJ2kStoreRgbaNativeJob,
};

#[test]
fn explicit_tail_fields_preserve_cuda_host_abi_sizes_and_offsets() {
    assert_eq!(size_of::<CudaJpegEntropyCheckpoint>(), 40);
    assert_eq!(offset_of!(CudaJpegEntropyCheckpoint, reserved_tail), 36);
    assert_eq!(size_of::<CudaHtj2kCleanupMultiKernelJob>(), 64);
    assert_eq!(
        offset_of!(CudaHtj2kCleanupMultiKernelJob, reserved_tail),
        60
    );
    assert_eq!(size_of::<CudaHtj2kDequantizeKernelJob>(), 40);
    assert_eq!(offset_of!(CudaHtj2kDequantizeKernelJob, reserved_tail), 36);
    assert_eq!(size_of::<CudaClassicKernelJob>(), 72);
    assert_eq!(offset_of!(CudaClassicKernelJob, dequantization_step), 68);
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
    assert_eq!(size_of::<CudaJ2kIdwtMultiKernelJob>(), 128);
    assert_eq!(offset_of!(CudaJ2kIdwtMultiKernelJob, reserved_tail), 124);
    assert_eq!(size_of::<CudaJ2kStoreRgb8MctBatchJob>(), 128);
    assert_eq!(offset_of!(CudaJ2kStoreRgb8MctBatchJob, reserved_tail), 124);
    assert_eq!(size_of::<CudaJ2kStoreRgbNativeJob>(), 96);
    assert_eq!(size_of::<CudaJ2kStoreRgbNativeBatchJob>(), 128);
    assert_eq!(offset_of!(CudaJ2kStoreRgbNativeBatchJob, job), 32);
    assert_eq!(size_of::<CudaJ2kStoreRgbaNativeJob>(), 116);
    assert_eq!(size_of::<CudaJ2kStoreRgbaNativeBatchJob>(), 160);
    assert_eq!(offset_of!(CudaJ2kStoreRgbaNativeBatchJob, job), 40);
    assert_eq!(
        offset_of!(CudaJ2kStoreRgbaNativeBatchJob, reserved_tail),
        156
    );
    assert_eq!(size_of::<CudaJ2kStoreGray8BatchJob>(), 64);
    assert_eq!(offset_of!(CudaJ2kStoreGray8BatchJob, reserved_tail), 60);
    assert_eq!(size_of::<CudaJ2kStoreGray16BatchJob>(), 64);
    assert_eq!(offset_of!(CudaJ2kStoreGray16BatchJob, reserved_tail), 60);
    assert_eq!(size_of::<CudaJ2kStoreGrayI16BatchJob>(), 64);
    assert_eq!(offset_of!(CudaJ2kStoreGrayI16BatchJob, reserved_tail), 60);
}

#[test]
fn explicit_cuda_tail_fields_are_part_of_safe_byte_views() {
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
    let checkpoint_bytes = <CudaJpegEntropyCheckpoint as GpuAbi>::as_bytes(&checkpoint);
    assert_eq!(checkpoint_bytes.len(), 40);
    assert_eq!(&checkpoint_bytes[36..40], &0x4433_2211u32.to_ne_bytes());

    let jobs = [CudaHtj2kDequantizeKernelJob {
        output_ptr: 1,
        width: 2,
        height: 3,
        output_stride: 4,
        output_offset: 5,
        num_bitplanes: 6,
        reserved: 0,
        dequantization_step: 1.0,
        reserved_tail: 0x8877_6655,
    }];
    let job_bytes = <CudaHtj2kDequantizeKernelJob as GpuAbi>::slice_as_bytes(&jobs);
    assert_eq!(job_bytes.len(), 40);
    assert_eq!(&job_bytes[36..40], &0x8877_6655u32.to_ne_bytes());
}
