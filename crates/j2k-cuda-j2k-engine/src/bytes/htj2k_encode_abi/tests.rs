// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use crate::j2k_decode::{
    CudaJ2kIdwtMultiKernelJob, CudaJ2kStoreGray16BatchJob, CudaJ2kStoreGray8BatchJob,
    CudaJ2kStoreGrayI16BatchJob, CudaJ2kStoreRgb8MctBatchJob, CudaJ2kStoreRgbNativeBatchJob,
    CudaJ2kStoreRgbNativeJob, CudaJ2kStoreRgbaNativeBatchJob, CudaJ2kStoreRgbaNativeJob,
};

#[test]
fn explicit_tail_fields_preserve_cuda_host_abi_sizes_and_offsets() {
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
