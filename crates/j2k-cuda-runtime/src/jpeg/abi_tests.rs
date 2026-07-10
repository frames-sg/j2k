// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    CudaJpeg420Params, CudaJpegBaselineEncodeFormat, CudaJpegBaselineEncodeHuffmanTable,
    CudaJpegBaselineEncodeParams, CudaJpegBaselineEncodeStatus, CudaJpegDecodeStatus,
    CudaJpegEntropyCheckpoint, CudaJpegEntropyChunkParams, CudaJpegEntropyOverflowState,
    CudaJpegEntropySyncState, CudaJpegHuffmanTable,
};

#[test]
fn cuda_jpeg_host_abi_layouts_remain_stable() {
    use core::mem::{align_of, offset_of, size_of};

    assert_eq!(size_of::<CudaJpegHuffmanTable>(), 396);
    assert_eq!(align_of::<CudaJpegHuffmanTable>(), 4);
    assert_eq!(offset_of!(CudaJpegHuffmanTable, val_offset), 68);
    assert_eq!(offset_of!(CudaJpegHuffmanTable, values), 136);
    assert_eq!(offset_of!(CudaJpegHuffmanTable, values_len), 392);

    assert_eq!(size_of::<CudaJpegEntropyCheckpoint>(), 40);
    assert_eq!(align_of::<CudaJpegEntropyCheckpoint>(), 8);
    assert_eq!(offset_of!(CudaJpegEntropyCheckpoint, bit_acc), 8);
    assert_eq!(offset_of!(CudaJpegEntropyCheckpoint, reserved), 32);

    assert_eq!(size_of::<CudaJpegBaselineEncodeParams>(), 84);
    assert_eq!(
        offset_of!(CudaJpegBaselineEncodeParams, entropy_capacity),
        80
    );
    assert_eq!(size_of::<CudaJpegBaselineEncodeHuffmanTable>(), 768);
    assert_eq!(offset_of!(CudaJpegBaselineEncodeHuffmanTable, lens), 512);

    assert_eq!(size_of::<CudaJpegEntropySyncState>(), 32);
    assert_eq!(offset_of!(CudaJpegEntropySyncState, reserved), 28);
    assert_eq!(size_of::<CudaJpegEntropyOverflowState>(), 32);
    assert_eq!(offset_of!(CudaJpegEntropyOverflowState, reserved), 20);

    assert_eq!(size_of::<CudaJpeg420Params>(), 32);
    assert_eq!(size_of::<CudaJpegEntropyChunkParams>(), 32);
    assert_eq!(size_of::<CudaJpegBaselineEncodeStatus>(), 16);
    assert_eq!(size_of::<CudaJpegDecodeStatus>(), 16);

    assert_eq!(CudaJpegBaselineEncodeFormat::Gray8.abi(), 0);
    assert_eq!(CudaJpegBaselineEncodeFormat::Rgb8.abi(), 1);
}
