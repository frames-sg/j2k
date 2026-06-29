// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_core::{BackendKind, DeviceMemoryRange};
use j2k_transcode::{
    ResidentBufferRef, ResidentCodestreamBuffer, ResidentColorModel, ResidentComponentGeometry,
    ResidentDctCoefficientOrder, ResidentDctGridLayout, ResidentDwtSubband, ResidentDwtSubbandKind,
    ResidentDwtSubbandLayout, ResidentHandoffError, ResidentJpegDctGrid, ResidentSampleInfo,
    ResidentSampling,
};

#[test]
fn resident_buffer_ref_rejects_empty_and_out_of_bounds_ranges() {
    let empty = DeviceMemoryRange::new(BackendKind::Metal, 7, 0, 0);
    assert_eq!(
        ResidentBufferRef::new(empty),
        Err(ResidentHandoffError::EmptyRange)
    );

    let overflow = DeviceMemoryRange::new(BackendKind::Metal, 7, usize::MAX, 2);
    assert_eq!(
        ResidentBufferRef::new(overflow),
        Err(ResidentHandoffError::OffsetOverflow)
    );

    let out_of_bounds = DeviceMemoryRange::new(BackendKind::Metal, 7, 32, 96);
    assert_eq!(
        ResidentBufferRef::with_allocation_len(out_of_bounds, 64),
        Err(ResidentHandoffError::RangeExceedsAllocation)
    );
}

#[test]
fn resident_component_metadata_validates_sampling_dimensions_and_bit_depth() {
    assert_eq!(
        ResidentSampling::new(0, 1),
        Err(ResidentHandoffError::ZeroSampling)
    );
    assert_eq!(
        ResidentSampleInfo::new(0, true),
        Err(ResidentHandoffError::InvalidBitDepth)
    );
    assert_eq!(
        ResidentComponentGeometry::new(
            0,
            0,
            16,
            ResidentSampling::new(1, 1).expect("valid sampling"),
        ),
        Err(ResidentHandoffError::ZeroDimension)
    );
}

#[test]
fn resident_dct_grid_carries_required_transcode_handoff_metadata() {
    let memory = DeviceMemoryRange::new(BackendKind::Metal, 99, 16, 16_384);
    let buffer = ResidentBufferRef::with_allocation_len(memory, 32_768).expect("valid range");
    let component = ResidentComponentGeometry::new(
        2,
        128,
        64,
        ResidentSampling::new(2, 2).expect("valid sampling"),
    )
    .expect("valid geometry");
    let sample = ResidentSampleInfo::new(12, true).expect("valid sample metadata");
    let layout = ResidentDctGridLayout {
        block_cols: 16,
        block_rows: 8,
        row_pitch_bytes: 2048,
        bytes_per_coefficient: 2,
        coefficient_order: ResidentDctCoefficientOrder::Natural,
    };

    let grid =
        ResidentJpegDctGrid::new(buffer, component, sample, ResidentColorModel::YCbCr, layout)
            .expect("valid resident DCT grid");

    assert_eq!(grid.buffer.memory_range(), memory);
    assert_eq!(grid.component.component_index, 2);
    assert_eq!(grid.component.sampling.x_rsiz, 2);
    assert_eq!(grid.sample.bit_depth, 12);
    assert!(grid.sample.signed);
    assert_eq!(grid.color, ResidentColorModel::YCbCr);
    assert_eq!(grid.block_cols, 16);
    assert_eq!(grid.coefficient_order, ResidentDctCoefficientOrder::Natural);
    assert!(grid.require_backend(BackendKind::Metal).is_ok());
    assert!(matches!(
        grid.require_backend(BackendKind::Cuda),
        Err(ResidentHandoffError::BackendMismatch {
            expected: BackendKind::Cuda,
            actual: BackendKind::Metal,
        })
    ));

    let short_layout = ResidentDctGridLayout {
        row_pitch_bytes: 1024,
        ..layout
    };
    assert_eq!(
        ResidentJpegDctGrid::new(
            buffer,
            component,
            sample,
            ResidentColorModel::YCbCr,
            short_layout,
        ),
        Err(ResidentHandoffError::LayoutExceedsBuffer)
    );
}

#[test]
fn resident_dwt_subband_and_codestream_validate_layout_and_capacity() {
    let buffer = ResidentBufferRef::with_allocation_len(
        DeviceMemoryRange::new(BackendKind::Metal, 11, 0, 2048),
        2048,
    )
    .expect("valid range");
    let component = ResidentComponentGeometry::new(
        0,
        64,
        64,
        ResidentSampling::new(1, 1).expect("valid sampling"),
    )
    .expect("valid geometry");
    let sample = ResidentSampleInfo::new(16, true).expect("valid sample metadata");

    let bad_layout = ResidentDwtSubbandLayout {
        level: 1,
        subband: ResidentDwtSubbandKind::LowLow,
        width: 32,
        height: 32,
        row_pitch_bytes: 0,
        bytes_per_coefficient: 2,
    };
    assert_eq!(
        ResidentDwtSubband::new(
            buffer,
            component,
            sample,
            ResidentColorModel::Grayscale,
            bad_layout,
        ),
        Err(ResidentHandoffError::ZeroByteStride)
    );

    let short_row_layout = ResidentDwtSubbandLayout {
        row_pitch_bytes: 32,
        ..bad_layout
    };
    assert_eq!(
        ResidentDwtSubband::new(
            buffer,
            component,
            sample,
            ResidentColorModel::Grayscale,
            short_row_layout,
        ),
        Err(ResidentHandoffError::LayoutExceedsBuffer)
    );

    let too_many_rows_layout = ResidentDwtSubbandLayout {
        height: 33,
        row_pitch_bytes: 64,
        ..bad_layout
    };
    assert_eq!(
        ResidentDwtSubband::new(
            buffer,
            component,
            sample,
            ResidentColorModel::Grayscale,
            too_many_rows_layout,
        ),
        Err(ResidentHandoffError::LayoutExceedsBuffer)
    );

    let good_layout = ResidentDwtSubbandLayout {
        row_pitch_bytes: 64,
        ..bad_layout
    };
    let subband = ResidentDwtSubband::new(
        buffer,
        component,
        sample,
        ResidentColorModel::Grayscale,
        good_layout,
    )
    .expect("valid subband");
    assert_eq!(subband.subband, ResidentDwtSubbandKind::LowLow);
    assert!(subband.require_backend(BackendKind::Metal).is_ok());
    assert!(matches!(
        subband.require_backend(BackendKind::Cuda),
        Err(ResidentHandoffError::BackendMismatch {
            expected: BackendKind::Cuda,
            actual: BackendKind::Metal,
        })
    ));

    assert_eq!(
        ResidentCodestreamBuffer::new(buffer, 4096, 4096),
        Err(ResidentHandoffError::CodestreamExceedsCapacity)
    );
    let codestream = ResidentCodestreamBuffer::new(buffer, 128, 2048).expect("valid codestream");
    assert_eq!(codestream.byte_len, 128);
    assert_eq!(codestream.capacity, 2048);
    assert!(codestream.require_backend(BackendKind::Metal).is_ok());
    assert!(matches!(
        codestream.require_backend(BackendKind::Cuda),
        Err(ResidentHandoffError::BackendMismatch {
            expected: BackendKind::Cuda,
            actual: BackendKind::Metal,
        })
    ));
}
