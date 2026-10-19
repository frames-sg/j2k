// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{Decoder, JpegError, MarkerKind};

use super::{
    lossless_output_order, restart_index_allocation_bytes, restart_index_for_stream,
    restart_segment_capacity, RestartSegment,
};

fn baseline_info() -> crate::Info {
    Decoder::new(j2k_test_support::JPEG_BASELINE_420_16X16)
        .expect("baseline fixture")
        .info()
        .clone()
}

#[test]
fn restart_segment_capacity_has_an_exact_shared_cap_boundary() {
    let max_segments =
        j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES / core::mem::size_of::<RestartSegment>();
    let max_mcus = u32::try_from(max_segments).expect("restart boundary fits u32");
    assert_eq!(
        restart_segment_capacity(max_mcus, 1).expect("exact restart boundary"),
        max_segments
    );
    assert!(matches!(
        restart_segment_capacity(max_mcus + 1, 1),
        Err(JpegError::MemoryCapExceeded { requested, cap }) if requested > cap
    ));
}

#[test]
fn restart_index_disabled_needs_no_scan_offset_or_allocation() {
    let info = baseline_info();
    assert_eq!(restart_index_allocation_bytes(&info, None), Ok(0));
    assert_eq!(restart_index_for_stream(&[], None, &info, None), Ok(None));
    assert_eq!(
        restart_index_for_stream(&[], None, &info, Some(0)),
        Ok(None)
    );
}

#[test]
fn restart_index_requires_a_scan_offset_when_enabled() {
    let error = restart_index_for_stream(&[], None, &baseline_info(), Some(1))
        .expect_err("enabled restart parsing requires SOS");
    assert_eq!(
        error,
        JpegError::MissingMarker {
            marker: MarkerKind::Sos
        }
    );
}

#[test]
fn restart_index_parses_stuffing_fill_and_ordered_markers() {
    let mut info = baseline_info();
    info.mcu_geometry.count = 3;
    let bytes = [
        0x11, 0xff, 0x00, 0x22, 0xff, 0xff, 0xd0, 0x33, 0xff, 0xd1, 0xff, 0xd9,
    ];
    let index = restart_index_for_stream(&bytes, Some(0), &info, Some(1))
        .expect("valid restart stream")
        .expect("restart index");

    assert_eq!(index.interval_mcus, 1);
    assert_eq!(index.segments.len(), 3);
    assert_eq!(index.segments[0].start_mcu, 0);
    assert_eq!(index.segments[1].start_mcu, 1);
    assert_eq!(index.segments[1].marker, Some(0xd0));
    assert_eq!(index.segments[2].start_mcu, 2);
    assert_eq!(index.segments[2].marker, Some(0xd1));
}

#[test]
fn restart_index_classifies_marker_failures() {
    let mut info = baseline_info();
    info.mcu_geometry.count = 2;

    assert!(matches!(
        restart_index_for_stream(&[0xff, 0xd1], Some(0), &info, Some(1)),
        Err(JpegError::RestartMismatch {
            expected: 0,
            found: 0xd1,
            ..
        })
    ));
    assert!(matches!(
        restart_index_for_stream(&[0xff, 0xd9], Some(0), &info, Some(1)),
        Err(JpegError::UnexpectedEoi {
            mcu_at: 1,
            mcu_total: 2
        })
    ));
    assert!(matches!(
        restart_index_for_stream(&[0xff, 0xc0], Some(0), &info, Some(1)),
        Err(JpegError::UnexpectedMarker { found: 0xc0, .. })
    ));
    assert!(matches!(
        restart_index_for_stream(&[0xff], Some(0), &info, Some(1)),
        Err(JpegError::Truncated {
            offset: 0,
            expected: 1
        })
    ));
    assert_eq!(
        restart_index_for_stream(&[0], Some(0), &info, Some(1)),
        Err(JpegError::MissingMarker {
            marker: MarkerKind::Eoi
        })
    );
}

#[test]
fn restart_index_rejects_more_markers_than_geometry_allows() {
    let info = baseline_info();
    assert!(matches!(
        restart_index_for_stream(&[0xff, 0xd0], Some(0), &info, Some(1)),
        Err(JpegError::UnexpectedMarker { found: 0xd0, .. })
    ));
}

#[test]
fn output_order_follows_frame_component_order() {
    for bytes in [
        j2k_test_support::lossless_predictor_rgb_3x3_jpeg(1),
        j2k_test_support::lossless_predictor_ycbcr_3x3_jpeg(1),
    ] {
        let decoder = Decoder::new(&bytes).expect("lossless color fixture");
        assert_eq!(lossless_output_order(&decoder.plan), Ok([0, 1, 2]));
    }
    let gray_bytes = j2k_test_support::lossless_predictor_grayscale_3x3_jpeg(1);
    let gray = Decoder::new(&gray_bytes).expect("lossless gray fixture");
    assert!(matches!(
        lossless_output_order(&gray.plan),
        Err(JpegError::UnsupportedComponentCount { count: 1 })
    ));
}
