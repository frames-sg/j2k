// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec;

use crate::{ColorSpace, Decoder, JpegError, MarkerKind};

use super::{
    lossless_predictor_color_into, lossless_predictor_color_rows, lossless_predictor_gray_rows,
    lossless_predictor_plane, lossless_predictor_value, lossless_predictor_value_u16,
    read_gray16_sample, restart_index_allocation_bytes, restart_index_for_stream,
    restart_segment_capacity, upsample_h2v1_u16_at, upsample_h2v1_u8_at, upsample_h2v2_u16_at,
    upsample_h2v2_u8_at, write_lossless_color16_sampled_output,
    write_lossless_color8_sampled_output, LosslessColorPlanes, LosslessColorSampling,
    RestartSegment,
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
fn predictors_read_interleaved_rows_and_planes_consistently() {
    let gray = [10, 20, 30, 40];
    assert_eq!(lossless_predictor_value(1, &gray, 2, 1, 1), 30);
    assert_eq!(
        lossless_predictor_gray_rows::<u8>(2, &[30, 40], &[10, 20], 1, 1),
        20
    );

    let color = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
    assert_eq!(
        lossless_predictor_color_into::<u8>(1, &color, 6, 1, 1, 2),
        9
    );
    assert_eq!(
        lossless_predictor_color_rows::<u8>(
            2,
            &[7, 8, 9, 10, 11, 12],
            &[1, 2, 3, 4, 5, 6],
            1,
            1,
            1
        ),
        5
    );
    assert_eq!(
        lossless_predictor_plane(7, &[10u8, 20, 30, 40], 2, 1, 1),
        25
    );
}

#[test]
fn gray16_helpers_use_little_endian_samples() {
    let samples = [0x34, 0x12, 0x78, 0x56, 0xbc, 0x9a, 0xf0, 0xde];
    assert_eq!(read_gray16_sample(&samples, 2), 0x5678);
    assert_eq!(lossless_predictor_value_u16(1, &samples, 4, 1, 1), 0x9abc);
}

#[test]
fn sampled_upsamplers_cover_horizontal_and_vertical_edges() {
    assert_eq!(upsample_h2v1_u8_at(&[10, 30], 0), 10);
    assert_eq!(upsample_h2v1_u8_at(&[10, 30], 3), 30);
    assert_eq!(upsample_h2v2_u8_at(&[10, 30, 50, 70], 2, 2, 4, 2, 3), 65);

    assert_eq!(upsample_h2v1_u16_at(&[1_000, 3_000], 0), 1_000);
    assert_eq!(upsample_h2v1_u16_at(&[1_000, 3_000], 3), 3_000);
    assert_eq!(
        upsample_h2v2_u16_at(&[1_000, 3_000, 5_000, 7_000], 2, 2, 4, 2, 3),
        6_500
    );
}

#[test]
fn sampled_rgb_writers_preserve_rgb_planes_and_stride_padding() {
    let c0 = [10u8, 20, 30, 40];
    let c1 = [50u8, 60];
    let c2 = [70u8, 80];
    let mut out = vec![0xee; 16];
    write_lossless_color8_sampled_output(
        &mut out,
        8,
        ColorSpace::Rgb,
        LosslessColorSampling::S422,
        (2, 2),
        LosslessColorPlanes {
            c0: &c0,
            c1: &c1,
            c2: &c2,
        },
    );

    assert_eq!(&out[..6], &[10, 50, 70, 20, 50, 70]);
    assert_eq!(&out[8..14], &[30, 60, 80, 40, 60, 80]);
    assert_eq!(&out[6..8], &[0xee; 2]);
    assert_eq!(&out[14..], &[0xee; 2]);
}

#[test]
fn sampled_rgb16_writer_handles_odd_420_dimensions() {
    let c0 = [1_000u16, 2_000, 3_000, 4_000, 5_000, 6_000];
    let c1 = [10_000u16, 20_000];
    let c2 = [30_000u16, 40_000];
    let mut out = vec![0u8; 3 * 2 * 6];
    write_lossless_color16_sampled_output(
        &mut out,
        18,
        ColorSpace::Rgb,
        LosslessColorSampling::S420,
        (3, 2),
        LosslessColorPlanes {
            c0: &c0,
            c1: &c1,
            c2: &c2,
        },
    );

    assert_eq!(u16::from_le_bytes([out[0], out[1]]), 1_000);
    assert_eq!(u16::from_le_bytes([out[2], out[3]]), 10_000);
    assert_eq!(u16::from_le_bytes([out[4], out[5]]), 30_000);
    assert_eq!(u16::from_le_bytes([out[30], out[31]]), 6_000);
}
