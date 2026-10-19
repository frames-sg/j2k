// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec;

use super::{fill_lossless_plane_row, lossless_plane_layout, LosslessPlane, LosslessPlaneLayout};
use crate::info::{ColorSpace, Info, McuGeometry, SamplingFactors, SofKind};

fn lossless_info(dimensions: (u32, u32), sampling: &[(u8, u8)], bit_depth: u8) -> Info {
    let sampling = SamplingFactors::from_components(sampling).expect("sampling");
    Info {
        dimensions,
        color_space: ColorSpace::Rgb,
        sampling,
        sof_kind: SofKind::Lossless,
        bit_depth,
        restart_interval: None,
        mcu_geometry: McuGeometry::from_sampling(dimensions, sampling),
        scan_count: 1,
    }
}

/// Planes for a `width`-pixel frame: full-size luma, chroma `h_expand` by
/// `v_expand` subsampled, sized as `lossless_plane_layout` sizes them.
fn layout(width: usize, height: usize, h_expand: usize, v_expand: usize) -> LosslessPlaneLayout {
    let (cw, ch) = (width.div_ceil(h_expand), height.div_ceil(v_expand));
    let plane = |offset, width, height, h_expand, v_expand| LosslessPlane {
        offset,
        width,
        height,
        h_expand,
        v_expand,
    };
    LosslessPlaneLayout {
        planes: [
            plane(0, width, height, 1, 1),
            plane(width * height, cw, ch, h_expand, v_expand),
            plane(width * height + cw * ch, cw, ch, h_expand, v_expand),
        ],
        total_samples: width * height + 2 * cw * ch,
        total_bytes: 0,
    }
}

#[test]
fn plane_layout_sizes_odd_frames_and_any_whole_ratio() {
    let layout = lossless_plane_layout(
        &lossless_info((33, 17), &[(2, 2), (1, 1), (1, 1)], 8),
        usize::MAX,
    )
    .expect("layout")
    .expect("4:2:0 renders through planes");
    assert_eq!(
        layout
            .planes
            .map(|plane| (plane.width, plane.height, plane.h_expand, plane.v_expand)),
        [(33, 17, 1, 1), (17, 9, 2, 2), (17, 9, 2, 2)]
    );
    assert_eq!(layout.total_samples, 33 * 17 + 2 * 17 * 9);
    assert_eq!(layout.total_bytes, layout.total_samples);

    let layout = lossless_plane_layout(
        &lossless_info((9, 5), &[(4, 1), (1, 1), (1, 1)], 12),
        usize::MAX,
    )
    .expect("layout")
    .expect("4:1:1 renders through planes");
    assert_eq!(layout.planes[1].width, 3);
    assert_eq!(layout.planes[1].h_expand, 4);
    assert_eq!(layout.total_bytes, layout.total_samples * 2);

    // 3x2 against 2x1 does not divide: no replication ratio exists.
    assert_eq!(
        lossless_plane_layout(
            &lossless_info((9, 5), &[(3, 2), (2, 1), (1, 1)], 8),
            usize::MAX
        )
        .expect("layout"),
        None
    );
}

#[test]
fn plane_rows_replicate_subsampled_components() {
    // 4x2 4:2:0: chroma sample (0, 0) covers pixels x 0..2, rows 0..2.
    let samples: [u8; 12] = [10, 20, 30, 40, 50, 60, 70, 80, 1, 2, 5, 6];
    let layout = layout(4, 2, 2, 2);
    let mut row = vec![0u8; 12];
    fill_lossless_plane_row::<u8>(&mut row, 1, 4, (&samples, &layout), (ColorSpace::Rgb, 8));
    assert_eq!(row, [50, 1, 5, 60, 1, 5, 70, 2, 6, 80, 2, 6]);
}

#[test]
fn plane_rows_preserve_rgb_planes_and_stride_padding() {
    let samples = [10u8, 20, 30, 40, 50, 60, 70, 80];
    let layout = layout(2, 2, 2, 1);
    let mut out = [0xee; 16];
    for y in 0..2 {
        fill_lossless_plane_row::<u8>(
            &mut out[y * 8..y * 8 + 6],
            y,
            2,
            (&samples, &layout),
            (ColorSpace::Rgb, 8),
        );
    }
    assert_eq!(&out[..6], &[10, 50, 70, 20, 50, 70]);
    assert_eq!(&out[8..14], &[30, 60, 80, 40, 60, 80]);
    assert_eq!(&out[6..8], &[0xee; 2]);
    assert_eq!(&out[14..], &[0xee; 2]);
}

#[test]
fn plane_rows_handle_odd_420_dimensions_at_16_bits() {
    let samples = [
        1_000u16, 2_000, 3_000, 4_000, 5_000, 6_000, 10_000, 20_000, 30_000, 40_000,
    ];
    let layout = layout(3, 2, 2, 2);
    let mut row = vec![0u8; 3 * 3 * 2];
    let value = |row: &[u8], index: usize| u16::from_le_bytes([row[index * 2], row[index * 2 + 1]]);
    fill_lossless_plane_row::<u16>(&mut row, 1, 3, (&samples, &layout), (ColorSpace::Rgb, 16));
    assert_eq!(
        (0..9)
            .map(|i| value(&row, i))
            .collect::<alloc::vec::Vec<_>>(),
        [4_000, 10_000, 30_000, 5_000, 10_000, 30_000, 6_000, 20_000, 40_000]
    );
}

#[test]
fn plane_rows_convert_ycbcr_at_the_frame_precision() {
    // 12-bit neutral chroma keeps luma; the 8- and 16-bit centers would not.
    let samples = [1_000u16, 2_048, 2_048];
    let layout = layout(1, 1, 1, 1);
    let mut row = vec![0u8; 6];
    fill_lossless_plane_row::<u16>(&mut row, 0, 1, (&samples, &layout), (ColorSpace::YCbCr, 12));
    assert_eq!(row, [0xE8, 0x03, 0xE8, 0x03, 0xE8, 0x03]);
}
