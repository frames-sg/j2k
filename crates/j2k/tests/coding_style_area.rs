// SPDX-License-Identifier: MIT OR Apache-2.0

//! Main-header coding-style inspection and the T.800 decode area of a region.

use j2k::{J2kDecoder, J2kError, J2kProgressionOrder};
use j2k_core::Rect;
use j2k_native::{encode, EncodeOptions, EncodeProgressionOrder};
use j2k_test_support::{wrap_jp2_codestream, OPENJPEG_ORIGIN_23X17_AT_3_5};

fn encode_gray(width: u32, height: u32, options: &EncodeOptions) -> Vec<u8> {
    let pixels = (0..width * height)
        .map(|index| (index % 251) as u8)
        .collect::<Vec<_>>();
    encode(&pixels, width, height, 1, 8, false, options).expect("encode gray fixture")
}

#[test]
fn coding_style_reports_main_cod_defaults_for_codestreams_and_files() {
    let codestream = encode_gray(
        40,
        24,
        &EncodeOptions {
            reversible: true,
            num_decomposition_levels: 3,
            num_layers: 2,
            progression_order: EncodeProgressionOrder::Rpcl,
            code_block_width_exp: 3,
            code_block_height_exp: 2,
            ..EncodeOptions::default()
        },
    );
    let file = wrap_jp2_codestream(&codestream, 40, 24, 1, 8, 17);
    for (shape, bytes) in [("codestream", &codestream), ("JP2", &file)] {
        let style = J2kDecoder::inspect_coding_style(bytes).expect("coding style");
        assert_eq!(
            style.progression_order,
            J2kProgressionOrder::Rpcl,
            "{shape}"
        );
        assert_eq!(style.quality_layers, 2, "{shape}");
        assert_eq!(style.decomposition_levels, 3, "{shape}");
        assert_eq!(style.code_block_size, (32, 16), "{shape}");
        let info = J2kDecoder::inspect(bytes).expect("inspect");
        assert_eq!(
            u32::from(style.decomposition_levels) + 1,
            u32::from(info.resolution_levels),
            "{shape}"
        );
    }
}

#[test]
fn coding_style_rejects_inputs_that_are_not_jpeg_2000() {
    let error = J2kDecoder::inspect_coding_style(b"not a codestream").expect_err("not J2K");
    assert!(matches!(error, J2kError::Unsupported(_)), "{error:?}");
}

/// `[ceil((origin + start) / s), ceil((origin + end) / s))`, relative to the
/// reduced image's first sample `ceil(origin / s)`.
fn t800_area_axis(start: u32, length: u32, origin: u32, levels: u8) -> (u32, u32) {
    let s = 1_u32 << levels;
    let low = (origin + start).div_ceil(s);
    let high = (origin + start + length).div_ceil(s);
    (low - origin.div_ceil(s), high - low)
}

#[test]
fn reduced_area_follows_t800_and_lies_within_the_covering_region() {
    let grid = J2kDecoder::inspect_reference_grid(OPENJPEG_ORIGIN_23X17_AT_3_5).expect("grid");
    assert_eq!((grid.image_origin, grid.image_size), ((3, 5), (23, 17)));
    for levels in 0..=4_u8 {
        for (x, w) in [(0, 23), (0, 1), (1, 7), (4, 9), (13, 10), (22, 1)] {
            for (y, h) in [(0, 17), (2, 3), (11, 6), (16, 1)] {
                let region = Rect { x, y, w, h };
                let area = grid.reduced_area(region, levels).expect("area");
                let (ax, aw) = t800_area_axis(x, w, 3, levels);
                let (ay, ah) = t800_area_axis(y, h, 5, levels);
                assert_eq!(
                    area,
                    Rect {
                        x: ax,
                        y: ay,
                        w: aw,
                        h: ah
                    },
                    "{region:?} r{levels}"
                );
                let covering = grid.reduced_region(region, levels).expect("covering");
                assert!(
                    area.x >= covering.x && area.y >= covering.y,
                    "{region:?} r{levels}"
                );
                assert_eq!(
                    area.x + area.w,
                    covering.x + covering.w,
                    "{region:?} r{levels}"
                );
                assert_eq!(
                    area.y + area.h,
                    covering.y + covering.h,
                    "{region:?} r{levels}"
                );
            }
        }
    }
    let full = Rect {
        x: 0,
        y: 0,
        w: 23,
        h: 17,
    };
    assert_eq!(grid.reduced_area(full, 2), grid.reduced_region(full, 2));
    // A region whose only overlap is a partial leading footprint is empty.
    let sliver = grid
        .reduced_area(
            Rect {
                x: 2,
                y: 0,
                w: 1,
                h: 17,
            },
            2,
        )
        .expect("sliver");
    assert_eq!(sliver.w, 0);
}

#[test]
fn reduced_area_rejects_empty_outside_and_overdeep_requests() {
    let grid = J2kDecoder::inspect_reference_grid(OPENJPEG_ORIGIN_23X17_AT_3_5).expect("grid");
    assert_eq!(
        grid.reduced_area(
            Rect {
                x: 0,
                y: 0,
                w: 0,
                h: 4
            },
            0
        ),
        None
    );
    assert_eq!(
        grid.reduced_area(
            Rect {
                x: 20,
                y: 0,
                w: 4,
                h: 4
            },
            0
        ),
        None
    );
    assert_eq!(
        grid.reduced_area(
            Rect {
                x: 0,
                y: 0,
                w: 4,
                h: 4
            },
            32
        ),
        None
    );
}
