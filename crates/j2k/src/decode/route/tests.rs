// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{full_image_route, FullImageRoute};
use crate::backend::image_at_scale;
use crate::DecodeSettings;
use j2k_core::{Downscale, PixelFormat};
use j2k_native::{encode, EncodeOptions};

fn codestream(components: u16, options: &EncodeOptions) -> Vec<u8> {
    let (width, height) = (96_u32, 80_u32);
    let pixels: Vec<u8> = (0..width * height * u32::from(components))
        .map(|index| (index.wrapping_mul(2_654_435_761) >> 24) as u8)
        .collect();
    encode(&pixels, width, height, components, 8, false, options).expect("encode fixture")
}

fn route(bytes: &[u8], scale: Downscale, fmt: PixelFormat, budget: usize) -> FullImageRoute {
    let image = image_at_scale(bytes, DecodeSettings::default(), scale).expect("image");
    full_image_route(&image, fmt, budget)
}

/// The floor the native decoder plans for the planes of `bytes` at `scale`.
fn floor(bytes: &[u8], scale: Downscale) -> usize {
    let image = image_at_scale(bytes, DecodeSettings::default(), scale).expect("image");
    image
        .full_decode_planes_floor(0)
        .expect("plan")
        .expect("a streamable image has a floor")
}

#[test]
fn an_image_streams_only_when_its_planes_pass_the_budget() {
    let fixtures = [
        codestream(
            3,
            &EncodeOptions {
                num_decomposition_levels: 3,
                reversible: false,
                ..EncodeOptions::default()
            },
        ),
        codestream(
            1,
            &EncodeOptions {
                num_decomposition_levels: 3,
                tile_size: Some((32, 32)),
                use_ht_block_coding: true,
                ..EncodeOptions::default()
            },
        ),
    ];
    for bytes in &fixtures {
        for scale in [Downscale::None, Downscale::Half, Downscale::Quarter] {
            let floor = floor(bytes, scale);
            // The planes alone of the smallest output are more than this.
            assert!(floor > 20 * 24 * 4, "{scale:?}: {floor}");
            for fmt in [PixelFormat::Rgb8, PixelFormat::Rgba8, PixelFormat::Gray8] {
                assert_eq!(
                    route(bytes, scale, fmt, floor - 1),
                    FullImageRoute::StreamedRegion
                );
                assert_eq!(route(bytes, scale, fmt, floor), FullImageRoute::Planes);
                assert_eq!(
                    route(bytes, scale, fmt, j2k_native::DEFAULT_MAX_DECODE_BYTES),
                    FullImageRoute::Planes
                );
            }
        }
    }
}

#[test]
fn reduced_decodes_plan_full_resolution_coefficients() {
    let bytes = codestream(
        1,
        &EncodeOptions {
            num_decomposition_levels: 3,
            ..EncodeOptions::default()
        },
    );
    // A quarter of the planes and IDWT buffers, but every coefficient.
    let full = floor(&bytes, Downscale::None);
    let half = floor(&bytes, Downscale::Half);
    assert!(half < full, "{half} < {full}");
    assert!(half > 96 * 80 * 4, "{half}");
}

#[test]
fn sixteen_bit_output_keeps_the_planes() {
    let bytes = codestream(3, &EncodeOptions::default());
    for fmt in [PixelFormat::Rgb16, PixelFormat::Rgba16, PixelFormat::Gray16] {
        assert_eq!(
            route(&bytes, Downscale::None, fmt, 0),
            FullImageRoute::Planes
        );
    }
}

/// A tile whose coarse resolutions are empty while finer ones are not keeps
/// the planes, which reconstruct it, whatever the budget.
#[test]
fn tiles_with_empty_coarse_levels_keep_the_planes() {
    let gray: &[u8] = include_bytes!(
        "../../../../j2k-test-support/fixtures/empty_coarse_levels/openjpeg_gray_8x6_tile8x5_53.j2k"
    );
    let rgb: &[u8] = include_bytes!(
        "../../../../j2k-test-support/fixtures/empty_coarse_levels/openjpeg_rgb_6x6_tile5x5_97.j2k"
    );
    for (bytes, fmt) in [(gray, PixelFormat::Gray8), (rgb, PixelFormat::Rgb8)] {
        let image =
            image_at_scale(bytes, DecodeSettings::default(), Downscale::None).expect("image");
        assert_eq!(image.full_decode_planes_floor(0).expect("plan"), None);
        assert_eq!(full_image_route(&image, fmt, 0), FullImageRoute::Planes);
    }
}
