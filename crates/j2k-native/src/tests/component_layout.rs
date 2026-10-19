// SPDX-License-Identifier: MIT OR Apache-2.0

//! Decoding one component at a time into a shared coefficient buffer must
//! match the whole-tile layout bit for bit. A declining code-block decoder
//! hook keeps the decode on the whole-tile layout while every stage still
//! runs on the CPU, so the two layouts can be compared directly.

use alloc::vec::Vec;

use super::gradient_pixels;
use crate::{
    encode, encode_component_planes_53, DecodeSettings, DecoderContext, EncodeComponentPlane,
    EncodeOptions, HtCodeBlockDecoder, Image,
};

const WIDTH: u32 = 61;
const HEIGHT: u32 = 47;

struct DecliningDecoder;

impl HtCodeBlockDecoder for DecliningDecoder {}

#[derive(Clone, Copy, Debug)]
enum Components {
    Gray,
    RgbWithMct,
    RgbWithoutMct,
    /// Full-size luma with chroma subsampled by two on both axes.
    Subsampled,
}

fn options(reversible: bool, ht: bool, tiled: bool) -> EncodeOptions {
    EncodeOptions {
        reversible,
        use_ht_block_coding: ht,
        num_decomposition_levels: 3,
        code_block_width_exp: 1,
        code_block_height_exp: 1,
        tile_size: tiled.then_some((32, 32)),
        ..EncodeOptions::default()
    }
}

fn codestream(components: Components, reversible: bool, ht: bool, tiled: bool) -> Vec<u8> {
    match components {
        Components::Gray => encode(
            &gradient_pixels(WIDTH, HEIGHT, 1),
            WIDTH,
            HEIGHT,
            1,
            8,
            false,
            &EncodeOptions {
                use_mct: false,
                ..options(reversible, ht, tiled)
            },
        ),
        Components::RgbWithMct | Components::RgbWithoutMct => encode(
            &gradient_pixels(WIDTH, HEIGHT, 3),
            WIDTH,
            HEIGHT,
            3,
            8,
            false,
            &EncodeOptions {
                use_mct: matches!(components, Components::RgbWithMct),
                ..options(reversible, ht, tiled)
            },
        ),
        Components::Subsampled => {
            let luma = gradient_pixels(WIDTH, HEIGHT, 1);
            let chroma_size = (WIDTH.div_ceil(2), HEIGHT.div_ceil(2));
            let blue = gradient_pixels(chroma_size.0, chroma_size.1, 2)
                .into_iter()
                .step_by(2)
                .collect::<Vec<_>>();
            let red = gradient_pixels(chroma_size.0, chroma_size.1, 3)
                .into_iter()
                .step_by(3)
                .collect::<Vec<_>>();
            let plane = |data, sampling| EncodeComponentPlane {
                data,
                x_rsiz: sampling,
                y_rsiz: sampling,
            };
            let planes = [plane(&luma, 1), plane(&blue, 2), plane(&red, 2)];
            encode_component_planes_53(
                &planes,
                WIDTH,
                HEIGHT,
                8,
                false,
                &EncodeOptions {
                    use_mct: false,
                    ..options(true, ht, tiled)
                },
            )
        }
    }
    .expect("encode layout fixture")
}

/// Component planes as sample bits, and the coefficients the decode stored.
fn decode_planes(
    image: &Image<'_>,
    roi: Option<(u32, u32, u32, u32)>,
    hook: bool,
) -> (Vec<Vec<u32>>, usize) {
    let mut context = DecoderContext::default();
    let mut hook_decoder = DecliningDecoder;
    let decoded = match (roi, hook) {
        (None, false) => image.decode_components_with_context(&mut context),
        (None, true) => image.decode_components_with_ht_decoder(&mut context, &mut hook_decoder),
        (Some(roi), false) => image.decode_region_components_with_context(roi, &mut context),
        (Some(roi), true) => {
            image.decode_region_components_with_ht_decoder(&mut context, roi, &mut hook_decoder)
        }
    }
    .expect("decode layout fixture");
    let planes = decoded
        .planes()
        .iter()
        .map(|plane| {
            plane
                .samples()
                .iter()
                .map(|sample| sample.to_bits())
                .collect()
        })
        .collect();
    drop(decoded);
    (
        planes,
        context
            .tile_decode_context
            .debug_counters
            .stored_coefficients,
    )
}

#[test]
fn component_at_a_time_decode_matches_the_whole_tile_layout_bit_for_bit() {
    let cases = [
        (Components::Gray, true),
        (Components::Gray, false),
        (Components::RgbWithMct, true),
        (Components::RgbWithMct, false),
        (Components::RgbWithoutMct, false),
        (Components::Subsampled, true),
    ];
    for (components, reversible) in cases {
        for ht in [false, true] {
            for tiled in [false, true] {
                // The encoder does not tile subsampled components.
                if tiled && matches!(components, Components::Subsampled) {
                    continue;
                }
                let bytes = codestream(components, reversible, ht, tiled);
                for reduction in [0_u8, 1] {
                    let image =
                        Image::new_with_reduction(&bytes, &DecodeSettings::default(), reduction)
                            .expect("parse layout fixture");
                    let (width, height) = (image.width(), image.height());
                    let regions = [
                        None,
                        Some((0, 0, width, height)),
                        Some((width / 4, height / 5, width / 3, height / 3)),
                        Some((
                            width / 2,
                            height / 2,
                            width - width / 2,
                            height - height / 2,
                        )),
                        Some((width - 1, height - 1, 1, 1)),
                    ];
                    for roi in regions {
                        let label = format!(
                            "{components:?} reversible={reversible} ht={ht} tiled={tiled} \
                             reduction={reduction} roi={roi:?}"
                        );
                        let (tile_layout, tile_coefficients) = decode_planes(&image, roi, true);
                        let (component_layout, component_coefficients) =
                            decode_planes(&image, roi, false);
                        assert_eq!(component_layout, tile_layout, "{label}");
                        if !matches!(components, Components::Gray) {
                            // The shared buffer holds the largest component,
                            // not all three, on every decoded tile.
                            assert!(
                                component_coefficients < tile_coefficients,
                                "{label}: {component_coefficients} >= {tile_coefficients}"
                            );
                        }
                    }
                }
            }
        }
    }
}
