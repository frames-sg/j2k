// SPDX-License-Identifier: MIT OR Apache-2.0

//! Decoding one component at a time into a shared coefficient buffer must
//! match the whole-tile layout bit for bit. A declining code-block decoder
//! hook keeps the decode on the whole-tile layout while every stage still
//! runs on the CPU, so the two layouts can be compared directly.

use alloc::vec::Vec;

use super::gradient_pixels;
use crate::{
    encode, encode_component_planes_53, encode_typed_component_planes_53, j2c, DecodeSettings,
    DecoderContext, EncodeComponentPlane, EncodeOptions, EncodeTypedComponentPlane,
    HtCodeBlockDecoder, Image,
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

/// Precision above the f32 path, so the decode keeps exact `i64` samples.
const HIGH_BIT_DEPTH: u8 = 29;

/// Little-endian `u32` samples spanning the full 29-bit range.
fn high_bit_samples(width: u32, height: u32, seed: u32) -> Vec<u8> {
    let mask = (1_u32 << HIGH_BIT_DEPTH) - 1;
    (0..height)
        .flat_map(|y| (0..width).map(move |x| (x, y)))
        .map(|(x, y)| {
            x.wrapping_mul(0x0135_7b9f)
                .wrapping_add(y.wrapping_mul(0x00ae_4c23))
                .wrapping_add(seed.wrapping_mul(0x0d1f_0e57))
                & mask
        })
        .flat_map(u32::to_le_bytes)
        .collect()
}

fn high_bit_codestream(components: Components, ht: bool, tiled: bool) -> Vec<u8> {
    let chroma_sampling = if matches!(components, Components::Subsampled) {
        2
    } else {
        1
    };
    let chroma_size = (
        WIDTH.div_ceil(u32::from(chroma_sampling)),
        HEIGHT.div_ceil(u32::from(chroma_sampling)),
    );
    let luma = high_bit_samples(WIDTH, HEIGHT, 0);
    let blue = high_bit_samples(chroma_size.0, chroma_size.1, 1);
    let red = high_bit_samples(chroma_size.0, chroma_size.1, 2);
    let plane = |data, sampling| EncodeTypedComponentPlane {
        data,
        x_rsiz: sampling,
        y_rsiz: sampling,
        bit_depth: HIGH_BIT_DEPTH,
        signed: false,
    };
    let planes = [
        plane(&luma, 1),
        plane(&blue, chroma_sampling),
        plane(&red, chroma_sampling),
    ];
    let component_count = if matches!(components, Components::Gray) {
        1
    } else {
        planes.len()
    };
    encode_typed_component_planes_53(
        &planes[..component_count],
        WIDTH,
        HEIGHT,
        &EncodeOptions {
            use_mct: matches!(components, Components::RgbWithMct),
            ..options(true, ht, tiled)
        },
    )
    .expect("encode high-bit layout fixture")
}

/// Exact integer component planes, and the peak coefficient bytes, which
/// include the `i64` buffer. No public API returns these planes through a hook, so this calls
/// the codestream decoder directly. Whole images only: 25-38 bit region
/// decode is not supported yet.
fn decode_exact_planes(image: &Image<'_>, hook: bool) -> (Vec<Vec<i64>>, usize) {
    let mut context = DecoderContext::default();
    let mut hook_decoder = DecliningDecoder;
    let mut ht_decoder: Option<&mut dyn HtCodeBlockDecoder> =
        if hook { Some(&mut hook_decoder) } else { None };
    j2c::decode(
        image.codestream,
        &image.header,
        0,
        &mut context,
        &mut ht_decoder,
        #[cfg(feature = "std")]
        None,
    )
    .expect("decode high-bit layout fixture");
    let planes = context
        .tile_decode_context
        .channel_data
        .iter()
        .map(|component| {
            component
                .integer_container
                .clone()
                .expect("29-bit decode keeps exact integer samples")
        })
        .collect();
    (
        planes,
        context
            .tile_decode_context
            .debug_counters
            .workspace_peak
            .bytes()
            .coefficients,
    )
}

#[test]
fn high_bit_component_at_a_time_decode_matches_the_whole_tile_layout() {
    let cases = [
        Components::Gray,
        Components::RgbWithMct,
        Components::RgbWithoutMct,
        Components::Subsampled,
    ];
    for components in cases {
        for ht in [false, true] {
            for tiled in [false, true] {
                let bytes = high_bit_codestream(components, ht, tiled);
                for reduction in [0_u8, 1] {
                    let image =
                        Image::new_with_reduction(&bytes, &DecodeSettings::default(), reduction)
                            .expect("parse high-bit layout fixture");
                    let label = format!(
                        "{components:?} ht={ht} tiled={tiled} \
                         reduction={reduction}"
                    );
                    let (tile_layout, tile_coefficients) = decode_exact_planes(&image, true);
                    let (component_layout, component_coefficients) =
                        decode_exact_planes(&image, false);
                    assert_eq!(component_layout, tile_layout, "{label}");
                    if !matches!(components, Components::Gray) {
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
