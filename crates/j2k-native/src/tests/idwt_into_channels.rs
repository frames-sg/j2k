// SPDX-License-Identifier: MIT OR Apache-2.0

//! A region decode that one tile covers writes the last IDWT level into the
//! padding of each `channel_data` plane and compacts it there. That must
//! match keeping the level in its own window and copying it in `store`, bit
//! for bit, for component planes and for packed integer output.

use alloc::vec::Vec;

use crate::{encode, DecodeSettings, DecoderContext, EncodeOptions, Image};

const WIDTH: u32 = 256;
const HEIGHT: u32 = 192;
const TILE: u32 = 64;

/// Deterministic noise, so every sub-band carries significant coefficients.
fn noisy_pixels(components: u8) -> Vec<u8> {
    let mut state = 0x2545_f491_u32;
    (0..WIDTH * HEIGHT * u32::from(components))
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state >> 24) as u8
        })
        .collect()
}

fn codestream(components: u8, reversible: bool, tiled: bool) -> Vec<u8> {
    encode(
        &noisy_pixels(components),
        WIDTH,
        HEIGHT,
        u16::from(components),
        8,
        false,
        &EncodeOptions {
            reversible,
            num_decomposition_levels: 4,
            tile_size: tiled.then_some((TILE, TILE)),
            ..EncodeOptions::default()
        },
    )
    .expect("encode channel IDWT fixture")
}

struct Decoded {
    planes: Vec<Vec<u32>>,
    /// Packed integer output, which rounds irreversible samples in `store`
    /// when there is no MCT. Decoded for one-component images only.
    packed: Vec<u8>,
    peak_idwt: usize,
    peak_total: usize,
}

fn decode(image: &Image<'_>, roi: (u32, u32, u32, u32), separate_last_level: bool) -> Decoded {
    let mut context = DecoderContext::default();
    // The row-streamed IDWT replaces the padded last level, so compare the
    // whole-window paths.
    context.set_line_region_idwt(false);
    context.tile_decode_context.separate_last_idwt_level = separate_last_level;
    // Off the tile-wide route, whose worker workspaces would make Tier-1 the
    // peak of these small images and hide the IDWT bytes compared here.
    #[cfg(feature = "parallel")]
    {
        context.tile_decode_context.force_per_sub_band = true;
    }
    let planes: Vec<Vec<u32>> = {
        let decoded = image
            .decode_region_components_with_context(roi, &mut context)
            .expect("decode channel IDWT planes");
        decoded
            .planes()
            .iter()
            .map(|plane| {
                plane
                    .samples()
                    .iter()
                    .map(|sample| sample.to_bits())
                    .collect()
            })
            .collect()
    };
    let peak = context
        .tile_decode_context
        .debug_counters
        .workspace_peak
        .bytes();
    let packed = if planes.len() == 1 {
        image
            .decode_region_with_context(roi, &mut context)
            .expect("decode channel IDWT pixels")
            .data
    } else {
        Vec::new()
    };
    Decoded {
        planes,
        packed,
        peak_idwt: peak.idwt,
        peak_total: peak.total(),
    }
}

#[test]
fn last_idwt_level_in_channel_data_matches_a_separate_window_bit_for_bit() {
    for components in [1_u8, 3] {
        for reversible in [true, false] {
            for tiled in [false, true] {
                let bytes = codestream(components, reversible, tiled);
                for reduction in [0_u8, 1] {
                    let image =
                        Image::new_with_reduction(&bytes, &DecodeSettings::default(), reduction)
                            .expect("parse channel IDWT fixture");
                    let (width, height) = (image.width(), image.height());
                    let tile = TILE >> reduction;
                    let large = (
                        width / 24,
                        height / 24,
                        width - width / 12,
                        height - height / 12,
                    );
                    let regions = [
                        (0, 0, width, height),
                        large,
                        (width / 3, height / 3, width / 12, height / 12),
                        // Inside one tile of the tiled fixture.
                        (tile + 2, tile + 2, tile - 4, tile - 4),
                        (width - 1, height - 1, 1, 1),
                    ];
                    for roi in regions {
                        let label = format!(
                            "components={components} reversible={reversible} tiled={tiled} \
                             reduction={reduction} roi={roi:?}"
                        );
                        let separate = decode(&image, roi, true);
                        let in_channels = decode(&image, roi, false);
                        assert_eq!(in_channels.planes, separate.planes, "{label}");
                        assert_eq!(in_channels.packed, separate.packed, "{label}");
                        if roi == large && !tiled && reduction == 0 {
                            assert!(
                                in_channels.peak_idwt < separate.peak_idwt
                                    && in_channels.peak_total < separate.peak_total,
                                "{label}: idwt {} vs {}, total {} vs {}",
                                in_channels.peak_idwt,
                                separate.peak_idwt,
                                in_channels.peak_total,
                                separate.peak_total
                            );
                        }
                    }
                }
            }
        }
    }
}
