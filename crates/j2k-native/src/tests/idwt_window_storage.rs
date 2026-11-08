// SPDX-License-Identifier: MIT OR Apache-2.0

//! Region decode stores each sub-band only where the IDWT reads it, so
//! code-blocks on the edge of that window keep part of their samples. That
//! must match storing whole code-blocks bit for bit on every Tier-1 writer:
//! serial, per-sub-band parallel, and the tile-wide route.

use alloc::vec::Vec;

use crate::{encode, CpuDecodeParallelism, DecodeSettings, DecoderContext, EncodeOptions, Image};

const WIDTH: u32 = 157;
const HEIGHT: u32 = 131;

/// Deterministic noise, so every sub-band carries significant coefficients.
fn noisy_pixels(components: u8) -> Vec<u8> {
    let mut state = 0x9e37_79b9_u32;
    (0..WIDTH * HEIGHT * u32::from(components))
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state >> 24) as u8
        })
        .collect()
}

fn codestream(reversible: bool, ht: bool, tiled: bool) -> Vec<u8> {
    encode(
        &noisy_pixels(3),
        WIDTH,
        HEIGHT,
        3,
        8,
        false,
        &EncodeOptions {
            reversible,
            use_ht_block_coding: ht,
            num_decomposition_levels: 4,
            code_block_width_exp: 3,
            code_block_height_exp: 3,
            tile_size: tiled.then_some((64, 64)),
            ..EncodeOptions::default()
        },
    )
    .expect("encode window storage fixture")
}

#[derive(Clone, Copy, Debug)]
enum Route {
    Serial,
    Parallel,
    /// Parallel, with tiles kept off the tile-wide route.
    PerSubBand,
}

/// Region planes as sample bits, and the coefficients the decode stored.
fn decode_region(
    image: &Image<'_>,
    roi: (u32, u32, u32, u32),
    route: Route,
    whole_code_blocks: bool,
) -> (Vec<Vec<u32>>, usize) {
    let mut context = DecoderContext::default();
    context.set_cpu_decode_parallelism(match route {
        Route::Serial => CpuDecodeParallelism::Serial,
        Route::Parallel | Route::PerSubBand => CpuDecodeParallelism::Auto,
    });
    #[cfg(feature = "parallel")]
    {
        context.tile_decode_context.force_per_sub_band = matches!(route, Route::PerSubBand);
    }
    context.tile_decode_context.store_whole_code_blocks = whole_code_blocks;
    let decoded = image
        .decode_region_components_with_context(roi, &mut context)
        .expect("decode window storage fixture");
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
fn window_clipped_region_storage_matches_whole_code_blocks_bit_for_bit() {
    for reversible in [true, false] {
        for ht in [false, true] {
            for tiled in [false, true] {
                let bytes = codestream(reversible, ht, tiled);
                for reduction in [0_u8, 1] {
                    let image =
                        Image::new_with_reduction(&bytes, &DecodeSettings::default(), reduction)
                            .expect("parse window storage fixture");
                    let (width, height) = (image.width(), image.height());
                    let interior = (width / 4, height / 5, width / 3, height / 3);
                    let regions = [
                        interior,
                        (0, 0, width / 6, height / 6),
                        (
                            width - width / 4,
                            height - height / 4,
                            width / 4,
                            height / 4,
                        ),
                        (width / 2, height / 2, 1, 1),
                    ];
                    for roi in regions {
                        for route in [Route::Serial, Route::Parallel, Route::PerSubBand] {
                            let label = format!(
                                "reversible={reversible} ht={ht} tiled={tiled} \
                                 reduction={reduction} roi={roi:?} route={route:?}"
                            );
                            let (whole, whole_coefficients) =
                                decode_region(&image, roi, route, true);
                            let (clipped, clipped_coefficients) =
                                decode_region(&image, roi, route, false);
                            assert_eq!(clipped, whole, "{label}");
                            assert!(
                                clipped_coefficients <= whole_coefficients,
                                "{label}: {clipped_coefficients} > {whole_coefficients}"
                            );
                            // At 1/2 resolution the 9/7 margins reach every
                            // stored block of this small fixture.
                            if roi == interior && !tiled && reduction == 0 {
                                assert!(
                                    clipped_coefficients < whole_coefficients,
                                    "{label}: {clipped_coefficients} >= {whole_coefficients}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn exact_integer_region_uses_existing_pruning_and_fits_below_full_decode_budget() {
    let (width, height) = (512_u32, 384_u32);
    let pixels = (0..width * height)
        .flat_map(|index| (index.wrapping_mul(1_000_003) & ((1 << 29) - 1)).to_le_bytes())
        .collect::<Vec<_>>();
    let bytes = encode(
        &pixels,
        width,
        height,
        1,
        29,
        false,
        &EncodeOptions {
            reversible: true,
            num_decomposition_levels: 4,
            use_mct: false,
            ..EncodeOptions::default()
        },
    )
    .expect("exact integer source");
    let image = Image::new(&bytes, &DecodeSettings::strict()).unwrap();
    let mut full_context = DecoderContext::default();
    full_context.set_cpu_decode_parallelism(CpuDecodeParallelism::Serial);
    assert_eq!(
        image
            .decode_native_with_context(&mut full_context)
            .unwrap()
            .data,
        pixels
    );
    let full_work = full_context.tile_decode_context.debug_counters;
    drop(full_context);

    let roi = (191_u32, 129_u32, 17_u32, 13_u32);
    let expected = pixels
        .chunks_exact(width as usize * 4)
        .skip(roi.1 as usize)
        .take(roi.3 as usize)
        .flat_map(|row| {
            row[roi.0 as usize * 4..(roi.0 + roi.2) as usize * 4]
                .iter()
                .copied()
        })
        .collect::<Vec<_>>();
    let control = crate::DecodeOperationControl::new(Some(512 * 1024), None);
    let mut full_limited = DecoderContext::default();
    full_limited.set_cpu_decode_parallelism(CpuDecodeParallelism::Serial);
    let refused = full_limited.with_operation_control(control.clone(), |context| {
        image.decode_native_with_context(context).map(|_| ())
    });
    assert!(
        refused.as_ref().is_err_and(crate::j2c::is_capacity_error),
        "the small working allowance must reject full-image reconstruction"
    );
    drop(full_limited);

    for component_output in [false, true] {
        let mut context = DecoderContext::default();
        context.set_cpu_decode_parallelism(CpuDecodeParallelism::Serial);
        let actual = context
            .with_operation_control(control.clone(), |context| {
                if component_output {
                    image
                        .decode_native_region_components_with_context(roi, context)
                        .map(|mut image| image.planes.remove(0).data)
                } else {
                    image
                        .decode_native_region_with_context(roi, context)
                        .map(|image| image.data)
                }
            })
            .expect("a small exact integer ROI fits without full reconstruction");
        assert_eq!(actual, expected);
        let work = &context.tile_decode_context.debug_counters;
        assert!(work.decoded_code_blocks < full_work.decoded_code_blocks);
        assert!(work.stored_coefficients < full_work.stored_coefficients);
        assert!(work.idwt_output_samples < full_work.idwt_output_samples);
    }
}
