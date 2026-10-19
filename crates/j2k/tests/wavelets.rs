// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use j2k::{
    encode_j2k_lossless, encode_j2k_lossless_typed_components, encode_j2k_lossy,
    BatchDecodeOptions, BatchLayout, CpuBatchDecoder, CpuBatchSamples, EncodedImage,
    J2kBlockCodingMode, J2kDecoder, J2kLosslessEncodeOptions, J2kLosslessSamples,
    J2kLosslessTypedComponentPlane, J2kLosslessTypedComponentSamples, J2kLossyEncodeOptions,
    J2kLossySamples, WaveletBandKind, WaveletComponent, WaveletSamples, WaveletSelection,
    WaveletTransform,
};

fn pixels() -> Vec<u8> {
    (0..65 * 67)
        .map(|i| u8::try_from((i * 37 + i / 65 * 19) % 256).unwrap())
        .collect()
}

fn lossless(ht: bool) -> Vec<u8> {
    encode_j2k_lossless(
        J2kLosslessSamples::new(&pixels(), 65, 67, 1, 8, false).unwrap(),
        &J2kLosslessEncodeOptions::default()
            .with_max_decomposition_levels(Some(2))
            .with_block_coding_mode(if ht {
                J2kBlockCodingMode::HighThroughput
            } else {
                J2kBlockCodingMode::Classic
            }),
    )
    .unwrap()
    .codestream
}

// Independent lifting oracles, including the phase of odd tile origins.
fn inverse53(low: &[i64], high: &[i64], origin: u32) -> Vec<i64> {
    if high.is_empty() {
        return low.to_vec();
    }
    if low.is_empty() {
        return high.iter().map(|&v| v >> 1).collect();
    }
    let low_start = i64::from(origin.div_ceil(2));
    let high_start = i64::from(origin / 2);
    let high_at = |i: i64| {
        high[usize::try_from((i - high_start).clamp(0, i64::try_from(high.len()).unwrap() - 1))
            .unwrap()]
    };
    let mut out = vec![0; low.len() + high.len()];
    for (i, &value) in low.iter().enumerate() {
        let x = low_start + i64::try_from(i).unwrap();
        out[usize::try_from(2 * x - i64::from(origin)).unwrap()] =
            value - ((high_at(x - 1) + high_at(x) + 2) >> 2);
    }
    for (i, &value) in high.iter().enumerate() {
        let x = high_start + i64::try_from(i).unwrap();
        let even = |index: i64| {
            let index = index.clamp(low_start, low_start + i64::try_from(low.len()).unwrap() - 1);
            out[usize::try_from(2 * index - i64::from(origin)).unwrap()]
        };
        out[usize::try_from(2 * x + 1 - i64::from(origin)).unwrap()] =
            value + ((even(x) + even(x + 1)) >> 1);
    }
    out
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "the independent F64 lifting oracle returns the API's F32 coefficient precision"
)]
fn inverse97(low: &[f32], high: &[f32], origin: u32) -> Vec<f32> {
    if low.len() + high.len() == 1 {
        return if high.is_empty() {
            low.to_vec()
        } else {
            vec![high[0] * 0.5]
        };
    }
    let low_start = i64::from(origin.div_ceil(2));
    let high_start = i64::from(origin / 2);
    let mut lo: Vec<f64> = low
        .iter()
        .map(|&v| f64::from(v) * 1.230_174_104_914_001)
        .collect();
    let mut hi: Vec<f64> = high
        .iter()
        .map(|&v| f64::from(v) / 1.230_174_104_914_001)
        .collect();
    for (even, coefficient) in [
        (true, 0.443_506_852_043_971),
        (false, 0.882_911_075_530_934),
        (true, -0.052_980_118_572_961),
        (false, -1.586_134_342_059_924),
    ] {
        if even {
            for (i, value) in lo.iter_mut().enumerate() {
                let x = low_start + i64::try_from(i).unwrap();
                let at = |n: i64| {
                    hi[usize::try_from(
                        (n - high_start).clamp(0, i64::try_from(hi.len()).unwrap() - 1),
                    )
                    .unwrap()]
                };
                *value -= coefficient * (at(x - 1) + at(x));
            }
        } else {
            for (i, value) in hi.iter_mut().enumerate() {
                let x = high_start + i64::try_from(i).unwrap();
                let at = |n: i64| {
                    lo[usize::try_from(
                        (n - low_start).clamp(0, i64::try_from(lo.len()).unwrap() - 1),
                    )
                    .unwrap()]
                };
                *value -= coefficient * (at(x) + at(x + 1));
            }
        }
    }
    let mut out = vec![0.0; low.len() + high.len()];
    for (i, value) in lo.into_iter().enumerate() {
        out[usize::try_from(2 * (low_start + i64::try_from(i).unwrap()) - i64::from(origin))
            .unwrap()] = value as f32;
    }
    for (i, value) in hi.into_iter().enumerate() {
        out[usize::try_from(
            2 * (high_start + i64::try_from(i).unwrap()) + 1 - i64::from(origin),
        )
        .unwrap()] = value as f32;
    }
    out
}

fn integers(samples: &WaveletSamples) -> &[i64] {
    let WaveletSamples::I64(values) = samples else {
        panic!("reversible coefficients must be exact integers")
    };
    values
}

fn floats(samples: &WaveletSamples) -> &[f32] {
    let WaveletSamples::F32(values) = samples else {
        panic!("irreversible coefficients must be dequantized floats")
    };
    values
}

fn reconstruct<T: Copy>(
    component: &WaveletComponent,
    samples: fn(&WaveletSamples) -> &[T],
    inverse: fn(&[T], &[T], u32) -> Vec<T>,
) -> Vec<T> {
    let ll = &component.bands[0];
    assert_eq!(ll.kind, WaveletBandKind::LowLow);
    let mut values = samples(&ll.samples).to_vec();
    let mut width = ll.rect.width() as usize;
    let mut height = ll.rect.height() as usize;
    for resolution in 1..=component.decomposition_levels {
        let band = |kind| {
            component
                .bands
                .iter()
                .find(|b| b.resolution == resolution && b.kind == kind)
                .unwrap()
        };
        let hl = band(WaveletBandKind::HighLow);
        let lh = band(WaveletBandKind::LowHigh);
        let hh = band(WaveletBandKind::HighHigh);
        let high_width = hl.rect.width() as usize;
        let high_height = lh.rect.height() as usize;
        let denominator = 1_u32 << (component.decomposition_levels - resolution);
        let origin_x = component.rect.x0.div_ceil(denominator);
        let origin_y = component.rect.y0.div_ceil(denominator);
        // Synthesis applies horizontal lifting before vertical lifting.
        let low_rows = (0..height)
            .map(|y| {
                inverse(
                    &values[y * width..(y + 1) * width],
                    &samples(&hl.samples)[y * high_width..(y + 1) * high_width],
                    origin_x,
                )
            })
            .collect::<Vec<_>>();
        let high_rows = (0..high_height)
            .map(|y| {
                inverse(
                    &samples(&lh.samples)[y * width..(y + 1) * width],
                    &samples(&hh.samples)[y * high_width..(y + 1) * high_width],
                    origin_x,
                )
            })
            .collect::<Vec<_>>();
        let columns = (0..width + high_width)
            .map(|x| {
                inverse(
                    &(0..height).map(|y| low_rows[y][x]).collect::<Vec<_>>(),
                    &(0..high_height)
                        .map(|y| high_rows[y][x])
                        .collect::<Vec<_>>(),
                    origin_y,
                )
            })
            .collect::<Vec<_>>();
        values = (0..height + high_height)
            .flat_map(|y| columns.iter().map(move |column| column[y]))
            .collect();
        width += high_width;
        height += high_height;
    }
    assert_eq!(
        (width, height),
        (
            component.rect.width() as usize,
            component.rect.height() as usize
        )
    );
    values
}

#[test]
fn classic_and_ht_bands_reconstruct_original_odd_sized_image() {
    let mut previous = None;
    for ht in [false, true] {
        let bytes = lossless(ht);
        let decoded = J2kDecoder::new(&bytes)
            .unwrap()
            .decode_wavelets(&WaveletSelection::default())
            .unwrap();
        assert_eq!((decoded.rect.width(), decoded.rect.height()), (65, 67));
        assert_eq!(decoded.tiles.len(), 1);
        let component = &decoded.tiles[0].components[0];
        assert_eq!(component.transform, WaveletTransform::Reversible53);
        assert_eq!(component.decomposition_levels, 2);
        let reconstructed = reconstruct(component, integers, inverse53);
        for (index, (actual, expected)) in reconstructed.iter().zip(pixels()).enumerate() {
            assert_eq!(*actual, i64::from(expected) - 128, "pixel {index}, HT={ht}");
        }
        let bands = component
            .bands
            .iter()
            .map(|b| integers(&b.samples).to_vec())
            .collect::<Vec<_>>();
        if let Some(expected) = previous {
            assert_eq!(bands, expected);
        }
        previous = Some(bands);
    }
}

#[test]
fn selected_bands_match_full_decode_and_invalid_selection_is_rejected() {
    let bytes = lossless(true);
    let mut decoder = J2kDecoder::new(&bytes).unwrap();
    let full = decoder
        .decode_wavelets(&WaveletSelection::default())
        .unwrap();
    let selection = WaveletSelection {
        resolutions: vec![1],
        bands: vec![WaveletBandKind::HighLow],
        ..Default::default()
    };
    let selected = decoder.decode_wavelets(&selection).unwrap();
    let bands = &selected.tiles[0].components[0].bands;
    assert_eq!(bands.len(), 1);
    let expected = full.tiles[0].components[0]
        .bands
        .iter()
        .find(|b| b.resolution == 1 && b.kind == WaveletBandKind::HighLow)
        .unwrap();
    assert_eq!(integers(&bands[0].samples), integers(&expected.samples));
    for invalid in [
        WaveletSelection {
            components: vec![1],
            ..Default::default()
        },
        WaveletSelection {
            tiles: vec![1],
            ..Default::default()
        },
        WaveletSelection {
            resolutions: vec![99],
            ..Default::default()
        },
        WaveletSelection {
            resolutions: vec![0],
            bands: vec![WaveletBandKind::HighHigh],
            ..Default::default()
        },
    ] {
        assert!(decoder.decode_wavelets(&invalid).is_err());
    }
    assert_eq!(
        decoder.decode_wavelets(&selection).unwrap().tiles[0].components[0]
            .bands
            .len(),
        1
    );
}

#[test]
fn irreversible_coefficients_reconstruct_within_one_lsb_of_pixel_decode() {
    for ht in [false, true] {
        let bytes = encode_j2k_lossy(
            J2kLossySamples::new(&pixels(), 65, 67, 1, 8, false).unwrap(),
            &J2kLossyEncodeOptions::default().with_block_coding_mode(if ht {
                J2kBlockCodingMode::HighThroughput
            } else {
                J2kBlockCodingMode::Classic
            }),
        )
        .unwrap()
        .codestream;
        let mut decoder = J2kDecoder::new(&bytes).unwrap();
        let output = decoder
            .decode_wavelets(&WaveletSelection::default())
            .unwrap();
        let component = &output.tiles[0].components[0];
        assert_eq!(component.transform, WaveletTransform::Irreversible97);
        assert!(component.decomposition_levels > 0);
        let expected = j2k_native::Image::new(&bytes, &j2k_native::DecodeSettings::default())
            .unwrap()
            .decode_native()
            .unwrap();
        for (i, (actual, expected)) in reconstruct(component, floats, inverse97)
            .into_iter()
            .zip(expected.data)
            .enumerate()
        {
            let actual = (actual.round() + 128.0).clamp(0.0, 255.0);
            assert!(
                (actual - f32::from(expected)).abs() <= 1.0,
                "HT={ht}, pixel={i}, actual={actual}, expected={expected}"
            );
        }
    }
}

#[test]
fn independent_signed_multitile_coefficients_preserve_origins_and_selection() {
    let bytes =
        include_bytes!("../../j2k-test-support/fixtures/htj2k/openjph_batch/gray_s16_53.j2c");
    let oracle = include_bytes!(
        "../../j2k-test-support/fixtures/htj2k/openjph_batch/gray_s16_53.oracle.raw"
    );
    let mut decoder = J2kDecoder::new(bytes).unwrap();
    let full = decoder
        .decode_wavelets(&WaveletSelection::default())
        .unwrap();
    assert_eq!(full.tiles.len(), 4);
    for tile in &full.tiles {
        let component = &tile.components[0];
        assert!(component.signed);
        assert_eq!(component.precision, 16);
        for (i, actual) in reconstruct(component, integers, inverse53)
            .into_iter()
            .enumerate()
        {
            let x = i % component.rect.width() as usize + component.rect.x0 as usize;
            let y = i / component.rect.width() as usize + component.rect.y0 as usize;
            let offset = 2 * (y * 19 + x);
            assert_eq!(
                actual,
                i64::from(i16::from_le_bytes([oracle[offset], oracle[offset + 1]])),
                "tile={}, pixel=({x},{y})",
                tile.index
            );
        }
    }
    let selected = decoder
        .decode_wavelets(&WaveletSelection {
            tiles: vec![3],
            ..Default::default()
        })
        .unwrap();
    assert_eq!(selected.tiles.len(), 1);
    assert_eq!(selected.tiles[0].index, 3);
    assert_eq!(
        reconstruct(&selected.tiles[0].components[0], integers, inverse53),
        reconstruct(&full.tiles[3].components[0], integers, inverse53)
    );
}

fn decoded_unsigned_planes(bytes: &[u8]) -> Vec<f64> {
    let mut decoder = CpuBatchDecoder::new(BatchDecodeOptions {
        layout: BatchLayout::Nchw,
        ..BatchDecodeOptions::default()
    });
    let result = decoder
        .decode(vec![EncodedImage::full(Arc::from(bytes))])
        .unwrap();
    assert!(result.errors().is_empty(), "{:?}", result.errors());
    match result.groups()[0].samples() {
        CpuBatchSamples::U8(values) => values.iter().map(|&v| f64::from(v)).collect(),
        CpuBatchSamples::U16(values) => values.iter().map(|&v| f64::from(v)).collect(),
        _ => panic!("expected unsigned RGB samples"),
    }
}

fn component_plane(component: &WaveletComponent) -> Vec<f64> {
    match component.transform {
        WaveletTransform::Reversible53 => reconstruct(component, integers, inverse53)
            .into_iter()
            .map(|v| f64::from(i32::try_from(v).unwrap()))
            .collect(),
        WaveletTransform::Irreversible97 => reconstruct(component, floats, inverse97)
            .into_iter()
            .map(f64::from)
            .collect(),
    }
}

#[test]
fn color_transformed_components_invert_to_the_decoded_rgb_image() {
    for (name, bytes, tolerance) in [
        (
            "classic 5/3 RCT",
            &include_bytes!("../../j2k-test-support/fixtures/openjpeg_progression/pcrl_97x61.j2k")
                [..],
            0.0,
        ),
        (
            "classic 5/3 RCT, two tiles",
            &include_bytes!(
                "../../j2k-test-support/fixtures/openjpeg_progression/pcrl_97x61_tiles_plt.j2k"
            )[..],
            0.0,
        ),
        (
            "HTJ2K 9/7 ICT, 12-bit",
            &include_bytes!("../../j2k-test-support/fixtures/htj2k/openhtj2k_hifi_ht1_02.j2k")[..],
            1.0,
        ),
    ] {
        let output = J2kDecoder::new(bytes)
            .unwrap()
            .decode_wavelets(&WaveletSelection::default())
            .unwrap();
        let width = output.rect.width() as usize;
        let height = output.rect.height() as usize;
        let pixels = decoded_unsigned_planes(bytes);
        let mut worst = 0.0_f64;
        for tile in &output.tiles {
            assert!(tile.uses_color_transform, "{name}");
            let components = &tile.components;
            assert_eq!(
                components.iter().map(|c| c.index).collect::<Vec<_>>(),
                [0, 1, 2]
            );
            let first = &components[0];
            assert!(!first.signed && first.sampling == (1, 1), "{name}");
            let [y, cb, cr] = [0, 1, 2].map(|i| component_plane(&components[i]));
            // Inverse RCT (T.800 G.2) or ICT (G.3); component 1 is B-side, 2 is R-side.
            let rgb: [Vec<f64>; 3] = if first.transform == WaveletTransform::Reversible53 {
                let g = (0..y.len())
                    .map(|i| y[i] - ((cb[i] + cr[i]) / 4.0).floor())
                    .collect::<Vec<_>>();
                [
                    (0..y.len()).map(|i| cr[i] + g[i]).collect(),
                    g.clone(),
                    (0..y.len()).map(|i| cb[i] + g[i]).collect(),
                ]
            } else {
                [
                    (0..y.len()).map(|i| y[i] + 1.402 * cr[i]).collect(),
                    (0..y.len())
                        .map(|i| y[i] - 0.344_136 * cb[i] - 0.714_136 * cr[i])
                        .collect(),
                    (0..y.len()).map(|i| y[i] + 1.772 * cb[i]).collect(),
                ]
            };
            let shift = f64::from(1_u32 << (first.precision - 1));
            let max = f64::from((1_u32 << first.precision) - 1);
            let tile_width = first.rect.width() as usize;
            let x0 = (first.rect.x0 - output.rect.x0) as usize;
            let y0 = (first.rect.y0 - output.rect.y0) as usize;
            for (channel, plane) in rgb.iter().enumerate() {
                for (i, value) in plane.iter().enumerate() {
                    let actual = if first.transform == WaveletTransform::Reversible53 {
                        value + shift
                    } else {
                        (value + shift).round().clamp(0.0, max)
                    };
                    let (x, y) = (x0 + i % tile_width, y0 + i / tile_width);
                    let expected = pixels[(channel * height + y) * width + x];
                    worst = worst.max((actual - expected).abs());
                }
            }
        }
        assert!(worst <= tolerance, "{name}: max error {worst}");
    }
}

#[test]
fn high_precision_coefficients_never_round_through_float_storage() {
    let values = (0..65 * 67)
        .map(|i| ((i % 1024) - 512) * 65_536 + (i % 3))
        .collect::<Vec<i32>>();
    let raw = values
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>();
    let planes = [J2kLosslessTypedComponentPlane {
        data: &raw,
        x_rsiz: 1,
        y_rsiz: 1,
        bit_depth: 29,
        signed: true,
    }];
    let bytes = encode_j2k_lossless_typed_components(
        J2kLosslessTypedComponentSamples::new(&planes, 65, 67).unwrap(),
        &J2kLosslessEncodeOptions::default().with_max_decomposition_levels(Some(2)),
    )
    .unwrap()
    .codestream;
    let output = J2kDecoder::new(&bytes)
        .unwrap()
        .decode_wavelets(&WaveletSelection::default())
        .unwrap();
    assert_eq!(output.tiles[0].components[0].decomposition_levels, 2);
    for (actual, expected) in reconstruct(&output.tiles[0].components[0], integers, inverse53)
        .into_iter()
        .zip(values)
    {
        assert_eq!(actual, i64::from(expected));
    }
}
