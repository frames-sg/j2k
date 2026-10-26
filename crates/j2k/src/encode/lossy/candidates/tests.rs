// SPDX-License-Identifier: MIT OR Apache-2.0

//! The CPU search against independent per-candidate encodes.

use alloc::{format, string::String};

use j2k_native::DEFAULT_MAX_DECODE_BYTES;

use super::super::{
    encode_cpu_lossy_targeted, encode_lossy_targeted, J2kBlockCodingMode, J2kRateTarget,
    LossyAttempt,
};
use super::*;
use crate::encode::samples::raw_pixel_bytes_per_sample;

fn textured(width: u32, height: u32, components: u16, bit_depth: u8) -> Vec<u8> {
    let bytes_per_sample = raw_pixel_bytes_per_sample(bit_depth);
    let count = width as usize * height as usize * usize::from(components);
    let mask = (1_u64 << bit_depth) - 1;
    let mut state = u64::from(width) * 7_919 + u64::from(components);
    let mut pixels = Vec::with_capacity(count * bytes_per_sample);
    for index in 0..count as u64 {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let gradient = index.wrapping_mul(mask / 61 + 1);
        let value = (gradient ^ ((state >> 33) & (mask >> 3))) & mask;
        pixels.extend_from_slice(&value.to_le_bytes()[..bytes_per_sample]);
    }
    pixels
}

fn options(block_coding_mode: J2kBlockCodingMode, budget: u8) -> J2kLossyEncodeOptions {
    J2kLossyEncodeOptions {
        block_coding_mode,
        psnr_iteration_budget: budget,
        ..J2kLossyEncodeOptions::default()
    }
}

fn outcome(result: Result<LossyAttempt, J2kError>) -> Result<(Vec<u8>, u32), String> {
    result
        .map(|attempt| (attempt.codestream, attempt.quantization_scale.to_bits()))
        .map_err(|error| format!("{error:?}"))
}

#[test]
fn targeted_cpu_search_matches_independent_candidates() {
    use J2kBlockCodingMode::{Classic, HighThroughput};
    let (mut searches, mut successes) = (0_usize, 0_usize);
    for (width, height, components, bit_depth, signed, constant) in [
        (48, 40, 3, 8, false, false),
        (63, 65, 3, 8, false, false),
        (63, 65, 3, 8, false, true),
        (1, 37, 1, 16, false, false),
        (24, 20, 3, 12, true, false),
    ] {
        let mut pixels = textured(width, height, components, bit_depth);
        if constant {
            pixels.fill(0);
        }
        let samples = J2kLossySamples::new(&pixels, width, height, components, bit_depth, signed)
            .expect("valid synthetic samples");
        let roi = [NativeEncodeRoiRegion {
            component: components - 1,
            x: width / 3,
            y: height / 4,
            width: (width / 2).max(1),
            height: (height / 2).max(1),
            // Maxshift must cover the background bitplanes at fine scales.
            shift: bit_depth + 6,
        }];
        for target in [
            J2kRateTarget::Bytes(pixels.len() as u64 / 5 + 1),
            J2kRateTarget::BitsPerPixel(0.5),
            J2kRateTarget::PsnrDb(28.0),
            J2kRateTarget::PsnrDb(200.0),
        ] {
            for (mode, budget, with_roi, layer_count) in [
                (HighThroughput, 1, false, 0),
                (HighThroughput, 8, true, 0),
                (Classic, 8, false, 0),
                (Classic, 1, true, 0),
                (HighThroughput, 8, false, 1),
                (Classic, 8, false, 2),
            ] {
                let mut options = options(mode, budget);
                if layer_count != 0 {
                    if matches!(target, J2kRateTarget::PsnrDb(_)) {
                        continue;
                    }
                    if layer_count == 2 {
                        options
                            .quality_layers
                            .push(crate::J2kQualityLayer::new(J2kRateTarget::Bytes(1)));
                    }
                    options
                        .quality_layers
                        .push(crate::J2kQualityLayer::new(target));
                }
                let target = if layer_count == 0 {
                    Some(target)
                } else {
                    super::super::effective_lossy_target(&options).unwrap()
                };
                let roi = with_roi.then_some(roi.as_slice());
                let actual = encode_cpu_lossy_targeted(samples, &options, target, roi);
                let independent = &mut |scale| {
                    roi.map_or_else(
                        || encode_cpu_lossy(samples, &options, scale),
                        |roi| encode_cpu_lossy_with_roi_regions(samples, &options, scale, roi),
                    )
                };
                let expected = encode_lossy_targeted(samples, &options, target, independent);
                let expected = outcome(expected);
                let what = format!("{width}x{height} {target:?} {mode:?} {budget} roi {with_roi}");
                assert_eq!(outcome(actual), expected, "{what}");
                searches += 1;
                successes += usize::from(expected.is_ok());
            }
        }
    }
    // Most searches must reach a codestream for the comparison to mean much.
    assert!(
        successes * 3 >= searches * 2,
        "{successes} of {searches} searches succeeded"
    );
}

#[test]
fn refused_quality_decode_releases_the_transform() {
    let pixels = textured(48, 40, 3, 8);
    let samples = J2kLossySamples::new(&pixels, 48, 40, 3, 8, false).expect("valid samples");
    for mode in [
        J2kBlockCodingMode::HighThroughput,
        J2kBlockCodingMode::Classic,
    ] {
        let options = options(mode, 8);
        let mut candidates = CpuCandidates::new(samples, &options, None, true);
        let codestream = candidates.encode_candidate(1.0).expect("first candidate");
        assert!(candidates.encoder.is_some(), "{mode:?}");
        // A retained charge as large as the decode cap refuses the decode.
        let psnr = candidates
            .measure(samples, &codestream, DEFAULT_MAX_DECODE_BYTES)
            .expect("plain decode fallback");
        let plain = decoded_psnr(samples, &codestream, 0).expect("plain decode");
        assert_eq!(psnr.to_bits(), plain.to_bits(), "{mode:?}");
        assert!(
            candidates.encoder.is_none() && !candidates.reuse,
            "{mode:?}"
        );
        let next = candidates
            .encode_candidate(2.0)
            .expect("independent candidate");
        let independent = encode_cpu_lossy(samples, &options, 2.0).expect("reference");
        assert_eq!(next, independent, "{mode:?}");
    }
}
