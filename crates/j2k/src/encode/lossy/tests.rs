// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    target_bytes_for_bpp, J2kError, J2kLossyEncodeOptions, J2kLossySamples, J2kRateTarget,
    Unsupported,
};
use crate::{encode_j2k_lossy, J2kProgressionOrder};

mod rate_validation;
mod target_contracts;

const TWO_TO_THE_64: f64 = 18_446_744_073_709_551_616.0;

fn single_pixel_samples() -> J2kLossySamples<'static> {
    J2kLossySamples::new(&[0], 1, 1, 1, 8, false).expect("valid single-pixel samples")
}

#[test]
fn bits_per_pixel_target_rejects_exactly_two_to_the_64_bytes() {
    let bits_per_pixel = TWO_TO_THE_64 * 8.0;
    let options = J2kLossyEncodeOptions::default()
        .with_rate_target(Some(J2kRateTarget::BitsPerPixel(bits_per_pixel)));

    let result = encode_j2k_lossy(single_pixel_samples(), &options);

    assert!(matches!(
        result,
        Err(J2kError::Unsupported(Unsupported {
            what: "JPEG 2000 lossy bits-per-pixel target overflows byte target"
        }))
    ));
}

#[test]
fn bits_per_pixel_target_accepts_largest_f64_below_two_to_the_64_bytes() {
    let largest_representable_byte_target = TWO_TO_THE_64.next_down();
    let bits_per_pixel = largest_representable_byte_target * 8.0;

    let target = target_bytes_for_bpp(single_pixel_samples(), bits_per_pixel)
        .expect("largest representable f64 byte target below 2^64 should fit");

    assert_eq!(target, u64::MAX - 2_047);
}

#[test]
fn explicit_lossy_depth_and_component_profile_survive_rate_search() {
    let pixels = (0..64 * 64 * 3)
        .map(|i| u8::try_from(i % 251).unwrap())
        .collect::<Vec<_>>();
    let samples = J2kLossySamples::new(&pixels, 64, 64, 3, 8, false).unwrap();
    let options = J2kLossyEncodeOptions::default()
        .with_cpu_only_backend()
        .with_max_decomposition_levels(Some(2))
        .with_quantization_profile(Some(vec![vec![1.0; 7], vec![2.0; 7], vec![3.0; 7]]))
        .with_rate_target(Some(J2kRateTarget::BitsPerPixel(8.0)));
    let encoded = encode_j2k_lossy(samples, &options).unwrap();
    assert!(encoded.report.actual_bytes <= 4096 + super::byte_target_tolerance(4096));
    let header = j2k_native::inspect_j2k_codestream_header(&encoded.codestream).unwrap();
    assert_eq!(header.resolution_levels, 3);
}

fn edge_tile_options(profile: Option<Vec<Vec<f32>>>) -> J2kLossyEncodeOptions {
    J2kLossyEncodeOptions::default()
        .with_cpu_only_backend()
        .with_max_decomposition_levels(Some(5))
        .with_tile_size(Some((64, 64)))
        .with_quantization_profile(profile)
}

#[test]
fn component_profile_rows_follow_the_smallest_edge_tile_depth() {
    // A 1 px edge column allows no decomposition; a 16 px one allows four.
    for (width, levels) in [(65_u32, 0_usize), (80, 4)] {
        let pixels = (0..width * 64 * 3)
            .map(|i| u8::try_from(i % 251).unwrap())
            .collect::<Vec<_>>();
        let samples = J2kLossySamples::new(&pixels, width, 64, 3, 8, false).unwrap();
        let bands = 1 + 3 * levels;
        let default = encode_j2k_lossy(samples, &edge_tile_options(None)).unwrap();
        let ones =
            encode_j2k_lossy(samples, &edge_tile_options(Some(vec![vec![1.0; bands]; 3]))).unwrap();
        assert_eq!(ones.codestream, default.codestream, "{width} px all-ones");

        let mut weights = vec![vec![1.0; bands]; 3];
        weights[1][0] = 4.0;
        let weighted = encode_j2k_lossy(samples, &edge_tile_options(Some(weights))).unwrap();
        assert_ne!(
            weighted.codestream, default.codestream,
            "{width} px weighted"
        );
        let header = j2k_native::inspect_j2k_codestream_header(&weighted.codestream).unwrap();
        assert_eq!(usize::from(header.resolution_levels), levels + 1);
        j2k_native::Image::new(&weighted.codestream, &j2k_native::DecodeSettings::default())
            .unwrap()
            .decode()
            .unwrap();

        // Rows sized for the requested five levels ignore the edge tile.
        let requested_depth = Some(vec![vec![1.0; 16]; 3]);
        assert!(
            matches!(
                encode_j2k_lossy(samples, &edge_tile_options(requested_depth)),
                Err(J2kError::Unsupported(_))
            ),
            "{width} px requested-depth rows"
        );
    }
}

#[test]
fn rate_search_stops_at_the_component_profile_scale_ceiling() {
    let mut state = 0x2545_f491_u32;
    let pixels = (0..128 * 128 * 3)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            u8::try_from(state >> 24).unwrap()
        })
        .collect::<Vec<_>>();
    let samples = J2kLossySamples::new(&pixels, 128, 128, 3, 8, false).unwrap();
    let options = J2kLossyEncodeOptions::default()
        .with_cpu_only_backend()
        .with_max_decomposition_levels(Some(2))
        .with_quantization_profile(Some(vec![
            vec![
                1.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0
            ];
            3
        ]));
    // Detail weights cap the scale near 2 while LL is still finely quantized.
    let ceiling = super::profile_scale_ceiling(samples, &options).unwrap();
    assert!(ceiling > 1.0 && ceiling < 4.0);

    // Unreachable without pushing detail steps out of range.
    let tiny = options
        .clone()
        .with_rate_target(Some(J2kRateTarget::Bytes(1)));
    assert!(matches!(
        encode_j2k_lossy(samples, &tiny),
        Err(J2kError::RateTargetUnreachable { .. })
    ));

    // Every scale meets 1 dB, so the search climbs until the ceiling stops it.
    let mut psnr = options.with_rate_target(Some(J2kRateTarget::PsnrDb(1.0)));
    psnr.psnr_iteration_budget = 16;
    let encoded = encode_j2k_lossy(samples, &psnr).unwrap();
    assert!(encoded.report.quantization_scale <= ceiling);
    assert!(encoded.report.quantization_scale > ceiling / 2.0);
}

#[test]
fn explicit_lossy_max_decomposition_levels_selects_that_depth() {
    // LRCP and RLCP default to one level; an explicit value selects that many,
    // capped by geometry, as the lossless option does.
    let pixels = (0..256 * 256)
        .map(|i| u8::try_from(i % 251).unwrap())
        .collect::<Vec<_>>();
    let samples = J2kLossySamples::new(&pixels, 256, 256, 1, 8, false).unwrap();
    for progression in [J2kProgressionOrder::Lrcp, J2kProgressionOrder::Rlcp] {
        let levels = |max| {
            let options = J2kLossyEncodeOptions::default()
                .with_cpu_only_backend()
                .with_progression(progression)
                .with_max_decomposition_levels(max);
            let encoded = encode_j2k_lossy(samples, &options).unwrap();
            j2k_native::inspect_j2k_codestream_header(&encoded.codestream)
                .unwrap()
                .resolution_levels
                - 1
        };
        for (max, expected) in [(None, 1), (Some(0), 0), (Some(5), 5), (Some(32), 8)] {
            assert_eq!(levels(max), expected, "{progression:?} max {max:?}");
        }
    }
}
