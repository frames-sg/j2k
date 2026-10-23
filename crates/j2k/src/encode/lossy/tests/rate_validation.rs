// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::{vec, vec::Vec};

use super::super::candidates::LossyCandidates;
use super::super::{
    encode_lossy_to_byte_target, encode_lossy_to_psnr_target, J2kError, J2kLossyEncodeOptions,
    J2kLossySamples,
};
use super::single_pixel_samples;
use crate::{BackendErrorKind, J2kEncodeValidation};

/// A deterministic synthetic size curve that, like the CPU candidates, lets the
/// byte search return the probe it selects instead of encoding it again, so
/// these tests count the search's probes alone.
struct DeterministicCurve<F>(F);

impl<F: FnMut(f32) -> Result<Vec<u8>, J2kError>> LossyCandidates for DeterministicCurve<F> {
    fn reuse_terminal_byte_output(&self) -> bool {
        true
    }

    fn encode_candidate(&mut self, quantization_scale: f32) -> Result<Vec<u8>, J2kError> {
        (self.0)(quantization_scale)
    }
}

#[test]
fn byte_target_accepts_first_under_limit_candidate_after_large_overshoot() {
    let options = J2kLossyEncodeOptions {
        psnr_iteration_budget: 1,
        ..J2kLossyEncodeOptions::default()
    };
    let mut scales = Vec::new();

    let attempt = encode_lossy_to_byte_target(
        single_pixel_samples(),
        &options,
        1_000,
        &mut |scale: f32| {
            scales.push(scale);
            let len = if scale <= 1.0 { 1_600 } else { 300 };
            Ok(vec![0_u8; len])
        },
    )
    .expect("an under-limit candidate is reachable");

    // The budget-exhausted search re-encodes its closest fitting scale.
    assert!(attempt.quantization_scale > 1.0);
    assert_eq!(attempt.codestream.len(), 300);
    assert_eq!(scales.last(), Some(&attempt.quantization_scale));
}

#[test]
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "synthetic positive byte counts"
)]
fn byte_target_lands_within_the_allowance_in_a_few_encodes() {
    let options = J2kLossyEncodeOptions::default();
    // Log-linear size curves, the third with a 1.6x drop past scale 18 (a band
    // falling below its step); bisection needs 13-15 encodes on each. On the
    // last, a small target, the second probe is within the 512-byte allowance
    // but more than 1% short.
    for (bytes_at_one, slope, drop_above, target) in [
        (942_572.0, -0.75, f32::INFINITY, 224_420_u64),
        (500_000.0, -1.2, f32::INFINITY, 12_288),
        (300_000.0, -0.6, 18.0, 65_536),
        (3_600.0, -0.75, f32::INFINITY, 2_963),
    ] {
        let mut scales = Vec::new();
        let attempt = encode_lossy_to_byte_target(
            single_pixel_samples(),
            &options,
            target,
            &mut DeterministicCurve(|scale: f32| {
                scales.push(scale);
                let drop = if scale > drop_above { 1.6 } else { 1.0 };
                Ok(vec![
                    0_u8;
                    (bytes_at_one * f64::from(scale).powf(slope) / drop)
                        as usize
                ])
            }),
        )
        .expect("reachable target");

        let len = attempt.codestream.len() as u64;
        assert!(
            len.abs_diff(target) <= target.div_ceil(100),
            "{target}: {len}"
        );
        assert!(scales.len() <= 4, "{target}: {scales:?}");
        assert_eq!(scales.last(), Some(&attempt.quantization_scale));
    }
}

#[test]
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "synthetic positive byte counts"
)]
fn byte_target_search_starts_above_the_floor_and_moves_both_ways() {
    // At 2.14 bits per pixel of a 4096x2048 RGB image the first probe is well
    // above the finest scale. The first curve is too small there, the second
    // too large, and the third fits even at the finest scale.
    let samples = J2kLossySamples {
        data: &[],
        width: 4096,
        height: 2048,
        components: 3,
        bit_depth: 8,
        signed: false,
    };
    let options = J2kLossyEncodeOptions::default();
    let (floor, _) = super::super::quantization_scale_bounds(samples, &options).unwrap();
    let target = 2_243_952_u64;
    for (bytes_at_one, slope, finest_fits) in [
        (9_425_726.0, -0.75, false),
        (40_000_000.0, -0.9, false),
        (2_000_000.0, -0.8, true),
    ] {
        let mut scales = Vec::new();
        let mut curve = DeterministicCurve(|scale: f32| {
            scales.push(scale);
            Ok(vec![
                0_u8;
                (bytes_at_one * f64::from(scale).powf(slope)) as usize
            ])
        });
        let attempt = encode_lossy_to_byte_target(samples, &options, target, &mut curve)
            .expect("reachable target");

        assert!(scales[0] > floor, "{bytes_at_one}: {scales:?}");
        if finest_fits {
            assert_eq!(attempt.quantization_scale, floor);
        } else {
            let len = attempt.codestream.len() as u64;
            assert!(len.abs_diff(target) <= target.div_ceil(100), "{len}");
            assert!(scales.len() <= 4, "{bytes_at_one}: {scales:?}");
        }
        assert_eq!(scales.last(), Some(&attempt.quantization_scale));
    }
}

#[test]
fn byte_target_revalidates_the_final_stateful_encode() {
    let options = J2kLossyEncodeOptions {
        psnr_iteration_budget: 1,
        ..J2kLossyEncodeOptions::default()
    };
    let mut fitted = Vec::new();

    let result = encode_lossy_to_byte_target(
        single_pixel_samples(),
        &options,
        1_000,
        &mut |scale: f32| {
            // A stateful encoder: every scale fits once, then grows.
            let len = if scale <= 1.0 {
                1_600
            } else if fitted.contains(&scale.to_bits()) {
                2_000
            } else {
                fitted.push(scale.to_bits());
                300
            };
            Ok(vec![0_u8; len])
        },
    );

    assert!(matches!(
        result,
        Err(J2kError::RateTargetUnreachable { target, best })
            if target == "1000 bytes" && best == "2000 bytes"
    ));
}

#[test]
fn external_psnr_target_rejects_final_wrong_signedness_metadata() {
    let pixels = vec![7_u8; 64];
    let samples = J2kLossySamples::new(&pixels, 8, 8, 1, 8, false).expect("fixture samples");
    let source = j2k_native::encode(
        &pixels,
        8,
        8,
        1,
        8,
        false,
        &j2k_native::EncodeOptions {
            reversible: false,
            num_decomposition_levels: 0,
            ..j2k_native::EncodeOptions::default()
        },
    )
    .expect("fixture codestream");
    let options = J2kLossyEncodeOptions {
        validation: J2kEncodeValidation::External,
        psnr_iteration_budget: 1,
        ..J2kLossyEncodeOptions::default()
    };
    let mut seen_scales = Vec::new();
    let mut final_encode_seen = false;

    let result = encode_lossy_to_psnr_target(samples, &options, 1.0, &mut |scale: f32| {
        let repeated = seen_scales.contains(&scale.to_bits());
        seen_scales.push(scale.to_bits());
        let mut codestream = source.clone();
        if repeated {
            final_encode_seen = true;
            let siz = codestream
                .windows(2)
                .position(|window| window == [0xff, 0x51])
                .expect("SIZ marker");
            codestream[siz + 40] |= 0x80;
        }
        Ok(codestream)
    });

    assert!(
        final_encode_seen,
        "fixture must reach the selected final re-encode"
    );
    assert!(matches!(
        result,
        Err(J2kError::Backend(error))
            if error.kind() == BackendErrorKind::Validation
                && error.message() == "JPEG 2000 PSNR validation metadata mismatch"
    ));
}

#[test]
fn psnr_search_keeps_a_feasible_candidate_after_initial_quality_dip() {
    let pixels = [128_u8; 64];
    let samples = J2kLossySamples::new(&pixels, 8, 8, 1, 8, false).unwrap();
    let options = J2kLossyEncodeOptions {
        psnr_iteration_budget: 2,
        ..Default::default()
    };
    let encode = |value| {
        j2k_native::encode(
            &[value; 64],
            8,
            8,
            1,
            8,
            false,
            &j2k_native::EncodeOptions {
                num_decomposition_levels: 0,
                ..Default::default()
            },
        )
        .unwrap()
    };
    let low_quality = encode(0);
    let feasible = encode(128);
    let result = encode_lossy_to_psnr_target(samples, &options, 40.0, &mut |scale: f32| {
        Ok(if scale.to_bits() == 2.0_f32.to_bits() {
            feasible.clone()
        } else {
            low_quality.clone()
        })
    })
    .expect("the measured scale-two candidate meets the minimum PSNR");
    assert_eq!(result.codestream, feasible);
}

#[test]
fn byte_search_evaluates_a_profile_boundary_between_doubling_steps() {
    let options =
        J2kLossyEncodeOptions::default().with_quantization_profile(Some(vec![vec![1500.0]]));
    let samples = single_pixel_samples();
    let (_, ceiling) = super::super::quantization_scale_bounds(samples, &options).unwrap();
    assert!(ceiling > 1.0 && ceiling < 2.0);
    let mut boundary_seen = false;
    let result = encode_lossy_to_byte_target(samples, &options, 1000, &mut |scale: f32| {
        assert!(scale <= ceiling);
        if scale.to_bits() == ceiling.to_bits() {
            boundary_seen = true;
            Ok(vec![0; 300])
        } else {
            Ok(vec![0; 1600])
        }
    })
    .unwrap();
    assert!(boundary_seen);
    assert_eq!(result.codestream.len(), 300);
    assert_eq!(result.quantization_scale, ceiling);
}
