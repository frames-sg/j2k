// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::{vec, vec::Vec};

use super::super::{
    encode_lossy_to_byte_target, encode_lossy_to_psnr_target, J2kError, J2kLossyEncodeOptions,
    J2kLossySamples,
};
use super::single_pixel_samples;
use crate::{BackendErrorKind, J2kEncodeValidation};

#[test]
fn byte_target_accepts_first_under_limit_candidate_after_large_overshoot() {
    let options = J2kLossyEncodeOptions {
        psnr_iteration_budget: 1,
        ..J2kLossyEncodeOptions::default()
    };
    let mut scales = Vec::new();

    let attempt = encode_lossy_to_byte_target(single_pixel_samples(), &options, 1_000, |scale| {
        scales.push(scale);
        let len = if scale <= 1.0 { 1_600 } else { 300 };
        Ok(vec![0_u8; len])
    })
    .expect("an under-limit candidate is reachable");

    assert_eq!(attempt.quantization_scale.to_bits(), 2.0_f32.to_bits());
    assert_eq!(attempt.codestream.len(), 300);
    assert_eq!(scales, vec![1.0, 2.0, 1.5, 2.0]);
}

#[test]
fn byte_target_revalidates_the_final_stateful_encode() {
    let options = J2kLossyEncodeOptions {
        psnr_iteration_budget: 1,
        ..J2kLossyEncodeOptions::default()
    };
    let mut scale_two_calls = 0_u8;

    let result = encode_lossy_to_byte_target(single_pixel_samples(), &options, 1_000, |scale| {
        let len = if scale <= 1.0 {
            1_600
        } else if scale.to_bits() == 2.0_f32.to_bits() {
            scale_two_calls += 1;
            if scale_two_calls == 1 {
                300
            } else {
                2_000
            }
        } else {
            300
        };
        Ok(vec![0_u8; len])
    });

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

    let result = encode_lossy_to_psnr_target(samples, &options, 1.0, |scale| {
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
    let result = encode_lossy_to_psnr_target(samples, &options, 40.0, |scale| {
        Ok(if scale == 2.0 {
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
    let result = encode_lossy_to_byte_target(samples, &options, 1000, |scale| {
        assert!(scale <= ceiling);
        if scale == ceiling {
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
