// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

#[cfg(target_os = "macos")]
#[test]
fn metal_encode_deinterleave_invalid_component_count_errors_without_dispatch() {
    let pixels = [0_u8; 10];
    let mut accelerator = MetalEncodeStageAccelerator::default();

    let err = accelerator
        .encode_deinterleave(J2kDeinterleaveToF32Job {
            pixels: &pixels,
            num_pixels: 2,
            num_components: 5,
            bit_depth: 8,
            signed: false,
        })
        .unwrap_err();

    assert_eq!(
        err,
        j2k::J2kEncodeStageError::unsupported(
            "J2K Metal encode deinterleave supports 1-4 component samples",
        )
    );
    assert_eq!(accelerator.deinterleave_attempts(), 1);
    assert_eq!(accelerator.deinterleave_dispatches(), 0);
    assert_eq!(accelerator.dispatch_report().deinterleave, 0);
}

#[cfg(target_os = "macos")]
#[test]
fn metal_encode_deinterleave_compute_rejects_invalid_shape_structured() {
    let pixels = [0_u8; 10];

    let err = compute::encode_deinterleave_to_f32(J2kDeinterleaveToF32Job {
        pixels: &pixels,
        num_pixels: 2,
        num_components: 5,
        bit_depth: 8,
        signed: false,
    })
    .unwrap_err();

    assert!(matches!(
        err,
        crate::Error::UnsupportedMetalRequest {
            reason: "J2K Metal encode deinterleave supports 1-4 component samples"
        }
    ));
}

#[cfg(target_os = "macos")]
#[test]
fn metal_quantize_subband_kernel_matches_cpu_reference() {
    #[derive(Clone, Copy)]
    struct Case {
        name: &'static str,
        step_exponent: u16,
        step_mantissa: u16,
        range_bits: u8,
        reversible: bool,
    }

    if !should_run_metal_runtime() {
        return;
    }

    let coefficients = (0_u16..257)
        .map(|idx| {
            let centered = f32::from(idx) - 128.0;
            centered * 0.375 + f32::from(idx % 7) * 0.125 - if idx % 5 == 0 { 0.5 } else { 0.0 }
        })
        .collect::<Vec<_>>();

    for case in [
        Case {
            name: "reversible",
            step_exponent: 12,
            step_mantissa: 0,
            range_bits: 8,
            reversible: true,
        },
        Case {
            name: "irreversible_delta_1",
            step_exponent: 8,
            step_mantissa: 0,
            range_bits: 8,
            reversible: false,
        },
        Case {
            name: "irreversible_fractional_delta",
            step_exponent: 9,
            step_mantissa: 512,
            range_bits: 8,
            reversible: false,
        },
        Case {
            name: "irreversible_large_delta",
            step_exponent: 6,
            step_mantissa: 1024,
            range_bits: 10,
            reversible: false,
        },
    ] {
        let expected = quantize_reference(
            &coefficients,
            case.step_exponent,
            case.step_mantissa,
            case.range_bits,
            case.reversible,
        );
        let actual = compute::encode_quantize_subband(J2kQuantizeSubbandJob {
            coefficients: &coefficients,
            step_exponent: case.step_exponent,
            step_mantissa: case.step_mantissa,
            range_bits: case.range_bits,
            reversible: case.reversible,
        })
        .unwrap_or_else(|err| panic!("Metal quantize_subband failed for {}: {err}", case.name));

        assert_eq!(actual, expected, "{}", case.name);
    }
}

#[cfg(target_os = "macos")]
#[test]
fn metal_quantize_subband_stage_reports_dispatch() {
    if !should_run_metal_runtime() {
        return;
    }

    let coefficients = [-4.5, -1.25, -0.5, 0.0, 0.5, 1.25, 4.5];
    let expected = quantize_reference(&coefficients, 8, 0, 8, true);
    let mut accelerator = MetalEncodeStageAccelerator::default();

    let actual = accelerator
        .encode_quantize_subband(J2kQuantizeSubbandJob {
            coefficients: &coefficients,
            step_exponent: 8,
            step_mantissa: 0,
            range_bits: 8,
            reversible: true,
        })
        .expect("Metal quantize_subband stage should not error")
        .expect("Metal quantize_subband should dispatch");

    assert_eq!(actual, expected);
    assert_eq!(accelerator.quantize_subband_attempts(), 1);
    assert_eq!(accelerator.quantize_subband_dispatches(), 1);
    assert_eq!(accelerator.dispatch_report().quantize_subband, 1);
}

#[cfg(target_os = "macos")]
#[test]
fn metal_quantize_subband_invalid_shape_errors_without_dispatch() {
    let coefficients = [1.0, -2.0, 3.0];
    let mut accelerator = MetalEncodeStageAccelerator::default();

    let err = accelerator
        .encode_quantize_subband(J2kQuantizeSubbandJob {
            coefficients: &coefficients,
            step_exponent: 8,
            step_mantissa: 2048,
            range_bits: 8,
            reversible: false,
        })
        .unwrap_err();

    assert_eq!(
        err,
        j2k::J2kEncodeStageError::unsupported(
            "J2K Metal encode quantize_subband supports step mantissas <= 2047",
        )
    );
    assert_eq!(accelerator.quantize_subband_attempts(), 1);
    assert_eq!(accelerator.quantize_subband_dispatches(), 0);
    assert_eq!(accelerator.dispatch_report().quantize_subband, 0);
}

#[cfg(target_os = "macos")]
#[test]
fn metal_quantize_subband_compute_rejects_invalid_shape_structured() {
    let coefficients = [1.0, -2.0, 3.0];
    let err = compute::encode_quantize_subband(J2kQuantizeSubbandJob {
        coefficients: &coefficients,
        step_exponent: 8,
        step_mantissa: 2048,
        range_bits: 8,
        reversible: false,
    })
    .unwrap_err();

    assert!(matches!(
        err,
        crate::Error::UnsupportedMetalRequest {
            reason: "J2K Metal encode quantize_subband supports step mantissas <= 2047"
        }
    ));
}

#[test]
fn auto_encode_deinterleave_disabled_fallback_still_encodes() {
    let mut pixels = Vec::with_capacity(8 * 8 * 2);
    for idx in 0..8 * 8 {
        let sample = u16::try_from((idx * 257 + 19) & 0xffff).expect("masked sample fits u16");
        pixels.extend_from_slice(&sample.to_le_bytes());
    }
    let samples =
        J2kLosslessSamples::new(&pixels, 8, 8, 1, 16, false).expect("valid Gray16 samples");
    let options = J2kLosslessEncodeOptions::default().with_max_decomposition_levels(Some(0));
    let mut accelerator = MetalEncodeStageAccelerator::for_auto_host_output();

    let encoded = encode_j2k_lossless_with_accelerator(
        samples,
        &options,
        j2k_core::BackendKind::Metal,
        &mut accelerator,
    )
    .expect("Gray16 encode should succeed with CPU deinterleave fallback");
    let decoded = Image::new(&encoded.codestream, &DecodeSettings::default())
        .expect("codestream parses")
        .decode_native()
        .expect("codestream decodes");

    assert_eq!(decoded.data, pixels);
    assert_eq!(encoded.dispatch_report.deinterleave, 0);
    assert_eq!(accelerator.deinterleave_attempts(), 1);
    assert_eq!(accelerator.deinterleave_dispatches(), 0);
}

#[cfg(target_os = "macos")]
#[test]
fn metal_dispatch_option_treats_unavailable_as_no_dispatch() {
    let result: j2k::J2kEncodeStageResult<Option<u8>> =
        super::super::metal_dispatch_option(Err(crate::Error::MetalUnavailable), "kernel failed");

    assert!(matches!(result, Ok(None)));
}

#[cfg(target_os = "macos")]
#[test]
fn metal_dispatch_option_preserves_kernel_errors() {
    let result: j2k::J2kEncodeStageResult<Option<u8>> = super::super::metal_dispatch_option(
        Err(crate::Error::MetalKernel {
            message: "bad status".to_string(),
        }),
        "kernel failed",
    );

    let error = result.expect_err("kernel failure must not be downgraded");
    assert_eq!(error.kind(), j2k::J2kEncodeStageErrorKind::Backend);
    assert_eq!(error.reason(), "kernel failed");
    assert!(matches!(
        error,
        j2k::J2kEncodeStageError::Backend { source, .. }
            if source.to_string().contains("bad status")
    ));
}

#[test]
fn metal_encode_stage_accelerator_preserves_cpu_codestream_validity() {
    #[cfg(target_os = "macos")]
    if !should_run_metal_runtime() {
        return;
    }

    let pixels: Vec<u8> = (0..8 * 8 * 3)
        .map(|i| u8::try_from(i & 0xFF).expect("masked pixel fits u8"))
        .collect();
    let samples = J2kLosslessSamples::new(&pixels, 8, 8, 3, 8, false).expect("valid RGB samples");
    let options = J2kLosslessEncodeOptions::default()
        .with_backend(EncodeBackendPreference::Auto)
        .with_max_decomposition_levels(Some(1));
    let mut accelerator = MetalEncodeStageAccelerator::default();

    let encoded = encode_j2k_lossless_with_accelerator(
        samples,
        &options,
        j2k_core::BackendKind::Metal,
        &mut accelerator,
    )
    .expect("encode with metal stage accelerator");
    let decoded = Image::new(&encoded.codestream, &DecodeSettings::default())
        .expect("codestream parses")
        .decode_native()
        .expect("codestream decodes");

    assert_eq!(decoded.width, 8);
    assert_eq!(decoded.height, 8);
    assert_eq!(decoded.num_components, 3);
    assert_eq!(decoded.bit_depth, 8);
    assert_eq!(accelerator.forward_rct_attempts(), 1);
    assert_eq!(accelerator.forward_dwt53_attempts(), 3);
    assert!(accelerator.tier1_code_block_attempts() > 0);
    assert_eq!(accelerator.packetization_attempts(), 1);
}

#[test]
fn metal_encode_stage_accelerator_can_leave_forward_rct_on_cpu() {
    let mut plane0 = vec![0.0, 64.0, 128.0, 255.0];
    let mut plane1 = vec![3.0, 67.0, 131.0, 252.0];
    let mut plane2 = vec![7.0, 71.0, 135.0, 248.0];
    let original = (plane0.clone(), plane1.clone(), plane2.clone());
    let mut accelerator = MetalEncodeStageAccelerator::with_cpu_forward_rct();

    let dispatched = accelerator
        .encode_forward_rct(J2kForwardRctJob {
            plane0: &mut plane0,
            plane1: &mut plane1,
            plane2: &mut plane2,
        })
        .expect("CPU RCT fallback should be selectable");

    assert!(!dispatched);
    assert_eq!(accelerator.forward_rct_attempts(), 1);
    assert_eq!(accelerator.forward_rct_dispatches(), 0);
    assert_eq!((plane0, plane1, plane2), original);
}

#[cfg(target_os = "macos")]
#[test]
fn metal_forward_ict_dispatch_matches_cpu_reference() {
    fn forward_ict_reference(
        plane0: &[f32],
        plane1: &[f32],
        plane2: &[f32],
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut out0 = Vec::with_capacity(plane0.len());
        let mut out1 = Vec::with_capacity(plane1.len());
        let mut out2 = Vec::with_capacity(plane2.len());
        for ((&r, &g), &b) in plane0.iter().zip(plane1).zip(plane2) {
            out0.push(0.299 * r + 0.587 * g + 0.114 * b);
            out1.push(-0.16875 * r - 0.33126 * g + 0.5 * b);
            out2.push(0.5 * r - 0.41869 * g - 0.08131 * b);
        }
        (out0, out1, out2)
    }

    fn assert_near(actual: &[f32], expected: &[f32], label: &str) {
        assert_eq!(actual.len(), expected.len(), "{label} length mismatch");
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= 0.0001,
                "{label}[{index}] mismatch: actual={actual}, expected={expected}"
            );
        }
    }

    if !should_run_metal_runtime() {
        return;
    }

    let mut plane0 = vec![0.0, 64.0, 128.0, 255.0, -12.5, 42.25];
    let mut plane1 = vec![3.0, 67.0, 131.0, 252.0, 19.75, -8.5];
    let mut plane2 = vec![7.0, 71.0, 135.0, 248.0, 33.5, 128.0];
    let expected = forward_ict_reference(&plane0, &plane1, &plane2);
    let mut accelerator = MetalEncodeStageAccelerator::default();

    let dispatched = accelerator
        .encode_forward_ict(J2kForwardIctJob {
            plane0: &mut plane0,
            plane1: &mut plane1,
            plane2: &mut plane2,
        })
        .expect("Metal ICT dispatch");

    assert!(dispatched);
    assert_near(&plane0, &expected.0, "Y");
    assert_near(&plane1, &expected.1, "Cb");
    assert_near(&plane2, &expected.2, "Cr");
    assert_eq!(accelerator.forward_ict_attempts(), 1);
    assert_eq!(accelerator.forward_ict_dispatches(), 1);
    let report = accelerator.dispatch_report();
    assert_eq!(report.forward_ict, 1);
    assert_eq!(report.forward_rct, 0);
}

#[cfg(target_os = "macos")]
#[test]
fn metal_forward_ict_invalid_plane_lengths_error_without_dispatch() {
    let mut plane0 = vec![0.0, 64.0];
    let mut plane1 = vec![3.0];
    let mut plane2 = vec![7.0, 71.0];
    let original = (plane0.clone(), plane1.clone(), plane2.clone());
    let mut accelerator = MetalEncodeStageAccelerator::default();

    let err = accelerator
        .encode_forward_ict(J2kForwardIctJob {
            plane0: &mut plane0,
            plane1: &mut plane1,
            plane2: &mut plane2,
        })
        .unwrap_err();

    assert_eq!(
        err,
        j2k::J2kEncodeStageError::unsupported("J2K Metal forward ICT plane lengths must match")
    );
    assert_eq!(accelerator.forward_ict_attempts(), 1);
    assert_eq!(accelerator.forward_ict_dispatches(), 0);
    assert_eq!(accelerator.dispatch_report().forward_ict, 0);
    assert_eq!((plane0, plane1, plane2), original);
}

#[cfg(target_os = "macos")]
#[test]
fn metal_forward_ict_compute_rejects_invalid_shape_structured() {
    let mut plane0 = vec![0.0, 64.0];
    let mut plane1 = vec![3.0];
    let mut plane2 = vec![7.0, 71.0];

    let err = compute::encode_forward_ict(&mut plane0, &mut plane1, &mut plane2).unwrap_err();

    assert!(matches!(
        err,
        crate::Error::UnsupportedMetalRequest {
            reason: "J2K Metal forward ICT plane lengths must match"
        }
    ));
}

#[cfg(target_os = "macos")]
#[test]
#[expect(
    clippy::cast_sign_loss,
    reason = "bounded synthetic sample expression is nonnegative"
)]
fn metal_forward_rct_dispatch_round_trips_rgb8_lossless_tile() {
    if !should_run_metal_runtime() {
        return;
    }

    let pixels: Vec<u8> = (0..7 * 5 * 3).map(|i| ((i * 17) & 0xFF) as u8).collect();
    let samples = J2kLosslessSamples::new(&pixels, 7, 5, 3, 8, false).expect("valid RGB samples");
    let options = J2kLosslessEncodeOptions::default()
        .with_backend(EncodeBackendPreference::RequireDevice)
        .with_max_decomposition_levels(Some(0));
    let mut accelerator = MetalEncodeStageAccelerator::default();

    let encoded = encode_j2k_lossless_with_accelerator(
        samples,
        &options,
        BackendKind::Metal,
        &mut accelerator,
    )
    .expect("encode with metal forward RCT");
    let decoded = Image::new(&encoded.codestream, &DecodeSettings::default())
        .expect("codestream parses")
        .decode_native()
        .expect("codestream decodes");

    assert_eq!(decoded.data, pixels);
    assert_eq!(encoded.dispatch_report.deinterleave, 1);
    assert_eq!(accelerator.deinterleave_attempts(), 1);
    assert_eq!(accelerator.deinterleave_dispatches(), 1);
    assert_eq!(accelerator.forward_rct_attempts(), 1);
    assert_eq!(accelerator.forward_rct_dispatches(), 1);
}

#[cfg(target_os = "macos")]
#[test]
#[expect(
    clippy::cast_sign_loss,
    reason = "bounded synthetic byte expression is nonnegative"
)]
fn metal_deinterleave_gray16_lossless_facade_dispatches_and_round_trips() {
    if !should_run_metal_runtime() {
        return;
    }

    let mut pixels = Vec::with_capacity(8 * 8 * 2);
    for idx in 0..8 * 8 {
        let sample = ((idx * 1021 + 0x2345) & 0xffff) as u16;
        pixels.extend_from_slice(&sample.to_le_bytes());
    }
    let samples =
        J2kLosslessSamples::new(&pixels, 8, 8, 1, 16, false).expect("valid Gray16 samples");
    let mut accelerator = MetalEncodeStageAccelerator::default();

    let encoded = encode_j2k_lossless_with_accelerator(
        samples,
        &lossless_options! {
            backend: EncodeBackendPreference::RequireDevice,
            max_decomposition_levels: Some(0),
        },
        BackendKind::Metal,
        &mut accelerator,
    )
    .expect("Metal-accelerated Gray16 lossless encode");
    let decoded = Image::new(&encoded.codestream, &DecodeSettings::default())
        .expect("codestream parses")
        .decode_native()
        .expect("codestream decodes");

    assert_eq!(decoded.data, pixels);
    assert_eq!(encoded.dispatch_report.deinterleave, 1);
    assert_eq!(accelerator.deinterleave_attempts(), 1);
    assert_eq!(accelerator.deinterleave_dispatches(), 1);
}

#[cfg(target_os = "macos")]
#[test]
#[expect(
    clippy::cast_sign_loss,
    reason = "bounded synthetic pixel expression is nonnegative"
)]
fn metal_validation_decodes_and_compares_lossless_codestream_on_device() {
    if !should_run_metal_runtime() {
        return;
    }

    let pixels: Vec<u8> = (0..16 * 16 * 3).map(|i| ((i * 29) & 0xFF) as u8).collect();
    let samples = J2kLosslessSamples::new(&pixels, 16, 16, 3, 8, false).unwrap();
    let encoded = j2k::encode_j2k_lossless(
        samples,
        &lossless_options! {
            backend: EncodeBackendPreference::CpuOnly,
        },
    )
    .expect("lossless encode");

    super::super::validate_lossless_roundtrip_on_metal(samples, &encoded.codestream)
        .expect("Metal lossless validation");
}
