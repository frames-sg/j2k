// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

#[cfg(target_os = "macos")]
#[test]
fn metal_forward_dwt53_handles_single_sample_edge_dimensions() {
    if !should_run_metal_runtime() {
        return;
    }

    for (width, height) in [(1, 8), (8, 1)] {
        let samples: Vec<f32> = (0..width * height)
            .map(|i| {
                f32::from(
                    u8::try_from((i * 11 + width * 3 + height * 5) & 0xFF)
                        .expect("masked sample fits in u8"),
                ) - 128.0
            })
            .collect();
        let mut accelerator = MetalEncodeStageAccelerator::default();

        let output = accelerator
            .encode_forward_dwt53(J2kForwardDwt53Job {
                samples: &samples,
                width,
                height,
                num_levels: 1,
            })
            .expect("metal DWT 5/3 stage")
            .expect("metal DWT 5/3 dispatch");

        assert_eq!(output.ll_width, width.div_ceil(2));
        assert_eq!(output.ll_height, height.div_ceil(2));
        assert_eq!(output.levels.len(), 1);
        assert_eq!(accelerator.forward_dwt53_attempts(), 1);
        assert_eq!(accelerator.forward_dwt53_dispatches(), 1);
    }
}

#[cfg(target_os = "macos")]
#[test]
fn metal_forward_dwt53_matches_reference_for_fractional_stage_samples() {
    fn assert_slice_near(actual: &[f32], expected: &[f32], label: &str) {
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

    let width = 8;
    let height = 8;
    let samples = (0..width * height)
        .map(|idx| f32::from(u16::try_from(idx).expect("test index fits u16")) * 0.5 - 15.25)
        .collect::<Vec<_>>();
    let expected =
        forward_dwt53_reference(&samples, width, height, 1).expect("native 5/3 reference DWT");
    let mut accelerator = MetalEncodeStageAccelerator::default();

    let actual = accelerator
        .encode_forward_dwt53(J2kForwardDwt53Job {
            samples: &samples,
            width,
            height,
            num_levels: 1,
        })
        .expect("metal DWT 5/3 stage")
        .expect("metal DWT 5/3 dispatch");

    assert_eq!(actual.ll_width, expected.ll_width);
    assert_eq!(actual.ll_height, expected.ll_height);
    assert_slice_near(&actual.ll, &expected.ll, "LL");
    assert_eq!(actual.levels.len(), expected.levels.len());
    for (index, (actual, expected)) in actual.levels.iter().zip(&expected.levels).enumerate() {
        assert_eq!(actual.width, expected.width, "level {index} width");
        assert_eq!(actual.height, expected.height, "level {index} height");
        assert_eq!(
            actual.low_width, expected.low_width,
            "level {index} low width"
        );
        assert_eq!(
            actual.low_height, expected.low_height,
            "level {index} low height"
        );
        assert_eq!(
            actual.high_width, expected.high_width,
            "level {index} high width"
        );
        assert_eq!(
            actual.high_height, expected.high_height,
            "level {index} high height"
        );
        assert_slice_near(&actual.hl, &expected.hl, "HL");
        assert_slice_near(&actual.lh, &expected.lh, "LH");
        assert_slice_near(&actual.hh, &expected.hh, "HH");
    }
}

#[cfg(target_os = "macos")]
fn native_lossy_dwt97_options(num_decomposition_levels: u8) -> EncodeOptions {
    EncodeOptions {
        num_decomposition_levels,
        reversible: false,
        guard_bits: 2,
        use_ht_block_coding: true,
        ..Default::default()
    }
}

#[cfg(target_os = "macos")]
fn assert_metal_dwt97_matches_native_encode(
    pixels: &[u8],
    width: u32,
    height: u32,
    num_decomposition_levels: u8,
) {
    let options = native_lossy_dwt97_options(num_decomposition_levels);
    let expected = j2k_native::encode(pixels, width, height, 1, 8, false, &options)
        .expect("native lossy DWT 9/7 encode");
    let mut accelerator = MetalEncodeStageAccelerator::for_forward_dwt97_encode();
    let actual = {
        j2k_native::encode_with_accelerator(
            pixels,
            width,
            height,
            1,
            8,
            false,
            &options,
            &mut accelerator,
        )
        .expect("Metal DWT 9/7 lossy encode")
    };

    assert_eq!(actual, expected);
    assert_eq!(accelerator.forward_dwt97_attempts(), 1);
    assert_eq!(accelerator.forward_dwt97_dispatches(), 1);
    let report = accelerator.dispatch_report();
    assert_eq!(report.forward_dwt97, 1);
    assert_eq!(report.forward_dwt53, 0);
}

#[cfg(target_os = "macos")]
#[test]
fn metal_forward_dwt97_single_level_matches_native_encode_output() {
    if !should_run_metal_runtime() {
        return;
    }

    let width = 17;
    let height = 15;
    let pixels = (0..width * height)
        .map(|idx| ((idx * 29 + idx / 5 + 17) & 0xFF) as u8)
        .collect::<Vec<_>>();

    assert_metal_dwt97_matches_native_encode(&pixels, width, height, 1);
}

#[cfg(target_os = "macos")]
#[test]
fn metal_forward_dwt97_multi_level_matches_native_encode_output() {
    if !should_run_metal_runtime() {
        return;
    }

    let width = 32;
    let height = 16;
    let pixels = (0..width * height)
        .map(|idx| ((idx * 43 + idx / 7 + 91) & 0xFF) as u8)
        .collect::<Vec<_>>();

    assert_metal_dwt97_matches_native_encode(&pixels, width, height, 3);
}
