// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec;
use alloc::vec::Vec;

use super::super::codestream::WaveletTransform;
use super::super::rect::IntRect;
use super::direct::apply_single_decomposition_idwt_job;
use super::horizontal::{filter_horizontal, filter_horizontal_i64};
use super::model::CoefficientSource;
use super::roi::interleave_samples_roi;
use super::vertical::{filter_vertical, filter_vertical_i64};
use crate::error::{DecodeError, DecodingError};
use crate::{J2kIdwtBand, J2kRect, J2kSingleDecompositionIdwtJob, J2kWaveletTransform, Result};

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

fn direct_job<'a>(
    rect: J2kRect,
    transform: J2kWaveletTransform,
    ll: J2kIdwtBand<'a>,
    hl: J2kIdwtBand<'a>,
    lh: J2kIdwtBand<'a>,
    hh: J2kIdwtBand<'a>,
) -> J2kSingleDecompositionIdwtJob<'a> {
    J2kSingleDecompositionIdwtJob {
        rect,
        transform,
        ll,
        hl,
        lh,
        hh,
    }
}

fn band<'a>(rect: J2kRect, coefficients: &'a [f32]) -> J2kIdwtBand<'a> {
    J2kIdwtBand { rect, coefficients }
}

#[test]
fn reversible_53_even_direct_job_is_bit_exact() {
    const EXPECTED: [u32; 16] = [
        3_225_419_776,
        3_231_711_232,
        3_225_419_776,
        3_229_614_080,
        3_212_836_864,
        1_093_664_768,
        0,
        1_096_810_496,
        3_221_225_472,
        3_221_225_472,
        3_221_225_472,
        3_212_836_864,
        1_065_353_216,
        1_098_907_648,
        1_073_741_824,
        1_100_480_512,
    ];
    let rect = J2kRect {
        x0: 0,
        y0: 0,
        x1: 4,
        y1: 4,
    };
    let band_rect = J2kRect {
        x0: 0,
        y0: 0,
        x1: 2,
        y1: 2,
    };
    let ll = [1.0, 2.0, 3.0, 4.0];
    let hl = [5.0, 6.0, 7.0, 8.0];
    let lh = [9.0, 10.0, 11.0, 12.0];
    let hh = [13.0, 14.0, 15.0, 16.0];
    let mut output = Vec::with_capacity(32);
    let capacity = output.capacity();

    apply_single_decomposition_idwt_job(
        direct_job(
            rect,
            J2kWaveletTransform::Reversible53,
            band(band_rect, &ll),
            band(band_rect, &hl),
            band(band_rect, &lh),
            band(band_rect, &hh),
        ),
        &mut output,
    )
    .expect("valid reversible direct IDWT job");

    assert_eq!(bits(&output).as_slice(), &EXPECTED);
    assert_eq!(
        output.capacity(),
        capacity,
        "target capacity must be reused"
    );
}

#[test]
fn irreversible_97_odd_direct_job_is_bit_exact() {
    const EXPECTED: [u32; 15] = [
        3_243_516_307,
        1_088_535_889,
        1_086_781_832,
        3_237_068_116,
        1_072_899_983,
        1_090_811_482,
        3_205_449_560,
        3_240_528_043,
        1_057_965_919,
        1_095_801_595,
        3_240_069_383,
        1_090_703_406,
        1_072_530_023,
        3_238_711_343,
        1_091_758_981,
    ];
    let rect = J2kRect {
        x0: 0,
        y0: 0,
        x1: 5,
        y1: 3,
    };
    let ll_rect = J2kRect {
        x0: 0,
        y0: 0,
        x1: 3,
        y1: 2,
    };
    let hl_rect = J2kRect {
        x0: 0,
        y0: 0,
        x1: 2,
        y1: 2,
    };
    let lh_rect = J2kRect {
        x0: 0,
        y0: 0,
        x1: 3,
        y1: 1,
    };
    let hh_rect = J2kRect {
        x0: 0,
        y0: 0,
        x1: 2,
        y1: 1,
    };
    let ll = [0.5, -1.25, 2.0, 3.5, -4.25, 5.75];
    let hl = [6.5, -7.0, 8.25, -9.5];
    let lh = [10.0, -11.5, 12.75];
    let hh = [-13.0, 14.25];
    let mut output = Vec::new();

    apply_single_decomposition_idwt_job(
        direct_job(
            rect,
            J2kWaveletTransform::Irreversible97,
            band(ll_rect, &ll),
            band(hl_rect, &hl),
            band(lh_rect, &lh),
            band(hh_rect, &hh),
        ),
        &mut output,
    )
    .expect("valid irreversible direct IDWT job");

    assert_eq!(bits(&output).as_slice(), &EXPECTED);
}

#[test]
fn reversible_53_two_levels_are_bit_exact() {
    const EXPECTED: [u32; 16] = [
        3_242_196_992,
        1_073_741_824,
        3_233_808_384,
        3_247_964_160,
        1_088_421_888,
        1_073_741_824,
        1_086_324_736,
        1_092_616_192,
        3_225_419_776,
        1_082_130_432,
        3_221_225_472,
        3_239_051_264,
        3_249_537_024,
        1_097_859_072,
        1_092_616_192,
        3_243_245_568,
    ];
    let unit_rect = J2kRect {
        x0: 0,
        y0: 0,
        x1: 1,
        y1: 1,
    };
    let two_rect = J2kRect {
        x0: 0,
        y0: 0,
        x1: 2,
        y1: 2,
    };
    let four_rect = J2kRect {
        x0: 0,
        y0: 0,
        x1: 4,
        y1: 4,
    };
    let mut level_one = Vec::new();
    apply_single_decomposition_idwt_job(
        direct_job(
            two_rect,
            J2kWaveletTransform::Reversible53,
            band(unit_rect, &[1.0]),
            band(unit_rect, &[2.0]),
            band(unit_rect, &[3.0]),
            band(unit_rect, &[4.0]),
        ),
        &mut level_one,
    )
    .expect("valid first IDWT level");

    let high_a = [5.0, -6.0, 7.0, -8.0];
    let high_b = [9.0, 10.0, -11.0, 12.0];
    let high_c = [-13.0, 14.0, 15.0, -16.0];
    let mut output = Vec::new();
    apply_single_decomposition_idwt_job(
        direct_job(
            four_rect,
            J2kWaveletTransform::Reversible53,
            band(two_rect, &level_one),
            band(two_rect, &high_a),
            band(two_rect, &high_b),
            band(two_rect, &high_c),
        ),
        &mut output,
    )
    .expect("valid second IDWT level");

    assert_eq!(bits(&output).as_slice(), &EXPECTED);
}

#[test]
fn reversible_i64_filters_preserve_even_and_odd_goldens() {
    const EVEN_EXPECTED: [i64; 16] = [-1, -2, -1, -1, 2, 10, 3, 14, 2, 7, 2, 8, 8, 27, 9, 31];
    const ODD_EXPECTED: [i64; 15] = [
        2_473_901_162_495,
        824_633_720_829,
        2_473_901_162_495,
        824_633_720_831,
        2_473_901_162_496,
        824_633_720_835,
        274_877_906_958,
        824_633_720_837,
        274_877_906_963,
        824_633_720_839,
        2_473_901_162_500,
        824_633_720_844,
        2_473_901_162_500,
        824_633_720_846,
        2_473_901_162_501,
    ];
    let mut even: Vec<i64> = (1..=16).collect();
    let even_rect = IntRect::from_xywh(0, 0, 4, 4);
    filter_horizontal_i64(&mut even, even_rect);
    filter_vertical_i64(&mut even, even_rect);
    assert_eq!(even.as_slice(), &EVEN_EXPECTED);

    let base = 1_i64 << 40;
    let mut odd = vec![
        base + 1,
        -base + 2,
        base + 3,
        -base + 4,
        base + 5,
        -base + 6,
        base + 7,
        -base + 8,
        base + 9,
        -base + 10,
        base + 11,
        -base + 12,
        base + 13,
        -base + 14,
        base + 15,
    ];
    let odd_rect = IntRect::from_xywh(0, 0, 5, 3);
    filter_horizontal_i64(&mut odd, odd_rect);
    filter_vertical_i64(&mut odd, odd_rect);
    assert_eq!(odd.as_slice(), &ODD_EXPECTED);
}

#[test]
fn irreversible_97_nonzero_origin_roi_window_is_bit_exact() {
    const EXPECTED: [u32; 12] = [
        1_098_644_013,
        1_084_670_383,
        3_239_209_924,
        1_072_899_983,
        3_241_441_274,
        3_239_628_159,
        1_073_078_266,
        1_095_801_595,
        1_094_877_885,
        1_066_907_503,
        3_239_826_101,
        1_091_758_981,
    ];
    let ll = [0.5, -1.25, 2.0, 3.5, -4.25, 5.75];
    let hl = [6.5, -7.0, 8.25, -9.5];
    let lh = [10.0, -11.5, 12.75];
    let hh = [-13.0, 14.25];
    let output_window = IntRect::from_ltrb(1, 0, 5, 3);
    let decomposition_rect = IntRect::from_ltrb(0, 0, 5, 3);
    let mut output = vec![0.0; 12];

    interleave_samples_roi(
        CoefficientSource::new(&ll, IntRect::from_ltrb(0, 0, 3, 2), 3),
        CoefficientSource::new(&hl, IntRect::from_ltrb(0, 0, 2, 2), 2),
        CoefficientSource::new(&lh, IntRect::from_ltrb(0, 0, 3, 1), 3),
        CoefficientSource::new(&hh, IntRect::from_ltrb(0, 0, 2, 1), 2),
        &mut output,
        output_window,
        decomposition_rect,
    );
    filter_horizontal(&mut output, output_window, WaveletTransform::Irreversible97);
    filter_vertical(&mut output, output_window, WaveletTransform::Irreversible97);

    assert_eq!(bits(&output).as_slice(), &EXPECTED);
}

#[test]
fn invalid_direct_band_preserves_target_and_error_class() {
    let rect = J2kRect {
        x0: 0,
        y0: 0,
        x1: 2,
        y1: 2,
    };
    let band_rect = J2kRect {
        x0: 0,
        y0: 0,
        x1: 1,
        y1: 1,
    };
    let empty = [];
    let valid = [1.0];
    let mut target = vec![42.0, -7.0];

    let result = apply_single_decomposition_idwt_job(
        direct_job(
            rect,
            J2kWaveletTransform::Reversible53,
            band(band_rect, &empty),
            band(band_rect, &valid),
            band(band_rect, &valid),
            band(band_rect, &valid),
        ),
        &mut target,
    );

    assert!(matches!(
        result,
        Err(DecodeError::Decoding(DecodingError::CodeBlockDecodeFailure))
    ));
    assert_eq!(target, [42.0, -7.0]);
}

#[test]
fn public_idwt_function_signatures_stay_stable() {
    let _: fn(J2kSingleDecompositionIdwtJob<'_>, &mut Vec<f32>) -> Result<()> =
        apply_single_decomposition_idwt_job;
}

#[test]
fn idwt_module_boundaries_stay_focused() {
    const MODULES: &[(&str, &str, usize)] = &[
        ("coordinator", include_str!("../idwt.rs"), 40),
        ("direct", include_str!("direct.rs"), 220),
        ("filter common", include_str!("filter_common.rs"), 70),
        ("horizontal", include_str!("horizontal.rs"), 290),
        ("f32 interleave", include_str!("interleave.rs"), 210),
        ("i64 interleave", include_str!("interleave_i64.rs"), 180),
        ("model", include_str!("model.rs"), 170),
        ("orchestration", include_str!("orchestrate.rs"), 340),
        ("ROI", include_str!("roi.rs"), 240),
        ("vertical", include_str!("vertical.rs"), 390),
    ];

    for &(name, source, line_limit) in MODULES {
        let line_count = source.lines().count();
        assert!(
            line_count <= line_limit,
            "IDWT {name} module grew to {line_count} lines (limit {line_limit})"
        );
        assert!(
            !source.contains("include!("),
            "IDWT {name} must use real Rust modules"
        );
        assert!(
            !source.contains("allow(unused"),
            "IDWT {name} must not add a broad unused allowance"
        );
        assert!(
            !source.contains("allow(clippy::too_many_lines"),
            "IDWT {name} must not suppress file decomposition pressure"
        );
    }
}
