// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_t803::{
    compare_peak_samples, compare_samples, normalize_component, parse_pgx, Component, ErrorBounds,
    NormalizationTarget,
};

fn pgx_bytes(header: &[u8], payload: &[u8]) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(header.len() + payload.len())
        .expect("allocate PGX fixture");
    bytes.extend_from_slice(header);
    bytes.extend_from_slice(payload);
    bytes
}

#[test]
fn pgx_parses_unsigned_one_byte_samples() {
    let bytes = pgx_bytes(b"PG ML +8 2 2\n", &[0, 1, 127, 255]);

    let image = parse_pgx(&bytes).expect("valid PGX");

    assert_eq!(image.width, 2);
    assert_eq!(image.height, 2);
    assert_eq!(image.bit_depth, 8);
    assert!(!image.signed);
    assert_eq!(image.samples, [0, 1, 127, 255]);
}

#[test]
fn pgx_parses_spaced_sign_crlf_and_signed_big_endian_samples() {
    let bytes = pgx_bytes(
        b"PG ML - 12 4 1\r\n",
        &[0xf8, 0x00, 0xff, 0xff, 0x00, 0x00, 0x07, 0xff],
    );

    let image = parse_pgx(&bytes).expect("valid signed PGX");

    assert_eq!(image.bit_depth, 12);
    assert!(image.signed);
    assert_eq!(image.samples, [-2048, -1, 0, 2047]);
}

#[test]
fn pgx_parses_official_little_endian_reference_storage() {
    let bytes = pgx_bytes(b"PG LM 12 2 1\n", &[0x34, 0x02, 0xcd, 0x0a]);

    let image = parse_pgx(&bytes).expect("valid little-endian PGX");

    assert_eq!(image.samples, [0x234, 0xacd]);
}

#[test]
fn pgx_parses_official_unsigned_header_with_blank_sign_column() {
    let bytes = pgx_bytes(b"PG ML  8 1 1\n", &[0x7f]);

    let image = parse_pgx(&bytes).expect("valid unsigned PGX");

    assert!(!image.signed);
    assert_eq!(image.samples, [0x7f]);
}

#[test]
fn pgx_parses_four_byte_unsigned_samples() {
    let bytes = pgx_bytes(b"PG ML 32 2 1\n", &[0, 0, 0, 1, 0xff, 0xff, 0xff, 0xff]);

    let image = parse_pgx(&bytes).expect("valid 32-bit PGX");

    assert_eq!(image.samples, [1, 4_294_967_295]);
}

#[test]
fn pgx_rejects_non_normative_or_malformed_storage() {
    let cases: &[(&[u8], &str)] = &[
        (b"PG MM +8 1 1\n\x00", "byte order"),
        (b"PG ML +0 1 1\n\x00", "bit depth"),
        (b"PG ML +33 1 1\n\x00", "bit depth"),
        (b"PG ML +8 0 1\n", "dimensions"),
        (b"PG ML +8 1 1\n", "payload length"),
        (b"PG ML +8 1 1\n\x00\x01", "payload length"),
        (b"PG ML +8 18446744073709551615 2\n", "width"),
        (b"PG ML -12 1 1\n\x0f\xff", "sign extension"),
        (b"PG ML +12 1 1\n\xf0\x00", "precision"),
        (b"PG ML +8\t1 1\n\x00", "header"),
    ];

    for (bytes, expected) in cases {
        let error = parse_pgx(bytes).expect_err("malformed PGX must fail");
        assert!(
            error.to_string().contains(expected),
            "{error:?} did not mention {expected:?}"
        );
    }
}

#[test]
fn normalization_clips_scales_and_crops_the_upper_left() {
    let samples = [-600, -5, 7, 511, 100, 200, 300, 400, -1, 0, 1, 2];
    let component = Component {
        width: 4,
        height: 3,
        bit_depth: 10,
        signed: true,
        post_decode_subsampling: (1, 1),
        samples: &samples,
    };
    let target = NormalizationTarget {
        width: 3,
        height: 2,
        bit_depth: 8,
        signed: true,
    };

    let normalized = normalize_component(component, target).expect("normalization succeeds");

    assert_eq!(normalized, [-128, -2, 1, 25, 50, 75]);
}

#[test]
fn normalization_subsamples_replicated_decoder_output_before_cropping() {
    let samples = [
        128, 128, 128, 128, 0, 0, 0, 0, 64, 64, 64, 64, 255, 255, 255, 255,
    ];
    let component = Component {
        width: 8,
        height: 2,
        bit_depth: 8,
        signed: false,
        post_decode_subsampling: (4, 1),
        samples: &samples,
    };
    let target = NormalizationTarget {
        width: 2,
        height: 2,
        bit_depth: 8,
        signed: false,
    };

    let normalized = normalize_component(component, target).expect("normalization succeeds");

    assert_eq!(normalized, [128, 0, 64, 255]);
}

#[test]
fn normalization_rejects_incompatible_shape_or_signedness() {
    let samples = [0];
    let component = Component {
        width: 1,
        height: 1,
        bit_depth: 8,
        signed: false,
        post_decode_subsampling: (1, 1),
        samples: &samples,
    };

    let shape_error = normalize_component(
        component,
        NormalizationTarget {
            width: 2,
            height: 1,
            bit_depth: 8,
            signed: false,
        },
    )
    .expect_err("oversized target must fail");
    assert!(shape_error.to_string().contains("dimensions"));

    let signedness_error = normalize_component(
        component,
        NormalizationTarget {
            width: 1,
            height: 1,
            bit_depth: 8,
            signed: true,
        },
    )
    .expect_err("signedness mismatch must fail");
    assert!(signedness_error.to_string().contains("signedness"));
}

#[test]
fn comparison_uses_inclusive_peak_and_mse_bounds() {
    let comparison = compare_samples(
        &[0, 4, 8, 12],
        &[2, 2, 10, 10],
        ErrorBounds { peak: 2, mse: 4.0 },
    )
    .expect("comparable samples");

    assert_eq!(comparison.peak, 2);
    assert!((comparison.mse - 4.0).abs() < f64::EPSILON);
    assert!(comparison.passed);

    let too_strict = compare_samples(
        &[0, 4, 8, 12],
        &[2, 2, 10, 10],
        ErrorBounds { peak: 1, mse: 4.0 },
    )
    .expect("comparable samples");
    assert!(!too_strict.passed);
}

#[test]
fn comparison_rejects_invalid_inputs_and_bounds() {
    assert!(compare_samples(&[], &[], ErrorBounds { peak: 0, mse: 0.0 }).is_err());
    assert!(compare_samples(&[0], &[], ErrorBounds { peak: 0, mse: 0.0 }).is_err());
    assert!(compare_samples(
        &[0],
        &[0],
        ErrorBounds {
            peak: 0,
            mse: f64::NAN,
        }
    )
    .is_err());
}

#[test]
fn comparison_preserves_finite_mse_at_the_i64_error_boundary() {
    let comparison = compare_samples(
        &[i64::MIN, 0],
        &[i64::MAX, 0],
        ErrorBounds {
            peak: u64::MAX,
            mse: f64::MAX,
        },
    )
    .expect("the exact u128 sum remains representable as finite f64 MSE");

    assert_eq!(comparison.peak, u64::MAX);
    assert_eq!(comparison.mse.to_bits(), 2.0_f64.powi(127).to_bits());
    assert!(comparison.passed);
}

#[test]
fn peak_only_comparison_uses_an_inclusive_annex_g_bound() {
    assert!(
        compare_peak_samples(&[0, 4, 8], &[2, 2, 10], 2)
            .expect("comparison")
            .passed
    );
    assert!(
        !compare_peak_samples(&[0, 4, 8], &[3, 2, 10], 2)
            .expect("comparison")
            .passed
    );
    assert!(compare_peak_samples(&[], &[], 0).is_err());
}
