// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::{vec, vec::Vec};

use j2k_native::{DecodeSettings, EncodeRoiRegion, Image};

use super::super::{
    bits_per_pixel, byte_target_diff, byte_target_tolerance, effective_lossy_target,
    encode_cpu_lossy, encode_cpu_lossy_with_roi_regions, encode_lossy_targeted,
    lossy_quality_layer_byte_targets, lossy_quality_layer_count, lossy_report,
    validate_lossy_options, validate_rate_target, J2kBlockCodingMode, J2kError,
    J2kLossyEncodeOptions, J2kRateTarget, LossyAttempt,
};
use super::single_pixel_samples;
use crate::{J2kEncodeValidation, J2kLossySamples, J2kQualityLayer};

fn layer(target: J2kRateTarget) -> J2kQualityLayer {
    J2kQualityLayer::new(target)
}

#[test]
fn target_dispatch_preserves_none_bytes_and_bpp_semantics() {
    let samples = single_pixel_samples();
    let options = J2kLossyEncodeOptions {
        psnr_iteration_budget: 1,
        ..J2kLossyEncodeOptions::default()
    };
    let mut scales = Vec::new();
    let attempt = encode_lossy_targeted(samples, &options, None, |scale| {
        scales.push(scale);
        Ok(vec![1, 2, 3])
    })
    .unwrap();
    assert_eq!(scales, [1.0]);
    assert_eq!(attempt.codestream, [1, 2, 3]);
    assert_eq!(attempt.quantization_scale.to_bits(), 1.0_f32.to_bits());

    for target in [J2kRateTarget::Bytes(1), J2kRateTarget::BitsPerPixel(8.0)] {
        let attempt = encode_lossy_targeted(samples, &options, Some(target), |scale| {
            let len = if scale <= 1.0 { 600 } else { 1 };
            Ok(vec![0; len])
        })
        .unwrap();
        assert_eq!(attempt.codestream.len(), 1);
        assert!(attempt.quantization_scale >= 1.0);
    }
}

#[test]
fn cpu_lossy_adapters_and_report_preserve_decode_and_metric_contracts() {
    let pixels: Vec<u8> = (0_u8..64).map(|value| value.wrapping_mul(5)).collect();
    let samples = J2kLossySamples::new(&pixels, 8, 8, 1, 8, false).unwrap();
    let options = J2kLossyEncodeOptions {
        block_coding_mode: J2kBlockCodingMode::HighThroughput,
        max_decomposition_levels: Some(0),
        validation: J2kEncodeValidation::External,
        ..J2kLossyEncodeOptions::default()
    };
    let plain = encode_cpu_lossy(samples, &options, 1.5).unwrap();
    let roi = encode_cpu_lossy_with_roi_regions(
        samples,
        &options,
        1.5,
        &[EncodeRoiRegion {
            component: 0,
            x: 1,
            y: 2,
            width: 5,
            height: 4,
            shift: 12,
        }],
    )
    .unwrap();

    for codestream in [&plain, &roi] {
        let image = Image::new(codestream, &DecodeSettings::default()).unwrap();
        assert_eq!((image.width(), image.height()), (8, 8));
        assert_eq!(image.decode().unwrap().len(), pixels.len());
    }

    let report = lossy_report(
        samples,
        &options,
        Some(J2kRateTarget::Bytes(plain.len() as u64)),
        &LossyAttempt {
            codestream: plain,
            quantization_scale: 1.5,
        },
    )
    .unwrap();
    assert_eq!(
        report.target,
        Some(J2kRateTarget::Bytes(report.actual_bytes))
    );
    assert_eq!(report.quality_layers, 1);
    assert_eq!(report.quantization_scale.to_bits(), 1.5_f32.to_bits());
    assert!(report.actual_bits_per_pixel > 0.0);
    assert_eq!(report.psnr_db, None);
    assert_eq!(report.ht_rate_granularity_bytes, Some(report.actual_bytes));
}

#[test]
fn effective_target_requires_single_and_layer_targets_to_agree() {
    let bytes = J2kRateTarget::Bytes(100);
    let psnr = J2kRateTarget::PsnrDb(30.0);

    assert_eq!(
        effective_lossy_target(&J2kLossyEncodeOptions::default()).unwrap(),
        None
    );
    assert_eq!(
        effective_lossy_target(&J2kLossyEncodeOptions::default().with_rate_target(Some(bytes)))
            .unwrap(),
        Some(bytes)
    );
    assert_eq!(
        effective_lossy_target(
            &J2kLossyEncodeOptions::default().with_quality_layers(vec![layer(bytes)])
        )
        .unwrap(),
        Some(bytes)
    );
    assert_eq!(
        effective_lossy_target(
            &J2kLossyEncodeOptions::default()
                .with_rate_target(Some(bytes))
                .with_quality_layers(vec![layer(bytes)])
        )
        .unwrap(),
        Some(bytes)
    );
    assert!(effective_lossy_target(
        &J2kLossyEncodeOptions::default()
            .with_rate_target(Some(psnr))
            .with_quality_layers(vec![layer(bytes)])
    )
    .is_err());

    let layers = vec![layer(J2kRateTarget::Bytes(50)), layer(bytes)];
    assert_eq!(
        effective_lossy_target(
            &J2kLossyEncodeOptions::default().with_quality_layers(layers.clone())
        )
        .unwrap(),
        Some(bytes)
    );
    assert_eq!(
        effective_lossy_target(
            &J2kLossyEncodeOptions::default()
                .with_rate_target(Some(bytes))
                .with_quality_layers(layers.clone())
        )
        .unwrap(),
        Some(bytes)
    );
    assert!(effective_lossy_target(
        &J2kLossyEncodeOptions::default()
            .with_rate_target(Some(psnr))
            .with_quality_layers(layers)
    )
    .is_err());
}

#[test]
fn quality_layer_byte_targets_convert_bpp_and_reject_nonmonotonic_layers() {
    let samples = single_pixel_samples();
    let empty = J2kLossyEncodeOptions::default();
    assert!(lossy_quality_layer_byte_targets(samples, &empty)
        .unwrap()
        .is_empty());
    assert_eq!(lossy_quality_layer_count(&empty), 1);

    let options = empty.clone().with_quality_layers(vec![
        layer(J2kRateTarget::Bytes(2)),
        layer(J2kRateTarget::BitsPerPixel(24.0)),
    ]);
    assert_eq!(
        lossy_quality_layer_byte_targets(samples, &options).unwrap(),
        [2, 3]
    );
    assert_eq!(lossy_quality_layer_count(&options), 2);

    let psnr = empty.clone().with_quality_layers(vec![
        layer(J2kRateTarget::Bytes(1)),
        layer(J2kRateTarget::PsnrDb(30.0)),
    ]);
    assert!(lossy_quality_layer_byte_targets(samples, &psnr)
        .unwrap()
        .is_empty());

    let descending = empty.with_quality_layers(vec![
        layer(J2kRateTarget::Bytes(3)),
        layer(J2kRateTarget::Bytes(2)),
    ]);
    assert!(lossy_quality_layer_byte_targets(samples, &descending).is_err());
}

#[test]
fn option_validation_rejects_each_invalid_boundary() {
    let invalid = [
        J2kLossyEncodeOptions {
            quality_layers: vec![layer(J2kRateTarget::Bytes(1)); 33],
            ..J2kLossyEncodeOptions::default()
        },
        J2kLossyEncodeOptions {
            tile_size: Some((0, 1)),
            ..J2kLossyEncodeOptions::default()
        },
        J2kLossyEncodeOptions {
            precinct_exponents: vec![(16, 1)],
            ..J2kLossyEncodeOptions::default()
        },
        J2kLossyEncodeOptions {
            psnr_tolerance_db: f64::NAN,
            ..J2kLossyEncodeOptions::default()
        },
        J2kLossyEncodeOptions {
            psnr_iteration_budget: 0,
            ..J2kLossyEncodeOptions::default()
        },
        J2kLossyEncodeOptions {
            rate_target: Some(J2kRateTarget::Bytes(0)),
            ..J2kLossyEncodeOptions::default()
        },
        J2kLossyEncodeOptions {
            quality_layers: vec![layer(J2kRateTarget::PsnrDb(f64::INFINITY))],
            ..J2kLossyEncodeOptions::default()
        },
    ];
    for options in invalid {
        assert!(matches!(
            validate_lossy_options(&options),
            Err(J2kError::Unsupported(_))
        ));
    }

    let valid = J2kLossyEncodeOptions {
        tile_size: Some((1, 1)),
        precinct_exponents: vec![(15, 15)],
        psnr_tolerance_db: 0.0,
        psnr_iteration_budget: 1,
        rate_target: Some(J2kRateTarget::BitsPerPixel(1.0)),
        block_coding_mode: J2kBlockCodingMode::HighThroughput,
        ..J2kLossyEncodeOptions::default()
    };
    assert!(validate_lossy_options(&valid).is_ok());
}

#[test]
fn rate_target_and_metric_helpers_cover_numeric_edges() {
    for target in [
        None,
        Some(J2kRateTarget::Bytes(1)),
        Some(J2kRateTarget::BitsPerPixel(0.5)),
        Some(J2kRateTarget::PsnrDb(1.0)),
    ] {
        assert!(validate_rate_target(target).is_ok());
    }
    for target in [
        Some(J2kRateTarget::Bytes(0)),
        Some(J2kRateTarget::BitsPerPixel(0.0)),
        Some(J2kRateTarget::BitsPerPixel(f64::NAN)),
        Some(J2kRateTarget::PsnrDb(0.0)),
        Some(J2kRateTarget::PsnrDb(f64::INFINITY)),
    ] {
        assert!(validate_rate_target(target).is_err());
    }

    assert_eq!(byte_target_tolerance(1), 512);
    assert_eq!(byte_target_tolerance(100_000), 1_000);
    assert_eq!(byte_target_diff(2, 9), 7);
    assert_eq!(byte_target_diff(9, 2), 7);
    assert_eq!(
        bits_per_pixel(single_pixel_samples(), 2).to_bits(),
        16.0_f64.to_bits()
    );
}
