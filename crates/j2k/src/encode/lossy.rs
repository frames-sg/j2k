// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::{format, vec::Vec};

use j2k_core::Unsupported;
use j2k_native::EncodeRoiRegion as NativeEncodeRoiRegion;

use super::contracts::{
    J2kBlockCodingMode, J2kLossyEncodeOptions, J2kLossyEncodeReport, J2kRateTarget,
};
use super::native::native_lossy_options;
use super::samples::J2kLossySamples;
use super::validation::{decoded_psnr, validate_lossy_roundtrip};
use crate::J2kError;

pub(super) struct LossyAttempt {
    pub(super) codestream: Vec<u8>,
    pub(super) quantization_scale: f32,
}

pub(super) fn encode_cpu_lossy(
    samples: J2kLossySamples<'_>,
    options: &J2kLossyEncodeOptions,
    quantization_scale: f32,
) -> Result<Vec<u8>, J2kError> {
    let options = native_lossy_options(samples, options, quantization_scale)?;
    j2k_native::encode(
        samples.data,
        samples.width,
        samples.height,
        samples.components,
        samples.bit_depth,
        samples.signed,
        &options,
    )
    .map_err(|err| J2kError::backend(format!("JPEG 2000 lossy encode failed: {err}")))
}

pub(super) fn encode_cpu_lossy_with_roi_regions(
    samples: J2kLossySamples<'_>,
    options: &J2kLossyEncodeOptions,
    quantization_scale: f32,
    roi_regions: &[NativeEncodeRoiRegion],
) -> Result<Vec<u8>, J2kError> {
    let options = native_lossy_options(samples, options, quantization_scale)?;
    j2k_native::encode_with_roi_regions(
        samples.data,
        samples.width,
        samples.height,
        samples.components,
        samples.bit_depth,
        samples.signed,
        &options,
        roi_regions,
    )
    .map_err(|err| J2kError::backend(format!("JPEG 2000 lossy ROI encode failed: {err}")))
}

pub(super) fn encode_lossy_targeted(
    samples: J2kLossySamples<'_>,
    options: &J2kLossyEncodeOptions,
    target: Option<J2kRateTarget>,
    mut encode_at_scale: impl FnMut(f32) -> Result<Vec<u8>, J2kError>,
) -> Result<LossyAttempt, J2kError> {
    match target {
        None => {
            let codestream = encode_at_scale(1.0)?;
            Ok(LossyAttempt {
                codestream,
                quantization_scale: 1.0,
            })
        }
        Some(J2kRateTarget::Bytes(bytes)) => {
            encode_lossy_to_byte_target(samples, options, bytes, encode_at_scale)
        }
        Some(J2kRateTarget::BitsPerPixel(bits_per_pixel)) => {
            let target_bytes = target_bytes_for_bpp(samples, bits_per_pixel)?;
            encode_lossy_to_byte_target(samples, options, target_bytes, encode_at_scale)
        }
        Some(J2kRateTarget::PsnrDb(psnr_db)) => {
            encode_lossy_to_psnr_target(samples, options, psnr_db, encode_at_scale)
        }
    }
}

pub(super) fn encode_lossy_to_byte_target(
    _samples: J2kLossySamples<'_>,
    options: &J2kLossyEncodeOptions,
    target_bytes: u64,
    mut encode_at_scale: impl FnMut(f32) -> Result<Vec<u8>, J2kError>,
) -> Result<LossyAttempt, J2kError> {
    let tolerance = byte_target_tolerance(target_bytes);
    let mut low = 1.0f32;
    let mut high = 1.0f32;
    let mut best = LossyAttempt {
        codestream: encode_at_scale(high)?,
        quantization_scale: high,
    };
    let mut best_diff = byte_target_diff(best.codestream.len() as u64, target_bytes);

    while best.codestream.len() as u64 > target_bytes.saturating_add(tolerance)
        && high < 1_048_576.0
    {
        low = high;
        high *= 2.0;
        let codestream = encode_at_scale(high)?;
        let diff = byte_target_diff(codestream.len() as u64, target_bytes);
        if diff < best_diff {
            best = LossyAttempt {
                codestream,
                quantization_scale: high,
            };
            best_diff = diff;
        }
    }

    if best.codestream.len() as u64 > target_bytes.saturating_add(tolerance) {
        return Err(J2kError::RateTargetUnreachable {
            target: format!("{target_bytes} bytes"),
            best: format!("{} bytes", best.codestream.len()),
        });
    }

    for _ in 0..options.psnr_iteration_budget.max(1) {
        let mid = (low + high) * 0.5;
        let codestream = encode_at_scale(mid)?;
        let len = codestream.len() as u64;
        let diff = byte_target_diff(len, target_bytes);
        if diff < best_diff {
            best = LossyAttempt {
                codestream,
                quantization_scale: mid,
            };
            best_diff = diff;
        }
        if len > target_bytes {
            low = mid;
        } else {
            high = mid;
        }
    }

    Ok(best)
}

pub(super) fn encode_lossy_to_psnr_target(
    samples: J2kLossySamples<'_>,
    options: &J2kLossyEncodeOptions,
    target_psnr_db: f64,
    mut encode_at_scale: impl FnMut(f32) -> Result<Vec<u8>, J2kError>,
) -> Result<LossyAttempt, J2kError> {
    let tolerance = options.psnr_tolerance_db;
    let mut low = 1.0f32;
    let mut high = 1.0f32;
    let mut best = LossyAttempt {
        codestream: encode_at_scale(high)?,
        quantization_scale: high,
    };
    let mut best_psnr = decoded_psnr(samples, &best.codestream)?;
    if best_psnr + tolerance < target_psnr_db {
        return Err(J2kError::RateTargetUnreachable {
            target: format!("{target_psnr_db:.3} dB"),
            best: format!("{best_psnr:.3} dB"),
        });
    }

    for _ in 0..options.psnr_iteration_budget.max(1) {
        high *= 2.0;
        let codestream = encode_at_scale(high)?;
        let psnr = decoded_psnr(samples, &codestream)?;
        if psnr + tolerance >= target_psnr_db {
            best = LossyAttempt {
                codestream,
                quantization_scale: high,
            };
            best_psnr = psnr;
            low = high;
        } else {
            break;
        }
    }

    for _ in 0..options.psnr_iteration_budget.max(1) {
        let mid = (low + high) * 0.5;
        let codestream = encode_at_scale(mid)?;
        let psnr = decoded_psnr(samples, &codestream)?;
        if psnr + tolerance >= target_psnr_db {
            best = LossyAttempt {
                codestream,
                quantization_scale: mid,
            };
            best_psnr = psnr;
            low = mid;
        } else {
            high = mid;
        }
    }

    let _ = best_psnr;
    Ok(best)
}

pub(super) fn lossy_quality_layer_byte_targets(
    samples: J2kLossySamples<'_>,
    options: &J2kLossyEncodeOptions,
) -> Result<Vec<u64>, J2kError> {
    if options.quality_layers.len() <= 1 {
        return Ok(Vec::new());
    }

    let mut targets = Vec::with_capacity(options.quality_layers.len());
    for layer in &options.quality_layers {
        match layer.target {
            J2kRateTarget::Bytes(bytes) => targets.push(bytes),
            J2kRateTarget::BitsPerPixel(bits_per_pixel) => {
                targets.push(target_bytes_for_bpp(samples, bits_per_pixel)?);
            }
            J2kRateTarget::PsnrDb(_) => return Ok(Vec::new()),
        }
    }
    if targets.windows(2).any(|pair| pair[0] > pair[1]) {
        return Err(J2kError::Unsupported(Unsupported {
            what: "JPEG 2000 lossy quality layer targets must be cumulative and monotonic",
        }));
    }
    Ok(targets)
}

pub(super) fn validate_lossy_options(options: &J2kLossyEncodeOptions) -> Result<(), J2kError> {
    if options.quality_layers.len() > 32 {
        return Err(J2kError::Unsupported(Unsupported {
            what: "JPEG 2000 lossy encode supports 1-32 quality layers",
        }));
    }
    if let Some((tile_width, tile_height)) = options.tile_size {
        if tile_width == 0 || tile_height == 0 {
            return Err(J2kError::Unsupported(Unsupported {
                what: "JPEG 2000 lossy tile dimensions must be non-zero",
            }));
        }
    }
    if options
        .precinct_exponents
        .iter()
        .any(|&(ppx, ppy)| ppx > 15 || ppy > 15)
    {
        return Err(J2kError::Unsupported(Unsupported {
            what: "JPEG 2000 lossy precinct exponents must be 0-15",
        }));
    }
    if !(options.psnr_tolerance_db.is_finite() && options.psnr_tolerance_db >= 0.0) {
        return Err(J2kError::Unsupported(Unsupported {
            what: "JPEG 2000 lossy PSNR tolerance must be finite and non-negative",
        }));
    }
    if options.psnr_iteration_budget == 0 {
        return Err(J2kError::Unsupported(Unsupported {
            what: "JPEG 2000 lossy PSNR iteration budget must be greater than zero",
        }));
    }
    validate_rate_target(options.rate_target)?;
    for layer in &options.quality_layers {
        validate_rate_target(Some(layer.target))?;
    }
    Ok(())
}

pub(super) fn effective_lossy_target(
    options: &J2kLossyEncodeOptions,
) -> Result<Option<J2kRateTarget>, J2kError> {
    match (options.rate_target, options.quality_layers.as_slice()) {
        (target, []) => Ok(target),
        (None, [layer]) => Ok(Some(layer.target)),
        (Some(target), [layer]) if target == layer.target => Ok(Some(target)),
        (Some(_), [_]) => Err(J2kError::Unsupported(Unsupported {
            what:
                "specify either a JPEG 2000 lossy rate target or one quality layer target, not both",
        })),
        (None, layers) => Ok(layers.last().map(|layer| layer.target)),
        (Some(target), layers) if layers.last().is_some_and(|layer| layer.target == target) => {
            Ok(Some(target))
        }
        (Some(_), _) => Err(J2kError::Unsupported(Unsupported {
            what: "when multiple JPEG 2000 quality layers are specified, the single rate target must match the final cumulative layer target",
        })),
    }
}

pub(super) fn validate_rate_target(target: Option<J2kRateTarget>) -> Result<(), J2kError> {
    match target {
        None => Ok(()),
        Some(J2kRateTarget::BitsPerPixel(bits_per_pixel))
            if bits_per_pixel.is_finite() && bits_per_pixel > 0.0 =>
        {
            Ok(())
        }
        Some(J2kRateTarget::Bytes(bytes)) if bytes > 0 => Ok(()),
        Some(J2kRateTarget::PsnrDb(psnr_db)) if psnr_db.is_finite() && psnr_db > 0.0 => Ok(()),
        Some(J2kRateTarget::BitsPerPixel(_)) => Err(J2kError::Unsupported(Unsupported {
            what: "JPEG 2000 lossy bits-per-pixel target must be finite and greater than zero",
        })),
        Some(J2kRateTarget::Bytes(_)) => Err(J2kError::Unsupported(Unsupported {
            what: "JPEG 2000 lossy byte target must be greater than zero",
        })),
        Some(J2kRateTarget::PsnrDb(_)) => Err(J2kError::Unsupported(Unsupported {
            what: "JPEG 2000 lossy PSNR target must be finite and greater than zero",
        })),
    }
}

pub(super) fn lossy_report(
    samples: J2kLossySamples<'_>,
    options: &J2kLossyEncodeOptions,
    target: Option<J2kRateTarget>,
    attempt: &LossyAttempt,
) -> Result<J2kLossyEncodeReport, J2kError> {
    let actual_bytes = attempt.codestream.len() as u64;
    Ok(J2kLossyEncodeReport {
        target,
        quality_layers: u16::from(lossy_quality_layer_count(options)),
        quantization_scale: attempt.quantization_scale,
        actual_bytes,
        actual_bits_per_pixel: bits_per_pixel(samples, actual_bytes),
        psnr_db: validate_lossy_roundtrip(samples, &attempt.codestream, options.validation)?,
        ht_rate_granularity_bytes: (options.block_coding_mode
            == J2kBlockCodingMode::HighThroughput)
            .then_some(actual_bytes),
    })
}

pub(super) fn lossy_quality_layer_count(options: &J2kLossyEncodeOptions) -> u8 {
    u8::try_from(options.quality_layers.len().max(1)).unwrap_or(32)
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "finite positive byte targets are range-checked against u64 before conversion"
)]
pub(super) fn target_bytes_for_bpp(
    samples: J2kLossySamples<'_>,
    bits_per_pixel: f64,
) -> Result<u64, J2kError> {
    let pixels = f64::from(samples.width) * f64::from(samples.height);
    let bytes = (pixels * bits_per_pixel / 8.0).ceil();
    if bytes.is_finite() && bytes > 0.0 && bytes <= 18_446_744_073_709_551_615.0 {
        Ok(bytes as u64)
    } else {
        Err(J2kError::Unsupported(Unsupported {
            what: "JPEG 2000 lossy bits-per-pixel target overflows byte target",
        }))
    }
}

pub(super) fn byte_target_tolerance(target_bytes: u64) -> u64 {
    target_bytes.div_ceil(100).max(512)
}

pub(super) fn byte_target_diff(actual: u64, target: u64) -> u64 {
    actual.abs_diff(target)
}

pub(super) fn bits_per_pixel(samples: J2kLossySamples<'_>, bytes: u64) -> f64 {
    (u64_to_f64(bytes) * 8.0) / (f64::from(samples.width) * f64::from(samples.height))
}

#[expect(
    clippy::cast_precision_loss,
    reason = "lossy rate metrics intentionally use approximate f64 ratios"
)]
pub(super) fn usize_to_f64(value: usize) -> f64 {
    value as f64
}

#[expect(
    clippy::cast_precision_loss,
    reason = "lossy rate metrics intentionally use approximate f64 ratios"
)]
pub(super) fn u64_to_f64(value: u64) -> f64 {
    value as f64
}
