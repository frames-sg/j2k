// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::{format, vec::Vec};

use j2k_core::Unsupported;
use j2k_native::EncodeRoiRegion as NativeEncodeRoiRegion;

use super::accelerator::resolve_encode_backend;
use super::contracts::{
    EncodedLossyJ2k, J2kBlockCodingMode, J2kLossyEncodeOptions, J2kLossyEncodeReport, J2kRateTarget,
};
use super::cpu::native_lossy_options;
use super::high_bit;
use super::samples::J2kLossySamples;
use super::validation::validate_lossy_roundtrip;
use crate::{J2kEncodeDispatchReport, J2kError};

mod candidates;
use self::candidates::{CpuCandidates, LossyCandidates};

/// Error contexts of CPU lossy encodes, shared with reused-transform candidates.
const LOSSY_ENCODE_CONTEXT: &str = "native JPEG 2000 lossy encode failed";
const LOSSY_ROI_ENCODE_CONTEXT: &str = "native JPEG 2000 lossy ROI encode failed";

pub(super) struct LossyAttempt {
    pub(super) codestream: Vec<u8>,
    pub(super) quantization_scale: f32,
}

pub(super) fn encode(
    samples: J2kLossySamples<'_>,
    options: &J2kLossyEncodeOptions,
) -> Result<EncodedLossyJ2k, J2kError> {
    validate_lossy_options(options)?;
    high_bit::validate_lossy_options(samples, options)?;
    let target = effective_lossy_target(options)?;
    let attempt = encode_cpu_lossy_targeted(samples, options, target, None)?;
    let report = lossy_report(samples, options, target, &attempt)?;
    Ok(EncodedLossyJ2k {
        codestream: attempt.codestream,
        backend: resolve_encode_backend(options.backend)?,
        dispatch_report: J2kEncodeDispatchReport::default(),
        width: samples.width,
        height: samples.height,
        components: samples.components,
        bit_depth: samples.bit_depth,
        signed: samples.signed,
        report,
    })
}

pub(super) fn encode_cpu_lossy(
    samples: J2kLossySamples<'_>,
    options: &J2kLossyEncodeOptions,
    quantization_scale: f32,
) -> Result<Vec<u8>, J2kError> {
    let native_options = native_lossy_options(samples, options, quantization_scale)?;
    let encode_result = options.qfactor.map_or_else(
        || {
            j2k_native::encode(
                samples.data,
                samples.width,
                samples.height,
                samples.components,
                samples.bit_depth,
                samples.signed,
                &native_options,
            )
        },
        |qfactor| {
            j2k_native::encode_htj2k_with_qfactor(
                samples.data,
                samples.width,
                samples.height,
                samples.components,
                samples.bit_depth,
                samples.signed,
                qfactor,
                &native_options,
            )
        },
    );
    encode_result.map_err(|source| {
        J2kError::from_native_encode_error_with_context(source, LOSSY_ENCODE_CONTEXT)
    })
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
    .map_err(|source| {
        J2kError::from_native_encode_error_with_context(source, LOSSY_ROI_ENCODE_CONTEXT)
    })
}

/// CPU lossy encode at the target's selected scale; `roi_regions` selects the
/// ROI encoder. Rate and PSNR searches share one colour transform and DWT
/// across candidates, each byte-identical to an independent encode.
pub(super) fn encode_cpu_lossy_targeted(
    samples: J2kLossySamples<'_>,
    options: &J2kLossyEncodeOptions,
    target: Option<J2kRateTarget>,
    roi_regions: Option<&[NativeEncodeRoiRegion]>,
) -> Result<LossyAttempt, J2kError> {
    // Fixed-scale encodes have one candidate; Qfactor has its own entry point.
    let reuse = target.is_some() && options.qfactor.is_none();
    let mut candidates = CpuCandidates::new(samples, options, roi_regions, reuse);
    encode_lossy_targeted(samples, options, target, &mut candidates)
}

pub(super) fn encode_lossy_targeted(
    samples: J2kLossySamples<'_>,
    options: &J2kLossyEncodeOptions,
    target: Option<J2kRateTarget>,
    candidates: &mut impl LossyCandidates,
) -> Result<LossyAttempt, J2kError> {
    match target {
        None => {
            let codestream = candidates.encode_candidate(1.0)?;
            Ok(LossyAttempt {
                codestream,
                quantization_scale: 1.0,
            })
        }
        Some(J2kRateTarget::Bytes(bytes)) => {
            encode_lossy_to_byte_target(samples, options, bytes, candidates)
        }
        Some(J2kRateTarget::BitsPerPixel(bits_per_pixel)) => {
            let target_bytes = target_bytes_for_bpp(samples, bits_per_pixel)?;
            encode_lossy_to_byte_target(samples, options, target_bytes, candidates)
        }
        Some(J2kRateTarget::PsnrDb(psnr_db)) => {
            encode_lossy_to_psnr_target(samples, options, psnr_db, candidates)
        }
    }
}

/// Resolve the entire admissible interval before encoding any rate candidate.
fn quantization_scale_bounds(
    samples: J2kLossySamples<'_>,
    options: &J2kLossyEncodeOptions,
) -> Result<(f32, f32), J2kError> {
    let native = native_lossy_options(samples, options, 1.0)?;
    let (lower, upper) =
        j2k_native::irreversible_quantization_scale_bounds(samples.bit_depth, &native);
    let lower = lower.max(1.0);
    if lower > upper {
        return Err(J2kError::RateTargetUnreachable {
            target: "a representable component quantization profile at scale >= 1".into(),
            best: "no common admissible scale".into(),
        });
    }
    Ok((lower, upper))
}

/// Size slope (d log bytes / d log scale) assumed until two probes measure one.
const INITIAL_SIZE_SLOPE: f64 = -0.8;
/// Least scale change per probe until the target is bracketed, so the floor or
/// the ceiling is reached in a bounded number of encodes.
const MIN_SCALE_GROWTH: f64 = 1.25;
/// First-probe rate model, fitted on natural 8-bit RGB images: the scale meeting
/// a target of `b` bits per sample is near `exp(START_LOG_SCALE) * b^START_RATE_EXPONENT`,
/// doubling per extra bit of depth, and `exp(UNTRANSFORMED_LOG_SCALE)` times
/// larger without the multi-component transform, whose chroma samples are the
/// cheapest. A poor guess costs probes, not the result, and needs no encode,
/// unlike starting from the finest scale.
const START_LOG_SCALE: f64 = 2.42;
const START_RATE_EXPONENT: f64 = -0.8;
const UNTRANSFORMED_LOG_SCALE: f64 = 0.8;

/// Every probe is a full encode at one global scale, the first one at the rate
/// model's guess. Until the target is bracketed, the next scale extrapolates
/// the last two (log scale, log bytes) points, up after a probe that is too
/// large and down after one that is too small; once it is bracketed, it
/// interpolates inside the bracket, bisecting whenever the bracket fails to
/// halve. The first probe within 1% of the target is returned as encoded, as is
/// the finest admissible scale when it fits and the coarsest when it is too
/// large but within the size allowance. After `psnr_iteration_budget`
/// bracketed probes, the probe closest to the target within the allowance is
/// re-encoded so the result is always the final encode.
pub(super) fn encode_lossy_to_byte_target(
    samples: J2kLossySamples<'_>,
    options: &J2kLossyEncodeOptions,
    target_bytes: u64,
    candidates: &mut impl LossyCandidates,
) -> Result<LossyAttempt, J2kError> {
    let tolerance = byte_target_tolerance(target_bytes);
    let maximum_bytes = target_bytes.saturating_add(tolerance);
    // The allowance's 512-byte floor is too loose for a small target, so only
    // a probe within 1% ends the search; others within it are fallbacks.
    let near = target_bytes.div_ceil(100);
    let near_target = target_bytes.saturating_sub(near)..=target_bytes.saturating_add(near);
    let (scale_floor, scale_ceiling) = quantization_scale_bounds(samples, options)?;
    let log_target = u64_to_f64(target_bytes).ln();
    let unreachable = |len: u64| J2kError::RateTargetUnreachable {
        target: format!("{target_bytes} bytes"),
        best: format!("{len} bytes"),
    };

    let mut next = start_log_scale(samples, target_bytes).max(f64::from(scale_floor).ln());
    let (mut too_large, mut too_small) = (None::<(f64, f64)>, None::<(f64, f64)>);
    let (mut previous, mut closest) = (None::<(f64, f64)>, None::<(u64, f32)>);
    let (mut bracket_width, mut bracketed_probes) = (f64::INFINITY, 0_u8);
    loop {
        let scale = f64_to_f32(next.exp()).clamp(scale_floor, scale_ceiling);
        let codestream = candidates.encode_candidate(scale)?;
        let len = codestream.len() as u64;
        // The finest scale is returned whenever it fits and the coarsest when it
        // is too large but within the allowance: no other scale moves closer.
        let bounded = (scale <= scale_floor || (scale >= scale_ceiling && len > target_bytes))
            && len <= maximum_bytes;
        if near_target.contains(&len) || bounded {
            if candidates.reuse_terminal_byte_output() {
                return Ok(LossyAttempt {
                    codestream,
                    quantization_scale: scale,
                });
            }
            // Accelerator candidates keep the final encode of the selected scale.
            drop(codestream);
            return encode_selected_byte_target_scale(
                candidates,
                scale,
                maximum_bytes,
                unreachable,
            );
        }
        // Retain only scalar search state so the next native encode is not run
        // while an earlier output allocation escapes its phase budget.
        drop(codestream);
        let diff = len.abs_diff(target_bytes);
        if len <= maximum_bytes && closest.is_none_or(|(best, _)| diff < best) {
            closest = Some((diff, scale));
        }
        let point = log_point(scale, len);
        if len < target_bytes {
            too_small = Some(point);
        } else if scale >= scale_ceiling {
            return Err(unreachable(len));
        } else {
            too_large = Some(point);
        }
        next = if let (Some(too_large), Some(fit)) = (too_large, too_small) {
            if bracketed_probes >= options.psnr_iteration_budget.max(1) {
                // The too-small probe fits, so a closest probe exists.
                let Some((_, fit_scale)) = closest else {
                    return Err(unreachable(len));
                };
                return encode_selected_byte_target_scale(
                    candidates,
                    fit_scale,
                    maximum_bytes,
                    unreachable,
                );
            }
            bracketed_probes += 1;
            let width = fit.0 - too_large.0;
            let secant = too_large.0 + (log_target - too_large.1) * width / (fit.1 - too_large.1);
            let halving = width <= bracket_width / 2.0;
            bracket_width = width;
            if halving && too_large.0 < secant && secant < fit.0 {
                secant
            } else {
                too_large.0 + width / 2.0
            }
        } else {
            let slope = previous.map_or(INITIAL_SIZE_SLOPE, |p| (point.1 - p.1) / (point.0 - p.0));
            let slope = if slope.is_finite() {
                slope.clamp(-3.0, -0.25)
            } else {
                INITIAL_SIZE_SLOPE
            };
            let (step, least) = ((log_target - point.1) / slope, MIN_SCALE_GROWTH.ln());
            point.0
                + if too_large.is_some() {
                    step.max(least)
                } else {
                    step.min(-least)
                }
        };
        previous = Some(point);
    }
}

/// The final encode of a byte-target search's selected scale, which must still
/// fit the size allowance.
fn encode_selected_byte_target_scale(
    candidates: &mut impl LossyCandidates,
    scale: f32,
    maximum_bytes: u64,
    unreachable: impl Fn(u64) -> J2kError,
) -> Result<LossyAttempt, J2kError> {
    let codestream = candidates.encode_candidate(scale)?;
    let len = codestream.len() as u64;
    if len > maximum_bytes {
        return Err(unreachable(len));
    }
    Ok(LossyAttempt {
        codestream,
        quantization_scale: scale,
    })
}

/// The first probe's log scale under the rate model.
fn start_log_scale(samples: J2kLossySamples<'_>, target_bytes: u64) -> f64 {
    let count =
        f64::from(samples.width) * f64::from(samples.height) * f64::from(samples.components);
    let bits_per_sample = u64_to_f64(target_bytes) * 8.0 / count;
    // As `native_lossy_options` decides the multi-component transform.
    let untransformed = if matches!(samples.components, 3 | 4) {
        0.0
    } else {
        UNTRANSFORMED_LOG_SCALE
    };
    START_LOG_SCALE
        + START_RATE_EXPONENT * bits_per_sample.ln()
        + (f64::from(samples.bit_depth) - 8.0) * core::f64::consts::LN_2
        + untransformed
}

fn log_point(scale: f32, len: u64) -> (f64, f64) {
    (f64::from(scale).ln(), u64_to_f64(len.max(1)).ln())
}

pub(super) fn encode_lossy_to_psnr_target(
    samples: J2kLossySamples<'_>,
    options: &J2kLossyEncodeOptions,
    target_psnr_db: f64,
    candidates: &mut impl LossyCandidates,
) -> Result<LossyAttempt, J2kError> {
    let tolerance = options.psnr_tolerance_db;
    let (scale_floor, scale_ceiling) = quantization_scale_bounds(samples, options)?;
    let mut low = scale_floor;
    let mut high = scale_floor;
    let initial_codestream = candidates.encode_candidate(high)?;
    let initial_psnr = candidates.candidate_psnr(samples, &initial_codestream)?;
    // The selected scale is re-encoded once after the search. Keeping the
    // earlier output here would undercount it during every later candidate.
    drop(initial_codestream);
    let mut best_scale = (initial_psnr + tolerance >= target_psnr_db).then_some(high);
    let mut best_psnr = initial_psnr;

    for _ in 0..options.psnr_iteration_budget.max(1) {
        if high >= scale_ceiling {
            break;
        }
        high = (high * 2.0).min(scale_ceiling);
        let codestream = candidates.encode_candidate(high)?;
        let psnr = candidates.candidate_psnr(samples, &codestream)?;
        best_psnr = best_psnr.max(psnr);
        if psnr + tolerance >= target_psnr_db {
            best_scale = Some(high);
            low = high;
        } else if best_scale.is_some() {
            break;
        }
        // A failed initial quality measurement is not a proof that every
        // coarser candidate fails: reconstructed PSNR need not be monotonic.
    }
    // Always inspect the admissible endpoint if no probe has met the target.
    if best_scale.is_none() && high < scale_ceiling {
        high = scale_ceiling;
        let codestream = candidates.encode_candidate(high)?;
        let psnr = candidates.candidate_psnr(samples, &codestream)?;
        best_psnr = best_psnr.max(psnr);
        if psnr + tolerance >= target_psnr_db {
            best_scale = Some(high);
            low = high;
        }
    }

    for _ in 0..options.psnr_iteration_budget.max(1) {
        let mid = f32::midpoint(low, high);
        if mid <= low || mid >= high {
            break;
        }
        let codestream = candidates.encode_candidate(mid)?;
        let psnr = candidates.candidate_psnr(samples, &codestream)?;
        best_psnr = best_psnr.max(psnr);
        if psnr + tolerance >= target_psnr_db {
            best_scale = Some(mid);
            low = mid;
        } else {
            high = mid;
        }
    }
    let best_scale = best_scale.ok_or_else(|| J2kError::RateTargetUnreachable {
        target: format!("{target_psnr_db:.3} dB"),
        best: format!("{best_psnr:.3} dB"),
    })?;
    let codestream = candidates.encode_candidate(best_scale)?;
    let final_psnr = candidates.candidate_psnr(samples, &codestream)?;
    if final_psnr + tolerance < target_psnr_db {
        return Err(J2kError::RateTargetUnreachable {
            target: format!("{target_psnr_db:.3} dB"),
            best: format!("{final_psnr:.3} dB"),
        });
    }
    Ok(LossyAttempt {
        codestream,
        quantization_scale: best_scale,
    })
}

pub(super) fn lossy_quality_layer_byte_targets(
    samples: J2kLossySamples<'_>,
    options: &J2kLossyEncodeOptions,
) -> Result<Vec<u64>, J2kError> {
    if options.quality_layers.len() <= 1 {
        return Ok(Vec::new());
    }

    let mut targets = super::allocation::try_vec(
        options.quality_layers.len(),
        "J2K lossy quality-layer targets",
    )?;
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
    if options.qfactor.is_some() && options.quantization_profile.is_some() {
        return Err(J2kError::Unsupported(Unsupported {
            what: "OpenHTJ2K Qfactor cannot be combined with a quantization profile",
        }));
    }
    if let Some(qfactor) = options.qfactor {
        if !(1..=100).contains(&qfactor) {
            return Err(J2kError::Unsupported(Unsupported {
                what: "OpenHTJ2K Qfactor must be in 1..=100",
            }));
        }
        if options.block_coding_mode != J2kBlockCodingMode::HighThroughput {
            return Err(J2kError::Unsupported(Unsupported {
                what: "OpenHTJ2K Qfactor requires high-throughput block coding",
            }));
        }
        if options.rate_target.is_some() || !options.quality_layers.is_empty() {
            return Err(J2kError::Unsupported(Unsupported {
                what:
                    "OpenHTJ2K Qfactor cannot be combined with lossy rate or quality-layer targets",
            }));
        }
    }
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

// `u64::MAX as f64` also rounds to 2^64, so this conversion bound must stay exclusive.
const U64_EXCLUSIVE_UPPER_BOUND_AS_F64: f64 = 18_446_744_073_709_551_616.0;

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    reason = "finite positive byte targets are checked below the exclusive 2^64 bound before conversion; a truncated target with a fraction is below 2^53"
)]
pub(super) fn target_bytes_for_bpp(
    samples: J2kLossySamples<'_>,
    bits_per_pixel: f64,
) -> Result<u64, J2kError> {
    let pixels = f64::from(samples.width) * f64::from(samples.height);
    let bytes = pixels * bits_per_pixel / 8.0;
    if bytes.is_finite() && bytes > 0.0 && bytes < U64_EXCLUSIVE_UPPER_BOUND_AS_F64 {
        // `bytes` rounded up without `f64::ceil`, a C library call on baseline
        // x86-64 (see `cargo xtask x86-codegen-audit`): the cast truncates, and
        // a target with a fraction is below 2^53, where the truncation is
        // exact. Every `f64` from 2^52 up is whole, so testing `bytes` rather
        // than its ceiling against the bounds accepts the same targets.
        let whole = bytes as u64;
        Ok(if (whole as f64) < bytes {
            whole + 1
        } else {
            whole
        })
    } else {
        Err(J2kError::Unsupported(Unsupported {
            what: "JPEG 2000 lossy bits-per-pixel target overflows byte target",
        }))
    }
}

pub(super) fn byte_target_tolerance(target_bytes: u64) -> u64 {
    target_bytes.div_ceil(100).max(512)
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

#[expect(
    clippy::cast_possible_truncation,
    reason = "search scales are clamped to the f32 admissible interval after narrowing"
)]
fn f64_to_f32(value: f64) -> f32 {
    value as f32
}

#[cfg(test)]
mod tests;
