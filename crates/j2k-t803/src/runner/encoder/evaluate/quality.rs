// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fmt::Write as _;

use j2k_compare::openjpeg::OpenJpegDecodedImage;

use crate::compare::u64_as_f64;
use crate::EncoderQualityStatus;

use super::super::input::GeneratedInput;
use crate::encoder::{EncoderCase, EncoderRateTarget};

#[derive(Clone, Copy)]
pub(super) struct EncodedMetrics {
    pub(super) bytes: u64,
    pub(super) bits_per_pixel: f64,
}

pub(super) fn encoded_metrics(
    case: &EncoderCase,
    encoded_bytes: usize,
) -> Result<EncodedMetrics, String> {
    let bytes = u64::try_from(encoded_bytes)
        .map_err(|_| "encoded codestream size exceeds the report range".to_string())?;
    let bytes_as_f64 = u64_as_f64(bytes).map_err(|error| error.to_string())?;
    let pixel_count = f64::from(case.width) * f64::from(case.height);
    Ok(EncodedMetrics {
        bytes,
        bits_per_pixel: bytes_as_f64 * 8.0 / pixel_count,
    })
}

#[derive(Clone, Copy)]
pub(super) struct Psnr {
    pub(super) db: Option<f64>,
    pub(super) infinite: bool,
}

pub(super) fn decoded_psnr(expected: &GeneratedInput, actual: &OpenJpegDecodedImage) -> Psnr {
    let mut squared_error = 0.0;
    let mut samples = 0.0_f64;
    let mut peak = 0.0_f64;
    for (expected, actual) in expected.components.iter().zip(&actual.components) {
        peak = peak.max(2_f64.powi(i32::from(expected.bit_depth)) - 1.0);
        for (&expected, &actual) in expected.samples.iter().zip(&actual.samples) {
            let error = f64::from(expected) - f64::from(actual);
            squared_error += error * error;
            samples += 1.0;
        }
    }
    if squared_error == 0.0 {
        return Psnr {
            db: None,
            infinite: true,
        };
    }
    let mse = squared_error / samples;
    Psnr {
        db: Some(10.0 * (peak * peak / mse).log10()),
        infinite: false,
    }
}

pub(super) struct QualityResult {
    pub(super) status: EncoderQualityStatus,
    pub(super) requirement: String,
    pub(super) error: Option<String>,
}

pub(super) fn evaluate_quality(
    case: &EncoderCase,
    psnr: Psnr,
    metrics: EncodedMetrics,
) -> QualityResult {
    let requirement = quality_requirement(case);
    let minimum_psnr = case
        .minimum_psnr_db
        .expect("validated lossy case has minimum PSNR");
    let mut failures = Vec::new();
    if psnr.db.is_some_and(|psnr_db| psnr_db < minimum_psnr) {
        let psnr_db = psnr.db.expect("finite PSNR was compared");
        failures.push(format!(
            "PSNR {psnr_db:.6} dB is below {minimum_psnr:.6} dB"
        ));
    }
    if let Some((target, overshoot)) = rate_gate(case) {
        match target {
            EncoderRateTarget::BitsPerPixel(target) => {
                let actual = metrics.bits_per_pixel;
                let one_byte = 8.0 / (f64::from(case.width) * f64::from(case.height));
                let maximum = target * (1.0 + overshoot / 100.0) + one_byte;
                if actual > maximum {
                    failures.push(format!("rate {actual:.6} bpp exceeds {maximum:.6} bpp"));
                }
            }
            EncoderRateTarget::Bytes(target) => {
                let encoded_bytes = metrics.bytes;
                match (u64_as_f64(target), u64_as_f64(encoded_bytes)) {
                    (Ok(target), Ok(actual)) => {
                        let maximum = target * (1.0 + overshoot / 100.0) + 1.0;
                        if actual > maximum {
                            failures.push(format!(
                                "codestream {encoded_bytes} bytes exceeds {maximum:.0} bytes"
                            ));
                        }
                    }
                    _ => failures.push("rate gate numeric conversion failed".to_string()),
                }
            }
            EncoderRateTarget::PsnrDb(_) => {}
        }
    }
    if failures.is_empty() {
        QualityResult {
            status: EncoderQualityStatus::Pass,
            requirement,
            error: None,
        }
    } else {
        QualityResult {
            status: EncoderQualityStatus::Fail,
            requirement,
            error: Some(failures.join("; ")),
        }
    }
}

pub(super) fn quality_requirement(case: &EncoderCase) -> String {
    let minimum_psnr = case.minimum_psnr_db.unwrap_or_default();
    let mut requirement = format!("PSNR >= {minimum_psnr:.6} dB");
    if let Some((target, overshoot)) = rate_gate(case) {
        let rate = match target {
            EncoderRateTarget::BitsPerPixel(value) => format!("{value:.6} bpp"),
            EncoderRateTarget::Bytes(value) => format!("{value} bytes"),
            EncoderRateTarget::PsnrDb(_) => return requirement,
        };
        let _ = write!(
            requirement,
            "; rate <= {rate} + {overshoot:.6}% + one-byte rounding"
        );
    }
    requirement
}

fn rate_gate(case: &EncoderCase) -> Option<(EncoderRateTarget, f64)> {
    let target = case
        .lossy_quality_layers
        .last()
        .copied()
        .or(case.lossy_rate_target)?;
    let overshoot = case.maximum_rate_overshoot_percent?;
    Some((target, overshoot))
}
