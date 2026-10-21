//! Forward quantization for JPEG 2000 encoding.
//!
//! - Lossless (reversible 5-3): No quantization, just sign/magnitude conversion
//! - Lossy (irreversible 9-7): Scalar deadzone quantization with step sizes
//!   derived from the DWT subband gain norms.

use alloc::vec;
use alloc::vec::Vec;

use crate::math::{floor_f32, log2_f32, pow2i, round_f32};
use crate::{
    EncodeError, EncodeResult, IrreversibleQuantizationStep, IrreversibleQuantizationSubbandScales,
    J2kSubBandType,
};

/// Largest exponent the 5-bit irreversible QCD/QCC step field stores.
const MAX_STEP_EXPONENT: u16 = 31;

/// Quantization parameters for a single subband.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct QuantStepSize {
    pub(crate) exponent: u16,
    pub(crate) mantissa: u16,
}

pub(crate) fn subband_scales_all_valid(scales: IrreversibleQuantizationSubbandScales) -> bool {
    [
        scales.low_low,
        scales.high_low,
        scales.low_high,
        scales.high_high,
    ]
    .iter()
    .all(|scale| scale.is_finite() && *scale > 0.0)
}

fn subband_scale_for_step_index(
    scales: IrreversibleQuantizationSubbandScales,
    index: usize,
) -> f32 {
    if index == 0 {
        return scales.low_low;
    }
    match (index - 1) % 3 {
        0 => scales.high_low,
        1 => scales.low_high,
        _ => scales.high_high,
    }
}

impl QuantStepSize {
    /// Compute the JPEG 2000 irreversible step size:
    /// `Δ = 2^(R_b - exponent) × (1 + mantissa / 2048)`.
    #[expect(
        clippy::trivially_copy_pass_by_ref,
        reason = "the stable codec boundary borrows shared Copy metadata used across nested calls"
    )]
    fn delta(&self, range_bits: u8) -> f32 {
        let rb = i32::from(range_bits) - i32::from(self.exponent);
        let base = pow2i(rb);
        base * (1.0 + f32::from(self.mantissa) / 2048.0)
    }

    /// Round a step to its stored exponent/mantissa pair, or `None` when the
    /// rounded pair needs an exponent outside the 5-bit QCD/QCC field.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the floored log2 of a finite positive f32 lies within -150..=127"
    )]
    fn try_from_delta(range_bits: u8, delta: f32) -> Option<Self> {
        if !delta.is_finite() || delta <= 0.0 {
            return None;
        }
        let floor_log2 = floor_f32(log2_f32(delta)) as i32;
        let exponent = i32::from(range_bits) - floor_log2;
        // A mantissa carry lowers the exponent by one, so 32 can still round
        // into range. Bit depths above 31 store steps of 2^32 and coarser, so
        // the exponent is the only range check.
        if !(0..=i32::from(MAX_STEP_EXPONENT) + 1).contains(&exponent) {
            return None;
        }
        let normalized = delta / pow2i(floor_log2);
        let mantissa = round_f32((normalized - 1.0) * 2048.0) as i32;
        let (exponent, mantissa) = if mantissa >= 2048 {
            (exponent - 1, 0)
        } else {
            (exponent, mantissa.max(0))
        };
        Some(Self {
            exponent: u16::try_from(exponent)
                .ok()
                .filter(|&exponent| exponent <= MAX_STEP_EXPONENT)?,
            mantissa: u16::try_from(mantissa).ok()?,
        })
    }

    /// Round a step to its stored pair, saturating at the coarsest or finest
    /// step the 5-bit exponent can store.
    fn from_delta(range_bits: u8, delta: f32) -> Self {
        Self::try_from_delta(range_bits, delta).unwrap_or_else(|| {
            if log2_f32(delta) >= f32::from(range_bits) {
                Self {
                    exponent: 0,
                    mantissa: 2047,
                }
            } else {
                Self {
                    exponent: MAX_STEP_EXPONENT,
                    mantissa: 0,
                }
            }
        })
    }
}

/// Unrounded irreversible step for QCD/QCC position `index`, including the
/// band's synthesis-norm weight and the validated component weight. Invalid
/// global/orientation multipliers are treated as one for the compatibility
/// step-query API.
#[expect(
    clippy::cast_possible_truncation,
    reason = "narrow the complete product once at the existing f32 quantization boundary"
)]
fn irreversible_step_delta(
    bit_depth: u8,
    guard_bits: u8,
    irreversible_quantization_scale: f32,
    irreversible_quantization_subband_scales: IrreversibleQuantizationSubbandScales,
    index: usize,
    norm_weight: f32,
    weight: f32,
) -> f32 {
    let base_step = QuantStepSize {
        exponent: u16::from(bit_depth) + u16::from(guard_bits),
        mantissa: 0,
    };
    let scale =
        if irreversible_quantization_scale.is_finite() && irreversible_quantization_scale > 0.0 {
            irreversible_quantization_scale
        } else {
            1.0
        };
    let subband_scales = if subband_scales_all_valid(irreversible_quantization_subband_scales) {
        irreversible_quantization_subband_scales
    } else {
        IrreversibleQuantizationSubbandScales::default()
    };
    let base = base_step.delta(bit_depth);
    let orientation = subband_scale_for_step_index(subband_scales, index);
    let scaled = base * scale;
    let oriented = scaled * orientation;
    // A unit norm weight leaves `oriented` exact, preserving earlier steps.
    let normed = oriented * norm_weight;
    let weighted = normed * weight;
    if scaled.is_normal() && oriented.is_normal() && normed.is_normal() && weighted.is_normal() {
        // Preserve the established f32 evaluation for ordinary inputs.
        weighted
    } else {
        // Five positive f32 factors fit in f64, including a subnormal scale
        // rescued by another factor. Narrow only after the complete product.
        (libm::scalbn(1.0, -i32::from(guard_bits))
            * f64::from(scale)
            * f64::from(orientation)
            * f64::from(norm_weight)
            * f64::from(weight)) as f32
    }
}

/// Compute the exact irreversible 9/7 quantization step tuple the native encoder
/// writes for one subband at `level` (1 is the finest; LL lies at `levels`) of a
/// `levels`-level decomposition, under a global plus per-subband profile.
///
/// Invalid scale arguments are treated as one; valid positive products
/// saturate monotonically at the complete stored-step endpoints.
#[must_use]
pub fn irreversible_quantization_step_for_subband(
    bit_depth: u8,
    guard_bits: u8,
    irreversible_quantization_scale: f32,
    irreversible_quantization_subband_scales: IrreversibleQuantizationSubbandScales,
    subband: J2kSubBandType,
    level: u8,
    levels: u8,
) -> IrreversibleQuantizationStep {
    // Only the orientation is read from this position.
    let index = match subband {
        J2kSubBandType::LowLow => 0,
        J2kSubBandType::HighLow => 1,
        J2kSubBandType::LowHigh => 2,
        J2kSubBandType::HighHigh => 3,
    };
    let step_size = QuantStepSize::from_delta(
        bit_depth,
        irreversible_step_delta(
            bit_depth,
            guard_bits,
            irreversible_quantization_scale,
            irreversible_quantization_subband_scales,
            index,
            synthesis_norm_weight(subband, level, levels),
            1.0,
        ),
    );
    IrreversibleQuantizationStep {
        exponent: u8::try_from(step_size.exponent).unwrap_or(u8::MAX),
        mantissa: step_size.mantissa,
    }
}

/// Compute default quantization step sizes for the irreversible 9-7 transform.
///
/// The step sizes are derived from the DWT 9-7 subband gain norms (Table E.1 in T.800).
/// For lossless mode, step sizes are not used (exponents store bit depth info only).
#[cfg(test)]
pub(crate) fn compute_step_sizes(
    bit_depth: u8,
    num_decompositions: u8,
    reversible: bool,
    guard_bits: u8,
) -> Vec<QuantStepSize> {
    compute_step_sizes_with_irreversible_scale(
        bit_depth,
        num_decompositions,
        reversible,
        guard_bits,
        1.0,
    )
}

/// Compute quantization step sizes with an irreversible 9-7 scale multiplier.
///
/// A scale of 1.0 preserves the quality-first default. Larger scales coarsen
/// the irreversible quantizer while keeping the same subband gain relationship.
#[cfg(test)]
pub(crate) fn compute_step_sizes_with_irreversible_scale(
    bit_depth: u8,
    num_decompositions: u8,
    reversible: bool,
    guard_bits: u8,
    irreversible_quantization_scale: f32,
) -> Vec<QuantStepSize> {
    compute_step_sizes_with_irreversible_profile(
        bit_depth,
        num_decompositions,
        reversible,
        guard_bits,
        irreversible_quantization_scale,
        IrreversibleQuantizationSubbandScales::default(),
    )
}

/// Compute quantization step sizes with global and per-subband irreversible
/// 9/7 scale multipliers.
#[cfg(test)]
pub(crate) fn compute_step_sizes_with_irreversible_profile(
    bit_depth: u8,
    num_decompositions: u8,
    reversible: bool,
    guard_bits: u8,
    irreversible_quantization_scale: f32,
    irreversible_quantization_subband_scales: IrreversibleQuantizationSubbandScales,
) -> Vec<QuantStepSize> {
    let mut step_sizes = Vec::new();
    append_step_sizes_with_irreversible_profile(
        &mut step_sizes,
        bit_depth,
        num_decompositions,
        reversible,
        guard_bits,
        irreversible_quantization_scale,
        irreversible_quantization_subband_scales,
    );
    step_sizes
}

pub(crate) fn append_step_sizes_with_irreversible_profile(
    step_sizes: &mut Vec<QuantStepSize>,
    bit_depth: u8,
    num_decompositions: u8,
    reversible: bool,
    guard_bits: u8,
    irreversible_quantization_scale: f32,
    irreversible_quantization_subband_scales: IrreversibleQuantizationSubbandScales,
) {
    if reversible {
        // For reversible 5-3, QCD stores the subband exponent only.
        // The decoder reconstructs the number of bitplanes as:
        //   Mb = guard_bits + exponent - 1
        // For lossless coding we therefore need exponents that reproduce the
        // reversible subband dynamic range:
        //   LL => bit_depth + 0
        //   HL/LH => bit_depth + 1
        //   HH => bit_depth + 2
        // This gain depends on subband orientation, not decomposition level.
        step_sizes.push(QuantStepSize {
            exponent: u16::from(bit_depth),
            mantissa: 0,
        });

        for _ in 0..num_decompositions {
            step_sizes.push(QuantStepSize {
                exponent: u16::from(bit_depth) + 1,
                mantissa: 0,
            });
            step_sizes.push(QuantStepSize {
                exponent: u16::from(bit_depth) + 1,
                mantissa: 0,
            });
            step_sizes.push(QuantStepSize {
                exponent: u16::from(bit_depth) + 2,
                mantissa: 0,
            });
        }
    } else {
        // Irreversible 9-7 default: each band's step shrinks with its synthesis
        // norm, and R_b = bit_depth + log_gain still makes LL finest and HH
        // coarsest within a level under the decoder's QCD formula.
        let step_count = 1usize + usize::from(num_decompositions) * 3;

        for index in 0..step_count {
            step_sizes.push(QuantStepSize::from_delta(
                bit_depth,
                irreversible_step_delta(
                    bit_depth,
                    guard_bits,
                    irreversible_quantization_scale,
                    irreversible_quantization_subband_scales,
                    index,
                    qcd_synthesis_norm_weight(index, num_decompositions),
                    1.0,
                ),
            ));
        }
    }
}

/// Decomposition levels implied by a QCD/QCC step or profile row length.
fn levels_for_band_count(bands: usize) -> EncodeResult<u8> {
    u8::try_from(bands.saturating_sub(1) / 3).map_err(|_| EncodeError::InvalidInput {
        what: "quantization profile band count exceeds the supported decomposition levels",
    })
}

/// Return whether every profile multiplier is exactly one, which keeps the
/// original quantization markers.
pub(crate) fn profile_is_unity(profile: &[Vec<f32>]) -> bool {
    profile
        .iter()
        .flatten()
        .all(|weight| weight.to_bits() == 1.0_f32.to_bits())
}

/// Replace one component's steps with its profile-weighted steps.
///
/// Weights multiply the unrounded step, and every band of a non-unity profile
/// must be representable. A tiny or huge weight, or a global scale that would
/// clamp some steps and not others, is rejected rather than silently changing
/// the profile's ratios.
pub(crate) fn apply_component_profile(
    steps: &mut [QuantStepSize],
    bit_depth: u8,
    guard_bits: u8,
    irreversible_quantization_scale: f32,
    irreversible_quantization_subband_scales: IrreversibleQuantizationSubbandScales,
    profile: Option<&[Vec<f32>]>,
    component: usize,
) -> EncodeResult<()> {
    let Some(profile) = profile else {
        return Ok(());
    };
    let weights = profile.get(component).ok_or(EncodeError::InvalidInput {
        what: "quantization profile component count does not match input",
    })?;
    if weights.len() != steps.len() {
        return Err(EncodeError::InvalidInput {
            what: "quantization profile band count must match actual decomposition levels",
        });
    }
    if profile_is_unity(profile) {
        return Ok(());
    }
    let levels = levels_for_band_count(steps.len())?;
    for (index, (step, &weight)) in steps.iter_mut().zip(weights).enumerate() {
        let delta = irreversible_step_delta(
            bit_depth,
            guard_bits,
            irreversible_quantization_scale,
            irreversible_quantization_subband_scales,
            index,
            qcd_synthesis_norm_weight(index, levels),
            weight,
        );
        *step =
            QuantStepSize::try_from_delta(bit_depth, delta).ok_or(EncodeError::InvalidInput {
                what: "quantization profile step is outside the representable QCD/QCC range",
            })?;
    }
    Ok(())
}

/// First and last positive finite f32 scales whose rounded weighted steps
/// fit both QCD/QCC and the coding mode's exponent limit. An inverted interval
/// means the bands have no common
/// admissible scale. With no active profile, include the saturated default
/// steps of a `levels`-level decomposition at scale one and stop once every
/// band has reached its coarsest step. Profile rows carry their own band count.
#[expect(
    clippy::cast_possible_truncation,
    reason = "rate-search scales are f32; the widened saturation calculation is bounded before narrowing"
)]
pub(crate) fn quantization_scale_bounds(
    bit_depth: u8,
    guard_bits: u8,
    subband_scales: IrreversibleQuantizationSubbandScales,
    profile: Option<&[Vec<f32>]>,
    levels: u8,
    maximum_exponent: u16,
) -> (f32, f32) {
    let profile = profile.filter(|profile| !profile_is_unity(profile));
    let finest = pow2i(i32::from(bit_depth) - i32::from(maximum_exponent));
    let default_bands = 1 + 3 * usize::from(levels);
    let within_bound = |scale, lower: bool| {
        let accepts = |index, levels, weight| {
            let delta = irreversible_step_delta(
                bit_depth,
                guard_bits,
                scale,
                subband_scales,
                index,
                qcd_synthesis_norm_weight(index, levels),
                weight,
            );
            QuantStepSize::try_from_delta(bit_depth, delta)
                .is_some_and(|step| step.exponent <= maximum_exponent)
                || if lower {
                    delta >= finest
                } else {
                    delta < finest
                }
        };
        profile.map_or_else(
            || (0..default_bands).all(|index| accepts(index, levels, 1.0)),
            |profile| {
                profile.iter().all(|row| {
                    let row_levels = levels_for_band_count(row.len()).unwrap_or(u8::MAX);
                    row.iter()
                        .enumerate()
                        .all(|(index, &weight)| accepts(index, row_levels, weight))
                })
            },
        )
    };
    // Positive f32 bit patterns are ordered by value. Bisect that finite set,
    // using zero/infinity as excluded sentinels, without exponential jumps.
    let boundary = |lower| {
        let mut low = 0;
        let mut high = f32::INFINITY.to_bits();
        while high - low > 1 {
            let middle = low + (high - low) / 2;
            if within_bound(f32::from_bits(middle), lower) == lower {
                high = middle;
            } else {
                low = middle;
            }
        }
        f32::from_bits(if lower { high } else { low })
    };
    let lower = boundary(true);
    let upper = boundary(false);
    if profile.is_some() {
        (lower, upper)
    } else {
        // Global scaling saturates independently in each band. Its last
        // changing band, not the first to saturate, determines search's end.
        let coarsest = pow2i(i32::from(bit_depth)) * (1.0 + 2047.0 / 2048.0);
        let smallest_factor = (0..default_bands)
            .map(|index| {
                f64::from(subband_scale_for_step_index(subband_scales, index))
                    * f64::from(qcd_synthesis_norm_weight(index, levels))
            })
            .fold(f64::MAX, f64::min);
        let saturation = (f64::from(coarsest) * libm::scalbn(1.0, i32::from(guard_bits))
            / smallest_factor)
            .min(f64::from(f32::MAX)) as f32;
        (1.0, saturation.max(1.0))
    }
}

/// Compatibility helper: an invalid starting scale is left for encode to reject.
pub(crate) fn profile_scale_ceiling(
    bit_depth: u8,
    guard_bits: u8,
    start: f32,
    subband_scales: IrreversibleQuantizationSubbandScales,
    profile: &[Vec<f32>],
) -> f32 {
    let levels = levels_for_band_count(profile.first().map_or(1, Vec::len)).unwrap_or(u8::MAX);
    let (lower, upper) = quantization_scale_bounds(
        bit_depth,
        guard_bits,
        subband_scales,
        Some(profile),
        levels,
        MAX_STEP_EXPONENT,
    );
    if start >= lower && start <= upper {
        upper
    } else {
        start
    }
}

/// One-dimensional CDF 9/7 synthesis energy gains `(lowpass, highpass)` for
/// decomposition levels 1 to 16, finest first. A 2-D band's squared synthesis
/// norm is the product of its two directional gains.
const D97_ENERGY_GAINS: [(f64, f64); 16] = [
    (1.965_907_314_575_295_7, 2.080_871_927_589_849),
    (4.122_409_873_969_023, 3.868_863_224_131_922),
    (8.416_744_177_952_724, 8.317_022_299_806_517),
    (16.935_572_073_021_724, 17.201_929_112_787_134),
    (33.924_926_802_207_46, 34.746_895_711_342_454),
    (67.877_165_259_517_36, 69.675_395_886_752_16),
    (135.768_047_117_209_76, 139.443_143_900_563_17),
    (271.542_960_998_980_6, 278.932_688_221_656_86),
    (543.089_356_530_109_5, 557.888_607_932_094_2),
    (1_086.180_430_479_691_7, 1_115.788_835_859_533_3),
    (2_172.361_719_689_337, 2_231.583_482_284_095_7),
    (4_344.723_868_746_226, 4_463.169_869_925_348),
    (8_689.447_952_176_557, 8_926.341_192_539_1),
    (17_378.896_011_694_746, 17_852.683_111_423_37),
    (34_757.792_077_060_57, 35_705.366_586_020_52),
    (69_515.584_180_957_42, 71_410.733_353_619_97),
];

/// L2 norm of the 2-D 9/7 synthesis basis of `subband` at `level`, where
/// level 1 is the finest, LL lies at the deepest level, and level 0 is an
/// untransformed component.
fn d97_synthesis_norm(subband: J2kSubBandType, level: u8) -> f64 {
    if level == 0 {
        return 1.0;
    }
    let tabulated = usize::from(level).min(D97_ENERGY_GAINS.len());
    let (low, high) = D97_ENERGY_GAINS[tabulated - 1];
    let energy = match subband {
        J2kSubBandType::LowLow => low * low,
        J2kSubBandType::HighLow | J2kSubBandType::LowHigh => low * high,
        J2kSubBandType::HighHigh => high * high,
    };
    // Each directional gain doubles per level beyond the table, so the norm does too.
    let extra_levels = usize::from(level) - tabulated;
    libm::scalbn(
        libm::sqrt(energy),
        i32::try_from(extra_levels).expect("at most 255 levels beyond the table"),
    )
}

/// Subband and level of QCD/QCC position `index` in a `levels`-level
/// decomposition: LL, then HL, LH and HH from the coarsest level to the finest.
fn qcd_band(index: usize, levels: u8) -> (J2kSubBandType, u8) {
    if index == 0 {
        return (J2kSubBandType::LowLow, levels);
    }
    let subband = match (index - 1) % 3 {
        0 => J2kSubBandType::HighLow,
        1 => J2kSubBandType::LowHigh,
        _ => J2kSubBandType::HighHigh,
    };
    let finer_levels = u8::try_from((index - 1) / 3).unwrap_or(u8::MAX);
    (subband, levels.saturating_sub(finer_levels))
}

/// Rate-distortion weight of a default irreversible step: steps shrink in
/// proportion to each band's 9/7 synthesis norm, as standard JPEG 2000 encoders
/// signal them. The band with the smallest norm (the finest HL/LH, or LL of a
/// one-level decomposition) keeps the unweighted step, so no band is quantized
/// more coarsely than with orientation-only steps.
#[expect(
    clippy::cast_possible_truncation,
    reason = "weights lie in (0, 1] and narrow once to the f32 step arithmetic"
)]
fn synthesis_norm_weight(subband: J2kSubBandType, level: u8, levels: u8) -> f32 {
    let smallest = if levels == 0 {
        1.0
    } else {
        d97_synthesis_norm(J2kSubBandType::LowLow, levels)
            .min(d97_synthesis_norm(J2kSubBandType::HighLow, 1))
    };
    (smallest / d97_synthesis_norm(subband, level)) as f32
}

fn qcd_synthesis_norm_weight(index: usize, levels: u8) -> f32 {
    let (subband, level) = qcd_band(index, levels);
    synthesis_norm_weight(subband, level, levels)
}

const OPENHTJ2K_VISUAL_WEIGHTS_444: [[f64; 15]; 3] = [
    [
        0.0901, 0.2758, 0.2758, 0.7018, 0.8378, 0.8378, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    ],
    [
        0.0263, 0.0863, 0.0863, 0.1362, 0.2564, 0.2564, 0.3346, 0.4691, 0.4691, 0.5444, 0.6523,
        0.6523, 0.7078, 0.7797, 0.7797,
    ],
    [
        0.0773, 0.1835, 0.1835, 0.2598, 0.4130, 0.4130, 0.5040, 0.6464, 0.6464, 0.7220, 0.8254,
        0.8254, 0.8769, 0.9424, 0.9424,
    ],
];

const OPENHTJ2K_COLOR_GAINS: [f64; 3] = [1.7321, 1.8051, 1.5734];

/// Append the expounded 9/7 QCD or QCC tuples used by `OpenHTJ2K`'s Qfactor
/// profile for grayscale or 4:4:4 YCbCr content.
pub(crate) fn append_openhtj2k_qfactor_step_sizes(
    step_sizes: &mut Vec<QuantStepSize>,
    bit_depth: u8,
    num_decompositions: u8,
    qfactor: u8,
    component: usize,
) -> EncodeResult<()> {
    if !(1..=100).contains(&qfactor) {
        return Err(EncodeError::InvalidInput {
            what: "OpenHTJ2K Qfactor must be in 1..=100",
        });
    }
    let Some(visual_weights) = OPENHTJ2K_VISUAL_WEIGHTS_444.get(component) else {
        return Err(EncodeError::InvalidInput {
            what: "OpenHTJ2K Qfactor supports grayscale or three-component 4:4:4 input",
        });
    };
    let level_count = usize::from(num_decompositions);
    if level_count > D97_ENERGY_GAINS.len() {
        return Err(EncodeError::Unsupported {
            what: "OpenHTJ2K Qfactor supports at most 16 decomposition levels",
        });
    }

    let qfactor = f64::from(qfactor);
    let magnitude = if qfactor < 50.0 {
        50.0 / qfactor
    } else {
        2.0 * (1.0 - qfactor / 100.0)
    };
    let mut power = 1.0;
    let mut alpha = 0.04;
    if qfactor >= 97.0 {
        power = 0.0;
        alpha = 0.10;
    } else if qfactor > 65.0 {
        let magnitude_t0 = 2.0 * (1.0 - 65.0 / 100.0);
        let magnitude_t1 = 2.0 * (1.0 - 97.0 / 100.0);
        power = (libm::log(magnitude_t1) - libm::log(magnitude))
            / (libm::log(magnitude_t1) - libm::log(magnitude_t0));
        alpha = 0.10 * libm::pow(0.04 / 0.10, power);
    }

    let epsilon = libm::sqrt(0.5) / libm::scalbn(1.0, i32::from(bit_depth));
    let delta_reference = alpha * magnitude * OPENHTJ2K_COLOR_GAINS[0] + epsilon;
    let component_gain = OPENHTJ2K_COLOR_GAINS[component];
    let start = step_sizes.len();
    for (level, &(low, high)) in D97_ENERGY_GAINS[..level_count].iter().enumerate() {
        let base = level * 3;
        for (offset, wmse) in [high * high, low * high, high * low]
            .into_iter()
            .enumerate()
        {
            let weight = visual_weights.get(base + offset).copied().unwrap_or(1.0);
            let delta =
                delta_reference / (libm::sqrt(wmse) * libm::pow(weight, power) * component_gain);
            step_sizes.push(openhtj2k_step_from_normalized_delta(delta));
        }
    }
    let low_energy = if level_count == 0 {
        1.0
    } else {
        let low = D97_ENERGY_GAINS[level_count - 1].0;
        low * low
    };
    let low_delta = delta_reference / (libm::sqrt(low_energy) * component_gain);
    step_sizes.push(openhtj2k_step_from_normalized_delta(low_delta));
    step_sizes[start..].reverse();
    Ok(())
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "the OpenHTJ2K Qfactor conversion clamps exponent and mantissa to their marker widths"
)]
fn openhtj2k_step_from_normalized_delta(mut delta: f64) -> QuantStepSize {
    let mut exponent = 0_i32;
    // The marker exponent saturates at 31, so any additional normalization
    // iterations cannot affect the encoded result.
    for _ in 0..=31 {
        if delta >= 1.0 {
            break;
        }
        delta *= 2.0;
        exponent += 1;
    }
    let mut mantissa = libm::floor((delta - 1.0) * 2048.0 + 0.5) as i32;
    if mantissa >= 2048 {
        mantissa = 0;
        exponent -= 1;
    }
    if exponent > 31 {
        exponent = 31;
        mantissa = 0;
    } else if exponent < 0 {
        exponent = 0;
        mantissa = 2047;
    }
    QuantStepSize {
        exponent: u16::try_from(exponent).expect("Qfactor exponent was clamped to 0..=31"),
        mantissa: u16::try_from(mantissa).expect("Qfactor mantissa was clamped to 0..=2047"),
    }
}

/// Quantize wavelet coefficients for a single subband.
///
/// For lossless: converts f32 to i32 (round to nearest integer).
/// For lossy: applies scalar deadzone quantization.
///
/// Returns (magnitude, sign) pairs packed as i32 values.
#[expect(
    clippy::trivially_copy_pass_by_ref,
    reason = "the stable codec boundary borrows shared Copy metadata used across nested calls"
)]
#[expect(
    clippy::cast_possible_truncation,
    reason = "finite wavelet coefficients are intentionally rounded at the codec quantization boundary"
)]
#[allow(
    clippy::disallowed_macros,
    reason = "the infallible scalar parity API preserves its established Vec-returning contract"
)]
pub(crate) fn quantize_subband(
    coefficients: &[f32],
    step_size: &QuantStepSize,
    range_bits: u8,
    reversible: bool,
) -> Vec<i32> {
    if reversible {
        // No quantization: round to nearest integer
        coefficients.iter().map(|&c| round_f32(c) as i32).collect()
    } else {
        let delta = step_size.delta(range_bits);
        if delta <= 0.0 {
            return vec![0i32; coefficients.len()];
        }
        coefficients
            .iter()
            .map(|&c| {
                // Deadzone quantization: q = sign(c) * floor(|c| / Δ)
                let sign = if c < 0.0 { -1 } else { 1 };
                let magnitude = floor_f32(c.abs() / delta) as i32;
                sign * magnitude
            })
            .collect()
    }
}

/// Fallible counterpart used by the allocation-bounded native encode path.
#[expect(
    clippy::trivially_copy_pass_by_ref,
    reason = "the stable codec boundary borrows shared Copy metadata used across nested calls"
)]
#[expect(
    clippy::cast_possible_truncation,
    reason = "finite wavelet coefficients are intentionally rounded at the codec quantization boundary"
)]
pub(crate) fn try_quantize_subband(
    coefficients: &[f32],
    step_size: &QuantStepSize,
    range_bits: u8,
    reversible: bool,
) -> EncodeResult<Vec<i32>> {
    let requested_bytes = coefficients
        .len()
        .checked_mul(core::mem::size_of::<i32>())
        .ok_or(EncodeError::ArithmeticOverflow {
            what: "quantized subband coefficients",
        })?;
    let mut quantized = Vec::new();
    quantized
        .try_reserve_exact(coefficients.len())
        .map_err(|_| EncodeError::HostAllocationFailed {
            what: "quantized subband coefficients",
            bytes: requested_bytes,
        })?;
    if reversible {
        for &coefficient in coefficients {
            quantized.push(round_f32(coefficient) as i32);
        }
        return Ok(quantized);
    }

    let delta = step_size.delta(range_bits);
    if delta <= 0.0 {
        quantized.resize(coefficients.len(), 0);
        return Ok(quantized);
    }
    for &coefficient in coefficients {
        let sign = if coefficient < 0.0 { -1 } else { 1 };
        let magnitude = floor_f32(coefficient.abs() / delta) as i32;
        quantized.push(sign * magnitude);
    }
    Ok(quantized)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn openhtj2k_qfactor_90_matches_reference_qcd_and_qcc() {
        let expected = [
            [
                0x65df, 0x65b4, 0x65b4, 0x658b, 0x5dc9, 0x5dc9, 0x5dad, 0x560f, 0x560f, 0x5625,
                0x4008, 0x4008, 0x410b, 0x3dab, 0x3dab, 0x337e,
            ],
            [
                0x654f, 0x66db, 0x66db, 0x6764, 0x5027, 0x5027, 0x50d7, 0x49c6, 0x49c6, 0x4b9b,
                0x45c4, 0x45c4, 0x39b0, 0x3396, 0x3396, 0x2a15,
            ],
            [
                0x6745, 0x6788, 0x6788, 0x67e6, 0x5056, 0x5056, 0x50d5, 0x4995, 0x4995, 0x4ae4,
                0x4481, 0x4481, 0x3819, 0x3130, 0x3130, 0x35a3,
            ],
        ];

        for (component, expected) in expected.iter().enumerate() {
            let mut actual = Vec::new();
            append_openhtj2k_qfactor_step_sizes(&mut actual, 8, 5, 90, component)
                .expect("valid OpenHTJ2K Qfactor profile");
            let actual = actual
                .iter()
                .map(|step| (step.exponent << 11) | step.mantissa)
                .collect::<Vec<_>>();
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn openhtj2k_normalized_step_conversion_saturates_at_marker_exponent() {
        assert_eq!(
            openhtj2k_step_from_normalized_delta(0.75),
            QuantStepSize {
                exponent: 1,
                mantissa: 1024,
            }
        );
        assert_eq!(
            openhtj2k_step_from_normalized_delta(f64::MIN_POSITIVE),
            QuantStepSize {
                exponent: 31,
                mantissa: 0,
            }
        );
    }

    #[test]
    fn test_lossless_quantize() {
        let coeffs = vec![10.0, -5.0, 3.7, -8.2, 0.0];
        let step = QuantStepSize {
            exponent: 12,
            mantissa: 0,
        };
        let result = quantize_subband(&coeffs, &step, 1, true);
        assert_eq!(result, vec![10, -5, 4, -8, 0]);
    }

    #[test]
    fn test_lossy_quantize() {
        let coeffs = vec![10.0, -5.0, 0.3, -0.1];
        let step = QuantStepSize {
            exponent: 8,
            mantissa: 0,
        };
        let delta = step.delta(8);
        assert!((delta - 1.0).abs() < 0.01);

        let result = quantize_subband(&coeffs, &step, 8, false);
        assert_eq!(result[0], 10);
        assert_eq!(result[1], -5);
        assert_eq!(result[2], 0); // Below deadzone
        assert_eq!(result[3], 0); // Below deadzone
    }

    #[test]
    fn irreversible_quantization_uses_direct_division_at_integer_boundary() {
        let coefficient = f32::from_bits(0x41fa_c8ff);
        let step = QuantStepSize {
            exponent: 8,
            mantissa: 23,
        };

        let result = quantize_subband(&[coefficient, -coefficient], &step, 8, false);

        assert_eq!(result, vec![30, -30]);
    }

    #[test]
    fn test_compute_step_sizes_reversible() {
        let steps = compute_step_sizes(8, 3, true, 1);
        // 1 LL + 3 levels × 3 subbands = 10
        assert_eq!(steps.len(), 10);
        // All mantissas should be 0 for reversible
        assert!(steps.iter().all(|s| s.mantissa == 0));
        let exponents: Vec<u16> = steps.iter().map(|s| s.exponent).collect();
        assert_eq!(exponents, vec![8, 9, 9, 10, 9, 9, 10, 9, 9, 10]);
    }

    #[test]
    fn test_compute_step_sizes_irreversible() {
        let steps = compute_step_sizes(8, 3, false, 1);
        assert_eq!(steps.len(), 10);
    }

    /// One-level 9/7 synthesis norms relative to LL, the smallest: HL/LH, HH.
    const ONE_LEVEL_DETAIL_WEIGHTS: [f32; 2] = [0.971_983, 0.944_752];

    /// Decoder deltas of a one-level LL, HL, LH, HH step set at `base`, the LL
    /// delta, checked to the 11-bit mantissa's relative precision.
    fn assert_one_level_deltas(steps: &[QuantStepSize], base: f32) {
        let [high_low, high_high] = ONE_LEVEL_DETAIL_WEIGHTS;
        let expected = [
            base,
            2.0 * base * high_low,
            2.0 * base * high_low,
            4.0 * base * high_high,
        ];
        for ((&range_bits, step), expected) in [8u8, 9, 9, 10].iter().zip(steps).zip(expected) {
            let delta = step.delta(range_bits);
            assert!(
                (delta / expected - 1.0).abs() < 1e-3,
                "{delta} vs {expected}"
            );
        }
    }

    #[test]
    fn irreversible_steps_match_decoder_qcd_contract() {
        let steps = compute_step_sizes(8, 1, false, 2);
        // LL has the smallest one-level norm and keeps the unweighted step.
        assert_eq!((steps[0].exponent, steps[0].mantissa), (10, 0));
        assert_one_level_deltas(&steps, 0.25);
    }

    #[test]
    fn irreversible_quantization_scale_coarsens_qcd_deltas() {
        let steps = compute_step_sizes_with_irreversible_scale(8, 1, false, 2, 4.0);
        assert_eq!((steps[0].exponent, steps[0].mantissa), (8, 0));
        assert_one_level_deltas(&steps, 1.0);
    }

    #[test]
    fn irreversible_quantization_scale_uses_mantissa_for_fractional_steps() {
        let steps = compute_step_sizes_with_irreversible_scale(8, 1, false, 2, 5.0);
        assert_eq!((steps[0].exponent, steps[0].mantissa), (8, 512));
        assert_one_level_deltas(&steps, 1.25);
    }

    #[test]
    fn irreversible_subband_scales_change_only_selected_97_steps() {
        let subband_scales = IrreversibleQuantizationSubbandScales {
            low_low: 1.0,
            high_low: 1.0,
            low_high: 1.0,
            high_high: 1.5,
        };

        let default_steps = compute_step_sizes_with_irreversible_profile(
            8,
            1,
            false,
            2,
            1.9,
            IrreversibleQuantizationSubbandScales::default(),
        );
        let shaped_steps =
            compute_step_sizes_with_irreversible_profile(8, 1, false, 2, 1.9, subband_scales);

        assert_eq!(shaped_steps[0], default_steps[0]);
        assert_eq!(shaped_steps[1], default_steps[1]);
        assert_eq!(shaped_steps[2], default_steps[2]);
        assert!(shaped_steps[3].delta(10) > default_steps[3].delta(10));
    }

    #[test]
    fn saturated_irreversible_coefficients_fit_declared_bitplanes() {
        let guard_bits = 2;
        let steps = compute_step_sizes(8, 1, false, guard_bits);
        let range_bits = [8u8, 9, 9, 10];

        for (&range_bits, step) in range_bits.iter().zip(&steps) {
            let quantized = quantize_subband(&[-128.0, 127.0], step, range_bits, false);
            let total_bitplanes = u16::from(guard_bits) + step.exponent - 1;
            let max_abs = quantized
                .iter()
                .map(|coefficient| coefficient.unsigned_abs())
                .max()
                .unwrap();
            assert!(
                max_abs < (1u32 << total_bitplanes),
                "range_bits={range_bits} step={step:?} quantized={quantized:?} total_bitplanes={total_bitplanes}"
            );
        }
    }

    #[test]
    fn step_conversion_rounds_into_range_or_rejects_at_both_limits() {
        let finest = QuantStepSize {
            exponent: 31,
            mantissa: 0,
        };
        let coarsest = QuantStepSize {
            exponent: 0,
            mantissa: 2047,
        };
        // 2^(8 - 31) is the finest step; just below it the mantissa carries in.
        for delta in [pow2i(-23), pow2i(-24) * (2.0 - 1.0 / 8192.0)] {
            assert_eq!(QuantStepSize::try_from_delta(8, delta), Some(finest));
        }
        assert_eq!(QuantStepSize::try_from_delta(8, 511.875), Some(coarsest));
        for delta in [
            pow2i(-24) * 1.5,
            511.968_75,
            512.0,
            1e-20,
            f32::MIN_POSITIVE,
            1e-45,
            1e30,
            f32::MAX,
            f32::INFINITY,
            0.0,
        ] {
            assert_eq!(QuantStepSize::try_from_delta(8, delta), None, "{delta}");
        }
        // Without a profile, out-of-range steps saturate instead of panicking.
        for (delta, expected) in [
            (1e-20, finest),
            (0.0, finest),
            (511.968_75, coarsest),
            (1e30, coarsest),
            (f32::INFINITY, coarsest),
        ] {
            assert_eq!(QuantStepSize::from_delta(8, delta), expected, "{delta}");
        }
    }

    #[test]
    fn global_scale_steps_never_decrease_past_the_coarsest_step() {
        let mut previous = 0.0;
        for scale in [1e-30, 1.0, 1024.0, 1536.0, 2048.0, 3072.0, 4096.0, 1e30] {
            let steps = compute_step_sizes_with_irreversible_scale(8, 0, false, 2, scale);
            let delta = steps[0].delta(8);
            assert!(delta >= previous, "scale {scale}: {delta} < {previous}");
            previous = delta;
        }
    }

    #[test]
    fn high_bit_depth_steps_never_decrease_and_reconstruct_past_2_pow_31() {
        let finest = QuantStepSize {
            exponent: 31,
            mantissa: 0,
        };
        let coarsest = QuantStepSize {
            exponent: 0,
            mantissa: 2047,
        };
        for bit_depth in [30_u8, 32, 33, 38] {
            let mut previous = 0.0;
            for scale in (-40..=60).flat_map(|exp| [pow2i(exp), pow2i(exp) * 1.5]) {
                let steps =
                    compute_step_sizes_with_irreversible_scale(bit_depth, 0, false, 2, scale);
                let delta = steps[0].delta(bit_depth);
                assert!(
                    delta.is_finite() && delta >= previous,
                    "bit depth {bit_depth}, scale {scale}: {delta} < {previous}"
                );
                previous = delta;
            }
            let at = |scale| {
                compute_step_sizes_with_irreversible_scale(bit_depth, 0, false, 2, scale)[0]
            };
            assert_eq!(at(pow2i(-40)), finest, "bit depth {bit_depth}");
            assert_eq!(at(pow2i(60)), coarsest, "bit depth {bit_depth}");
        }
        // A 38-bit step of 2^33 is stored exactly rather than saturated.
        let step = QuantStepSize::try_from_delta(38, pow2i(33)).unwrap();
        assert_eq!((step.exponent, step.mantissa), (5, 0));
        assert_eq!(step.delta(38), pow2i(33));
        assert_eq!(coarsest.delta(40), pow2i(40) * (1.0 + 2047.0 / 2048.0));
    }

    fn profiled_ll_delta(weight: f32, scale: f32) -> EncodeResult<f32> {
        let mut steps = compute_step_sizes_with_irreversible_scale(8, 2, false, 2, scale);
        let mut weights = vec![1.0; steps.len()];
        weights[0] = weight;
        apply_component_profile(
            &mut steps,
            8,
            2,
            scale,
            IrreversibleQuantizationSubbandScales::default(),
            Some(&[weights]),
            0,
        )?;
        Ok(steps[0].delta(8))
    }

    #[test]
    fn profile_steps_follow_weight_order_until_they_are_rejected() {
        // Guard bits 2 give an unweighted step of 0.25; a two-level LL's
        // synthesis norm refines it to 0.25 * 0.490629 * weight before rounding.
        let delta = profiled_ll_delta(1536.0, 1.0).unwrap();
        assert!((delta / 188.4015 - 1.0).abs() < 1e-3, "{delta}");
        assert!(profiled_ll_delta(4096.0, 1.0).is_ok());
        assert!(profiled_ll_delta(4400.0, 1.0).is_err());

        let mut previous = None;
        let mut leaving_range = false;
        let weights = core::iter::successors(Some(1e-12_f32), |weight| Some(weight * 1.13))
            .take_while(|&weight| weight < 1e12);
        for weight in weights {
            match profiled_ll_delta(weight, 1.0) {
                Ok(delta) => {
                    assert!(!leaving_range, "weight {weight} re-entered the range");
                    assert!(previous.is_none_or(|previous| delta >= previous));
                    previous = Some(delta);
                }
                Err(_) => leaving_range = previous.is_some(),
            }
        }
        assert!(leaving_range);
    }

    #[test]
    fn profile_weights_the_unrounded_globally_scaled_step() {
        // 0.25 * 4096 = 1024 is past the coarsest step for the finest bands, but
        // a quarter of it is not: the profile reproduces the default at 1024.
        let mut steps = compute_step_sizes_with_irreversible_scale(8, 2, false, 2, 4096.0);
        apply_component_profile(
            &mut steps,
            8,
            2,
            4096.0,
            IrreversibleQuantizationSubbandScales::default(),
            Some(&[vec![0.25; 7]]),
            0,
        )
        .unwrap();
        assert_eq!(
            steps,
            compute_step_sizes_with_irreversible_scale(8, 2, false, 2, 1024.0)
        );
        // Unity bands of an active profile must also stay representable.
        assert!(profiled_ll_delta(0.25, 4096.0).is_err());
    }

    #[test]
    fn profile_product_can_reenter_range_after_an_intermediate_overflow() {
        let scales = IrreversibleQuantizationSubbandScales {
            low_low: 1e20,
            ..IrreversibleQuantizationSubbandScales::default()
        };
        let mut steps = compute_step_sizes_with_irreversible_profile(8, 0, false, 2, 1e20, scales);
        apply_component_profile(&mut steps, 8, 2, 1e20, scales, Some(&[vec![1e-40]]), 0)
            .expect("the complete positive product is approximately 0.25");
        assert_eq!((steps[0].exponent, steps[0].mantissa), (10, 0));
    }

    #[test]
    fn global_product_preserves_a_subnormal_factor_before_rescaling() {
        let scales = IrreversibleQuantizationSubbandScales {
            low_low: f32::MAX,
            ..IrreversibleQuantizationSubbandScales::default()
        };
        let steps =
            compute_step_sizes_with_irreversible_profile(1, 0, false, 2, f32::from_bits(1), scales);
        // 2^-2 * 2^-149 * (2^128 - 2^104) rounds to 2^-23.
        assert_eq!((steps[0].exponent, steps[0].mantissa), (24, 0));
    }

    #[test]
    fn profile_interval_checks_both_limits_of_every_weighted_band() {
        let profile = [vec![1.0 / 16_777_216.0, 1.0, 3.0, 8.0]];
        let scales = IrreversibleQuantizationSubbandScales::default();
        let (lower, upper) =
            quantization_scale_bounds(8, 2, scales, Some(&profile), 1, MAX_STEP_EXPONENT);
        let accepted = |scale| {
            let mut steps = compute_step_sizes_with_irreversible_scale(8, 1, false, 2, scale);
            apply_component_profile(&mut steps, 8, 2, scale, scales, Some(&profile), 0).is_ok()
        };
        assert!(lower > 7.0 && lower <= 8.0);
        // HH weight 8 on a one-level HH norm weight of 0.944752: 511.875 / 1.889504.
        assert!(upper > 270.0 && upper < 271.5, "{upper}");
        assert!(accepted(lower) && accepted(upper));
        assert!(!accepted(lower.next_down()) && !accepted(upper.next_up()));
        let incompatible = [vec![1e-20, 1e20, 1.0, 1.0]];
        let (lower, upper) =
            quantization_scale_bounds(8, 2, scales, Some(&incompatible), 1, MAX_STEP_EXPONENT);
        assert!(lower > upper);
    }

    #[test]
    fn profile_scale_ceiling_is_the_last_representable_scale() {
        let profile = [vec![1.0, 3.0, 3.0, 3.0, 5.0, 5.0, 5.0]];
        let scales = IrreversibleQuantizationSubbandScales::default();
        let ceiling = profile_scale_ceiling(8, 2, 1.0, scales, &profile);
        let representable = |scale: f32| {
            let mut steps = compute_step_sizes_with_irreversible_scale(8, 2, false, 2, scale);
            apply_component_profile(&mut steps, 8, 2, scale, scales, Some(&profile), 0).is_ok()
        };
        assert!(ceiling > 1.0 && ceiling.to_bits() != ceiling.round().to_bits());
        assert!(representable(ceiling));
        assert!(!representable(ceiling.next_up()));
        // An out-of-range start is returned for encode to reject.
        assert_eq!(profile_scale_ceiling(8, 2, 1e9, scales, &profile), 1e9);
    }

    #[test]
    fn default_irreversible_steps_follow_97_synthesis_norms() {
        // OpenJPEG 2.5, Grok and OpenJPH signal HL steps about 2x larger per
        // finer level and about 17x larger at level 1 than at level 5.
        let steps = compute_step_sizes(8, 5, false, 2);
        let high_low = |level: usize| steps[1 + 3 * (5 - level)].delta(9);
        let ladder = high_low(1) / high_low(5);
        assert!((16.5..17.5).contains(&ladder), "{ladder}");
        for level in 1..5 {
            let ratio = high_low(level) / high_low(level + 1);
            assert!((1.9..2.2).contains(&ratio), "level {level}: {ratio}");
        }
    }

    #[test]
    fn default_irreversible_steps_keep_the_smallest_norm_band_and_refine_the_rest() {
        // Without norm weighting every band here signals exponent 10, mantissa 0.
        let unweighted = QuantStepSize {
            exponent: 10,
            mantissa: 0,
        };
        for levels in 1..=6_u8 {
            let steps = compute_step_sizes(8, levels, false, 2);
            // One level: LL has the smallest norm; otherwise the finest HL/LH.
            let kept = if levels == 1 {
                vec![0]
            } else {
                vec![steps.len() - 3, steps.len() - 2]
            };
            for (index, step) in steps.iter().enumerate() {
                if kept.contains(&index) {
                    assert_eq!(*step, unweighted, "{levels} levels, band {index}");
                } else {
                    assert!(
                        step.delta(8) < unweighted.delta(8),
                        "{levels} levels, band {index}: {step:?}"
                    );
                }
            }
        }
    }
}
