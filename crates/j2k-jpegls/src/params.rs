// SPDX-License-Identifier: MIT OR Apache-2.0

//! Preset coding parameters (T.87 C.2.4.1.1) and the coding variables
//! derived from them (T.87 A.2.1).

/// Default context reset threshold `RESET`.
const DEFAULT_RESET: u16 = 64;
/// Default thresholds for `MAXVAL = 255`, `NEAR = 0` (T.87 Table C.3).
const BASIC_T1: i32 = 3;
const BASIC_T2: i32 = 7;
const BASIC_T3: i32 = 21;

/// JPEG-LS preset coding parameters, the content of an LSE segment with ID 1.
///
/// In a stream, and in
/// [`JpegLsEncodeOptions::preset`](crate::JpegLsEncodeOptions::preset), a
/// zero field means "use the default".
/// [`ScanInfo::parameters`](crate::ScanInfo::parameters) reports the values a
/// scan uses, with the defaults filled in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct PresetCodingParameters {
    /// Maximum sample value `MAXVAL`. The default is `2^P - 1` for `P` bits
    /// per sample.
    pub max_value: u16,
    /// Gradient threshold `T1`.
    pub t1: u16,
    /// Gradient threshold `T2`.
    pub t2: u16,
    /// Gradient threshold `T3`.
    pub t3: u16,
    /// Context reset threshold `RESET`. The default is 64.
    pub reset: u16,
}

impl PresetCodingParameters {
    /// The default parameters of T.87 C.2.4.1.1 for a maximum sample value
    /// and `NEAR`.
    #[must_use]
    pub fn defaults(max_value: u16, near: u8) -> Self {
        let max = i32::from(max_value);
        let near = i32::from(near);
        let (t1, t2, t3) = if max >= 128 {
            let factor = (max.min(4095) + 128) / 256;
            let t1 = clamp_threshold(factor * (BASIC_T1 - 2) + 2 + 3 * near, near + 1, max);
            let t2 = clamp_threshold(factor * (BASIC_T2 - 3) + 3 + 5 * near, t1, max);
            let t3 = clamp_threshold(factor * (BASIC_T3 - 4) + 4 + 7 * near, t2, max);
            (t1, t2, t3)
        } else {
            let factor = 256 / (max + 1);
            let t1 = clamp_threshold((BASIC_T1 / factor + 3 * near).max(2), near + 1, max);
            let t2 = clamp_threshold((BASIC_T2 / factor + 5 * near).max(3), t1, max);
            let t3 = clamp_threshold((BASIC_T3 / factor + 7 * near).max(4), t2, max);
            (t1, t2, t3)
        };
        Self {
            max_value,
            t1: saturate_u16(t1),
            t2: saturate_u16(t2),
            t3: saturate_u16(t3),
            reset: DEFAULT_RESET,
        }
    }
}

/// `CLAMP(i, j, MAXVAL)` of T.87 C.2.4.1.1.
const fn clamp_threshold(value: i32, low: i32, max: i32) -> i32 {
    if value > max || value < low {
        low
    } else {
        value
    }
}

fn saturate_u16(value: i32) -> u16 {
    u16::try_from(value.max(0)).unwrap_or(u16::MAX)
}

/// Why preset coding parameters or `NEAR` are invalid for a scan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ParameterError {
    MaxValue,
    Near { near: u8, max: u16 },
    T1,
    T2,
    T3,
    Reset,
}

impl ParameterError {
    pub(crate) const fn reason(self) -> &'static str {
        match self {
            Self::MaxValue => "MAXVAL must be 1 to 2^P - 1",
            Self::Near { .. } => "NEAR is above min(255, MAXVAL / 2)",
            Self::T1 => "T1 must be NEAR + 1 to MAXVAL",
            Self::T2 => "T2 must be T1 to MAXVAL",
            Self::T3 => "T3 must be T2 to MAXVAL",
            Self::Reset => "RESET must be 3 to max(255, MAXVAL)",
        }
    }
}

/// The coding variables of one scan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CodingParameters {
    pub(crate) max_value: i32,
    pub(crate) near: i32,
    pub(crate) t1: i32,
    pub(crate) t2: i32,
    pub(crate) t3: i32,
    pub(crate) reset: i64,
    pub(crate) range: i32,
    /// `qbpp`: bits of a mapped error value in an escape code.
    pub(crate) qbpp: u32,
    /// `LIMIT`: the longest Golomb code in bits.
    pub(crate) limit: u32,
    /// Initial value of the `A` context counters.
    pub(crate) a_init: i64,
}

impl CodingParameters {
    /// Resolve the parameters of a scan from the LSE values in effect (zero
    /// fields take defaults), as T.87 C.2.4.1.1 specifies.
    pub(crate) fn resolve(
        preset: Option<PresetCodingParameters>,
        bit_depth: u8,
        near: u8,
    ) -> Result<Self, ParameterError> {
        let preset = preset.unwrap_or_default();
        let full_scale = (1_i32 << bit_depth.clamp(2, 16)) - 1;
        let max_value = if preset.max_value == 0 {
            full_scale
        } else if i32::from(preset.max_value) > full_scale {
            return Err(ParameterError::MaxValue);
        } else {
            i32::from(preset.max_value)
        };
        let max_near = (max_value / 2).min(255);
        if i32::from(near) > max_near {
            return Err(ParameterError::Near {
                near,
                max: saturate_u16(max_near),
            });
        }
        let defaults = PresetCodingParameters::defaults(saturate_u16(max_value), near);
        let near_i = i32::from(near);
        let t1 = pick(preset.t1, defaults.t1, near_i + 1, max_value).ok_or(ParameterError::T1)?;
        let t2 = pick(preset.t2, defaults.t2, t1, max_value).ok_or(ParameterError::T2)?;
        let t3 = pick(preset.t3, defaults.t3, t2, max_value).ok_or(ParameterError::T3)?;
        let reset = pick(preset.reset, DEFAULT_RESET, 3, max_value.max(255))
            .ok_or(ParameterError::Reset)?;

        let range = (max_value + 2 * near_i) / (2 * near_i + 1) + 1;
        let bpp = ceil_log2(max_value + 1).max(2);
        Ok(Self {
            max_value,
            near: near_i,
            t1,
            t2,
            t3,
            reset: i64::from(reset),
            range,
            qbpp: ceil_log2(range),
            limit: 2 * (bpp + bpp.max(8)),
            a_init: i64::from(((range + 32) / 64).max(2)),
        })
    }

    /// The parameters with defaults filled in, as reported to callers.
    pub(crate) fn preset(&self) -> PresetCodingParameters {
        PresetCodingParameters {
            max_value: saturate_u16(self.max_value),
            t1: saturate_u16(self.t1),
            t2: saturate_u16(self.t2),
            t3: saturate_u16(self.t3),
            reset: u16::try_from(self.reset).unwrap_or(u16::MAX),
        }
    }

    /// Clamp a prediction to `[0, MAXVAL]`.
    pub(crate) fn clamp_sample(&self, value: i32) -> i32 {
        value.clamp(0, self.max_value)
    }

    /// Quantize a prediction error for near-lossless coding (T.87 A.4.4).
    pub(crate) const fn quantize_error(&self, error: i32) -> i32 {
        if self.near == 0 {
            error
        } else if error > 0 {
            (error + self.near) / (2 * self.near + 1)
        } else {
            -(self.near - error) / (2 * self.near + 1)
        }
    }

    /// Reduce an error modulo `RANGE` (T.87 A.4.5).
    pub(crate) const fn reduce_error(&self, error: i32) -> i32 {
        let mut error = error;
        if error < 0 {
            error += self.range;
        }
        if error >= (self.range + 1) / 2 {
            error -= self.range;
        }
        error
    }

    /// Reconstruct a sample from a prediction and a signed, quantized error
    /// (T.87 A.4.4 and the decoder's modulo correction in Annex F).
    pub(crate) fn reconstruct(&self, prediction: i32, signed_error: i32) -> i32 {
        let scale = 2 * self.near + 1;
        let mut value = prediction + signed_error * scale;
        if value < -self.near {
            value += self.range * scale;
        } else if value > self.max_value + self.near {
            value -= self.range * scale;
        }
        self.clamp_sample(value)
    }

    /// Quantize a gradient into one of nine regions (T.87 A.3.3).
    pub(crate) const fn quantize_gradient(&self, gradient: i32) -> i32 {
        if gradient <= -self.t3 {
            -4
        } else if gradient <= -self.t2 {
            -3
        } else if gradient <= -self.t1 {
            -2
        } else if gradient < -self.near {
            -1
        } else if gradient <= self.near {
            0
        } else if gradient < self.t1 {
            1
        } else if gradient < self.t2 {
            2
        } else if gradient < self.t3 {
            3
        } else {
            4
        }
    }
}

/// Use `value` if nonzero and within `low..=high`, the default if zero.
fn pick(value: u16, default: u16, low: i32, high: i32) -> Option<i32> {
    if value == 0 {
        return Some(i32::from(default));
    }
    let value = i32::from(value);
    (low..=high).contains(&value).then_some(value)
}

/// Smallest `x` with `2^x >= n`.
const fn ceil_log2(n: i32) -> u32 {
    let mut x = 0;
    while x < 31 && (1_i32 << x) < n {
        x += 1;
    }
    x
}

#[cfg(test)]
mod tests {
    use super::{CodingParameters, ParameterError, PresetCodingParameters};

    #[test]
    fn eight_bit_lossless_defaults_match_table_c3() {
        let defaults = PresetCodingParameters::defaults(255, 0);
        assert_eq!(
            (defaults.t1, defaults.t2, defaults.t3, defaults.reset),
            (3, 7, 21, 64)
        );
    }

    #[test]
    fn defaults_scale_with_maxval_and_near() {
        // Values CharLS writes for these cases.
        let sixteen = PresetCodingParameters::defaults(65535, 0);
        assert_eq!((sixteen.t1, sixteen.t2, sixteen.t3), (18, 67, 276));
        let maxval_1000 = PresetCodingParameters::defaults(1000, 0);
        assert_eq!(
            (maxval_1000.t1, maxval_1000.t2, maxval_1000.t3),
            (6, 19, 72)
        );
        let near3 = PresetCodingParameters::defaults(255, 3);
        assert_eq!((near3.t1, near3.t2, near3.t3), (12, 22, 42));
        let two_bit = PresetCodingParameters::defaults(3, 0);
        assert_eq!((two_bit.t1, two_bit.t2, two_bit.t3), (2, 3, 3));
    }

    #[test]
    fn derived_variables_follow_annex_a() {
        let eight = CodingParameters::resolve(None, 8, 0).expect("valid");
        assert_eq!(
            (eight.range, eight.qbpp, eight.limit, eight.a_init),
            (256, 8, 32, 4)
        );
        let sixteen = CodingParameters::resolve(None, 16, 0).expect("valid");
        assert_eq!(
            (sixteen.range, sixteen.qbpp, sixteen.limit, sixteen.a_init),
            (65536, 16, 64, 1024)
        );
        let near = CodingParameters::resolve(None, 8, 3).expect("valid");
        assert_eq!((near.range, near.qbpp, near.limit), (38, 6, 32));
    }

    #[test]
    fn out_of_range_parameters_are_rejected() {
        assert_eq!(
            CodingParameters::resolve(None, 8, 128),
            Err(ParameterError::Near {
                near: 128,
                max: 127
            })
        );
        let preset = |max_value, t1, t2, t3, reset| {
            Some(PresetCodingParameters {
                max_value,
                t1,
                t2,
                t3,
                reset,
            })
        };
        assert_eq!(
            CodingParameters::resolve(preset(256, 0, 0, 0, 0), 8, 0),
            Err(ParameterError::MaxValue)
        );
        assert_eq!(
            CodingParameters::resolve(preset(0, 300, 0, 0, 0), 8, 0),
            Err(ParameterError::T1)
        );
        assert_eq!(
            CodingParameters::resolve(preset(0, 10, 5, 0, 0), 8, 0),
            Err(ParameterError::T2)
        );
        assert_eq!(
            CodingParameters::resolve(preset(0, 0, 0, 2, 0), 8, 0),
            Err(ParameterError::T3)
        );
        assert_eq!(
            CodingParameters::resolve(preset(0, 0, 0, 0, 2), 8, 0),
            Err(ParameterError::Reset)
        );
    }
}
