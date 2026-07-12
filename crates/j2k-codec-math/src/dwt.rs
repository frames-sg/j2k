//! JPEG 2000 discrete wavelet transform constants.

mod linearized53;
pub use linearized53::{
    linearized_dwt53_row, Dwt53Band, Dwt53LinearRow, Dwt53LinearTap, DWT53_MAX_HIGH_LINEAR_TAPS,
    DWT53_MAX_LINEAR_TAPS,
};

/// Return the maximum number of DWT decomposition levels supported by an
/// image geometry.
///
/// The shared encoder policy is `floor(log2(min(width, height)))`; a zero or
/// unit-length axis supports no decomposition levels. Keeping this
/// conservative geometry rule here ensures CPU and GPU paths apply the same
/// ceiling.
#[must_use]
pub const fn max_decomposition_levels(width: u32, height: u32) -> u8 {
    let mut minimum_dimension = if width < height { width } else { height };
    let mut levels = 0_u8;
    while minimum_dimension > 1 {
        minimum_dimension >>= 1;
        levels += 1;
    }
    levels
}

/// Forward irreversible 9/7 lifting step alpha, rounded for existing CPU/GPU paths.
pub const DWT97_ALPHA_F32: f32 = -1.586_134_3;
/// Forward irreversible 9/7 lifting step beta, rounded for existing CPU/GPU paths.
pub const DWT97_BETA_F32: f32 = -0.052_980_117;
/// Forward irreversible 9/7 lifting step gamma, rounded for existing CPU/GPU paths.
pub const DWT97_GAMMA_F32: f32 = 0.882_911_1;
/// Forward irreversible 9/7 lifting step delta, rounded for existing CPU/GPU paths.
pub const DWT97_DELTA_F32: f32 = 0.443_506_87;
/// Irreversible 9/7 scaling factor, rounded for existing CPU/GPU paths.
pub const DWT97_KAPPA_F32: f32 = 1.230_174_1;
/// Inverse irreversible 9/7 scaling factor, computed the same way as existing paths.
pub const DWT97_INV_KAPPA_F32: f32 = 1.0 / DWT97_KAPPA_F32;

/// Inverse 9/7 alpha step used by synthesis paths.
pub const IDWT97_NEG_ALPHA_F32: f32 = -DWT97_ALPHA_F32;
/// Inverse 9/7 beta step used by synthesis paths.
pub const IDWT97_NEG_BETA_F32: f32 = -DWT97_BETA_F32;
/// Inverse 9/7 gamma step used by synthesis paths.
pub const IDWT97_NEG_GAMMA_F32: f32 = -DWT97_GAMMA_F32;
/// Inverse 9/7 delta step used by synthesis paths.
pub const IDWT97_NEG_DELTA_F32: f32 = -DWT97_DELTA_F32;

/// Forward irreversible 9/7 lifting step alpha at f64 precision.
pub const DWT97_ALPHA_F64: f64 = -1.586_134_342_059_924;
/// Forward irreversible 9/7 lifting step beta at f64 precision.
pub const DWT97_BETA_F64: f64 = -0.052_980_118_572_961;
/// Forward irreversible 9/7 lifting step gamma at f64 precision.
pub const DWT97_GAMMA_F64: f64 = 0.882_911_075_530_934;
/// Forward irreversible 9/7 lifting step delta at f64 precision.
pub const DWT97_DELTA_F64: f64 = 0.443_506_852_043_971;
/// Irreversible 9/7 scaling factor at f64 precision.
pub const DWT97_KAPPA_F64: f64 = 1.230_174_104_914_001;
/// Inverse irreversible 9/7 scaling factor at f64 precision.
pub const DWT97_INV_KAPPA_F64: f64 = 1.0 / DWT97_KAPPA_F64;

#[cfg(test)]
mod tests {
    use super::*;

    const MAX_LEVELS_FOR_U32_GEOMETRY: u8 = max_decomposition_levels(u32::MAX, u32::MAX);

    #[test]
    fn maximum_decomposition_levels_are_const_and_use_the_shorter_axis() {
        assert_eq!(MAX_LEVELS_FOR_U32_GEOMETRY, 31);

        for (width, height, expected) in [
            (0, 0, 0),
            (0, u32::MAX, 0),
            (u32::MAX, 0, 0),
            (1, u32::MAX, 0),
            (u32::MAX, 1, 0),
            (2, 8, 1),
            (8, 2, 1),
            (3, 9, 1),
            (9, 3, 1),
            (7, 9, 2),
            (9, 7, 2),
            (u32::MAX, u32::MAX, 31),
        ] {
            assert_eq!(max_decomposition_levels(width, height), expected);
        }
    }

    #[test]
    fn maximum_decomposition_levels_match_power_of_two_boundaries() {
        for exponent in 1_u8..=31 {
            let power = 1_u32 << exponent;
            assert_eq!(max_decomposition_levels(power, power), exponent);
            assert_eq!(max_decomposition_levels(power - 1, u32::MAX), exponent - 1);
            assert_eq!(
                max_decomposition_levels(power.saturating_add(1), u32::MAX),
                exponent
            );
        }
    }

    #[test]
    fn f32_constants_match_existing_backend_rounding() {
        assert_eq!(DWT97_ALPHA_F32.to_bits(), (-1.586_134_3f32).to_bits());
        assert_eq!(DWT97_BETA_F32.to_bits(), (-0.052_980_117f32).to_bits());
        assert_eq!(DWT97_GAMMA_F32.to_bits(), 0.882_911_1f32.to_bits());
        assert_eq!(DWT97_DELTA_F32.to_bits(), 0.443_506_87f32.to_bits());
        assert_eq!(DWT97_KAPPA_F32.to_bits(), 1.230_174_1f32.to_bits());
        assert_eq!(
            DWT97_INV_KAPPA_F32.to_bits(),
            (1.0f32 / 1.230_174_1f32).to_bits()
        );
    }

    #[test]
    fn inverse_constants_are_exact_negations_of_forward_steps() {
        assert_eq!(IDWT97_NEG_ALPHA_F32.to_bits(), (-DWT97_ALPHA_F32).to_bits());
        assert_eq!(IDWT97_NEG_BETA_F32.to_bits(), (-DWT97_BETA_F32).to_bits());
        assert_eq!(IDWT97_NEG_GAMMA_F32.to_bits(), (-DWT97_GAMMA_F32).to_bits());
        assert_eq!(IDWT97_NEG_DELTA_F32.to_bits(), (-DWT97_DELTA_F32).to_bits());
    }
}
