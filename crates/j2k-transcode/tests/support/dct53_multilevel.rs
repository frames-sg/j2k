// SPDX-License-Identifier: MIT OR Apache-2.0

//! Multilevel DCT-to-5/3 experiments.
//!
//! The first decomposition level is produced by the direct DCT-domain mapping.
//! Additional levels use a conventional 5/3 transform over the LL band, which
//! keeps the validation surface smaller until profiling justifies direct LL2+
//! mappings.

use core::fmt;

use crate::dct53_2d::{
    dct8x8_blocks_then_dwt53_float, dct8x8_blocks_to_dwt53_float_linear,
    linearized_53_2d_from_plane,
};
use crate::Dwt53TwoDimensional;

/// Multilevel 5/3 decomposition result for one component plane.
#[derive(Debug, Clone, PartialEq)]
pub struct Dwt53MultiLevel<T> {
    /// Decomposition levels in order from full resolution to lowest LL.
    pub levels: Vec<Dwt53TwoDimensional<T>>,
    /// Lowest-resolution LL band after all requested levels.
    pub final_ll: Vec<T>,
    /// Width of `final_ll`.
    pub final_ll_width: usize,
    /// Height of `final_ll`.
    pub final_ll_height: usize,
}

impl Dwt53MultiLevel<f64> {
    /// Maximum absolute coefficient difference across matching levels.
    #[must_use]
    pub fn max_abs_diff(&self, other: &Self) -> f64 {
        assert_eq!(self.levels.len(), other.levels.len());
        assert_eq!(self.final_ll_width, other.final_ll_width);
        assert_eq!(self.final_ll_height, other.final_ll_height);

        let level_diff = self
            .levels
            .iter()
            .zip(other.levels.iter())
            .map(|(actual, expected)| actual.max_abs_diff(expected))
            .fold(0.0, f64::max);

        self.final_ll
            .iter()
            .zip(other.final_ll.iter())
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(level_diff, f64::max)
    }
}

/// Error returned when the requested decomposition level count is invalid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dwt53MultiLevelError {
    requested_levels: usize,
    available_levels: usize,
}

impl Dwt53MultiLevelError {
    /// Requested decomposition levels.
    #[must_use]
    pub const fn requested_levels(self) -> usize {
        self.requested_levels
    }

    /// Maximum levels supported by the supplied starting plane.
    #[must_use]
    pub const fn available_levels(self) -> usize {
        self.available_levels
    }
}

impl fmt::Display for Dwt53MultiLevelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "requested {} decomposition levels, but only {} are available",
            self.requested_levels, self.available_levels
        )
    }
}

impl std::error::Error for Dwt53MultiLevelError {}

/// Direct level-1 DCT-to-5/3 followed by conventional LL recursion.
pub fn dct8x8_to_dwt53_multilevel_float_linear(
    block: [[f64; 8]; 8],
    levels: usize,
) -> Result<Dwt53MultiLevel<f64>, Dwt53MultiLevelError> {
    validate_levels(levels, 8, 8)?;
    Ok(decompose_from_first_level(
        dct8x8_blocks_to_dwt53_float_linear(&[block], 1, 1, 8, 8).expect("valid single DCT block"),
        levels,
    ))
}

/// Reference multilevel path:
/// DCT coefficients -> float IDCT samples -> conventional 5/3 recursion.
pub fn idct8x8_then_dwt53_multilevel_float(
    block: [[f64; 8]; 8],
    levels: usize,
) -> Result<Dwt53MultiLevel<f64>, Dwt53MultiLevelError> {
    validate_levels(levels, 8, 8)?;
    Ok(decompose_from_first_level(
        dct8x8_blocks_then_dwt53_float(&[block], 1, 1, 8, 8).expect("valid single DCT block"),
        levels,
    ))
}

fn decompose_from_first_level(
    first_level: Dwt53TwoDimensional<f64>,
    levels: usize,
) -> Dwt53MultiLevel<f64> {
    let mut decomposition = Dwt53MultiLevel {
        final_ll: first_level.ll.clone(),
        final_ll_width: first_level.low_width,
        final_ll_height: first_level.low_height,
        levels: vec![first_level],
    };

    while decomposition.levels.len() < levels {
        let next = linearized_53_2d_from_plane(
            &decomposition.final_ll,
            decomposition.final_ll_width,
            decomposition.final_ll_height,
        );
        decomposition.final_ll.clone_from(&next.ll);
        decomposition.final_ll_width = next.low_width;
        decomposition.final_ll_height = next.low_height;
        decomposition.levels.push(next);
    }

    decomposition
}

fn validate_levels(
    requested_levels: usize,
    width: usize,
    height: usize,
) -> Result<(), Dwt53MultiLevelError> {
    let available_levels = available_levels(width, height);
    if requested_levels == 0 || requested_levels > available_levels {
        return Err(Dwt53MultiLevelError {
            requested_levels,
            available_levels,
        });
    }

    Ok(())
}

fn available_levels(mut width: usize, mut height: usize) -> usize {
    let mut levels = 0;
    while width >= 2 && height >= 2 {
        levels += 1;
        width = width.div_ceil(2);
        height = height.div_ceil(2);
    }
    levels
}
