// SPDX-License-Identifier: MIT OR Apache-2.0

//! Irreversible 9/7 forward-wavelet contracts.

use alloc::vec::Vec;

/// Forward irreversible 9/7 DWT job.
#[derive(Debug, Clone, Copy)]
pub struct J2kForwardDwt97Job<'a> {
    /// Source samples in row-major order.
    pub samples: &'a [f32],
    /// Source width in samples.
    pub width: u32,
    /// Source height in samples.
    pub height: u32,
    /// Number of decomposition levels requested.
    pub num_levels: u8,
}

/// Forward irreversible 9/7 DWT output.
#[derive(Debug)]
pub struct J2kForwardDwt97Output {
    /// LL subband coefficients from the lowest decomposition level.
    pub ll: Vec<f32>,
    /// LL subband width.
    pub ll_width: u32,
    /// LL subband height.
    pub ll_height: u32,
    /// Higher resolution detail levels, ordered from lowest to highest.
    pub levels: Vec<J2kForwardDwt97Level>,
}

/// One irreversible 9/7 DWT detail level.
#[derive(Debug)]
pub struct J2kForwardDwt97Level {
    /// HL subband coefficients.
    pub hl: Vec<f32>,
    /// LH subband coefficients.
    pub lh: Vec<f32>,
    /// HH subband coefficients.
    pub hh: Vec<f32>,
    /// Full-resolution width represented by this level.
    pub width: u32,
    /// Full-resolution height represented by this level.
    pub height: u32,
    /// Low-pass width at this level.
    pub low_width: u32,
    /// Low-pass height at this level.
    pub low_height: u32,
    /// High-pass width at this level.
    pub high_width: u32,
    /// High-pass height at this level.
    pub high_height: u32,
}

crate::move_only::assert_move_only!(J2kForwardDwt97Output, J2kForwardDwt97Level);
