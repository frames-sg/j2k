// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_codec_math::dwt;

pub(super) const ALPHA: f64 = dwt::DWT97_ALPHA_F64;
pub(super) const BETA: f64 = dwt::DWT97_BETA_F64;
pub(super) const GAMMA: f64 = dwt::DWT97_GAMMA_F64;
pub(super) const DELTA: f64 = dwt::DWT97_DELTA_F64;
pub(super) const KAPPA: f64 = dwt::DWT97_KAPPA_F64;
pub(super) const INV_KAPPA: f64 = dwt::DWT97_INV_KAPPA_F64;

#[derive(Clone, Copy)]
pub(super) enum WaveletKind {
    Reversible53,
    Irreversible97,
}

#[derive(Clone, Copy)]
pub(super) enum WaveletBand {
    Low,
    High,
}

impl WaveletKind {
    pub(super) const fn max_taps_per_row(self) -> usize {
        match self {
            Self::Reversible53 => 5,
            Self::Irreversible97 => 16,
        }
    }
}

pub(super) const fn low_len(sample_len: usize) -> usize {
    sample_len.div_ceil(2)
}

pub(super) const fn high_len(sample_len: usize) -> usize {
    sample_len / 2
}
