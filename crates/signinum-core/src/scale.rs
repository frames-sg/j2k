// SPDX-License-Identifier: Apache-2.0

/// Reduced-resolution decode factor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Downscale {
    /// Full resolution.
    None,
    /// One-half resolution in each axis.
    Half,
    /// One-quarter resolution in each axis.
    Quarter,
    /// One-eighth resolution in each axis.
    Eighth,
}

impl Downscale {
    /// Return the scale denominator.
    pub const fn denominator(self) -> u32 {
        match self {
            Self::None => 1,
            Self::Half => 2,
            Self::Quarter => 4,
            Self::Eighth => 8,
        }
    }

    /// Return the JPEG block edge represented by one output block at this scale.
    pub const fn output_block_size(self) -> u32 {
        match self {
            Self::None => 8,
            Self::Half => 4,
            Self::Quarter => 2,
            Self::Eighth => 1,
        }
    }
}
