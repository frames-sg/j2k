// SPDX-License-Identifier: MIT OR Apache-2.0

/// Power-of-two downscale requested during decode.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Downscale {
    /// Full-resolution output.
    #[default]
    None,
    /// Half-resolution output.
    Half,
    /// Quarter-resolution output.
    Quarter,
    /// Eighth-resolution output.
    Eighth,
}

impl Downscale {
    /// Return the integer scale denominator.
    pub const fn denominator(self) -> u32 {
        match self {
            Self::None => 1,
            Self::Half => 2,
            Self::Quarter => 4,
            Self::Eighth => 8,
        }
    }

    /// Return the decoded DCT block dimension after scaling.
    pub const fn output_block_size(self) -> u32 {
        match self {
            Self::None => 8,
            Self::Half => 4,
            Self::Quarter => 2,
            Self::Eighth => 1,
        }
    }
}
