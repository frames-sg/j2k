// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared lossless JPEG decode helpers.

pub(crate) mod scan;

/// Output container for reconstructed lossless samples: `u8` for 8-bit and
/// `u16` (little-endian) for 16-bit images.
pub(crate) trait LosslessSample: Copy + Default {
    const BIT_DEPTH: u8;
    const BYTES: usize;

    /// Narrow a reconstructed sample. [`scan::LosslessScan`] has already
    /// rejected values wider than the image precision.
    fn from_sample(sample: u16) -> Self;

    fn write_le(self, dst: &mut [u8]);

    /// Store one component row, shifted left by the point transform.
    fn store_row(samples: &[u16], shift: u8, dst: &mut [u8]) {
        for (dst, &sample) in dst.chunks_exact_mut(Self::BYTES).zip(samples) {
            Self::from_sample(sample << shift).write_le(dst);
        }
    }

    /// Interleave three component rows into one pixel row.
    fn store_interleaved(rows: [&[u16]; 3], shift: u8, dst: &mut [u8]) {
        let [c0, c1, c2] = rows;
        for (pixel, ((&s0, &s1), &s2)) in dst
            .chunks_exact_mut(3 * Self::BYTES)
            .zip(c0.iter().zip(c1).zip(c2))
        {
            let (first, rest) = pixel.split_at_mut(Self::BYTES);
            let (second, third) = rest.split_at_mut(Self::BYTES);
            Self::from_sample(s0 << shift).write_le(first);
            Self::from_sample(s1 << shift).write_le(second);
            Self::from_sample(s2 << shift).write_le(third);
        }
    }
}

impl LosslessSample for u8 {
    const BIT_DEPTH: u8 = 8;
    const BYTES: usize = 1;

    #[expect(
        clippy::cast_possible_truncation,
        reason = "8-bit lossless samples are range-checked to 0..=255 during reconstruction"
    )]
    fn from_sample(sample: u16) -> Self {
        sample as u8
    }

    fn write_le(self, dst: &mut [u8]) {
        dst[0] = self;
    }

    fn store_row(samples: &[u16], shift: u8, dst: &mut [u8]) {
        for (dst, &sample) in dst.iter_mut().zip(samples) {
            *dst = Self::from_sample(sample << shift);
        }
    }

    fn store_interleaved(rows: [&[u16]; 3], shift: u8, dst: &mut [u8]) {
        let [c0, c1, c2] = rows;
        for (pixel, ((&s0, &s1), &s2)) in dst.chunks_exact_mut(3).zip(c0.iter().zip(c1).zip(c2)) {
            pixel[0] = Self::from_sample(s0 << shift);
            pixel[1] = Self::from_sample(s1 << shift);
            pixel[2] = Self::from_sample(s2 << shift);
        }
    }
}

impl LosslessSample for u16 {
    const BIT_DEPTH: u8 = 16;
    const BYTES: usize = 2;

    fn from_sample(sample: u16) -> Self {
        sample
    }

    fn write_le(self, dst: &mut [u8]) {
        dst[..2].copy_from_slice(&self.to_le_bytes());
    }
}
