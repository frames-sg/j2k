// SPDX-License-Identifier: MIT OR Apache-2.0

//! Samples deeper than 16 bits that drive the reversible transforms' values
//! as high as real images can, for tests of lossless coding at precisions on
//! both sides of the 20 bits through which it runs in `f32`.

/// Sample precisions on both sides of the 20 bits through which reversible
/// coding runs in `f32`.
pub(crate) const DEEP_PRECISIONS: [u8; 6] = [17, 20, 21, 22, 23, 24];
pub(crate) const DEEP_WIDTH: u32 = 96;
pub(crate) const DEEP_HEIGHT: u32 = 80;

/// Samples at the extremes of the `depth`-bit range in checkerboards of
/// several scales, a small step back from each extreme: the input that
/// drives the reversible wavelet's values highest. With three components,
/// green opposes red and blue, which does the same for the chroma planes.
pub(crate) fn hostile_deep_samples(depth: u8, signed: bool, components: usize) -> Vec<i64> {
    let (low, high) = if signed {
        (-(1_i64 << (depth - 1)), (1_i64 << (depth - 1)) - 1)
    } else {
        (0, (1_i64 << depth) - 1)
    };
    let mut samples = Vec::new();
    for y in 0..DEEP_HEIGHT {
        for x in 0..DEEP_WIDTH {
            let scale = (x / 24 + y / 20) % 4;
            let raised = ((x >> scale) + (y >> scale)) % 2 == 0;
            for component in 0..components {
                let index = (y * DEEP_WIDTH + x) as usize * components + component;
                let step = i64::try_from(index.wrapping_mul(2_654_435_761) >> 7).unwrap() % 4;
                let opposed = components == 3 && component == 1;
                samples.push(if raised == opposed {
                    low + step
                } else {
                    high - step
                });
            }
        }
    }
    samples
}

/// Interleaved samples as the little-endian bytes the encoder takes.
pub(crate) fn deep_sample_bytes(samples: &[i64], depth: u8) -> Vec<u8> {
    let width = usize::from(depth).div_ceil(8);
    samples
        .iter()
        .flat_map(|sample| sample.to_le_bytes().into_iter().take(width))
        .collect()
}

/// The samples of packed little-endian bytes, as the decoder writes them.
pub(crate) fn deep_samples_from_bytes(bytes: &[u8], depth: u8, signed: bool) -> Vec<i64> {
    let width = usize::from(depth).div_ceil(8);
    let shift = 64 - u32::from(depth);
    bytes
        .chunks_exact(width)
        .map(|chunk| {
            let mut wide = [0_u8; 8];
            wide[..width].copy_from_slice(chunk);
            let raw = u64::from_le_bytes(wide) << shift;
            if signed {
                raw.cast_signed() >> shift
            } else {
                (raw >> shift).cast_signed()
            }
        })
        .collect()
}
