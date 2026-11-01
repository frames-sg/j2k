// SPDX-License-Identifier: MIT OR Apache-2.0

//! Code-blocks with selective bypass, per-pass termination, context resets
//! or segmentation symbols decode on the packed-column passes, raw segments
//! included. They must give the per-coefficient passes' samples bit for bit,
//! and the same outcome on corrupt data, strict or not.

use alloc::vec::Vec;

use crate::j2c::bitplane::FORCE_PER_COEFFICIENT;
use crate::{
    decode_j2k_code_block_scalar, encode_j2k_code_block_scalar_with_style, J2kCodeBlockDecodeJob,
    J2kCodeBlockSegment, J2kCodeBlockStyle, J2kSubBandType,
};

/// Xorshift, for reproducible blocks and mutations.
struct Random(u64);

impl Random {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// A value below `bound`.
    fn below(&mut self, bound: usize) -> usize {
        usize::try_from(self.next() % u64::try_from(bound).expect("bound")).expect("below bound")
    }
}

struct Block {
    data: Vec<u8>,
    segments: Vec<J2kCodeBlockSegment>,
    width: u32,
    height: u32,
    missing_bit_planes: u8,
    passes: u8,
    total_bitplanes: u8,
    sub_band_type: J2kSubBandType,
    style: J2kCodeBlockStyle,
}

/// Mostly small magnitudes, as in wavelet sub-bands, `density` in 16 of
/// them nonzero, below `2^total_bitplanes`.
fn coefficients(
    random: &mut Random,
    samples: u32,
    density: usize,
    total_bitplanes: u8,
) -> Vec<i32> {
    let limit = 1_usize << total_bitplanes;
    (0..samples)
        .map(|_| {
            if random.below(16) >= density {
                return 0;
            }
            let magnitude = random.below(limit) >> random.below(usize::from(total_bitplanes));
            let magnitude = i32::try_from(magnitude.min(limit - 1)).expect("magnitude");
            if random.below(2) == 0 {
                magnitude
            } else {
                -magnitude
            }
        })
        .collect()
}

/// The samples' bits, or the error, decoding `data` for `block` on the
/// per-coefficient passes or not.
fn decode(
    block: &Block,
    data: &[u8],
    strict: bool,
    per_coefficient: bool,
) -> Result<Vec<u32>, String> {
    let mut output = alloc::vec![0.0_f32; (block.width * block.height) as usize];
    FORCE_PER_COEFFICIENT.set(per_coefficient);
    let result = decode_j2k_code_block_scalar(
        J2kCodeBlockDecodeJob {
            data,
            segments: &block.segments,
            width: block.width,
            height: block.height,
            output_stride: block.width as usize,
            missing_bit_planes: block.missing_bit_planes,
            number_of_coding_passes: block.passes,
            total_bitplanes: block.total_bitplanes,
            roi_shift: 0,
            sub_band_type: block.sub_band_type,
            style: block.style,
            strict,
            dequantization_step: 1.0,
        },
        &mut output,
    );
    FORCE_PER_COEFFICIENT.set(false);
    result
        .map(|()| output.iter().map(|sample| sample.to_bits()).collect())
        .map_err(|error| alloc::format!("{error:?}"))
}

/// A copy of `data` with a few bits flipped or bytes set to 0xFF, sometimes
/// truncated.
fn mutate(random: &mut Random, data: &[u8]) -> Vec<u8> {
    let mut data = data.to_vec();
    for _ in 0..=random.below(3) {
        let at = random.below(data.len());
        if random.below(4) == 0 {
            data[at] = 0xFF;
        } else {
            data[at] ^= 1 << random.below(8);
        }
    }
    if random.below(5) == 0 {
        data.truncate(random.below(data.len()));
    }
    data
}

/// Decode `block` and corrupt copies of it both ways; returns how many
/// corrupt decodes failed.
fn check(random: &mut Random, block: &Block, label: &str) -> usize {
    for strict in [false, true] {
        let packed = decode(block, &block.data, strict, false);
        assert!(packed.is_ok(), "{label}: {packed:?}");
        assert_eq!(packed, decode(block, &block.data, strict, true), "{label}");
    }
    let mut failures = 0;
    if block.data.is_empty() {
        return failures;
    }
    for mutant in 0..24 {
        let data = mutate(random, &block.data);
        for strict in [false, true] {
            let per_coefficient = decode(block, &data, strict, true);
            failures += usize::from(per_coefficient.is_err());
            assert_eq!(
                decode(block, &data, strict, false),
                per_coefficient,
                "{label} mutant={mutant} strict={strict}"
            );
        }
    }
    failures
}

#[test]
fn packed_passes_match_per_coefficient_passes_for_every_non_causal_style() {
    let mut random = Random(0x9e37_79b9_7f4a_7c15);
    let sub_bands = [
        J2kSubBandType::LowLow,
        J2kSubBandType::HighLow,
        J2kSubBandType::LowHigh,
        J2kSubBandType::HighHigh,
    ];
    let sizes = [
        (1, 1),
        (3, 5),
        (4, 4),
        (7, 13),
        (16, 16),
        (33, 17),
        (64, 8),
        (64, 64),
    ];
    let (mut blocks, mut raw_blocks, mut failures) = (0, 0, 0);
    for styles in 0..16_u32 {
        let style = J2kCodeBlockStyle {
            selective_arithmetic_coding_bypass: styles & 1 != 0,
            reset_context_probabilities: styles & 2 != 0,
            termination_on_each_pass: styles & 4 != 0,
            vertically_causal_context: false,
            segmentation_symbols: styles & 8 != 0,
        };
        for (width, height) in sizes {
            for density in [1, 4, 12] {
                // Raw coding starts after the fourth bitplane.
                let total_bitplanes = 1 + u8::try_from(random.below(16)).expect("bitplanes");
                let coefficients =
                    coefficients(&mut random, width * height, density, total_bitplanes);
                let sub_band_type = sub_bands[random.below(4)];
                let encoded = encode_j2k_code_block_scalar_with_style(
                    &coefficients,
                    width,
                    height,
                    sub_band_type,
                    total_bitplanes,
                    style,
                )
                .expect("encode code-block");
                let block = Block {
                    data: encoded.data,
                    segments: encoded.segments,
                    width,
                    height,
                    missing_bit_planes: encoded.missing_bit_planes,
                    passes: encoded.number_of_coding_passes,
                    total_bitplanes,
                    sub_band_type,
                    style,
                };
                blocks += 1;
                raw_blocks +=
                    usize::from(block.segments.iter().any(|segment| !segment.use_arithmetic));
                let label = alloc::format!(
                    "style={styles:04b} {width}x{height} density={density} bitplanes={total_bitplanes}"
                );
                failures += check(&mut random, &block, &label);
            }
        }
    }
    assert!(
        raw_blocks > blocks / 4,
        "{raw_blocks} of {blocks} blocks had raw segments"
    );
    assert!(failures > 100, "{failures} corrupt decodes failed");
}
