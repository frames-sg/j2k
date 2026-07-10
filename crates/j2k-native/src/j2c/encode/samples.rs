// SPDX-License-Identifier: MIT OR Apache-2.0

use super::MAX_PART1_SAMPLE_BIT_DEPTH;

pub(super) fn raw_pixel_bytes_per_sample(bit_depth: u8) -> Result<usize, &'static str> {
    if bit_depth == 0 || bit_depth > MAX_PART1_SAMPLE_BIT_DEPTH {
        return Err("unsupported bit depth");
    }
    Ok(usize::from(bit_depth).div_ceil(8).max(1))
}

pub(super) fn read_le_sample_value(bytes: &[u8], bit_depth: u8) -> u64 {
    let mut raw = 0_u64;
    for (shift, byte) in bytes.iter().enumerate() {
        raw |= u64::from(*byte) << (shift * 8);
    }
    let mask = (1_u64 << bit_depth) - 1;
    raw & mask
}

pub(super) fn sign_extend_sample(raw: u64, bit_depth: u8) -> i64 {
    let shift = 64 - u32::from(bit_depth);
    i64::from_ne_bytes((raw << shift).to_ne_bytes()) >> shift
}

pub(super) fn native_samples_equal(
    expected: &[u8],
    actual: &[u8],
    bit_depth: u8,
    signed: bool,
) -> bool {
    if expected.len() != actual.len() {
        return false;
    }

    let Ok(bytes_per_sample) = raw_pixel_bytes_per_sample(bit_depth) else {
        return false;
    };
    let sample_count = expected.len() / bytes_per_sample;
    (0..sample_count).all(|sample_index| {
        decode_native_sample(expected, sample_index, bit_depth, signed)
            == decode_native_sample(actual, sample_index, bit_depth, signed)
    })
}

fn decode_native_sample(bytes: &[u8], sample_index: usize, bit_depth: u8, signed: bool) -> i64 {
    let bytes_per_sample = raw_pixel_bytes_per_sample(bit_depth).unwrap_or(1);
    let byte_offset = sample_index * bytes_per_sample;
    let raw = read_le_sample_value(
        &bytes[byte_offset..byte_offset + bytes_per_sample],
        bit_depth,
    );

    if signed {
        sign_extend_sample(raw, bit_depth)
    } else {
        i64::try_from(raw).expect("supported unsigned sample values fit in i64")
    }
}
