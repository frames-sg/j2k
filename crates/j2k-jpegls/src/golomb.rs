// SPDX-License-Identifier: MIT OR Apache-2.0

//! Limited-length Golomb codes and error mapping (T.87 A.5).

use crate::bitio::{BitReader, BitWriter, ReadError};

/// Read a limited-length Golomb code with parameter `k` (T.87 A.5.3).
///
/// A prefix of `limit - qbpp - 1` zeros is the escape: `qbpp` bits of the
/// mapped value minus one follow it.
pub(crate) fn read(
    reader: &mut BitReader<'_>,
    k: u32,
    limit: u32,
    qbpp: u32,
) -> Result<i64, ReadError> {
    let escape = limit.saturating_sub(qbpp + 1);
    let high = reader.read_unary(escape)?;
    if high == escape {
        return Ok(i64::from(reader.read_bits(qbpp)?) + 1);
    }
    if k == 0 {
        return Ok(i64::from(high));
    }
    if k > 32 {
        return Err(ReadError::InvalidCode);
    }
    Ok((i64::from(high) << k) | i64::from(reader.read_bits(k)?))
}

/// Write `mapped` as a limited-length Golomb code with parameter `k`.
pub(crate) fn write(writer: &mut BitWriter, mapped: u32, k: u32, limit: u32, qbpp: u32) {
    let escape = limit - qbpp - 1;
    let high = mapped >> k;
    if high < escape {
        writer.put_zeros(high);
        writer.put_bit(true);
        writer.put(mapped, k);
    } else {
        writer.put_zeros(escape);
        writer.put_bit(true);
        writer.put(mapped.wrapping_sub(1), qbpp);
    }
}

/// Map a regular-mode error to a non-negative value (T.87 A.5.2).
/// `inverted` selects the mapping for lossless `k = 0` with `2B <= -N`.
pub(crate) fn map_error(error: i32, inverted: bool) -> u32 {
    let error = if inverted { -error - 1 } else { error };
    if error >= 0 {
        error.unsigned_abs() * 2
    } else {
        error.unsigned_abs() * 2 - 1
    }
}

/// Invert [`map_error`].
pub(crate) const fn unmap_error(mapped: i64, inverted: bool) -> i64 {
    let error = if mapped & 1 == 0 {
        mapped >> 1
    } else {
        -((mapped + 1) >> 1)
    };
    if inverted {
        -error - 1
    } else {
        error
    }
}

#[cfg(test)]
mod tests {
    use super::{map_error, read, unmap_error, write};
    use crate::bitio::{BitReader, BitWriter};

    #[test]
    fn error_mapping_matches_annex_a() {
        // Normal mapping: 0, -1, 1, -2, 2 -> 0, 1, 2, 3, 4.
        let normal: Vec<u32> = [0, -1, 1, -2, 2]
            .iter()
            .map(|&e| map_error(e, false))
            .collect();
        assert_eq!(normal, [0, 1, 2, 3, 4]);
        // Inverted mapping: 0 -> 1, -1 -> 0, 1 -> 3, -2 -> 2.
        let inverted: Vec<u32> = [0, -1, 1, -2].iter().map(|&e| map_error(e, true)).collect();
        assert_eq!(inverted, [1, 0, 3, 2]);
        for error in -300..300 {
            for inverted in [false, true] {
                let mapped = i64::from(map_error(error, inverted));
                assert_eq!(unmap_error(mapped, inverted), i64::from(error));
            }
        }
    }

    #[test]
    fn golomb_codes_round_trip_including_escapes() {
        let (limit, qbpp) = (32, 8);
        let mut writer = BitWriter::new(Vec::new());
        let cases: Vec<(u32, u32)> = (0..256).map(|m| (m, m % 5)).collect();
        for &(mapped, k) in &cases {
            write(&mut writer, mapped, k, limit, qbpp);
        }
        writer.finish_segment();
        let bytes = writer.into_bytes();
        let mut reader = BitReader::new(&bytes);
        for &(mapped, k) in &cases {
            assert_eq!(read(&mut reader, k, limit, qbpp), Ok(i64::from(mapped)));
        }
    }
}
