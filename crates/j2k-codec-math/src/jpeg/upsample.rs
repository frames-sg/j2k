//!
//! Each output sample weights the nearer input sample three to one against
//! its neighbour, with libjpeg-turbo's alternating rounding. The functions
//! take the input samples around the output column, so a backend can read
//! them from whatever memory holds its rows; every JPEG decoder backend
//! upsamples with these formulas.

/// Output column `x` of a chroma row upsampled 2:1 horizontally (h2v1).
///
/// `current` is input sample `x / 2` of a row of `input_width` samples, and
/// `previous` and `next` are its left and right neighbours; at the row's ends,
/// where one does not exist, the formula does not read it. The first and last
/// output columns copy their input sample. Interior even columns round the
/// weighted sum down from the half and odd columns round it up, as
/// libjpeg-turbo's `h2v1_fancy_upsample` does.
#[inline]
#[expect(
    clippy::cast_possible_truncation,
    reason = "a weighted mean of u8 samples is in the u8 range"
)]
pub const fn h2v1_fancy(previous: u8, current: u8, next: u8, x: usize, input_width: usize) -> u8 {
    if x == 0 || x + 1 >= 2 * input_width {
        return current;
    }
    let weighted = 3 * current as u16;
    if x.is_multiple_of(2) {
        ((weighted + previous as u16 + 1) >> 2) as u8
    } else {
        ((weighted + next as u16 + 2) >> 2) as u8
    }
}

/// Column sum of h2v2 upsampling: three times the chroma sample of the row
/// nearer the output row plus the sample of the row beyond it.
#[inline]
pub const fn h2v2_column_sum(nearer: u8, farther: u8) -> u32 {
    3 * nearer as u32 + farther as u32
}

/// Output column `x` of a chroma row pair upsampled 2:1 in both directions
/// (h2v2), from the [`h2v2_column_sum`]s at input sample `x / 2` (`current`)
/// and its left and right neighbours.
///
/// `input_width` is the chroma row's sample count and `output_width` the
/// upsampled row's. The first column, and every column of a one-sample row,
/// takes its column sum with the upward rounding; the last column takes it
/// with the downward rounding; interior columns weight the nearer column
/// three to one, as libjpeg-turbo's `h2v2_fancy_upsample` does.
#[inline]
#[expect(
    clippy::cast_possible_truncation,
    reason = "a weighted mean of u8 samples is in the u8 range"
)]
pub const fn h2v2_fancy(
    previous: u32,
    current: u32,
    next: u32,
    x: usize,
    input_width: usize,
    output_width: usize,
) -> u8 {
    if x == 0 || input_width == 1 {
        return ((current * 4 + 8) >> 4) as u8;
    }
    if x + 1 == output_width {
        return ((current * 4 + 7) >> 4) as u8;
    }
    if x.is_multiple_of(2) {
        ((current * 3 + previous + 8) >> 4) as u8
    } else {
        ((current * 3 + next + 7) >> 4) as u8
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn h2v1_copies_the_ends_and_rounds_alternately_inside() {
        let row = [0u8, 255, 100];
        let upsampled: [u8; 6] = core::array::from_fn(|x| {
            let sample = x / 2;
            let previous = row[sample.saturating_sub(1)];
            let next = row[(sample + 1).min(row.len() - 1)];
            h2v1_fancy(previous, row[sample], next, x, row.len())
        });
        // (3*0+255+2)>>2, (3*255+0+1)>>2, (3*255+100+2)>>2, (3*100+255+1)>>2.
        assert_eq!(upsampled, [0, 64, 191, 216, 139, 100]);
    }

    #[test]
    fn h2v2_rounds_the_ends_and_alternates_inside() {
        let sums = [
            h2v2_column_sum(0, 0),
            h2v2_column_sum(255, 255),
            h2v2_column_sum(100, 0),
        ];
        let upsampled: [u8; 6] = core::array::from_fn(|x| {
            let sample = x / 2;
            let previous = sums[sample.saturating_sub(1)];
            let next = sums[(sample + 1).min(sums.len() - 1)];
            h2v2_fancy(previous, sums[sample], next, x, sums.len(), 6)
        });
        // (0+8)>>4, (0+1020+7)>>4, (3060+0+8)>>4, (3060+300+7)>>4,
        // (900+1020+8)>>4, (1200+7)>>4.
        assert_eq!(upsampled, [0, 64, 191, 210, 120, 75]);
    }

    #[test]
    fn constant_rows_upsample_to_the_same_constant() {
        for value in 0..=255u8 {
            let sum = h2v2_column_sum(value, value);
            for input_width in 1..=4 {
                for x in 0..2 * input_width {
                    assert_eq!(h2v1_fancy(value, value, value, x, input_width), value);
                    assert_eq!(
                        h2v2_fancy(sum, sum, sum, x, input_width, 2 * input_width),
                        value
                    );
                }
            }
        }
    }
}
