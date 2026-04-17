// SPDX-License-Identifier: Apache-2.0

//! Chroma upsamplers. Three shapes this milestone supports:
//!
//! - **`upsample_1x1`** (4:4:4): identity copy; no resampling.
//! - **`upsample_h2v1_fancy`** (4:2:2): 2× horizontal triangle filter.
//! - **`upsample_h2v2_fancy`** (4:2:0): 2× horizontal + 2× vertical triangle
//!   filter (the libjpeg-turbo default for 4:2:0).
//!
//! The "fancy" name is libjpeg-turbo's; the filter weights are `(3, 1)` for
//! the two nearest chroma samples. At image edges the far sample is clamped
//! to the nearest (replicate) so the filter always has valid taps.

#![allow(dead_code)]

/// Identity upsample: one output row is the input row unchanged. Output width
/// equals input width. Used for 4:4:4 where no upsample is needed.
pub(crate) fn upsample_1x1(input: &[u8], output: &mut [u8]) {
    let n = input.len().min(output.len());
    output[..n].copy_from_slice(&input[..n]);
}

/// Horizontal fancy upsample (4:2:2). `input_row` has length `input_cols`;
/// `output_row` must have length `2 * input_cols`.
pub(crate) fn upsample_h2v1_fancy(input_row: &[u8], output_row: &mut [u8]) {
    let n = input_row.len();
    assert_eq!(output_row.len(), n * 2, "output row must be 2× input width");
    if n == 0 {
        return;
    }
    if n == 1 {
        output_row[0] = input_row[0];
        output_row[1] = input_row[0];
        return;
    }
    output_row[0] = input_row[0];
    output_row[1] = ((3 * input_row[0] as u32 + input_row[1] as u32 + 2) / 4) as u8;
    for i in 1..n - 1 {
        let prev = input_row[i - 1] as u32;
        let curr = input_row[i] as u32;
        let next = input_row[i + 1] as u32;
        output_row[2 * i] = ((3 * curr + prev + 2) / 4) as u8;
        output_row[2 * i + 1] = ((3 * curr + next + 2) / 4) as u8;
    }
    let last = input_row[n - 1] as u32;
    let before = input_row[n - 2] as u32;
    output_row[2 * n - 2] = ((3 * last + before + 2) / 4) as u8;
    output_row[2 * n - 1] = input_row[n - 1];
}

/// Produce two output rows for 4:2:0 vertical+horizontal fancy upsample.
pub(crate) fn upsample_h2v2_fancy(
    prev: &[u8],
    curr: &[u8],
    next: &[u8],
    out_top: &mut [u8],
    out_bot: &mut [u8],
) {
    let n = curr.len();
    assert_eq!(prev.len(), n);
    assert_eq!(next.len(), n);
    assert_eq!(out_top.len(), 2 * n);
    assert_eq!(out_bot.len(), 2 * n);
    if n == 0 {
        return;
    }

    for i in 0..n {
        let c = curr[i] as u32;
        let p = prev[i] as u32;
        let nx = next[i] as u32;
        let top_co = 3 * c + p;
        let bot_co = 3 * c + nx;

        let left = if i == 0 { i } else { i - 1 };
        let lc = curr[left] as u32;
        let lp = prev[left] as u32;
        let lnx = next[left] as u32;
        let top_l = 3 * lc + lp;
        let bot_l = 3 * lc + lnx;

        let right = if i + 1 == n { i } else { i + 1 };
        let rc = curr[right] as u32;
        let rp = prev[right] as u32;
        let rnx = next[right] as u32;
        let top_r = 3 * rc + rp;
        let bot_r = 3 * rc + rnx;

        out_top[2 * i] = ((3 * top_co + top_l + 8) >> 4) as u8;
        out_top[2 * i + 1] = ((3 * top_co + top_r + 8) >> 4) as u8;
        out_bot[2 * i] = ((3 * bot_co + bot_l + 8) >> 4) as u8;
        out_bot[2 * i + 1] = ((3 * bot_co + bot_r + 8) >> 4) as u8;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn upsample_1x1_is_memcpy() {
        let input = vec![1u8, 2, 3, 4];
        let mut output = vec![0u8; 4];
        upsample_1x1(&input, &mut output);
        assert_eq!(output, input);
    }

    #[test]
    fn h2v1_fancy_replicates_edges_and_interpolates_middle() {
        let input = vec![10u8, 20, 30, 40];
        let mut output = vec![0u8; 8];
        upsample_h2v1_fancy(&input, &mut output);
        assert_eq!(output[0], 10);
        assert_eq!(output[1], 13);
        assert_eq!(output[2], 18);
        assert_eq!(output[3], 23);
        assert_eq!(output[4], 28);
        assert_eq!(output[5], 33);
        assert_eq!(output[6], 38);
        assert_eq!(output[7], 40);
    }

    #[test]
    fn h2v1_fancy_handles_single_sample_row() {
        let input = vec![42u8];
        let mut output = vec![0u8; 2];
        upsample_h2v1_fancy(&input, &mut output);
        assert_eq!(output, vec![42, 42]);
    }

    #[test]
    fn h2v2_fancy_produces_uniform_output_for_uniform_input() {
        let row = vec![100u8; 4];
        let mut top = vec![0u8; 8];
        let mut bot = vec![0u8; 8];
        upsample_h2v2_fancy(&row, &row, &row, &mut top, &mut bot);
        assert!(top.iter().all(|&v| v == 100));
        assert!(bot.iter().all(|&v| v == 100));
    }

    #[test]
    fn h2v2_fancy_blends_toward_adjacent_row_asymmetrically() {
        let prev = vec![0u8; 2];
        let curr = vec![200u8; 2];
        let next = vec![200u8; 2];
        let mut top = vec![0u8; 4];
        let mut bot = vec![0u8; 4];
        upsample_h2v2_fancy(&prev, &curr, &next, &mut top, &mut bot);
        assert_eq!(top[0], 150);
        assert_eq!(bot[0], 200);
    }
}
