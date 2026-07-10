// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared reversible integer 5/3 lifting helpers for transcode paths.

use core::convert::Infallible;

mod in_place;

pub(crate) use in_place::{floor_div_i32, reversible_lift_53_i32};

pub(crate) fn reversible_lift_53_low_at(
    sample_len: usize,
    low_idx: usize,
    mut sample: impl FnMut(usize) -> i32,
) -> i32 {
    let mut fallible_sample = |idx| Ok::<i32, Infallible>(sample(idx));
    match reversible_lift_53_low_at_fallible(sample_len, low_idx, &mut fallible_sample) {
        Ok(value) => value,
        Err(err) => match err {},
    }
}

pub(crate) fn reversible_lift_53_high_at(
    sample_len: usize,
    high_idx: usize,
    mut sample: impl FnMut(usize) -> i32,
) -> i32 {
    let mut fallible_sample = |idx| Ok::<i32, Infallible>(sample(idx));
    match reversible_lift_53_high_at_fallible(sample_len, high_idx, &mut fallible_sample) {
        Ok(value) => value,
        Err(err) => match err {},
    }
}

pub(crate) fn reversible_lift_53_low_at_fallible<E>(
    sample_len: usize,
    low_idx: usize,
    mut sample: impl FnMut(usize) -> Result<i32, E>,
) -> Result<i32, E> {
    reversible_lift_53_low_at_with(sample_len, low_idx, &mut sample)
}

pub(crate) fn reversible_lift_53_high_at_fallible<E>(
    sample_len: usize,
    high_idx: usize,
    mut sample: impl FnMut(usize) -> Result<i32, E>,
) -> Result<i32, E> {
    reversible_lift_53_high_at_with(sample_len, high_idx, &mut sample)
}

fn reversible_lift_53_low_at_with<E>(
    sample_len: usize,
    low_idx: usize,
    sample: &mut impl FnMut(usize) -> Result<i32, E>,
) -> Result<i32, E> {
    let even_idx = low_idx * 2;
    let current = sample(even_idx)?;
    if sample_len < 2 {
        return Ok(current);
    }

    if sample_len.is_multiple_of(2) {
        let right = reversible_lift_53_high_at_with(sample_len, low_idx, sample)?;
        if low_idx == 0 {
            return Ok(current + floor_div_i32(right + 1, 2));
        }
        let left = reversible_lift_53_high_at_with(sample_len, low_idx - 1, sample)?;
        return Ok(current + floor_div_i32(left + right + 2, 4));
    }

    let high_len = sample_len / 2;
    if high_len == 0 {
        return Ok(current);
    }
    let left = if low_idx > 0 {
        reversible_lift_53_high_at_with(sample_len, low_idx - 1, sample)?
    } else {
        reversible_lift_53_high_at_with(sample_len, 0, sample)?
    };
    let right = if low_idx < high_len {
        reversible_lift_53_high_at_with(sample_len, low_idx, sample)?
    } else {
        left
    };
    Ok(current + floor_div_i32(left + right + 2, 4))
}

fn reversible_lift_53_high_at_with<E>(
    sample_len: usize,
    high_idx: usize,
    sample: &mut impl FnMut(usize) -> Result<i32, E>,
) -> Result<i32, E> {
    let odd_idx = high_idx * 2 + 1;
    let current = sample(odd_idx)?;
    let left = sample(odd_idx - 1)?;
    if sample_len.is_multiple_of(2) && odd_idx + 1 == sample_len {
        return Ok(current - left);
    }

    let right_idx = if odd_idx + 1 < sample_len {
        odd_idx + 1
    } else {
        sample_len - 1
    };
    let right = sample(right_idx)?;
    Ok(current - floor_div_i32(left + right, 2))
}

#[cfg(test)]
mod tests {
    use super::{reversible_lift_53_high_at, reversible_lift_53_i32, reversible_lift_53_low_at};

    #[test]
    fn indexed_lift_matches_in_place_lift_for_varied_lengths() {
        for sample_len in 1_usize..=17 {
            let samples: Vec<i32> = (0..sample_len)
                .map(|idx| {
                    let idx = i32::try_from(idx).expect("test index fits in i32");
                    let value = (idx * 37) - (idx % 5) * 19;
                    if idx % 2 == 0 {
                        value
                    } else {
                        -value
                    }
                })
                .collect();
            let mut lifted = samples.clone();
            reversible_lift_53_i32(&mut lifted);

            let low_len = sample_len.div_ceil(2);
            for low_idx in 0..low_len {
                assert_eq!(
                    reversible_lift_53_low_at(sample_len, low_idx, |idx| samples[idx]),
                    lifted[low_idx * 2],
                    "low {low_idx} for len {sample_len}"
                );
            }
            let high_len = sample_len / 2;
            for high_idx in 0..high_len {
                assert_eq!(
                    reversible_lift_53_high_at(sample_len, high_idx, |idx| samples[idx]),
                    lifted[high_idx * 2 + 1],
                    "high {high_idx} for len {sample_len}"
                );
            }
        }
    }
}
