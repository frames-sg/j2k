// SPDX-License-Identifier: MIT OR Apache-2.0

/// Maximum number of input taps contributing to one linearized 5/3 output.
pub const DWT53_MAX_LINEAR_TAPS: usize = 5;
/// Maximum number of input taps contributing to one high-pass 5/3 output.
pub const DWT53_MAX_HIGH_LINEAR_TAPS: usize = 3;

/// Linearized 5/3 output band.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Dwt53Band {
    /// Low-pass output samples.
    Low,
    /// High-pass output samples.
    High,
}

/// One non-zero input contribution to a linearized 5/3 output sample.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dwt53LinearTap {
    sample_index: usize,
    weight: f64,
}

impl Dwt53LinearTap {
    const ZERO: Self = Self {
        sample_index: 0,
        weight: 0.0,
    };

    /// Input sample index.
    #[must_use]
    pub const fn sample_index(self) -> usize {
        self.sample_index
    }

    /// Weight applied to the input sample.
    #[must_use]
    pub const fn weight(self) -> f64 {
        self.weight
    }
}

/// Fixed-capacity sparse row for one linearized 5/3 output sample.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dwt53LinearRow {
    taps: [Dwt53LinearTap; DWT53_MAX_LINEAR_TAPS],
    len: usize,
}

impl Dwt53LinearRow {
    const EMPTY: Self = Self {
        taps: [Dwt53LinearTap::ZERO; DWT53_MAX_LINEAR_TAPS],
        len: 0,
    };

    /// Non-zero taps in increasing input-index order.
    #[must_use]
    pub fn taps(&self) -> &[Dwt53LinearTap] {
        &self.taps[..self.len]
    }

    fn push(&mut self, sample_index: usize, weight: f64) -> Option<()> {
        if weight == 0.0 {
            return Some(());
        }
        let tap = self.taps.get_mut(self.len)?;
        *tap = Dwt53LinearTap {
            sample_index,
            weight,
        };
        self.len += 1;
        Some(())
    }
}

/// Derive one conventional linearized 5/3 analysis row in constant work.
///
/// Returns `None` when `output_index` is outside the selected band's output
/// extent. Symmetric boundary extension is folded into the returned weights.
#[must_use]
pub fn linearized_dwt53_row(
    sample_len: usize,
    band: Dwt53Band,
    output_index: usize,
) -> Option<Dwt53LinearRow> {
    let output_len = match band {
        Dwt53Band::Low => low_len(sample_len),
        Dwt53Band::High => high_len(sample_len),
    };
    if output_index >= output_len {
        return None;
    }

    let parity = usize::from(matches!(band, Dwt53Band::High));
    let center = output_index.checked_mul(2)?.checked_add(parity)?;
    let radius = if matches!(band, Dwt53Band::Low) { 2 } else { 1 };
    let first = center.saturating_sub(radius);
    let last = center.saturating_add(radius).min(sample_len - 1);
    let mut row = Dwt53LinearRow::EMPTY;
    for sample_index in first..=last {
        let weight = match band {
            Dwt53Band::Low => low_weight(sample_len, output_index, sample_index),
            Dwt53Band::High => high_weight(sample_len, output_index, sample_index),
        };
        row.push(sample_index, weight)?;
    }
    Some(row)
}

const fn low_len(sample_len: usize) -> usize {
    sample_len / 2 + sample_len % 2
}

const fn high_len(sample_len: usize) -> usize {
    sample_len / 2
}

fn low_weight(sample_len: usize, low_index: usize, sample_index: usize) -> f64 {
    let even_index = low_index * 2;
    let mut weight = delta(sample_index, even_index);
    let high_count = high_len(sample_len);
    let left_high = low_index.checked_sub(1);
    let right_high = (low_index < high_count).then_some(low_index);
    match (left_high, right_high) {
        (Some(left), Some(right)) => {
            weight += 0.25 * high_weight(sample_len, left, sample_index);
            weight += 0.25 * high_weight(sample_len, right, sample_index);
        }
        (None, Some(right)) => {
            weight += 0.5 * high_weight(sample_len, right, sample_index);
        }
        (Some(left), None) => {
            weight += 0.5 * high_weight(sample_len, left, sample_index);
        }
        (None, None) => {}
    }
    weight
}

fn high_weight(sample_len: usize, high_index: usize, sample_index: usize) -> f64 {
    let odd_index = high_index * 2 + 1;
    let left_even = odd_index - 1;
    let right_even = if odd_index + 1 < sample_len {
        odd_index + 1
    } else {
        left_even
    };
    delta(sample_index, odd_index)
        - 0.5 * (delta(sample_index, left_even) + delta(sample_index, right_even))
}

fn delta(left: usize, right: usize) -> f64 {
    f64::from(left == right)
}

#[cfg(test)]
mod tests {
    use super::{linearized_dwt53_row, Dwt53Band, DWT53_MAX_LINEAR_TAPS};

    #[test]
    fn symbolic_rows_match_the_canonical_linearized_transform() {
        let samples = [
            11.0, -7.0, 23.0, 5.0, -19.0, 31.0, 2.0, 17.0, -13.0, 29.0, 3.0, -5.0, 37.0, -11.0,
            7.0, 41.0, -17.0,
        ];
        for sample_len in 1..=samples.len() {
            for low_index in 0..sample_len.div_ceil(2) {
                let row = linearized_dwt53_row(sample_len, Dwt53Band::Low, low_index)
                    .expect("in-range low row");
                assert_close(
                    apply(&row, &samples),
                    direct_low(&samples[..sample_len], low_index),
                );
            }
            for high_index in 0..sample_len / 2 {
                let row = linearized_dwt53_row(sample_len, Dwt53Band::High, high_index)
                    .expect("in-range high row");
                assert_close(
                    apply(&row, &samples),
                    direct_high(&samples[..sample_len], high_index),
                );
            }
        }
    }

    #[test]
    fn maximum_jpeg_axis_builds_one_fixed_size_row_per_output() {
        let sample_len = 65_535usize;
        let mut row_count = 0usize;
        let mut tap_count = 0usize;
        for (band, output_len) in [
            (Dwt53Band::Low, sample_len.div_ceil(2)),
            (Dwt53Band::High, sample_len / 2),
        ] {
            for output_index in 0..output_len {
                let row = linearized_dwt53_row(sample_len, band, output_index)
                    .expect("in-range large-axis row");
                assert!(row.taps().len() <= DWT53_MAX_LINEAR_TAPS);
                assert!(row.taps().iter().all(|tap| tap.sample_index() < sample_len));
                row_count += 1;
                tap_count += row.taps().len();
            }
        }
        assert_eq!(row_count, sample_len);
        assert!(tap_count <= sample_len * DWT53_MAX_LINEAR_TAPS);
    }

    fn apply(row: &super::Dwt53LinearRow, samples: &[f64]) -> f64 {
        row.taps()
            .iter()
            .map(|tap| samples[tap.sample_index()] * tap.weight())
            .sum()
    }

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() <= f64::EPSILON,
            "actual={actual} expected={expected}"
        );
    }

    fn direct_low(samples: &[f64], low_index: usize) -> f64 {
        let current = samples[low_index * 2];
        let left = low_index
            .checked_sub(1)
            .map(|index| direct_high(samples, index));
        let right = (low_index < samples.len() / 2).then(|| direct_high(samples, low_index));
        current
            + match (left, right) {
                (Some(left), Some(right)) => (left + right) * 0.25,
                (None, Some(right)) => right * 0.5,
                (Some(left), None) => left * 0.5,
                (None, None) => 0.0,
            }
    }

    fn direct_high(samples: &[f64], high_index: usize) -> f64 {
        let odd_index = high_index * 2 + 1;
        let left = samples[odd_index - 1];
        let right = samples.get(odd_index + 1).copied().unwrap_or(left);
        samples[odd_index] - (left + right) * 0.5
    }
}
