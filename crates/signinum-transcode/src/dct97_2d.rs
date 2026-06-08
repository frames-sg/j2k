// SPDX-License-Identifier: Apache-2.0

//! Constrained 2D DCT to irreversible 9/7 wavelet transforms.
//!
//! The production float path performs a separable 8x8 IDCT into a reusable
//! spatial plane, then applies the separable single-level 9/7 transform.

use core::f64::consts::PI;
use core::fmt;
use std::sync::LazyLock;

use rayon::prelude::*;

use crate::dct_grid::validate_dct_block_grid;

const ALPHA: f64 = -1.586_134_342_059_924;
const BETA: f64 = -0.052_980_118_572_961;
const GAMMA: f64 = 0.882_911_075_530_934;
const DELTA: f64 = 0.443_506_852_043_971;
const KAPPA: f64 = 1.230_174_104_914_001;
const INV_KAPPA: f64 = 1.0 / KAPPA;
const PARALLEL_IDCT_MIN_SAMPLES: usize = 64 * 64;

/// One separable single-level 2D 9/7 transform result.
#[derive(Debug, Clone, PartialEq)]
pub struct Dwt97TwoDimensional<T> {
    /// Low-horizontal, low-vertical band.
    pub ll: Vec<T>,
    /// High-horizontal, low-vertical band.
    pub hl: Vec<T>,
    /// Low-horizontal, high-vertical band.
    pub lh: Vec<T>,
    /// High-horizontal, high-vertical band.
    pub hh: Vec<T>,
    /// Width of horizontally low-pass bands.
    pub low_width: usize,
    /// Height of vertically low-pass bands.
    pub low_height: usize,
    /// Width of horizontally high-pass bands.
    pub high_width: usize,
    /// Height of vertically high-pass bands.
    pub high_height: usize,
}

/// Error returned when a DCT block grid cannot cover the requested component.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dct97GridError {
    block_count: usize,
    block_cols: usize,
    block_rows: usize,
    width: usize,
    height: usize,
}

impl Dct97GridError {
    /// Number of supplied 8x8 DCT blocks.
    #[must_use]
    pub const fn block_count(self) -> usize {
        self.block_count
    }

    /// Declared block columns.
    #[must_use]
    pub const fn block_cols(self) -> usize {
        self.block_cols
    }

    /// Declared block rows.
    #[must_use]
    pub const fn block_rows(self) -> usize {
        self.block_rows
    }

    /// Requested component width.
    #[must_use]
    pub const fn width(self) -> usize {
        self.width
    }

    /// Requested component height.
    #[must_use]
    pub const fn height(self) -> usize {
        self.height
    }
}

impl fmt::Display for Dct97GridError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DCT grid has {} blocks for {}x{} grid covering requested {}x{} samples",
            self.block_count, self.block_cols, self.block_rows, self.width, self.height
        )
    }
}

impl std::error::Error for Dct97GridError {}

/// Scratch storage for repeated DCT-grid to 9/7 transform calls.
#[derive(Debug, Default)]
pub struct Dct97GridScratch {
    spatial_samples: Vec<f64>,
    plane: Dwt97PlaneScratch,
}

#[derive(Debug, Default)]
struct Dwt97PlaneScratch {
    row_low: Vec<f64>,
    row_high: Vec<f64>,
    lift_workspace: Vec<f64>,
}

impl Dct97GridScratch {
    /// Capacity of the reusable spatial-sample buffer used by the IDCT-then
    /// 9/7 path.
    #[must_use]
    pub fn spatial_sample_capacity(&self) -> usize {
        self.spatial_samples.capacity()
    }
}

/// Reference path for a DCT block grid:
/// DCT coefficients -> float IDCT samples -> separable linearized 9/7.
pub fn dct8x8_blocks_then_dwt97_float(
    blocks: &[[[f64; 8]; 8]],
    block_cols: usize,
    block_rows: usize,
    width: usize,
    height: usize,
) -> Result<Dwt97TwoDimensional<f64>, Dct97GridError> {
    validate_grid(blocks.len(), block_cols, block_rows, width, height)?;

    let mut samples = Vec::with_capacity(width * height);
    for y in 0..height {
        let block_y = y / 8;
        let local_y = y % 8;
        for x in 0..width {
            let block_x = x / 8;
            let local_x = x % 8;
            let block = &blocks[block_y * block_cols + block_x];
            samples.push(idct8x8_sample(block, local_x, local_y));
        }
    }

    Ok(linearized_97_2d_from_plane(&samples, width, height))
}

/// Reference 9/7 path with caller-owned spatial-sample scratch:
/// DCT coefficients -> float IDCT samples -> separable linearized 9/7.
pub fn dct8x8_blocks_then_dwt97_float_with_scratch(
    blocks: &[[[f64; 8]; 8]],
    block_cols: usize,
    block_rows: usize,
    width: usize,
    height: usize,
    scratch: &mut Dct97GridScratch,
) -> Result<Dwt97TwoDimensional<f64>, Dct97GridError> {
    validate_grid(blocks.len(), block_cols, block_rows, width, height)?;

    let sample_count = width.saturating_mul(height);
    scratch.spatial_samples.clear();
    scratch.spatial_samples.resize(sample_count, 0.0);
    idct8x8_blocks_to_samples(
        blocks,
        block_cols,
        width,
        height,
        &mut scratch.spatial_samples,
    );

    Ok(linearized_97_2d_from_plane_with_plane_scratch(
        &scratch.spatial_samples,
        width,
        height,
        &mut scratch.plane,
    ))
}

pub(crate) fn linearized_97_2d_from_plane(
    samples: &[f64],
    width: usize,
    height: usize,
) -> Dwt97TwoDimensional<f64> {
    let mut scratch = Dct97GridScratch::default();
    linearized_97_2d_from_plane_with_scratch(samples, width, height, &mut scratch)
}

pub(crate) fn linearized_97_2d_from_plane_with_scratch(
    samples: &[f64],
    width: usize,
    height: usize,
    scratch: &mut Dct97GridScratch,
) -> Dwt97TwoDimensional<f64> {
    linearized_97_2d_from_plane_with_plane_scratch(samples, width, height, &mut scratch.plane)
}

fn linearized_97_2d_from_plane_with_plane_scratch(
    samples: &[f64],
    width: usize,
    height: usize,
    scratch: &mut Dwt97PlaneScratch,
) -> Dwt97TwoDimensional<f64> {
    debug_assert_eq!(samples.len(), width * height);

    let low_width = low_len(width);
    let low_height = low_len(height);
    let high_width = high_len(width);
    let high_height = high_len(height);

    scratch.row_low.clear();
    scratch.row_low.resize(height * low_width, 0.0);
    scratch.row_high.clear();
    scratch.row_high.resize(height * high_width, 0.0);

    for y in 0..height {
        let start = y * width;
        let row = &samples[start..start + width];
        let low_start = y * low_width;
        let high_start = y * high_width;
        linearized_97_split_contiguous_into(
            row,
            &mut scratch.row_low[low_start..low_start + low_width],
            &mut scratch.row_high[high_start..high_start + high_width],
            &mut scratch.lift_workspace,
        );
    }

    let mut ll = vec![0.0; low_width * low_height];
    let mut lh = vec![0.0; low_width * high_height];
    for x in 0..low_width {
        linearized_97_split_strided_into(
            &scratch.row_low,
            low_width,
            x,
            height,
            &mut ll,
            &mut lh,
            low_width,
            &mut scratch.lift_workspace,
        );
    }

    let mut hl = vec![0.0; high_width * low_height];
    let mut hh = vec![0.0; high_width * high_height];
    for x in 0..high_width {
        linearized_97_split_strided_into(
            &scratch.row_high,
            high_width,
            x,
            height,
            &mut hl,
            &mut hh,
            high_width,
            &mut scratch.lift_workspace,
        );
    }

    Dwt97TwoDimensional {
        ll,
        hl,
        lh,
        hh,
        low_width,
        low_height,
        high_width,
        high_height,
    }
}

fn idct8x8_sample(block: &[[f64; 8]; 8], x: usize, y: usize) -> f64 {
    let mut sample = 0.0;
    for (freq_y, row) in block.iter().enumerate() {
        let y_basis = idct8_basis(y, freq_y);
        for (freq_x, coefficient) in row.iter().copied().enumerate() {
            sample += coefficient * y_basis * idct8_basis(x, freq_x);
        }
    }
    sample
}

fn idct8x8_blocks_to_samples(
    blocks: &[[[f64; 8]; 8]],
    block_cols: usize,
    width: usize,
    height: usize,
    samples: &mut [f64],
) {
    debug_assert_eq!(samples.len(), width * height);
    let basis = idct8_basis_table();
    let active_block_cols = width.div_ceil(8);
    let active_block_rows = height.div_ceil(8);

    if width * height >= PARALLEL_IDCT_MIN_SAMPLES {
        samples
            .par_chunks_mut(width * 8)
            .enumerate()
            .take(active_block_rows)
            .for_each(|(block_y, sample_rows)| {
                idct8x8_block_row_to_samples(
                    blocks,
                    block_cols,
                    width,
                    height,
                    basis,
                    active_block_cols,
                    block_y,
                    sample_rows,
                );
            });
    } else {
        for block_y in 0..active_block_rows {
            let block_sample_y = block_y * 8;
            let output_rows = (height - block_sample_y).min(8);
            let row_start = block_sample_y * width;
            let row_end = row_start + output_rows * width;
            idct8x8_block_row_to_samples(
                blocks,
                block_cols,
                width,
                height,
                basis,
                active_block_cols,
                block_y,
                &mut samples[row_start..row_end],
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn idct8x8_block_row_to_samples(
    blocks: &[[[f64; 8]; 8]],
    block_cols: usize,
    width: usize,
    height: usize,
    basis: &[[f64; 8]; 8],
    active_block_cols: usize,
    block_y: usize,
    sample_rows: &mut [f64],
) {
    let block_sample_y = block_y * 8;
    let output_rows = (height - block_sample_y).min(8);
    for block_x in 0..active_block_cols {
        let block_sample_x = block_x * 8;
        let output_cols = (width - block_sample_x).min(8);
        let block = &blocks[block_y * block_cols + block_x];
        let mut vertical = [[0.0; 8]; 8];

        for (local_y, basis_row) in basis.iter().enumerate() {
            for freq_x in 0..8 {
                let mut sum = 0.0;
                for (freq_y, block_row) in block.iter().enumerate() {
                    sum += basis_row[freq_y] * block_row[freq_x];
                }
                vertical[local_y][freq_x] = sum;
            }
        }

        for (local_y, vertical_row) in vertical.iter().enumerate().take(output_rows) {
            let row_offset = local_y * width + block_sample_x;
            for local_x in 0..output_cols {
                let mut sample = 0.0;
                for (freq_x, vertical_value) in vertical_row.iter().enumerate() {
                    sample += *vertical_value * basis[local_x][freq_x];
                }
                sample_rows[row_offset + local_x] = sample;
            }
        }
    }
}

#[cfg(test)]
fn linearized_97_from_sample_slice(samples: &[f64]) -> Dwt97OneDimensional {
    let mut lifted = samples.to_vec();
    forward_lift_97(&mut lifted);

    Dwt97OneDimensional {
        low: lifted.iter().step_by(2).copied().collect(),
        high: lifted.iter().skip(1).step_by(2).copied().collect(),
    }
}

fn forward_lift_97(data: &mut [f64]) {
    let n = data.len();
    if n < 2 {
        return;
    }

    let last_even = if n.is_multiple_of(2) { n - 2 } else { n - 1 };

    for i in (1..n).step_by(2) {
        let left = data[i - 1];
        let right = if i + 1 < n {
            data[i + 1]
        } else {
            data[last_even]
        };
        data[i] += ALPHA * (left + right);
    }

    for i in (0..n).step_by(2) {
        let left = if i > 0 { data[i - 1] } else { data[1] };
        let right = if i + 1 < n { data[i + 1] } else { left };
        data[i] += BETA * (left + right);
    }

    for i in (1..n).step_by(2) {
        let left = data[i - 1];
        let right = if i + 1 < n {
            data[i + 1]
        } else {
            data[last_even]
        };
        data[i] += GAMMA * (left + right);
    }

    for i in (0..n).step_by(2) {
        let left = if i > 0 { data[i - 1] } else { data[1] };
        let right = if i + 1 < n { data[i + 1] } else { left };
        data[i] += DELTA * (left + right);
    }

    for i in (0..n).step_by(2) {
        data[i] *= INV_KAPPA;
    }
    for i in (1..n).step_by(2) {
        data[i] *= KAPPA;
    }
}

fn linearized_97_split_contiguous_into(
    samples: &[f64],
    low: &mut [f64],
    high: &mut [f64],
    workspace: &mut Vec<f64>,
) {
    debug_assert_eq!(low.len(), low_len(samples.len()));
    debug_assert_eq!(high.len(), high_len(samples.len()));

    workspace.clear();
    workspace.extend_from_slice(samples);
    forward_lift_97(workspace);

    for (target, value) in low.iter_mut().zip(workspace.iter().step_by(2)) {
        *target = *value;
    }
    for (target, value) in high.iter_mut().zip(workspace.iter().skip(1).step_by(2)) {
        *target = *value;
    }
}

#[allow(clippy::too_many_arguments)]
fn linearized_97_split_strided_into(
    samples: &[f64],
    stride: usize,
    x: usize,
    height: usize,
    low: &mut [f64],
    high: &mut [f64],
    band_width: usize,
    workspace: &mut Vec<f64>,
) {
    debug_assert_eq!(low.len(), band_width * low_len(height));
    debug_assert_eq!(high.len(), band_width * high_len(height));

    workspace.clear();
    workspace.extend((0..height).map(|y| samples[y * stride + x]));
    forward_lift_97(workspace);

    for (low_y, value) in workspace.iter().step_by(2).enumerate() {
        low[low_y * band_width + x] = *value;
    }
    for (high_y, value) in workspace.iter().skip(1).step_by(2).enumerate() {
        high[high_y * band_width + x] = *value;
    }
}

fn idct8_basis(sample_idx: usize, freq: usize) -> f64 {
    debug_assert!(sample_idx < 8);
    debug_assert!(freq < 8);

    idct8_basis_table()[sample_idx][freq]
}

fn idct8_basis_table() -> &'static [[f64; 8]; 8] {
    static BASIS: LazyLock<[[f64; 8]; 8]> = LazyLock::new(|| {
        let mut basis = [[0.0; 8]; 8];
        for (sample_idx, row) in basis.iter_mut().enumerate() {
            for (freq, value) in row.iter_mut().enumerate() {
                *value = idct8_basis_uncached(sample_idx, freq);
            }
        }
        basis
    });
    &BASIS
}

fn idct8_basis_uncached(sample_idx: usize, freq: usize) -> f64 {
    let scale = if freq == 0 {
        (1.0_f64 / 8.0).sqrt()
    } else {
        (2.0_f64 / 8.0).sqrt()
    };
    scale * (((sample_idx as f64 + 0.5) * freq as f64 * PI) / 8.0).cos()
}

fn low_len(sample_len: usize) -> usize {
    sample_len.div_ceil(2)
}

fn high_len(sample_len: usize) -> usize {
    sample_len / 2
}

fn validate_grid(
    block_count: usize,
    block_cols: usize,
    block_rows: usize,
    width: usize,
    height: usize,
) -> Result<(), Dct97GridError> {
    validate_dct_block_grid(block_count, block_cols, block_rows, width, height).map_err(|()| {
        Dct97GridError {
            block_count,
            block_cols,
            block_rows,
            width,
            height,
        }
    })
}

#[cfg(test)]
struct Dwt97OneDimensional {
    low: Vec<f64>,
    high: Vec<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_all_close(values: &[f64], expected: f64, epsilon: f64) {
        for &value in values {
            assert!(
                (value - expected).abs() < epsilon,
                "value={value} expected={expected} values={values:?}"
            );
        }
    }

    #[test]
    fn linearized_97_from_constant_signal_places_dc_in_low_pass() {
        for len in [2usize, 3, 8, 9, 64, 65] {
            let samples = vec![50.0; len];

            let transformed = linearized_97_from_sample_slice(&samples);

            assert_all_close(&transformed.low, 50.0, 0.001);
            assert_all_close(&transformed.high, 0.0, 0.001);
        }
    }

    #[test]
    fn linearized_97_2d_from_constant_plane_places_dc_in_ll() {
        for (width, height) in [(8usize, 8usize), (9, 7)] {
            let samples = vec![50.0; width * height];

            let transformed = linearized_97_2d_from_plane(&samples, width, height);

            assert_all_close(&transformed.ll, 50.0, 0.001);
            assert_all_close(&transformed.hl, 0.0, 0.001);
            assert_all_close(&transformed.lh, 0.0, 0.001);
            assert_all_close(&transformed.hh, 0.0, 0.001);
        }
    }

    // -------------------------------------------------------------------------
    // Independent CDF 9/7 ground truth.
    //
    // The CUDA 9/7 kernel is parity-tested against `forward_lift_97` /
    // `linearized_97_2d_from_plane`, so a bug in the lifting would be faithfully
    // reproduced by the kernel and pass that parity test unnoticed. These tests
    // close that gap by validating the lifting against an *independent*
    // implementation: a direct FIR filter bank using the canonical, fully
    // normalized CDF 9/7 analysis taps and JPEG2000 whole-sample symmetric
    // extension. Different arithmetic, same transform.
    //
    // The taps themselves are checked against their defining mathematical
    // properties (DC gains and high-pass vanishing moments) so they cannot
    // silently drift to "match" a buggy lifting.

    /// Canonical CDF 9/7 analysis low-pass filter (9 taps, even-symmetric).
    /// Fully normalized so its DC gain is 1 (a constant maps unchanged into the
    /// low band, matching the lifting's `INV_KAPPA` scaling).
    const REF_LP: [f64; 9] = [
        0.026_748_757_410_810,
        -0.016_864_118_442_875,
        -0.078_223_266_528_990,
        0.266_864_118_442_875,
        0.602_949_018_236_360,
        0.266_864_118_442_875,
        -0.078_223_266_528_990,
        -0.016_864_118_442_875,
        0.026_748_757_410_810,
    ];

    /// Canonical CDF 9/7 analysis high-pass filter (7 taps, even-symmetric).
    /// Fully normalized so its DC gain is 0 (matching the lifting's `KAPPA`
    /// scaling); it has four vanishing moments.
    const REF_HP: [f64; 7] = [
        0.091_271_763_114_250,
        -0.057_543_526_228_500,
        -0.591_271_763_114_247,
        1.115_087_052_456_994,
        -0.591_271_763_114_247,
        -0.057_543_526_228_500,
        0.091_271_763_114_250,
    ];

    /// Whole-sample symmetric reflection: mirror about index 0 and `n - 1`
    /// without repeating the endpoints. This is the boundary extension
    /// `forward_lift_97` implements at the array edges.
    fn ws_reflect(i: isize, n: usize) -> usize {
        debug_assert!(n >= 1);
        if n == 1 {
            return 0;
        }
        let n = isize::try_from(n).expect("signal length fits in isize");
        let period = 2 * (n - 1);
        let mut k = i.rem_euclid(period);
        if k >= n {
            k = period - k;
        }
        usize::try_from(k).expect("reflected index is non-negative")
    }

    /// Independent single-level forward 9/7 analysis via direct convolution.
    /// Returns `(low, high)` interleaved-position bands matching `forward_lift_97`
    /// (`low[m]` centered at sample `2m`, `high[m]` centered at sample `2m + 1`).
    fn ref_analysis_1d(signal: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = signal.len();
        if n < 2 {
            // The lifting leaves <2-length signals unchanged (low = the sample).
            return (signal.to_vec(), Vec::new());
        }
        let mut low = vec![0.0; low_len(n)];
        let mut high = vec![0.0; high_len(n)];
        for (m, out) in low.iter_mut().enumerate() {
            let center = 2 * isize::try_from(m).unwrap();
            *out = REF_LP
                .iter()
                .enumerate()
                .map(|(t, &tap)| {
                    tap * signal[ws_reflect(center + isize::try_from(t).unwrap() - 4, n)]
                })
                .sum();
        }
        for (m, out) in high.iter_mut().enumerate() {
            let center = 2 * isize::try_from(m).unwrap() + 1;
            *out = REF_HP
                .iter()
                .enumerate()
                .map(|(t, &tap)| {
                    tap * signal[ws_reflect(center + isize::try_from(t).unwrap() - 3, n)]
                })
                .sum();
        }
        (low, high)
    }

    /// Independent separable 2D forward 9/7 (rows then columns) producing the
    /// same four-band layout as `linearized_97_2d_from_plane`.
    fn ref_analysis_2d(samples: &[f64], width: usize, height: usize) -> Dwt97TwoDimensional<f64> {
        let low_width = low_len(width);
        let high_width = high_len(width);
        let low_height = low_len(height);
        let high_height = high_len(height);

        let mut row_low = vec![0.0; height * low_width];
        let mut row_high = vec![0.0; height * high_width];
        for y in 0..height {
            let (lo, hi) = ref_analysis_1d(&samples[y * width..y * width + width]);
            row_low[y * low_width..y * low_width + low_width].copy_from_slice(&lo);
            row_high[y * high_width..y * high_width + high_width].copy_from_slice(&hi);
        }

        let vertical_split = |source: &[f64], band_width: usize| -> (Vec<f64>, Vec<f64>) {
            let mut low = vec![0.0; band_width * low_height];
            let mut high = vec![0.0; band_width * high_height];
            for x in 0..band_width {
                let column: Vec<f64> = (0..height).map(|y| source[y * band_width + x]).collect();
                let (lo, hi) = ref_analysis_1d(&column);
                for (vy, &value) in lo.iter().enumerate() {
                    low[vy * band_width + x] = value;
                }
                for (vy, &value) in hi.iter().enumerate() {
                    high[vy * band_width + x] = value;
                }
            }
            (low, high)
        };

        let (ll, lh) = vertical_split(&row_low, low_width);
        let (hl, hh) = vertical_split(&row_high, high_width);

        Dwt97TwoDimensional {
            ll,
            hl,
            lh,
            hh,
            low_width,
            low_height,
            high_width,
            high_height,
        }
    }

    /// Small deterministic PRNG (LCG) for reproducible test signals in [-1, 1).
    fn next_unit(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((*state >> 11) as f64 / (1u64 << 53) as f64).mul_add(2.0, -1.0)
    }

    fn assert_bands_close(actual: &[f64], expected: &[f64], label: &str, epsilon: f64) {
        assert_eq!(actual.len(), expected.len(), "{label} band length");
        for (i, (a, b)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - b).abs() <= epsilon,
                "{label}[{i}] diverged: lifting={a} reference={b} (diff {})",
                (a - b).abs()
            );
        }
    }

    #[test]
    fn reference_cdf97_taps_satisfy_their_defining_properties() {
        // Low-pass DC gain 1, high-pass DC gain 0 — the normalization the
        // lifting's KAPPA scaling targets.
        let lp_dc: f64 = REF_LP.iter().sum();
        assert!((lp_dc - 1.0).abs() < 1e-9, "low-pass DC gain = {lp_dc}");
        let hp_dc: f64 = REF_HP.iter().sum();
        assert!(hp_dc.abs() < 1e-9, "high-pass DC gain = {hp_dc}");

        // Even symmetry.
        for k in 0..4 {
            assert!(
                (REF_LP[k] - REF_LP[8 - k]).abs() < 1e-15,
                "low-pass asymmetric at {k}"
            );
        }
        for k in 0..3 {
            assert!(
                (REF_HP[k] - REF_HP[6 - k]).abs() < 1e-15,
                "high-pass asymmetric at {k}"
            );
        }

        // Four vanishing moments: the high-pass annihilates polynomials of
        // degree <= 3 (so a wrong predict coefficient or sign cannot pass).
        for m in 1..=3 {
            let moment: f64 = REF_HP
                .iter()
                .enumerate()
                .map(|(k, &tap)| (k as f64 - 3.0).powi(m) * tap)
                .sum();
            assert!(moment.abs() < 1e-9, "high-pass moment {m} = {moment}");
        }
    }

    #[test]
    fn forward_lift_97_matches_independent_filter_bank_1d() {
        let mut state = 0x1234_5678_9abc_def0u64;
        for n in [2usize, 3, 4, 5, 8, 9, 12, 15, 16, 23, 32, 33, 64, 65] {
            let signal: Vec<f64> = (0..n).map(|_| next_unit(&mut state) * 100.0).collect();
            let lifted = linearized_97_from_sample_slice(&signal);
            let (low, high) = ref_analysis_1d(&signal);
            assert_bands_close(&lifted.low, &low, &format!("n={n} low"), 1e-9);
            assert_bands_close(&lifted.high, &high, &format!("n={n} high"), 1e-9);
        }
    }

    #[test]
    fn forward_lift_97_annihilates_low_degree_polynomials() {
        // Independent of the filter bank: a correct 9/7 high-pass kills cubics in
        // the interior (boundary coefficients use symmetric extension). This pins
        // the predict-step coefficients and signs directly from wavelet theory.
        let n = 40usize;
        let polynomials: [[f64; 4]; 4] = [
            [5.0, 0.0, 0.0, 0.0],
            [0.0, 2.5, 0.0, 0.0],
            [1.0, -0.7, 0.3, 0.0],
            [0.0, 0.0, 0.0, 0.05],
        ];
        for coeffs in polynomials {
            let signal: Vec<f64> = (0..n)
                .map(|i| {
                    let x = i as f64;
                    coeffs[3].mul_add(
                        x * x * x,
                        coeffs[2].mul_add(x * x, coeffs[1].mul_add(x, coeffs[0])),
                    )
                })
                .collect();
            let lifted = linearized_97_from_sample_slice(&signal);
            // Skip the first/last high-pass coefficients (boundary support).
            let interior = &lifted.high[3..lifted.high.len() - 3];
            assert_all_close(interior, 0.0, 1e-6);
        }
    }

    #[test]
    fn linearized_97_2d_matches_independent_separable_filter_bank() {
        let mut state = 0xfeed_face_dead_beefu64;
        for (width, height) in [
            (8usize, 8usize),
            (16, 16),
            (24, 16),
            (15, 13),
            (16, 23),
            (9, 7),
            (32, 32),
        ] {
            let samples: Vec<f64> = (0..width * height)
                .map(|_| next_unit(&mut state) * 100.0)
                .collect();
            let got = linearized_97_2d_from_plane(&samples, width, height);
            let want = ref_analysis_2d(&samples, width, height);
            assert_eq!(
                (
                    got.low_width,
                    got.low_height,
                    got.high_width,
                    got.high_height
                ),
                (
                    want.low_width,
                    want.low_height,
                    want.high_width,
                    want.high_height
                ),
                "band dimensions for {width}x{height}"
            );
            assert_bands_close(&got.ll, &want.ll, &format!("{width}x{height} ll"), 1e-9);
            assert_bands_close(&got.hl, &want.hl, &format!("{width}x{height} hl"), 1e-9);
            assert_bands_close(&got.lh, &want.lh, &format!("{width}x{height} lh"), 1e-9);
            assert_bands_close(&got.hh, &want.hh, &format!("{width}x{height} hh"), 1e-9);
        }
    }

    #[test]
    fn linearized_97_2d_separates_horizontal_and_vertical_detail() {
        // Catches an HL/LH swap or a row/column transpose independently of the
        // filter bank: a plane that varies only along x has no vertical detail
        // (LH and HH must vanish), and vice versa.
        let (width, height) = (16usize, 16usize);

        let varies_in_x: Vec<f64> = (0..width * height)
            .map(|i| ((i % width) as f64).sin().mul_add(30.0, 5.0))
            .collect();
        let t = linearized_97_2d_from_plane(&varies_in_x, width, height);
        assert_all_close(&t.lh, 0.0, 1e-9);
        assert_all_close(&t.hh, 0.0, 1e-9);

        let varies_in_y: Vec<f64> = (0..width * height)
            .map(|i| ((i / width) as f64).cos().mul_add(30.0, 5.0))
            .collect();
        let t = linearized_97_2d_from_plane(&varies_in_y, width, height);
        assert_all_close(&t.hl, 0.0, 1e-9);
        assert_all_close(&t.hh, 0.0, 1e-9);
    }

    // -------------------------------------------------------------------------
    // Ground truth: exact mathematical inverse DCT for the float 9/7 path.
    //
    // The 9/7 transcode oracle (`dct8x8_blocks_then_dwt97_float`) feeds
    // `idct8x8_sample` into the wavelet. Validate that IDCT against the defining
    // DCT-III cosine sum so a basis/normalization/transpose bug cannot hide
    // inside both the oracle and its CUDA port.
    fn exact_idct_sample(block: &[[f64; 8]; 8], x: usize, y: usize) -> f64 {
        let alpha = |k: usize| {
            if k == 0 {
                (1.0_f64 / 8.0).sqrt()
            } else {
                (2.0_f64 / 8.0).sqrt()
            }
        };
        let cos_term = |sample: usize, freq: usize| {
            (((2 * sample + 1) as f64) * freq as f64 * PI / 16.0).cos()
        };
        let mut acc = 0.0;
        for (v, row) in block.iter().enumerate() {
            for (u, &coeff) in row.iter().enumerate() {
                acc += alpha(u) * alpha(v) * coeff * cos_term(x, u) * cos_term(y, v);
            }
        }
        acc
    }

    #[test]
    fn idct8x8_sample_matches_exact_cosine_sum() {
        let mut state = 0x5151_aaaa_bbbb_ccccu64;
        for _ in 0..64 {
            let mut block = [[0.0f64; 8]; 8];
            for row in &mut block {
                for coeff in row {
                    *coeff = next_unit(&mut state) * 64.0;
                }
            }
            for y in 0..8 {
                for x in 0..8 {
                    let got = idct8x8_sample(&block, x, y);
                    let want = exact_idct_sample(&block, x, y);
                    assert!(
                        (got - want).abs() < 1e-9,
                        "idct8x8_sample({x},{y})={got} exact={want}"
                    );
                }
            }
        }
    }

    #[test]
    fn idct8x8_sample_dc_only_is_uniform() {
        // DC-only block -> uniform plane equal to F(0,0) / 8.
        let mut block = [[0.0f64; 8]; 8];
        block[0][0] = 320.0;
        for y in 0..8 {
            for x in 0..8 {
                assert!((idct8x8_sample(&block, x, y) - 40.0).abs() < 1e-9);
            }
        }
    }
}
