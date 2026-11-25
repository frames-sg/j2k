// SPDX-License-Identifier: MIT OR Apache-2.0

//! Integer reference of one level of the reversible 5/3 inverse wavelet
//! transform (T.800 F.3.8), for checking the GPU kernels against it.

/// A `width` x `height` window whose first sample is at reference-grid
/// coordinate (`x0`, `y0`).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Dwt53Window {
    pub width: u32,
    pub height: u32,
    pub x0: u32,
    pub y0: u32,
}

impl Dwt53Window {
    /// Creates a window.
    pub const fn new(width: u32, height: u32, x0: u32, y0: u32) -> Self {
        Self {
            width,
            height,
            x0,
            y0,
        }
    }

    /// Widths and heights of the window's LL, HL, LH and HH sub-bands.
    pub const fn band_dims(self) -> [(u32, u32); 4] {
        let (low_w, high_w) = low_high(self.x0, self.width);
        let (low_h, high_h) = low_high(self.y0, self.height);
        [
            (low_w, low_h),
            (high_w, low_h),
            (low_w, high_h),
            (high_w, high_h),
        ]
    }

    /// Number of samples in each sub-band.
    pub const fn band_lens(self) -> [usize; 4] {
        let [ll, hl, lh, hh] = self.band_dims();
        [
            (ll.0 * ll.1) as usize,
            (hl.0 * hl.1) as usize,
            (lh.0 * lh.1) as usize,
            (hh.0 * hh.1) as usize,
        ]
    }
}

/// Number of low-pass (even) and high-pass (odd) coordinates in
/// `origin..origin + len`.
const fn low_high(origin: u32, len: u32) -> (u32, u32) {
    let end = origin + len;
    (end.div_ceil(2) - origin.div_ceil(2), end / 2 - origin / 2)
}

/// Windows one sample wide or tall.
///
/// A lone sample at an odd coordinate is high-pass, and the inverse divides it
/// by two with floor division. The list has lone samples at every parity, lone
/// columns and rows that the other axis then lifts, and even origins as the
/// control: their lone sample is low-pass and stays as it is.
pub const DWT53_LONE_SAMPLE_WINDOWS: [Dwt53Window; 10] = [
    Dwt53Window::new(1, 1, 1, 0),
    Dwt53Window::new(1, 1, 0, 1),
    Dwt53Window::new(1, 1, 1, 1),
    Dwt53Window::new(1, 1, 0, 0),
    Dwt53Window::new(1, 7, 1, 0),
    Dwt53Window::new(1, 6, 3, 1),
    Dwt53Window::new(1, 5, 2, 0),
    Dwt53Window::new(7, 1, 0, 1),
    Dwt53Window::new(6, 1, 1, 3),
    Dwt53Window::new(5, 1, 0, 2),
];

/// Sample `index` of sub-band `band` (0 LL, 1 HL, 2 LH, 3 HH) of batch item
/// `instance`.
fn coefficient(instance: usize, band: usize, index: usize) -> i32 {
    let raw = (index * 37 + band * 12 + instance * 6 + 1) % 61;
    i32::try_from(raw).expect("coefficient below 61") - 30
}

/// Integer sub-band samples of `window` for batch item `instance`.
///
/// They are odd and even, positive and negative, and the first of each band
/// is odd, so every lone sample is.
pub fn dwt53_bands(window: Dwt53Window, instance: usize) -> [Vec<i32>; 4] {
    let lens = window.band_lens();
    std::array::from_fn(|band| {
        (0..lens[band])
            .map(|index| coefficient(instance, band, index))
            .collect()
    })
}

/// Interleaves `bands` into the window's samples.
///
/// The bands are LL, HL, LH and HH, each row-major at its
/// [`Dwt53Window::band_dims`] width. Low-pass samples sit at even
/// reference-grid coordinates.
pub fn dwt53_interleave(window: Dwt53Window, bands: &[Vec<i32>; 4]) -> Vec<i32> {
    let dims = window.band_dims();
    let mut plane = Vec::with_capacity((window.width * window.height) as usize);
    for y in window.y0..window.y0 + window.height {
        for x in window.x0..window.x0 + window.width {
            let (low_x, low_y) = (x % 2 == 0, y % 2 == 0);
            let band = usize::from(!low_x) + 2 * usize::from(!low_y);
            let band_x = if low_x {
                x.div_ceil(2) - window.x0.div_ceil(2)
            } else {
                x / 2 - window.x0 / 2
            };
            let band_y = if low_y {
                y.div_ceil(2) - window.y0.div_ceil(2)
            } else {
                y / 2 - window.y0 / 2
            };
            plane.push(bands[band][(band_y * dims[band].0 + band_x) as usize]);
        }
    }
    plane
}

/// The inverse 5/3 of `plane`, the interleaved samples of `window`: its rows
/// when `horizontal`, then its columns when `vertical`.
pub fn inverse_dwt53(window: Dwt53Window, plane: &mut [i32], horizontal: bool, vertical: bool) {
    let (width, height) = (window.width as usize, window.height as usize);
    if horizontal {
        for row in plane.chunks_exact_mut(width) {
            inverse_dwt53_line(row, window.x0);
        }
    }
    if vertical {
        let mut column = vec![0; height];
        for x in 0..width {
            for (y, sample) in column.iter_mut().enumerate() {
                *sample = plane[y * width + x];
            }
            inverse_dwt53_line(&mut column, window.y0);
            for (y, sample) in column.iter().enumerate() {
                plane[y * width + x] = *sample;
            }
        }
    }
}

/// The one-dimensional inverse 5/3 of `line`, whose first sample is at
/// coordinate `origin`, with symmetric extension at both ends.
fn inverse_dwt53_line(line: &mut [i32], origin: u32) {
    let len = line.len();
    if len == 1 {
        if origin % 2 == 1 {
            line[0] = line[0].div_euclid(2);
        }
        return;
    }
    let neighbours = |index: usize| {
        let left = if index == 0 { 1 } else { index - 1 };
        let right = if index + 1 == len { len - 2 } else { index + 1 };
        (left, right)
    };
    let first_low = usize::from(origin % 2 == 1);
    for index in (first_low..len).step_by(2) {
        let (left, right) = neighbours(index);
        line[index] -= (line[left] + line[right] + 2).div_euclid(4);
    }
    for index in (1 - first_low..len).step_by(2) {
        let (left, right) = neighbours(index);
        line[index] += (line[left] + line[right]).div_euclid(2);
    }
}

/// `values` as the integer-valued `f32` samples the kernels hold.
///
/// # Panics
///
/// Panics if a value does not fit `i16`, which these small references never
/// reach.
pub fn dwt53_samples_f32(values: &[i32]) -> Vec<f32> {
    values
        .iter()
        .map(|&value| f32::from(i16::try_from(value).expect("reference sample fits i16")))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{dwt53_bands, dwt53_interleave, inverse_dwt53, Dwt53Window};

    fn inverse(window: Dwt53Window, bands: &[Vec<i32>; 4]) -> Vec<i32> {
        let mut plane = dwt53_interleave(window, bands);
        inverse_dwt53(window, &mut plane, true, true);
        plane
    }

    #[test]
    fn lone_odd_samples_divide_by_two_with_floor_division() {
        let one = |value| [vec![value], vec![], vec![], vec![]];
        // At an odd column the lone sample is HL; at an odd row, LH; at both, HH.
        let hl = |value| [vec![], vec![value], vec![], vec![]];
        let lh = |value| [vec![], vec![], vec![value], vec![]];
        let hh = |value| [vec![], vec![], vec![], vec![value]];
        assert_eq!(inverse(Dwt53Window::new(1, 1, 0, 0), &one(-7)), [-7]);
        assert_eq!(inverse(Dwt53Window::new(1, 1, 1, 0), &hl(-7)), [-4]);
        assert_eq!(inverse(Dwt53Window::new(1, 1, 0, 1), &lh(7)), [3]);
        assert_eq!(inverse(Dwt53Window::new(1, 1, 1, 1), &hh(-7)), [-2]);
    }

    #[test]
    fn lone_column_halves_each_row_before_the_column_lifts() {
        // Rows at x = 1 are high-pass: HL at even y, HH at odd y.
        let window = Dwt53Window::new(1, 3, 1, 0);
        let bands = [vec![], vec![7, 9], vec![], vec![-5]];
        // Halved: 3, -3, 4. Predict the even rows from row 1, mirrored at the
        // ends: 3 - floor((-6 + 2) / 4) = 4 and 4 - floor((-6 + 2) / 4) = 5.
        // Update row 1 from them: -3 + floor((4 + 5) / 2) = 1.
        assert_eq!(inverse(window, &bands), [4, 1, 5]);
        assert_eq!(dwt53_bands(window, 0).map(|band| band.len()), [0, 2, 0, 1]);
    }
}
