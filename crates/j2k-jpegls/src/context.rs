// SPDX-License-Identifier: MIT OR Apache-2.0

//! Context modelling shared by the decoder and the encoder: the regular-mode
//! and run-interruption contexts (T.87 A.2 to A.7), prediction, and the line
//! buffers that hold the causal neighbourhood.

use crate::allocation::{zeroed, AllocationError};
use crate::params::CodingParameters;

/// Run-length order table `J` (T.87 A.7.1.2).
pub(crate) const J: [u32; 32] = [
    0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 9, 10, 11, 12, 13,
    14, 15,
];

/// Regular-mode contexts: `Q` in 1..=364 after sign merging. Index 0 is never
/// used because all-zero gradients select run mode.
const REGULAR_CONTEXTS: usize = 365;
const MIN_C: i32 = -128;
const MAX_C: i32 = 127;
/// Line and sample interleaving code at most this many components per scan.
pub(crate) const MAX_INTERLEAVED_COMPONENTS: usize = 4;

/// Context variables `A`, `B`, `C`, `N` of one regular-mode context.
#[derive(Debug, Clone, Copy)]
pub(crate) struct RegularContext {
    a: i64,
    b: i64,
    c: i32,
    n: i64,
}

impl RegularContext {
    const fn new(a_init: i64) -> Self {
        Self {
            a: a_init,
            b: 0,
            c: 0,
            n: 1,
        }
    }

    /// Golomb parameter `k` (T.87 A.5.1).
    pub(crate) fn golomb_k(&self) -> u32 {
        golomb_k(self.n, self.a)
    }

    /// Bias correction `C`.
    pub(crate) const fn bias(&self) -> i32 {
        self.c
    }

    /// Whether lossless coding with `k = 0` uses the inverted error mapping
    /// (`2 * B <= -N`, T.87 A.5.2).
    pub(crate) const fn inverts_mapping(&self) -> bool {
        2 * self.b + self.n - 1 < 0
    }

    /// Update `A`, `B`, `N` and the bias `C` after coding `error`
    /// (T.87 A.6.1 and A.6.2).
    pub(crate) fn update(&mut self, error: i32, near: i32, reset: i64) {
        self.a += i64::from(error.unsigned_abs());
        self.b += i64::from(error) * i64::from(2 * near + 1);
        if self.n == reset {
            self.a >>= 1;
            self.b >>= 1;
            self.n >>= 1;
        }
        self.n += 1;
        if self.b + self.n <= 0 {
            self.b += self.n;
            if self.b <= -self.n {
                self.b = -self.n + 1;
            }
            if self.c > MIN_C {
                self.c -= 1;
            }
        } else if self.b > 0 {
            self.b -= self.n;
            if self.b > 0 {
                self.b = 0;
            }
            if self.c < MAX_C {
                self.c += 1;
            }
        }
    }
}

/// Context variables of one run-interruption context (T.87 A.7.2).
#[derive(Debug, Clone, Copy)]
pub(crate) struct RunContext {
    /// `RItype`: 1 when the interruption sample is predicted from `Ra`.
    run_type: i64,
    a: i64,
    n: i64,
    nn: i64,
}

impl RunContext {
    const fn new(run_type: i64, a_init: i64) -> Self {
        Self {
            run_type,
            a: a_init,
            n: 1,
            nn: 0,
        }
    }

    pub(crate) const fn run_type(&self) -> i64 {
        self.run_type
    }

    /// Golomb parameter `k` from `TEMP` (T.87 A.7.2.1).
    pub(crate) fn golomb_k(&self) -> u32 {
        golomb_k(self.n, self.a + (self.n >> 1) * self.run_type)
    }

    /// The `map` bit of the error mapping (T.87 A.7.2.1).
    pub(crate) const fn map_bit(&self, error: i64, k: u32) -> bool {
        (k == 0 && error > 0 && 2 * self.nn < self.n)
            || (error < 0 && 2 * self.nn >= self.n)
            || (error < 0 && k != 0)
    }

    /// Recover the error from `EMErrval + RItype` (T.87 A.7.2.2).
    pub(crate) fn unmap(&self, temp: i64, k: u32) -> i64 {
        let map = temp & 1 == 1;
        let magnitude = temp.midpoint(i64::from(map));
        if (k != 0 || 2 * self.nn >= self.n) == map {
            -magnitude
        } else {
            magnitude
        }
    }

    /// Update after coding `error` with mapped value `mapped` (T.87 A.7.2.1).
    pub(crate) fn update(&mut self, error: i64, mapped: i64, reset: i64) {
        if error < 0 {
            self.nn += 1;
        }
        self.a += (mapped + 1 - self.run_type) >> 1;
        if self.n == reset {
            self.a >>= 1;
            self.n >>= 1;
            self.nn >>= 1;
        }
        self.n += 1;
    }
}

/// Smallest `k` with `n * 2^k >= a`. `n >= 1` always holds and `a` stays far
/// below `2^48`, so the bound only guards against an impossible state.
fn golomb_k(n: i64, a: i64) -> u32 {
    let mut k = 0;
    while k < 48 && (n << k) < a {
        k += 1;
    }
    k
}

/// Median edge detector prediction (T.87 A.4.1).
pub(crate) fn predict(ra: i32, rb: i32, rc: i32) -> i32 {
    let (low, high) = if ra < rb { (ra, rb) } else { (rb, ra) };
    if rc >= high {
        low
    } else if rc <= low {
        high
    } else {
        ra + rb - rc
    }
}

/// Map three quantized gradients to a context index in 1..=364 and the sign
/// of the first nonzero gradient (T.87 A.3.4). Returns `None` for run mode.
pub(crate) const fn context_index(q1: i32, q2: i32, q3: i32) -> Option<(usize, i32)> {
    let q = 81 * q1 + 9 * q2 + q3;
    if q == 0 {
        None
    } else if q < 0 {
        Some((q.unsigned_abs() as usize, -1))
    } else {
        Some((q.unsigned_abs() as usize, 1))
    }
}

/// Coding state of one scan: contexts, run indices, and two line buffers per
/// component. Each line buffer holds `width + 2` samples: index 0 is the
/// sample left of the line and index `width + 1` the one right of it.
pub(crate) struct ScanState {
    pub(crate) params: CodingParameters,
    pub(crate) width: usize,
    pub(crate) regular: [RegularContext; REGULAR_CONTEXTS],
    pub(crate) run: [RunContext; 2],
    pub(crate) run_index: [usize; MAX_INTERLEAVED_COMPONENTS],
    pub(crate) previous: Vec<Vec<i32>>,
    pub(crate) current: Vec<Vec<i32>>,
}

impl ScanState {
    pub(crate) fn new(
        params: CodingParameters,
        width: usize,
        components: usize,
    ) -> Result<Self, AllocationError> {
        let mut previous = Vec::new();
        let mut current = Vec::new();
        for _ in 0..components {
            previous.push(zeroed(width + 2)?);
            current.push(zeroed(width + 2)?);
        }
        let a_init = params.a_init;
        Ok(Self {
            params,
            width,
            regular: [RegularContext::new(a_init); REGULAR_CONTEXTS],
            run: [RunContext::new(0, a_init), RunContext::new(1, a_init)],
            run_index: [0; MAX_INTERLEAVED_COMPONENTS],
            previous,
            current,
        })
    }

    /// Reset contexts, run indices, and line buffers, as at the start of a
    /// scan or after a restart marker.
    pub(crate) fn reset(&mut self) {
        let a_init = self.params.a_init;
        self.regular = [RegularContext::new(a_init); REGULAR_CONTEXTS];
        self.run = [RunContext::new(0, a_init), RunContext::new(1, a_init)];
        self.run_index = [0; MAX_INTERLEAVED_COMPONENTS];
        for line in self.previous.iter_mut().chain(self.current.iter_mut()) {
            line.fill(0);
        }
    }

    /// Set the edge samples of each component's line (T.87 A.2.1): `Rd` past
    /// the right edge repeats `Rb`, and `Ra` left of the line is the sample
    /// above it.
    pub(crate) fn start_line(&mut self) {
        let width = self.width;
        for (previous, current) in self.previous.iter_mut().zip(self.current.iter_mut()) {
            previous[width + 1] = previous[width];
            current[0] = previous[1];
        }
    }

    /// Context (`None` for all-zero gradients, which selects run mode) and
    /// prediction of sample `x` of a component (T.87 A.3 and A.4.1).
    pub(crate) fn sample_context(&self, component: usize, x: usize) -> (Option<(usize, i32)>, i32) {
        let (previous, current) = (&self.previous[component], &self.current[component]);
        let ra = current[x];
        let (rc, rb, rd) = (previous[x], previous[x + 1], previous[x + 2]);
        let params = &self.params;
        let context = context_index(
            params.quantize_gradient(rd - rb),
            params.quantize_gradient(rb - rc),
            params.quantize_gradient(rc - ra),
        );
        (context, predict(ra, rb, rc))
    }

    /// [`Self::sample_context`] for every component of the pixel at `x`.
    pub(crate) fn pixel_contexts(
        &self,
        x: usize,
    ) -> [(Option<(usize, i32)>, i32); MAX_INTERLEAVED_COMPONENTS] {
        let mut contexts = [(None, 0); MAX_INTERLEAVED_COMPONENTS];
        for (component, context) in contexts.iter_mut().enumerate().take(self.current.len()) {
            *context = self.sample_context(component, x);
        }
        contexts
    }

    /// Make the line just coded the previous line.
    pub(crate) fn end_line(&mut self) {
        core::mem::swap(&mut self.previous, &mut self.current);
    }

    pub(crate) fn increment_run_index(&mut self, component: usize) {
        let index = &mut self.run_index[component];
        if *index < J.len() - 1 {
            *index += 1;
        }
    }

    pub(crate) fn decrement_run_index(&mut self, component: usize) {
        let index = &mut self.run_index[component];
        *index = index.saturating_sub(1);
    }
}

#[cfg(test)]
mod tests {
    use super::{context_index, predict};

    #[test]
    fn predictor_follows_the_median_edge_detector() {
        assert_eq!(predict(10, 20, 25), 10);
        assert_eq!(predict(10, 20, 5), 20);
        assert_eq!(predict(10, 20, 15), 15);
    }

    #[test]
    fn context_sign_follows_the_first_nonzero_gradient() {
        assert_eq!(context_index(0, 0, 0), None);
        assert_eq!(context_index(-1, 4, 4), Some((41, -1)));
        assert_eq!(context_index(0, 0, -4), Some((4, -1)));
        assert_eq!(context_index(4, 4, 4), Some((364, 1)));
    }
}
