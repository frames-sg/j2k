use alloc::vec;
use alloc::vec::Vec;

pub(crate) const SIMD_WIDTH: usize = 8;

/// Number of bits required to represent an unsigned 32-bit magnitude.
///
/// The count is kept in its natural `0..=32` range instead of being
/// calculated in a wider type and narrowed at each call site.
pub(crate) const fn bit_width_u32(mut value: u32) -> u8 {
    let mut bits = 0_u8;
    while value != 0 {
        bits += 1;
        value >>= 1;
    }
    bits
}

/// Number of bits required to represent an unsigned 64-bit magnitude.
pub(crate) const fn bit_width_u64(mut value: u64) -> u8 {
    let mut bits = 0_u8;
    while value != 0 {
        bits += 1;
        value >>= 1;
    }
    bits
}

/// Smallest exponent whose power of two is at least `value`.
pub(crate) const fn ceil_log2_u32(value: u32) -> u8 {
    if value <= 1 {
        0
    } else {
        bit_width_u32(value - 1)
    }
}

#[cfg(feature = "simd")]
mod inner {
    use super::SIMD_WIDTH;
    use core::ops::{Add, AddAssign, DivAssign, Mul, MulAssign, Sub, SubAssign};
    use fearless_simd::{SimdBase, SimdFloat};

    pub(crate) use fearless_simd::{dispatch, Level, Simd};

    #[derive(Copy, Clone)]
    #[repr(C, align(32))]
    pub(crate) struct f32x8<S: Simd> {
        inner: fearless_simd::f32x8<S>,
    }

    impl<S: Simd> f32x8<S> {
        #[expect(
            clippy::inline_always,
            reason = "tiny SIMD and scalar wrappers must disappear inside transform loops"
        )]
        #[inline(always)]
        pub(crate) fn from_slice(simd: S, slice: &[f32]) -> Self {
            Self {
                inner: fearless_simd::f32x8::from_slice(simd, slice),
            }
        }

        #[expect(
            clippy::inline_always,
            reason = "tiny SIMD and scalar wrappers must disappear inside transform loops"
        )]
        #[inline(always)]
        pub(crate) fn splat(simd: S, value: f32) -> Self {
            Self {
                inner: fearless_simd::f32x8::splat(simd, value),
            }
        }

        #[expect(
            clippy::inline_always,
            reason = "tiny SIMD and scalar wrappers must disappear inside transform loops"
        )]
        #[inline(always)]
        pub(crate) fn mul_add(self, mul: Self, addend: Self) -> Self {
            Self {
                inner: self.inner.mul_add(mul.inner, addend.inner),
            }
        }

        #[expect(
            clippy::inline_always,
            reason = "tiny SIMD and scalar wrappers must disappear inside transform loops"
        )]
        #[inline(always)]
        pub(crate) fn floor(self) -> Self {
            Self {
                inner: self.inner.floor(),
            }
        }

        #[expect(
            clippy::inline_always,
            reason = "tiny SIMD and scalar wrappers must disappear inside transform loops"
        )]
        #[inline(always)]
        pub(crate) fn store(self, slice: &mut [f32]) {
            self.inner.store_slice(&mut slice[..SIMD_WIDTH]);
        }

        #[expect(
            clippy::inline_always,
            reason = "tiny SIMD and scalar wrappers must disappear inside transform loops"
        )]
        #[inline(always)]
        pub(crate) fn zip_low(self, other: Self) -> Self {
            Self {
                inner: self.inner.zip_low(other.inner),
            }
        }

        #[expect(
            clippy::inline_always,
            reason = "tiny SIMD and scalar wrappers must disappear inside transform loops"
        )]
        #[inline(always)]
        pub(crate) fn zip_high(self, other: Self) -> Self {
            Self {
                inner: self.inner.zip_high(other.inner),
            }
        }

        #[expect(
            clippy::inline_always,
            reason = "tiny SIMD and scalar wrappers must disappear inside transform loops"
        )]
        #[inline(always)]
        pub(crate) fn min(self, other: Self) -> Self {
            Self {
                inner: self.inner.min(other.inner),
            }
        }

        #[expect(
            clippy::inline_always,
            reason = "tiny SIMD and scalar wrappers must disappear inside transform loops"
        )]
        #[inline(always)]
        pub(crate) fn max(self, other: Self) -> Self {
            Self {
                inner: self.inner.max(other.inner),
            }
        }
    }

    impl<S: Simd> Add for f32x8<S> {
        type Output = Self;
        #[expect(
            clippy::inline_always,
            reason = "tiny SIMD and scalar wrappers must disappear inside transform loops"
        )]
        #[inline(always)]
        fn add(self, rhs: Self) -> Self {
            Self {
                inner: self.inner + rhs.inner,
            }
        }
    }

    impl<S: Simd> Sub for f32x8<S> {
        type Output = Self;
        #[expect(
            clippy::inline_always,
            reason = "tiny SIMD and scalar wrappers must disappear inside transform loops"
        )]
        #[inline(always)]
        fn sub(self, rhs: Self) -> Self {
            Self {
                inner: self.inner - rhs.inner,
            }
        }
    }

    impl<S: Simd> Mul for f32x8<S> {
        type Output = Self;
        #[expect(
            clippy::inline_always,
            reason = "tiny SIMD and scalar wrappers must disappear inside transform loops"
        )]
        #[inline(always)]
        fn mul(self, rhs: Self) -> Self {
            Self {
                inner: self.inner * rhs.inner,
            }
        }
    }

    impl<S: Simd> Add<f32> for f32x8<S> {
        type Output = Self;
        #[expect(
            clippy::inline_always,
            reason = "tiny SIMD and scalar wrappers must disappear inside transform loops"
        )]
        #[inline(always)]
        fn add(self, rhs: f32) -> Self {
            Self {
                inner: self.inner + rhs,
            }
        }
    }

    impl<S: Simd> Mul<f32> for f32x8<S> {
        type Output = Self;
        #[expect(
            clippy::inline_always,
            reason = "tiny SIMD and scalar wrappers must disappear inside transform loops"
        )]
        #[inline(always)]
        fn mul(self, rhs: f32) -> Self {
            Self {
                inner: self.inner * rhs,
            }
        }
    }

    impl<S: Simd> AddAssign for f32x8<S> {
        #[expect(
            clippy::inline_always,
            reason = "tiny SIMD and scalar wrappers must disappear inside transform loops"
        )]
        #[inline(always)]
        fn add_assign(&mut self, rhs: Self) {
            self.inner = self.inner + rhs.inner;
        }
    }

    impl<S: Simd> SubAssign for f32x8<S> {
        #[expect(
            clippy::inline_always,
            reason = "tiny SIMD and scalar wrappers must disappear inside transform loops"
        )]
        #[inline(always)]
        fn sub_assign(&mut self, rhs: Self) {
            self.inner = self.inner - rhs.inner;
        }
    }

    impl<S: Simd> MulAssign<f32> for f32x8<S> {
        #[expect(
            clippy::inline_always,
            reason = "tiny SIMD and scalar wrappers must disappear inside transform loops"
        )]
        #[inline(always)]
        fn mul_assign(&mut self, rhs: f32) {
            self.inner = self.inner * rhs;
        }
    }

    impl<S: Simd> DivAssign<f32> for f32x8<S> {
        #[expect(
            clippy::inline_always,
            reason = "tiny SIMD and scalar wrappers must disappear inside transform loops"
        )]
        #[inline(always)]
        fn div_assign(&mut self, rhs: f32) {
            self.inner = self.inner / rhs;
        }
    }
}

#[cfg(not(feature = "simd"))]
mod inner {
    use super::SIMD_WIDTH;
    use core::marker::PhantomData;
    use core::ops::{Add, AddAssign, DivAssign, Mul, MulAssign, Sub, SubAssign};

    pub(crate) trait Simd: Copy + Clone {}

    #[derive(Copy, Clone)]
    pub(crate) struct ScalarSimd;
    impl Simd for ScalarSimd {}

    pub(crate) struct Level;
    impl Level {
        #[expect(
            clippy::inline_always,
            reason = "tiny SIMD and scalar wrappers must disappear inside transform loops"
        )]
        #[inline(always)]
        pub(crate) fn new() -> Self {
            Level
        }
    }

    #[derive(Copy, Clone)]
    #[repr(C, align(32))]
    pub(crate) struct f32x8<S: Simd> {
        val: [f32; SIMD_WIDTH],
        _marker: PhantomData<S>,
    }

    impl<S: Simd> f32x8<S> {
        #[expect(
            clippy::inline_always,
            reason = "tiny SIMD and scalar wrappers must disappear inside transform loops"
        )]
        #[inline(always)]
        pub(crate) fn from_slice(_simd: S, slice: &[f32]) -> Self {
            let mut val = [0.0f32; SIMD_WIDTH];
            val.copy_from_slice(&slice[..SIMD_WIDTH]);
            Self {
                val,
                _marker: PhantomData,
            }
        }

        #[expect(
            clippy::inline_always,
            reason = "tiny SIMD and scalar wrappers must disappear inside transform loops"
        )]
        #[inline(always)]
        pub(crate) fn splat(_simd: S, value: f32) -> Self {
            Self {
                val: [value; SIMD_WIDTH],
                _marker: PhantomData,
            }
        }

        #[expect(
            clippy::inline_always,
            reason = "tiny SIMD and scalar wrappers must disappear inside transform loops"
        )]
        #[expect(clippy::needless_range_loop, reason = "fixed-width scalar SIMD lanes")]
        #[inline(always)]
        pub(crate) fn mul_add(self, mul: Self, addend: Self) -> Self {
            let mut result = [0.0f32; SIMD_WIDTH];
            for i in 0..SIMD_WIDTH {
                result[i] = super::mul_add(self.val[i], mul.val[i], addend.val[i]);
            }
            Self {
                val: result,
                _marker: PhantomData,
            }
        }

        #[expect(
            clippy::inline_always,
            reason = "tiny SIMD and scalar wrappers must disappear inside transform loops"
        )]
        #[expect(clippy::needless_range_loop, reason = "fixed-width scalar SIMD lanes")]
        #[inline(always)]
        pub(crate) fn floor(self) -> Self {
            let mut result = [0.0f32; SIMD_WIDTH];
            for i in 0..SIMD_WIDTH {
                result[i] = super::floor_f32(self.val[i]);
            }
            Self {
                val: result,
                _marker: PhantomData,
            }
        }

        #[expect(
            clippy::inline_always,
            reason = "tiny SIMD and scalar wrappers must disappear inside transform loops"
        )]
        #[inline(always)]
        pub(crate) fn store(self, slice: &mut [f32]) {
            slice[..SIMD_WIDTH].copy_from_slice(&self.val);
        }

        #[expect(
            clippy::inline_always,
            reason = "tiny SIMD and scalar wrappers must disappear inside transform loops"
        )]
        #[inline(always)]
        pub(crate) fn zip_low(self, other: Self) -> Self {
            Self {
                val: [
                    self.val[0],
                    other.val[0],
                    self.val[1],
                    other.val[1],
                    self.val[2],
                    other.val[2],
                    self.val[3],
                    other.val[3],
                ],
                _marker: PhantomData,
            }
        }

        #[expect(
            clippy::inline_always,
            reason = "tiny SIMD and scalar wrappers must disappear inside transform loops"
        )]
        #[inline(always)]
        pub(crate) fn zip_high(self, other: Self) -> Self {
            Self {
                val: [
                    self.val[4],
                    other.val[4],
                    self.val[5],
                    other.val[5],
                    self.val[6],
                    other.val[6],
                    self.val[7],
                    other.val[7],
                ],
                _marker: PhantomData,
            }
        }

        #[expect(
            clippy::inline_always,
            reason = "tiny SIMD and scalar wrappers must disappear inside transform loops"
        )]
        #[expect(clippy::needless_range_loop, reason = "fixed-width scalar SIMD lanes")]
        #[inline(always)]
        pub(crate) fn min(self, other: Self) -> Self {
            let mut result = [0.0f32; SIMD_WIDTH];
            for i in 0..SIMD_WIDTH {
                result[i] = super::min_f32(self.val[i], other.val[i]);
            }
            Self {
                val: result,
                _marker: PhantomData,
            }
        }

        #[expect(
            clippy::inline_always,
            reason = "tiny SIMD and scalar wrappers must disappear inside transform loops"
        )]
        #[expect(clippy::needless_range_loop, reason = "fixed-width scalar SIMD lanes")]
        #[inline(always)]
        pub(crate) fn max(self, other: Self) -> Self {
            let mut result = [0.0f32; SIMD_WIDTH];
            for i in 0..SIMD_WIDTH {
                result[i] = super::max_f32(self.val[i], other.val[i]);
            }
            Self {
                val: result,
                _marker: PhantomData,
            }
        }
    }

    impl<S: Simd> Add for f32x8<S> {
        type Output = Self;
        #[expect(
            clippy::inline_always,
            reason = "tiny SIMD and scalar wrappers must disappear inside transform loops"
        )]
        #[expect(clippy::needless_range_loop, reason = "fixed-width scalar SIMD lanes")]
        #[inline(always)]
        fn add(self, rhs: Self) -> Self {
            let mut result = [0.0f32; SIMD_WIDTH];
            for i in 0..SIMD_WIDTH {
                result[i] = self.val[i] + rhs.val[i];
            }
            Self {
                val: result,
                _marker: PhantomData,
            }
        }
    }

    impl<S: Simd> Sub for f32x8<S> {
        type Output = Self;
        #[expect(
            clippy::inline_always,
            reason = "tiny SIMD and scalar wrappers must disappear inside transform loops"
        )]
        #[expect(clippy::needless_range_loop, reason = "fixed-width scalar SIMD lanes")]
        #[inline(always)]
        fn sub(self, rhs: Self) -> Self {
            let mut result = [0.0f32; SIMD_WIDTH];
            for i in 0..SIMD_WIDTH {
                result[i] = self.val[i] - rhs.val[i];
            }
            Self {
                val: result,
                _marker: PhantomData,
            }
        }
    }

    impl<S: Simd> Mul for f32x8<S> {
        type Output = Self;
        #[expect(
            clippy::inline_always,
            reason = "tiny SIMD and scalar wrappers must disappear inside transform loops"
        )]
        #[expect(clippy::needless_range_loop, reason = "fixed-width scalar SIMD lanes")]
        #[inline(always)]
        fn mul(self, rhs: Self) -> Self {
            let mut result = [0.0f32; SIMD_WIDTH];
            for i in 0..SIMD_WIDTH {
                result[i] = self.val[i] * rhs.val[i];
            }
            Self {
                val: result,
                _marker: PhantomData,
            }
        }
    }

    impl<S: Simd> Add<f32> for f32x8<S> {
        type Output = Self;
        #[expect(
            clippy::inline_always,
            reason = "tiny SIMD and scalar wrappers must disappear inside transform loops"
        )]
        #[expect(clippy::needless_range_loop, reason = "fixed-width scalar SIMD lanes")]
        #[inline(always)]
        fn add(self, rhs: f32) -> Self {
            let mut result = [0.0f32; SIMD_WIDTH];
            for i in 0..SIMD_WIDTH {
                result[i] = self.val[i] + rhs;
            }
            Self {
                val: result,
                _marker: PhantomData,
            }
        }
    }

    impl<S: Simd> Mul<f32> for f32x8<S> {
        type Output = Self;
        #[expect(
            clippy::inline_always,
            reason = "tiny SIMD and scalar wrappers must disappear inside transform loops"
        )]
        #[expect(clippy::needless_range_loop, reason = "fixed-width scalar SIMD lanes")]
        #[inline(always)]
        fn mul(self, rhs: f32) -> Self {
            let mut result = [0.0f32; SIMD_WIDTH];
            for i in 0..SIMD_WIDTH {
                result[i] = self.val[i] * rhs;
            }
            Self {
                val: result,
                _marker: PhantomData,
            }
        }
    }

    impl<S: Simd> AddAssign for f32x8<S> {
        #[expect(
            clippy::inline_always,
            reason = "tiny SIMD and scalar wrappers must disappear inside transform loops"
        )]
        #[inline(always)]
        fn add_assign(&mut self, rhs: Self) {
            for i in 0..SIMD_WIDTH {
                self.val[i] += rhs.val[i];
            }
        }
    }

    impl<S: Simd> SubAssign for f32x8<S> {
        #[expect(
            clippy::inline_always,
            reason = "tiny SIMD and scalar wrappers must disappear inside transform loops"
        )]
        #[inline(always)]
        fn sub_assign(&mut self, rhs: Self) {
            for i in 0..SIMD_WIDTH {
                self.val[i] -= rhs.val[i];
            }
        }
    }

    impl<S: Simd> MulAssign<f32> for f32x8<S> {
        #[expect(
            clippy::inline_always,
            reason = "tiny SIMD and scalar wrappers must disappear inside transform loops"
        )]
        #[inline(always)]
        fn mul_assign(&mut self, rhs: f32) {
            for i in 0..SIMD_WIDTH {
                self.val[i] *= rhs;
            }
        }
    }

    impl<S: Simd> DivAssign<f32> for f32x8<S> {
        #[expect(
            clippy::inline_always,
            reason = "tiny SIMD and scalar wrappers must disappear inside transform loops"
        )]
        #[inline(always)]
        fn div_assign(&mut self, rhs: f32) {
            for i in 0..SIMD_WIDTH {
                self.val[i] /= rhs;
            }
        }
    }

    /// Scalar fallback for SIMD dispatch.
    #[macro_export]
    macro_rules! simd_dispatch {
        ($level:expr, $simd:ident => $body:expr) => {{
            let _ = $level;
            let $simd = $crate::math::ScalarSimd;
            $body
        }};
    }

    pub(crate) use simd_dispatch as dispatch;
}

// Note that these polyfills can be very imprecise, but hopefully good enough
// for the vast majority of cases.

#[expect(
    clippy::inline_always,
    reason = "tiny SIMD and scalar wrappers must disappear inside transform loops"
)]
#[inline(always)]
pub(crate) fn mul_add(a: f32, b: f32, c: f32) -> f32 {
    #[cfg(all(
        feature = "std",
        any(
            all(
                any(target_arch = "x86", target_arch = "x86_64"),
                target_feature = "fma"
            ),
            all(target_arch = "aarch64", target_feature = "neon")
        )
    ))]
    {
        f32::mul_add(a, b, c)
    }
    #[cfg(not(all(
        feature = "std",
        any(
            all(
                any(target_arch = "x86", target_arch = "x86_64"),
                target_feature = "fma"
            ),
            all(target_arch = "aarch64", target_feature = "neon")
        )
    )))]
    {
        a * b + c
    }
}

#[expect(
    clippy::inline_always,
    reason = "tiny SIMD and scalar wrappers must disappear inside transform loops"
)]
#[cfg_attr(
    not(feature = "std"),
    expect(
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        reason = "the no-std floor polyfill converts through the truncated integer part"
    )
)]
#[inline(always)]
pub(crate) fn floor_f32(x: f32) -> f32 {
    #[cfg(feature = "std")]
    {
        x.floor()
    }
    #[cfg(not(feature = "std"))]
    {
        let xi = x as i32;
        let xf = xi as f32;
        if x < xf {
            xf - 1.0
        } else {
            xf
        }
    }
}

#[expect(
    clippy::inline_always,
    reason = "tiny SIMD and scalar wrappers must disappear inside transform loops"
)]
#[inline(always)]
pub(crate) fn round_f32(x: f32) -> f32 {
    #[cfg(feature = "std")]
    {
        x.round()
    }
    #[cfg(not(feature = "std"))]
    {
        if x >= 0.0 {
            floor_f32(x + 0.5)
        } else {
            -floor_f32(-x + 0.5)
        }
    }
}

#[expect(
    clippy::inline_always,
    reason = "tiny SIMD and scalar wrappers must disappear inside transform loops"
)]
#[inline(always)]
pub(crate) fn log2_f32(x: f32) -> f32 {
    #[cfg(feature = "std")]
    {
        x.log2()
    }
    #[cfg(not(feature = "std"))]
    {
        libm::log2f(x)
    }
}

#[expect(
    clippy::inline_always,
    reason = "tiny SIMD and scalar wrappers must disappear inside transform loops"
)]
#[expect(
    clippy::cast_precision_loss,
    reason = "powers of two in the supported exponent range are exactly representable in f32"
)]
#[inline(always)]
pub(crate) fn pow2i(exp: i32) -> f32 {
    if exp >= 0 {
        (1_u32 << exp) as f32
    } else {
        1.0 / (1_u32 << -exp) as f32
    }
}

#[expect(
    clippy::inline_always,
    reason = "tiny SIMD and scalar wrappers must disappear inside transform loops"
)]
#[inline(always)]
#[cfg(not(feature = "simd"))]
pub(crate) fn min_f32(a: f32, b: f32) -> f32 {
    #[cfg(feature = "std")]
    {
        a.min(b)
    }
    #[cfg(not(feature = "std"))]
    {
        if a < b {
            a
        } else {
            b
        }
    }
}

#[expect(
    clippy::inline_always,
    reason = "tiny SIMD and scalar wrappers must disappear inside transform loops"
)]
#[inline(always)]
#[cfg(not(feature = "simd"))]
pub(crate) fn max_f32(a: f32, b: f32) -> f32 {
    #[cfg(feature = "std")]
    {
        a.max(b)
    }
    #[cfg(not(feature = "std"))]
    {
        if a > b {
            a
        } else {
            b
        }
    }
}

#[cfg(not(feature = "simd"))]
pub(crate) use inner::{dispatch, f32x8, Level, ScalarSimd, Simd};
#[cfg(feature = "simd")]
pub(crate) use inner::{dispatch, f32x8, Level, Simd};

/// A wrapper around `Vec<f32>` that pads the vector to a multiple of `N` elements.
/// This allows SIMD operations to safely process the data without bounds checking
/// at the end of the buffer.
#[derive(Debug, Clone)]
pub(crate) struct SimdBuffer<const N: usize> {
    data: Vec<f32>,
    original_len: usize,
}

impl<const N: usize> SimdBuffer<N> {
    /// Create a new `SimdBuffer` from a `Vec<f32>`, padding it to a multiple of `N`.
    pub(crate) fn new(mut data: Vec<f32>) -> Self {
        let original_len = data.len();
        let padded_len = Self::padded_len(original_len);
        if padded_len > original_len {
            data.resize(padded_len, 0.0);
        }
        Self { data, original_len }
    }

    /// Create a new `SimdBuffer` filled with zeros.
    pub(crate) fn zeros(original_len: usize) -> Self {
        let padded_len = Self::padded_len(original_len);
        let data = vec![0.0; padded_len];
        Self { data, original_len }
    }

    /// Returns only the original (non-padded) data as an immutable slice.
    pub(crate) fn truncated(&self) -> &[f32] {
        &self.data[..self.original_len]
    }

    /// Returns the length padded to a multiple of `N`
    fn padded_len(original_len: usize) -> usize {
        let remainder = original_len % N;
        let padding = N - remainder;
        original_len + padding
    }
}

impl<const N: usize> core::ops::Deref for SimdBuffer<N> {
    type Target = [f32];

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<const N: usize> core::ops::DerefMut for SimdBuffer<N> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

#[cfg(test)]
mod integer_tests {
    use super::{bit_width_u32, bit_width_u64, ceil_log2_u32};

    #[test]
    fn integer_bit_ranges_cover_type_boundaries() {
        assert_eq!(bit_width_u32(0), 0);
        assert_eq!(bit_width_u32(1), 1);
        assert_eq!(bit_width_u32(u32::MAX), 32);
        assert_eq!(bit_width_u64(0), 0);
        assert_eq!(bit_width_u64(1_u64 << 63), 64);
        assert_eq!(bit_width_u64(u64::MAX), 64);
    }

    #[test]
    fn ceil_log2_covers_zero_and_full_u32_domain() {
        assert_eq!(ceil_log2_u32(0), 0);
        assert_eq!(ceil_log2_u32(1), 0);
        assert_eq!(ceil_log2_u32(2), 1);
        assert_eq!(ceil_log2_u32(3), 2);
        assert_eq!(ceil_log2_u32(u32::MAX), 32);
    }
}
