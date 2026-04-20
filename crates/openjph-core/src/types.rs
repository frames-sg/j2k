//! Core type definitions — port of `ojph_defs.h` and `ojph_base.h`.
//!
//! Provides fundamental numeric aliases, version constants, helper functions,
//! and geometric primitives used throughout the codec.

// ---------------------------------------------------------------------------
// Numeric type aliases (mirrors the C++ `ojph::` typedefs)
// ---------------------------------------------------------------------------

/// Unsigned 8-bit integer.
pub type Ui8 = u8;
/// Signed 8-bit integer.
pub type Si8 = i8;
/// Unsigned 16-bit integer.
pub type Ui16 = u16;
/// Signed 16-bit integer.
pub type Si16 = i16;
/// Unsigned 32-bit integer.
pub type Ui32 = u32;
/// Signed 32-bit integer.
pub type Si32 = i32;
/// Unsigned 64-bit integer.
pub type Ui64 = u64;
/// Signed 64-bit integer.
pub type Si64 = i64;

// ---------------------------------------------------------------------------
// Version constants
// ---------------------------------------------------------------------------

/// OpenJPH major version (upstream C++ v0.26.3).
pub const OPENJPH_VERSION_MAJOR: u32 = 0;
/// OpenJPH minor version.
pub const OPENJPH_VERSION_MINOR: u32 = 26;
/// OpenJPH patch version.
pub const OPENJPH_VERSION_PATCH: u32 = 3;

// ---------------------------------------------------------------------------
// Codec constants
// ---------------------------------------------------------------------------

/// Number of fractional bits used in fixed-point arithmetic.
pub const NUM_FRAC_BITS: u32 = 13;

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Integer ceiling division: `⌈a / b⌉`.
///
/// # Panics
///
/// Panics (debug) when `b == 0`.
#[inline]
pub const fn div_ceil(a: u32, b: u32) -> u32 {
    a.div_ceil(b)
}

/// Returns the larger of two values (const-evaluable).
#[inline]
pub const fn ojph_max(a: i32, b: i32) -> i32 {
    if a > b {
        a
    } else {
        b
    }
}

/// Returns the smaller of two values (const-evaluable).
#[inline]
pub const fn ojph_min(a: i32, b: i32) -> i32 {
    if a < b {
        a
    } else {
        b
    }
}

// ---------------------------------------------------------------------------
// Geometric primitives — port of `ojph_base.h`
// ---------------------------------------------------------------------------

/// Two-dimensional size with unsigned width and height.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Size {
    /// Width.
    pub w: u32,
    /// Height.
    pub h: u32,
}

impl Size {
    /// Creates a new `Size`.
    #[inline]
    pub const fn new(w: u32, h: u32) -> Self {
        Self { w, h }
    }

    /// Total number of elements (area) as `u64` to avoid overflow.
    #[inline]
    pub const fn area(&self) -> u64 {
        self.w as u64 * self.h as u64
    }
}

/// A point with unsigned coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Point {
    /// X coordinate.
    pub x: u32,
    /// Y coordinate.
    pub y: u32,
}

impl Point {
    /// Creates a new `Point`.
    #[inline]
    pub const fn new(x: u32, y: u32) -> Self {
        Self { x, y }
    }
}

/// Axis-aligned rectangle defined by its origin point and size.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Rect {
    /// Origin (top-left corner).
    pub org: Point,
    /// Size (width × height).
    pub siz: Size,
}

impl Rect {
    /// Creates a new `Rect`.
    #[inline]
    pub const fn new(org: Point, siz: Size) -> Self {
        Self { org, siz }
    }

    /// Total number of elements inside the rectangle.
    #[inline]
    pub const fn area(&self) -> u64 {
        self.siz.area()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn div_ceil_basic() {
        assert_eq!(div_ceil(10, 3), 4);
        assert_eq!(div_ceil(9, 3), 3);
        assert_eq!(div_ceil(1, 1), 1);
    }

    #[test]
    fn min_max() {
        assert_eq!(ojph_max(3, 5), 5);
        assert_eq!(ojph_min(3, 5), 3);
    }

    #[test]
    fn size_area() {
        let s = Size::new(1920, 1080);
        assert_eq!(s.area(), 1920 * 1080);
    }
}
