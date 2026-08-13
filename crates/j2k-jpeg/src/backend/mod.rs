// SPDX-License-Identifier: MIT OR Apache-2.0

//! Hot-path backend dispatch for interleaved RGB row production and the 8×8
//! inverse DCT.

use crate::idct;
#[cfg(target_arch = "x86_64")]
use crate::simd::x86::ExactAvx2;

pub(crate) mod scalar;

#[cfg(target_arch = "x86_64")]
mod x86;

#[cfg(target_arch = "aarch64")]
mod neon;

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
mod row_pair;

#[cfg(test)]
mod tests;

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BackendKind {
    Scalar,
    #[cfg(target_arch = "x86_64")]
    Avx2,
    #[cfg(target_arch = "aarch64")]
    Neon,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum Backend {
    Scalar,
    #[cfg(target_arch = "x86_64")]
    Avx2(ExactAvx2),
    #[cfg(target_arch = "aarch64")]
    Neon(fearless_simd::Neon),
}

#[derive(Clone, Copy)]
pub(crate) struct Rgb420ChromaRows<'a> {
    pub(crate) prev_cb: &'a [u8],
    pub(crate) curr_cb: &'a [u8],
    pub(crate) next_cb: &'a [u8],
    pub(crate) prev_cr: &'a [u8],
    pub(crate) curr_cr: &'a [u8],
    pub(crate) next_cr: &'a [u8],
}

impl<'a> Rgb420ChromaRows<'a> {
    pub(crate) fn new(
        prev_cb: &'a [u8],
        curr_cb: &'a [u8],
        next_cb: &'a [u8],
        prev_cr: &'a [u8],
        curr_cr: &'a [u8],
        next_cr: &'a [u8],
    ) -> Self {
        Self {
            prev_cb,
            curr_cb,
            next_cb,
            prev_cr,
            curr_cr,
            next_cr,
        }
    }

    pub(crate) fn min_width(self) -> usize {
        self.prev_cb
            .len()
            .min(self.curr_cb.len())
            .min(self.next_cb.len())
            .min(self.prev_cr.len())
            .min(self.curr_cr.len())
            .min(self.next_cr.len())
    }
}

pub(crate) struct Rgb420RowPair<'a> {
    pub(crate) y_top: &'a [u8],
    pub(crate) y_bottom: Option<&'a [u8]>,
    pub(crate) chroma: Rgb420ChromaRows<'a>,
    pub(crate) dst_top: &'a mut [u8],
    pub(crate) dst_bottom: Option<&'a mut [u8]>,
}

impl<'a> Rgb420RowPair<'a> {
    pub(crate) fn new(
        y_top: &'a [u8],
        y_bottom: Option<&'a [u8]>,
        chroma: Rgb420ChromaRows<'a>,
        dst_top: &'a mut [u8],
        dst_bottom: Option<&'a mut [u8]>,
    ) -> Self {
        Self {
            y_top,
            y_bottom,
            chroma,
            dst_top,
            dst_bottom,
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct Rgb420Crop {
    pub(crate) start: usize,
    pub(crate) width: usize,
}

impl Rgb420Crop {
    pub(crate) fn new(start: usize, width: usize) -> Self {
        Self { start, width }
    }
}

pub(crate) struct Rgb420CroppedRowPair<'a> {
    pub(crate) rows: Rgb420RowPair<'a>,
    pub(crate) crop: Rgb420Crop,
}

impl<'a> Rgb420CroppedRowPair<'a> {
    pub(crate) fn new(rows: Rgb420RowPair<'a>, crop: Rgb420Crop) -> Self {
        Self { rows, crop }
    }
}

impl Backend {
    pub(crate) fn detect() -> Self {
        if cfg!(feature = "scalar-only") {
            return Self::Scalar;
        }

        #[cfg(target_arch = "x86_64")]
        {
            if let Some(avx2) = ExactAvx2::detect() {
                return Self::Avx2(avx2);
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            if let Some(neon) = fearless_simd::Level::new().as_neon() {
                return Self::Neon(neon);
            }
        }

        Self::Scalar
    }

    #[cfg(test)]
    pub(super) fn for_test_kind(kind: BackendKind) -> Option<Self> {
        match kind {
            BackendKind::Scalar => Some(Self::Scalar),
            #[cfg(target_arch = "x86_64")]
            BackendKind::Avx2 => ExactAvx2::detect().map(Self::Avx2),
            #[cfg(target_arch = "aarch64")]
            BackendKind::Neon => fearless_simd::Level::new().as_neon().map(Self::Neon),
        }
    }

    #[cfg(test)]
    pub(super) const fn kind(self) -> BackendKind {
        match self {
            Self::Scalar => BackendKind::Scalar,
            #[cfg(target_arch = "x86_64")]
            Self::Avx2(_) => BackendKind::Avx2,
            #[cfg(target_arch = "aarch64")]
            Self::Neon(_) => BackendKind::Neon,
        }
    }

    #[expect(
        clippy::inline_always,
        reason = "same-host SIMD benchmarks confirmed this dispatch must inline outside the row loop"
    )]
    #[inline(always)]
    pub(crate) fn fill_rgb_row_from_gray(self, gray_row: &[u8], dst: &mut [u8]) {
        match self {
            Self::Scalar => scalar::fill_rgb_row_from_gray(gray_row, dst),
            #[cfg(target_arch = "x86_64")]
            Self::Avx2(_) => x86::fill_rgb_row_from_gray(gray_row, dst),
            #[cfg(target_arch = "aarch64")]
            Self::Neon(neon) => neon::fill_rgb_row_from_gray(neon, gray_row, dst),
        }
    }

    #[inline]
    pub(crate) fn fill_rgb_row_from_rgb(
        self,
        r_row: &[u8],
        g_row: &[u8],
        b_row: &[u8],
        dst: &mut [u8],
    ) {
        match self {
            Self::Scalar => scalar::fill_rgb_row_from_rgb(r_row, g_row, b_row, dst),
            #[cfg(target_arch = "x86_64")]
            Self::Avx2(_) => x86::fill_rgb_row_from_rgb(r_row, g_row, b_row, dst),
            #[cfg(target_arch = "aarch64")]
            Self::Neon(neon) => neon::fill_rgb_row_from_rgb(neon, r_row, g_row, b_row, dst),
        }
    }

    pub(crate) fn fill_rgb_row_from_ycbcr(
        self,
        y_row: &[u8],
        cb_row: &[u8],
        cr_row: &[u8],
        dst: &mut [u8],
    ) {
        match self {
            Self::Scalar => scalar::fill_rgb_row_from_ycbcr(y_row, cb_row, cr_row, dst),
            #[cfg(target_arch = "x86_64")]
            Self::Avx2(avx2) => x86::fill_rgb_row_from_ycbcr(avx2, y_row, cb_row, cr_row, dst),
            #[cfg(target_arch = "aarch64")]
            Self::Neon(neon) => neon::fill_rgb_row_from_ycbcr(neon, y_row, cb_row, cr_row, dst),
        }
    }

    pub(crate) fn fill_rgba_row_from_gray(self, gray_row: &[u8], dst: &mut [u8], alpha: u8) {
        match self {
            Self::Scalar => scalar::fill_rgba_row_from_gray(gray_row, dst, alpha),
            #[cfg(target_arch = "x86_64")]
            Self::Avx2(_) => scalar::fill_rgba_row_from_gray(gray_row, dst, alpha),
            #[cfg(target_arch = "aarch64")]
            Self::Neon(_) => scalar::fill_rgba_row_from_gray(gray_row, dst, alpha),
        }
    }

    pub(crate) fn fill_rgba_row_from_rgb(
        self,
        r_row: &[u8],
        g_row: &[u8],
        b_row: &[u8],
        dst: &mut [u8],
        alpha: u8,
    ) {
        match self {
            Self::Scalar => scalar::fill_rgba_row_from_rgb(r_row, g_row, b_row, dst, alpha),
            #[cfg(target_arch = "x86_64")]
            Self::Avx2(_) => scalar::fill_rgba_row_from_rgb(r_row, g_row, b_row, dst, alpha),
            #[cfg(target_arch = "aarch64")]
            Self::Neon(_) => scalar::fill_rgba_row_from_rgb(r_row, g_row, b_row, dst, alpha),
        }
    }

    pub(crate) fn fill_rgba_row_from_ycbcr(
        self,
        y_row: &[u8],
        cb_row: &[u8],
        cr_row: &[u8],
        dst: &mut [u8],
        alpha: u8,
    ) {
        match self {
            Self::Scalar => {
                scalar::fill_rgba_row_from_ycbcr(y_row, cb_row, cr_row, dst, alpha);
            }
            #[cfg(target_arch = "x86_64")]
            Self::Avx2(_) => {
                scalar::fill_rgba_row_from_ycbcr(y_row, cb_row, cr_row, dst, alpha);
            }
            #[cfg(target_arch = "aarch64")]
            Self::Neon(_) => {
                scalar::fill_rgba_row_from_ycbcr(y_row, cb_row, cr_row, dst, alpha);
            }
        }
    }

    #[inline]
    pub(crate) fn fill_rgb_row_pair_from_420(self, request: Rgb420RowPair<'_>) {
        match self {
            Self::Scalar => scalar::fill_rgb_row_pair_from_420(request),
            #[cfg(target_arch = "x86_64")]
            Self::Avx2(avx2) => x86::fill_rgb_row_pair_from_420(avx2, request),
            #[cfg(target_arch = "aarch64")]
            Self::Neon(neon) => neon::fill_rgb_row_pair_from_420(neon, request),
        }
    }

    #[inline]
    pub(crate) fn fill_rgb_row_pair_from_420_cropped(self, request: Rgb420CroppedRowPair<'_>) {
        match self {
            Self::Scalar => scalar::fill_rgb_row_pair_from_420_cropped(request),
            #[cfg(target_arch = "x86_64")]
            Self::Avx2(avx2) => x86::fill_rgb_row_pair_from_420_cropped(avx2, request),
            #[cfg(target_arch = "aarch64")]
            Self::Neon(neon) => neon::fill_rgb_row_pair_from_420_cropped(neon, request),
        }
    }

    pub(crate) fn prefers_cropped_420_region(self, row_width: usize, crop_width: usize) -> bool {
        if crop_width == 0 || crop_width >= row_width {
            return false;
        }
        match self {
            Self::Scalar => true,
            #[cfg(target_arch = "x86_64")]
            Self::Avx2(_) => true,
            #[cfg(target_arch = "aarch64")]
            Self::Neon(_) => true,
        }
    }

    /// 8×8 inverse DCT of a dequantized coefficient block. Output is
    /// level-shifted by +128 and clamped to `[0, 255]` — bit-exact with
    /// [`idct::scalar::idct_islow`] on every legal JPEG input.
    pub(crate) fn idct(self, input: &[i16; 64], output: &mut [u8; 64]) {
        match self {
            Self::Scalar => idct::scalar::idct_islow(input, output),
            #[cfg(target_arch = "x86_64")]
            Self::Avx2(avx2) => idct::avx2::idct_islow(avx2, input, output),
            #[cfg(target_arch = "aarch64")]
            Self::Neon(neon) => idct::neon::idct_islow(neon, input, output),
        }
    }

    pub(crate) fn idct_bottom_half_zero(self, input: &[i16; 64], output: &mut [u8; 64]) {
        match self {
            Self::Scalar => idct::scalar::idct_islow_bottom_half_zero(input, output),
            #[cfg(target_arch = "x86_64")]
            Self::Avx2(avx2) => idct::avx2::idct_islow(avx2, input, output),
            #[cfg(target_arch = "aarch64")]
            Self::Neon(neon) => idct::neon::idct_islow_bottom_half_zero(neon, input, output),
        }
    }
}
