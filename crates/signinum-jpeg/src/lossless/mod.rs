// SPDX-License-Identifier: Apache-2.0

//! Shared lossless JPEG decode helpers.

use crate::error::{HuffmanFailure, JpegError};

pub(crate) trait LosslessSample: Copy + Default + Into<i32> {
    const RESTART_PREDICTOR: i32;
    const BIT_DEPTH: u8;
    const BYTES: usize;

    fn from_i32(value: i32) -> Result<Self, JpegError>;

    fn read_le(src: &[u8]) -> i32;

    fn write_le(self, dst: &mut [u8]);
}

impl LosslessSample for u8 {
    const RESTART_PREDICTOR: i32 = 128;
    const BIT_DEPTH: u8 = 8;
    const BYTES: usize = 1;

    fn from_i32(value: i32) -> Result<Self, JpegError> {
        u8::try_from(value).map_err(|_| invalid_lossless_symbol())
    }

    fn read_le(src: &[u8]) -> i32 {
        i32::from(src[0])
    }

    fn write_le(self, dst: &mut [u8]) {
        dst[0] = self;
    }
}

impl LosslessSample for u16 {
    const RESTART_PREDICTOR: i32 = 32_768;
    const BIT_DEPTH: u8 = 16;
    const BYTES: usize = 2;

    fn from_i32(value: i32) -> Result<Self, JpegError> {
        u16::try_from(value).map_err(|_| invalid_lossless_symbol())
    }

    fn read_le(src: &[u8]) -> i32 {
        i32::from(u16::from_le_bytes([src[0], src[1]]))
    }

    fn write_le(self, dst: &mut [u8]) {
        dst[..2].copy_from_slice(&self.to_le_bytes());
    }
}

fn invalid_lossless_symbol() -> JpegError {
    JpegError::HuffmanDecode {
        mcu: 0,
        reason: HuffmanFailure::InvalidSymbol,
    }
}

/// Spec predictor (ITU-T T.81 H.1.2.1) over a sample accessor.
///
/// Edge rules: the first sample predicts `bias` (1 << (P - 1)); the first row
/// predicts Ra; the first column predicts Rb. `at(x, y)` must return the
/// reconstructed sample at the given coordinates; only `(x-1, y)`, `(x, y-1)`
/// and `(x-1, y-1)` are ever requested.
#[allow(clippy::inline_always)] // per-sample hot path: keep the accessor closure monomorphized
#[inline(always)]
pub(crate) fn lossless_predict(
    predictor: u8,
    bias: i32,
    x: usize,
    y: usize,
    at: impl Fn(usize, usize) -> i32,
) -> i32 {
    if x == 0 && y == 0 {
        return bias;
    }
    if y == 0 {
        return at(x - 1, 0);
    }
    if x == 0 {
        return at(0, y - 1);
    }
    let ra = at(x - 1, y);
    let rb = at(x, y - 1);
    let rc = at(x - 1, y - 1);
    match predictor {
        1 => ra,
        2 => rb,
        3 => rc,
        4 => ra + rb - rc,
        5 => ra + ((rb - rc) >> 1),
        6 => rb + ((ra - rc) >> 1),
        7 => (ra + rb) >> 1,
        _ => bias,
    }
}
