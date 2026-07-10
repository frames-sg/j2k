// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec;
use alloc::vec::Vec;

use super::super::build::SubBand;
use super::super::decode::DecompositionStorage;
use super::super::rect::IntRect;

/// The output from performing the IDWT operation.
pub(crate) struct IDWTOutput {
    /// The buffer that will hold the final coefficients.
    pub(crate) coefficients: Vec<f32>,
    /// The buffer that will hold exact reversible integer coefficients.
    pub(crate) coefficients_i64: Vec<i64>,
    /// The rect that the coefficients belong to. This will be equivalent
    /// to the rectangle that forms the smallest decomposition level. It does
    /// not have to be equivalent to the original size of the tile, as the
    /// sub-bands that form a tile aren't necessarily aligned to it. Therefore,
    /// the samples need to be trimmed to the tile rectangle afterward.
    pub(crate) rect: IntRect,
}

impl Default for IDWTOutput {
    fn default() -> Self {
        Self {
            coefficients: vec![],
            coefficients_i64: vec![],
            rect: IntRect::from_ltrb(0, 0, u32::MAX, u32::MAX),
        }
    }
}

pub(super) struct IDWTTempOutput {
    pub(super) rect: IntRect,
}

#[derive(Clone, Copy)]
pub(super) enum InputSource {
    SubBand,
    Scratch,
    Output,
}

#[derive(Clone, Copy)]
pub(super) struct CoefficientSource<'a> {
    pub(super) coefficients: &'a [f32],
    pub(super) rect: IntRect,
    pub(super) stride: u32,
}

impl<'a> CoefficientSource<'a> {
    pub(super) fn new(coefficients: &'a [f32], rect: IntRect, stride: u32) -> Self {
        Self {
            coefficients,
            rect,
            stride,
        }
    }

    pub(super) fn from_sub_band(
        sub_band: &'a SubBand,
        storage: &'a DecompositionStorage<'_>,
    ) -> Self {
        Self {
            coefficients: &storage.coefficients[sub_band.coefficients.clone()],
            rect: sub_band.rect,
            stride: sub_band.rect.width(),
        }
    }

    pub(super) fn get(self, x: u32, y: u32) -> f32 {
        if x < self.rect.x0 || x >= self.rect.x1 || y < self.rect.y0 || y >= self.rect.y1 {
            return 0.0;
        }
        let local_x = (x - self.rect.x0) as usize;
        let local_y = (y - self.rect.y0) as usize;
        self.coefficients[local_y * self.stride as usize + local_x]
    }
}

#[derive(Clone, Copy)]
pub(super) struct IDWTInputI64<'a> {
    pub(super) coefficients: &'a [i64],
}

impl<'a> IDWTInputI64<'a> {
    pub(super) fn from_sub_band(
        sub_band: &'a SubBand,
        storage: &'a DecompositionStorage<'_>,
    ) -> Self {
        IDWTInputI64 {
            coefficients: &storage.coefficients_i64[sub_band.coefficients.clone()],
        }
    }

    pub(super) fn from_output(coefficients: &'a [i64], _rect: IntRect) -> Self {
        IDWTInputI64 { coefficients }
    }
}

#[derive(Clone, Copy)]
pub(super) struct IDWTInput<'a> {
    pub(super) rect: IntRect,
    pub(super) coefficients: &'a [f32],
}

impl<'a> IDWTInput<'a> {
    pub(super) fn from_sub_band(
        sub_band: &'a SubBand,
        storage: &'a DecompositionStorage<'_>,
    ) -> Self {
        IDWTInput {
            rect: sub_band.rect,
            coefficients: &storage.coefficients[sub_band.coefficients.clone()],
        }
    }

    pub(super) fn from_output(coefficients: &'a [f32], rect: IntRect) -> Self {
        IDWTInput { rect, coefficients }
    }
}
