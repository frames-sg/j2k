// SPDX-License-Identifier: MIT OR Apache-2.0

//! Borrowed row-scratch variants selected by sequential output routing.

use crate::internal::scratch::{RgbGenericRows, YCbCr420Rows, YCbCrGenericRows};

pub(super) enum OutputScratch<'a> {
    Grayscale,
    YCbCr420(&'a mut YCbCr420Rows),
    YCbCrGeneric(&'a mut YCbCrGenericRows),
    RgbGeneric(&'a mut RgbGenericRows),
}

pub(super) enum RgbOutputScratch<'a> {
    None,
    YCbCr420,
    YCbCrGeneric(&'a mut YCbCrGenericRows),
    RgbGeneric(&'a mut RgbGenericRows),
}
