// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_core::{CompressedPayloadKind, CompressedTransferSyntax, PixelFormat};

use crate::{CudaSession, Error};

// Minimum qualified cells from verified CUDA Auto-routing artifact
// ded1eb045f9673e5bbe64dc873be3ba227ecb61ec11b6c9ad53653dbcc993f44.
// These thresholds apply only to measured raw Part 1 Gray8/Rgb8 surfaces,
// including host readback. No CUDA encode cell qualified.
const RGB_SMALL_FULL: (u32, u32) = (256, 149);
const RGB_SMALL_ROI: (u32, u32) = (128, 74);
const MEDIUM_FULL: (u32, u32) = (640, 480);
const MEDIUM_HALF: (u32, u32) = (320, 240);
const GRAY_LARGE_FULL: (u32, u32) = (3323, 891);
const GRAY_LARGE_ROI: (u32, u32) = (1661, 445);
const GRAY_LARGE_HALF: (u32, u32) = (1662, 446);
const RGB_LARGE_HALF: (u32, u32) = (1296, 972);
const REPEATED_DECODE_MIN_COUNT: usize = 16;

// Additional Part 15 cells from verified development artifact
// 77370c83710ebf578139ad0bfa2608ffad989d83faec8d5eee213691290c0088.
// Replace this provisional hash with an exact-SHA release rerun before release.
const HT_RAW_LOSSY_FULL: (u32, u32) = (640, 480);
const HT_RAW_LOSSY_HALF: (u32, u32) = (320, 240);
const HT_JPH_LOSSLESS_FULL: (u32, u32) = (768, 512);
const HT_JPH_LOSSLESS_HALF: (u32, u32) = (384, 256);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum AutoDecodeOperation {
    Full,
    Region,
    ScaledHalf,
}

pub(crate) fn auto_decode_uses_cuda(
    work_dimensions: (u32, u32),
    source_components: u16,
    fmt: PixelFormat,
    transfer_syntax: CompressedTransferSyntax,
    payload_kind: CompressedPayloadKind,
    operation: AutoDecodeOperation,
) -> bool {
    let ht_thresholds = match (source_components, fmt, transfer_syntax, payload_kind) {
        (
            3,
            PixelFormat::Rgb8,
            CompressedTransferSyntax::HtJpeg2000Lossy,
            CompressedPayloadKind::Jpeg2000Codestream,
        ) => Some((HT_RAW_LOSSY_FULL, HT_RAW_LOSSY_HALF)),
        (
            3,
            PixelFormat::Rgb8,
            CompressedTransferSyntax::HtJpeg2000Lossless,
            CompressedPayloadKind::JphFile,
        ) => Some((HT_JPH_LOSSLESS_FULL, HT_JPH_LOSSLESS_HALF)),
        _ => None,
    };
    if let Some((full, partial)) = ht_thresholds {
        let minimum = match operation {
            AutoDecodeOperation::Full => full,
            AutoDecodeOperation::Region | AutoDecodeOperation::ScaledHalf => partial,
        };
        return dimensions_at_least(work_dimensions, minimum);
    }
    if payload_kind != CompressedPayloadKind::Jpeg2000Codestream {
        return false;
    }
    let minimum = match (operation, source_components, fmt, transfer_syntax) {
        (
            AutoDecodeOperation::Full,
            3,
            PixelFormat::Rgb8,
            CompressedTransferSyntax::Jpeg2000Lossless,
        ) => RGB_SMALL_FULL,
        (
            AutoDecodeOperation::Region,
            3,
            PixelFormat::Rgb8,
            CompressedTransferSyntax::Jpeg2000Lossless,
        ) => RGB_SMALL_ROI,
        (
            AutoDecodeOperation::Full
            | AutoDecodeOperation::Region
            | AutoDecodeOperation::ScaledHalf,
            3,
            PixelFormat::Rgb8,
            CompressedTransferSyntax::Jpeg2000Lossy,
        ) => match operation {
            AutoDecodeOperation::Full => MEDIUM_FULL,
            AutoDecodeOperation::Region | AutoDecodeOperation::ScaledHalf => MEDIUM_HALF,
        },
        (
            AutoDecodeOperation::Full,
            1,
            PixelFormat::Gray8,
            CompressedTransferSyntax::Jpeg2000Lossless,
        ) => MEDIUM_FULL,
        (
            AutoDecodeOperation::Full
            | AutoDecodeOperation::Region
            | AutoDecodeOperation::ScaledHalf,
            1,
            PixelFormat::Gray8,
            CompressedTransferSyntax::Jpeg2000Lossy,
        ) => match operation {
            AutoDecodeOperation::Full => GRAY_LARGE_FULL,
            AutoDecodeOperation::Region => GRAY_LARGE_ROI,
            AutoDecodeOperation::ScaledHalf => GRAY_LARGE_HALF,
        },
        (
            AutoDecodeOperation::ScaledHalf,
            3,
            PixelFormat::Rgb8,
            CompressedTransferSyntax::Jpeg2000Lossless,
        ) => RGB_LARGE_HALF,
        _ => return false,
    };
    dimensions_at_least(work_dimensions, minimum)
}

pub(crate) fn auto_repeated_decode_uses_cuda(
    dimensions: (u32, u32),
    source_components: u16,
    fmt: PixelFormat,
    transfer_syntax: CompressedTransferSyntax,
    payload_kind: CompressedPayloadKind,
    count: usize,
) -> bool {
    if count < REPEATED_DECODE_MIN_COUNT {
        return false;
    }
    let ht_minimum = match (source_components, fmt, transfer_syntax, payload_kind) {
        (
            3,
            PixelFormat::Rgb8,
            CompressedTransferSyntax::HtJpeg2000Lossy,
            CompressedPayloadKind::Jpeg2000Codestream,
        ) => Some(HT_RAW_LOSSY_FULL),
        (
            3,
            PixelFormat::Rgb8,
            CompressedTransferSyntax::HtJpeg2000Lossless,
            CompressedPayloadKind::JphFile,
        ) => Some(HT_JPH_LOSSLESS_FULL),
        _ => None,
    };
    if let Some(minimum) = ht_minimum {
        return dimensions_at_least(dimensions, minimum);
    }
    if payload_kind != CompressedPayloadKind::Jpeg2000Codestream {
        return false;
    }
    let minimum = match (source_components, fmt, transfer_syntax) {
        (
            3,
            PixelFormat::Rgb8,
            CompressedTransferSyntax::Jpeg2000Lossless | CompressedTransferSyntax::Jpeg2000Lossy,
        ) => RGB_SMALL_FULL,
        (1, PixelFormat::Gray8, CompressedTransferSyntax::Jpeg2000Lossless) => MEDIUM_FULL,
        (1, PixelFormat::Gray8, CompressedTransferSyntax::Jpeg2000Lossy) => GRAY_LARGE_FULL,
        _ => return false,
    };
    dimensions_at_least(dimensions, minimum)
}

pub(crate) fn inputs_repeat_one_slice(inputs: &[&[u8]]) -> bool {
    let Some(first) = inputs.first().copied() else {
        return false;
    };
    inputs
        .iter()
        .copied()
        .all(|input| core::ptr::eq(input, first))
}

pub(crate) fn auto_cuda_available(session: &mut CudaSession) -> Result<bool, Error> {
    #[cfg(feature = "cuda-runtime")]
    {
        match session.cuda_context() {
            Ok(_) => Ok(true),
            Err(Error::CudaUnavailable) => Ok(false),
            Err(error) => Err(error),
        }
    }
    #[cfg(not(feature = "cuda-runtime"))]
    {
        let _ = session;
        Ok(false)
    }
}

fn dimensions_at_least(actual: (u32, u32), minimum: (u32, u32)) -> bool {
    actual.0 >= minimum.0 && actual.1 >= minimum.1
}

#[cfg(test)]
mod tests;
