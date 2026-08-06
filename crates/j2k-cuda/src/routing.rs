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
    if count < REPEATED_DECODE_MIN_COUNT
        || payload_kind != CompressedPayloadKind::Jpeg2000Codestream
    {
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
mod tests {
    use j2k_core::{CompressedPayloadKind, CompressedTransferSyntax, PixelFormat};

    use super::{
        auto_decode_uses_cuda, auto_repeated_decode_uses_cuda, inputs_repeat_one_slice,
        AutoDecodeOperation,
    };

    #[test]
    fn single_image_thresholds_match_verified_external_cells() {
        use AutoDecodeOperation::{Full, Region, ScaledHalf};
        use CompressedTransferSyntax::{Jpeg2000Lossless as Lossless, Jpeg2000Lossy as Lossy};
        const RAW: CompressedPayloadKind = CompressedPayloadKind::Jpeg2000Codestream;

        let cases = [
            ((256, 149), 3, PixelFormat::Rgb8, Lossless, Full, true),
            ((128, 74), 3, PixelFormat::Rgb8, Lossless, Region, true),
            ((128, 75), 3, PixelFormat::Rgb8, Lossless, ScaledHalf, false),
            ((256, 149), 3, PixelFormat::Rgb8, Lossy, Full, false),
            ((640, 480), 3, PixelFormat::Rgb8, Lossy, Full, true),
            ((320, 240), 3, PixelFormat::Rgb8, Lossy, Region, true),
            ((320, 240), 3, PixelFormat::Rgb8, Lossy, ScaledHalf, true),
            ((640, 480), 1, PixelFormat::Gray8, Lossless, Full, true),
            ((320, 240), 1, PixelFormat::Gray8, Lossless, Region, false),
            (
                (320, 240),
                1,
                PixelFormat::Gray8,
                Lossless,
                ScaledHalf,
                false,
            ),
            ((3323, 891), 1, PixelFormat::Gray8, Lossy, Full, true),
            ((1661, 445), 1, PixelFormat::Gray8, Lossy, Region, true),
            ((1662, 446), 1, PixelFormat::Gray8, Lossy, ScaledHalf, true),
            (
                (1296, 972),
                3,
                PixelFormat::Rgb8,
                Lossless,
                ScaledHalf,
                true,
            ),
        ];
        for (dimensions, components, fmt, transfer_syntax, operation, expected) in cases {
            assert_eq!(
                auto_decode_uses_cuda(dimensions, components, fmt, transfer_syntax, RAW, operation,),
                expected,
                "{dimensions:?} {fmt:?} {transfer_syntax:?} {operation:?}",
            );
        }
    }

    #[test]
    fn auto_decode_keeps_unmeasured_surfaces_on_cpu() {
        for transfer_syntax in [
            CompressedTransferSyntax::HtJpeg2000Lossless,
            CompressedTransferSyntax::HtJpeg2000Lossy,
        ] {
            assert!(!auto_decode_uses_cuda(
                (4096, 4096),
                3,
                PixelFormat::Rgb8,
                transfer_syntax,
                CompressedPayloadKind::Jpeg2000Codestream,
                AutoDecodeOperation::Full,
            ));
        }
        for fmt in [
            PixelFormat::Gray16,
            PixelFormat::Rgb16,
            PixelFormat::Rgba8,
            PixelFormat::Rgba16,
        ] {
            assert!(!auto_decode_uses_cuda(
                (4096, 4096),
                match fmt {
                    PixelFormat::Gray16 => 1,
                    PixelFormat::Rgb16 => 3,
                    PixelFormat::Rgba8 | PixelFormat::Rgba16 => 4,
                    _ => unreachable!("test enumerates only higher-depth and alpha formats"),
                },
                fmt,
                CompressedTransferSyntax::Jpeg2000Lossy,
                CompressedPayloadKind::Jpeg2000Codestream,
                AutoDecodeOperation::Full,
            ));
        }
    }

    #[test]
    fn auto_decode_requires_the_measured_source_component_count() {
        assert!(!auto_decode_uses_cuda(
            (2592, 1944),
            1,
            PixelFormat::Rgb8,
            CompressedTransferSyntax::Jpeg2000Lossless,
            CompressedPayloadKind::Jpeg2000Codestream,
            AutoDecodeOperation::Full,
        ));
        assert!(!auto_decode_uses_cuda(
            (3323, 891),
            3,
            PixelFormat::Gray8,
            CompressedTransferSyntax::Jpeg2000Lossy,
            CompressedPayloadKind::Jpeg2000Codestream,
            AutoDecodeOperation::Full,
        ));
    }

    #[test]
    fn auto_decode_keeps_wrapped_and_below_threshold_work_on_cpu() {
        assert!(!auto_decode_uses_cuda(
            (640, 480),
            3,
            PixelFormat::Rgb8,
            CompressedTransferSyntax::Jpeg2000Lossy,
            CompressedPayloadKind::Jp2File,
            AutoDecodeOperation::Full,
        ));
        assert!(!auto_decode_uses_cuda(
            (319, 240),
            3,
            PixelFormat::Rgb8,
            CompressedTransferSyntax::Jpeg2000Lossy,
            CompressedPayloadKind::Jpeg2000Codestream,
            AutoDecodeOperation::Region,
        ));
    }

    #[test]
    fn repeated_batch_thresholds_match_verified_external_cells() {
        use CompressedTransferSyntax::{Jpeg2000Lossless as Lossless, Jpeg2000Lossy as Lossy};
        const RAW: CompressedPayloadKind = CompressedPayloadKind::Jpeg2000Codestream;

        let cases = [
            ((256, 149), 3, PixelFormat::Rgb8, Lossy, 16, true),
            ((640, 480), 1, PixelFormat::Gray8, Lossless, 16, true),
            ((256, 149), 1, PixelFormat::Gray8, Lossless, 16, false),
            ((2592, 1944), 3, PixelFormat::Rgb8, Lossless, 15, false),
        ];
        for (dimensions, components, fmt, transfer_syntax, count, expected) in cases {
            assert_eq!(
                auto_repeated_decode_uses_cuda(
                    dimensions,
                    components,
                    fmt,
                    transfer_syntax,
                    RAW,
                    count,
                ),
                expected,
                "{dimensions:?} {fmt:?} {transfer_syntax:?} count={count}",
            );
        }
    }

    #[test]
    fn repeated_batch_requires_one_shared_input_slice() {
        let bytes = [1, 2, 3, 4];
        let copied = bytes;

        assert!(inputs_repeat_one_slice(&[&bytes, &bytes]));
        assert!(!inputs_repeat_one_slice(&[&bytes, &copied]));
        assert!(!inputs_repeat_one_slice(&[]));
    }
}
