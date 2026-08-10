// SPDX-License-Identifier: MIT OR Apache-2.0

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
    for (transfer_syntax, payload_kind) in [
        (
            CompressedTransferSyntax::HtJpeg2000Lossless,
            CompressedPayloadKind::Jpeg2000Codestream,
        ),
        (
            CompressedTransferSyntax::HtJpeg2000Lossy,
            CompressedPayloadKind::JphFile,
        ),
    ] {
        assert!(!auto_decode_uses_cuda(
            (4096, 4096),
            3,
            PixelFormat::Rgb8,
            transfer_syntax,
            payload_kind,
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
fn part15_single_image_thresholds_match_verified_external_cells() {
    use AutoDecodeOperation::{Full, Region, ScaledHalf};
    use CompressedPayloadKind::{Jpeg2000Codestream as Raw, JphFile as Jph};
    use CompressedTransferSyntax::{HtJpeg2000Lossless as Lossless, HtJpeg2000Lossy as Lossy};

    let cases = [
        ((640, 480), Lossy, Raw, Full, true),
        ((639, 480), Lossy, Raw, Full, false),
        ((320, 240), Lossy, Raw, Region, true),
        ((319, 240), Lossy, Raw, Region, false),
        ((320, 240), Lossy, Raw, ScaledHalf, true),
        ((768, 512), Lossless, Jph, Full, true),
        ((767, 512), Lossless, Jph, Full, false),
        ((384, 256), Lossless, Jph, Region, true),
        ((383, 256), Lossless, Jph, Region, false),
        ((384, 256), Lossless, Jph, ScaledHalf, true),
    ];
    for (dimensions, transfer_syntax, payload_kind, operation, expected) in cases {
        assert_eq!(
            auto_decode_uses_cuda(
                dimensions,
                3,
                PixelFormat::Rgb8,
                transfer_syntax,
                payload_kind,
                operation,
            ),
            expected,
            "{dimensions:?} {transfer_syntax:?} {payload_kind:?} {operation:?}",
        );
    }

    assert!(!auto_decode_uses_cuda(
        (768, 512),
        1,
        PixelFormat::Rgb8,
        Lossless,
        Jph,
        Full,
    ));
    assert!(!auto_decode_uses_cuda(
        (768, 512),
        3,
        PixelFormat::Rgba8,
        Lossless,
        Jph,
        Full,
    ));
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
fn part15_repeated_batch_thresholds_match_verified_external_cells() {
    use CompressedPayloadKind::{Jpeg2000Codestream as Raw, JphFile as Jph};
    use CompressedTransferSyntax::{HtJpeg2000Lossless as Lossless, HtJpeg2000Lossy as Lossy};

    let cases = [
        ((640, 480), Lossy, Raw, 16, true),
        ((639, 480), Lossy, Raw, 16, false),
        ((640, 480), Lossy, Raw, 15, false),
        ((768, 512), Lossless, Jph, 16, true),
        ((767, 512), Lossless, Jph, 16, false),
        ((768, 512), Lossless, Jph, 15, false),
        ((768, 512), Lossless, Raw, 16, false),
        ((768, 512), Lossy, Jph, 16, false),
    ];
    for (dimensions, transfer_syntax, payload_kind, count, expected) in cases {
        assert_eq!(
            auto_repeated_decode_uses_cuda(
                dimensions,
                3,
                PixelFormat::Rgb8,
                transfer_syntax,
                payload_kind,
                count,
            ),
            expected,
            "{dimensions:?} {transfer_syntax:?} {payload_kind:?} count={count}",
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
