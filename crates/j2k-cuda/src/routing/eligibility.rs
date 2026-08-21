// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_core::{CompressedPayloadKind, CompressedTransferSyntax, PixelFormat};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum AutoDecodeOperation {
    Full,
    Region,
    ScaledHalf,
}

pub(super) fn decode_is_eligible(
    source_components: u16,
    fmt: PixelFormat,
    transfer_syntax: CompressedTransferSyntax,
    payload_kind: CompressedPayloadKind,
) -> bool {
    matches!(
        (source_components, fmt, transfer_syntax, payload_kind),
        (
            1,
            PixelFormat::Gray8,
            CompressedTransferSyntax::Jpeg2000Lossless | CompressedTransferSyntax::Jpeg2000Lossy,
            CompressedPayloadKind::Jpeg2000Codestream,
        ) | (
            3,
            PixelFormat::Rgb8,
            CompressedTransferSyntax::Jpeg2000Lossless
                | CompressedTransferSyntax::Jpeg2000Lossy
                | CompressedTransferSyntax::HtJpeg2000Lossy,
            CompressedPayloadKind::Jpeg2000Codestream,
        ) | (
            3,
            PixelFormat::Rgb8,
            CompressedTransferSyntax::HtJpeg2000Lossless,
            CompressedPayloadKind::JphFile,
        )
    )
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
