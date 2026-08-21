// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_core::{CompressedPayloadKind, CompressedTransferSyntax, Downscale, PixelFormat};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PromotionOperation {
    ScaledHalf,
    Repeated,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct PromotionCell {
    pub(crate) format: PixelFormat,
    pub(crate) transfer_syntax: CompressedTransferSyntax,
    pub(crate) payload_kind: CompressedPayloadKind,
    pub(crate) operation: PromotionOperation,
    pub(crate) minimum_width: u32,
    pub(crate) minimum_height: u32,
    pub(crate) minimum_pixels: u64,
    pub(crate) minimum_count: usize,
    pub(crate) source_evidence: &'static str,
}

pub(crate) fn auto_scaled_decode_uses_metal(
    dimensions: (u32, u32),
    source_components: u16,
    format: PixelFormat,
    transfer_syntax: CompressedTransferSyntax,
    payload_kind: CompressedPayloadKind,
    scale: Downscale,
) -> bool {
    source_components == 3
        && scale == Downscale::Half
        && qualifies(
            dimensions,
            format,
            transfer_syntax,
            payload_kind,
            PromotionOperation::ScaledHalf,
            1,
        )
}

pub(crate) fn auto_repeated_decode_uses_metal(
    dimensions: (u32, u32),
    format: PixelFormat,
    count: usize,
    transfer_syntax: CompressedTransferSyntax,
    payload_kind: CompressedPayloadKind,
) -> bool {
    qualifies(
        dimensions,
        format,
        transfer_syntax,
        payload_kind,
        PromotionOperation::Repeated,
        count,
    )
}

fn qualifies(
    dimensions: (u32, u32),
    format: PixelFormat,
    transfer_syntax: CompressedTransferSyntax,
    payload_kind: CompressedPayloadKind,
    operation: PromotionOperation,
    count: usize,
) -> bool {
    let pixels = u64::from(dimensions.0) * u64::from(dimensions.1);
    crate::generated::promotion::PROMOTION_CELLS
        .iter()
        .any(|cell| {
            cell.format == format
                && cell.transfer_syntax == transfer_syntax
                && cell.payload_kind == payload_kind
                && cell.operation == operation
                && dimensions.0 >= cell.minimum_width
                && dimensions.1 >= cell.minimum_height
                && pixels >= cell.minimum_pixels
                && count >= cell.minimum_count
                && crate::generated::promotion::SOURCE_EVIDENCE.contains(&cell.source_evidence)
        })
}
