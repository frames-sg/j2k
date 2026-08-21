// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_core::{CompressedPayloadKind, CompressedTransferSyntax, PixelFormat};

use super::eligibility::AutoDecodeOperation;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PromotionOperation {
    Full,
    Region,
    ScaledHalf,
    Repeated,
}

impl From<AutoDecodeOperation> for PromotionOperation {
    fn from(value: AutoDecodeOperation) -> Self {
        match value {
            AutoDecodeOperation::Full => Self::Full,
            AutoDecodeOperation::Region => Self::Region,
            AutoDecodeOperation::ScaledHalf => Self::ScaledHalf,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct PromotionCell {
    pub(crate) source_components: u16,
    pub(crate) format: PixelFormat,
    pub(crate) transfer_syntax: CompressedTransferSyntax,
    pub(crate) payload_kind: CompressedPayloadKind,
    pub(crate) operation: PromotionOperation,
    pub(crate) minimum_width: u32,
    pub(crate) minimum_height: u32,
    pub(crate) minimum_count: usize,
    pub(crate) source_evidence: &'static str,
}

pub(super) fn qualifies(
    dimensions: (u32, u32),
    source_components: u16,
    format: PixelFormat,
    transfer_syntax: CompressedTransferSyntax,
    payload_kind: CompressedPayloadKind,
    operation: PromotionOperation,
    count: usize,
) -> bool {
    crate::generated::promotion::PROMOTION_CELLS
        .iter()
        .any(|cell| {
            cell.source_components == source_components
                && cell.format == format
                && cell.transfer_syntax == transfer_syntax
                && cell.payload_kind == payload_kind
                && cell.operation == operation
                && dimensions.0 >= cell.minimum_width
                && dimensions.1 >= cell.minimum_height
                && count >= cell.minimum_count
                && crate::generated::promotion::SOURCE_EVIDENCE.contains(&cell.source_evidence)
        })
}
