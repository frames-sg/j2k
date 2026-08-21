// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_core::{CompressedPayloadKind, CompressedTransferSyntax, PixelFormat};

use super::{
    eligibility::{decode_is_eligible, AutoDecodeOperation},
    promotion::{qualifies, PromotionOperation},
    rejection::AutoCudaRejection,
    telemetry,
};

pub(crate) fn auto_decode_uses_cuda(
    dimensions: (u32, u32),
    source_components: u16,
    format: PixelFormat,
    transfer_syntax: CompressedTransferSyntax,
    payload_kind: CompressedPayloadKind,
    operation: AutoDecodeOperation,
) -> bool {
    decide(
        dimensions,
        source_components,
        format,
        transfer_syntax,
        payload_kind,
        operation.into(),
        1,
    )
}

pub(crate) fn auto_repeated_decode_uses_cuda(
    dimensions: (u32, u32),
    source_components: u16,
    format: PixelFormat,
    transfer_syntax: CompressedTransferSyntax,
    payload_kind: CompressedPayloadKind,
    count: usize,
) -> bool {
    decide(
        dimensions,
        source_components,
        format,
        transfer_syntax,
        payload_kind,
        PromotionOperation::Repeated,
        count,
    )
}

fn decide(
    dimensions: (u32, u32),
    source_components: u16,
    format: PixelFormat,
    transfer_syntax: CompressedTransferSyntax,
    payload_kind: CompressedPayloadKind,
    operation: PromotionOperation,
    count: usize,
) -> bool {
    if !decode_is_eligible(source_components, format, transfer_syntax, payload_kind) {
        let _ = telemetry::observe(false, Some(AutoCudaRejection::Ineligible));
        return false;
    }
    let promoted = qualifies(
        dimensions,
        source_components,
        format,
        transfer_syntax,
        payload_kind,
        operation,
        count,
    );
    let _ = telemetry::observe(
        promoted,
        (!promoted).then_some(AutoCudaRejection::NotBenchmarkQualified),
    );
    promoted
}
