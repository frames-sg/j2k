// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_core::{CompressedPayloadKind, CompressedTransferSyntax};

use super::{AutoRoutingCodec, AutoRoutingContainer, AutoRoutingWorkload, AutoRoutingWorkloadKind};

/// Check a declared decode workload identity against production inspection.
///
/// # Errors
///
/// Returns an error when the workload is not a decode input or its declared
/// coding system/container does not match the inspected bytes.
pub fn validate_auto_routing_decode_identity(
    workload: &AutoRoutingWorkload,
    transfer_syntax: CompressedTransferSyntax,
    payload_kind: CompressedPayloadKind,
) -> Result<(), String> {
    if workload.kind != AutoRoutingWorkloadKind::Decode {
        return Err(format!(
            "Auto-routing workload {} is not a decode input",
            workload.id
        ));
    }
    let inspected_codec = match transfer_syntax {
        CompressedTransferSyntax::Jpeg2000Lossless | CompressedTransferSyntax::Jpeg2000Lossy => {
            AutoRoutingCodec::Jpeg2000Part1
        }
        CompressedTransferSyntax::HtJpeg2000Lossless
        | CompressedTransferSyntax::HtJpeg2000Lossy => AutoRoutingCodec::Htj2kPart15,
        _ => {
            return Err(format!(
                "Auto-routing workload {} is not JPEG 2000",
                workload.id
            ))
        }
    };
    let inspected_container = match payload_kind {
        CompressedPayloadKind::Jpeg2000Codestream => AutoRoutingContainer::Codestream,
        CompressedPayloadKind::Jp2File => AutoRoutingContainer::Jp2,
        CompressedPayloadKind::JphFile => AutoRoutingContainer::Jph,
        _ => {
            return Err(format!(
                "Auto-routing workload {} has an unsupported payload kind",
                workload.id
            ))
        }
    };
    if (workload.codec, workload.container) != (inspected_codec, inspected_container) {
        return Err(format!(
            "Auto-routing workload {} declared {:?}/{:?}, which does not match inspected {:?}/{:?}",
            workload.id, workload.codec, workload.container, inspected_codec, inspected_container
        ));
    }
    Ok(())
}
