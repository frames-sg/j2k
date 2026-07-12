// SPDX-License-Identifier: MIT OR Apache-2.0

//! Ownership-aware final codestream/JPH output routing.

use alloc::vec::Vec;

use super::allocation::checked_add_owned_bytes;
use crate::{parse::ParsedImageInfo, wrap::wrap_recode_jph_codestream, J2kError};
use j2k_core::{CompressedPayloadKind, Unsupported};

pub(super) fn finalize_owned(
    codestream: Vec<u8>,
    payload_kind: CompressedPayloadKind,
    parsed: &ParsedImageInfo,
    preserve_file_metadata: bool,
) -> Result<Vec<u8>, J2kError> {
    match payload_kind {
        CompressedPayloadKind::Jpeg2000Codestream => Ok(codestream),
        CompressedPayloadKind::JphFile => {
            let parsed_bytes = parsed.allocated_bytes()?;
            checked_add_owned_bytes(
                parsed_bytes,
                codestream.capacity(),
                "HTJ2K codestream plus retained recode metadata",
            )?;
            wrap_recode_jph_codestream(
                &codestream,
                parsed.file_metadata.as_ref(),
                preserve_file_metadata,
                parsed_bytes,
                codestream.capacity(),
            )
        }
        _ => Err(Unsupported {
            what: "J2K to HTJ2K recode output must be a raw HTJ2K codestream or JPH file",
        }
        .into()),
    }
}

pub(super) fn wrap_borrowed_jph(
    codestream: &[u8],
    parsed: &ParsedImageInfo,
    preserve_file_metadata: bool,
) -> Result<Vec<u8>, J2kError> {
    let parsed_bytes = parsed.allocated_bytes()?;
    wrap_recode_jph_codestream(
        codestream,
        parsed.file_metadata.as_ref(),
        preserve_file_metadata,
        parsed_bytes,
        0,
    )
}
