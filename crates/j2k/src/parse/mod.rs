// SPDX-License-Identifier: MIT OR Apache-2.0

mod boxes;
mod codestream;

use self::boxes::{extract_jp2_codestream_payload, parse_jp2};
use self::codestream::{parse_codestream, CodestreamInfo};
use crate::{J2kComponentInfo, J2kError, J2kFileMetadata, J2kSupportInfo};
use j2k_core::{
    Colorspace, CompressedPayloadKind, CompressedTransferSyntax, Info, TileLayout, Unsupported,
};

pub(crate) fn parse_info(input: &[u8]) -> Result<Info, J2kError> {
    Ok(parse_image_info(input)?.info)
}

pub(crate) fn parse_image_info(input: &[u8]) -> Result<ParsedImageInfo, J2kError> {
    if boxes::looks_like_jp2(input) {
        return parse_jp2(input);
    }
    if codestream::looks_like_codestream(input) {
        let parsed = parse_codestream(input)?;
        let info = parsed.clone().into_info(None);
        let components = parsed.siz.component_info.clone();
        return Ok(ParsedImageInfo {
            info,
            transfer_syntax: parsed.transfer_syntax(),
            payload_kind: CompressedPayloadKind::Jpeg2000Codestream,
            components,
            file_metadata: None,
        });
    }
    Err(J2kError::Unsupported(Unsupported {
        what: "input is not a JP2 container or raw JPEG 2000 codestream",
    }))
}

/// Borrowed raw codestream slice extracted from raw JPEG 2000 bytes or a JP2/JPH wrapper.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct J2kCodestreamPayload<'a> {
    codestream: &'a [u8],
    payload_kind: CompressedPayloadKind,
    codestream_offset: usize,
}

impl<'a> J2kCodestreamPayload<'a> {
    /// Raw JPEG 2000 codestream bytes.
    #[must_use]
    pub const fn codestream(self) -> &'a [u8] {
        self.codestream
    }

    /// Encapsulation shape the codestream was extracted from.
    #[must_use]
    pub const fn payload_kind(self) -> CompressedPayloadKind {
        self.payload_kind
    }

    /// Byte offset where the codestream payload starts in the original input.
    #[must_use]
    pub const fn codestream_offset(self) -> usize {
        self.codestream_offset
    }
}

/// Return the raw JPEG 2000 codestream payload from raw codestream, JP2, or JPH input.
///
/// This helper validates only the wrapper framing needed to locate the borrowed
/// codestream slice. Full JP2 metadata and ordering validation remains part of
/// the decoder inspect path.
pub fn extract_j2k_codestream_payload(input: &[u8]) -> Result<J2kCodestreamPayload<'_>, J2kError> {
    if codestream::looks_like_codestream(input) {
        return Ok(J2kCodestreamPayload {
            codestream: input,
            payload_kind: CompressedPayloadKind::Jpeg2000Codestream,
            codestream_offset: 0,
        });
    }
    if boxes::looks_like_jp2(input) {
        let (payload_kind, codestream_offset, codestream) = extract_jp2_codestream_payload(input)?;
        return Ok(J2kCodestreamPayload {
            codestream,
            payload_kind,
            codestream_offset,
        });
    }
    Err(J2kError::Unsupported(Unsupported {
        what: "input is not a JP2 container or raw JPEG 2000 codestream",
    }))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ParsedImageInfo {
    pub(crate) info: Info,
    pub(crate) transfer_syntax: CompressedTransferSyntax,
    pub(crate) payload_kind: CompressedPayloadKind,
    pub(crate) components: Vec<J2kComponentInfo>,
    pub(crate) file_metadata: Option<J2kFileMetadata>,
}

impl ParsedImageInfo {
    pub(crate) fn into_support_info(self) -> J2kSupportInfo {
        J2kSupportInfo {
            info: self.info,
            transfer_syntax: self.transfer_syntax,
            payload_kind: self.payload_kind,
            components: self.components,
            file_metadata: self.file_metadata,
        }
    }
}

fn infer_colorspace(components: u16, has_mct: bool, reversible: bool) -> Colorspace {
    match (components, has_mct, reversible) {
        (1, _, _) => Colorspace::SGray,
        (3, false, _) => Colorspace::Rgb,
        (3, true, false) => Colorspace::Ict,
        (3, true, true) => Colorspace::Rct,
        _ => Colorspace::IccTagged,
    }
}

#[derive(Debug, Clone)]
struct ParsedSiz {
    dimensions: (u32, u32),
    components: u16,
    bit_depth: u8,
    tile_layout: TileLayout,
    component_info: Vec<J2kComponentInfo>,
}

#[derive(Debug, Clone, Copy)]
struct ParsedCod {
    resolution_levels: u8,
    has_mct: bool,
    reversible: bool,
    high_throughput: bool,
}

impl CodestreamInfo {
    fn into_info(self, colorspace: Option<Colorspace>) -> Info {
        Info {
            dimensions: self.siz.dimensions,
            components: self.siz.components,
            colorspace: colorspace.unwrap_or_else(|| {
                infer_colorspace(self.siz.components, self.cod.has_mct, self.cod.reversible)
            }),
            bit_depth: self.siz.bit_depth,
            tile_layout: Some(self.siz.tile_layout),
            coded_unit_layout: None,
            restart_interval: None,
            resolution_levels: self.cod.resolution_levels,
        }
    }

    fn transfer_syntax(self) -> CompressedTransferSyntax {
        match (self.cod.high_throughput, self.cod.reversible) {
            (false, true) => CompressedTransferSyntax::Jpeg2000Lossless,
            (false, false) => CompressedTransferSyntax::Jpeg2000Lossy,
            (true, true) => CompressedTransferSyntax::HtJpeg2000Lossless,
            (true, false) => CompressedTransferSyntax::HtJpeg2000Lossy,
        }
    }
}
