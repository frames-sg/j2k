// SPDX-License-Identifier: Apache-2.0

mod boxes;
mod codestream;

use self::boxes::parse_jp2;
use self::codestream::{parse_codestream, CodestreamInfo};
use crate::J2kError;
use signinum_core::{
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
    pub(crate) components: Vec<ParsedComponentInfo>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ParsedComponentInfo {
    pub(crate) bit_depth: u8,
    pub(crate) signed: bool,
    pub(crate) x_rsiz: u8,
    pub(crate) y_rsiz: u8,
}

fn infer_colorspace(components: u8, has_mct: bool, reversible: bool) -> Colorspace {
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
    components: u8,
    bit_depth: u8,
    tile_layout: TileLayout,
    component_info: Vec<ParsedComponentInfo>,
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
