// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec::Vec;

use j2k_core::{Colorspace, CompressedPayloadKind, CompressedTransferSyntax, Unsupported};

use crate::{
    parse::parse_image_info, J2kChannelDefinition, J2kColorSpec, J2kComponentMapping, J2kError,
    J2kFileMetadata, J2kPaletteMetadata,
};

mod allocation;
mod color;
mod metadata;
mod plan;
mod writer;

const JP2_SIGNATURE_PAYLOAD: &[u8; 4] = &[0x0d, 0x0a, 0x87, 0x0a];
const JP2_BRAND: [u8; 4] = *b"jp2 ";
const JPH_BRAND: [u8; 4] = *b"jph ";
const JP2_COMPRESSION_TYPE: u8 = 7;

/// Color metadata to write into a JP2/JPH Colour Specification box.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum J2kFileColorSpec<'a> {
    /// Infer an enumerated JP2 colorspace from the codestream component count
    /// and parsed JPEG 2000 color transform metadata.
    Infer,
    /// Write an enumerated JP2 colorspace.
    Enumerated(Colorspace),
    /// Write an ICC-profile Colour Specification box.
    IccProfile(&'a [u8]),
}

impl<'a> J2kFileColorSpec<'a> {
    /// Return a directly representable JP2/JPH colour specification borrowed
    /// from inspected file metadata.
    #[must_use]
    pub fn from_inspected(color_spec: &'a J2kColorSpec) -> Option<Self> {
        file_color_spec_from_inspected_colr(color_spec)
    }

    /// Return the first directly representable JP2/JPH colour specification
    /// from inspected file metadata.
    #[must_use]
    pub fn from_file_metadata(metadata: &'a J2kFileMetadata) -> Option<Self> {
        metadata.color_specs.iter().find_map(Self::from_inspected)
    }
}

/// Optional JP2/JPH metadata boxes to write in the JP2 Header box.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct J2kFileBoxMetadata<'a> {
    /// Palette box metadata to write as `pclr`.
    pub palette: Option<&'a J2kPaletteMetadata>,
    /// Component Mapping box entries to write as `cmap`.
    ///
    /// When [`Self::palette`] is present and this slice is empty, the writer
    /// emits standard palette mappings from codestream component 0 to each
    /// palette column.
    pub component_mappings: &'a [J2kComponentMapping],
    /// Channel Definition box entries to write as `cdef`.
    pub channel_definitions: &'a [J2kChannelDefinition],
}

impl J2kFileBoxMetadata<'_> {
    /// Empty JP2/JPH metadata-box selection.
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            palette: None,
            component_mappings: &[],
            channel_definitions: &[],
        }
    }
}

impl<'a> J2kFileBoxMetadata<'a> {
    /// Borrow directly representable JP2/JPH metadata boxes from inspected
    /// file metadata for rewrapping.
    #[must_use]
    pub fn from_file_metadata(metadata: &'a J2kFileMetadata) -> Self {
        Self {
            palette: metadata.palette.as_ref(),
            component_mappings: &metadata.component_mappings,
            channel_definitions: &metadata.channel_definitions,
        }
    }
}

/// Options for wrapping a raw JPEG 2000 / HTJ2K codestream as a JP2/JPH file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct J2kFileWrapOptions<'a> {
    /// Desired file wrapper.
    pub payload_kind: CompressedPayloadKind,
    /// Color metadata to place in the JP2 Header box.
    pub color: J2kFileColorSpec<'a>,
    /// Explicit Colour Specification boxes to place in the JP2 Header box.
    ///
    /// When non-empty, these are written in order and [`Self::color`] is used
    /// only for legacy single-COLR callers.
    pub color_specs: &'a [J2kFileColorSpec<'a>],
    /// Optional JP2/JPH file metadata boxes to place in the JP2 Header box.
    pub metadata: J2kFileBoxMetadata<'a>,
}

impl J2kFileWrapOptions<'_> {
    /// Create standard JP2 wrapper options.
    #[must_use]
    pub const fn jp2() -> Self {
        Self {
            payload_kind: CompressedPayloadKind::Jp2File,
            color: J2kFileColorSpec::Infer,
            color_specs: &[],
            metadata: J2kFileBoxMetadata::empty(),
        }
    }

    /// Create standard JPH wrapper options for an HTJ2K codestream.
    #[must_use]
    pub const fn jph() -> Self {
        Self {
            payload_kind: CompressedPayloadKind::JphFile,
            color: J2kFileColorSpec::Infer,
            color_specs: &[],
            metadata: J2kFileBoxMetadata::empty(),
        }
    }
}

impl<'a> J2kFileWrapOptions<'a> {
    /// Return options with explicit color metadata.
    #[must_use]
    pub const fn with_color(mut self, color: J2kFileColorSpec<'a>) -> Self {
        self.color = color;
        self.color_specs = &[];
        self
    }

    /// Return options with explicit ordered JP2/JPH Colour Specification boxes.
    #[must_use]
    pub const fn with_color_specs(mut self, color_specs: &'a [J2kFileColorSpec<'a>]) -> Self {
        self.color_specs = color_specs;
        self
    }

    /// Return options with explicit JP2/JPH metadata boxes.
    #[must_use]
    pub const fn with_metadata(mut self, metadata: J2kFileBoxMetadata<'a>) -> Self {
        self.metadata = metadata;
        self
    }
}

/// Wrap a raw JPEG 2000 / HTJ2K codestream in a JP2 or JPH file container.
///
/// The codestream is inspected with the existing parser before any boxes are
/// written. Component precision, signedness, and sampling metadata come from
/// the codestream SIZ marker. Mixed precision or mixed signedness is emitted via
/// the JP2 `bpcc` box.
pub fn wrap_j2k_codestream(
    codestream: &[u8],
    options: J2kFileWrapOptions<'_>,
) -> Result<Vec<u8>, J2kError> {
    wrap_with_options(codestream, options, 0, 0)
}

fn wrap_with_options(
    codestream: &[u8],
    options: J2kFileWrapOptions<'_>,
    external_retained_bytes: usize,
    source_owned_bytes: usize,
) -> Result<Vec<u8>, J2kError> {
    let colors = color::ColorSelection::Options {
        legacy: options.color,
        explicit: options.color_specs,
    };
    wrap_with_selection(
        codestream,
        options.payload_kind,
        colors,
        options.metadata,
        external_retained_bytes,
        source_owned_bytes,
    )
}

pub(crate) fn wrap_recode_jph_codestream(
    codestream: &[u8],
    input_metadata: Option<&J2kFileMetadata>,
    preserve_file_metadata: bool,
    external_retained_bytes: usize,
    source_owned_bytes: usize,
) -> Result<Vec<u8>, J2kError> {
    let reusable_metadata = input_metadata.filter(|metadata| {
        preserve_file_metadata
            || (metadata.palette.is_none() && metadata.component_mappings.is_empty())
    });
    let colors = reusable_metadata.map_or(
        color::ColorSelection::Options {
            legacy: J2kFileColorSpec::Infer,
            explicit: &[],
        },
        |metadata| color::ColorSelection::Inspected(&metadata.color_specs),
    );
    let metadata = if preserve_file_metadata {
        input_metadata.map_or_else(J2kFileBoxMetadata::empty, |metadata| {
            J2kFileBoxMetadata::from_file_metadata(metadata)
        })
    } else {
        J2kFileBoxMetadata::empty()
    };
    wrap_with_selection(
        codestream,
        CompressedPayloadKind::JphFile,
        colors,
        metadata,
        external_retained_bytes,
        source_owned_bytes,
    )
}

fn wrap_with_selection<'a>(
    codestream: &'a [u8],
    payload_kind: CompressedPayloadKind,
    colors: color::ColorSelection<'a>,
    metadata: J2kFileBoxMetadata<'a>,
    external_retained_bytes: usize,
    source_owned_bytes: usize,
) -> Result<Vec<u8>, J2kError> {
    let parsed = parse_image_info(codestream)?;
    if parsed.payload_kind != CompressedPayloadKind::Jpeg2000Codestream {
        return Err(J2kError::Unsupported(Unsupported {
            what: "JP2/JPH wrapping expects a raw JPEG 2000 codestream",
        }));
    }
    validate_wrapper_kind(payload_kind, parsed.transfer_syntax)?;

    let brand = match payload_kind {
        CompressedPayloadKind::Jp2File => JP2_BRAND,
        CompressedPayloadKind::JphFile => JPH_BRAND,
        _ => {
            return Err(J2kError::Unsupported(Unsupported {
                what: "JP2/JPH wrapping requires Jp2File or JphFile output",
            }))
        }
    };

    let retained_bytes = allocation::checked_retained_bytes(
        external_retained_bytes,
        source_owned_bytes,
        "JP2/JPH retained source owners",
    )?;
    let retained_bytes = allocation::checked_retained_bytes(
        retained_bytes,
        parsed.allocated_bytes()?,
        "JP2/JPH retained inspection metadata",
    )?;
    let plan = plan::WrapPlan::build(codestream, brand, &parsed, colors, metadata)?;
    writer::write(&plan, retained_bytes)
}

fn validate_wrapper_kind(
    payload_kind: CompressedPayloadKind,
    transfer_syntax: CompressedTransferSyntax,
) -> Result<(), J2kError> {
    let htj2k = matches!(
        transfer_syntax,
        CompressedTransferSyntax::HtJpeg2000Lossless | CompressedTransferSyntax::HtJpeg2000Lossy
    );
    match (payload_kind, htj2k) {
        (CompressedPayloadKind::Jp2File, false) | (CompressedPayloadKind::JphFile, true) => Ok(()),
        (CompressedPayloadKind::Jp2File, true) => Err(J2kError::Unsupported(Unsupported {
            what: "HTJ2K codestreams should be wrapped as JPH files",
        })),
        (CompressedPayloadKind::JphFile, false) => Err(J2kError::Unsupported(Unsupported {
            what: "JPH wrapping requires an HTJ2K codestream",
        })),
        _ => Err(J2kError::Unsupported(Unsupported {
            what: "JP2/JPH wrapping requires Jp2File or JphFile output",
        })),
    }
}

fn file_color_spec_from_inspected_colr(color_spec: &J2kColorSpec) -> Option<J2kFileColorSpec<'_>> {
    match color_spec {
        J2kColorSpec::Enumerated { value } => {
            inspected_enumerated_colorspace(*value).map(J2kFileColorSpec::Enumerated)
        }
        J2kColorSpec::IccProfile { profile } => Some(J2kFileColorSpec::IccProfile(profile)),
        J2kColorSpec::Unknown { .. } => None,
    }
}

fn inspected_enumerated_colorspace(value: u32) -> Option<Colorspace> {
    match value {
        16 => Some(Colorspace::SRgb),
        17 => Some(Colorspace::SGray),
        18 => Some(Colorspace::YCbCr),
        _ => None,
    }
}
