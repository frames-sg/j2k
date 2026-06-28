// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{codestream::parse_codestream, ParsedImageInfo};
use crate::{
    J2kChannelAssociation, J2kChannelDefinition, J2kChannelType, J2kColorSpec, J2kComponentInfo,
    J2kComponentMapping, J2kComponentMappingType, J2kError, J2kFileMetadata, J2kPaletteColumn,
    J2kPaletteMetadata,
};
use j2k_core::{Colorspace, CompressedPayloadKind, InputError};

const JP2_SIGNATURE: [u8; 12] = [0, 0, 0, 12, b'j', b'P', b' ', b' ', 0x0D, 0x0A, 0x87, 0x0A];
const JP2_SIGNATURE_PREFIX: [u8; 8] = [0, 0, 0, 12, b'j', b'P', b' ', b' '];

pub(crate) fn looks_like_jp2(input: &[u8]) -> bool {
    input.starts_with(&JP2_SIGNATURE_PREFIX)
}

pub(crate) fn parse_jp2(input: &[u8]) -> Result<ParsedImageInfo, J2kError> {
    if input.len() < JP2_SIGNATURE.len() {
        return Err(InputError::TooShort {
            need: JP2_SIGNATURE.len(),
            have: input.len(),
        }
        .into());
    }

    let mut offset = 0usize;
    let mut saw_signature = false;
    let mut saw_ftyp = false;
    let mut saw_jp2h = false;
    let mut saw_ihdr = false;
    let mut image_header = None;
    let mut file_metadata = None;
    let mut codestream = None;
    let mut payload_kind = CompressedPayloadKind::Jp2File;

    while offset < input.len() {
        let header = read_box_header(input, offset)?;
        if header.end > input.len() {
            return Err(InputError::TruncatedAt {
                offset,
                segment: "box payload",
            }
            .into());
        }
        let payload = &input[header.payload_start..header.end];
        match &header.box_type {
            b"jP  " => {
                if saw_signature || offset != 0 {
                    return Err(J2kError::InvalidBox {
                        offset,
                        what: "signature box must appear exactly once at the start of the file",
                    });
                }
                if payload != &JP2_SIGNATURE[8..] {
                    return Err(J2kError::InvalidBox {
                        offset,
                        what: "invalid JP2 signature payload",
                    });
                }
                saw_signature = true;
            }
            b"ftyp" => {
                if !saw_signature || saw_ftyp || saw_jp2h || codestream.is_some() {
                    return Err(J2kError::InvalidBox {
                        offset,
                        what: "file type box must appear exactly once before jp2h and jp2c",
                    });
                }
                if payload.len() < 8 {
                    return Err(J2kError::InvalidBox {
                        offset,
                        what: "ftyp payload shorter than 8 bytes",
                    });
                }
                payload_kind = classify_file_type(payload);
                saw_ftyp = true;
            }
            b"jp2h" => {
                if !saw_ftyp || saw_jp2h || codestream.is_some() {
                    return Err(J2kError::InvalidBox {
                        offset,
                        what: "jp2h must appear exactly once after ftyp and before jp2c",
                    });
                }
                let (ihdr, metadata) = parse_jp2h(payload, header.payload_start)?;
                saw_jp2h = true;
                saw_ihdr = ihdr.is_some();
                image_header = ihdr;
                file_metadata = Some(metadata);
            }
            b"jp2c" => {
                if !saw_jp2h || codestream.is_some() {
                    return Err(J2kError::InvalidBox {
                        offset,
                        what: "jp2c must appear exactly once after jp2h",
                    });
                }
                codestream = Some(payload);
            }
            _ => {}
        }
        offset = header.end;
    }

    if !saw_signature {
        return Err(J2kError::MissingRequiredBox { box_type: "jP  " });
    }
    if !saw_ftyp {
        return Err(J2kError::MissingRequiredBox { box_type: "ftyp" });
    }
    if !saw_jp2h {
        return Err(J2kError::MissingRequiredBox { box_type: "jp2h" });
    }
    if !saw_ihdr {
        return Err(J2kError::MissingRequiredBox { box_type: "ihdr" });
    }
    let codestream = codestream.ok_or(J2kError::MissingRequiredBox { box_type: "jp2c" })?;
    let parsed = parse_codestream(codestream)?;
    if let Some(ihdr) = image_header {
        validate_ihdr_matches_codestream(ihdr, &parsed.siz)?;
        validate_component_metadata(ihdr, file_metadata.as_ref(), &parsed.siz)?;
    }
    if payload_kind == CompressedPayloadKind::JphFile && !parsed.cod.high_throughput {
        return Err(J2kError::InvalidBox {
            offset: 0,
            what: "JPH file type requires an HTJ2K codestream",
        });
    }
    if payload_kind == CompressedPayloadKind::Jp2File && parsed.cod.high_throughput {
        return Err(J2kError::InvalidBox {
            offset: 0,
            what: "JP2 file type must not wrap an HTJ2K codestream; use JPH",
        });
    }
    let colorspace = file_metadata
        .as_ref()
        .and_then(primary_colorspace_from_file_metadata);
    let info = parsed.clone().into_info(colorspace);
    let components = parsed.siz.component_info.clone();
    Ok(ParsedImageInfo {
        info,
        transfer_syntax: parsed.transfer_syntax(),
        payload_kind,
        components,
        file_metadata,
    })
}

fn classify_file_type(payload: &[u8]) -> CompressedPayloadKind {
    if payload.len() >= 4 && &payload[..4] == b"jph " {
        return CompressedPayloadKind::JphFile;
    }
    if payload[8..]
        .chunks_exact(4)
        .any(|compatible_brand| compatible_brand == b"jph ")
    {
        return CompressedPayloadKind::JphFile;
    }
    CompressedPayloadKind::Jp2File
}

#[derive(Debug, Clone, Copy)]
struct Jp2ImageHeader {
    offset: usize,
    width: u32,
    height: u32,
    components: u16,
    bits_per_component: Option<J2kComponentInfo>,
}

fn parse_jp2h(
    payload: &[u8],
    base_offset: usize,
) -> Result<(Option<Jp2ImageHeader>, J2kFileMetadata), J2kError> {
    let mut offset = 0usize;
    let mut image_header = None;
    let mut metadata = J2kFileMetadata {
        bits_per_component: Vec::new(),
        color_specs: Vec::new(),
        palette: None,
        component_mappings: Vec::new(),
        channel_definitions: Vec::new(),
        has_palette: false,
        has_component_mapping: false,
        has_channel_definition: false,
    };

    while offset < payload.len() {
        let header = read_box_header(payload, offset)?;
        if header.end > payload.len() {
            return Err(InputError::TruncatedAt {
                offset: base_offset + offset,
                segment: "box payload",
            }
            .into());
        }
        let inner = &payload[header.payload_start..header.end];
        match &header.box_type {
            b"ihdr" => {
                if image_header.is_some() {
                    return Err(J2kError::InvalidBox {
                        offset: base_offset + offset,
                        what: "ihdr must appear exactly once",
                    });
                }
                image_header = Some(parse_ihdr(inner, base_offset + offset)?);
            }
            b"colr" => {
                metadata.color_specs.push(parse_colr(inner));
            }
            b"bpcc" => {
                metadata.bits_per_component = parse_bpcc(inner, base_offset + offset)?;
            }
            b"pclr" => {
                metadata.has_palette = true;
                metadata.palette = Some(parse_pclr(inner, base_offset + offset)?);
            }
            b"cmap" => {
                metadata.has_component_mapping = true;
                metadata.component_mappings = parse_cmap(inner, base_offset + offset)?;
            }
            b"cdef" => {
                metadata.has_channel_definition = true;
                metadata.channel_definitions = parse_cdef(inner, base_offset + offset)?;
            }
            _ => {}
        }
        offset = header.end;
    }

    if metadata.color_specs.is_empty() {
        return Err(J2kError::MissingRequiredBox { box_type: "colr" });
    }

    Ok((image_header, metadata))
}

fn parse_ihdr(payload: &[u8], offset: usize) -> Result<Jp2ImageHeader, J2kError> {
    if payload.len() < 14 {
        return Err(J2kError::InvalidBox {
            offset,
            what: "ihdr payload shorter than 14 bytes",
        });
    }
    if payload[11] != 7 {
        return Err(J2kError::InvalidBox {
            offset,
            what: "ihdr compression type must be JPEG 2000",
        });
    }
    Ok(Jp2ImageHeader {
        offset,
        height: u32::from_be_bytes([payload[0], payload[1], payload[2], payload[3]]),
        width: u32::from_be_bytes([payload[4], payload[5], payload[6], payload[7]]),
        components: u16::from_be_bytes([payload[8], payload[9]]),
        bits_per_component: if payload[10] == 0xff {
            None
        } else {
            Some(parse_component_descriptor(payload[10], offset, "ihdr bpc")?)
        },
    })
}

fn validate_ihdr_matches_codestream(
    ihdr: Jp2ImageHeader,
    siz: &super::ParsedSiz,
) -> Result<(), J2kError> {
    if (ihdr.width, ihdr.height) != siz.dimensions {
        return Err(J2kError::InvalidBox {
            offset: ihdr.offset,
            what: "ihdr dimensions must match codestream image dimensions",
        });
    }
    Ok(())
}

fn validate_component_metadata(
    ihdr: Jp2ImageHeader,
    metadata: Option<&J2kFileMetadata>,
    siz: &super::ParsedSiz,
) -> Result<(), J2kError> {
    let resolved = metadata.and_then(|metadata| resolved_image_component_metadata(metadata, siz));
    let resolved = resolved.as_deref().unwrap_or(&siz.component_info);
    if resolved.len() != ihdr.components as usize {
        return Err(J2kError::InvalidBox {
            offset: ihdr.offset,
            what: "ihdr component count must match resolved JP2 image components",
        });
    }

    if let Some(descriptor) = ihdr.bits_per_component {
        if metadata.is_some_and(|metadata| !metadata.bits_per_component.is_empty()) {
            return Err(J2kError::InvalidBox {
                offset: ihdr.offset,
                what: "bpcc must not be present when ihdr bpc is explicit",
            });
        }
        if !resolved
            .iter()
            .all(|component| same_component_precision(*component, descriptor))
        {
            return Err(J2kError::InvalidBox {
                offset: ihdr.offset,
                what: "ihdr bpc must match resolved JP2 image component precision",
            });
        }
    } else {
        let Some(metadata) = metadata else {
            return Err(J2kError::InvalidBox {
                offset: ihdr.offset,
                what: "ihdr bpc=255 requires bpcc",
            });
        };
        if metadata.bits_per_component.len() != ihdr.components as usize {
            return Err(J2kError::InvalidBox {
                offset: ihdr.offset,
                what: "bpcc component count must match ihdr component count",
            });
        }
        for (component, descriptor) in resolved.iter().zip(metadata.bits_per_component.iter()) {
            if !same_component_precision(*component, *descriptor) {
                return Err(J2kError::InvalidBox {
                    offset: ihdr.offset,
                    what: "bpcc entries must match resolved JP2 image component precision",
                });
            }
        }
    }

    Ok(())
}

fn resolved_image_component_metadata(
    metadata: &J2kFileMetadata,
    siz: &super::ParsedSiz,
) -> Option<Vec<J2kComponentInfo>> {
    if metadata.component_mappings.is_empty() {
        if let Some(palette) = metadata.palette.as_ref() {
            return Some(
                palette
                    .columns
                    .iter()
                    .copied()
                    .map(component_from_palette_column)
                    .collect(),
            );
        }
        return Some(siz.component_info.clone());
    }

    let mut resolved = Vec::with_capacity(metadata.component_mappings.len());
    for mapping in &metadata.component_mappings {
        match mapping.mapping_type {
            J2kComponentMappingType::Direct => {
                let component = siz.component_info.get(mapping.component_index as usize)?;
                resolved.push(*component);
            }
            J2kComponentMappingType::Palette { column } => {
                let palette = metadata.palette.as_ref()?;
                let column = palette.columns.get(column as usize)?;
                resolved.push(component_from_palette_column(*column));
            }
            J2kComponentMappingType::Unknown { .. } => return None,
        }
    }

    Some(resolved)
}

fn component_from_palette_column(column: J2kPaletteColumn) -> J2kComponentInfo {
    J2kComponentInfo {
        bit_depth: column.bit_depth,
        signed: column.signed,
        x_rsiz: 1,
        y_rsiz: 1,
    }
}

fn same_component_precision(left: J2kComponentInfo, right: J2kComponentInfo) -> bool {
    left.bit_depth == right.bit_depth && left.signed == right.signed
}

fn parse_colr(payload: &[u8]) -> J2kColorSpec {
    if payload.len() < 3 {
        return J2kColorSpec::Unknown { method: 0 };
    }
    match payload[0] {
        1 if payload.len() >= 7 => J2kColorSpec::Enumerated {
            value: u32::from_be_bytes([payload[3], payload[4], payload[5], payload[6]]),
        },
        2 => J2kColorSpec::IccProfile {
            profile: payload[3..].to_vec(),
        },
        method => J2kColorSpec::Unknown { method },
    }
}

fn primary_colorspace_from_file_metadata(metadata: &J2kFileMetadata) -> Option<Colorspace> {
    metadata
        .color_specs
        .first()
        .map(|color_spec| match color_spec {
            J2kColorSpec::Enumerated { value: 16 } => Colorspace::SRgb,
            J2kColorSpec::Enumerated { value: 17 } => Colorspace::SGray,
            J2kColorSpec::Enumerated { value: 18 } => Colorspace::YCbCr,
            J2kColorSpec::Enumerated { .. } | J2kColorSpec::IccProfile { .. } => {
                Colorspace::IccTagged
            }
            J2kColorSpec::Unknown { .. } => Colorspace::IccTagged,
        })
}

fn parse_bpcc(payload: &[u8], offset: usize) -> Result<Vec<J2kComponentInfo>, J2kError> {
    if payload.is_empty() {
        return Err(J2kError::InvalidBox {
            offset,
            what: "bpcc payload must not be empty",
        });
    }
    payload
        .iter()
        .map(|descriptor| parse_component_descriptor(*descriptor, offset, "bpcc component bpc"))
        .collect()
}

fn parse_component_descriptor(
    descriptor: u8,
    offset: usize,
    what: &'static str,
) -> Result<J2kComponentInfo, J2kError> {
    let bit_depth = (descriptor & 0x7f) + 1;
    if bit_depth > 38 {
        return Err(J2kError::InvalidBox { offset, what });
    }
    Ok(J2kComponentInfo {
        bit_depth,
        signed: (descriptor & 0x80) != 0,
        x_rsiz: 1,
        y_rsiz: 1,
    })
}

fn parse_pclr(payload: &[u8], offset: usize) -> Result<J2kPaletteMetadata, J2kError> {
    if payload.len() < 3 {
        return Err(J2kError::InvalidBox {
            offset,
            what: "pclr payload shorter than header",
        });
    }
    let entry_count = u16::from_be_bytes([payload[0], payload[1]]) as usize;
    let column_count = usize::from(payload[2]);
    if entry_count == 0 || column_count == 0 {
        return Err(J2kError::InvalidBox {
            offset,
            what: "pclr entry and column counts must be non-zero",
        });
    }
    let descriptors_end = 3usize
        .checked_add(column_count)
        .ok_or(J2kError::InvalidBox {
            offset,
            what: "pclr column descriptor length overflow",
        })?;
    if descriptors_end > payload.len() {
        return Err(J2kError::InvalidBox {
            offset,
            what: "pclr column descriptors are truncated",
        });
    }

    let columns = payload[3..descriptors_end]
        .iter()
        .map(|descriptor| J2kPaletteColumn {
            bit_depth: (descriptor & 0x7f) + 1,
            signed: (descriptor & 0x80) != 0,
        })
        .collect::<Vec<_>>();

    let mut cursor = descriptors_end;
    let mut entries = Vec::with_capacity(entry_count);
    for _ in 0..entry_count {
        let mut row = Vec::with_capacity(column_count);
        for column in &columns {
            let byte_count = usize::from(column.bit_depth).div_ceil(8).max(1);
            let end = cursor.checked_add(byte_count).ok_or(J2kError::InvalidBox {
                offset,
                what: "pclr entry length overflow",
            })?;
            let bytes = payload.get(cursor..end).ok_or(J2kError::InvalidBox {
                offset,
                what: "pclr entries are truncated",
            })?;
            let mut raw = 0u64;
            for &byte in bytes {
                raw = (raw << 8) | u64::from(byte);
            }
            row.push(raw);
            cursor = end;
        }
        entries.push(row);
    }

    Ok(J2kPaletteMetadata { columns, entries })
}

fn parse_cmap(payload: &[u8], offset: usize) -> Result<Vec<J2kComponentMapping>, J2kError> {
    if payload.is_empty() || !payload.len().is_multiple_of(4) {
        return Err(J2kError::InvalidBox {
            offset,
            what: "cmap payload length must be a non-zero multiple of 4",
        });
    }
    Ok(payload
        .chunks_exact(4)
        .map(|entry| {
            let component_index = u16::from_be_bytes([entry[0], entry[1]]);
            let mapping_type = match entry[2] {
                0 => J2kComponentMappingType::Direct,
                1 => J2kComponentMappingType::Palette { column: entry[3] },
                value => J2kComponentMappingType::Unknown {
                    value,
                    column: entry[3],
                },
            };
            J2kComponentMapping {
                component_index,
                mapping_type,
            }
        })
        .collect())
}

fn parse_cdef(payload: &[u8], offset: usize) -> Result<Vec<J2kChannelDefinition>, J2kError> {
    if payload.len() < 2 {
        return Err(J2kError::InvalidBox {
            offset,
            what: "cdef payload shorter than entry count",
        });
    }
    let entry_count = u16::from_be_bytes([payload[0], payload[1]]) as usize;
    if entry_count == 0 {
        return Err(J2kError::InvalidBox {
            offset,
            what: "cdef entry count must be non-zero",
        });
    }
    let required_len = 2usize
        .checked_add(entry_count.checked_mul(6).ok_or(J2kError::InvalidBox {
            offset,
            what: "cdef entry length overflow",
        })?)
        .ok_or(J2kError::InvalidBox {
            offset,
            what: "cdef payload length overflow",
        })?;
    if required_len > payload.len() {
        return Err(J2kError::InvalidBox {
            offset,
            what: "cdef entries are truncated",
        });
    }

    Ok(payload[2..required_len]
        .chunks_exact(6)
        .map(|entry| {
            let channel_index = u16::from_be_bytes([entry[0], entry[1]]);
            let raw_type = u16::from_be_bytes([entry[2], entry[3]]);
            let raw_association = u16::from_be_bytes([entry[4], entry[5]]);
            J2kChannelDefinition {
                channel_index,
                channel_type: match raw_type {
                    0 => J2kChannelType::Color,
                    1 => J2kChannelType::Opacity,
                    2 => J2kChannelType::PremultipliedOpacity,
                    u16::MAX => J2kChannelType::Unspecified,
                    value => J2kChannelType::Unknown { value },
                },
                association: match raw_association {
                    0 => J2kChannelAssociation::WholeImage,
                    u16::MAX => J2kChannelAssociation::Unspecified,
                    index => J2kChannelAssociation::Color { index },
                },
            }
        })
        .collect())
}

#[derive(Debug, Clone, Copy)]
struct BoxHeader {
    box_type: [u8; 4],
    payload_start: usize,
    end: usize,
}

fn read_box_header(input: &[u8], offset: usize) -> Result<BoxHeader, J2kError> {
    if offset + 8 > input.len() {
        return Err(InputError::TruncatedAt {
            offset,
            segment: "box header",
        }
        .into());
    }
    let lbox = u32::from_be_bytes([
        input[offset],
        input[offset + 1],
        input[offset + 2],
        input[offset + 3],
    ]);
    let box_type = [
        input[offset + 4],
        input[offset + 5],
        input[offset + 6],
        input[offset + 7],
    ];

    let (payload_start, end) = match lbox {
        0 => (offset + 8, input.len()),
        1 => {
            if offset + 16 > input.len() {
                return Err(InputError::TruncatedAt {
                    offset,
                    segment: "extended box header",
                }
                .into());
            }
            let xlbox = u64::from_be_bytes([
                input[offset + 8],
                input[offset + 9],
                input[offset + 10],
                input[offset + 11],
                input[offset + 12],
                input[offset + 13],
                input[offset + 14],
                input[offset + 15],
            ]);
            if xlbox < 16 {
                return Err(J2kError::InvalidBox {
                    offset,
                    what: "extended box length smaller than header",
                });
            }
            let end = offset
                .checked_add(xlbox as usize)
                .ok_or(J2kError::InvalidBox {
                    offset,
                    what: "extended box length overflow",
                })?;
            (offset + 16, end)
        }
        length if length < 8 => {
            return Err(J2kError::InvalidBox {
                offset,
                what: "box length smaller than header",
            })
        }
        length => {
            let end = offset
                .checked_add(length as usize)
                .ok_or(J2kError::InvalidBox {
                    offset,
                    what: "box length overflow",
                })?;
            (offset + 8, end)
        }
    };

    Ok(BoxHeader {
        box_type,
        payload_start,
        end,
    })
}
