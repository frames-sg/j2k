// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_core::TileLayout;

use super::super::{validate_component_metadata, validate_ihdr_matches_codestream, Jp2ImageHeader};
use crate::parse::ParsedSiz;
use crate::{
    J2kComponentInfo, J2kComponentMapping, J2kComponentMappingType, J2kError, J2kFileMetadata,
    J2kPaletteColumn, J2kPaletteMetadata,
};

fn component(bit_depth: u8, signed: bool) -> J2kComponentInfo {
    J2kComponentInfo {
        bit_depth,
        signed,
        x_rsiz: 1,
        y_rsiz: 1,
    }
}

fn siz(components: Vec<J2kComponentInfo>) -> ParsedSiz {
    ParsedSiz {
        dimensions: (4, 3),
        components: u16::try_from(components.len()).unwrap(),
        bit_depth: components
            .iter()
            .map(|entry| entry.bit_depth)
            .max()
            .unwrap_or(0),
        tile_layout: TileLayout {
            tile_width: 4,
            tile_height: 3,
            tiles_x: 1,
            tiles_y: 1,
        },
        component_info: components,
    }
}

fn metadata() -> J2kFileMetadata {
    J2kFileMetadata {
        bits_per_component: Vec::new(),
        color_specs: Vec::new(),
        palette: None,
        component_mappings: Vec::new(),
        channel_definitions: Vec::new(),
        has_palette: false,
        has_component_mapping: false,
        has_channel_definition: false,
    }
}

fn header(components: u16, bits_per_component: Option<J2kComponentInfo>) -> Jp2ImageHeader {
    Jp2ImageHeader {
        offset: 17,
        width: 4,
        height: 3,
        components,
        bits_per_component,
    }
}

#[test]
fn image_header_validation_rejects_only_dimension_mismatches() {
    let siz = siz(vec![component(8, false)]);
    assert!(validate_ihdr_matches_codestream(header(1, Some(component(8, false))), &siz).is_ok());

    let mismatched = Jp2ImageHeader {
        width: 5,
        ..header(1, Some(component(8, false)))
    };
    assert!(matches!(
        validate_ihdr_matches_codestream(mismatched, &siz),
        Err(J2kError::InvalidBox {
            offset: 17,
            what: "ihdr dimensions must match codestream image dimensions"
        })
    ));
}

#[test]
fn palette_channels_do_not_replace_ihdr_codestream_component_metadata() {
    let siz = siz(vec![component(8, false)]);
    let mut metadata = metadata();
    metadata.palette = Some(J2kPaletteMetadata {
        columns: vec![
            J2kPaletteColumn {
                bit_depth: 6,
                signed: false,
            },
            J2kPaletteColumn {
                bit_depth: 7,
                signed: false,
            },
            J2kPaletteColumn {
                bit_depth: 8,
                signed: false,
            },
        ],
        entries: Vec::new(),
    });
    metadata.component_mappings = (0..3)
        .map(|column| J2kComponentMapping {
            component_index: 0,
            mapping_type: J2kComponentMappingType::Palette { column },
        })
        .collect();

    assert!(
        validate_component_metadata(header(1, Some(component(8, false))), &metadata, &siz,).is_ok()
    );
    assert!(matches!(
        validate_component_metadata(header(3, Some(component(8, false))), &metadata, &siz),
        Err(J2kError::InvalidBox {
            what: "ihdr component count must match codestream components",
            ..
        })
    ));
}

#[test]
fn explicit_ihdr_precision_must_match_every_codestream_component_and_forbids_bpcc() {
    let uniform = siz(vec![component(8, false), component(8, false)]);
    let empty = metadata();
    assert!(
        validate_component_metadata(header(2, Some(component(8, false))), &empty, &uniform).is_ok()
    );

    let mixed = siz(vec![component(8, false), component(12, false)]);
    assert!(matches!(
        validate_component_metadata(header(2, Some(component(8, false))), &empty, &mixed),
        Err(J2kError::InvalidBox {
            what: "ihdr bpc must match codestream component precision",
            ..
        })
    ));

    let mut with_bpcc = metadata();
    with_bpcc.bits_per_component = vec![component(8, false), component(8, false)];
    assert!(matches!(
        validate_component_metadata(header(2, Some(component(8, false))), &with_bpcc, &uniform),
        Err(J2kError::InvalidBox {
            what: "bpcc must not be present when ihdr bpc is explicit",
            ..
        })
    ));
}

#[test]
fn variable_precision_requires_complete_matching_bpcc_metadata() {
    let siz = siz(vec![component(8, false), component(12, true)]);
    let mut metadata = metadata();
    metadata.bits_per_component = vec![component(8, false), component(12, true)];
    assert!(validate_component_metadata(header(2, None), &metadata, &siz).is_ok());

    metadata.bits_per_component.pop();
    assert!(matches!(
        validate_component_metadata(header(2, None), &metadata, &siz),
        Err(J2kError::InvalidBox {
            what: "bpcc component count must match ihdr component count",
            ..
        })
    ));

    metadata.bits_per_component = vec![component(8, false), component(11, true)];
    assert!(matches!(
        validate_component_metadata(header(2, None), &metadata, &siz),
        Err(J2kError::InvalidBox {
            what: "bpcc entries must match codestream component precision",
            ..
        })
    ));

    assert!(matches!(
        validate_component_metadata(header(3, None), &metadata, &siz),
        Err(J2kError::InvalidBox {
            what: "ihdr component count must match codestream components",
            ..
        })
    ));
}
