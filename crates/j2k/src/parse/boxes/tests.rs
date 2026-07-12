// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_core::{Colorspace, CompressedPayloadKind, InputError};
use j2k_native::{
    DecodeError as NativeDecodeError, FormatError as NativeFormatError,
    Jp2ChannelAssociation as NativeChannelAssociation,
    Jp2ChannelDefinition as NativeChannelDefinition, Jp2ChannelType as NativeChannelType,
    Jp2ColorSpec as NativeColorSpec, Jp2ComponentMapping as NativeComponentMapping,
    Jp2ComponentMappingType as NativeComponentMappingType,
    Jp2ComponentMetadata as NativeComponentMetadata, Jp2FileKind, Jp2FileMetadata,
    Jp2ImageHeaderMetadata, Jp2PaletteColumn as NativePaletteColumn,
    Jp2PaletteMetadata as NativePaletteMetadata,
};

use super::{
    file_metadata_from_native, image_header_from_native, map_native_jp2_error,
    metadata_allocated_bytes, payload_kind_from_native, primary_colorspace_from_file_metadata,
};
use crate::{
    J2kChannelAssociation, J2kChannelType, J2kColorSpec, J2kComponentMappingType, J2kError,
};

fn comprehensive_native_metadata() -> Jp2FileMetadata {
    Jp2FileMetadata {
        bits_per_component: vec![
            NativeComponentMetadata {
                bit_depth: 8,
                signed: false,
            },
            NativeComponentMetadata {
                bit_depth: 12,
                signed: true,
            },
        ],
        color_specs: vec![
            NativeColorSpec::Enumerated { value: 16 },
            NativeColorSpec::IccProfile {
                profile: vec![1, 2, 3],
            },
            NativeColorSpec::Unknown { method: 7 },
        ],
        palette: Some(NativePaletteMetadata {
            columns: vec![
                NativePaletteColumn {
                    bit_depth: 8,
                    signed: false,
                },
                NativePaletteColumn {
                    bit_depth: 10,
                    signed: true,
                },
            ],
            entries: vec![vec![1, 2], vec![3, 4]],
        }),
        component_mappings: vec![
            NativeComponentMapping {
                component_index: 0,
                mapping_type: NativeComponentMappingType::Direct,
            },
            NativeComponentMapping {
                component_index: 1,
                mapping_type: NativeComponentMappingType::Palette { column: 1 },
            },
            NativeComponentMapping {
                component_index: 2,
                mapping_type: NativeComponentMappingType::Unknown {
                    value: 9,
                    column: 4,
                },
            },
        ],
        channel_definitions: vec![
            NativeChannelDefinition {
                channel_index: 0,
                channel_type: NativeChannelType::Color,
                association: NativeChannelAssociation::WholeImage,
            },
            NativeChannelDefinition {
                channel_index: 1,
                channel_type: NativeChannelType::Opacity,
                association: NativeChannelAssociation::Color { index: 2 },
            },
            NativeChannelDefinition {
                channel_index: 2,
                channel_type: NativeChannelType::PremultipliedOpacity,
                association: NativeChannelAssociation::Unspecified,
            },
            NativeChannelDefinition {
                channel_index: 3,
                channel_type: NativeChannelType::Unspecified,
                association: NativeChannelAssociation::WholeImage,
            },
            NativeChannelDefinition {
                channel_index: 4,
                channel_type: NativeChannelType::Unknown { value: 99 },
                association: NativeChannelAssociation::WholeImage,
            },
        ],
        has_palette: true,
        has_component_mapping: true,
        has_channel_definition: true,
    }
}

#[test]
fn native_metadata_conversion_preserves_variants_and_capacity_accounting() {
    let native = comprehensive_native_metadata();

    let (metadata, reported_bytes) =
        file_metadata_from_native(native).expect("bounded metadata conversion succeeds");

    assert_eq!(reported_bytes, metadata_allocated_bytes(&metadata).unwrap());
    assert_eq!(metadata.bits_per_component[0].bit_depth, 8);
    assert!(metadata.bits_per_component[1].signed);
    assert!(matches!(
        metadata.color_specs.as_slice(),
        [
            J2kColorSpec::Enumerated { value: 16 },
            J2kColorSpec::IccProfile { profile },
            J2kColorSpec::Unknown { method: 7 }
        ] if profile == &[1, 2, 3]
    ));
    let palette = metadata.palette.as_ref().expect("palette retained");
    assert_eq!(palette.columns[1].bit_depth, 10);
    assert!(palette.columns[1].signed);
    assert_eq!(palette.entries, [vec![1, 2], vec![3, 4]]);
    assert!(matches!(
        metadata.component_mappings[0].mapping_type,
        J2kComponentMappingType::Direct
    ));
    assert!(matches!(
        metadata.component_mappings[1].mapping_type,
        J2kComponentMappingType::Palette { column: 1 }
    ));
    assert!(matches!(
        metadata.component_mappings[2].mapping_type,
        J2kComponentMappingType::Unknown {
            value: 9,
            column: 4
        }
    ));
    assert_eq!(
        metadata.channel_definitions[0].channel_type,
        J2kChannelType::Color
    );
    assert_eq!(
        metadata.channel_definitions[1].association,
        J2kChannelAssociation::Color { index: 2 }
    );
    assert_eq!(
        metadata.channel_definitions[2].channel_type,
        J2kChannelType::PremultipliedOpacity
    );
    assert_eq!(
        metadata.channel_definitions[3].channel_type,
        J2kChannelType::Unspecified
    );
    assert_eq!(
        metadata.channel_definitions[4].channel_type,
        J2kChannelType::Unknown { value: 99 }
    );
    assert!(metadata.has_palette);
    assert!(metadata.has_component_mapping);
    assert!(metadata.has_channel_definition);
}

#[test]
fn wrapper_conversions_preserve_file_kind_header_and_primary_colorspace() {
    assert_eq!(
        payload_kind_from_native(Jp2FileKind::Jp2),
        CompressedPayloadKind::Jp2File
    );
    assert_eq!(
        payload_kind_from_native(Jp2FileKind::Jph),
        CompressedPayloadKind::JphFile
    );

    let header = image_header_from_native(Jp2ImageHeaderMetadata {
        width: 320,
        height: 240,
        components: 3,
        bits_per_component: Some(NativeComponentMetadata {
            bit_depth: 12,
            signed: true,
        }),
    });
    assert_eq!(
        (header.width, header.height, header.components),
        (320, 240, 3)
    );
    let component = header.bits_per_component.expect("explicit precision");
    assert_eq!((component.bit_depth, component.signed), (12, true));

    for (color_spec, expected) in [
        (J2kColorSpec::Enumerated { value: 16 }, Colorspace::SRgb),
        (J2kColorSpec::Enumerated { value: 17 }, Colorspace::SGray),
        (J2kColorSpec::Enumerated { value: 18 }, Colorspace::YCbCr),
        (
            J2kColorSpec::Enumerated { value: 999 },
            Colorspace::IccTagged,
        ),
        (
            J2kColorSpec::IccProfile { profile: vec![1] },
            Colorspace::IccTagged,
        ),
        (J2kColorSpec::Unknown { method: 9 }, Colorspace::IccTagged),
    ] {
        let (metadata, _) = file_metadata_from_native(Jp2FileMetadata {
            bits_per_component: Vec::new(),
            color_specs: vec![match color_spec {
                J2kColorSpec::Enumerated { value } => NativeColorSpec::Enumerated { value },
                J2kColorSpec::IccProfile { profile } => NativeColorSpec::IccProfile { profile },
                J2kColorSpec::Unknown { method } => NativeColorSpec::Unknown { method },
            }],
            palette: None,
            component_mappings: Vec::new(),
            channel_definitions: Vec::new(),
            has_palette: false,
            has_component_mapping: false,
            has_channel_definition: false,
        })
        .expect("color metadata conversion");
        assert_eq!(
            primary_colorspace_from_file_metadata(&metadata),
            Some(expected)
        );
    }
}

#[test]
fn native_container_errors_keep_specific_facade_categories() {
    assert!(matches!(
        map_native_jp2_error(NativeDecodeError::Format(
            NativeFormatError::InvalidSignature
        )),
        J2kError::InvalidBox { offset: 0, .. }
    ));
    assert!(matches!(
        map_native_jp2_error(NativeDecodeError::Format(NativeFormatError::TooShort {
            need: 12,
            have: 8
        })),
        J2kError::Input(InputError::TooShort { need: 12, have: 8 })
    ));
    assert!(matches!(
        map_native_jp2_error(NativeDecodeError::Format(
            NativeFormatError::MissingRequiredBox("jp2h")
        )),
        J2kError::MissingRequiredBox { box_type: "jp2h" }
    ));
    assert!(matches!(
        map_native_jp2_error(NativeDecodeError::Format(
            NativeFormatError::MissingCodestream
        )),
        J2kError::MissingRequiredBox { box_type: "jp2c" }
    ));
    assert!(matches!(
        map_native_jp2_error(NativeDecodeError::AllocationTooLarge {
            what: "metadata",
            requested: 9,
            cap: 8
        }),
        J2kError::NativeDecode { .. }
    ));
}
