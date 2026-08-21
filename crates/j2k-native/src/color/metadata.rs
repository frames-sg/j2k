// SPDX-License-Identifier: MIT OR Apache-2.0

//! Container color metadata and channel-order resolution.

use alloc::vec::Vec;

use super::icc::{resolve_icc_color_space, try_clone_color_profile};
use super::ColorSpace;
use crate::error::bail;
use crate::image::DecodeOwnerBudget;
use crate::j2c::{ComponentData, Header};
use crate::jp2::cdef::{ChannelAssociation, ChannelDefinitionBox, ChannelType};
use crate::jp2::colr::EnumeratedColorspace;
use crate::jp2::{self, ImageBoxes};
use crate::{
    try_reserve_decode_elements, try_resize_decode_elements, DecodeSettings, FormatError, Result,
    ValidationError, DEFAULT_MAX_DECODE_BYTES,
};

const BITS_PER_ASSOCIATION_WORD: usize = 64;

pub(crate) fn resolve_alpha_and_color_space(
    boxes: &ImageBoxes,
    header: &Header<'_>,
    settings: &DecodeSettings,
    retained_baseline_bytes: usize,
) -> Result<(ColorSpace, bool, bool)> {
    let mut num_components = header.component_infos.len();
    if settings.resolve_palette_indices {
        if let Some(palette_box) = &boxes.palette {
            num_components = palette_box.columns.len();
        }
    }

    let mut has_alpha = boxes.channel_definition.as_ref().is_some_and(|cdef| {
        cdef.channel_definitions.iter().any(|definition| {
            matches!(
                definition.channel_type,
                ChannelType::Opacity | ChannelType::PremultipliedOpacity
            )
        })
    });

    let mut color_space = if !settings.resolve_palette_indices && boxes.palette.is_some() {
        has_alpha = false;
        ColorSpace::Gray
    } else {
        let retained_container_bytes =
            crate::image::retained_container_metadata_bytes(header, boxes)?
                .checked_add(retained_baseline_bytes)
                .ok_or(ValidationError::ImageTooLarge)?;
        if retained_container_bytes > DEFAULT_MAX_DECODE_BYTES {
            return Err(ValidationError::ImageTooLarge.into());
        }
        get_color_space(boxes, num_components, retained_container_bytes)?
    };

    let actual_num_components = header.component_infos.len();
    let mut used_lenient_metadata_recovery = false;
    if boxes.palette.is_none()
        && actual_num_components != usize::from(color_space.num_channels() + u16::from(has_alpha))
    {
        if !settings.strict
            && actual_num_components == usize::from(color_space.num_channels()) + 1
            && !has_alpha
        {
            has_alpha = true;
            used_lenient_metadata_recovery = true;
        } else if actual_num_components == 1 || (actual_num_components == 2 && has_alpha) {
            color_space = ColorSpace::Gray;
        } else if actual_num_components == 3 {
            color_space = ColorSpace::RGB;
        } else if actual_num_components == 4 {
            color_space = if has_alpha {
                ColorSpace::RGB
            } else {
                ColorSpace::CMYK
            };
        } else {
            color_space = ColorSpace::Unknown {
                num_channels: u16::try_from(actual_num_components)
                    .map_err(|_| ValidationError::TooManyChannels)?,
            };
        }
    }

    Ok((color_space, has_alpha, used_lenient_metadata_recovery))
}

fn get_color_space(
    boxes: &ImageBoxes,
    num_components: usize,
    retained_container_bytes: usize,
) -> Result<ColorSpace> {
    match boxes
        .primary_color_specification()
        .map_or(&jp2::colr::ColorSpace::Unknown, |specification| {
            &specification.color_space
        }) {
        jp2::colr::ColorSpace::Enumerated(enumerated) => match enumerated {
            EnumeratedColorspace::Cmyk => Ok(ColorSpace::CMYK),
            EnumeratedColorspace::Srgb
            | EnumeratedColorspace::EsRgb
            | EnumeratedColorspace::Sycc => Ok(ColorSpace::RGB),
            EnumeratedColorspace::RommRgb => Ok(ColorSpace::Icc {
                profile: try_clone_color_profile(
                    include_bytes!("../../assets/ProPhoto-v2-micro.icc"),
                    retained_container_bytes,
                )?,
                num_channels: 3,
            }),
            EnumeratedColorspace::Greyscale => Ok(ColorSpace::Gray),
            EnumeratedColorspace::CieLab(_) => Ok(ColorSpace::Icc {
                profile: try_clone_color_profile(
                    include_bytes!("../../assets/LAB.icc"),
                    retained_container_bytes,
                )?,
                num_channels: 3,
            }),
            _ => Err(FormatError::Unsupported.into()),
        },
        jp2::colr::ColorSpace::Icc(profile) => {
            Ok(resolve_icc_color_space(profile, retained_container_bytes)?
                .unwrap_or(ColorSpace::RGB))
        }
        jp2::colr::ColorSpace::Unknown => Ok(match num_components {
            1 => ColorSpace::Gray,
            3 => ColorSpace::RGB,
            4 => ColorSpace::CMYK,
            _ => ColorSpace::Unknown {
                num_channels: u16::try_from(num_components).unwrap_or(u16::MAX),
            },
        }),
    }
}

pub(crate) fn validate_and_reorder_channels(
    cdef: &ChannelDefinitionBox,
    components: &mut Vec<ComponentData>,
    retained_image_bytes: usize,
) -> Result<()> {
    let component_count = components.len();
    if cdef.channel_definitions.len() != component_count {
        bail!(ValidationError::InvalidChannelDefinition);
    }

    let word_count = component_count.div_ceil(BITS_PER_ASSOCIATION_WORD);
    let mut validation_budget =
        DecodeOwnerBudget::for_components(retained_image_bytes, components, components.capacity())?;
    validation_budget.include_elements::<u64>(word_count)?;
    let mut seen = Vec::new();
    try_resize_decode_elements(&mut seen, word_count, 0_u64)?;
    validation_budget.include_capacity_overage::<u64>(word_count, seen.capacity())?;
    for definition in &cdef.channel_definitions {
        let index = usize::from(definition.channel_index);
        if index >= component_count {
            bail!(ValidationError::InvalidChannelDefinition);
        }
        let word = index / BITS_PER_ASSOCIATION_WORD;
        let mask = 1_u64 << (index % BITS_PER_ASSOCIATION_WORD);
        if seen[word] & mask != 0 {
            bail!(ValidationError::InvalidChannelDefinition);
        }
        seen[word] |= mask;
    }
    seen.fill(0);
    for definition in &cdef.channel_definitions {
        if let ChannelAssociation::Colour(association) = definition.association {
            let Some(index) = association.checked_sub(1).map(usize::from) else {
                bail!(ValidationError::InvalidChannelDefinition);
            };
            if index >= component_count {
                bail!(ValidationError::InvalidChannelDefinition);
            }
            let word = index / BITS_PER_ASSOCIATION_WORD;
            let mask = 1_u64 << (index % BITS_PER_ASSOCIATION_WORD);
            if seen[word] & mask != 0 {
                bail!(ValidationError::InvalidChannelDefinition);
            }
            seen[word] |= mask;
        }
    }
    drop(seen);

    let mut reorder_budget =
        DecodeOwnerBudget::for_components(retained_image_bytes, components, components.capacity())?;
    reorder_budget.include_elements::<usize>(component_count)?;
    reorder_budget.include_elements::<usize>(component_count)?;
    let mut source_order = Vec::new();
    let mut destination_by_source = Vec::new();
    try_reserve_decode_elements(&mut source_order, component_count)?;
    reorder_budget.include_capacity_overage::<usize>(component_count, source_order.capacity())?;
    try_resize_decode_elements(&mut destination_by_source, component_count, 0_usize)?;
    reorder_budget
        .include_capacity_overage::<usize>(component_count, destination_by_source.capacity())?;
    source_order.extend(0..component_count);
    source_order.sort_unstable_by_key(|&entry_idx| {
        let definition = &cdef.channel_definitions[entry_idx];
        (
            channel_association_sort_key(definition.association),
            definition.channel_index,
        )
    });
    for entry_idx in &mut source_order {
        *entry_idx = usize::from(cdef.channel_definitions[*entry_idx].channel_index);
    }
    for (destination, &source) in source_order.iter().enumerate() {
        destination_by_source[source] = destination;
    }
    drop(source_order);

    for source in 0..component_count {
        while destination_by_source[source] != source {
            let destination = destination_by_source[source];
            components.swap(source, destination);
            destination_by_source.swap(source, destination);
        }
    }
    Ok(())
}

const fn channel_association_sort_key(association: ChannelAssociation) -> u16 {
    match association {
        ChannelAssociation::Colour(index) => index,
        ChannelAssociation::WholeImage | ChannelAssociation::Unspecified => u16::MAX,
    }
}

#[cfg(test)]
mod tests {
    use alloc::{vec, vec::Vec};
    use core::mem::size_of;

    use super::validate_and_reorder_channels;
    use crate::image::DecodeOwnerBudget;
    use crate::j2c::ComponentData;
    use crate::jp2::cdef::{
        ChannelAssociation, ChannelDefinition, ChannelDefinitionBox, ChannelType,
    };
    use crate::math::{SimdBuffer, SIMD_WIDTH};

    fn component(value: f32) -> ComponentData {
        ComponentData {
            container: SimdBuffer::<SIMD_WIDTH>::new(vec![value]),
            integer_container: None,
            bit_depth: 8,
            signed: false,
        }
    }

    #[test]
    fn channel_reorder_moves_component_owners_without_cloning_payloads() {
        let mut components = vec![component(10.0), component(20.0), component(30.0)];
        let pointers = [
            components[0].container.truncated().as_ptr(),
            components[1].container.truncated().as_ptr(),
            components[2].container.truncated().as_ptr(),
        ];
        let cdef = ChannelDefinitionBox {
            channel_definitions: vec![
                ChannelDefinition {
                    channel_index: 0,
                    channel_type: ChannelType::Colour,
                    association: ChannelAssociation::Colour(2),
                },
                ChannelDefinition {
                    channel_index: 1,
                    channel_type: ChannelType::Colour,
                    association: ChannelAssociation::Colour(3),
                },
                ChannelDefinition {
                    channel_index: 2,
                    channel_type: ChannelType::Colour,
                    association: ChannelAssociation::Colour(1),
                },
            ],
        };

        validate_and_reorder_channels(&cdef, &mut components, 0).expect("valid channel mapping");
        assert_eq!(
            components[0].container.truncated()[0].to_bits(),
            30.0_f32.to_bits()
        );
        assert_eq!(
            components[1].container.truncated()[0].to_bits(),
            10.0_f32.to_bits()
        );
        assert_eq!(
            components[2].container.truncated()[0].to_bits(),
            20.0_f32.to_bits()
        );
        assert_eq!(components[0].container.truncated().as_ptr(), pointers[2]);
        assert_eq!(components[1].container.truncated().as_ptr(), pointers[0]);
        assert_eq!(components[2].container.truncated().as_ptr(), pointers[1]);
    }

    #[test]
    fn channel_reorder_uses_channel_indices_when_cdef_entries_are_reversed() {
        let mut components = vec![component(10.0), component(20.0), component(30.0)];
        let cdef = ChannelDefinitionBox {
            channel_definitions: vec![
                ChannelDefinition {
                    channel_index: 2,
                    channel_type: ChannelType::Colour,
                    association: ChannelAssociation::Colour(1),
                },
                ChannelDefinition {
                    channel_index: 1,
                    channel_type: ChannelType::Colour,
                    association: ChannelAssociation::Colour(2),
                },
                ChannelDefinition {
                    channel_index: 0,
                    channel_type: ChannelType::Colour,
                    association: ChannelAssociation::Colour(3),
                },
            ],
        };

        validate_and_reorder_channels(&cdef, &mut components, 0).expect("valid channel mapping");
        let values = components
            .iter()
            .map(|component| component.container.truncated()[0])
            .collect::<Vec<_>>();
        assert_eq!(values, [30.0, 20.0, 10.0]);
    }

    #[test]
    fn shared_decode_budget_uses_simd_and_integer_capacities() {
        let mut integer = Vec::new();
        integer.try_reserve_exact(5).expect("test integer capacity");
        integer.push(1_i64);
        let components = vec![ComponentData {
            container: SimdBuffer::<SIMD_WIDTH>::new(vec![1.0]),
            integer_container: Some(integer),
            bit_depth: 16,
            signed: false,
        }];
        let budget = DecodeOwnerBudget::for_components(0, &components, components.capacity())
            .expect("small budget");
        assert!(budget.bytes() > size_of::<ComponentData>() + size_of::<f32>() + size_of::<i64>());
    }
}
