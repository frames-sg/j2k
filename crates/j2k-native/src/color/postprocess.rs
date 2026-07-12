// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fallible, aggregate-budgeted palette resolution and channel reordering.

use super::{palette_index, sign_extend_palette_value};
use crate::error::{bail, Result, ValidationError};
use crate::image::DecodeOwnerBudget;
use crate::j2c::ComponentData;
use crate::jp2::cdef::{ChannelAssociation, ChannelDefinitionBox};
use crate::jp2::cmap::ComponentMappingType;
use crate::jp2::pclr::PaletteBox;
use crate::jp2::ImageBoxes;
use crate::math::{SimdBuffer, SIMD_WIDTH};
use crate::{try_reserve_decode_elements, try_resize_decode_elements, ColorError, DecodingError};
use alloc::vec::Vec;

const BITS_PER_ASSOCIATION_WORD: usize = 64;
const MAX_EXACT_F32_INTEGER_BITS: u8 = 24;

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
    let mut seen_color_associations = Vec::new();
    try_resize_decode_elements(&mut seen_color_associations, word_count, 0_u64)?;
    validation_budget
        .include_capacity_overage::<u64>(word_count, seen_color_associations.capacity())?;
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
            if seen_color_associations[word] & mask != 0 {
                bail!(ValidationError::InvalidChannelDefinition);
            }
            seen_color_associations[word] |= mask;
        }
    }
    drop(seen_color_associations);

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
    source_order.sort_unstable_by_key(|&source_idx| {
        (
            channel_association_sort_key(cdef.channel_definitions[source_idx].association),
            source_idx,
        )
    });
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

#[expect(
    clippy::cast_precision_loss,
    reason = "palette integer values are intentionally exposed through the decoder's f32 component representation"
)]
pub(crate) fn resolve_palette_indices(
    components: Vec<ComponentData>,
    boxes: &ImageBoxes,
    retained_image_bytes: usize,
) -> Result<Vec<ComponentData>> {
    let Some(palette) = boxes.palette.as_ref() else {
        return Ok(components);
    };
    let Some(mapping) = boxes.component_mapping.as_ref() else {
        bail!(ColorError::PaletteResolutionFailed);
    };
    if mapping.entries.is_empty() {
        bail!(ColorError::PaletteResolutionFailed);
    }

    let mut logical_budget = DecodeOwnerBudget::for_components(
        retained_image_bytes,
        &components,
        components.capacity(),
    )?;
    logical_budget.include_elements::<ComponentData>(mapping.entries.len())?;
    for entry in &mapping.entries {
        let component = components
            .get(usize::from(entry.component_index))
            .ok_or(ColorError::PaletteResolutionFailed)?;
        include_mapped_component(&mut logical_budget, component, palette, entry.mapping_type)?;
    }

    let mut resolved = Vec::new();
    try_reserve_decode_elements(&mut resolved, mapping.entries.len())?;
    logical_budget
        .include_capacity_overage::<ComponentData>(mapping.entries.len(), resolved.capacity())?;
    for entry in &mapping.entries {
        let component = components
            .get(usize::from(entry.component_index))
            .ok_or(ColorError::PaletteResolutionFailed)?;
        let resolved_component = match entry.mapping_type {
            ComponentMappingType::Direct => try_clone_component(component, &mut logical_budget)?,
            ComponentMappingType::Palette { column } => {
                let column_idx = usize::from(column);
                let column_info = palette
                    .columns
                    .get(column_idx)
                    .ok_or(ColorError::PaletteResolutionFailed)?;
                let sample_count = component.container.truncated().len();
                let mut mapped = SimdBuffer::<SIMD_WIDTH>::try_zeros(sample_count)
                    .map_err(|_| DecodingError::HostAllocationFailed)?;
                let planned_capacity = SimdBuffer::<SIMD_WIDTH>::padded_len(sample_count)
                    .ok_or(ValidationError::ImageTooLarge)?;
                logical_budget
                    .include_capacity_overage::<f32>(planned_capacity, mapped.capacity())?;
                let mut exact_values = if column_info.bit_depth > MAX_EXACT_F32_INTEGER_BITS {
                    let mut values = Vec::new();
                    try_reserve_decode_elements(&mut values, sample_count)?;
                    logical_budget
                        .include_capacity_overage::<i64>(sample_count, values.capacity())?;
                    Some(values)
                } else {
                    None
                };
                for (sample_idx, &sample) in component.container.truncated().iter().enumerate() {
                    let index = palette_index(sample)?;
                    let raw = palette
                        .map(index, column_idx)
                        .ok_or(ColorError::PaletteResolutionFailed)?;
                    let exact = if column_info.signed {
                        sign_extend_palette_value(raw, column_info.bit_depth)
                    } else {
                        i64::try_from(raw).map_err(|_| ColorError::PaletteResolutionFailed)?
                    };
                    mapped[sample_idx] = exact as f32;
                    if let Some(values) = &mut exact_values {
                        values.push(exact);
                    }
                }
                ComponentData {
                    container: mapped,
                    integer_container: exact_values,
                    bit_depth: column_info.bit_depth,
                    signed: column_info.signed,
                }
            }
            ComponentMappingType::Unknown { .. } => {
                bail!(ColorError::PaletteResolutionFailed)
            }
        };
        resolved.push(resolved_component);
    }

    let mut actual_budget = DecodeOwnerBudget::for_components(
        retained_image_bytes,
        &components,
        components.capacity(),
    )?;
    actual_budget.include_components(&resolved, resolved.capacity())?;
    Ok(resolved)
}

fn include_mapped_component(
    budget: &mut DecodeOwnerBudget,
    component: &ComponentData,
    palette: &PaletteBox,
    mapping_type: ComponentMappingType,
) -> Result<()> {
    match mapping_type {
        ComponentMappingType::Direct => include_component_clone(budget, component),
        ComponentMappingType::Palette { column } => {
            let column = palette
                .columns
                .get(usize::from(column))
                .ok_or(ColorError::PaletteResolutionFailed)?;
            include_palette_component(budget, component, column.bit_depth)
        }
        ComponentMappingType::Unknown { .. } => Err(ColorError::PaletteResolutionFailed.into()),
    }
}

fn include_component_clone(
    budget: &mut DecodeOwnerBudget,
    component: &ComponentData,
) -> Result<()> {
    let padded = SimdBuffer::<SIMD_WIDTH>::padded_len(component.container.truncated().len())
        .ok_or(ValidationError::ImageTooLarge)?;
    budget.include_elements::<f32>(padded)?;
    if let Some(integers) = &component.integer_container {
        budget.include_elements::<i64>(integers.len())?;
    }
    Ok(())
}

fn include_palette_component(
    budget: &mut DecodeOwnerBudget,
    component: &ComponentData,
    bit_depth: u8,
) -> Result<()> {
    let sample_count = component.container.truncated().len();
    let padded =
        SimdBuffer::<SIMD_WIDTH>::padded_len(sample_count).ok_or(ValidationError::ImageTooLarge)?;
    budget.include_elements::<f32>(padded)?;
    if bit_depth > MAX_EXACT_F32_INTEGER_BITS {
        budget.include_elements::<i64>(sample_count)?;
    }
    Ok(())
}

fn try_clone_component(
    component: &ComponentData,
    budget: &mut DecodeOwnerBudget,
) -> Result<ComponentData> {
    let sample_count = component.container.truncated().len();
    let mut container = SimdBuffer::<SIMD_WIDTH>::try_zeros(sample_count)
        .map_err(|_| DecodingError::HostAllocationFailed)?;
    let planned_capacity =
        SimdBuffer::<SIMD_WIDTH>::padded_len(sample_count).ok_or(ValidationError::ImageTooLarge)?;
    budget.include_capacity_overage::<f32>(planned_capacity, container.capacity())?;
    container[..sample_count].copy_from_slice(component.container.truncated());
    let integer_container = component
        .integer_container
        .as_ref()
        .map(|source| -> Result<Vec<i64>> {
            let mut cloned = Vec::new();
            try_reserve_decode_elements(&mut cloned, source.len())?;
            budget.include_capacity_overage::<i64>(source.len(), cloned.capacity())?;
            cloned.extend_from_slice(source);
            Ok(cloned)
        })
        .transpose()?;
    Ok(ComponentData {
        container,
        integer_container,
        bit_depth: component.bit_depth,
        signed: component.signed,
    })
}

#[cfg(test)]
mod tests {
    use super::validate_and_reorder_channels;
    use crate::image::DecodeOwnerBudget;
    use crate::j2c::ComponentData;
    use crate::jp2::cdef::{
        ChannelAssociation, ChannelDefinition, ChannelDefinitionBox, ChannelType,
    };
    use crate::math::{SimdBuffer, SIMD_WIDTH};
    use alloc::{vec, vec::Vec};
    use core::mem::size_of;

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
