// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fallible, aggregate-budgeted palette resolution.

use alloc::vec::Vec;

use crate::error::{bail, Result, ValidationError};
use crate::image::DecodeOwnerBudget;
use crate::j2c::ComponentData;
use crate::jp2::cmap::ComponentMappingType;
use crate::jp2::pclr::PaletteBox;
use crate::jp2::ImageBoxes;
use crate::math::{self, SimdBuffer, SIMD_WIDTH};
use crate::{try_reserve_decode_elements, ColorError, DecodingError};

const MAX_EXACT_F32_INTEGER_BITS: u8 = 24;

#[expect(
    clippy::cast_possible_truncation,
    reason = "Rust's saturating float-to-integer conversion is retained before rejecting negative indices"
)]
fn palette_index(sample: f32) -> Result<usize> {
    let rounded = math::round_f32(sample) as i64;
    usize::try_from(rounded).map_err(|_| ColorError::PaletteResolutionFailed.into())
}

fn sign_extend_palette_value(raw: u64, bit_depth: u8) -> i64 {
    if bit_depth == 0 || bit_depth >= 64 {
        return raw.cast_signed();
    }
    let mask = (1_u64 << bit_depth) - 1;
    let value = raw & mask;
    let shift = 64 - u32::from(bit_depth);
    (value << shift).cast_signed() >> shift
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
    use super::palette_index;

    #[test]
    fn palette_indices_reject_negative_samples_without_wrapping() {
        assert!(palette_index(-1.0).is_err());
        assert_eq!(palette_index(2.4).expect("valid palette index"), 2);
    }
}
