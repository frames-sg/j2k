// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fallible construction of per-component header metadata.

use alloc::vec::Vec;

use super::super::{
    allocation::{try_clone_coding_style, try_clone_quantization_info},
    CodingStyleComponent, CodingStyleDefault, ComponentInfo, ComponentSizeInfo, QuantizationInfo,
    StepSize,
};
use super::allocation::{account_component_metadata_peak, HeaderMarkerBudget};
use crate::error::{Result, ValidationError};

pub(super) fn build_component_infos(
    component_sizes: &[ComponentSizeInfo],
    coding_overrides: &[Option<CodingStyleComponent>],
    quantization_overrides: &[Option<QuantizationInfo>],
    roi_shifts: &[Option<u8>],
    coding_default: &CodingStyleDefault,
    quantization_default: &QuantizationInfo,
    budget: &mut HeaderMarkerBudget,
) -> Result<Vec<ComponentInfo>> {
    let component_count = component_sizes.len();
    if coding_overrides.len() != component_count
        || quantization_overrides.len() != component_count
        || roi_shifts.len() != component_count
    {
        return Err(ValidationError::InvalidComponentMetadata.into());
    }

    account_component_metadata_peak(
        component_sizes,
        coding_overrides,
        quantization_overrides,
        coding_default,
        quantization_default,
        budget,
    )?;

    let mut component_infos = Vec::new();
    crate::try_reserve_decode_elements(&mut component_infos, component_count)?;
    budget
        .account_capacity_overage::<ComponentInfo>(component_count, component_infos.capacity())?;

    for (idx, size_info) in component_sizes.iter().copied().enumerate() {
        let coding_source = coding_overrides[idx]
            .as_ref()
            .unwrap_or(&coding_default.component_parameters);
        let mut coding_style = try_clone_coding_style(coding_source)?;
        budget.account_capacity_overage::<(u8, u8)>(
            coding_source.parameters.precinct_exponents.len(),
            coding_style.parameters.precinct_exponents.capacity(),
        )?;
        coding_style.flags.raw |= coding_default.component_parameters.flags.raw;

        let quantization_source = quantization_overrides[idx]
            .as_ref()
            .unwrap_or(quantization_default);
        let quantization_info = try_clone_quantization_info(quantization_source)?;
        budget.account_capacity_overage::<StepSize>(
            quantization_source.step_sizes.len(),
            quantization_info.step_sizes.capacity(),
        )?;

        component_infos.push(ComponentInfo {
            size_info,
            coding_style,
            quantization_info,
            roi_shift: roi_shifts[idx].unwrap_or(0),
        });
    }

    Ok(component_infos)
}
