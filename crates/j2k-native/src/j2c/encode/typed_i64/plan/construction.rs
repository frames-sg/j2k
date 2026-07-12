// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fallible typed high-bit plan construction helpers.

use alloc::vec::Vec;

use crate::j2c::encode::allocation::{
    checked_add_bytes, checked_element_bytes, host_allocation_failed,
};
use crate::j2c::encode::multitile::try_copy_slice;
use crate::j2c::encode::{
    quantize, EncodeComponentSampleInfo, EncodeOptions, EncodeTypedComponentPlane,
    NativeEncodePipelineResult, NativeEncodeSession, QuantStepSize,
};

pub(super) struct PlanConstruction<'a, 'input> {
    session: &'a NativeEncodeSession<'input>,
    retained_base_bytes: usize,
    live_bytes: usize,
}

impl<'a, 'input> PlanConstruction<'a, 'input> {
    pub(super) const fn new(
        session: &'a NativeEncodeSession<'input>,
        retained_base_bytes: usize,
    ) -> Self {
        Self {
            session,
            retained_base_bytes,
            live_bytes: 0,
        }
    }

    fn before(&self, requested_bytes: usize, what: &'static str) -> NativeEncodePipelineResult<()> {
        self.session.checked_phase(
            checked_add_bytes(
                self.retained_base_bytes,
                checked_add_bytes(self.live_bytes, requested_bytes, what)?,
                what,
            )?,
            what,
        )?;
        Ok(())
    }

    fn retain(
        &mut self,
        actual_bytes: usize,
        what: &'static str,
    ) -> NativeEncodePipelineResult<()> {
        self.live_bytes = checked_add_bytes(self.live_bytes, actual_bytes, what)?;
        self.session.checked_phase(
            checked_add_bytes(self.retained_base_bytes, self.live_bytes, what)?,
            what,
        )?;
        Ok(())
    }
}

pub(super) fn try_step_sizes(
    bit_depth: u8,
    num_levels: u8,
    guard_bits: u8,
    options: &EncodeOptions,
    construction: &mut PlanConstruction<'_, '_>,
) -> NativeEncodePipelineResult<Vec<QuantStepSize>> {
    let count = usize::from(num_levels) * 3 + 1;
    let requested = checked_element_bytes::<QuantStepSize>(count, "typed i64 step sizes")?;
    construction.before(requested, "typed i64 step sizes")?;
    let mut steps = Vec::new();
    steps
        .try_reserve_exact(count)
        .map_err(|_| host_allocation_failed("typed i64 step sizes", requested))?;
    quantize::append_step_sizes_with_irreversible_profile(
        &mut steps,
        bit_depth,
        num_levels,
        true,
        guard_bits,
        options.irreversible_quantization_scale,
        options.irreversible_quantization_subband_scales,
    );
    construction.retain(
        checked_element_bytes::<QuantStepSize>(steps.capacity(), "typed i64 step sizes")?,
        "typed i64 step sizes",
    )?;
    Ok(steps)
}

pub(super) fn try_component_sample_info(
    planes: &[EncodeTypedComponentPlane<'_>],
    construction: &mut PlanConstruction<'_, '_>,
) -> NativeEncodePipelineResult<Vec<EncodeComponentSampleInfo>> {
    let requested = checked_element_bytes::<EncodeComponentSampleInfo>(
        planes.len(),
        "typed i64 component metadata",
    )?;
    construction.before(requested, "typed i64 component metadata")?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(planes.len())
        .map_err(|_| host_allocation_failed("typed i64 component metadata", requested))?;
    values.extend(planes.iter().map(|plane| EncodeComponentSampleInfo {
        bit_depth: plane.bit_depth,
        signed: plane.signed,
    }));
    construction.retain(
        checked_element_bytes::<EncodeComponentSampleInfo>(
            values.capacity(),
            "typed i64 component metadata",
        )?,
        "typed i64 component metadata",
    )?;
    Ok(values)
}

pub(super) fn try_component_step_sizes(
    component_info: &[EncodeComponentSampleInfo],
    num_levels: u8,
    guard_bits: u8,
    options: &EncodeOptions,
    construction: &mut PlanConstruction<'_, '_>,
) -> NativeEncodePipelineResult<Vec<Vec<QuantStepSize>>> {
    let outer_requested = checked_element_bytes::<Vec<QuantStepSize>>(
        component_info.len(),
        "typed i64 component step owners",
    )?;
    construction.before(outer_requested, "typed i64 component step owners")?;
    let mut components = Vec::new();
    components
        .try_reserve_exact(component_info.len())
        .map_err(|_| host_allocation_failed("typed i64 component step owners", outer_requested))?;
    construction.retain(
        checked_element_bytes::<Vec<QuantStepSize>>(
            components.capacity(),
            "typed i64 component step owners",
        )?,
        "typed i64 component step owners",
    )?;
    for info in component_info {
        components.push(try_step_sizes(
            info.bit_depth,
            num_levels,
            guard_bits,
            options,
            construction,
        )?);
    }
    Ok(components)
}

pub(super) fn try_quantization(
    steps: &[QuantStepSize],
    construction: &mut PlanConstruction<'_, '_>,
) -> NativeEncodePipelineResult<Vec<(u16, u16)>> {
    let requested = checked_element_bytes::<(u16, u16)>(steps.len(), "typed i64 quantization")?;
    construction.before(requested, "typed i64 quantization")?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(steps.len())
        .map_err(|_| host_allocation_failed("typed i64 quantization", requested))?;
    values.extend(steps.iter().map(|step| (step.exponent, step.mantissa)));
    construction.retain(
        checked_element_bytes::<(u16, u16)>(values.capacity(), "typed i64 quantization")?,
        "typed i64 quantization",
    )?;
    Ok(values)
}

pub(super) fn try_component_quantization(
    component_steps: &[Vec<QuantStepSize>],
    construction: &mut PlanConstruction<'_, '_>,
) -> NativeEncodePipelineResult<Vec<Vec<(u16, u16)>>> {
    let requested = checked_element_bytes::<Vec<(u16, u16)>>(
        component_steps.len(),
        "typed i64 component quantization owners",
    )?;
    construction.before(requested, "typed i64 component quantization owners")?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(component_steps.len())
        .map_err(|_| {
            host_allocation_failed("typed i64 component quantization owners", requested)
        })?;
    construction.retain(
        checked_element_bytes::<Vec<(u16, u16)>>(
            values.capacity(),
            "typed i64 component quantization owners",
        )?,
        "typed i64 component quantization owners",
    )?;
    for steps in component_steps {
        values.push(try_quantization(steps, construction)?);
    }
    Ok(values)
}

pub(super) fn try_component_sampling(
    planes: &[EncodeTypedComponentPlane<'_>],
    construction: &mut PlanConstruction<'_, '_>,
) -> NativeEncodePipelineResult<Vec<(u8, u8)>> {
    let requested =
        checked_element_bytes::<(u8, u8)>(planes.len(), "typed i64 component sampling")?;
    construction.before(requested, "typed i64 component sampling")?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(planes.len())
        .map_err(|_| host_allocation_failed("typed i64 component sampling", requested))?;
    values.extend(planes.iter().map(|plane| (plane.x_rsiz, plane.y_rsiz)));
    construction.retain(
        checked_element_bytes::<(u8, u8)>(values.capacity(), "typed i64 component sampling")?,
        "typed i64 component sampling",
    )?;
    Ok(values)
}

pub(in crate::j2c::encode::typed_i64) fn try_precinct_exponents(
    options: &EncodeOptions,
    num_levels: u8,
    retained_base_bytes: usize,
    session: &NativeEncodeSession<'_>,
) -> NativeEncodePipelineResult<Vec<(u8, u8)>> {
    super::super::super::validate_precinct_exponents_for_options(options, num_levels)
        .map_err(super::super::super::NativeEncodePipelineError::invalid_input)?;
    let requested = checked_element_bytes::<(u8, u8)>(
        options.precinct_exponents.len(),
        "typed i64 precinct exponents",
    )?;
    session.checked_phase(
        checked_add_bytes(
            retained_base_bytes,
            requested,
            "typed i64 precinct exponents",
        )?,
        "typed i64 precinct exponents",
    )?;
    let values = try_copy_slice(&options.precinct_exponents, "typed i64 precinct exponents")?;
    session.checked_phase(
        checked_add_bytes(
            retained_base_bytes,
            checked_element_bytes::<(u8, u8)>(values.capacity(), "typed i64 precinct exponents")?,
            "typed i64 precinct exponents",
        )?,
        "typed i64 precinct exponents",
    )?;
    Ok(values)
}
