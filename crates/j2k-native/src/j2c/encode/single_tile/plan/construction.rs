// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fallible, phase-accounted construction for the standard single-tile plan.

use alloc::vec::Vec;

use crate::j2c::encode::allocation::{
    checked_add_bytes, checked_element_bytes, host_allocation_failed,
};
use crate::j2c::encode::{
    quantize, EncodeComponentSampleInfo, EncodeOptions, NativeEncodePipelineResult,
    NativeEncodeSession, QuantStepSize,
};

mod roi;
mod sampling;

pub(super) use roi::try_roi_plans;
pub(super) use sampling::{try_component_sampling, validate_component_sampling};

pub(super) struct PlanConstruction<'session, 'input> {
    session: &'session NativeEncodeSession<'input>,
    live_bytes: usize,
}

impl<'session, 'input> PlanConstruction<'session, 'input> {
    pub(super) const fn new(
        session: &'session NativeEncodeSession<'input>,
        already_live_bytes: usize,
    ) -> Self {
        Self {
            session,
            live_bytes: already_live_bytes,
        }
    }

    fn before(&self, requested_bytes: usize, what: &'static str) -> NativeEncodePipelineResult<()> {
        self.session.checked_phase(
            checked_add_bytes(self.live_bytes, requested_bytes, what)?,
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
        self.session.checked_phase(self.live_bytes, what)?;
        Ok(())
    }

    pub(super) fn try_vec<T>(
        &mut self,
        count: usize,
        what: &'static str,
    ) -> NativeEncodePipelineResult<Vec<T>> {
        let requested = checked_element_bytes::<T>(count, what)?;
        self.before(requested, what)?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(count)
            .map_err(|_| host_allocation_failed(what, requested))?;
        self.retain(checked_element_bytes::<T>(values.capacity(), what)?, what)?;
        Ok(values)
    }

    pub(super) fn try_copy_slice<T: Copy>(
        &mut self,
        source: &[T],
        what: &'static str,
    ) -> NativeEncodePipelineResult<Vec<T>> {
        let mut values = self.try_vec(source.len(), what)?;
        values.extend_from_slice(source);
        Ok(values)
    }

    pub(super) fn try_map_slice<S, T>(
        &mut self,
        source: &[S],
        what: &'static str,
        mut map: impl FnMut(&S) -> T,
    ) -> NativeEncodePipelineResult<Vec<T>> {
        let mut values = self.try_vec(source.len(), what)?;
        values.extend(source.iter().map(&mut map));
        Ok(values)
    }

    pub(super) fn try_step_sizes(
        &mut self,
        bit_depth: u8,
        num_levels: u8,
        reversible: bool,
        guard_bits: u8,
        options: &EncodeOptions,
    ) -> NativeEncodePipelineResult<Vec<QuantStepSize>> {
        let count = usize::from(num_levels)
            .checked_mul(3)
            .and_then(|count| count.checked_add(1))
            .ok_or(crate::EncodeError::ArithmeticOverflow {
                what: "single-tile quantization step count",
            })?;
        let mut steps = self.try_vec(count, "single-tile quantization steps")?;
        quantize::append_step_sizes_with_irreversible_profile(
            &mut steps,
            bit_depth,
            num_levels,
            reversible,
            guard_bits,
            options.irreversible_quantization_scale,
            options.irreversible_quantization_subband_scales,
        );
        Ok(steps)
    }

    pub(super) fn try_component_step_sizes(
        &mut self,
        component_info: &[EncodeComponentSampleInfo],
        num_levels: u8,
        reversible: bool,
        guard_bits: u8,
        options: &EncodeOptions,
    ) -> NativeEncodePipelineResult<Vec<Vec<QuantStepSize>>> {
        let mut components = self.try_vec::<Vec<QuantStepSize>>(
            component_info.len(),
            "single-tile component step owners",
        )?;
        for info in component_info {
            components.push(self.try_step_sizes(
                info.bit_depth,
                num_levels,
                reversible,
                guard_bits,
                options,
            )?);
        }
        Ok(components)
    }

    pub(super) fn try_quantization(
        &mut self,
        steps: &[QuantStepSize],
        what: &'static str,
    ) -> NativeEncodePipelineResult<Vec<(u16, u16)>> {
        self.try_map_slice(steps, what, |step| (step.exponent, step.mantissa))
    }

    pub(super) fn try_component_quantization(
        &mut self,
        component_steps: &[Vec<QuantStepSize>],
    ) -> NativeEncodePipelineResult<Vec<Vec<(u16, u16)>>> {
        let mut components = self.try_vec::<Vec<(u16, u16)>>(
            component_steps.len(),
            "single-tile component quantization owners",
        )?;
        for steps in component_steps {
            components.push(self.try_quantization(steps, "single-tile component quantization")?);
        }
        Ok(components)
    }
}
