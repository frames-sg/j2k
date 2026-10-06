// SPDX-License-Identifier: MIT OR Apache-2.0

//! Grayscale batch planning and retained input contracts.

use super::{
    build_cuda_classic_grayscale_plans_from_referenced_with_profile,
    build_cuda_htj2k_grayscale_plan_from_bytes_for_device_plan_with_profile_and_cap,
    build_cuda_htj2k_grayscale_plans_from_referenced_with_profile, profile, CudaHtj2kDecodePlan,
    CudaHtj2kProfileReport, DecodeSettings, DeviceDecodePlan, Error, HostPhaseBudget,
    NativeDecoderContext, PixelFormat,
};
use crate::decoder::decode_profile::share_plan_wall_time;
use crate::decoder::plan::{
    next_payload_base, plan_in_waves, referenced_classic_payload_bytes, referenced_ht_payload_bytes,
};

pub(super) struct PreparedGrayscaleBatch {
    pub(super) plans: Vec<CudaHtj2kDecodePlan>,
    pub(super) reports: Vec<CudaHtj2kProfileReport>,
    /// Per-input payloads, uploaded back to back in this order. Plan block
    /// offsets already address that combined layout.
    pub(super) payload_parts: Vec<Vec<u8>>,
    pub(super) output_indices: Vec<usize>,
    pub(super) output_dimensions: Vec<(u32, u32)>,
    pub(super) source_indices: Vec<usize>,
}

/// Borrowed encoded bytes plus normalized geometry for one shared CUDA batch
/// plan. `None` retains the legacy full-frame path that discovers dimensions
/// while parsing.
#[derive(Clone, Copy)]
pub(crate) struct GrayscaleBatchInput<'a> {
    pub(crate) source_index: usize,
    pub(crate) bytes: &'a [u8],
    pub(crate) device_plan: Option<DeviceDecodePlan>,
    pub(crate) referenced_plan: Option<&'a j2k_native::J2kReferencedHtj2kPlan>,
    pub(crate) referenced_classic_plan: Option<&'a j2k_native::J2kReferencedClassicPlan>,
}

impl<'a> GrayscaleBatchInput<'a> {
    pub(crate) const fn full(bytes: &'a [u8]) -> Self {
        Self {
            source_index: 0,
            bytes,
            device_plan: None,
            referenced_plan: None,
            referenced_classic_plan: None,
        }
    }
}

/// One input's tile plans, with block offsets relative to its own payload.
struct BuiltGrayscaleInput {
    plans: Vec<(CudaHtj2kDecodePlan, CudaHtj2kProfileReport)>,
    payload: Vec<u8>,
}

impl BuiltGrayscaleInput {
    fn account_host_owners(&self, budget: &mut HostPhaseBudget) -> Result<(), Error> {
        budget.account_vec(&self.payload)?;
        for (plan, _) in &self.plans {
            plan.account_host_owners(budget)?;
        }
        Ok(())
    }
}

pub(super) fn prepare_grayscale_batch(
    inputs: &[GrayscaleBatchInput<'_>],
    fmt: PixelFormat,
    settings: DecodeSettings,
) -> Result<PreparedGrayscaleBatch, Error> {
    prepare_grayscale_batch_with_cap(
        inputs,
        fmt,
        settings,
        j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
    )
}

pub(super) fn prepare_grayscale_batch_with_cap<'a>(
    inputs: &[GrayscaleBatchInput<'a>],
    fmt: PixelFormat,
    settings: DecodeSettings,
    host_cap: usize,
) -> Result<PreparedGrayscaleBatch, Error> {
    const PLAN_OWNERS: &str = "j2k CUDA grayscale batch plan owners";
    let mut initial_budget = HostPhaseBudget::with_cap(PLAN_OWNERS, host_cap);
    let plan_started = profile::profile_now(true);
    let plan_capacity = inputs.iter().try_fold(0usize, |total, input| {
        let additional = input
            .referenced_plan
            .map_or_else(
                || input.referenced_classic_plan.map(|plan| plan.tiles().len()),
                |plan| Some(plan.tiles().len()),
            )
            .unwrap_or(1);
        total
            .checked_add(additional)
            .ok_or(Error::capability_rejected(
                j2k_core::CapabilityRejection::resource_limit(
                    "prepared CUDA grayscale tile count overflows",
                ),
            ))
    })?;
    let mut prepared = PreparedGrayscaleBatch {
        plans: initial_budget.try_vec_with_capacity(plan_capacity)?,
        reports: initial_budget.try_vec_with_capacity(plan_capacity)?,
        payload_parts: initial_budget.try_vec_with_capacity(inputs.len())?,
        output_indices: initial_budget.try_vec_with_capacity(plan_capacity)?,
        output_dimensions: initial_budget.try_vec_with_capacity(inputs.len())?,
        source_indices: initial_budget.try_vec_with_capacity(plan_capacity)?,
    };
    let mut payload_base = 0_u64;
    plan_in_waves(
        inputs,
        host_cap,
        &mut initial_budget,
        |input: &GrayscaleBatchInput<'a>, context: &mut NativeDecoderContext<'a>, worker_cap| {
            build_grayscale_input_with_cap(input, fmt, settings, context, worker_cap)
        },
        |budget, input_index, built| {
            built.account_host_owners(budget)?;
            append_grayscale_input(
                &mut prepared,
                input_index,
                &inputs[input_index],
                built,
                &mut payload_base,
            )
        },
    )?;
    let plan_wall_us = profile::elapsed_us(plan_started);
    share_plan_wall_time(&mut prepared.reports, plan_wall_us, |report| report);
    Ok(prepared)
}

fn build_grayscale_input_with_cap<'a>(
    input: &GrayscaleBatchInput<'a>,
    fmt: PixelFormat,
    settings: DecodeSettings,
    native_context: &mut NativeDecoderContext<'a>,
    host_cap: usize,
) -> Result<BuiltGrayscaleInput, Error> {
    let mut payload: Vec<u8>;
    // The phase cap covers adapter-owned planning temporaries and returned
    // CUDA plans. Encoded bytes and prepared native plans are borrowed from
    // the caller and remain outside this operation's ownership budget.
    let mut host_budget =
        HostPhaseBudget::with_cap("j2k CUDA bounded grayscale input planning", host_cap);
    let plans = match (input.referenced_plan, input.referenced_classic_plan) {
        (Some(referenced), None) => {
            payload = host_budget
                .try_vec_with_capacity(referenced_ht_payload_bytes(referenced.payloads())?)?;
            let device_plan = input.device_plan.ok_or(Error::capability_rejected(
                j2k_core::CapabilityRejection::geometry_mismatch(
                    "prepared CUDA HTJ2K plan is missing normalized output geometry",
                ),
            ))?;
            build_cuda_htj2k_grayscale_plans_from_referenced_with_profile(
                input.bytes,
                referenced,
                fmt,
                device_plan,
                &mut payload,
                &mut host_budget,
            )?
        }
        (None, Some(referenced)) => {
            payload = host_budget
                .try_vec_with_capacity(referenced_classic_payload_bytes(referenced.tiles())?)?;
            let device_plan = input.device_plan.ok_or(Error::capability_rejected(
                j2k_core::CapabilityRejection::geometry_mismatch(
                    "prepared CUDA classic plan is missing normalized output geometry",
                ),
            ))?;
            build_cuda_classic_grayscale_plans_from_referenced_with_profile(
                input.bytes,
                referenced,
                fmt,
                device_plan,
                &mut payload,
                &mut host_budget,
            )?
        }
        (None, None) => {
            let plan_settings = if input.device_plan.is_some() {
                settings
            } else {
                DecodeSettings::default()
            };
            let (mut plan, report) =
                build_cuda_htj2k_grayscale_plan_from_bytes_for_device_plan_with_profile_and_cap(
                    input.bytes,
                    fmt,
                    input.device_plan,
                    plan_settings,
                    native_context,
                    host_cap,
                )?;
            payload = plan.take_payload();
            host_budget.account_vec(&payload)?;
            plan.account_host_owners(&mut host_budget)?;
            let mut plans = host_budget.try_vec_with_capacity(1)?;
            plans.push((plan, report));
            plans
        }
        (Some(_), Some(_)) => {
            return Err(Error::capability_rejected(
                j2k_core::CapabilityRejection::contract_violation(
                    "prepared CUDA grayscale input contains conflicting codec plans",
                ),
            ))
        }
    };
    Ok(BuiltGrayscaleInput { plans, payload })
}

fn append_grayscale_input(
    prepared: &mut PreparedGrayscaleBatch,
    output_index: usize,
    input: &GrayscaleBatchInput<'_>,
    built: BuiltGrayscaleInput,
    payload_base: &mut u64,
) -> Result<(), Error> {
    let BuiltGrayscaleInput {
        plans: mut input_plans,
        payload,
    } = built;
    let Some(first) = input_plans.first() else {
        return Err(Error::capability_rejected(
            j2k_core::CapabilityRejection::missing_prepared_plan(
                "prepared CUDA grayscale input produced no executable tile plans",
            ),
        ));
    };
    let dimensions = input
        .device_plan
        .map_or(first.0.dimensions(), DeviceDecodePlan::output_dims);
    for (plan, _) in &mut input_plans {
        plan.rebase_payload_offsets(*payload_base)?;
    }
    *payload_base = next_payload_base(*payload_base, payload.len())?;

    for (plan, report) in input_plans {
        prepared.plans.push(plan);
        prepared.reports.push(report);
        prepared.output_indices.push(output_index);
        prepared.source_indices.push(input.source_index);
    }
    prepared.payload_parts.push(payload);
    prepared.output_dimensions.push(dimensions);
    Ok(())
}

pub(super) fn grayscale_owner_budget(
    plans: &Vec<CudaHtj2kDecodePlan>,
    reports: &Vec<CudaHtj2kProfileReport>,
    payload_parts: &Vec<Vec<u8>>,
    what: &'static str,
) -> Result<HostPhaseBudget, Error> {
    let mut budget = HostPhaseBudget::new(what);
    budget.account_vec(plans)?;
    budget.account_vec(reports)?;
    budget.account_vec(payload_parts)?;
    for part in payload_parts {
        budget.account_vec(part)?;
    }
    for plan in plans {
        plan.account_host_owners(&mut budget)?;
    }
    Ok(budget)
}
