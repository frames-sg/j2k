// SPDX-License-Identifier: MIT OR Apache-2.0

//! Grayscale batch planning and retained input contracts.

use super::{
    build_cuda_classic_grayscale_plans_from_referenced_with_profile,
    build_cuda_htj2k_grayscale_plan_from_bytes_for_device_plan_with_profile,
    build_cuda_htj2k_grayscale_plan_from_bytes_with_profile,
    build_cuda_htj2k_grayscale_plans_from_referenced_with_profile, CudaHtj2kDecodePlan,
    CudaHtj2kProfileReport, DecodeSettings, DeviceDecodePlan, Error, HostPhaseBudget,
    NativeDecoderContext, PixelFormat,
};

pub(super) struct PreparedGrayscaleBatch {
    pub(super) plans: Vec<CudaHtj2kDecodePlan>,
    pub(super) reports: Vec<CudaHtj2kProfileReport>,
    pub(super) shared_payload: Vec<u8>,
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

pub(super) fn prepare_grayscale_batch(
    inputs: &[GrayscaleBatchInput<'_>],
    fmt: PixelFormat,
    settings: DecodeSettings,
) -> Result<PreparedGrayscaleBatch, Error> {
    let mut initial_budget = HostPhaseBudget::new("j2k CUDA grayscale batch plan owners");
    let mut prepared = PreparedGrayscaleBatch {
        plans: initial_budget.try_vec_with_capacity(inputs.len())?,
        reports: initial_budget.try_vec_with_capacity(inputs.len())?,
        shared_payload: Vec::new(),
        output_indices: initial_budget.try_vec_with_capacity(inputs.len())?,
        output_dimensions: initial_budget.try_vec_with_capacity(inputs.len())?,
        source_indices: initial_budget.try_vec_with_capacity(inputs.len())?,
    };
    let mut native_context = NativeDecoderContext::default();

    for (output_index, input) in inputs.iter().enumerate() {
        let (input_plans, payload_is_shared) = prepare_grayscale_input(
            input,
            fmt,
            settings,
            &mut native_context,
            &mut initial_budget,
            &mut prepared,
        )?;
        append_grayscale_input(
            &mut prepared,
            output_index,
            input,
            input_plans,
            payload_is_shared,
        )?;
    }

    Ok(prepared)
}

fn prepare_grayscale_input<'a>(
    input: &GrayscaleBatchInput<'a>,
    fmt: PixelFormat,
    settings: DecodeSettings,
    native_context: &mut NativeDecoderContext<'a>,
    initial_budget: &mut HostPhaseBudget,
    prepared: &mut PreparedGrayscaleBatch,
) -> Result<(Vec<(CudaHtj2kDecodePlan, CudaHtj2kProfileReport)>, bool), Error> {
    match (input.referenced_plan, input.referenced_classic_plan) {
        (Some(referenced), None) => {
            let device_plan = input.device_plan.ok_or(Error::UnsupportedCudaRequest {
                reason: "prepared CUDA HTJ2K plan is missing normalized output geometry",
            })?;
            let mut budget = grayscale_owner_budget(
                &prepared.plans,
                &prepared.reports,
                &prepared.shared_payload,
                None,
                "j2k CUDA referenced grayscale batch plan owners",
            )?;
            build_cuda_htj2k_grayscale_plans_from_referenced_with_profile(
                input.bytes,
                referenced,
                fmt,
                device_plan,
                &mut prepared.shared_payload,
                &mut budget,
            )
            .map(|plans| (plans, true))
        }
        (None, Some(referenced)) => {
            let device_plan = input.device_plan.ok_or(Error::UnsupportedCudaRequest {
                reason: "prepared CUDA classic plan is missing normalized output geometry",
            })?;
            let mut budget = grayscale_owner_budget(
                &prepared.plans,
                &prepared.reports,
                &prepared.shared_payload,
                None,
                "j2k CUDA referenced classic grayscale batch plan owners",
            )?;
            build_cuda_classic_grayscale_plans_from_referenced_with_profile(
                input.bytes,
                referenced,
                fmt,
                device_plan,
                &mut prepared.shared_payload,
                &mut budget,
            )
            .map(|plans| (plans, true))
        }
        (None, None) => {
            let (plan, report) = match input.device_plan {
                Some(device_plan) => {
                    build_cuda_htj2k_grayscale_plan_from_bytes_for_device_plan_with_profile(
                        input.bytes,
                        fmt,
                        Some(device_plan),
                        settings,
                        native_context,
                    )?
                }
                None => build_cuda_htj2k_grayscale_plan_from_bytes_with_profile(
                    input.bytes,
                    fmt,
                    native_context,
                )?,
            };
            let mut plans = initial_budget.try_vec_with_capacity(1)?;
            plans.push((plan, report));
            Ok((plans, false))
        }
        (Some(_), Some(_)) => Err(Error::UnsupportedCudaRequest {
            reason: "prepared CUDA grayscale input contains conflicting codec plans",
        }),
    }
}

fn append_grayscale_input(
    prepared: &mut PreparedGrayscaleBatch,
    output_index: usize,
    input: &GrayscaleBatchInput<'_>,
    mut input_plans: Vec<(CudaHtj2kDecodePlan, CudaHtj2kProfileReport)>,
    payload_is_shared: bool,
) -> Result<(), Error> {
    let Some(first) = input_plans.first() else {
        return Err(Error::UnsupportedCudaRequest {
            reason: "prepared CUDA grayscale input produced no executable tile plans",
        });
    };
    let dimensions = input
        .device_plan
        .map_or(first.0.dimensions(), DeviceDecodePlan::output_dims);
    if !payload_is_shared {
        for (plan, _) in &mut input_plans {
            let mut budget = grayscale_owner_budget(
                &prepared.plans,
                &prepared.reports,
                &prepared.shared_payload,
                Some(plan),
                "j2k CUDA grayscale batch plan owners",
            )?;
            plan.append_payload_to_shared_with_budget(&mut prepared.shared_payload, &mut budget)?;
        }
    }

    let mut budget = grayscale_owner_budget(
        &prepared.plans,
        &prepared.reports,
        &prepared.shared_payload,
        None,
        "j2k CUDA grayscale tile owner append",
    )?;
    budget.account_vec(&prepared.output_indices)?;
    budget.account_vec(&prepared.output_dimensions)?;
    budget.account_vec(&prepared.source_indices)?;
    budget.try_vec_reserve(&mut prepared.plans, input_plans.len())?;
    budget.try_vec_reserve(&mut prepared.reports, input_plans.len())?;
    budget.try_vec_reserve(&mut prepared.output_indices, input_plans.len())?;
    budget.try_vec_reserve(&mut prepared.source_indices, input_plans.len())?;
    for (plan, report) in input_plans {
        prepared.plans.push(plan);
        prepared.reports.push(report);
        prepared.output_indices.push(output_index);
        prepared.source_indices.push(input.source_index);
    }
    prepared.output_dimensions.push(dimensions);
    Ok(())
}

pub(super) fn grayscale_owner_budget(
    plans: &Vec<CudaHtj2kDecodePlan>,
    reports: &Vec<CudaHtj2kProfileReport>,
    payload: &Vec<u8>,
    pending: Option<&CudaHtj2kDecodePlan>,
    what: &'static str,
) -> Result<HostPhaseBudget, Error> {
    let mut budget = HostPhaseBudget::new(what);
    budget.account_vec(plans)?;
    budget.account_vec(reports)?;
    budget.account_vec(payload)?;
    for plan in plans {
        plan.account_host_owners(&mut budget)?;
    }
    if let Some(plan) = pending {
        plan.account_host_owners(&mut budget)?;
    }
    Ok(budget)
}
