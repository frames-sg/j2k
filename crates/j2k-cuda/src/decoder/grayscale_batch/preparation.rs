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

#[expect(
    clippy::too_many_lines,
    reason = "one preparation boundary keeps tile plans, shared payload ownership, and dense output identities aligned"
)]
pub(super) fn prepare_grayscale_batch(
    inputs: &[GrayscaleBatchInput<'_>],
    fmt: PixelFormat,
    settings: DecodeSettings,
) -> Result<PreparedGrayscaleBatch, Error> {
    let mut initial_budget = HostPhaseBudget::new("j2k CUDA grayscale batch plan owners");
    let mut plans = initial_budget.try_vec_with_capacity(inputs.len())?;
    let mut reports = initial_budget.try_vec_with_capacity(inputs.len())?;
    let mut output_indices = initial_budget.try_vec_with_capacity(inputs.len())?;
    let mut output_dimensions = initial_budget.try_vec_with_capacity(inputs.len())?;
    let mut source_indices = initial_budget.try_vec_with_capacity(inputs.len())?;
    let mut shared_payload = Vec::new();
    let mut native_context = NativeDecoderContext::default();

    for (output_index, input) in inputs.iter().enumerate() {
        let (mut input_plans, payload_is_shared) =
            match (input.referenced_plan, input.referenced_classic_plan) {
                (Some(referenced), None) => {
                    let device_plan = input.device_plan.ok_or(Error::UnsupportedCudaRequest {
                        reason: "prepared CUDA HTJ2K plan is missing normalized output geometry",
                    })?;
                    let mut append_budget = grayscale_owner_budget(
                        &plans,
                        &reports,
                        &shared_payload,
                        None,
                        "j2k CUDA referenced grayscale batch plan owners",
                    )?;
                    let plans = build_cuda_htj2k_grayscale_plans_from_referenced_with_profile(
                        input.bytes,
                        referenced,
                        fmt,
                        device_plan,
                        &mut shared_payload,
                        &mut append_budget,
                    )?;
                    (plans, true)
                }
                (None, Some(referenced)) => {
                    let device_plan = input.device_plan.ok_or(Error::UnsupportedCudaRequest {
                        reason: "prepared CUDA classic plan is missing normalized output geometry",
                    })?;
                    let mut append_budget = grayscale_owner_budget(
                        &plans,
                        &reports,
                        &shared_payload,
                        None,
                        "j2k CUDA referenced classic grayscale batch plan owners",
                    )?;
                    let plans = build_cuda_classic_grayscale_plans_from_referenced_with_profile(
                        input.bytes,
                        referenced,
                        fmt,
                        device_plan,
                        &mut shared_payload,
                        &mut append_budget,
                    )?;
                    (plans, true)
                }
                (None, None) if input.device_plan.is_some() => {
                    let device_plan = input.device_plan.expect("checked prepared device plan");
                    let (plan, report) =
                        build_cuda_htj2k_grayscale_plan_from_bytes_for_device_plan_with_profile(
                            input.bytes,
                            fmt,
                            Some(device_plan),
                            settings,
                            &mut native_context,
                        )?;
                    let mut one = initial_budget.try_vec_with_capacity(1)?;
                    one.push((plan, report));
                    (one, false)
                }
                (None, None) => {
                    let (plan, report) = build_cuda_htj2k_grayscale_plan_from_bytes_with_profile(
                        input.bytes,
                        fmt,
                        &mut native_context,
                    )?;
                    let mut one = initial_budget.try_vec_with_capacity(1)?;
                    one.push((plan, report));
                    (one, false)
                }
                (Some(_), Some(_)) => {
                    return Err(Error::UnsupportedCudaRequest {
                        reason: "prepared CUDA grayscale input contains conflicting codec plans",
                    });
                }
            };
        if input_plans.is_empty() {
            return Err(Error::UnsupportedCudaRequest {
                reason: "prepared CUDA grayscale input produced no executable tile plans",
            });
        }
        if !payload_is_shared {
            for (plan, _) in &mut input_plans {
                let mut append_budget = grayscale_owner_budget(
                    &plans,
                    &reports,
                    &shared_payload,
                    Some(plan),
                    "j2k CUDA grayscale batch plan owners",
                )?;
                plan.append_payload_to_shared_with_budget(&mut shared_payload, &mut append_budget)?;
            }
        }
        let dimensions = input
            .device_plan
            .map_or(input_plans[0].0.dimensions(), DeviceDecodePlan::output_dims);
        let mut append_budget = grayscale_owner_budget(
            &plans,
            &reports,
            &shared_payload,
            None,
            "j2k CUDA grayscale tile owner append",
        )?;
        append_budget.account_vec(&output_indices)?;
        append_budget.account_vec(&output_dimensions)?;
        append_budget.account_vec(&source_indices)?;
        append_budget.try_vec_reserve(&mut plans, input_plans.len())?;
        append_budget.try_vec_reserve(&mut reports, input_plans.len())?;
        append_budget.try_vec_reserve(&mut output_indices, input_plans.len())?;
        append_budget.try_vec_reserve(&mut source_indices, input_plans.len())?;
        for (plan, report) in input_plans {
            plans.push(plan);
            reports.push(report);
            output_indices.push(output_index);
            source_indices.push(input.source_index);
        }
        output_dimensions.push(dimensions);
    }

    Ok(PreparedGrayscaleBatch {
        plans,
        reports,
        shared_payload,
        output_indices,
        output_dimensions,
        source_indices,
    })
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
