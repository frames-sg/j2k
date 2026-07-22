// SPDX-License-Identifier: MIT OR Apache-2.0

//! Scratch sizing and retained workspace accounting for referenced plans.

mod budget;
mod storage;
mod view;

pub(in super::super) use view::{max_referenced_classic_dimensions, max_referenced_ht_dimensions};

use super::{
    DecodeError, DirectWorkspaceBudget, J2kDirectCpuScratch, J2kReferencedClassicPlan,
    J2kReferencedHtj2kPlan, Result, ValidationError,
};
use budget::validate_referenced_aggregate_plan;
use storage::{normalize_referenced_scratch, reserve_referenced_scratch};
use view::{
    max_referenced_classic_payload_bytes, max_referenced_payload_bytes, ReferencedPlanView,
};

pub(in super::super) fn prepare_referenced_direct_scratch(
    plan: &J2kReferencedHtj2kPlan,
    scratch: &mut J2kDirectCpuScratch,
) -> Result<DirectWorkspaceBudget> {
    let view = ReferencedPlanView::Htj2k(plan);
    prepare_referenced_component_scratch(
        view,
        plan.retained_allocation_bytes()?,
        max_referenced_payload_bytes(plan)?,
        None,
        max_referenced_ht_dimensions(plan),
        scratch,
    )
}

pub(in super::super) fn prepare_referenced_classic_scratch(
    plan: &J2kReferencedClassicPlan,
    scratch: &mut J2kDirectCpuScratch,
) -> Result<DirectWorkspaceBudget> {
    let view = ReferencedPlanView::Classic(plan);
    prepare_referenced_component_scratch(
        view,
        plan.retained_allocation_bytes()?,
        max_referenced_classic_payload_bytes(plan),
        max_referenced_classic_dimensions(plan),
        None,
        scratch,
    )
}

pub(in super::super) fn prepare_referenced_htj2k_staged_scratch(
    plan: &J2kReferencedHtj2kPlan,
    scratch: &mut J2kDirectCpuScratch,
) -> Result<DirectWorkspaceBudget> {
    prepare_referenced_component_scratch(
        ReferencedPlanView::Htj2k(plan),
        plan.retained_allocation_bytes()?,
        0,
        None,
        None,
        scratch,
    )
}

pub(in super::super) fn prepare_referenced_classic_staged_scratch(
    plan: &J2kReferencedClassicPlan,
    scratch: &mut J2kDirectCpuScratch,
) -> Result<DirectWorkspaceBudget> {
    prepare_referenced_component_scratch(
        ReferencedPlanView::Classic(plan),
        plan.retained_allocation_bytes()?,
        0,
        None,
        None,
        scratch,
    )
}

fn prepare_referenced_component_scratch(
    plan: ReferencedPlanView<'_>,
    retained_plan_bytes: usize,
    compressed_payload_bytes: usize,
    retained_classic_workspace_dimensions: Option<(u32, u32)>,
    retained_ht_workspace_dimensions: Option<(u32, u32)>,
    scratch: &mut J2kDirectCpuScratch,
) -> Result<DirectWorkspaceBudget> {
    normalize_referenced_scratch(
        plan,
        compressed_payload_bytes,
        retained_classic_workspace_dimensions.is_some(),
        retained_ht_workspace_dimensions.is_some(),
        scratch,
    )?;
    if let Err(error) = validate_referenced_aggregate_plan(
        plan,
        retained_plan_bytes,
        compressed_payload_bytes,
        retained_classic_workspace_dimensions,
        retained_ht_workspace_dimensions,
        scratch,
    ) {
        if !matches!(
            error,
            DecodeError::Validation(ValidationError::ImageTooLarge)
        ) {
            return Err(error);
        }
        scratch.clear();
        validate_referenced_aggregate_plan(
            plan,
            retained_plan_bytes,
            compressed_payload_bytes,
            retained_classic_workspace_dimensions,
            retained_ht_workspace_dimensions,
            scratch,
        )?;
    }

    if let Err(error) = reserve_referenced_scratch(
        plan,
        compressed_payload_bytes,
        retained_classic_workspace_dimensions,
        retained_ht_workspace_dimensions,
        scratch,
    ) {
        scratch.clear();
        return Err(error);
    }
    match validate_referenced_aggregate_plan(
        plan,
        retained_plan_bytes,
        compressed_payload_bytes,
        retained_classic_workspace_dimensions,
        retained_ht_workspace_dimensions,
        scratch,
    ) {
        Ok(workspace_budget) => Ok(workspace_budget),
        Err(error) => {
            scratch.clear();
            Err(error)
        }
    }
}
