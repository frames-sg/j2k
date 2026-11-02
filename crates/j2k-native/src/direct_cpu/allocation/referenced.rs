// SPDX-License-Identifier: MIT OR Apache-2.0

//! Scratch sizing and retained workspace accounting for referenced plans.

mod budget;
mod storage;
mod view;

pub(in super::super) use view::{max_referenced_classic_dimensions, max_referenced_ht_dimensions};

use super::{
    ClassicTaskShape, DecodeError, DirectWorkspaceBudget, J2kDirectCpuScratch,
    J2kReferencedClassicPlan, J2kReferencedHtj2kPlan, Result, ValidationError,
};
use budget::validate_referenced_aggregate_plan;
use storage::{normalize_referenced_scratch, reserve_referenced_scratch};
use view::{
    max_referenced_classic_fragmented_payload_bytes, max_referenced_classic_payload_bytes,
    max_referenced_classic_rows, max_referenced_dimensions, max_referenced_payload_bytes,
    ReferencedPlanView,
};

/// Accounting inputs shared by an estimator and the matching reservation.
#[derive(Clone, Copy)]
struct ReferencedRequirements<'a> {
    plan: ReferencedPlanView<'a>,
    parallel: bool,
    retained_plan_bytes: usize,
    compressed_payload_bytes: usize,
    retained_classic_workspace_dimensions: Option<(u32, u32)>,
    retained_ht_workspace_dimensions: Option<(u32, u32)>,
    classic_tasks: Option<ClassicTaskShape>,
}

impl<'a> ReferencedRequirements<'a> {
    fn htj2k(plan: &'a J2kReferencedHtj2kPlan, parallel: bool) -> Result<Self> {
        let view = ReferencedPlanView::Htj2k(plan);
        Ok(Self {
            plan: view,
            parallel,
            retained_plan_bytes: plan.retained_allocation_bytes()?,
            compressed_payload_bytes: max_referenced_payload_bytes(plan)?,
            retained_classic_workspace_dimensions: max_referenced_dimensions(view, false),
            retained_ht_workspace_dimensions: max_referenced_dimensions(view, true),
            classic_tasks: None,
        })
    }

    fn classic(plan: &'a J2kReferencedClassicPlan, parallel: bool) -> Result<Self> {
        let dimensions = max_referenced_classic_dimensions(plan);
        Ok(Self {
            plan: ReferencedPlanView::Classic(plan),
            parallel,
            retained_plan_bytes: plan.retained_allocation_bytes()?,
            compressed_payload_bytes: max_referenced_classic_payload_bytes(plan),
            retained_classic_workspace_dimensions: dimensions,
            retained_ht_workspace_dimensions: None,
            classic_tasks: classic_task_shape(plan),
        })
    }

    fn staged_htj2k(plan: &'a J2kReferencedHtj2kPlan) -> Result<Self> {
        Ok(Self {
            plan: ReferencedPlanView::Htj2k(plan),
            parallel: false,
            retained_plan_bytes: plan.retained_allocation_bytes()?,
            compressed_payload_bytes: 0,
            retained_classic_workspace_dimensions: None,
            retained_ht_workspace_dimensions: None,
            classic_tasks: None,
        })
    }

    fn staged_classic(plan: &'a J2kReferencedClassicPlan) -> Result<Self> {
        Ok(Self {
            plan: ReferencedPlanView::Classic(plan),
            parallel: false,
            retained_plan_bytes: plan.retained_allocation_bytes()?,
            compressed_payload_bytes: 0,
            retained_classic_workspace_dimensions: None,
            retained_ht_workspace_dimensions: None,
            classic_tasks: None,
        })
    }

    fn validate(self, scratch: &J2kDirectCpuScratch) -> Result<DirectWorkspaceBudget> {
        validate_referenced_aggregate_plan(
            self.plan,
            self.parallel,
            self.retained_plan_bytes,
            self.compressed_payload_bytes,
            self.retained_classic_workspace_dimensions,
            self.retained_ht_workspace_dimensions,
            self.classic_tasks,
            scratch,
        )
    }
}

pub(in super::super) fn prepare_referenced_direct_scratch(
    plan: &J2kReferencedHtj2kPlan,
    scratch: &mut J2kDirectCpuScratch,
) -> Result<DirectWorkspaceBudget> {
    prepare_referenced_component_scratch(
        ReferencedRequirements::htj2k(plan, scratch.parallel)?,
        scratch,
    )
}

/// Planned peak bytes of one referenced HTJ2K execution into empty scratch,
/// from the accounting execution enforces, without allocating.
pub(in super::super) fn planned_referenced_htj2k_execution_bytes(
    plan: &J2kReferencedHtj2kPlan,
    parallel: bool,
) -> Result<usize> {
    let mut scratch = J2kDirectCpuScratch::new();
    scratch.parallel = parallel;
    ReferencedRequirements::htj2k(plan, parallel)?
        .validate(&scratch)
        .map(DirectWorkspaceBudget::peak_bytes)
}

/// Planned peak bytes of one referenced classic execution into empty
/// scratch, from the accounting execution enforces, without allocating.
pub(in super::super) fn planned_referenced_classic_execution_bytes(
    plan: &J2kReferencedClassicPlan,
    parallel: bool,
) -> Result<usize> {
    let mut scratch = J2kDirectCpuScratch::new();
    scratch.parallel = parallel;
    ReferencedRequirements::classic(plan, parallel)?
        .validate(&scratch)
        .map(DirectWorkspaceBudget::peak_bytes)
}

/// Shape of the extra classic workspaces parallel execution of `plan` uses.
fn classic_task_shape(plan: &J2kReferencedClassicPlan) -> Option<ClassicTaskShape> {
    Some(ClassicTaskShape {
        dimensions: max_referenced_classic_dimensions(plan)?,
        rows: max_referenced_classic_rows(plan),
        fragmented_payload_bytes: max_referenced_classic_fragmented_payload_bytes(plan),
    })
}

pub(in super::super) fn prepare_referenced_classic_scratch(
    plan: &J2kReferencedClassicPlan,
    scratch: &mut J2kDirectCpuScratch,
) -> Result<DirectWorkspaceBudget> {
    prepare_referenced_component_scratch(
        ReferencedRequirements::classic(plan, scratch.parallel)?,
        scratch,
    )
}

pub(in super::super) fn prepare_referenced_htj2k_staged_scratch(
    plan: &J2kReferencedHtj2kPlan,
    scratch: &mut J2kDirectCpuScratch,
) -> Result<DirectWorkspaceBudget> {
    prepare_referenced_component_scratch(ReferencedRequirements::staged_htj2k(plan)?, scratch)
}

pub(in super::super) fn prepare_referenced_classic_staged_scratch(
    plan: &J2kReferencedClassicPlan,
    scratch: &mut J2kDirectCpuScratch,
) -> Result<DirectWorkspaceBudget> {
    prepare_referenced_component_scratch(ReferencedRequirements::staged_classic(plan)?, scratch)
}

fn prepare_referenced_component_scratch(
    requirements: ReferencedRequirements<'_>,
    scratch: &mut J2kDirectCpuScratch,
) -> Result<DirectWorkspaceBudget> {
    normalize_referenced_scratch(
        requirements.plan,
        requirements.compressed_payload_bytes,
        requirements.retained_classic_workspace_dimensions.is_some(),
        requirements.retained_ht_workspace_dimensions.is_some(),
        requirements.classic_tasks.is_some(),
        scratch,
    )?;
    if let Err(error) = requirements.validate(scratch) {
        if !matches!(
            error,
            DecodeError::Validation(ValidationError::ImageTooLarge)
        ) {
            return Err(error);
        }
        scratch.clear();
        requirements.validate(scratch)?;
    }

    if let Err(error) = reserve_referenced_scratch(
        requirements.plan,
        requirements.compressed_payload_bytes,
        requirements.retained_classic_workspace_dimensions,
        requirements.retained_ht_workspace_dimensions,
        requirements.classic_tasks,
        scratch,
    ) {
        scratch.clear();
        return Err(error);
    }
    match requirements.validate(scratch) {
        Ok(workspace_budget) => Ok(workspace_budget),
        Err(error) => {
            scratch.clear();
            Err(error)
        }
    }
}
