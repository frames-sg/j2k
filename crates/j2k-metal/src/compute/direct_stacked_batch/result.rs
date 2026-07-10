// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::{record_hybrid_stacked_component_batch, DirectTier1Mode, Error};
use super::resources::StackedComponentResources;
use super::validation::StackedComponentBatchPlan;
use super::StackedDirectComponentPlane;

pub(super) fn assemble_stacked_component_result(
    resources: StackedComponentResources,
    plan: &StackedComponentBatchPlan<'_>,
    tier1_mode: DirectTier1Mode,
) -> Result<StackedDirectComponentPlane, Error> {
    let buffer = resources.final_plane.ok_or_else(|| Error::MetalKernel {
        message: "J2K MetalDirect color component batch did not produce a final plane".to_string(),
    })?;
    record_hybrid_stacked_component_batch(tier1_mode);
    Ok(StackedDirectComponentPlane {
        buffer,
        dimensions: plan.first.dimensions,
        count: plan.count,
    })
}
