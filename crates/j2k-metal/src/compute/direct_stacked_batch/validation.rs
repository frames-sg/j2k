// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::{
    classic_group_shapes_match, classic_sub_band_shapes_match, ht_group_shapes_match,
    ht_sub_band_shapes_match, idwt_shapes_match, store_shapes_match, DirectTier1Mode, Error,
    PreparedDirectGrayscalePlan, PreparedDirectGrayscaleStep,
};

pub(super) struct StackedComponentBatchPlan<'p> {
    pub(super) first: &'p PreparedDirectGrayscalePlan,
    pub(super) count: usize,
    pub(super) broadcast_tier1_inputs: bool,
}

pub(super) fn plan_stacked_component_batch<'p>(
    plans: &[&'p PreparedDirectGrayscalePlan],
    tier1_mode: DirectTier1Mode,
) -> Result<StackedComponentBatchPlan<'p>, Error> {
    let Some(first) = plans.first().copied() else {
        return Err(Error::MetalKernel {
            message: "J2K MetalDirect color batch has no component plans".to_string(),
        });
    };

    Ok(StackedComponentBatchPlan {
        first,
        count: plans.len(),
        broadcast_tier1_inputs: tier1_mode == DirectTier1Mode::CpuUpload
            && plans.iter().all(|plan| std::ptr::eq(*plan, first)),
    })
}

pub(in super::super) fn supports_stacked_direct_component_plane_batch(
    plans: &[&PreparedDirectGrayscalePlan],
) -> bool {
    let Some(first) = plans.first() else {
        return false;
    };
    if plans.iter().any(|plan| {
        plan.dimensions != first.dimensions
            || plan.bit_depth != first.bit_depth
            || plan.steps.len() != first.steps.len()
    }) {
        return false;
    }

    let mut step_idx = 0;
    while step_idx < first.steps.len() {
        if let Some(group) = first.classic_group_starting_at(step_idx) {
            if group.end_step <= step_idx
                || !plans.iter().all(|plan| {
                    plan.classic_group_starting_at(step_idx)
                        .is_some_and(|other| classic_group_shapes_match(group, other))
                })
            {
                return false;
            }
            step_idx = group.end_step;
            continue;
        }
        if let Some(group) = first.ht_group_starting_at(step_idx) {
            if group.end_step <= step_idx
                || !plans.iter().all(|plan| {
                    plan.ht_group_starting_at(step_idx)
                        .is_some_and(|other| ht_group_shapes_match(group, other))
                })
            {
                return false;
            }
            step_idx = group.end_step;
            continue;
        }

        match &first.steps[step_idx] {
            PreparedDirectGrayscaleStep::ClassicSubBand(sub_band) => {
                if !plans.iter().all(|plan| {
                    matches!(
                        &plan.steps[step_idx],
                        PreparedDirectGrayscaleStep::ClassicSubBand(other)
                            if classic_sub_band_shapes_match(sub_band, other)
                    )
                }) {
                    return false;
                }
            }
            PreparedDirectGrayscaleStep::HtSubBand(sub_band) => {
                if !plans.iter().all(|plan| {
                    matches!(
                        &plan.steps[step_idx],
                        PreparedDirectGrayscaleStep::HtSubBand(other)
                            if ht_sub_band_shapes_match(sub_band, other)
                    )
                }) {
                    return false;
                }
            }
            PreparedDirectGrayscaleStep::Idwt(idwt) => {
                if !plans.iter().all(|plan| {
                    matches!(
                        &plan.steps[step_idx],
                        PreparedDirectGrayscaleStep::Idwt(other)
                            if idwt_shapes_match(idwt, other)
                    )
                }) {
                    return false;
                }
            }
            PreparedDirectGrayscaleStep::Store(store) => {
                if !plans.iter().all(|plan| {
                    matches!(
                        &plan.steps[step_idx],
                        PreparedDirectGrayscaleStep::Store(other)
                            if store_shapes_match(store, other)
                    )
                }) {
                    return false;
                }
            }
        }
        step_idx += 1;
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_component_batch_preserves_validation_error() {
        let Err(error) = plan_stacked_component_batch(&[], DirectTier1Mode::Metal) else {
            panic!("empty component batch must fail validation");
        };

        assert!(matches!(
            error,
            Error::MetalKernel { message }
                if message == "J2K MetalDirect color batch has no component plans"
        ));
    }
}
