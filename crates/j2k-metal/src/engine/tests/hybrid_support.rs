// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::{
    PreparedDirectColorPlan, PreparedDirectGrayscalePlan, PreparedDirectGrayscaleStep,
};
use std::sync::Mutex;

pub(super) static HYBRID_COUNTER_TEST_LOCK: Mutex<()> = Mutex::new(());

pub(super) fn prepared_direct_color_tier1_input_count(plan: &PreparedDirectColorPlan) -> usize {
    plan.component_plans
        .iter()
        .map(prepared_direct_component_tier1_input_count)
        .sum()
}

fn prepared_direct_component_tier1_input_count(plan: &PreparedDirectGrayscalePlan) -> usize {
    let mut count = 0;
    let mut step_idx = 0;
    while step_idx < plan.steps.len() {
        if let Some(group) = plan.classic_group_starting_at(step_idx) {
            count += 1;
            step_idx = group.end_step;
            continue;
        }
        if let Some(group) = plan.ht_group_starting_at(step_idx) {
            count += 1;
            step_idx = group.end_step;
            continue;
        }
        if matches!(
            &plan.steps[step_idx],
            PreparedDirectGrayscaleStep::ClassicSubBand(_)
                | PreparedDirectGrayscaleStep::HtSubBand(_)
        ) {
            count += 1;
        }
        step_idx += 1;
    }
    count
}

pub(super) fn cached_direct_color_tier1_input_count(plan: &PreparedDirectColorPlan) -> usize {
    plan.component_plans
        .iter()
        .map(cached_direct_component_tier1_input_count)
        .sum()
}

fn cached_direct_component_tier1_input_count(plan: &PreparedDirectGrayscalePlan) -> usize {
    let mut count = 0;
    let mut step_idx = 0;
    while step_idx < plan.steps.len() {
        if let Some(group) = plan.classic_group_starting_at(step_idx) {
            if has_cached_cpu_tier1_coefficients(plan, step_idx, group.total_coefficients) {
                count += 1;
            }
            step_idx = group.end_step;
            continue;
        }
        if let Some(group) = plan.ht_group_starting_at(step_idx) {
            if has_cached_cpu_tier1_coefficients(plan, step_idx, group.total_coefficients) {
                count += 1;
            }
            step_idx = group.end_step;
            continue;
        }
        match &plan.steps[step_idx] {
            PreparedDirectGrayscaleStep::ClassicSubBand(sub_band) => {
                let output_len = sub_band.width as usize * sub_band.height as usize;
                if has_cached_cpu_tier1_coefficients(plan, step_idx, output_len) {
                    count += 1;
                }
            }
            PreparedDirectGrayscaleStep::HtSubBand(sub_band) => {
                let output_len = sub_band.width as usize * sub_band.height as usize;
                if has_cached_cpu_tier1_coefficients(plan, step_idx, output_len) {
                    count += 1;
                }
            }
            PreparedDirectGrayscaleStep::Idwt(_) | PreparedDirectGrayscaleStep::Store(_) => {}
        }
        step_idx += 1;
    }
    count
}

fn has_cached_cpu_tier1_coefficients(
    plan: &PreparedDirectGrayscalePlan,
    step_idx: usize,
    output_len: usize,
) -> bool {
    let mut budget =
        crate::batch_allocation::BatchMetadataBudget::new("J2K Metal test CPU Tier-1 cache lookup");
    plan.cached_cpu_tier1_coefficients(&mut budget, step_idx, output_len)
        .expect("CPU Tier-1 cache lookup")
        .is_some()
}
