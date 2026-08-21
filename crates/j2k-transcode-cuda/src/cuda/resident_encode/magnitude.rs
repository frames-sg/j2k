// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::{checked_element_product, CudaTranscodeError, HostPhaseBudget};
use super::planning::{ResidentSubbandEncodePlan, ResidentSubbandGroupPlans};

pub(super) fn required_magnitude_bounds_for_plans(
    item_count: usize,
    plans: &[ResidentSubbandEncodePlan<'_>],
    maxima: impl Iterator<Item = u32>,
    budget: &mut HostPhaseBudget,
) -> Result<Vec<u8>, CudaTranscodeError> {
    if item_count == 0 {
        return Err(CudaTranscodeError::Kernel(
            "CUDA resident HTJ2K magnitude aggregation received no items",
        ));
    }
    let mut maxima = maxima;
    let bounds = required_magnitude_bounds_for_items(
        item_count,
        plans.iter().map(|plan| plan.jobs.len() / item_count),
        &mut maxima,
        1,
        budget,
    )?;
    require_all_maxima_consumed(&mut maxima)?;
    Ok(bounds)
}

pub(super) fn required_magnitude_bounds_for_groups<J>(
    groups: &[ResidentSubbandGroupPlans<'_, J>],
    maxima: impl Iterator<Item = u32>,
    budget: &mut HostPhaseBudget,
) -> Result<Vec<Vec<u8>>, CudaTranscodeError> {
    let mut maxima = maxima;
    let mut grouped = budget.try_vec_with_capacity_named(
        groups.len(),
        "CUDA grouped resident magnitude-bound outputs",
    )?;
    for group in groups {
        let item_count = group.jobs.len();
        if item_count == 0 {
            return Err(CudaTranscodeError::Kernel(
                "CUDA grouped resident HTJ2K magnitude aggregation received an empty group",
            ));
        }
        grouped.push(required_magnitude_bounds_for_items(
            item_count,
            group
                .plans()
                .into_iter()
                .map(|plan| plan.jobs.len() / item_count),
            &mut maxima,
            1,
            budget,
        )?);
    }
    require_all_maxima_consumed(&mut maxima)?;
    Ok(grouped)
}

fn require_all_maxima_consumed(
    maxima: &mut impl Iterator<Item = u32>,
) -> Result<(), CudaTranscodeError> {
    if maxima.next().is_some() {
        return Err(CudaTranscodeError::Kernel(
            "CUDA resident HTJ2K magnitude status count mismatch",
        ));
    }
    Ok(())
}

pub(super) fn required_magnitude_bounds_for_items(
    item_count: usize,
    blocks_per_item_by_subband: impl IntoIterator<Item = usize>,
    maxima: &mut impl Iterator<Item = u32>,
    decomposition_level: u8,
    budget: &mut HostPhaseBudget,
) -> Result<Vec<u8>, CudaTranscodeError> {
    let mut required = budget
        .try_vec_with_capacity_named(item_count, "CUDA resident HTJ2K magnitude-bound outputs")?;
    required.resize(item_count, 8);

    for blocks_per_item in blocks_per_item_by_subband {
        checked_element_product(
            &[item_count, blocks_per_item],
            "CUDA resident HTJ2K magnitude-bound status count",
        )?;
        for item_required in &mut required {
            let mut maximum = 0_u32;
            for _ in 0..blocks_per_item {
                maximum = maximum.max(maxima.next().ok_or(CudaTranscodeError::Kernel(
                    "CUDA resident HTJ2K magnitude status count mismatch",
                ))?);
            }
            *item_required = (*item_required).max(j2k_native::htj2k_required_magnitude_bound(
                u64::from(maximum),
                false,
                decomposition_level,
            ));
        }
    }

    Ok(required)
}

#[cfg(test)]
mod tests {
    use super::required_magnitude_bounds_for_items;
    use crate::cuda::allocation::HostPhaseBudget;

    #[test]
    fn resident_bounds_preserve_each_items_observed_cleanup_maximum() {
        let mut budget = HostPhaseBudget::new("resident magnitude-bound test");
        let mut maxima = [259_u32, 4, 32, 8, 1, 0].into_iter();

        let bounds = required_magnitude_bounds_for_items(2, [2, 1], &mut maxima, 1, &mut budget)
            .expect("aggregate resident magnitude bounds");

        assert_eq!(bounds, [9, 8]);
        assert_eq!(maxima.next(), None);
    }
}
