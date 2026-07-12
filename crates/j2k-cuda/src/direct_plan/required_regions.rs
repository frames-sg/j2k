// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_native::{
    idwt_required_input_windows, idwt_required_output_margin, J2kDirectGrayscalePlan,
    J2kDirectGrayscaleStep, J2kDirectIdwtStep, J2kRequiredBandRegion as RequiredBandRegion,
};

use super::{CudaHtj2kBandId, Error, PLAN_OUTPUT_RECT_MISMATCH, PLAN_PAYLOAD_TOO_LARGE};
use crate::allocation::{try_vec_reserve, try_vec_with_capacity, HostPhaseBudget};

const REQUIRED_REGIONS: &str = "CUDA direct-plan required regions";
const DIRECT_PLAN_OWNERS: &str = "CUDA direct-plan owner graph";

pub(super) struct RequiredBandRegions {
    entries: Vec<(CudaHtj2kBandId, RequiredBandRegion)>,
}

impl RequiredBandRegions {
    pub(super) fn get(&self, band_id: CudaHtj2kBandId) -> Option<&RequiredBandRegion> {
        self.entries
            .binary_search_by_key(&band_id, |(existing, _)| *existing)
            .ok()
            .map(|index| &self.entries[index].1)
    }
}

pub(super) fn required_regions_for_direct_plan(
    plan: &J2kDirectGrayscalePlan,
    retained_plan_bytes: usize,
) -> Result<RequiredBandRegions, Error> {
    let capacity = required_region_capacity(plan)?;
    let entries = try_vec_with_capacity(capacity, REQUIRED_REGIONS)?;
    let mut budget = HostPhaseBudget::new(DIRECT_PLAN_OWNERS);
    budget.account_bytes(retained_plan_bytes)?;
    budget.account_capacity::<(CudaHtj2kBandId, RequiredBandRegion)>(entries.capacity())?;
    let mut required = RequiredBandRegions { entries };

    for step in &plan.steps {
        let J2kDirectGrayscaleStep::Store(store) = step else {
            continue;
        };
        let source_right =
            store
                .source_x
                .checked_add(store.copy_width)
                .ok_or(Error::UnsupportedCudaRequest {
                    reason: PLAN_OUTPUT_RECT_MISMATCH,
                })?;
        let source_bottom =
            store
                .source_y
                .checked_add(store.copy_height)
                .ok_or(Error::UnsupportedCudaRequest {
                    reason: PLAN_OUTPUT_RECT_MISMATCH,
                })?;
        if let Some(region) =
            RequiredBandRegion::new(store.source_x, store.source_y, source_right, source_bottom)
        {
            add_required_region(&mut required, store.input_band_id, region)?;
        }
    }

    for step in plan.steps.iter().rev() {
        let J2kDirectGrayscaleStep::Idwt(idwt) = step else {
            continue;
        };
        let Some(output_region) = required.get(idwt.output_band_id).copied() else {
            continue;
        };
        let expanded = output_region.expanded_within_band(
            idwt_required_output_margin(idwt.transform),
            idwt.rect.width(),
            idwt.rect.height(),
        );
        add_idwt_input_required_regions(&mut required, idwt, expanded)?;
    }
    Ok(required)
}

fn required_region_capacity(plan: &J2kDirectGrayscalePlan) -> Result<usize, Error> {
    plan.steps.iter().try_fold(0usize, |capacity, step| {
        let additional = match step {
            J2kDirectGrayscaleStep::Store(_) => 1,
            J2kDirectGrayscaleStep::Idwt(_) => 4,
            J2kDirectGrayscaleStep::HtSubBand(_) | J2kDirectGrayscaleStep::ClassicSubBand(_) => 0,
        };
        capacity
            .checked_add(additional)
            .ok_or(Error::UnsupportedCudaRequest {
                reason: PLAN_PAYLOAD_TOO_LARGE,
            })
    })
}

fn add_required_region(
    required: &mut RequiredBandRegions,
    band_id: CudaHtj2kBandId,
    region: RequiredBandRegion,
) -> Result<(), Error> {
    match required
        .entries
        .binary_search_by_key(&band_id, |(existing, _)| *existing)
    {
        Ok(index) => required.entries[index].1 = required.entries[index].1.union(region),
        Err(index) => {
            try_vec_reserve(&mut required.entries, 1, REQUIRED_REGIONS)?;
            required.entries.insert(index, (band_id, region));
        }
    }
    Ok(())
}

fn add_idwt_input_required_regions(
    required: &mut RequiredBandRegions,
    idwt: &J2kDirectIdwtStep,
    output_region: RequiredBandRegion,
) -> Result<(), Error> {
    let windows = idwt_required_input_windows(idwt, output_region);
    add_required_region(required, idwt.ll_band_id, windows.ll)?;
    add_required_region(required, idwt.hl_band_id, windows.hl)?;
    add_required_region(required, idwt.lh_band_id, windows.lh)?;
    add_required_region(required, idwt.hh_band_id, windows.hh)
}

#[cfg(test)]
mod tests {
    use crate::allocation::HostPhaseBudget;
    use crate::Error;

    #[test]
    fn direct_plan_actual_capacities_accept_exact_cap_and_reject_one_over() {
        let mut exact = HostPhaseBudget::with_cap("test direct plan", 16);
        exact.account_bytes(12).expect("retained plan fits");
        exact.account_capacity::<u32>(1).expect("exact cap fits");

        let mut one_over = HostPhaseBudget::with_cap("test direct plan", 16);
        one_over.account_bytes(12).expect("retained plan fits");
        assert!(matches!(
            one_over.account_capacity::<u32>(2),
            Err(Error::HostAllocationTooLarge {
                requested: 20,
                cap: 16,
                what: "test direct plan",
            })
        ));
    }
}
