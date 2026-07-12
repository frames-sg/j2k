// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    allocation::{host_allocation_error, HostPhaseBudget},
    error::CudaError,
};
use std::{cmp::Reverse, collections::BinaryHeap};

use super::Htj2kOutputRegion;

fn sweep_state_error(message: &'static str) -> CudaError {
    CudaError::StatePoisoned {
        message: message.to_string(),
    }
}

pub(super) fn validate_cross_stride_spans(
    regions: &mut [Htj2kOutputRegion],
    live_region_bytes: usize,
) -> Result<(), CudaError> {
    regions.sort_unstable_by_key(|region| (region.linear_start, region.linear_end, region.stride));
    let mut host_budget = HostPhaseBudget::with_live_bytes(
        "CUDA HTJ2K cross-stride output sweep",
        live_region_bytes,
    )?;
    let mut active = BinaryHeap::new();
    active
        .try_reserve(regions.len())
        .map_err(|_| host_allocation_error::<Reverse<(usize, usize)>>(regions.len()))?;
    host_budget.account_capacity::<Reverse<(usize, usize)>>(active.capacity())?;
    let mut active_by_stride: Vec<(usize, usize)> =
        host_budget.try_vec_with_capacity(regions.len())?;
    let mut active_count = 0usize;
    for region in regions {
        while active
            .peek()
            .is_some_and(|Reverse((end, _))| *end <= region.linear_start)
        {
            let Some(Reverse((_, stride))) = active.pop() else {
                break;
            };
            active_count = active_count
                .checked_sub(1)
                .ok_or_else(|| sweep_state_error("HTJ2K output sweep active count underflow"))?;
            let Ok(stride_index) =
                active_by_stride.binary_search_by_key(&stride, |(stride, _)| *stride)
            else {
                return Err(sweep_state_error(
                    "HTJ2K output sweep lost an active stride count",
                ));
            };
            let count = &mut active_by_stride[stride_index].1;
            *count = count
                .checked_sub(1)
                .ok_or_else(|| sweep_state_error("HTJ2K output sweep stride count underflow"))?;
            if *count == 0 {
                active_by_stride.remove(stride_index);
            }
        }
        let stride_index =
            active_by_stride.binary_search_by_key(&region.stride, |(stride, _)| *stride);
        let same_stride_count = stride_index
            .ok()
            .map_or(0, |index| active_by_stride[index].1);
        if active_count != same_stride_count {
            return Err(CudaError::InvalidArgument {
                message: "different-stride HTJ2K output spans must be disjoint".to_string(),
            });
        }
        active.push(Reverse((region.linear_end, region.stride)));
        active_count = active_count
            .checked_add(1)
            .ok_or_else(|| sweep_state_error("HTJ2K output sweep active count overflow"))?;
        let count = match stride_index {
            Ok(index) => &mut active_by_stride[index].1,
            Err(index) => {
                active_by_stride.insert(index, (region.stride, 0));
                &mut active_by_stride[index].1
            }
        };
        *count = count
            .checked_add(1)
            .ok_or_else(|| sweep_state_error("HTJ2K output sweep stride count overflow"))?;
    }
    Ok(())
}
