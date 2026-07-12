// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    allocation::{host_allocation_error, HostPhaseBudget},
    error::CudaError,
};
use std::{cmp::Reverse, collections::BinaryHeap};

mod cross_stride;

use self::cross_stride::validate_cross_stride_spans;

#[derive(Clone, Copy)]
pub(super) struct Htj2kOutputRect {
    pub(super) row_start: usize,
    pub(super) row_end: usize,
    pub(super) column_start: usize,
    pub(super) column_end: usize,
}

#[derive(Clone, Copy)]
pub(super) struct Htj2kOutputRegion {
    pub(super) stride: usize,
    pub(super) rect: Htj2kOutputRect,
    pub(super) linear_start: usize,
    pub(super) linear_end: usize,
}

fn validate_same_stride_rects(
    regions: &[Htj2kOutputRegion],
    live_region_bytes: usize,
) -> Result<(), CudaError> {
    let mut host_budget =
        HostPhaseBudget::with_live_bytes("CUDA HTJ2K same-stride output sweep", live_region_bytes)?;
    let mut active_by_row_end = BinaryHeap::new();
    active_by_row_end
        .try_reserve(regions.len())
        .map_err(|_| host_allocation_error::<Reverse<(usize, usize)>>(regions.len()))?;
    host_budget.account_capacity::<Reverse<(usize, usize)>>(active_by_row_end.capacity())?;
    let mut active_by_column: Vec<(usize, (usize, usize))> =
        host_budget.try_vec_with_capacity(regions.len())?;
    for region in regions {
        let rect = region.rect;
        while active_by_row_end
            .peek()
            .is_some_and(|Reverse((row_end, _))| *row_end <= rect.row_start)
        {
            let Some(Reverse((row_end, column_start))) = active_by_row_end.pop() else {
                break;
            };
            if let Ok(index) = active_by_column
                .binary_search_by_key(&column_start, |(column_start, _)| *column_start)
            {
                if active_by_column[index].1 .1 == row_end {
                    active_by_column.remove(index);
                }
            }
        }

        // Successful insertions keep active column intervals mutually
        // disjoint, so the greatest start below `column_end` is sufficient.
        let insertion_index =
            active_by_column.partition_point(|(column_start, _)| *column_start < rect.column_end);
        let overlaps =
            insertion_index != 0 && active_by_column[insertion_index - 1].1 .0 > rect.column_start;
        if overlaps {
            return Err(CudaError::InvalidArgument {
                message: "HTJ2K jobs sharing one output must write disjoint regions".to_string(),
            });
        }
        let insertion_index =
            active_by_column.partition_point(|(column_start, _)| *column_start < rect.column_start);
        active_by_column.insert(
            insertion_index,
            (rect.column_start, (rect.column_end, rect.row_end)),
        );
        active_by_row_end.push(Reverse((rect.row_end, rect.column_start)));
    }
    Ok(())
}

pub(super) fn validate_disjoint_output_regions(
    regions: &mut [Htj2kOutputRegion],
    live_region_bytes: usize,
) -> Result<(), CudaError> {
    validate_cross_stride_spans(regions, live_region_bytes)?;
    regions.sort_unstable_by_key(|region| {
        let rect = region.rect;
        (
            region.stride,
            rect.row_start,
            rect.row_end,
            rect.column_start,
            rect.column_end,
        )
    });
    let mut start = 0usize;
    while start < regions.len() {
        let stride = regions[start].stride;
        let end = regions[start..]
            .iter()
            .position(|region| region.stride != stride)
            .map_or(regions.len(), |offset| start + offset);
        validate_same_stride_rects(&regions[start..end], live_region_bytes)?;
        start = end;
    }
    Ok(())
}
