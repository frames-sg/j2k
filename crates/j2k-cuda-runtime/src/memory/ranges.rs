// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    allocation::{try_vec_reserve, try_vec_with_capacity},
    context::CudaContext,
    driver::CuDevicePtr,
    error::CudaError,
};

use super::CudaDeviceBuffer;

const ONE_OWNING_CONTEXT_REQUIRED: &str =
    "CUDA buffer overlap comparison requires one owning context";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CheckedDeviceBufferRange {
    start: CuDevicePtr,
    end: CuDevicePtr,
    original_index: usize,
}

pub(crate) struct CheckedDeviceBufferRanges {
    context: CudaContext,
    sorted: Vec<CheckedDeviceBufferRange>,
}

fn one_owning_context_error() -> CudaError {
    CudaError::InvalidArgument {
        message: ONE_OWNING_CONTEXT_REQUIRED.to_string(),
    }
}

fn checked_nonempty_device_range(
    original_index: usize,
    (start, len): (CuDevicePtr, usize),
) -> Result<Option<CheckedDeviceBufferRange>, CudaError> {
    if len == 0 {
        return Ok(None);
    }
    let extent = u64::try_from(len).map_err(|_| CudaError::LengthTooLarge { len })?;
    let end = start
        .checked_add(extent)
        .ok_or(CudaError::LengthTooLarge { len })?;
    Ok(Some(CheckedDeviceBufferRange {
        start,
        end,
        original_index,
    }))
}

fn sort_ranges(ranges: &mut [CheckedDeviceBufferRange]) {
    ranges.sort_unstable_by_key(|range| (range.start, range.end, range.original_index));
}

fn first_self_overlap(sorted: &[CheckedDeviceBufferRange]) -> Option<(usize, usize)> {
    // In start order, any interval that reaches a later interval also reaches
    // its immediate successor, so adjacent checks are sufficient.
    sorted.windows(2).find_map(|pair| {
        let left = pair[0];
        let right = pair[1];
        (right.start < left.end).then_some((left.original_index, right.original_index))
    })
}

fn first_cross_overlap(
    left: &[CheckedDeviceBufferRange],
    right: &[CheckedDeviceBufferRange],
) -> Option<(usize, usize)> {
    // Advancing the range that ends first cannot skip a future intersection
    // because both sets are ordered by nondecreasing start address.
    let mut left_index = 0usize;
    let mut right_index = 0usize;
    while let (Some(left_range), Some(right_range)) = (left.get(left_index), right.get(right_index))
    {
        if left_range.end <= right_range.start {
            left_index += 1;
        } else if right_range.end <= left_range.start {
            right_index += 1;
        } else {
            return Some((left_range.original_index, right_range.original_index));
        }
    }
    None
}

fn checked_device_ranges_overlap(
    same_context: bool,
    left: (CuDevicePtr, usize),
    right: (CuDevicePtr, usize),
) -> Result<bool, CudaError> {
    if !same_context {
        return Err(one_owning_context_error());
    }
    let Some(left) = checked_nonempty_device_range(0, left)? else {
        return Ok(false);
    };
    let Some(right) = checked_nonempty_device_range(1, right)? else {
        return Ok(false);
    };
    Ok(left.start < right.end && right.start < left.end)
}

impl CheckedDeviceBufferRanges {
    pub(crate) fn from_same_context<'a>(
        context: &CudaContext,
        buffers: impl IntoIterator<Item = (usize, &'a CudaDeviceBuffer)>,
    ) -> Result<Self, CudaError> {
        let buffers = buffers.into_iter();
        let minimum_count = buffers.size_hint().0;
        let mut sorted = try_vec_with_capacity(minimum_count)?;
        for (original_index, buffer) in buffers {
            if !buffer.is_owned_by(context) {
                return Err(one_owning_context_error());
            }
            if let Some(range) = checked_nonempty_device_range(
                original_index,
                (buffer.device_ptr(), buffer.byte_len()),
            )? {
                // Flat-map iterators may have a zero lower size hint, so every
                // growth beyond the initial reservation must remain fallible.
                try_vec_reserve(&mut sorted, 1)?;
                sorted.push(range);
            }
        }
        sort_ranges(&mut sorted);
        Ok(Self {
            context: context.clone(),
            sorted,
        })
    }

    pub(crate) fn first_self_overlap(&self) -> Option<(usize, usize)> {
        first_self_overlap(&self.sorted)
    }

    pub(crate) fn first_cross_overlap(
        &self,
        other: &Self,
    ) -> Result<Option<(usize, usize)>, CudaError> {
        if !self.context.is_same_context(&other.context) {
            return Err(one_owning_context_error());
        }
        Ok(first_cross_overlap(&self.sorted, &other.sorted))
    }
}

impl CudaDeviceBuffer {
    pub(crate) fn overlaps(&self, other: &Self) -> Result<bool, CudaError> {
        checked_device_ranges_overlap(
            self.context.is_same_context(&other.context),
            (self.ptr, self.len),
            (other.ptr, other.len),
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::{driver::CuDevicePtr, error::CudaError};

    use super::{
        checked_device_ranges_overlap, checked_nonempty_device_range, first_cross_overlap,
        first_self_overlap, sort_ranges, CheckedDeviceBufferRange,
    };

    const LARGE_RANGE_COUNT: usize = 20_000;

    fn checked_ranges(
        ranges: impl IntoIterator<Item = (usize, CuDevicePtr, usize)>,
    ) -> Vec<CheckedDeviceBufferRange> {
        let mut checked = ranges
            .into_iter()
            .filter_map(|(index, start, len)| {
                checked_nonempty_device_range(index, (start, len)).expect("valid test range")
            })
            .collect::<Vec<_>>();
        sort_ranges(&mut checked);
        checked
    }

    fn spaced_start(index: usize, spacing: u64) -> u64 {
        u64::try_from(index).expect("test index fits u64") * spacing
    }

    #[test]
    fn cross_context_overlap_check_rejects_even_empty_or_equal_raw_ranges() {
        assert!(matches!(
            checked_device_ranges_overlap(false, (0, 0), (0, 0)),
            Err(CudaError::InvalidArgument { .. })
        ));
        assert!(matches!(
            checked_device_ranges_overlap(false, (4096, 64), (4096, 64)),
            Err(CudaError::InvalidArgument { .. })
        ));
    }

    #[test]
    fn same_context_overlap_check_uses_checked_half_open_ranges() {
        assert!(checked_device_ranges_overlap(true, (100, 16), (108, 16)).expect("overlap"));
        assert!(!checked_device_ranges_overlap(true, (100, 8), (108, 8)).expect("adjacent"));
        assert!(!checked_device_ranges_overlap(true, (100, 0), (100, 8)).expect("empty"));
        assert!(matches!(
            checked_device_ranges_overlap(true, (u64::MAX, 1), (0, 1)),
            Err(CudaError::LengthTooLarge { len: 1 })
        ));
    }

    #[test]
    fn range_sets_ignore_empty_ranges_and_preserve_original_indices() {
        let left = checked_ranges([(41, 100, 0), (17, 100, 8)]);
        let right = checked_ranges([(53, 104, 1), (61, 108, 0)]);

        assert_eq!(first_self_overlap(&left), None);
        assert_eq!(first_cross_overlap(&left, &right), Some((17, 53)));
    }

    #[test]
    fn large_disjoint_self_sweep_avoids_quadratic_pair_scanning() {
        let ranges =
            checked_ranges((0..LARGE_RANGE_COUNT).map(|index| (index, spaced_start(index, 16), 8)));

        assert_eq!(first_self_overlap(&ranges), None);
    }

    #[test]
    fn self_sweep_finds_overlap_hidden_after_many_disjoint_ranges() {
        let mut raw = (0..LARGE_RANGE_COUNT)
            .map(|index| (index, spaced_start(index, 16), 8))
            .collect::<Vec<_>>();
        raw.push((
            LARGE_RANGE_COUNT,
            spaced_start(LARGE_RANGE_COUNT - 1, 16) + 4,
            8,
        ));
        let ranges = checked_ranges(raw);

        assert_eq!(
            first_self_overlap(&ranges),
            Some((LARGE_RANGE_COUNT - 1, LARGE_RANGE_COUNT))
        );
    }

    #[test]
    fn large_disjoint_cross_sweep_avoids_quadratic_pair_scanning() {
        let left =
            checked_ranges((0..LARGE_RANGE_COUNT).map(|index| (index, spaced_start(index, 32), 8)));
        let right = checked_ranges(
            (0..LARGE_RANGE_COUNT).map(|index| (index, spaced_start(index, 32) + 8, 8)),
        );

        assert_eq!(first_cross_overlap(&left, &right), None);
    }

    #[test]
    fn cross_sweep_finds_overlap_hidden_at_end_of_large_sets() {
        let left =
            checked_ranges((0..LARGE_RANGE_COUNT).map(|index| (index, spaced_start(index, 32), 8)));
        let mut right = (0..LARGE_RANGE_COUNT)
            .map(|index| (index, spaced_start(index, 32) + 8, 8))
            .collect::<Vec<_>>();
        right.push((
            LARGE_RANGE_COUNT,
            spaced_start(LARGE_RANGE_COUNT - 1, 32) + 4,
            1,
        ));
        let right = checked_ranges(right);

        assert_eq!(
            first_cross_overlap(&left, &right),
            Some((LARGE_RANGE_COUNT - 1, LARGE_RANGE_COUNT))
        );
    }

    #[test]
    fn cross_sweep_handles_adjacent_then_nested_input_ranges() {
        let outputs = checked_ranges([(7, 100, 8)]);
        let inputs = checked_ranges([(11, 0, 100), (13, 20, 200)]);

        assert_eq!(first_cross_overlap(&outputs, &inputs), Some((7, 13)));
    }
}
