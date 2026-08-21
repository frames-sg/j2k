// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    allocation::HostPhaseBudget,
    error::CudaError,
    memory::{checked_image_words, CudaDeviceBufferRange},
};

use super::destination::validate_store_destination;

#[derive(Clone, Copy, Debug)]
pub(super) struct DenseDestinationRegion {
    pub(super) output_index: usize,
    pub(super) output_width: u32,
    pub(super) output_height: u32,
    pub(super) output_x: u32,
    pub(super) output_y: u32,
    pub(super) copy_width: u32,
    pub(super) copy_height: u32,
}

pub(super) struct DenseDestinationPlan {
    pub(super) ranges: Vec<CudaDeviceBufferRange>,
    pub(super) item_range_indices: Vec<usize>,
    pub(super) total_bytes: usize,
    pub(super) requires_zero_fill: bool,
}

pub(super) fn plan_dense_destination_regions(
    regions: &[DenseDestinationRegion],
    channels: usize,
    bytes_per_sample: usize,
    budget: &mut HostPhaseBudget,
) -> Result<DenseDestinationPlan, CudaError> {
    if channels == 0 || bytes_per_sample == 0 {
        return Err(CudaError::InvalidArgument {
            message: "dense J2K destination requires nonzero channels and sample size".to_string(),
        });
    }
    let mut ranges = budget.try_vec_with_capacity(regions.len())?;
    let mut item_range_indices = budget.try_vec_with_capacity(regions.len())?;
    let mut total_bytes = 0usize;
    let mut requires_zero_fill = false;
    let mut group_start = 0usize;
    let mut group_area = 0usize;
    let mut group_pixels = 0usize;

    for (item_index, region) in regions.iter().copied().enumerate() {
        let new_group =
            item_index == 0 || regions[item_index - 1].output_index != region.output_index;
        if new_group {
            if item_index != 0 {
                requires_zero_fill |= group_area != group_pixels;
            }
            if region.output_index != ranges.len() {
                return Err(CudaError::InvalidArgument {
                    message: "dense J2K destination output indices must be contiguous and grouped"
                        .to_string(),
                });
            }
            group_start = item_index;
            group_area = 0;
            group_pixels = checked_image_words(region.output_width, region.output_height, 1)?;
            let output_samples = group_pixels
                .checked_mul(channels)
                .ok_or(CudaError::LengthTooLarge { len: group_pixels })?;
            let output_bytes =
                output_samples
                    .checked_mul(bytes_per_sample)
                    .ok_or(CudaError::LengthTooLarge {
                        len: output_samples,
                    })?;
            ranges.push(CudaDeviceBufferRange {
                offset: total_bytes,
                len: output_bytes,
            });
            total_bytes = total_bytes
                .checked_add(output_bytes)
                .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
        } else {
            let first = regions[group_start];
            if (region.output_width, region.output_height)
                != (first.output_width, first.output_height)
            {
                return Err(CudaError::InvalidArgument {
                    message: "tile stores for one J2K output disagree on destination dimensions"
                        .to_string(),
                });
            }
        }

        validate_store_destination(
            region.output_width,
            region.output_height,
            region.output_x,
            region.output_y,
            region.copy_width,
            region.copy_height,
            u32::try_from(channels).map_err(|_| CudaError::LengthTooLarge { len: channels })?,
        )?;
        for prior in &regions[group_start..item_index] {
            if rectangles_overlap(*prior, region) {
                return Err(CudaError::InvalidArgument {
                    message: format!("tile stores for J2K output {} overlap", region.output_index),
                });
            }
        }
        let area = checked_image_words(region.copy_width, region.copy_height, 1)?;
        group_area = group_area
            .checked_add(area)
            .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
        if group_area > group_pixels {
            return Err(CudaError::InvalidArgument {
                message: format!(
                    "tile stores for J2K output {} exceed its destination area",
                    region.output_index
                ),
            });
        }
        item_range_indices.push(region.output_index);
    }
    if !regions.is_empty() {
        requires_zero_fill |= group_area != group_pixels;
    }

    Ok(DenseDestinationPlan {
        ranges,
        item_range_indices,
        total_bytes,
        requires_zero_fill,
    })
}

fn rectangles_overlap(a: DenseDestinationRegion, b: DenseDestinationRegion) -> bool {
    if a.copy_width == 0 || a.copy_height == 0 || b.copy_width == 0 || b.copy_height == 0 {
        return false;
    }
    let a_end_x = a.output_x + a.copy_width;
    let a_end_y = a.output_y + a.copy_height;
    let b_end_x = b.output_x + b.copy_width;
    let b_end_y = b.output_y + b.copy_height;
    a.output_x < b_end_x && b.output_x < a_end_x && a.output_y < b_end_y && b.output_y < a_end_y
}

#[cfg(test)]
mod tests {
    use super::*;

    fn region(
        output_index: usize,
        output_x: u32,
        output_y: u32,
        copy_width: u32,
        copy_height: u32,
    ) -> DenseDestinationRegion {
        DenseDestinationRegion {
            output_index,
            output_width: 19,
            output_height: 13,
            output_x,
            output_y,
            copy_width,
            copy_height,
        }
    }

    #[test]
    fn four_tiles_share_one_dense_output_range() {
        let regions = [
            region(0, 0, 0, 11, 7),
            region(0, 11, 0, 8, 7),
            region(0, 0, 7, 11, 6),
            region(0, 11, 7, 8, 6),
        ];
        let mut budget = HostPhaseBudget::new("dense destination test");
        let plan = plan_dense_destination_regions(&regions, 3, 2, &mut budget)
            .expect("plan disjoint multi-tile destination");

        assert_eq!(plan.ranges.len(), 1);
        assert_eq!(
            plan.ranges[0],
            CudaDeviceBufferRange {
                offset: 0,
                len: 19 * 13 * 3 * 2
            }
        );
        assert_eq!(plan.item_range_indices, [0, 0, 0, 0]);
        assert!(!plan.requires_zero_fill);
    }

    #[test]
    fn overlapping_tiles_are_rejected() {
        let regions = [region(0, 0, 0, 12, 13), region(0, 11, 0, 8, 13)];
        let mut budget = HostPhaseBudget::new("dense destination overlap test");
        let error = plan_dense_destination_regions(&regions, 1, 2, &mut budget)
            .err()
            .expect("overlap must fail");
        assert!(matches!(
            error,
            CudaError::InvalidArgument { message } if message.contains("overlap")
        ));
    }

    #[test]
    fn output_indices_must_be_dense_and_grouped() {
        let regions = [region(1, 0, 0, 19, 13)];
        let mut budget = HostPhaseBudget::new("dense destination index test");
        let error = plan_dense_destination_regions(&regions, 1, 1, &mut budget)
            .err()
            .expect("sparse output index must fail");
        assert!(matches!(error, CudaError::InvalidArgument { .. }));
    }
}
