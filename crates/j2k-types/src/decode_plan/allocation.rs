// SPDX-License-Identifier: MIT OR Apache-2.0

use core::mem::size_of;

use super::{
    HtOwnedCodeBlockBatchJob, J2kDirectColorPlan, J2kDirectGrayscalePlan, J2kDirectGrayscaleStep,
    J2kDirectRgbaPlan, J2kOwnedCodeBlockBatchJob, J2kReferencedClassicPlan, J2kReferencedHtj2kPlan,
    J2kReferencedTilePlan, DEFAULT_MAX_DECODE_BYTES,
};
use crate::{
    HtCodeBlockPayloadRanges, J2kClassicCodeBlockPayload, J2kCodeBlockSegment, J2kCodestreamRange,
};

/// Retained decode-plan capacity overflow or limit violation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodePlanAllocationError;

#[derive(Default)]
struct Budget {
    bytes: usize,
}

impl Budget {
    fn include_capacity<T>(&mut self, capacity: usize) -> Result<(), DecodePlanAllocationError> {
        let bytes = capacity
            .checked_mul(size_of::<T>())
            .ok_or(DecodePlanAllocationError)?;
        self.bytes = self
            .bytes
            .checked_add(bytes)
            .ok_or(DecodePlanAllocationError)?;
        (self.bytes <= DEFAULT_MAX_DECODE_BYTES)
            .then_some(())
            .ok_or(DecodePlanAllocationError)
    }
}

impl J2kDirectGrayscalePlan {
    /// Return allocator capacities retained by this plan's owner graph.
    pub fn retained_allocation_bytes(&self) -> Result<usize, DecodePlanAllocationError> {
        let mut budget = Budget::default();
        include_grayscale(&mut budget, self)?;
        Ok(budget.bytes)
    }
}

impl J2kDirectColorPlan {
    /// Return allocator capacities retained by this plan's owner graph.
    pub fn retained_allocation_bytes(&self) -> Result<usize, DecodePlanAllocationError> {
        retained_components(&self.component_plans, self.component_plans.capacity())
    }
}

impl J2kDirectRgbaPlan {
    /// Return allocator capacities retained by this plan's owner graph.
    pub fn retained_allocation_bytes(&self) -> Result<usize, DecodePlanAllocationError> {
        retained_components(&self.component_plans, self.component_plans.capacity())
    }
}

impl J2kReferencedHtj2kPlan {
    /// Return allocator capacities retained by geometry and payload ranges.
    pub fn retained_allocation_bytes(&self) -> Result<usize, DecodePlanAllocationError> {
        let mut budget = Budget::default();
        match self {
            Self::Grayscale {
                tiles, payloads, ..
            }
            | Self::Color {
                tiles, payloads, ..
            }
            | Self::Rgba {
                tiles, payloads, ..
            } => {
                include_tiles(&mut budget, tiles, tiles.capacity())?;
                budget.include_capacity::<HtCodeBlockPayloadRanges>(payloads.capacity())?;
            }
        }
        Ok(budget.bytes)
    }
}

impl J2kReferencedClassicPlan {
    /// Return allocator capacities retained by geometry and payload ranges.
    pub fn retained_allocation_bytes(&self) -> Result<usize, DecodePlanAllocationError> {
        let mut budget = Budget::default();
        match self {
            Self::Grayscale {
                tiles,
                payloads,
                ranges,
                ..
            }
            | Self::Color {
                tiles,
                payloads,
                ranges,
                ..
            }
            | Self::Rgba {
                tiles,
                payloads,
                ranges,
                ..
            } => {
                include_tiles(&mut budget, tiles, tiles.capacity())?;
                include_classic(&mut budget, payloads.capacity(), ranges.capacity())?;
            }
        }
        Ok(budget.bytes)
    }
}

fn include_tiles(
    budget: &mut Budget,
    tiles: &[J2kReferencedTilePlan],
    capacity: usize,
) -> Result<(), DecodePlanAllocationError> {
    budget.include_capacity::<J2kReferencedTilePlan>(capacity)?;
    for tile in tiles {
        include_classic(
            budget,
            tile.classic_payloads.capacity(),
            tile.classic_ranges.capacity(),
        )?;
        if let Some(plan) = tile.grayscale_geometry() {
            include_grayscale(budget, plan)?;
        } else if let Some(plan) = tile.color_geometry() {
            include_components(
                budget,
                &plan.component_plans,
                plan.component_plans.capacity(),
            )?;
        } else if let Some(plan) = tile.rgba_geometry() {
            include_components(
                budget,
                &plan.component_plans,
                plan.component_plans.capacity(),
            )?;
        } else {
            return Err(DecodePlanAllocationError);
        }
    }
    Ok(())
}

fn include_classic(
    budget: &mut Budget,
    payload_capacity: usize,
    range_capacity: usize,
) -> Result<(), DecodePlanAllocationError> {
    budget.include_capacity::<J2kClassicCodeBlockPayload>(payload_capacity)?;
    budget.include_capacity::<J2kCodestreamRange>(range_capacity)
}

fn retained_components(
    components: &[J2kDirectGrayscalePlan],
    capacity: usize,
) -> Result<usize, DecodePlanAllocationError> {
    let mut budget = Budget::default();
    include_components(&mut budget, components, capacity)?;
    Ok(budget.bytes)
}

fn include_components(
    budget: &mut Budget,
    components: &[J2kDirectGrayscalePlan],
    capacity: usize,
) -> Result<(), DecodePlanAllocationError> {
    budget.include_capacity::<J2kDirectGrayscalePlan>(capacity)?;
    for component in components {
        include_grayscale(budget, component)?;
    }
    Ok(())
}

fn include_grayscale(
    budget: &mut Budget,
    plan: &J2kDirectGrayscalePlan,
) -> Result<(), DecodePlanAllocationError> {
    budget.include_capacity::<J2kDirectGrayscaleStep>(plan.steps.capacity())?;
    for step in &plan.steps {
        match step {
            J2kDirectGrayscaleStep::ClassicSubBand(subband) => {
                budget.include_capacity::<J2kOwnedCodeBlockBatchJob>(subband.jobs.capacity())?;
                for job in &subband.jobs {
                    budget.include_capacity::<u8>(job.data.capacity())?;
                    budget.include_capacity::<J2kCodeBlockSegment>(job.segments.capacity())?;
                }
            }
            J2kDirectGrayscaleStep::HtSubBand(subband) => {
                budget.include_capacity::<HtOwnedCodeBlockBatchJob>(subband.jobs.capacity())?;
                for job in &subband.jobs {
                    budget.include_capacity::<u8>(job.data.capacity())?;
                }
            }
            J2kDirectGrayscaleStep::Idwt(_) | J2kDirectGrayscaleStep::Store(_) => {}
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use super::*;
    use crate::{J2kCodeBlockStyle, J2kSubBandType};

    #[test]
    fn retained_bytes_use_nested_vector_capacities() {
        let mut jobs = Vec::new();
        jobs.try_reserve_exact(3).expect("job capacity");
        jobs.push(J2kOwnedCodeBlockBatchJob {
            output_x: 0,
            output_y: 0,
            data: {
                let mut data = Vec::new();
                data.try_reserve_exact(11).expect("data capacity");
                data
            },
            segments: {
                let mut segments = Vec::new();
                segments.try_reserve_exact(5).expect("segment capacity");
                segments
            },
            width: 1,
            height: 1,
            output_stride: 1,
            missing_bit_planes: 0,
            number_of_coding_passes: 0,
            total_bitplanes: 0,
            roi_shift: 0,
            sub_band_type: J2kSubBandType::LowLow,
            style: J2kCodeBlockStyle {
                selective_arithmetic_coding_bypass: false,
                reset_context_probabilities: false,
                termination_on_each_pass: false,
                vertically_causal_context: false,
                segmentation_symbols: false,
            },
            strict: false,
            dequantization_step: 1.0,
        });
        let mut steps = Vec::new();
        steps.try_reserve_exact(4).expect("step capacity");
        steps.push(J2kDirectGrayscaleStep::ClassicSubBand(
            super::super::J2kOwnedSubBandPlan {
                band_id: 0,
                rect: super::super::J2kRect {
                    x0: 0,
                    y0: 0,
                    x1: 1,
                    y1: 1,
                },
                width: 1,
                height: 1,
                irreversible_midpoint: false,
                jobs,
            },
        ));
        let plan = J2kDirectGrayscalePlan {
            dimensions: (1, 1),
            bit_depth: 8,
            steps,
        };

        assert_eq!(
            plan.retained_allocation_bytes().expect("retained bytes"),
            4 * size_of::<J2kDirectGrayscaleStep>()
                + 3 * size_of::<J2kOwnedCodeBlockBatchJob>()
                + 11
                + 5 * size_of::<J2kCodeBlockSegment>()
        );
    }

    #[test]
    fn direct_plan_owner_types_remain_move_only_values() {
        fn assert_debug<T: core::fmt::Debug>() {}

        assert_debug::<J2kDirectColorPlan>();
        assert_debug::<J2kDirectGrayscalePlan>();
        assert_debug::<super::super::J2kOwnedSubBandPlan>();
        assert_debug::<super::super::HtOwnedSubBandPlan>();
        assert_debug::<J2kOwnedCodeBlockBatchJob>();
        assert_debug::<HtOwnedCodeBlockBatchJob>();
    }
}
