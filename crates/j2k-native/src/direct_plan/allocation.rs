// SPDX-License-Identifier: MIT OR Apache-2.0

//! Retained-allocation accounting for native direct-plan owner graphs.

use core::mem::size_of;

use crate::{
    HtCodeBlockPayloadRanges, HtOwnedCodeBlockBatchJob, J2kClassicCodeBlockPayload,
    J2kCodeBlockSegment, J2kCodestreamRange, J2kDirectColorPlan, J2kDirectGrayscalePlan,
    J2kDirectGrayscaleStep, J2kDirectRgbaPlan, J2kOwnedCodeBlockBatchJob, J2kReferencedClassicPlan,
    J2kReferencedHtj2kPlan, J2kReferencedTilePlan, Result, ValidationError,
    DEFAULT_MAX_DECODE_BYTES,
};

#[derive(Default)]
struct RetainedPlanBudget {
    bytes: usize,
}

impl RetainedPlanBudget {
    fn include_capacity<T>(&mut self, capacity: usize) -> Result<()> {
        let bytes = capacity
            .checked_mul(size_of::<T>())
            .ok_or(ValidationError::ImageTooLarge)?;
        self.include_bytes(bytes)
    }

    fn include_bytes(&mut self, bytes: usize) -> Result<()> {
        self.bytes = self
            .bytes
            .checked_add(bytes)
            .ok_or(ValidationError::ImageTooLarge)?;
        if self.bytes > DEFAULT_MAX_DECODE_BYTES {
            return Err(ValidationError::ImageTooLarge.into());
        }
        Ok(())
    }
}

impl J2kDirectGrayscalePlan {
    /// Return the allocator capacities retained by this direct-plan owner graph.
    ///
    /// The root plan value itself is not included. Callers that place the root
    /// in a separate heap allocation must account that allocation separately.
    ///
    /// # Errors
    ///
    /// Returns an error if nested capacity arithmetic overflows or exceeds the
    /// native decode allocation ceiling.
    #[doc(hidden)]
    pub fn retained_allocation_bytes(&self) -> Result<usize> {
        let mut budget = RetainedPlanBudget::default();
        include_grayscale_plan(&mut budget, self)?;
        Ok(budget.bytes)
    }
}

impl J2kDirectColorPlan {
    /// Return the allocator capacities retained by this direct-plan owner graph.
    ///
    /// The root plan value itself is not included. Callers that place the root
    /// in a separate heap allocation must account that allocation separately.
    ///
    /// # Errors
    ///
    /// Returns an error if nested capacity arithmetic overflows or exceeds the
    /// native decode allocation ceiling.
    #[doc(hidden)]
    pub fn retained_allocation_bytes(&self) -> Result<usize> {
        retained_color_component_plan_bytes(&self.component_plans, self.component_plans.capacity())
    }
}

impl J2kDirectRgbaPlan {
    /// Return the allocator capacities retained by this direct-plan owner graph.
    ///
    /// The root plan value itself is not included. Callers that place the root
    /// in a separate heap allocation must account that allocation separately.
    ///
    /// # Errors
    ///
    /// Returns an error if nested capacity arithmetic overflows or exceeds the
    /// native decode allocation ceiling.
    #[doc(hidden)]
    pub fn retained_allocation_bytes(&self) -> Result<usize> {
        retained_color_component_plan_bytes(&self.component_plans, self.component_plans.capacity())
    }
}

impl J2kReferencedHtj2kPlan {
    /// Return allocator capacities retained by referenced geometry and payload ranges.
    ///
    /// The encoded payload bytes themselves remain caller-owned and are not counted.
    ///
    /// # Errors
    ///
    /// Returns an error if nested capacity arithmetic overflows or exceeds the
    /// native decode allocation ceiling.
    #[doc(hidden)]
    pub fn retained_allocation_bytes(&self) -> Result<usize> {
        let mut budget = RetainedPlanBudget::default();
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
                include_referenced_tiles(&mut budget, tiles, tiles.capacity())?;
                budget.include_capacity::<HtCodeBlockPayloadRanges>(payloads.capacity())?;
            }
        }
        Ok(budget.bytes)
    }
}

impl J2kReferencedClassicPlan {
    /// Return allocator capacities retained by referenced geometry and payload fragments.
    ///
    /// The encoded payload bytes themselves remain caller-owned and are not counted.
    ///
    /// # Errors
    ///
    /// Returns an error if nested capacity arithmetic overflows or exceeds the
    /// native decode allocation ceiling.
    #[doc(hidden)]
    pub fn retained_allocation_bytes(&self) -> Result<usize> {
        let mut budget = RetainedPlanBudget::default();
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
                include_referenced_tiles(&mut budget, tiles, tiles.capacity())?;
                include_classic_references(&mut budget, payloads.capacity(), ranges.capacity())?;
            }
        }
        Ok(budget.bytes)
    }
}

fn include_referenced_tiles(
    budget: &mut RetainedPlanBudget,
    tiles: &[J2kReferencedTilePlan],
    retained_capacity: usize,
) -> Result<()> {
    budget.include_capacity::<J2kReferencedTilePlan>(retained_capacity)?;
    for tile in tiles {
        if let Some(geometry) = tile.grayscale_geometry() {
            include_grayscale_plan(budget, geometry)?;
        } else if let Some(geometry) = tile.color_geometry() {
            include_color_component_plans(
                budget,
                &geometry.component_plans,
                geometry.component_plans.capacity(),
            )?;
        } else if let Some(geometry) = tile.rgba_geometry() {
            include_color_component_plans(
                budget,
                &geometry.component_plans,
                geometry.component_plans.capacity(),
            )?;
        } else {
            return Err(ValidationError::ImageTooLarge.into());
        }
    }
    Ok(())
}

fn include_classic_references(
    budget: &mut RetainedPlanBudget,
    payload_capacity: usize,
    range_capacity: usize,
) -> Result<()> {
    budget.include_capacity::<J2kClassicCodeBlockPayload>(payload_capacity)?;
    budget.include_capacity::<J2kCodestreamRange>(range_capacity)
}

fn retained_color_component_plan_bytes(
    components: &[J2kDirectGrayscalePlan],
    retained_capacity: usize,
) -> Result<usize> {
    let mut budget = RetainedPlanBudget::default();
    include_color_component_plans(&mut budget, components, retained_capacity)?;
    Ok(budget.bytes)
}

fn include_color_component_plans(
    budget: &mut RetainedPlanBudget,
    components: &[J2kDirectGrayscalePlan],
    retained_capacity: usize,
) -> Result<()> {
    budget.include_capacity::<J2kDirectGrayscalePlan>(retained_capacity)?;
    for component in components {
        include_grayscale_plan(budget, component)?;
    }
    Ok(())
}

fn include_grayscale_plan(
    budget: &mut RetainedPlanBudget,
    plan: &J2kDirectGrayscalePlan,
) -> Result<()> {
    budget.include_capacity::<J2kDirectGrayscaleStep>(plan.steps.capacity())?;
    for step in &plan.steps {
        match step {
            J2kDirectGrayscaleStep::ClassicSubBand(sub_band) => {
                budget.include_capacity::<J2kOwnedCodeBlockBatchJob>(sub_band.jobs.capacity())?;
                for job in &sub_band.jobs {
                    budget.include_capacity::<u8>(job.data.capacity())?;
                    budget.include_capacity::<J2kCodeBlockSegment>(job.segments.capacity())?;
                }
            }
            J2kDirectGrayscaleStep::HtSubBand(sub_band) => {
                budget.include_capacity::<HtOwnedCodeBlockBatchJob>(sub_band.jobs.capacity())?;
                for job in &sub_band.jobs {
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
    use crate::{HtOwnedSubBandPlan, J2kOwnedSubBandPlan, J2kRect};

    #[test]
    fn retained_bytes_use_nested_vector_capacities() {
        let mut jobs = Vec::new();
        jobs.try_reserve_exact(3).expect("job test capacity");
        jobs.push(J2kOwnedCodeBlockBatchJob {
            output_x: 0,
            output_y: 0,
            data: {
                let mut data = Vec::new();
                data.try_reserve_exact(11).expect("data test capacity");
                data
            },
            segments: {
                let mut segments = Vec::new();
                segments
                    .try_reserve_exact(5)
                    .expect("segment test capacity");
                segments
            },
            width: 1,
            height: 1,
            output_stride: 1,
            missing_bit_planes: 0,
            number_of_coding_passes: 0,
            total_bitplanes: 0,
            roi_shift: 0,
            sub_band_type: crate::J2kSubBandType::LowLow,
            style: crate::J2kCodeBlockStyle {
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
        steps.try_reserve_exact(4).expect("step test capacity");
        steps.push(J2kDirectGrayscaleStep::ClassicSubBand(
            J2kOwnedSubBandPlan {
                band_id: 0,
                rect: J2kRect {
                    x0: 0,
                    y0: 0,
                    x1: 1,
                    y1: 1,
                },
                width: 1,
                height: 1,
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
    fn all_direct_plan_owner_types_are_move_only() {
        fn assert_debug<T: core::fmt::Debug>() {}

        assert_debug::<J2kDirectColorPlan>();
        assert_debug::<J2kDirectGrayscalePlan>();
        assert_debug::<J2kOwnedSubBandPlan>();
        assert_debug::<HtOwnedSubBandPlan>();
        assert_debug::<J2kOwnedCodeBlockBatchJob>();
        assert_debug::<HtOwnedCodeBlockBatchJob>();
    }
}
