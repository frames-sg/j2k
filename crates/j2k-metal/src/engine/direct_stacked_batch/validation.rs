// SPDX-License-Identifier: MIT OR Apache-2.0

use std::mem::size_of;
use std::sync::Arc;

use super::super::{
    classic_group_shapes_match, classic_sub_band_shapes_match, ht_group_shapes_match,
    ht_sub_band_shapes_match, idwt_shapes_match, repeated_shared_direct_color_plan_count,
    store_shapes_match, DirectTier1Mode, Error, PreparedDirectColorPlan,
    PreparedDirectGrayscalePlan, PreparedDirectGrayscaleStep,
};

pub(super) struct StackedComponentBatchPlan<'p> {
    pub(super) first: &'p PreparedDirectGrayscalePlan,
    pub(super) count: usize,
    pub(super) broadcast_tier1_inputs: bool,
}

pub(super) struct StackedColorBatchPreflight<'p> {
    pub(super) first: &'p PreparedDirectColorPlan,
    pub(super) execution_plans: &'p [Arc<PreparedDirectColorPlan>],
    pub(super) repeated_count: Option<usize>,
    pub(super) component_plan_refs: [Vec<&'p PreparedDirectGrayscalePlan>; 3],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct CheckedF32BatchSpan {
    pub(super) per_instance_elements: usize,
    pub(super) instance_count: usize,
    pub(super) total_elements: usize,
    pub(super) stride_bytes: usize,
    pub(super) total_bytes: usize,
}

fn span_overflow(context: &'static str, operation: &'static str) -> Error {
    Error::MetalKernel {
        message: format!("{context} {operation} overflow"),
    }
}

pub(super) fn checked_f32_batch_span(
    per_instance_elements: usize,
    instance_count: usize,
    context: &'static str,
) -> Result<CheckedF32BatchSpan, Error> {
    let total_elements = per_instance_elements
        .checked_mul(instance_count)
        .ok_or_else(|| span_overflow(context, "element count"))?;
    let stride_bytes = per_instance_elements
        .checked_mul(size_of::<f32>())
        .ok_or_else(|| span_overflow(context, "instance byte stride"))?;
    let total_bytes = stride_bytes
        .checked_mul(instance_count)
        .ok_or_else(|| span_overflow(context, "batch byte count"))?;
    Ok(CheckedF32BatchSpan {
        per_instance_elements,
        instance_count,
        total_elements,
        stride_bytes,
        total_bytes,
    })
}

pub(super) fn checked_f32_dimension_span(
    width: u32,
    height: u32,
    instance_count: usize,
    context: &'static str,
) -> Result<CheckedF32BatchSpan, Error> {
    let per_instance_elements = usize::try_from(width)
        .ok()
        .and_then(|width| {
            usize::try_from(height)
                .ok()
                .and_then(|height| width.checked_mul(height))
        })
        .ok_or_else(|| span_overflow(context, "plane element count"))?;
    checked_f32_batch_span(per_instance_elements, instance_count, context)
}

pub(super) fn checked_f32_instance_offset(
    span: &CheckedF32BatchSpan,
    instance_index: usize,
    context: &'static str,
) -> Result<usize, Error> {
    if instance_index >= span.instance_count {
        return Err(Error::MetalKernel {
            message: format!("{context} instance index is outside its batch"),
        });
    }
    instance_index
        .checked_mul(span.stride_bytes)
        .ok_or_else(|| span_overflow(context, "instance byte offset"))
}

pub(super) fn checked_f32_element_offset(
    span: &CheckedF32BatchSpan,
    instance_index: usize,
    element_offset: usize,
    context: &'static str,
) -> Result<usize, Error> {
    if element_offset >= span.per_instance_elements {
        return Err(Error::MetalKernel {
            message: format!("{context} element offset is outside its instance"),
        });
    }
    let instance_offset = checked_f32_instance_offset(span, instance_index, context)?;
    let member_offset = element_offset
        .checked_mul(size_of::<f32>())
        .ok_or_else(|| span_overflow(context, "member byte offset"))?;
    instance_offset
        .checked_add(member_offset)
        .ok_or_else(|| span_overflow(context, "aggregate byte offset"))
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

pub(super) fn preflight_stacked_mct_rgb8_color_batch(
    plans: &[Arc<PreparedDirectColorPlan>],
) -> Result<Option<StackedColorBatchPreflight<'_>>, Error> {
    let Some(first) = plans.first().map(AsRef::as_ref) else {
        return Ok(None);
    };
    let repeated_count = repeated_shared_direct_color_plan_count(plans);
    if plans.len() <= 1
        || !first.mct
        || first.component_plans.len() != 3
        || !plans.iter().all(|plan| {
            plan.mct
                && plan.dimensions == first.dimensions
                && plan.bit_depths == first.bit_depths
                && plan.transform == first.transform
                && plan.component_plans.len() == 3
        })
    {
        return Ok(None);
    }
    let execution_plans = if repeated_count.is_some() {
        &plans[..1]
    } else {
        plans
    };
    let mut budget = crate::batch_allocation::BatchMetadataBudget::new(
        "J2K Metal stacked component plan references",
    );
    let mut component_plan_refs = [
        budget.try_vec(
            execution_plans.len(),
            "J2K Metal stacked component 0 plan reference slots",
        )?,
        budget.try_vec(
            execution_plans.len(),
            "J2K Metal stacked component 1 plan reference slots",
        )?,
        budget.try_vec(
            execution_plans.len(),
            "J2K Metal stacked component 2 plan reference slots",
        )?,
    ];
    for plan in execution_plans {
        for (component_idx, references) in component_plan_refs.iter_mut().enumerate() {
            references.push(&plan.component_plans[component_idx]);
        }
    }
    if component_plan_refs.iter().any(|references| {
        !supports_stacked_direct_component_plane_batch(references)
            || references
                .first()
                .is_none_or(|component| component.dimensions != first.dimensions)
    }) {
        return Ok(None);
    }
    Ok(Some(StackedColorBatchPreflight {
        first,
        execution_plans,
        repeated_count,
        component_plan_refs,
    }))
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

    #[test]
    fn distinct_stacked_f32_span_accepts_exact_boundary_and_rejects_one_over() {
        let exact_elements = usize::MAX / size_of::<f32>();
        let span = checked_f32_batch_span(
            exact_elements,
            1,
            "J2K MetalDirect stacked distinct test span",
        )
        .expect("largest exactly representable f32 byte span");
        assert_eq!(span.per_instance_elements, exact_elements);
        assert_eq!(span.total_elements, exact_elements);
        assert_eq!(span.stride_bytes, exact_elements * size_of::<f32>());
        assert_eq!(span.total_bytes, exact_elements * size_of::<f32>());

        assert!(matches!(
            checked_f32_batch_span(
                exact_elements + 1,
                1,
                "J2K MetalDirect stacked distinct test span",
            ),
            Err(Error::MetalKernel { message }) if message.contains("overflow")
        ));
        assert!(matches!(
            checked_f32_batch_span(
                2,
                usize::MAX,
                "J2K MetalDirect stacked distinct test span",
            ),
            Err(Error::MetalKernel { message }) if message.contains("overflow")
        ));
    }

    #[test]
    fn repeated_f32_offsets_are_checked_against_the_validated_instance_span() {
        let exact_per_instance = usize::MAX / 8;
        let exact = checked_f32_batch_span(
            exact_per_instance,
            2,
            "J2K MetalDirect repeated grayscale test span",
        )
        .expect("largest two-instance f32 span below usize::MAX");
        assert_eq!(exact.total_bytes, exact_per_instance * 8);
        assert!(matches!(
            checked_f32_batch_span(
                exact_per_instance + 1,
                2,
                "J2K MetalDirect repeated grayscale test span",
            ),
            Err(Error::MetalKernel { message }) if message.contains("overflow")
        ));

        let span = checked_f32_batch_span(4, 2, "J2K MetalDirect repeated grayscale test span")
            .expect("small repeated span");
        assert_eq!(
            checked_f32_element_offset(
                &span,
                1,
                3,
                "J2K MetalDirect repeated grayscale test offset",
            )
            .expect("last element of last instance"),
            7 * size_of::<f32>()
        );
        assert!(matches!(
            checked_f32_element_offset(
                &span,
                1,
                4,
                "J2K MetalDirect repeated grayscale test offset",
            ),
            Err(Error::MetalKernel { message }) if message.contains("outside its instance")
        ));
        assert!(matches!(
            checked_f32_element_offset(
                &span,
                usize::MAX,
                0,
                "J2K MetalDirect repeated grayscale test offset",
            ),
            Err(Error::MetalKernel { message }) if message.contains("outside its batch")
        ));
    }

    #[test]
    fn f32_dimension_span_rejects_width_height_byte_overflow() {
        assert!(matches!(
            checked_f32_dimension_span(
                u32::MAX,
                u32::MAX,
                1,
                "J2K MetalDirect repeated grayscale dimension span",
            ),
            Err(Error::MetalKernel { message }) if message.contains("overflow")
        ));
    }
}
