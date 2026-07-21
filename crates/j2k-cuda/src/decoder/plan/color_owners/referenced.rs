// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_core::PixelFormat;
use j2k_native::{
    HtCodeBlockPayloadRanges, J2kClassicCodeBlockPayload, J2kCodestreamRange,
    J2kDirectGrayscalePlan, J2kDirectGrayscaleStep,
};

use crate::allocation::HostPhaseBudget;
use crate::{CudaHtj2kDecodePlan, Error};

#[expect(
    clippy::too_many_arguments,
    reason = "referenced tile bridge keeps destination geometry and retained owners explicit"
)]
pub(in crate::decoder::plan) fn flatten_referenced_cuda_color_tile_components(
    component_plans: &[J2kDirectGrayscalePlan],
    payloads: &[HtCodeBlockPayloadRanges],
    encoded: &[u8],
    format: PixelFormat,
    output_origin: (u32, u32),
    output_dimensions: (u32, u32),
    shared_payload: &mut Vec<u8>,
    host_budget: &mut HostPhaseBudget,
) -> Result<Vec<CudaHtj2kDecodePlan>, Error> {
    let mut components = host_budget.try_vec_with_capacity(component_plans.len())?;
    let mut payload_offset = 0usize;
    for component_plan in component_plans {
        let component_payload_count = component_plan
            .steps
            .iter()
            .map(|step| match step {
                J2kDirectGrayscaleStep::HtSubBand(subband) => subband.jobs.len(),
                _ => 0,
            })
            .try_fold(0usize, usize::checked_add)
            .ok_or(Error::UnsupportedCudaRequest {
                reason: super::super::super::CUDA_HTJ2K_BATCH_PAYLOAD_TOO_LARGE,
            })?;
        let payload_end = payload_offset.checked_add(component_payload_count).ok_or(
            Error::UnsupportedCudaRequest {
                reason: super::super::super::CUDA_HTJ2K_BATCH_PAYLOAD_TOO_LARGE,
            },
        )?;
        let component_payloads =
            payloads
                .get(payload_offset..payload_end)
                .ok_or(Error::UnsupportedCudaRequest {
                    reason:
                        "prepared CUDA color tile payload ranges do not match component geometry",
                })?;
        components.push(
            CudaHtj2kDecodePlan::from_referenced_tile_grayscale_plan_into_shared(
                component_plan,
                component_payloads,
                encoded,
                format,
                output_origin,
                output_dimensions,
                shared_payload,
                host_budget,
            )?,
        );
        payload_offset = payload_end;
    }
    if payload_offset != payloads.len() {
        return Err(Error::UnsupportedCudaRequest {
            reason: "prepared CUDA color tile payload ranges contain trailing jobs",
        });
    }
    Ok(components)
}

#[expect(
    clippy::too_many_arguments,
    reason = "referenced classic tile bridge keeps ranges and destination geometry explicit"
)]
pub(in crate::decoder::plan) fn flatten_referenced_classic_cuda_color_tile_components(
    component_plans: &[J2kDirectGrayscalePlan],
    payloads: &[J2kClassicCodeBlockPayload],
    ranges: &[J2kCodestreamRange],
    encoded: &[u8],
    format: PixelFormat,
    output_origin: (u32, u32),
    output_dimensions: (u32, u32),
    shared_payload: &mut Vec<u8>,
    host_budget: &mut HostPhaseBudget,
) -> Result<Vec<CudaHtj2kDecodePlan>, Error> {
    let mut components = host_budget.try_vec_with_capacity(component_plans.len())?;
    let mut payload_offset = 0usize;
    for component_plan in component_plans {
        let component_payload_count = component_plan
            .steps
            .iter()
            .map(|step| match step {
                J2kDirectGrayscaleStep::ClassicSubBand(subband) => subband.jobs.len(),
                _ => 0,
            })
            .try_fold(0usize, usize::checked_add)
            .ok_or(Error::UnsupportedCudaRequest {
                reason: super::super::super::CUDA_HTJ2K_BATCH_PAYLOAD_TOO_LARGE,
            })?;
        let payload_end = payload_offset.checked_add(component_payload_count).ok_or(
            Error::UnsupportedCudaRequest {
                reason: super::super::super::CUDA_HTJ2K_BATCH_PAYLOAD_TOO_LARGE,
            },
        )?;
        let component_payloads =
            payloads
                .get(payload_offset..payload_end)
                .ok_or(Error::UnsupportedCudaRequest {
                    reason:
                        "prepared CUDA classic color tile payloads do not match component geometry",
                })?;
        components.push(
            CudaHtj2kDecodePlan::from_referenced_classic_tile_grayscale_plan_into_shared(
                component_plan,
                component_payloads,
                ranges,
                encoded,
                format,
                output_origin,
                output_dimensions,
                shared_payload,
                host_budget,
            )?,
        );
        payload_offset = payload_end;
    }
    if payload_offset != payloads.len() {
        return Err(Error::UnsupportedCudaRequest {
            reason: "prepared CUDA classic color tile payloads contain trailing jobs",
        });
    }
    Ok(components)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use j2k::{prepare_batch, BatchDecodeOptions, EncodedImage};
    use j2k_native::{J2kDirectGrayscaleStep, J2kReferencedHtj2kPlan};

    use super::*;

    #[test]
    fn referenced_color_plan_retains_zero_entropy_components() {
        let encoded = Arc::<[u8]>::from(j2k_test_support::openhtj2k_sigprop_overlap_fixture());
        let prepared = prepare_batch(
            vec![EncodedImage::full(Arc::clone(&encoded))],
            BatchDecodeOptions::default(),
        )
        .expect("prepare independent refinement-overlap fixture");
        let image = &prepared.groups()[0].images()[0];
        let referenced = image
            .htj2k_plan()
            .expect("retained HTJ2K plan")
            .adapter_view()
            .downcast_ref::<J2kReferencedHtj2kPlan>()
            .expect("native referenced HTJ2K plan adapter");
        let tile = &referenced.tiles()[0];
        let geometry = tile.color_geometry().expect("RGB tile geometry");
        let job_counts = geometry
            .component_plans
            .iter()
            .map(|plan| {
                plan.steps
                    .iter()
                    .map(|step| match step {
                        J2kDirectGrayscaleStep::HtSubBand(subband) => subband.jobs.len(),
                        _ => 0,
                    })
                    .sum::<usize>()
            })
            .collect::<Vec<_>>();
        assert!(
            job_counts.contains(&0),
            "fixture must retain its zero-entropy component: {job_counts:?}"
        );

        let mut shared_payload = Vec::new();
        let mut budget = HostPhaseBudget::new("referenced zero-entropy color-plan test");
        let span = tile.payload_records();
        let payload_end = span.end_record().expect("tile payload span end");
        let components = flatten_referenced_cuda_color_tile_components(
            &geometry.component_plans,
            &referenced.payloads()[span.first_record..payload_end],
            &encoded,
            PixelFormat::Rgb8,
            {
                let output = image.plan().output_rect();
                (output.x, output.y)
            },
            image.plan().output_dims(),
            &mut shared_payload,
            &mut budget,
        )
        .expect("zero-entropy component remains an executable zero-coefficient plan");

        assert_eq!(components.len(), 3);
        assert_eq!(
            components
                .iter()
                .map(CudaHtj2kDecodePlan::block_count)
                .collect::<Vec<_>>(),
            job_counts
        );
    }
}
