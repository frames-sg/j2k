// SPDX-License-Identifier: MIT OR Apache-2.0

//! Parse-free CPU execution for referenced classic JPEG 2000 plans.

use crate::error::{bail, DecodingError, Result};
use crate::{
    decode_j2k_code_block_scalar_with_workspace, try_reserve_decode_elements,
    J2kCodeBlockDecodeJob, J2kCodeBlockDecodeWorkspace, J2kDirectGrayscalePlan,
    J2kDirectGrayscaleStep, J2kOwnedSubBandPlan, J2kReferencedClassicPlan, J2kWaveletTransform,
};

use super::allocation::prepare_referenced_classic_scratch;
use super::referenced::{decoded_color_components, decoded_plane, payload_slice};
use super::{
    apply_inverse_mct_region, checked_sub_band_job_output_range, execute_idwt_step,
    prepare_sub_band_output, store_component, DirectComponentBandScratch, DirectComponentPlane,
    J2kDirectCpuScratch, J2kDirectDecodedComponents, SubBandJobOutputRange,
};

/// Execute retained per-tile Gray/RGB/RGBA classic JPEG 2000 geometry
/// without reparsing packet headers.
///
/// Ordered compressed fragments are validated against `encoded_input` and
/// concatenated per code block in one retained scratch buffer. Reconstructed
/// component owners remain in `scratch` and are borrowed by the returned view.
#[doc(hidden)]
pub fn execute_referenced_classic_plan<'scratch>(
    plan: &J2kReferencedClassicPlan,
    encoded_input: &[u8],
    signed: bool,
    scratch: &'scratch mut J2kDirectCpuScratch,
) -> Result<J2kDirectDecodedComponents<'scratch>> {
    execute_referenced_classic_plan_with_payloads(
        plan,
        encoded_input,
        plan.payloads(),
        plan.ranges(),
        signed,
        scratch,
    )
}

/// Execute retained classic geometry from caller-flattened payload ranges.
///
/// Payload descriptors remain in geometry traversal order. Their fragment
/// ranges may point anywhere inside the shared `payload_arena`.
#[doc(hidden)]
pub fn execute_referenced_classic_plan_from_payloads<'scratch>(
    plan: &J2kReferencedClassicPlan,
    payload_arena: &[u8],
    payloads: &[crate::J2kClassicCodeBlockPayload],
    ranges: &[crate::J2kCodestreamRange],
    signed: bool,
    scratch: &'scratch mut J2kDirectCpuScratch,
) -> Result<J2kDirectDecodedComponents<'scratch>> {
    execute_referenced_classic_plan_with_payloads(
        plan,
        payload_arena,
        payloads,
        ranges,
        signed,
        scratch,
    )
}

fn execute_referenced_classic_plan_with_payloads<'scratch>(
    plan: &J2kReferencedClassicPlan,
    encoded_input: &[u8],
    payload_descriptors: &[crate::J2kClassicCodeBlockPayload],
    payload_ranges: &[crate::J2kCodestreamRange],
    signed: bool,
    scratch: &'scratch mut J2kDirectCpuScratch,
) -> Result<J2kDirectDecodedComponents<'scratch>> {
    if payload_descriptors.len() != plan.payloads().len() {
        bail!(DecodingError::CodeBlockDecodeFailure);
    }
    validate_payload_ranges(encoded_input, payload_descriptors, payload_ranges)?;
    prepare_referenced_classic_scratch(plan, scratch)?;
    {
        let J2kDirectCpuScratch {
            component_band_sets,
            component_planes,
            compressed_payload,
            classic_workspace,
            ht_workspace: _,
            staged_state: _,
        } = scratch;
        let mut payloads = ClassicPayloadCursor::new(
            payload_descriptors,
            payload_ranges,
            encoded_input,
            compressed_payload,
        );
        let mut output_initialized = [false; 4];

        for tile in plan.tiles() {
            if let Some(geometry) = tile.grayscale_geometry() {
                let bands = component_band_sets
                    .first_mut()
                    .ok_or(DecodingError::CodeBlockDecodeFailure)?;
                let output = component_planes
                    .first_mut()
                    .ok_or(DecodingError::CodeBlockDecodeFailure)?;
                execute_component_plan_referenced(
                    geometry,
                    bands,
                    output,
                    &mut payloads,
                    classic_workspace,
                    &mut output_initialized[0],
                )?;
            } else if let Some(geometry) = tile.color_geometry() {
                execute_color_components_referenced(
                    &geometry.component_plans,
                    3,
                    geometry.bit_depths,
                    geometry.mct,
                    geometry.transform,
                    signed,
                    component_band_sets,
                    component_planes,
                    &mut payloads,
                    classic_workspace,
                    &mut output_initialized,
                    tile.destination_rect(),
                )?;
            } else if let Some(geometry) = tile.rgba_geometry() {
                execute_color_components_referenced(
                    &geometry.component_plans,
                    4,
                    [
                        geometry.bit_depths[0],
                        geometry.bit_depths[1],
                        geometry.bit_depths[2],
                    ],
                    geometry.mct,
                    geometry.transform,
                    signed,
                    component_band_sets,
                    component_planes,
                    &mut payloads,
                    classic_workspace,
                    &mut output_initialized,
                    tile.destination_rect(),
                )?;
            } else {
                bail!(DecodingError::CodeBlockDecodeFailure);
            }
        }
        payloads.ensure_exhausted()?;
    }

    decoded_components(plan, scratch)
}

#[expect(
    clippy::too_many_arguments,
    reason = "the parse-free RGB-family executor keeps geometry, transform metadata, retained owners, payload cursor, and scalar workspace explicit"
)]
fn execute_color_components_referenced(
    component_plans: &[J2kDirectGrayscalePlan],
    expected_component_count: usize,
    rgb_bit_depths: [u8; 3],
    mct: bool,
    transform: J2kWaveletTransform,
    signed: bool,
    component_band_sets: &mut [DirectComponentBandScratch],
    reconstructed_planes: &mut [DirectComponentPlane],
    payloads: &mut ClassicPayloadCursor<'_, '_>,
    classic_workspace: &mut J2kCodeBlockDecodeWorkspace,
    output_initialized: &mut [bool; 4],
    destination: crate::J2kRect,
) -> Result<()> {
    if component_plans.len() != expected_component_count
        || !matches!(expected_component_count, 3 | 4)
        || component_band_sets.len() < component_plans.len()
        || reconstructed_planes.len() < component_plans.len()
    {
        bail!(DecodingError::CodeBlockDecodeFailure);
    }
    for (component_index, component_plan) in component_plans.iter().enumerate() {
        execute_component_plan_referenced(
            component_plan,
            &mut component_band_sets[component_index],
            &mut reconstructed_planes[component_index],
            payloads,
            classic_workspace,
            &mut output_initialized[component_index],
        )?;
    }
    if mct {
        let [plane0, plane1, plane2, ..] = reconstructed_planes else {
            bail!(DecodingError::CodeBlockDecodeFailure);
        };
        apply_inverse_mct_region(
            transform,
            rgb_bit_depths,
            signed,
            destination,
            plane0,
            plane1,
            plane2,
        )?;
    }
    Ok(())
}

fn execute_component_plan_referenced(
    plan: &J2kDirectGrayscalePlan,
    bands: &mut DirectComponentBandScratch,
    output: &mut DirectComponentPlane,
    payloads: &mut ClassicPayloadCursor<'_, '_>,
    classic_workspace: &mut J2kCodeBlockDecodeWorkspace,
    output_initialized: &mut bool,
) -> Result<()> {
    bands.reset();
    let mut stored = false;
    for step in &plan.steps {
        match step {
            J2kDirectGrayscaleStep::ClassicSubBand(sub_band) => {
                execute_classic_sub_band_referenced(sub_band, bands, payloads, classic_workspace)?;
            }
            J2kDirectGrayscaleStep::HtSubBand(_) => {
                bail!(DecodingError::UnsupportedFeature(
                    "referenced classic CPU plan encountered HT code blocks"
                ));
            }
            J2kDirectGrayscaleStep::Idwt(step) => execute_idwt_step(step, bands)?,
            J2kDirectGrayscaleStep::Store(store) => {
                store_component(store, bands.active(), output, output_initialized)?;
                stored = true;
            }
        }
    }
    if stored {
        Ok(())
    } else {
        Err(DecodingError::CodeBlockDecodeFailure.into())
    }
}

fn execute_classic_sub_band_referenced(
    plan: &J2kOwnedSubBandPlan,
    bands: &mut DirectComponentBandScratch,
    payloads: &mut ClassicPayloadCursor<'_, '_>,
    workspace: &mut J2kCodeBlockDecodeWorkspace,
) -> Result<()> {
    let (output, sub_band_width) =
        prepare_sub_band_output(bands, plan.band_id, plan.rect, plan.width, plan.height)?;
    for job in &plan.jobs {
        if !job.data.is_empty() {
            bail!(DecodingError::CodeBlockDecodeFailure);
        }
        let output_range = checked_sub_band_job_output_range(&SubBandJobOutputRange {
            output_x: job.output_x,
            output_y: job.output_y,
            output_stride: job.output_stride,
            width: job.width,
            height: job.height,
            sub_band_width,
            plan_width: plan.width,
            plan_height: plan.height,
            output_len: output.len(),
        })?;
        let data = payloads.next_data()?;
        let code_block = J2kCodeBlockDecodeJob {
            data,
            segments: &job.segments,
            width: job.width,
            height: job.height,
            output_stride: job.output_stride,
            missing_bit_planes: job.missing_bit_planes,
            number_of_coding_passes: job.number_of_coding_passes,
            total_bitplanes: job.total_bitplanes,
            roi_shift: job.roi_shift,
            sub_band_type: job.sub_band_type,
            style: job.style,
            strict: job.strict,
            dequantization_step: job.dequantization_step,
        };
        decode_j2k_code_block_scalar_with_workspace(
            code_block,
            &mut output[output_range],
            workspace,
        )?;
    }
    Ok(())
}

fn validate_payload_ranges(
    encoded_input: &[u8],
    payloads: &[crate::J2kClassicCodeBlockPayload],
    ranges: &[crate::J2kCodestreamRange],
) -> Result<()> {
    let mut next_range = 0usize;
    for payload in payloads {
        if payload.first_range != next_range {
            bail!(DecodingError::CodeBlockDecodeFailure);
        }
        let end_range = payload
            .end_range()
            .ok_or(DecodingError::CodeBlockDecodeFailure)?;
        let fragments = ranges
            .get(payload.first_range..end_range)
            .ok_or(DecodingError::CodeBlockDecodeFailure)?;
        let mut combined_length = 0usize;
        for range in fragments {
            payload_slice(encoded_input, *range)?;
            combined_length = combined_length
                .checked_add(range.length)
                .ok_or(DecodingError::CodeBlockDecodeFailure)?;
        }
        if combined_length != payload.combined_length {
            bail!(DecodingError::CodeBlockDecodeFailure);
        }
        next_range = end_range;
    }
    if next_range != ranges.len() {
        bail!(DecodingError::CodeBlockDecodeFailure);
    }
    Ok(())
}

struct ClassicPayloadCursor<'plan, 'scratch> {
    encoded_input: &'plan [u8],
    payloads: &'plan [crate::J2kClassicCodeBlockPayload],
    ranges: &'plan [crate::J2kCodestreamRange],
    combined: &'scratch mut alloc::vec::Vec<u8>,
    next: usize,
}

impl<'plan, 'scratch> ClassicPayloadCursor<'plan, 'scratch> {
    fn new(
        payloads: &'plan [crate::J2kClassicCodeBlockPayload],
        ranges: &'plan [crate::J2kCodestreamRange],
        encoded_input: &'plan [u8],
        combined: &'scratch mut alloc::vec::Vec<u8>,
    ) -> Self {
        Self {
            encoded_input,
            payloads,
            ranges,
            combined,
            next: 0,
        }
    }

    fn next_data(&mut self) -> Result<&[u8]> {
        let payload = self
            .payloads
            .get(self.next)
            .ok_or(DecodingError::CodeBlockDecodeFailure)?;
        self.next = self
            .next
            .checked_add(1)
            .ok_or(DecodingError::CodeBlockDecodeFailure)?;
        let end_range = payload
            .end_range()
            .ok_or(DecodingError::CodeBlockDecodeFailure)?;
        let fragments = self
            .ranges
            .get(payload.first_range..end_range)
            .ok_or(DecodingError::CodeBlockDecodeFailure)?;

        if let [range] = fragments {
            if range.length != payload.combined_length {
                bail!(DecodingError::CodeBlockDecodeFailure);
            }
            return payload_slice(self.encoded_input, *range);
        }

        self.combined.clear();
        try_reserve_decode_elements(self.combined, payload.combined_length)?;
        for range in fragments {
            self.combined
                .extend_from_slice(payload_slice(self.encoded_input, *range)?);
        }
        if self.combined.len() != payload.combined_length {
            bail!(DecodingError::CodeBlockDecodeFailure);
        }
        Ok(self.combined)
    }

    fn ensure_exhausted(&self) -> Result<()> {
        if self.next == self.payloads.len() {
            Ok(())
        } else {
            Err(DecodingError::CodeBlockDecodeFailure.into())
        }
    }
}

pub(super) fn decoded_components<'scratch>(
    plan: &J2kReferencedClassicPlan,
    scratch: &'scratch J2kDirectCpuScratch,
) -> Result<J2kDirectDecodedComponents<'scratch>> {
    let dimensions = (plan.output_rect().width(), plan.output_rect().height());
    let first = plan
        .tiles()
        .first()
        .ok_or(DecodingError::CodeBlockDecodeFailure)?;
    match plan {
        J2kReferencedClassicPlan::Grayscale { .. } => {
            let geometry = first
                .grayscale_geometry()
                .ok_or(DecodingError::CodeBlockDecodeFailure)?;
            for tile in plan.tiles() {
                let current = tile
                    .grayscale_geometry()
                    .ok_or(DecodingError::CodeBlockDecodeFailure)?;
                if current.dimensions != dimensions || current.bit_depth != geometry.bit_depth {
                    bail!(DecodingError::CodeBlockDecodeFailure);
                }
            }
            let plane = decoded_plane(
                scratch
                    .component_planes
                    .first()
                    .ok_or(DecodingError::CodeBlockDecodeFailure)?,
                dimensions,
                geometry.bit_depth,
            )?;
            Ok(J2kDirectDecodedComponents {
                dimensions,
                planes: [Some(plane), None, None, None],
                component_count: 1,
            })
        }
        J2kReferencedClassicPlan::Color { .. } => {
            let geometry = first
                .color_geometry()
                .ok_or(DecodingError::CodeBlockDecodeFailure)?;
            for tile in plan.tiles() {
                let current = tile
                    .color_geometry()
                    .ok_or(DecodingError::CodeBlockDecodeFailure)?;
                if current.dimensions != dimensions
                    || current.bit_depths != geometry.bit_depths
                    || current.component_plans.len() != 3
                {
                    bail!(DecodingError::CodeBlockDecodeFailure);
                }
            }
            decoded_color_components(dimensions, &geometry.bit_depths, 3, scratch)
        }
        J2kReferencedClassicPlan::Rgba { .. } => {
            let geometry = first
                .rgba_geometry()
                .ok_or(DecodingError::CodeBlockDecodeFailure)?;
            for tile in plan.tiles() {
                let current = tile
                    .rgba_geometry()
                    .ok_or(DecodingError::CodeBlockDecodeFailure)?;
                if current.dimensions != dimensions
                    || current.bit_depths != geometry.bit_depths
                    || current.component_plans.len() != 4
                {
                    bail!(DecodingError::CodeBlockDecodeFailure);
                }
            }
            decoded_color_components(dimensions, &geometry.bit_depths, 4, scratch)
        }
    }
}
