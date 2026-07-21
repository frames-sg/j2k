// SPDX-License-Identifier: MIT OR Apache-2.0

use std::time::Instant;

use metal::Buffer;

use crate::batch_allocation::{BatchMetadataBudget, BatchMetadataRequest};
use crate::compute::decode_dispatch::{
    encode_distinct_ht_sub_band_groups_to_buffer_in_command_buffer,
    encode_distinct_ht_sub_band_groups_to_buffer_in_encoder,
    encode_distinct_ht_sub_bands_to_buffer_in_command_buffer,
    encode_distinct_ht_sub_bands_to_buffer_in_encoder,
};
use crate::compute::direct_grayscale_execute::upload_cpu_decoded_coefficients;
use crate::compute::direct_roi::BandRequiredRegion;
use crate::compute::{
    decode_ht_inputs_on_cpu_with_plan_cache, direct_preflight_invariant, elapsed_us,
    take_f32_scratch_buffer, CpuTier1DecodeSubstageCounters, DirectTier1Mode, Error,
    HtCpuDecodeInput, PreparedDirectGrayscalePlan, PreparedDirectGrayscaleStep, PreparedHtSubBand,
    PreparedHtSubBandGroup,
};

use super::super::resources::{retain_metal_tier1_output, DirectBandSlice};
use super::super::validation::{
    checked_f32_batch_span, checked_f32_dimension_span, checked_f32_element_offset,
    checked_f32_instance_offset,
};
use super::{planned_cpu_input_count, try_collect_submission_items, SubmissionContext};

impl SubmissionContext<'_, '_, '_> {
    pub(super) fn submit_ht_group(
        &mut self,
        first: &PreparedDirectGrayscalePlan,
        step_idx: usize,
        group: &PreparedHtSubBandGroup,
    ) -> Result<(), Error> {
        let input_count = planned_cpu_input_count(
            self.tier1_mode,
            self.flattened_cpu_tier1_cache.is_some(),
            self.broadcast_tier1_inputs,
            self.plans.len(),
        );
        let mut metadata_budget =
            BatchMetadataBudget::new("J2K Metal stacked HT group submission metadata");
        metadata_budget.preflight(&[
            BatchMetadataRequest::of::<&PreparedHtSubBandGroup>(self.plans.len()),
            BatchMetadataRequest::of::<HtCpuDecodeInput<'_>>(input_count),
        ])?;
        let groups = try_collect_submission_items(
            &mut metadata_budget,
            self.plans.iter().map(|plan| {
                plan.ht_group_starting_at(step_idx).ok_or_else(|| {
                    direct_preflight_invariant("HT group step mismatch in stacked component batch")
                })
            }),
            "J2K Metal stacked HT group references",
        )?;
        let span = checked_f32_batch_span(
            group.total_coefficients,
            self.count,
            "J2K MetalDirect stacked HT group",
        )?;
        let buffer = match self.tier1_mode {
            DirectTier1Mode::Metal => {
                let output = take_f32_scratch_buffer(self.runtime, span.total_elements)?;
                let (buffers, status_check) = if let Some(encoder) = self.compute_encoder {
                    encode_distinct_ht_sub_band_groups_to_buffer_in_encoder(
                        self.runtime,
                        encoder,
                        &groups,
                        &output.buffer,
                    )?
                } else {
                    encode_distinct_ht_sub_band_groups_to_buffer_in_command_buffer(
                        self.runtime,
                        self.command_buffers.default,
                        &groups,
                        &output.buffer,
                    )?
                };
                if let Some(encoder) = self.compute_encoder {
                    encoder.memory_barrier_with_resources(&[&output.buffer]);
                }
                retain_metal_tier1_output(
                    output,
                    buffers,
                    status_check,
                    self.retained_buffers,
                    self.status_checks,
                    self.scratch_buffers,
                )?
            }
            DirectTier1Mode::CpuUpload => self.prepare_ht_group_cpu_buffer(
                first,
                step_idx,
                group,
                &groups,
                &mut metadata_budget,
            )?,
        };

        for (instance_idx, bands) in self.resources.band_sets.iter_mut().enumerate() {
            let source_idx = if self.broadcast_tier1_inputs {
                0
            } else {
                instance_idx
            };
            let source_group = groups[source_idx];
            for member in &source_group.members {
                bands.push(DirectBandSlice {
                    band_id: member.band_id,
                    buffer: buffer.clone(),
                    offset_bytes: checked_f32_element_offset(
                        &span,
                        source_idx,
                        member.offset_elements,
                        "J2K MetalDirect stacked HT group member",
                    )?,
                    window: member.window,
                });
            }
        }
        Ok(())
    }

    fn prepare_ht_group_cpu_buffer(
        &mut self,
        first: &PreparedDirectGrayscalePlan,
        step_idx: usize,
        group: &PreparedHtSubBandGroup,
        groups: &[&PreparedHtSubBandGroup],
        metadata_budget: &mut BatchMetadataBudget,
    ) -> Result<Buffer, Error> {
        let input_groups = if self.broadcast_tier1_inputs {
            &groups[..1]
        } else {
            groups
        };
        if let Some(cache) = self.flattened_cpu_tier1_cache {
            return cache.buffer_for(
                self.component_idx,
                step_idx,
                group.total_coefficients,
                input_groups.len(),
            );
        }

        let inputs = try_collect_submission_items(
            metadata_budget,
            input_groups.iter().map(|group| {
                Ok(HtCpuDecodeInput {
                    payload_source: &group.payload_source,
                    jobs: &group.jobs,
                    output_len: group.total_coefficients,
                })
            }),
            "J2K Metal stacked HT group CPU inputs",
        )?;
        let decode_started = self.profile_stages.then(Instant::now);
        let cpu_tier1_counters = self
            .profile_stages
            .then(CpuTier1DecodeSubstageCounters::default);
        let coefficients = decode_ht_inputs_on_cpu_with_plan_cache(
            first,
            step_idx,
            &inputs,
            cpu_tier1_counters.as_ref(),
        )?;
        if let Some(started) = decode_started {
            self.stage_timings.cpu_tier1 += elapsed_us(started);
        }
        if let Some(counters) = &cpu_tier1_counters {
            counters.add_to_stage_timings(self.stage_timings);
        }
        let upload_started = self.profile_stages.then(Instant::now);
        let buffer =
            upload_cpu_decoded_coefficients(self.runtime, &coefficients, self.retained_buffers)?;
        if let Some(started) = upload_started {
            self.stage_timings.coefficient_upload += elapsed_us(started);
        }
        Ok(buffer)
    }

    pub(super) fn submit_ht_sub_band(
        &mut self,
        first: &PreparedDirectGrayscalePlan,
        step_idx: usize,
        sub_band: &PreparedHtSubBand,
    ) -> Result<(), Error> {
        let input_count = planned_cpu_input_count(
            self.tier1_mode,
            self.flattened_cpu_tier1_cache.is_some(),
            self.broadcast_tier1_inputs,
            self.plans.len(),
        );
        let mut metadata_budget =
            BatchMetadataBudget::new("J2K Metal stacked HT sub-band submission metadata");
        metadata_budget.preflight(&[
            BatchMetadataRequest::of::<&PreparedHtSubBand>(self.plans.len()),
            BatchMetadataRequest::of::<HtCpuDecodeInput<'_>>(input_count),
        ])?;
        let sub_bands = try_collect_submission_items(
            &mut metadata_budget,
            self.plans
                .iter()
                .map(|plan| match plan.steps.get(step_idx) {
                    Some(PreparedDirectGrayscaleStep::HtSubBand(other)) => Ok(other),
                    _ => Err(direct_preflight_invariant(
                        "HT sub-band step mismatch in stacked component batch",
                    )),
                }),
            "J2K Metal stacked HT sub-band references",
        )?;
        let span = checked_f32_dimension_span(
            sub_band.width,
            sub_band.height,
            self.count,
            "J2K MetalDirect stacked HT sub-band",
        )?;
        let buffer = match self.tier1_mode {
            DirectTier1Mode::Metal => {
                let output = take_f32_scratch_buffer(self.runtime, span.total_elements)?;
                let (buffers, status_check) = if let Some(encoder) = self.compute_encoder {
                    encode_distinct_ht_sub_bands_to_buffer_in_encoder(
                        self.runtime,
                        encoder,
                        &sub_bands,
                        &output.buffer,
                    )?
                } else {
                    encode_distinct_ht_sub_bands_to_buffer_in_command_buffer(
                        self.runtime,
                        self.command_buffers.default,
                        &sub_bands,
                        &output.buffer,
                    )?
                };
                if let Some(encoder) = self.compute_encoder {
                    encoder.memory_barrier_with_resources(&[&output.buffer]);
                }
                retain_metal_tier1_output(
                    output,
                    buffers,
                    status_check,
                    self.retained_buffers,
                    self.status_checks,
                    self.scratch_buffers,
                )?
            }
            DirectTier1Mode::CpuUpload => self.prepare_ht_sub_band_cpu_buffer(
                first,
                step_idx,
                span.per_instance_elements,
                &sub_bands,
                &mut metadata_budget,
            )?,
        };

        for (instance_idx, bands) in self.resources.band_sets.iter_mut().enumerate() {
            let source_idx = if self.broadcast_tier1_inputs {
                0
            } else {
                instance_idx
            };
            let source_sub_band = sub_bands[source_idx];
            bands.push(DirectBandSlice {
                band_id: source_sub_band.band_id,
                buffer: buffer.clone(),
                offset_bytes: checked_f32_instance_offset(
                    &span,
                    source_idx,
                    "J2K MetalDirect stacked HT sub-band",
                )?,
                window: BandRequiredRegion::full(source_sub_band.width, source_sub_band.height),
            });
        }
        Ok(())
    }

    fn prepare_ht_sub_band_cpu_buffer(
        &mut self,
        first: &PreparedDirectGrayscalePlan,
        step_idx: usize,
        per_instance_len: usize,
        sub_bands: &[&PreparedHtSubBand],
        metadata_budget: &mut BatchMetadataBudget,
    ) -> Result<Buffer, Error> {
        let input_sub_bands = if self.broadcast_tier1_inputs {
            &sub_bands[..1]
        } else {
            sub_bands
        };
        if let Some(cache) = self.flattened_cpu_tier1_cache {
            return cache.buffer_for(
                self.component_idx,
                step_idx,
                per_instance_len,
                input_sub_bands.len(),
            );
        }

        let inputs = try_collect_submission_items(
            metadata_budget,
            input_sub_bands.iter().map(|sub_band| {
                Ok(HtCpuDecodeInput {
                    payload_source: &sub_band.payload_source,
                    jobs: &sub_band.jobs,
                    output_len: per_instance_len,
                })
            }),
            "J2K Metal stacked HT sub-band CPU inputs",
        )?;
        let decode_started = self.profile_stages.then(Instant::now);
        let cpu_tier1_counters = self
            .profile_stages
            .then(CpuTier1DecodeSubstageCounters::default);
        let coefficients = decode_ht_inputs_on_cpu_with_plan_cache(
            first,
            step_idx,
            &inputs,
            cpu_tier1_counters.as_ref(),
        )?;
        if let Some(started) = decode_started {
            self.stage_timings.cpu_tier1 += elapsed_us(started);
        }
        if let Some(counters) = &cpu_tier1_counters {
            counters.add_to_stage_timings(self.stage_timings);
        }
        let upload_started = self.profile_stages.then(Instant::now);
        let buffer =
            upload_cpu_decoded_coefficients(self.runtime, &coefficients, self.retained_buffers)?;
        if let Some(started) = upload_started {
            self.stage_timings.coefficient_upload += elapsed_us(started);
        }
        Ok(buffer)
    }
}
