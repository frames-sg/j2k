// SPDX-License-Identifier: MIT OR Apache-2.0

use std::mem::size_of;

use metal::Buffer;

use super::super::{
    decode_classic_inputs_on_cpu_with_plan_cache, decode_ht_inputs_on_cpu_with_plan_cache,
    direct_preflight_invariant,
    dispatch_irreversible97_single_decomposition_buffers_in_command_buffer_with_offsets,
    dispatch_reversible53_repeated_buffers_in_command_buffer_with_offsets,
    dispatch_store_component_repeated_in_command_buffer, elapsed_us,
    encode_distinct_classic_sub_band_groups_to_buffer_in_command_buffer,
    encode_distinct_classic_sub_bands_to_buffer_in_command_buffer,
    encode_distinct_ht_sub_band_groups_to_buffer_in_command_buffer,
    encode_distinct_ht_sub_bands_to_buffer_in_command_buffer, idwt_input_windows_from_slices,
    metal_profile_stages_enabled, prepared_idwt_output_len, prepared_idwt_params,
    repeated_idwt_params, take_f32_scratch_buffer, upload_cpu_decoded_coefficients,
    BandRequiredRegion, ClassicCpuDecodeInput, CpuTier1DecodeSubstageCounters,
    DirectColorBatchCommandBuffers, DirectHybridStageTimings, DirectScratchBuffer,
    DirectStatusCheck, DirectTier1Mode, Error, FlattenedCpuTier1Cache, HtCpuDecodeInput,
    IdwtSubBandBuffers, Instant, J2kRepeatedStoreParams, J2kWaveletTransform, MetalRuntime,
    PreparedClassicSubBand, PreparedClassicSubBandGroup, PreparedDirectGrayscalePlan,
    PreparedDirectGrayscaleStep, PreparedDirectIdwt, PreparedHtSubBand, PreparedHtSubBandGroup,
    PreparedIdwtInputStrides, RepeatedIdwtDispatch, SingleIdwtDispatch,
};
use super::resources::{
    lookup_direct_band_slice_entry, lookup_repeated_direct_band_layout_entry,
    retain_metal_tier1_output, DirectBandSlice, StackedComponentResources,
};
use super::validation::StackedComponentBatchPlan;
use super::StackedDirectComponentPlaneBatchRequest;

struct SubmissionContext<'a, 'p, 'r> {
    runtime: &'a MetalRuntime,
    command_buffers: DirectColorBatchCommandBuffers<'a>,
    plans: &'a [&'p PreparedDirectGrayscalePlan],
    component_idx: usize,
    flattened_cpu_tier1_cache: Option<&'a FlattenedCpuTier1Cache>,
    tier1_mode: DirectTier1Mode,
    stage_timings: &'a mut DirectHybridStageTimings,
    retained_buffers: &'a mut Vec<Buffer>,
    retained_cpu_coefficients: &'a mut Vec<Vec<f32>>,
    status_checks: &'a mut Vec<DirectStatusCheck>,
    scratch_buffers: &'a mut Vec<DirectScratchBuffer>,
    count: usize,
    broadcast_tier1_inputs: bool,
    profile_stages: bool,
    resources: &'r mut StackedComponentResources,
}

pub(super) fn submit_stacked_component_commands<'p>(
    request: StackedDirectComponentPlaneBatchRequest<'_, 'p>,
    plan: &StackedComponentBatchPlan<'p>,
    resources: &mut StackedComponentResources,
) -> Result<(), Error> {
    let StackedDirectComponentPlaneBatchRequest {
        runtime,
        command_buffers,
        plans,
        component_idx,
        flattened_cpu_tier1_cache,
        tier1_mode,
        stage_timings,
        retained_buffers,
        retained_cpu_coefficients,
        status_checks,
        scratch_buffers,
    } = request;
    let profile_stages = tier1_mode == DirectTier1Mode::CpuUpload && metal_profile_stages_enabled();
    let mut context = SubmissionContext {
        runtime,
        command_buffers,
        plans,
        component_idx,
        flattened_cpu_tier1_cache,
        tier1_mode,
        stage_timings,
        retained_buffers,
        retained_cpu_coefficients,
        status_checks,
        scratch_buffers,
        count: plan.count,
        broadcast_tier1_inputs: plan.broadcast_tier1_inputs,
        profile_stages,
        resources,
    };

    let first = plan.first;
    let mut step_idx = 0;
    while step_idx < first.steps.len() {
        if let Some(group) = first.classic_group_starting_at(step_idx) {
            context.submit_classic_group(first, step_idx, group)?;
            step_idx = group.end_step;
            continue;
        }
        if let Some(group) = first.ht_group_starting_at(step_idx) {
            context.submit_ht_group(first, step_idx, group)?;
            step_idx = group.end_step;
            continue;
        }

        match &first.steps[step_idx] {
            PreparedDirectGrayscaleStep::ClassicSubBand(sub_band) => {
                context.submit_classic_sub_band(first, step_idx, sub_band)?;
            }
            PreparedDirectGrayscaleStep::HtSubBand(sub_band) => {
                context.submit_ht_sub_band(first, step_idx, sub_band)?;
            }
            PreparedDirectGrayscaleStep::Idwt(idwt) => {
                context.submit_idwt(step_idx, idwt)?;
            }
            PreparedDirectGrayscaleStep::Store(store) => {
                context.submit_store(store)?;
            }
        }
        step_idx += 1;
    }

    Ok(())
}

impl SubmissionContext<'_, '_, '_> {
    fn submit_classic_group(
        &mut self,
        first: &PreparedDirectGrayscalePlan,
        step_idx: usize,
        group: &PreparedClassicSubBandGroup,
    ) -> Result<(), Error> {
        let groups = self
            .plans
            .iter()
            .map(|plan| {
                plan.classic_group_starting_at(step_idx)
                    .expect("preflight validated classic group")
            })
            .collect::<Vec<_>>();
        let buffer = match self.tier1_mode {
            DirectTier1Mode::Metal => {
                let output =
                    take_f32_scratch_buffer(self.runtime, group.total_coefficients * self.count)?;
                let (buffers, status_check) =
                    encode_distinct_classic_sub_band_groups_to_buffer_in_command_buffer(
                        self.runtime,
                        self.command_buffers.default,
                        &groups,
                        &output.buffer,
                        self.scratch_buffers,
                    )?;
                retain_metal_tier1_output(
                    output,
                    buffers,
                    status_check,
                    self.retained_buffers,
                    self.status_checks,
                    self.scratch_buffers,
                )
            }
            DirectTier1Mode::CpuUpload => {
                self.prepare_classic_group_cpu_buffer(first, step_idx, group, &groups)?
            }
        };

        let stride_bytes = group.total_coefficients * size_of::<f32>();
        for (instance_idx, bands) in self.resources.band_sets.iter_mut().enumerate() {
            let source_group = if self.broadcast_tier1_inputs {
                groups[0]
            } else {
                groups[instance_idx]
            };
            let instance_offset = if self.broadcast_tier1_inputs {
                0
            } else {
                instance_idx * stride_bytes
            };
            for member in &source_group.members {
                bands.push(DirectBandSlice {
                    band_id: member.band_id,
                    buffer: buffer.clone(),
                    offset_bytes: instance_offset + member.offset_elements * size_of::<f32>(),
                    window: member.window,
                });
            }
        }
        Ok(())
    }

    fn prepare_classic_group_cpu_buffer(
        &mut self,
        first: &PreparedDirectGrayscalePlan,
        step_idx: usize,
        group: &PreparedClassicSubBandGroup,
        groups: &[&PreparedClassicSubBandGroup],
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

        let inputs = input_groups
            .iter()
            .map(|group| ClassicCpuDecodeInput {
                coded_data: &group.coded_data,
                segments: &group.segments,
                jobs: &group.jobs,
                output_len: group.total_coefficients,
            })
            .collect::<Vec<_>>();
        let decode_started = self.profile_stages.then(Instant::now);
        let cpu_tier1_counters = self
            .profile_stages
            .then(CpuTier1DecodeSubstageCounters::default);
        let coefficients = decode_classic_inputs_on_cpu_with_plan_cache(
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
        let buffer = upload_cpu_decoded_coefficients(
            self.runtime,
            coefficients,
            self.retained_buffers,
            self.retained_cpu_coefficients,
        );
        if let Some(started) = upload_started {
            self.stage_timings.coefficient_upload += elapsed_us(started);
        }
        Ok(buffer)
    }

    fn submit_ht_group(
        &mut self,
        first: &PreparedDirectGrayscalePlan,
        step_idx: usize,
        group: &PreparedHtSubBandGroup,
    ) -> Result<(), Error> {
        let groups = self
            .plans
            .iter()
            .map(|plan| {
                plan.ht_group_starting_at(step_idx)
                    .expect("preflight validated HT group")
            })
            .collect::<Vec<_>>();
        let buffer = match self.tier1_mode {
            DirectTier1Mode::Metal => {
                let output =
                    take_f32_scratch_buffer(self.runtime, group.total_coefficients * self.count)?;
                let (buffers, status_check) =
                    encode_distinct_ht_sub_band_groups_to_buffer_in_command_buffer(
                        self.runtime,
                        self.command_buffers.default,
                        &groups,
                        &output.buffer,
                    )?;
                retain_metal_tier1_output(
                    output,
                    buffers,
                    status_check,
                    self.retained_buffers,
                    self.status_checks,
                    self.scratch_buffers,
                )
            }
            DirectTier1Mode::CpuUpload => {
                self.prepare_ht_group_cpu_buffer(first, step_idx, group, &groups)?
            }
        };

        let stride_bytes = group.total_coefficients * size_of::<f32>();
        for (instance_idx, bands) in self.resources.band_sets.iter_mut().enumerate() {
            let source_group = if self.broadcast_tier1_inputs {
                groups[0]
            } else {
                groups[instance_idx]
            };
            let instance_offset = if self.broadcast_tier1_inputs {
                0
            } else {
                instance_idx * stride_bytes
            };
            for member in &source_group.members {
                bands.push(DirectBandSlice {
                    band_id: member.band_id,
                    buffer: buffer.clone(),
                    offset_bytes: instance_offset + member.offset_elements * size_of::<f32>(),
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

        let inputs = input_groups
            .iter()
            .map(|group| HtCpuDecodeInput {
                coded_data: &group.coded_arena.data,
                jobs: &group.jobs,
                output_len: group.total_coefficients,
            })
            .collect::<Vec<_>>();
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
        let buffer = upload_cpu_decoded_coefficients(
            self.runtime,
            coefficients,
            self.retained_buffers,
            self.retained_cpu_coefficients,
        );
        if let Some(started) = upload_started {
            self.stage_timings.coefficient_upload += elapsed_us(started);
        }
        Ok(buffer)
    }

    fn submit_classic_sub_band(
        &mut self,
        first: &PreparedDirectGrayscalePlan,
        step_idx: usize,
        sub_band: &PreparedClassicSubBand,
    ) -> Result<(), Error> {
        let sub_bands = self
            .plans
            .iter()
            .map(|plan| match &plan.steps[step_idx] {
                PreparedDirectGrayscaleStep::ClassicSubBand(other) => Ok(other),
                _ => Err(direct_preflight_invariant(
                    "classic sub-band step mismatch in stacked component batch",
                )),
            })
            .collect::<Result<Vec<_>, Error>>()?;
        let per_instance_len = sub_band.width as usize * sub_band.height as usize;
        let buffer = match self.tier1_mode {
            DirectTier1Mode::Metal => {
                let output = take_f32_scratch_buffer(self.runtime, per_instance_len * self.count)?;
                let (buffers, status_check) =
                    encode_distinct_classic_sub_bands_to_buffer_in_command_buffer(
                        self.runtime,
                        self.command_buffers.default,
                        &sub_bands,
                        &output.buffer,
                        self.scratch_buffers,
                    )?;
                retain_metal_tier1_output(
                    output,
                    buffers,
                    status_check,
                    self.retained_buffers,
                    self.status_checks,
                    self.scratch_buffers,
                )
            }
            DirectTier1Mode::CpuUpload => self.prepare_classic_sub_band_cpu_buffer(
                first,
                step_idx,
                per_instance_len,
                &sub_bands,
            )?,
        };

        let stride_bytes = per_instance_len * size_of::<f32>();
        for (instance_idx, bands) in self.resources.band_sets.iter_mut().enumerate() {
            let source_sub_band = if self.broadcast_tier1_inputs {
                sub_bands[0]
            } else {
                sub_bands[instance_idx]
            };
            let instance_offset = if self.broadcast_tier1_inputs {
                0
            } else {
                instance_idx * stride_bytes
            };
            bands.push(DirectBandSlice {
                band_id: source_sub_band.band_id,
                buffer: buffer.clone(),
                offset_bytes: instance_offset,
                window: BandRequiredRegion::full(source_sub_band.width, source_sub_band.height),
            });
        }
        Ok(())
    }

    fn prepare_classic_sub_band_cpu_buffer(
        &mut self,
        first: &PreparedDirectGrayscalePlan,
        step_idx: usize,
        per_instance_len: usize,
        sub_bands: &[&PreparedClassicSubBand],
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

        let inputs = input_sub_bands
            .iter()
            .map(|sub_band| ClassicCpuDecodeInput {
                coded_data: &sub_band.coded_data,
                segments: &sub_band.segments,
                jobs: &sub_band.jobs,
                output_len: per_instance_len,
            })
            .collect::<Vec<_>>();
        let decode_started = self.profile_stages.then(Instant::now);
        let cpu_tier1_counters = self
            .profile_stages
            .then(CpuTier1DecodeSubstageCounters::default);
        let coefficients = decode_classic_inputs_on_cpu_with_plan_cache(
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
        let buffer = upload_cpu_decoded_coefficients(
            self.runtime,
            coefficients,
            self.retained_buffers,
            self.retained_cpu_coefficients,
        );
        if let Some(started) = upload_started {
            self.stage_timings.coefficient_upload += elapsed_us(started);
        }
        Ok(buffer)
    }

    fn submit_ht_sub_band(
        &mut self,
        first: &PreparedDirectGrayscalePlan,
        step_idx: usize,
        sub_band: &PreparedHtSubBand,
    ) -> Result<(), Error> {
        let sub_bands = self
            .plans
            .iter()
            .map(|plan| match &plan.steps[step_idx] {
                PreparedDirectGrayscaleStep::HtSubBand(other) => Ok(other),
                _ => Err(direct_preflight_invariant(
                    "HT sub-band step mismatch in stacked component batch",
                )),
            })
            .collect::<Result<Vec<_>, Error>>()?;
        let per_instance_len = sub_band.width as usize * sub_band.height as usize;
        let buffer = match self.tier1_mode {
            DirectTier1Mode::Metal => {
                let output = take_f32_scratch_buffer(self.runtime, per_instance_len * self.count)?;
                let (buffers, status_check) =
                    encode_distinct_ht_sub_bands_to_buffer_in_command_buffer(
                        self.runtime,
                        self.command_buffers.default,
                        &sub_bands,
                        &output.buffer,
                    )?;
                retain_metal_tier1_output(
                    output,
                    buffers,
                    status_check,
                    self.retained_buffers,
                    self.status_checks,
                    self.scratch_buffers,
                )
            }
            DirectTier1Mode::CpuUpload => {
                self.prepare_ht_sub_band_cpu_buffer(first, step_idx, per_instance_len, &sub_bands)?
            }
        };

        let stride_bytes = per_instance_len * size_of::<f32>();
        for (instance_idx, bands) in self.resources.band_sets.iter_mut().enumerate() {
            let source_sub_band = if self.broadcast_tier1_inputs {
                sub_bands[0]
            } else {
                sub_bands[instance_idx]
            };
            let instance_offset = if self.broadcast_tier1_inputs {
                0
            } else {
                instance_idx * stride_bytes
            };
            bands.push(DirectBandSlice {
                band_id: source_sub_band.band_id,
                buffer: buffer.clone(),
                offset_bytes: instance_offset,
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

        let inputs = input_sub_bands
            .iter()
            .map(|sub_band| HtCpuDecodeInput {
                coded_data: &sub_band.coded_data,
                jobs: &sub_band.jobs,
                output_len: per_instance_len,
            })
            .collect::<Vec<_>>();
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
        let buffer = upload_cpu_decoded_coefficients(
            self.runtime,
            coefficients,
            self.retained_buffers,
            self.retained_cpu_coefficients,
        );
        if let Some(started) = upload_started {
            self.stage_timings.coefficient_upload += elapsed_us(started);
        }
        Ok(buffer)
    }

    fn submit_idwt(&mut self, step_idx: usize, idwt: &PreparedDirectIdwt) -> Result<(), Error> {
        let per_instance_len = prepared_idwt_output_len(idwt);
        let output = take_f32_scratch_buffer(self.runtime, per_instance_len * self.count)?;
        let encode_started = self.profile_stages.then(Instant::now);
        match idwt.step.transform {
            J2kWaveletTransform::Reversible53 => {
                let (ll, low_low_stride) = lookup_repeated_direct_band_layout_entry(
                    &self.resources.band_sets,
                    idwt.step.ll_band_id,
                    idwt.step.ll,
                )?;
                let (hl, high_low_stride) = lookup_repeated_direct_band_layout_entry(
                    &self.resources.band_sets,
                    idwt.step.hl_band_id,
                    idwt.step.hl,
                )?;
                let (lh, low_high_stride) = lookup_repeated_direct_band_layout_entry(
                    &self.resources.band_sets,
                    idwt.step.lh_band_id,
                    idwt.step.lh,
                )?;
                let (hh, high_high_stride) = lookup_repeated_direct_band_layout_entry(
                    &self.resources.band_sets,
                    idwt.step.hh_band_id,
                    idwt.step.hh,
                )?;
                let params = repeated_idwt_params(
                    idwt,
                    idwt_input_windows_from_slices(&ll, &hl, &lh, &hh),
                    PreparedIdwtInputStrides {
                        ll: low_low_stride,
                        hl: high_low_stride,
                        lh: low_high_stride,
                        hh: high_high_stride,
                    },
                    self.count,
                    "color",
                )?;
                dispatch_reversible53_repeated_buffers_in_command_buffer_with_offsets(
                    self.command_buffers.idwt,
                    RepeatedIdwtDispatch {
                        runtime: self.runtime,
                        sub_bands: IdwtSubBandBuffers {
                            ll: &ll.buffer,
                            ll_offset: ll.offset_bytes,
                            hl: &hl.buffer,
                            hl_offset: hl.offset_bytes,
                            lh: &lh.buffer,
                            lh_offset: lh.offset_bytes,
                            hh: &hh.buffer,
                            hh_offset: hh.offset_bytes,
                        },
                        params,
                        decoded: &output.buffer,
                    },
                );
            }
            J2kWaveletTransform::Irreversible97 => {
                for (instance_idx, bands) in self.resources.band_sets.iter().enumerate() {
                    let PreparedDirectGrayscaleStep::Idwt(step) =
                        &self.plans[instance_idx].steps[step_idx]
                    else {
                        return Err(direct_preflight_invariant(
                            "IDWT step mismatch in stacked component batch",
                        ));
                    };
                    let ll =
                        lookup_direct_band_slice_entry(bands, step.step.ll_band_id, step.step.ll)?;
                    let hl =
                        lookup_direct_band_slice_entry(bands, step.step.hl_band_id, step.step.hl)?;
                    let lh =
                        lookup_direct_band_slice_entry(bands, step.step.lh_band_id, step.step.lh)?;
                    let hh =
                        lookup_direct_band_slice_entry(bands, step.step.hh_band_id, step.step.hh)?;
                    let params = prepared_idwt_params(
                        step,
                        idwt_input_windows_from_slices(&ll, &hl, &lh, &hh),
                    );
                    self.status_checks.push(
                        dispatch_irreversible97_single_decomposition_buffers_in_command_buffer_with_offsets(
                            self.command_buffers.idwt.interleave,
                            SingleIdwtDispatch {
                                runtime: self.runtime,
                                sub_bands: IdwtSubBandBuffers {
                                    ll: &ll.buffer,
                                    ll_offset: ll.offset_bytes,
                                    hl: &hl.buffer,
                                    hl_offset: hl.offset_bytes,
                                    lh: &lh.buffer,
                                    lh_offset: lh.offset_bytes,
                                    hh: &hh.buffer,
                                    hh_offset: hh.offset_bytes,
                                },
                                params,
                                decoded: &output.buffer,
                                decoded_offset: instance_idx
                                    * per_instance_len
                                    * size_of::<f32>(),
                            },
                        ),
                    );
                }
            }
        }
        if let Some(started) = encode_started {
            self.stage_timings.metal_idwt_encode += elapsed_us(started);
        }

        let stride_bytes = per_instance_len * size_of::<f32>();
        for (instance_idx, bands) in self.resources.band_sets.iter_mut().enumerate() {
            let PreparedDirectGrayscaleStep::Idwt(step) = &self.plans[instance_idx].steps[step_idx]
            else {
                return Err(direct_preflight_invariant(
                    "IDWT output step mismatch in stacked component batch",
                ));
            };
            bands.push(DirectBandSlice {
                band_id: step.step.output_band_id,
                buffer: output.buffer.clone(),
                offset_bytes: instance_idx * stride_bytes,
                window: step.output_window,
            });
        }
        self.scratch_buffers.push(output);
        Ok(())
    }

    fn submit_store(&mut self, store: &j2k_native::J2kDirectStoreStep) -> Result<(), Error> {
        let (input, input_instance_stride) = lookup_repeated_direct_band_layout_entry(
            &self.resources.band_sets,
            store.input_band_id,
            store.input_rect,
        )?;
        let per_instance_len = store.output_width as usize * store.output_height as usize;
        let output = take_f32_scratch_buffer(self.runtime, per_instance_len * self.count)?;
        let encode_started = self.profile_stages.then(Instant::now);
        dispatch_store_component_repeated_in_command_buffer(
            self.runtime,
            self.command_buffers.store,
            &input.buffer,
            input.offset_bytes,
            &output.buffer,
            J2kRepeatedStoreParams {
                input_width: store.input_rect.width(),
                input_height: store.input_rect.height(),
                input_instance_stride,
                source_x: store.source_x,
                source_y: store.source_y,
                copy_width: store.copy_width,
                copy_height: store.copy_height,
                output_width: store.output_width,
                output_height: store.output_height,
                output_x: store.output_x,
                output_y: store.output_y,
                addend: store.addend,
                batch_count: u32::try_from(self.count).map_err(|_| Error::MetalKernel {
                    message: "J2K MetalDirect color store batch count exceeds u32".to_string(),
                })?,
            },
        );
        if let Some(started) = encode_started {
            self.stage_timings.metal_store_encode += elapsed_us(started);
        }
        self.resources.final_plane = Some(output.buffer.clone());
        self.scratch_buffers.push(output);
        Ok(())
    }
}
