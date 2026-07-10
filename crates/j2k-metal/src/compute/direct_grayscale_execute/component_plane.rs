// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_native::J2kDirectStoreStep;
use metal::ComputeCommandEncoderRef;

use super::{
    borrow_mut_slice_buffer, decode_prepared_classic_sub_band_group_on_cpu_profile,
    decode_prepared_classic_sub_band_on_cpu_profile,
    decode_prepared_ht_sub_band_group_on_cpu_profile, decode_prepared_ht_sub_band_on_cpu_profile,
    dispatch_irreversible97_single_decomposition_buffers_in_encoder_with_offsets,
    dispatch_reversible53_single_decomposition_buffers_in_encoder_with_offsets,
    dispatch_store_component_buffer_in_encoder_with_offsets, elapsed_us,
    encode_prepared_classic_sub_band_group_to_buffer_in_encoder,
    encode_prepared_classic_sub_band_to_buffer_in_encoder,
    encode_prepared_ht_sub_band_group_to_buffer_in_encoder,
    encode_prepared_ht_sub_band_to_buffer_in_encoder, hybrid_stage_signpost,
    idwt_input_windows_from_slices, lookup_direct_band_slice, lookup_direct_band_slice_entry,
    metal_profile_stages_enabled, prepared_idwt_output_len, prepared_idwt_params, size_of,
    take_f32_scratch_buffer, BandRequiredRegion, Buffer, CommandBufferRef,
    CpuTier1DecodeSubstageCounters, DirectBandSlice, DirectHybridStageTimings, DirectScratchBuffer,
    DirectStatusCheck, DirectTier1Mode, Error, IdwtSubBandBuffers, Instant, J2kStoreParams,
    J2kWaveletTransform, MetalRuntime, PreparedDirectGrayscalePlan, PreparedDirectGrayscaleStep,
    SingleIdwtDispatch, SIGNPOST_DECODE_HYBRID_COEFFICIENT_UPLOAD,
};
use crate::compute::{
    PreparedClassicSubBand, PreparedClassicSubBandGroup, PreparedDirectIdwt, PreparedHtSubBand,
    PreparedHtSubBandGroup,
};

#[cfg(target_os = "macos")]
pub(in crate::compute) fn checked_coefficient_len(
    width: u32,
    height: u32,
    message: &str,
) -> Result<usize, Error> {
    (width as usize)
        .checked_mul(height as usize)
        .ok_or_else(|| Error::MetalKernel {
            message: message.to_string(),
        })
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn upload_cpu_decoded_coefficients(
    runtime: &MetalRuntime,
    mut coefficients: Vec<f32>,
    retained_buffers: &mut Vec<Buffer>,
    retained_cpu_coefficients: &mut Vec<Vec<f32>>,
) -> Buffer {
    let _signpost = hybrid_stage_signpost(SIGNPOST_DECODE_HYBRID_COEFFICIENT_UPLOAD);
    let buffer = borrow_mut_slice_buffer(&runtime.device, &mut coefficients);
    retained_buffers.push(buffer.clone());
    retained_cpu_coefficients.push(coefficients);
    buffer
}

#[cfg(target_os = "macos")]
pub(in crate::compute) struct DirectComponentPlaneRequest<'a> {
    pub(in crate::compute) runtime: &'a MetalRuntime,
    pub(in crate::compute) command_buffer: &'a CommandBufferRef,
    pub(in crate::compute) plan: &'a PreparedDirectGrayscalePlan,
    pub(in crate::compute) tier1_mode: DirectTier1Mode,
    pub(in crate::compute) stage_timings: &'a mut DirectHybridStageTimings,
    pub(in crate::compute) retained_buffers: &'a mut Vec<Buffer>,
    pub(in crate::compute) retained_cpu_coefficients: &'a mut Vec<Vec<f32>>,
    pub(in crate::compute) status_checks: &'a mut Vec<DirectStatusCheck>,
    pub(in crate::compute) scratch_buffers: &'a mut Vec<DirectScratchBuffer>,
}

struct ComponentPlaneExecution<'a> {
    runtime: &'a MetalRuntime,
    encoder: &'a ComputeCommandEncoderRef,
    tier1_mode: DirectTier1Mode,
    profile_stages: bool,
    stage_timings: &'a mut DirectHybridStageTimings,
    retained_buffers: &'a mut Vec<Buffer>,
    retained_cpu_coefficients: &'a mut Vec<Vec<f32>>,
    status_checks: &'a mut Vec<DirectStatusCheck>,
    scratch_buffers: &'a mut Vec<DirectScratchBuffer>,
    bands: Vec<DirectBandSlice>,
    final_plane: Option<Buffer>,
}

impl ComponentPlaneExecution<'_> {
    fn decode_and_upload_cpu(
        &mut self,
        decode: impl FnOnce(Option<&CpuTier1DecodeSubstageCounters>) -> Result<Vec<f32>, Error>,
    ) -> Result<Buffer, Error> {
        let decode_started = self.profile_stages.then(Instant::now);
        let cpu_tier1_counters = self
            .profile_stages
            .then(CpuTier1DecodeSubstageCounters::default);
        let coefficients = decode(cpu_tier1_counters.as_ref())?;
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

    fn encode_classic_group(&mut self, group: &PreparedClassicSubBandGroup) -> Result<(), Error> {
        let buffer = match self.tier1_mode {
            DirectTier1Mode::Metal => {
                let output = take_f32_scratch_buffer(self.runtime, group.total_coefficients)?;
                let (buffers, status_check) =
                    encode_prepared_classic_sub_band_group_to_buffer_in_encoder(
                        self.runtime,
                        self.encoder,
                        group,
                        &output.buffer,
                        self.scratch_buffers,
                    )?;
                self.retained_buffers.extend(buffers);
                self.status_checks.push(status_check);
                let buffer = output.buffer.clone();
                self.scratch_buffers.push(output);
                buffer
            }
            DirectTier1Mode::CpuUpload => self.decode_and_upload_cpu(|counters| {
                decode_prepared_classic_sub_band_group_on_cpu_profile(group, counters)
            })?,
        };
        for member in &group.members {
            self.bands.push(DirectBandSlice {
                band_id: member.band_id,
                buffer: buffer.clone(),
                offset_bytes: member.offset_elements * size_of::<f32>(),
                window: member.window,
            });
        }
        Ok(())
    }

    fn encode_ht_group(&mut self, group: &PreparedHtSubBandGroup) -> Result<(), Error> {
        let buffer = match self.tier1_mode {
            DirectTier1Mode::Metal => {
                let output = take_f32_scratch_buffer(self.runtime, group.total_coefficients)?;
                let (buffers, status_check) =
                    encode_prepared_ht_sub_band_group_to_buffer_in_encoder(
                        self.runtime,
                        self.encoder,
                        group,
                        &output.buffer,
                    )?;
                self.retained_buffers.extend(buffers);
                self.status_checks.push(status_check);
                let buffer = output.buffer.clone();
                self.scratch_buffers.push(output);
                buffer
            }
            DirectTier1Mode::CpuUpload => self.decode_and_upload_cpu(|counters| {
                decode_prepared_ht_sub_band_group_on_cpu_profile(group, counters)
            })?,
        };
        for member in &group.members {
            self.bands.push(DirectBandSlice {
                band_id: member.band_id,
                buffer: buffer.clone(),
                offset_bytes: member.offset_elements * size_of::<f32>(),
                window: member.window,
            });
        }
        Ok(())
    }

    fn encode_classic_sub_band(&mut self, sub_band: &PreparedClassicSubBand) -> Result<(), Error> {
        let buffer = match self.tier1_mode {
            DirectTier1Mode::Metal => {
                let output = take_f32_scratch_buffer(
                    self.runtime,
                    sub_band.width as usize * sub_band.height as usize,
                )?;
                let (buffers, status_check) =
                    encode_prepared_classic_sub_band_to_buffer_in_encoder(
                        self.runtime,
                        self.encoder,
                        sub_band,
                        &output.buffer,
                        self.scratch_buffers,
                    )?;
                self.retained_buffers.extend(buffers);
                self.status_checks.push(status_check);
                let buffer = output.buffer.clone();
                self.scratch_buffers.push(output);
                buffer
            }
            DirectTier1Mode::CpuUpload => self.decode_and_upload_cpu(|counters| {
                decode_prepared_classic_sub_band_on_cpu_profile(sub_band, counters)
            })?,
        };
        self.bands.push(DirectBandSlice {
            band_id: sub_band.band_id,
            buffer,
            offset_bytes: 0,
            window: BandRequiredRegion::full(sub_band.width, sub_band.height),
        });
        Ok(())
    }

    fn encode_ht_sub_band(&mut self, sub_band: &PreparedHtSubBand) -> Result<(), Error> {
        let buffer = match self.tier1_mode {
            DirectTier1Mode::Metal => {
                let output = take_f32_scratch_buffer(
                    self.runtime,
                    sub_band.width as usize * sub_band.height as usize,
                )?;
                let (buffers, status_check) = encode_prepared_ht_sub_band_to_buffer_in_encoder(
                    self.runtime,
                    self.encoder,
                    sub_band,
                    &output.buffer,
                )?;
                self.retained_buffers.extend(buffers);
                self.status_checks.push(status_check);
                let buffer = output.buffer.clone();
                self.scratch_buffers.push(output);
                buffer
            }
            DirectTier1Mode::CpuUpload => self.decode_and_upload_cpu(|counters| {
                decode_prepared_ht_sub_band_on_cpu_profile(sub_band, counters)
            })?,
        };
        self.bands.push(DirectBandSlice {
            band_id: sub_band.band_id,
            buffer,
            offset_bytes: 0,
            window: BandRequiredRegion::full(sub_band.width, sub_band.height),
        });
        Ok(())
    }

    fn encode_idwt(&mut self, idwt: &PreparedDirectIdwt) -> Result<(), Error> {
        let ll = lookup_direct_band_slice_entry(&self.bands, idwt.step.ll_band_id, idwt.step.ll)?;
        let hl = lookup_direct_band_slice_entry(&self.bands, idwt.step.hl_band_id, idwt.step.hl)?;
        let lh = lookup_direct_band_slice_entry(&self.bands, idwt.step.lh_band_id, idwt.step.lh)?;
        let hh = lookup_direct_band_slice_entry(&self.bands, idwt.step.hh_band_id, idwt.step.hh)?;
        let params = prepared_idwt_params(idwt, idwt_input_windows_from_slices(&ll, &hl, &lh, &hh));
        let output = take_f32_scratch_buffer(self.runtime, prepared_idwt_output_len(idwt))?;
        let encode_started = self.profile_stages.then(Instant::now);
        let dispatch = SingleIdwtDispatch {
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
            decoded_offset: 0,
        };
        match idwt.step.transform {
            J2kWaveletTransform::Reversible53 => {
                dispatch_reversible53_single_decomposition_buffers_in_encoder_with_offsets(
                    self.encoder,
                    dispatch,
                );
            }
            J2kWaveletTransform::Irreversible97 => {
                self.status_checks.push(
                    dispatch_irreversible97_single_decomposition_buffers_in_encoder_with_offsets(
                        self.encoder,
                        dispatch,
                    ),
                );
            }
        }
        if let Some(started) = encode_started {
            self.stage_timings.metal_idwt_encode += elapsed_us(started);
        }
        self.bands.push(DirectBandSlice {
            band_id: idwt.step.output_band_id,
            buffer: output.buffer.clone(),
            offset_bytes: 0,
            window: idwt.output_window,
        });
        self.scratch_buffers.push(output);
        Ok(())
    }

    fn encode_store(&mut self, store: &J2kDirectStoreStep) -> Result<(), Error> {
        let (input, input_offset) =
            lookup_direct_band_slice(&self.bands, store.input_band_id, store.input_rect)?;
        let output = take_f32_scratch_buffer(
            self.runtime,
            store.output_width as usize * store.output_height as usize,
        )?;
        let encode_started = self.profile_stages.then(Instant::now);
        dispatch_store_component_buffer_in_encoder_with_offsets(
            self.runtime,
            self.encoder,
            &input,
            input_offset,
            &output.buffer,
            0,
            J2kStoreParams {
                input_width: store.input_rect.width(),
                source_x: store.source_x,
                source_y: store.source_y,
                copy_width: store.copy_width,
                copy_height: store.copy_height,
                output_width: store.output_width,
                output_x: store.output_x,
                output_y: store.output_y,
                addend: store.addend,
            },
        );
        if let Some(started) = encode_started {
            self.stage_timings.metal_store_encode += elapsed_us(started);
        }
        self.final_plane = Some(output.buffer.clone());
        self.scratch_buffers.push(output);
        Ok(())
    }

    fn encode_step(&mut self, step: &PreparedDirectGrayscaleStep) -> Result<(), Error> {
        match step {
            PreparedDirectGrayscaleStep::ClassicSubBand(sub_band) => {
                self.encode_classic_sub_band(sub_band)
            }
            PreparedDirectGrayscaleStep::HtSubBand(sub_band) => self.encode_ht_sub_band(sub_band),
            PreparedDirectGrayscaleStep::Idwt(idwt) => self.encode_idwt(idwt),
            PreparedDirectGrayscaleStep::Store(store) => self.encode_store(store),
        }
    }

    fn finish(self) -> Result<Buffer, Error> {
        self.final_plane.ok_or_else(|| Error::MetalKernel {
            message: "J2K MetalDirect component plan did not produce a stored plane".to_string(),
        })
    }
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn encode_prepared_direct_component_plane_in_command_buffer(
    request: DirectComponentPlaneRequest<'_>,
) -> Result<Buffer, Error> {
    let DirectComponentPlaneRequest {
        runtime,
        command_buffer,
        plan,
        tier1_mode,
        stage_timings,
        retained_buffers,
        retained_cpu_coefficients,
        status_checks,
        scratch_buffers,
    } = request;
    let encoder = command_buffer.new_compute_command_encoder();
    let mut execution = ComponentPlaneExecution {
        runtime,
        encoder,
        tier1_mode,
        profile_stages: metal_profile_stages_enabled(),
        stage_timings,
        retained_buffers,
        retained_cpu_coefficients,
        status_checks,
        scratch_buffers,
        bands: Vec::new(),
        final_plane: None,
    };
    let result = (|| {
        let mut step_idx = 0;
        while step_idx < plan.steps.len() {
            if let Some(group) = plan.classic_group_starting_at(step_idx) {
                execution.encode_classic_group(group)?;
                step_idx = group.end_step;
                continue;
            }
            if let Some(group) = plan.ht_group_starting_at(step_idx) {
                execution.encode_ht_group(group)?;
                step_idx = group.end_step;
                continue;
            }
            execution.encode_step(&plan.steps[step_idx])?;
            step_idx += 1;
        }
        execution.finish()
    })();
    encoder.end_encoding();
    result
}
