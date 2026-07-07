// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{MetalRuntime, CommandBufferRef, PreparedDirectGrayscalePlan, PixelFormat, Buffer, DirectStatusCheck, DirectScratchBuffer, Surface, Error, DirectBandSlice, take_f32_scratch_buffer, encode_prepared_classic_sub_band_group_to_buffer_in_encoder, size_of, encode_prepared_ht_sub_band_group_to_buffer_in_encoder, PreparedDirectGrayscaleStep, encode_prepared_classic_sub_band_to_buffer_in_encoder, BandRequiredRegion, encode_prepared_ht_sub_band_to_buffer_in_encoder, lookup_direct_band_slice_entry, prepared_idwt_params, idwt_input_windows_from_slices, prepared_idwt_output_len, J2kWaveletTransform, IdwtSubBandBuffers, SingleIdwtDispatch, dispatch_reversible53_single_decomposition_buffers_in_encoder_with_offsets, dispatch_irreversible97_single_decomposition_buffers_in_encoder_with_offsets, lookup_direct_band_slice, j2k_scalar_pack_params, encode_gray_store_to_surface_in_encoder, J2kGrayStoreParams, J2kStoreParams, dispatch_store_component_buffer_in_encoder_with_offsets, encode_gray_plane_to_surface_in_encoder, hybrid_stage_signpost, SIGNPOST_DECODE_HYBRID_COEFFICIENT_UPLOAD, borrow_mut_slice_buffer, DirectTier1Mode, DirectHybridStageTimings, metal_profile_stages_enabled, Instant, CpuTier1DecodeSubstageCounters, decode_prepared_classic_sub_band_group_on_cpu_profile, elapsed_us, decode_prepared_ht_sub_band_group_on_cpu_profile, decode_prepared_classic_sub_band_on_cpu_profile, decode_prepared_ht_sub_band_on_cpu_profile, with_runtime, encode_repeated_direct_grayscale_plan_in_command_buffer, RepeatedDirectGrayscalePlanRequest, commit_and_wait_metal, validate_direct_status, recycle_scratch_buffers, Device, with_runtime_for_device, Arc, supports_stacked_direct_component_plane_batch, encode_stacked_direct_component_plane_batch, StackedDirectComponentPlaneBatchRequest, DirectColorBatchCommandBuffers, encode_repeated_gray_plane_to_surfaces_in_command_buffer, PreparedDirectColorPlan, prepared_direct_color_plan_supports_runtime, metal_profile_decode_split_commands_enabled, DecodeHybridSplitCommandBuffers, try_encode_stacked_mct_rgb8_direct_color_batch, StackedDirectColorBatchRequest, SIGNPOST_DECODE_HYBRID_COMMAND_WAIT, wait_for_completion_metal, record_completed_decode_split_gpu_stages, emit_direct_hybrid_stage_timings, label_command_buffer, completed_command_buffer_gpu_duration, encode_prepared_direct_color_plan_in_command_buffer, DirectColorPlanRequest};

#[cfg(target_os = "macos")]
pub(super) fn encode_prepared_direct_grayscale_plan_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    plan: &PreparedDirectGrayscalePlan,
    fmt: PixelFormat,
    retained_buffers: &mut Vec<Buffer>,
    status_checks: &mut Vec<DirectStatusCheck>,
    scratch_buffers: &mut Vec<DirectScratchBuffer>,
) -> Result<Surface, Error> {
    let encoder = command_buffer.new_compute_command_encoder();
    let result = (|| {
        let mut bands = Vec::<DirectBandSlice>::new();
        let mut final_surface = None;
        let mut step_idx = 0;

        while step_idx < plan.steps.len() {
            if let Some(group) = plan.classic_group_starting_at(step_idx) {
                let output = take_f32_scratch_buffer(runtime, group.total_coefficients)?;
                let (buffers, status_check) =
                    encode_prepared_classic_sub_band_group_to_buffer_in_encoder(
                        runtime,
                        encoder,
                        group,
                        &output.buffer,
                        scratch_buffers,
                    )?;
                retained_buffers.extend(buffers);
                status_checks.push(status_check);
                for member in &group.members {
                    bands.push(DirectBandSlice {
                        band_id: member.band_id,
                        buffer: output.buffer.clone(),
                        offset_bytes: member.offset_elements * size_of::<f32>(),
                        window: member.window,
                    });
                }
                scratch_buffers.push(output);
                step_idx = group.end_step;
                continue;
            }

            if let Some(group) = plan.ht_group_starting_at(step_idx) {
                let output = take_f32_scratch_buffer(runtime, group.total_coefficients)?;
                let (buffers, status_check) =
                    encode_prepared_ht_sub_band_group_to_buffer_in_encoder(
                        runtime,
                        encoder,
                        group,
                        &output.buffer,
                    )?;
                retained_buffers.extend(buffers);
                status_checks.push(status_check);
                for member in &group.members {
                    bands.push(DirectBandSlice {
                        band_id: member.band_id,
                        buffer: output.buffer.clone(),
                        offset_bytes: member.offset_elements * size_of::<f32>(),
                        window: member.window,
                    });
                }
                scratch_buffers.push(output);
                step_idx = group.end_step;
                continue;
            }

            let step = &plan.steps[step_idx];
            match step {
                PreparedDirectGrayscaleStep::ClassicSubBand(sub_band) => {
                    let output = take_f32_scratch_buffer(
                        runtime,
                        sub_band.width as usize * sub_band.height as usize,
                    )?;
                    let (buffers, status_check) =
                        encode_prepared_classic_sub_band_to_buffer_in_encoder(
                            runtime,
                            encoder,
                            sub_band,
                            &output.buffer,
                            scratch_buffers,
                        )?;
                    retained_buffers.extend(buffers);
                    status_checks.push(status_check);
                    bands.push(DirectBandSlice {
                        band_id: sub_band.band_id,
                        buffer: output.buffer.clone(),
                        offset_bytes: 0,
                        window: BandRequiredRegion::full(sub_band.width, sub_band.height),
                    });
                    scratch_buffers.push(output);
                }
                PreparedDirectGrayscaleStep::HtSubBand(sub_band) => {
                    let output = take_f32_scratch_buffer(
                        runtime,
                        sub_band.width as usize * sub_band.height as usize,
                    )?;
                    let (buffers, status_check) = encode_prepared_ht_sub_band_to_buffer_in_encoder(
                        runtime,
                        encoder,
                        sub_band,
                        &output.buffer,
                    )?;
                    retained_buffers.extend(buffers);
                    status_checks.push(status_check);
                    bands.push(DirectBandSlice {
                        band_id: sub_band.band_id,
                        buffer: output.buffer.clone(),
                        offset_bytes: 0,
                        window: BandRequiredRegion::full(sub_band.width, sub_band.height),
                    });
                    scratch_buffers.push(output);
                }
                PreparedDirectGrayscaleStep::Idwt(idwt) => {
                    let ll =
                        lookup_direct_band_slice_entry(&bands, idwt.step.ll_band_id, idwt.step.ll)?;
                    let hl =
                        lookup_direct_band_slice_entry(&bands, idwt.step.hl_band_id, idwt.step.hl)?;
                    let lh =
                        lookup_direct_band_slice_entry(&bands, idwt.step.lh_band_id, idwt.step.lh)?;
                    let hh =
                        lookup_direct_band_slice_entry(&bands, idwt.step.hh_band_id, idwt.step.hh)?;
                    let params = prepared_idwt_params(
                        idwt,
                        idwt_input_windows_from_slices(&ll, &hl, &lh, &hh),
                    );
                    let output = take_f32_scratch_buffer(runtime, prepared_idwt_output_len(idwt))?;
                    match idwt.step.transform {
                        J2kWaveletTransform::Reversible53 => {
                            dispatch_reversible53_single_decomposition_buffers_in_encoder_with_offsets(
                                encoder,
                                SingleIdwtDispatch {
                                    runtime,
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
                                },
                            );
                        }
                        J2kWaveletTransform::Irreversible97 => {
                            let status_check =
                                dispatch_irreversible97_single_decomposition_buffers_in_encoder_with_offsets(
                                    encoder,
                                    SingleIdwtDispatch {
                                        runtime,
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
                                    },
                                );
                            status_checks.push(status_check);
                        }
                    }
                    bands.push(DirectBandSlice {
                        band_id: idwt.step.output_band_id,
                        buffer: output.buffer.clone(),
                        offset_bytes: 0,
                        window: idwt.output_window,
                    });
                    scratch_buffers.push(output);
                }
                PreparedDirectGrayscaleStep::Store(store) => {
                    let (input, input_offset) =
                        lookup_direct_band_slice(&bands, store.input_band_id, store.input_rect)?;
                    if matches!(fmt, PixelFormat::Gray8 | PixelFormat::Gray16) {
                        let scale = j2k_scalar_pack_params(u32::from(plan.bit_depth));
                        final_surface = Some(encode_gray_store_to_surface_in_encoder(
                            runtime,
                            encoder,
                            &input,
                            input_offset,
                            J2kGrayStoreParams {
                                input_width: store.input_rect.width(),
                                source_x: store.source_x,
                                source_y: store.source_y,
                                copy_width: store.copy_width,
                                copy_height: store.copy_height,
                                output_width: store.output_width,
                                output_x: store.output_x,
                                output_y: store.output_y,
                                addend: store.addend,
                                max_value: scale.max_value,
                                u8_scale: scale.u8_scale,
                                u16_scale: scale.u16_scale,
                            },
                            plan.dimensions,
                            fmt,
                        )?);
                    } else {
                        let output = take_f32_scratch_buffer(
                            runtime,
                            store.output_width as usize * store.output_height as usize,
                        )?;
                        let params = J2kStoreParams {
                            input_width: store.input_rect.width(),
                            source_x: store.source_x,
                            source_y: store.source_y,
                            copy_width: store.copy_width,
                            copy_height: store.copy_height,
                            output_width: store.output_width,
                            output_x: store.output_x,
                            output_y: store.output_y,
                            addend: store.addend,
                        };
                        dispatch_store_component_buffer_in_encoder_with_offsets(
                            runtime,
                            encoder,
                            &input,
                            input_offset,
                            &output.buffer,
                            0,
                            params,
                        );
                        retained_buffers.push(output.buffer.clone());
                        final_surface = Some(encode_gray_plane_to_surface_in_encoder(
                            runtime,
                            encoder,
                            &output.buffer,
                            plan.dimensions,
                            plan.bit_depth,
                            fmt,
                        )?);
                        scratch_buffers.push(output);
                    }
                }
            }
            step_idx += 1;
        }

        final_surface.ok_or_else(|| Error::MetalKernel {
            message: "J2K MetalDirect prepared grayscale plan did not produce a final stored plane"
                .to_string(),
        })
    })();
    encoder.end_encoding();
    result
}

#[cfg(target_os = "macos")]
pub(super) fn checked_coefficient_len(width: u32, height: u32, message: &str) -> Result<usize, Error> {
    (width as usize)
        .checked_mul(height as usize)
        .ok_or_else(|| Error::MetalKernel {
            message: message.to_string(),
        })
}

#[cfg(target_os = "macos")]
pub(super) fn upload_cpu_decoded_coefficients(
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
pub(super) struct DirectComponentPlaneRequest<'a> {
    pub(super) runtime: &'a MetalRuntime,
    pub(super) command_buffer: &'a CommandBufferRef,
    pub(super) plan: &'a PreparedDirectGrayscalePlan,
    pub(super) tier1_mode: DirectTier1Mode,
    pub(super) stage_timings: &'a mut DirectHybridStageTimings,
    pub(super) retained_buffers: &'a mut Vec<Buffer>,
    pub(super) retained_cpu_coefficients: &'a mut Vec<Vec<f32>>,
    pub(super) status_checks: &'a mut Vec<DirectStatusCheck>,
    pub(super) scratch_buffers: &'a mut Vec<DirectScratchBuffer>,
}

#[cfg(target_os = "macos")]
pub(super) fn encode_prepared_direct_component_plane_in_command_buffer(
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
    let result = (|| {
        let mut bands = Vec::<DirectBandSlice>::new();
        let mut final_plane = None;
        let mut step_idx = 0;
        let profile_stages = metal_profile_stages_enabled();

        while step_idx < plan.steps.len() {
            if let Some(group) = plan.classic_group_starting_at(step_idx) {
                let buffer = match tier1_mode {
                    DirectTier1Mode::Metal => {
                        let output = take_f32_scratch_buffer(runtime, group.total_coefficients)?;
                        let (buffers, status_check) =
                            encode_prepared_classic_sub_band_group_to_buffer_in_encoder(
                                runtime,
                                encoder,
                                group,
                                &output.buffer,
                                scratch_buffers,
                            )?;
                        retained_buffers.extend(buffers);
                        status_checks.push(status_check);
                        let buffer = output.buffer.clone();
                        scratch_buffers.push(output);
                        buffer
                    }
                    DirectTier1Mode::CpuUpload => {
                        let decode_started = profile_stages.then(Instant::now);
                        let cpu_tier1_counters =
                            profile_stages.then(CpuTier1DecodeSubstageCounters::default);
                        let coefficients = decode_prepared_classic_sub_band_group_on_cpu_profile(
                            group,
                            cpu_tier1_counters.as_ref(),
                        )?;
                        if let Some(started) = decode_started {
                            stage_timings.cpu_tier1 += elapsed_us(started);
                        }
                        if let Some(counters) = &cpu_tier1_counters {
                            counters.add_to_stage_timings(stage_timings);
                        }
                        let upload_started = profile_stages.then(Instant::now);
                        let buffer = upload_cpu_decoded_coefficients(
                            runtime,
                            coefficients,
                            retained_buffers,
                            retained_cpu_coefficients,
                        );
                        if let Some(started) = upload_started {
                            stage_timings.coefficient_upload += elapsed_us(started);
                        }
                        buffer
                    }
                };
                for member in &group.members {
                    bands.push(DirectBandSlice {
                        band_id: member.band_id,
                        buffer: buffer.clone(),
                        offset_bytes: member.offset_elements * size_of::<f32>(),
                        window: member.window,
                    });
                }
                step_idx = group.end_step;
                continue;
            }

            if let Some(group) = plan.ht_group_starting_at(step_idx) {
                let buffer = match tier1_mode {
                    DirectTier1Mode::Metal => {
                        let output = take_f32_scratch_buffer(runtime, group.total_coefficients)?;
                        let (buffers, status_check) =
                            encode_prepared_ht_sub_band_group_to_buffer_in_encoder(
                                runtime,
                                encoder,
                                group,
                                &output.buffer,
                            )?;
                        retained_buffers.extend(buffers);
                        status_checks.push(status_check);
                        let buffer = output.buffer.clone();
                        scratch_buffers.push(output);
                        buffer
                    }
                    DirectTier1Mode::CpuUpload => {
                        let decode_started = profile_stages.then(Instant::now);
                        let cpu_tier1_counters =
                            profile_stages.then(CpuTier1DecodeSubstageCounters::default);
                        let coefficients = decode_prepared_ht_sub_band_group_on_cpu_profile(
                            group,
                            cpu_tier1_counters.as_ref(),
                        )?;
                        if let Some(started) = decode_started {
                            stage_timings.cpu_tier1 += elapsed_us(started);
                        }
                        if let Some(counters) = &cpu_tier1_counters {
                            counters.add_to_stage_timings(stage_timings);
                        }
                        let upload_started = profile_stages.then(Instant::now);
                        let buffer = upload_cpu_decoded_coefficients(
                            runtime,
                            coefficients,
                            retained_buffers,
                            retained_cpu_coefficients,
                        );
                        if let Some(started) = upload_started {
                            stage_timings.coefficient_upload += elapsed_us(started);
                        }
                        buffer
                    }
                };
                for member in &group.members {
                    bands.push(DirectBandSlice {
                        band_id: member.band_id,
                        buffer: buffer.clone(),
                        offset_bytes: member.offset_elements * size_of::<f32>(),
                        window: member.window,
                    });
                }
                step_idx = group.end_step;
                continue;
            }

            match &plan.steps[step_idx] {
                PreparedDirectGrayscaleStep::ClassicSubBand(sub_band) => {
                    let buffer = match tier1_mode {
                        DirectTier1Mode::Metal => {
                            let output = take_f32_scratch_buffer(
                                runtime,
                                sub_band.width as usize * sub_band.height as usize,
                            )?;
                            let (buffers, status_check) =
                                encode_prepared_classic_sub_band_to_buffer_in_encoder(
                                    runtime,
                                    encoder,
                                    sub_band,
                                    &output.buffer,
                                    scratch_buffers,
                                )?;
                            retained_buffers.extend(buffers);
                            status_checks.push(status_check);
                            let buffer = output.buffer.clone();
                            scratch_buffers.push(output);
                            buffer
                        }
                        DirectTier1Mode::CpuUpload => {
                            let decode_started = profile_stages.then(Instant::now);
                            let cpu_tier1_counters =
                                profile_stages.then(CpuTier1DecodeSubstageCounters::default);
                            let coefficients = decode_prepared_classic_sub_band_on_cpu_profile(
                                sub_band,
                                cpu_tier1_counters.as_ref(),
                            )?;
                            if let Some(started) = decode_started {
                                stage_timings.cpu_tier1 += elapsed_us(started);
                            }
                            if let Some(counters) = &cpu_tier1_counters {
                                counters.add_to_stage_timings(stage_timings);
                            }
                            let upload_started = profile_stages.then(Instant::now);
                            let buffer = upload_cpu_decoded_coefficients(
                                runtime,
                                coefficients,
                                retained_buffers,
                                retained_cpu_coefficients,
                            );
                            if let Some(started) = upload_started {
                                stage_timings.coefficient_upload += elapsed_us(started);
                            }
                            buffer
                        }
                    };
                    bands.push(DirectBandSlice {
                        band_id: sub_band.band_id,
                        buffer,
                        offset_bytes: 0,
                        window: BandRequiredRegion::full(sub_band.width, sub_band.height),
                    });
                }
                PreparedDirectGrayscaleStep::HtSubBand(sub_band) => {
                    let buffer = match tier1_mode {
                        DirectTier1Mode::Metal => {
                            let output = take_f32_scratch_buffer(
                                runtime,
                                sub_band.width as usize * sub_band.height as usize,
                            )?;
                            let (buffers, status_check) =
                                encode_prepared_ht_sub_band_to_buffer_in_encoder(
                                    runtime,
                                    encoder,
                                    sub_band,
                                    &output.buffer,
                                )?;
                            retained_buffers.extend(buffers);
                            status_checks.push(status_check);
                            let buffer = output.buffer.clone();
                            scratch_buffers.push(output);
                            buffer
                        }
                        DirectTier1Mode::CpuUpload => {
                            let decode_started = profile_stages.then(Instant::now);
                            let cpu_tier1_counters =
                                profile_stages.then(CpuTier1DecodeSubstageCounters::default);
                            let coefficients = decode_prepared_ht_sub_band_on_cpu_profile(
                                sub_band,
                                cpu_tier1_counters.as_ref(),
                            )?;
                            if let Some(started) = decode_started {
                                stage_timings.cpu_tier1 += elapsed_us(started);
                            }
                            if let Some(counters) = &cpu_tier1_counters {
                                counters.add_to_stage_timings(stage_timings);
                            }
                            let upload_started = profile_stages.then(Instant::now);
                            let buffer = upload_cpu_decoded_coefficients(
                                runtime,
                                coefficients,
                                retained_buffers,
                                retained_cpu_coefficients,
                            );
                            if let Some(started) = upload_started {
                                stage_timings.coefficient_upload += elapsed_us(started);
                            }
                            buffer
                        }
                    };
                    bands.push(DirectBandSlice {
                        band_id: sub_band.band_id,
                        buffer,
                        offset_bytes: 0,
                        window: BandRequiredRegion::full(sub_band.width, sub_band.height),
                    });
                }
                PreparedDirectGrayscaleStep::Idwt(idwt) => {
                    let ll =
                        lookup_direct_band_slice_entry(&bands, idwt.step.ll_band_id, idwt.step.ll)?;
                    let hl =
                        lookup_direct_band_slice_entry(&bands, idwt.step.hl_band_id, idwt.step.hl)?;
                    let lh =
                        lookup_direct_band_slice_entry(&bands, idwt.step.lh_band_id, idwt.step.lh)?;
                    let hh =
                        lookup_direct_band_slice_entry(&bands, idwt.step.hh_band_id, idwt.step.hh)?;
                    let params = prepared_idwt_params(
                        idwt,
                        idwt_input_windows_from_slices(&ll, &hl, &lh, &hh),
                    );
                    let output = take_f32_scratch_buffer(runtime, prepared_idwt_output_len(idwt))?;
                    let encode_started = profile_stages.then(Instant::now);
                    match idwt.step.transform {
                        J2kWaveletTransform::Reversible53 => {
                            dispatch_reversible53_single_decomposition_buffers_in_encoder_with_offsets(
                                encoder,
                                SingleIdwtDispatch {
                                    runtime,
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
                                },
                            );
                        }
                        J2kWaveletTransform::Irreversible97 => {
                            status_checks.push(
                                dispatch_irreversible97_single_decomposition_buffers_in_encoder_with_offsets(
                                    encoder,
                                    SingleIdwtDispatch {
                                        runtime,
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
                                    },
                                ),
                            );
                        }
                    }
                    if let Some(started) = encode_started {
                        stage_timings.metal_idwt_encode += elapsed_us(started);
                    }
                    bands.push(DirectBandSlice {
                        band_id: idwt.step.output_band_id,
                        buffer: output.buffer.clone(),
                        offset_bytes: 0,
                        window: idwt.output_window,
                    });
                    scratch_buffers.push(output);
                }
                PreparedDirectGrayscaleStep::Store(store) => {
                    let (input, input_offset) =
                        lookup_direct_band_slice(&bands, store.input_band_id, store.input_rect)?;
                    let output = take_f32_scratch_buffer(
                        runtime,
                        store.output_width as usize * store.output_height as usize,
                    )?;
                    let encode_started = profile_stages.then(Instant::now);
                    dispatch_store_component_buffer_in_encoder_with_offsets(
                        runtime,
                        encoder,
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
                        stage_timings.metal_store_encode += elapsed_us(started);
                    }
                    final_plane = Some(output.buffer.clone());
                    scratch_buffers.push(output);
                }
            }
            step_idx += 1;
        }

        final_plane.ok_or_else(|| Error::MetalKernel {
            message: "J2K MetalDirect component plan did not produce a stored plane".to_string(),
        })
    })();
    encoder.end_encoding();
    result
}

#[cfg(target_os = "macos")]
pub(crate) fn execute_repeated_prepared_direct_grayscale_plan(
    plan: &PreparedDirectGrayscalePlan,
    fmt: PixelFormat,
    count: usize,
) -> Result<Vec<Surface>, Error> {
    with_runtime(|runtime| {
        let command_buffer = runtime.queue.new_command_buffer();
        let mut retained_buffers = Vec::new();
        let mut status_checks = Vec::new();
        let mut scratch_buffers = Vec::new();
        let surfaces = encode_repeated_direct_grayscale_plan_in_command_buffer(
            RepeatedDirectGrayscalePlanRequest {
            runtime,
            command_buffer,
            plan,
            fmt,
            count,
            retained_buffers: &mut retained_buffers,
            status_checks: &mut status_checks,
            scratch_buffers: &mut scratch_buffers,
        })?;
        commit_and_wait_metal(command_buffer)?;
        for status_check in status_checks {
            validate_direct_status(status_check)?;
        }
        drop(retained_buffers);
        recycle_scratch_buffers(runtime, scratch_buffers)?;
        Ok(surfaces)
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn execute_prepared_direct_grayscale_plan(
    plan: &PreparedDirectGrayscalePlan,
    fmt: PixelFormat,
) -> Result<Surface, Error> {
    with_runtime(|runtime| {
        let command_buffer = runtime.queue.new_command_buffer();
        let mut retained_buffers = Vec::new();
        let mut status_checks = Vec::new();
        let mut scratch_buffers = Vec::new();
        let surface = encode_prepared_direct_grayscale_plan_in_command_buffer(
            runtime,
            command_buffer,
            plan,
            fmt,
            &mut retained_buffers,
            &mut status_checks,
            &mut scratch_buffers,
        )?;
        commit_and_wait_metal(command_buffer)?;
        for status_check in status_checks {
            validate_direct_status(status_check)?;
        }
        drop(retained_buffers);
        recycle_scratch_buffers(runtime, scratch_buffers)?;
        Ok(surface)
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn execute_prepared_direct_grayscale_plan_with_device(
    plan: &PreparedDirectGrayscalePlan,
    fmt: PixelFormat,
    device: &Device,
) -> Result<Surface, Error> {
    with_runtime_for_device(device, |_| {
        execute_prepared_direct_grayscale_plan(plan, fmt)
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn execute_prepared_direct_grayscale_plan_batch(
    plans: &[Arc<PreparedDirectGrayscalePlan>],
    fmt: PixelFormat,
) -> Result<Vec<Surface>, Error> {
    if plans.is_empty() {
        return Ok(Vec::new());
    }

    with_runtime(|runtime| {
        let command_buffer = runtime.queue.new_command_buffer();
        let mut retained_buffers = Vec::new();
        let mut retained_cpu_coefficients = Vec::<Vec<f32>>::new();
        let mut status_checks = Vec::new();
        let mut scratch_buffers = Vec::new();
        let mut stage_timings = DirectHybridStageTimings::default();
        let mut surfaces = Vec::with_capacity(plans.len());

        let component_plan_refs = plans.iter().map(Arc::as_ref).collect::<Vec<_>>();
        if plans.len() > 1 && supports_stacked_direct_component_plane_batch(&component_plan_refs) {
            let stacked_plane = encode_stacked_direct_component_plane_batch(
                StackedDirectComponentPlaneBatchRequest {
                runtime,
                command_buffers: DirectColorBatchCommandBuffers::single(command_buffer),
                plans: &component_plan_refs,
                component_idx: 0,
                flattened_cpu_tier1_cache: None,
                tier1_mode: DirectTier1Mode::Metal,
                stage_timings: &mut stage_timings,
                retained_buffers: &mut retained_buffers,
                retained_cpu_coefficients: &mut retained_cpu_coefficients,
                status_checks: &mut status_checks,
                scratch_buffers: &mut scratch_buffers,
            })?;
            let first = plans.first().expect("plans is not empty");
            if stacked_plane.dimensions == first.dimensions && stacked_plane.count == plans.len() {
                surfaces = encode_repeated_gray_plane_to_surfaces_in_command_buffer(
                    runtime,
                    command_buffer,
                    &stacked_plane.buffer,
                    first.dimensions,
                    first.bit_depth,
                    fmt,
                    plans.len(),
                )?;
            }
        }

        for plan in plans {
            if !surfaces.is_empty() {
                break;
            }
            surfaces.push(encode_prepared_direct_grayscale_plan_in_command_buffer(
                runtime,
                command_buffer,
                plan,
                fmt,
                &mut retained_buffers,
                &mut status_checks,
                &mut scratch_buffers,
            )?);
        }

        commit_and_wait_metal(command_buffer)?;
        for status_check in status_checks {
            validate_direct_status(status_check)?;
        }
        drop(retained_buffers);
        drop(retained_cpu_coefficients);
        recycle_scratch_buffers(runtime, scratch_buffers)?;
        Ok(surfaces)
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn execute_prepared_direct_color_plan(
    plan: &PreparedDirectColorPlan,
    fmt: PixelFormat,
) -> Result<Surface, Error> {
    let plans = [Arc::new(plan.clone())];
    let mut surfaces = execute_prepared_direct_color_plan_batch(&plans, fmt)?;
    surfaces.pop().ok_or_else(|| Error::MetalKernel {
        message: "J2K MetalDirect color plan produced no surface".to_string(),
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn execute_prepared_direct_color_plan_with_device(
    plan: &PreparedDirectColorPlan,
    fmt: PixelFormat,
    device: &Device,
) -> Result<Surface, Error> {
    with_runtime_for_device(device, |_| execute_prepared_direct_color_plan(plan, fmt))
}

#[cfg(target_os = "macos")]
pub(crate) fn execute_prepared_direct_color_plan_batch(
    plans: &[Arc<PreparedDirectColorPlan>],
    fmt: PixelFormat,
) -> Result<Vec<Surface>, Error> {
    execute_direct_color_plan_batch_with_tier1(plans, fmt, DirectTier1Mode::Metal)
}

#[cfg(target_os = "macos")]
pub(crate) fn execute_hybrid_cpu_tier1_direct_color_plan(
    plan: &PreparedDirectColorPlan,
    fmt: PixelFormat,
) -> Result<Surface, Error> {
    let plans = [Arc::new(plan.clone())];
    let mut surfaces = execute_hybrid_cpu_tier1_direct_color_plan_batch(&plans, fmt)?;
    surfaces.pop().ok_or_else(|| Error::MetalKernel {
        message: "J2K MetalDirect hybrid color plan produced no surface".to_string(),
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn execute_hybrid_cpu_tier1_direct_color_plan_with_device(
    plan: &PreparedDirectColorPlan,
    fmt: PixelFormat,
    device: &Device,
) -> Result<Surface, Error> {
    with_runtime_for_device(device, |_| {
        execute_hybrid_cpu_tier1_direct_color_plan(plan, fmt)
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn execute_hybrid_cpu_tier1_direct_color_plan_batch(
    plans: &[Arc<PreparedDirectColorPlan>],
    fmt: PixelFormat,
) -> Result<Vec<Surface>, Error> {
    execute_direct_color_plan_batch_with_tier1(plans, fmt, DirectTier1Mode::CpuUpload)
}

#[cfg(target_os = "macos")]
pub(super) fn execute_direct_color_plan_batch_with_tier1(
    plans: &[Arc<PreparedDirectColorPlan>],
    fmt: PixelFormat,
    tier1_mode: DirectTier1Mode,
) -> Result<Vec<Surface>, Error> {
    execute_direct_color_plan_batch_with_tier1_options(plans, fmt, tier1_mode, false)
}

#[cfg(all(target_os = "macos", test))]
pub(super) fn execute_flattened_hybrid_cpu_tier1_direct_color_plan_batch_for_test(
    plans: &[Arc<PreparedDirectColorPlan>],
    fmt: PixelFormat,
) -> Result<Vec<Surface>, Error> {
    execute_direct_color_plan_batch_with_tier1_options(plans, fmt, DirectTier1Mode::CpuUpload, true)
}

#[cfg(target_os = "macos")]
pub(super) fn execute_direct_color_plan_batch_with_tier1_options(
    plans: &[Arc<PreparedDirectColorPlan>],
    fmt: PixelFormat,
    tier1_mode: DirectTier1Mode,
    force_flattened_cpu_tier1: bool,
) -> Result<Vec<Surface>, Error> {
    if plans.is_empty() {
        return Ok(Vec::new());
    }
    if tier1_mode == DirectTier1Mode::Metal
        && plans
            .iter()
            .any(|plan| !prepared_direct_color_plan_supports_runtime(plan, fmt))
    {
        return Err(Error::MetalDirectFallback {
            message: "unsupported classic kernel input in direct component plan".to_string(),
            reason: crate::MetalDirectFallbackReason::UnsupportedRuntimeInput,
        });
    }

    with_runtime(|runtime| {
        let mut retained_buffers = Vec::new();
        let mut retained_cpu_coefficients = Vec::<Vec<f32>>::new();
        let mut status_checks = Vec::new();
        let mut scratch_buffers = Vec::new();
        let mut stage_timings = DirectHybridStageTimings::default();
        let profile_hybrid_stages =
            tier1_mode == DirectTier1Mode::CpuUpload && metal_profile_stages_enabled();

        if fmt == PixelFormat::Rgb8
            && profile_hybrid_stages
            && metal_profile_decode_split_commands_enabled()
        {
            let split_command_buffers = DecodeHybridSplitCommandBuffers::new(runtime);
            if let Some(surfaces) = try_encode_stacked_mct_rgb8_direct_color_batch(
                StackedDirectColorBatchRequest {
                runtime,
                command_buffers: split_command_buffers.refs(),
                plans,
                tier1_mode,
                force_flattened_cpu_tier1,
                stage_timings: &mut stage_timings,
                retained_buffers: &mut retained_buffers,
                retained_cpu_coefficients: &mut retained_cpu_coefficients,
                status_checks: &mut status_checks,
                scratch_buffers: &mut scratch_buffers,
            })? {
                split_command_buffers.commit_in_order();
                let wait_started = Instant::now();
                let _wait_signpost = hybrid_stage_signpost(SIGNPOST_DECODE_HYBRID_COMMAND_WAIT);
                wait_for_completion_metal(&split_command_buffers.mct_pack)?;
                stage_timings.command_wait += elapsed_us(wait_started);
                record_completed_decode_split_gpu_stages(
                    &mut stage_timings,
                    &split_command_buffers,
                );
                for status_check in status_checks {
                    validate_direct_status(status_check)?;
                }
                emit_direct_hybrid_stage_timings(&stage_timings, fmt, plans.len());
                drop(retained_buffers);
                drop(retained_cpu_coefficients);
                recycle_scratch_buffers(runtime, scratch_buffers)?;
                return Ok(surfaces);
            }

            drop(split_command_buffers);
            retained_buffers.clear();
            retained_cpu_coefficients.clear();
            status_checks.clear();
            scratch_buffers.clear();
            stage_timings = DirectHybridStageTimings::default();
        }

        let command_buffer = runtime.queue.new_command_buffer();
        if profile_hybrid_stages {
            label_command_buffer(command_buffer, "j2k decode hybrid direct color batch");
        }

        if fmt == PixelFormat::Rgb8 {
            if let Some(surfaces) = try_encode_stacked_mct_rgb8_direct_color_batch(
                StackedDirectColorBatchRequest {
                runtime,
                command_buffers: DirectColorBatchCommandBuffers::single(command_buffer),
                plans,
                tier1_mode,
                force_flattened_cpu_tier1,
                stage_timings: &mut stage_timings,
                retained_buffers: &mut retained_buffers,
                retained_cpu_coefficients: &mut retained_cpu_coefficients,
                status_checks: &mut status_checks,
                scratch_buffers: &mut scratch_buffers,
            })? {
                command_buffer.commit();
                let wait_started = profile_hybrid_stages.then(Instant::now);
                let _wait_signpost = hybrid_stage_signpost(SIGNPOST_DECODE_HYBRID_COMMAND_WAIT);
                wait_for_completion_metal(command_buffer)?;
                if let Some(started) = wait_started {
                    stage_timings.command_wait += elapsed_us(started);
                }
                if profile_hybrid_stages {
                    if let Some(duration) = completed_command_buffer_gpu_duration(command_buffer) {
                        stage_timings.gpu_command += duration.as_micros();
                    }
                }
                for status_check in status_checks {
                    validate_direct_status(status_check)?;
                }
                if tier1_mode == DirectTier1Mode::CpuUpload {
                    emit_direct_hybrid_stage_timings(&stage_timings, fmt, plans.len());
                }
                drop(retained_buffers);
                drop(retained_cpu_coefficients);
                recycle_scratch_buffers(runtime, scratch_buffers)?;
                return Ok(surfaces);
            }
        }

        let mut surfaces = Vec::with_capacity(plans.len());

        for plan in plans {
            let surface = encode_prepared_direct_color_plan_in_command_buffer(
                DirectColorPlanRequest {
                runtime,
                command_buffer,
                plan,
                fmt,
                tier1_mode,
                stage_timings: &mut stage_timings,
                retained_buffers: &mut retained_buffers,
                retained_cpu_coefficients: &mut retained_cpu_coefficients,
                status_checks: &mut status_checks,
                scratch_buffers: &mut scratch_buffers,
            })?;
            surfaces.push(surface);
        }

        command_buffer.commit();
        let wait_started = profile_hybrid_stages.then(Instant::now);
        let _wait_signpost = hybrid_stage_signpost(SIGNPOST_DECODE_HYBRID_COMMAND_WAIT);
        wait_for_completion_metal(command_buffer)?;
        if let Some(started) = wait_started {
            stage_timings.command_wait += elapsed_us(started);
        }
        if profile_hybrid_stages {
            if let Some(duration) = completed_command_buffer_gpu_duration(command_buffer) {
                stage_timings.gpu_command += duration.as_micros();
            }
        }
        for status_check in status_checks {
            validate_direct_status(status_check)?;
        }
        if tier1_mode == DirectTier1Mode::CpuUpload {
            emit_direct_hybrid_stage_timings(&stage_timings, fmt, plans.len());
        }
        drop(retained_buffers);
        drop(retained_cpu_coefficients);
        recycle_scratch_buffers(runtime, scratch_buffers)?;
        Ok(surfaces)
    })
}
