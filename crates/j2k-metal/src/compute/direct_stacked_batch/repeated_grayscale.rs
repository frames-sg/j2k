// SPDX-License-Identifier: MIT OR Apache-2.0

use std::mem::size_of;

use metal::{Buffer, CommandBufferRef};

use super::super::{
    dispatch_irreversible97_single_decomposition_buffers_in_command_buffer_with_offsets,
    dispatch_reversible53_repeated_buffers_in_command_buffer_with_offsets,
    dispatch_reversible53_single_decomposition_buffers_in_command_buffer_with_offsets,
    dispatch_store_component_buffer_in_command_buffer_with_offsets,
    dispatch_store_component_repeated_in_command_buffer,
    encode_gray_plane_to_surface_in_command_buffer_with_offset,
    encode_repeated_classic_sub_band_group_to_buffer_in_command_buffer,
    encode_repeated_classic_sub_band_to_buffer_in_command_buffer,
    encode_repeated_gray_plane_to_surfaces_in_command_buffer,
    encode_repeated_gray_store_to_surfaces_in_command_buffer,
    encode_repeated_ht_sub_band_group_to_buffer_in_command_buffer,
    encode_repeated_ht_sub_band_to_buffer_in_command_buffer, idwt_input_windows_from_slices,
    j2k_scalar_pack_params, prepared_idwt_output_len, prepared_idwt_params, repeated_idwt_params,
    take_f32_scratch_buffer, BandRequiredRegion, DirectIdwtCommandBuffers, DirectScratchBuffer,
    DirectStatusCheck, Error, IdwtSubBandBuffers, J2kRepeatedGrayStoreParams,
    J2kRepeatedStoreParams, J2kStoreParams, J2kWaveletTransform, MetalRuntime, PixelFormat,
    PreparedDirectGrayscalePlan, PreparedDirectGrayscaleStep, PreparedIdwtInputStrides,
    RepeatedIdwtDispatch, SingleIdwtDispatch, Surface,
};
use super::{
    lookup_direct_band_slice, lookup_direct_band_slice_entry,
    lookup_repeated_direct_band_layout_entry, DirectBandSlice,
};

pub(in super::super) struct RepeatedDirectGrayscalePlanRequest<'a> {
    pub(in super::super) runtime: &'a MetalRuntime,
    pub(in super::super) command_buffer: &'a CommandBufferRef,
    pub(in super::super) plan: &'a PreparedDirectGrayscalePlan,
    pub(in super::super) fmt: PixelFormat,
    pub(in super::super) count: usize,
    pub(in super::super) retained_buffers: &'a mut Vec<Buffer>,
    pub(in super::super) status_checks: &'a mut Vec<DirectStatusCheck>,
    pub(in super::super) scratch_buffers: &'a mut Vec<DirectScratchBuffer>,
}

#[cfg(target_os = "macos")]
pub(in super::super) fn encode_repeated_direct_grayscale_plan_in_command_buffer(
    request: RepeatedDirectGrayscalePlanRequest<'_>,
) -> Result<Vec<Surface>, Error> {
    let RepeatedDirectGrayscalePlanRequest {
        runtime,
        command_buffer,
        plan,
        fmt,
        count,
        retained_buffers,
        status_checks,
        scratch_buffers,
    } = request;
    let mut band_sets = vec![Vec::<DirectBandSlice>::new(); count];
    let mut surfaces = Vec::with_capacity(count);
    let mut stacked_outputs = true;
    let mut step_idx = 0;

    while step_idx < plan.steps.len() {
        if let Some(group) = plan.classic_group_starting_at(step_idx) {
            let per_instance_len = group.total_coefficients;
            let output = take_f32_scratch_buffer(runtime, per_instance_len * count)?;
            let (buffers, status_check) =
                encode_repeated_classic_sub_band_group_to_buffer_in_command_buffer(
                    runtime,
                    command_buffer,
                    group,
                    count,
                    &output.buffer,
                    scratch_buffers,
                )?;
            retained_buffers.extend(buffers);
            status_checks.push(status_check);
            let stride_bytes = per_instance_len * size_of::<f32>();
            for (instance_idx, bands) in band_sets.iter_mut().enumerate() {
                for member in &group.members {
                    bands.push(DirectBandSlice {
                        band_id: member.band_id,
                        buffer: output.buffer.clone(),
                        offset_bytes: instance_idx * stride_bytes
                            + member.offset_elements * size_of::<f32>(),
                        window: member.window,
                    });
                }
            }
            scratch_buffers.push(output);
            step_idx = group.end_step;
            continue;
        }

        if let Some(group) = plan.ht_group_starting_at(step_idx) {
            let per_instance_len = group.total_coefficients;
            let output = take_f32_scratch_buffer(runtime, per_instance_len * count)?;
            let (buffers, status_check) =
                encode_repeated_ht_sub_band_group_to_buffer_in_command_buffer(
                    runtime,
                    command_buffer,
                    group,
                    count,
                    &output.buffer,
                )?;
            retained_buffers.extend(buffers);
            status_checks.push(status_check);
            let stride_bytes = per_instance_len * size_of::<f32>();
            for (instance_idx, bands) in band_sets.iter_mut().enumerate() {
                for member in &group.members {
                    bands.push(DirectBandSlice {
                        band_id: member.band_id,
                        buffer: output.buffer.clone(),
                        offset_bytes: instance_idx * stride_bytes
                            + member.offset_elements * size_of::<f32>(),
                        window: member.window,
                    });
                }
            }
            scratch_buffers.push(output);
            step_idx = group.end_step;
            continue;
        }

        let step = &plan.steps[step_idx];
        match step {
            PreparedDirectGrayscaleStep::ClassicSubBand(sub_band) => {
                let per_instance_len = sub_band.width as usize * sub_band.height as usize;
                let output = take_f32_scratch_buffer(runtime, per_instance_len * count)?;
                let (buffers, status_check) =
                    encode_repeated_classic_sub_band_to_buffer_in_command_buffer(
                        runtime,
                        command_buffer,
                        sub_band,
                        count,
                        &output.buffer,
                        scratch_buffers,
                    )?;
                retained_buffers.extend(buffers);
                status_checks.push(status_check);
                let stride_bytes = per_instance_len * size_of::<f32>();
                for (instance_idx, bands) in band_sets.iter_mut().enumerate() {
                    bands.push(DirectBandSlice {
                        band_id: sub_band.band_id,
                        buffer: output.buffer.clone(),
                        offset_bytes: instance_idx * stride_bytes,
                        window: BandRequiredRegion::full(sub_band.width, sub_band.height),
                    });
                }
                scratch_buffers.push(output);
            }
            PreparedDirectGrayscaleStep::HtSubBand(sub_band) => {
                let per_instance_len = sub_band.width as usize * sub_band.height as usize;
                let output = take_f32_scratch_buffer(runtime, per_instance_len * count)?;
                let (buffers, status_check) =
                    encode_repeated_ht_sub_band_to_buffer_in_command_buffer(
                        runtime,
                        command_buffer,
                        sub_band,
                        count,
                        &output.buffer,
                    )?;
                retained_buffers.extend(buffers);
                status_checks.push(status_check);
                let stride_bytes = per_instance_len * size_of::<f32>();
                for (instance_idx, bands) in band_sets.iter_mut().enumerate() {
                    bands.push(DirectBandSlice {
                        band_id: sub_band.band_id,
                        buffer: output.buffer.clone(),
                        offset_bytes: instance_idx * stride_bytes,
                        window: BandRequiredRegion::full(sub_band.width, sub_band.height),
                    });
                }
                scratch_buffers.push(output);
            }
            PreparedDirectGrayscaleStep::Idwt(idwt) => match idwt.step.transform {
                J2kWaveletTransform::Reversible53 if stacked_outputs => {
                    let (ll, low_low_stride) = lookup_repeated_direct_band_layout_entry(
                        &band_sets,
                        idwt.step.ll_band_id,
                        idwt.step.ll,
                    )?;
                    let (hl, high_low_stride) = lookup_repeated_direct_band_layout_entry(
                        &band_sets,
                        idwt.step.hl_band_id,
                        idwt.step.hl,
                    )?;
                    let (lh, low_high_stride) = lookup_repeated_direct_band_layout_entry(
                        &band_sets,
                        idwt.step.lh_band_id,
                        idwt.step.lh,
                    )?;
                    let (hh, high_high_stride) = lookup_repeated_direct_band_layout_entry(
                        &band_sets,
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
                        count,
                        "repeated",
                    )?;
                    let per_instance_len = prepared_idwt_output_len(idwt);
                    let output = take_f32_scratch_buffer(runtime, per_instance_len * count)?;
                    dispatch_reversible53_repeated_buffers_in_command_buffer_with_offsets(
                        DirectIdwtCommandBuffers::single(command_buffer),
                        RepeatedIdwtDispatch {
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
                        },
                    );
                    let stride_bytes = per_instance_len * size_of::<f32>();
                    for (instance_idx, bands) in band_sets.iter_mut().enumerate() {
                        bands.push(DirectBandSlice {
                            band_id: idwt.step.output_band_id,
                            buffer: output.buffer.clone(),
                            offset_bytes: instance_idx * stride_bytes,
                            window: idwt.output_window,
                        });
                    }
                    scratch_buffers.push(output);
                }
                _ => {
                    stacked_outputs = false;
                    for bands in &mut band_sets {
                        let ll = lookup_direct_band_slice_entry(
                            bands,
                            idwt.step.ll_band_id,
                            idwt.step.ll,
                        )?;
                        let hl = lookup_direct_band_slice_entry(
                            bands,
                            idwt.step.hl_band_id,
                            idwt.step.hl,
                        )?;
                        let lh = lookup_direct_band_slice_entry(
                            bands,
                            idwt.step.lh_band_id,
                            idwt.step.lh,
                        )?;
                        let hh = lookup_direct_band_slice_entry(
                            bands,
                            idwt.step.hh_band_id,
                            idwt.step.hh,
                        )?;
                        let params = prepared_idwt_params(
                            idwt,
                            idwt_input_windows_from_slices(&ll, &hl, &lh, &hh),
                        );
                        let output =
                            take_f32_scratch_buffer(runtime, prepared_idwt_output_len(idwt))?;
                        match idwt.step.transform {
                                J2kWaveletTransform::Reversible53 => {
                                    dispatch_reversible53_single_decomposition_buffers_in_command_buffer_with_offsets(
                                        command_buffer,
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
                                J2kWaveletTransform::Irreversible97 => status_checks.push(
                                    dispatch_irreversible97_single_decomposition_buffers_in_command_buffer_with_offsets(
                                        command_buffer,
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
                                ),
                            }
                        bands.push(DirectBandSlice {
                            band_id: idwt.step.output_band_id,
                            buffer: output.buffer.clone(),
                            offset_bytes: 0,
                            window: idwt.output_window,
                        });
                        scratch_buffers.push(output);
                    }
                }
            },
            PreparedDirectGrayscaleStep::Store(store) => {
                if stacked_outputs {
                    let (input, _) = lookup_direct_band_slice(
                        &band_sets[0],
                        store.input_band_id,
                        store.input_rect,
                    )?;
                    let batch_count = u32::try_from(count).map_err(|_| Error::MetalKernel {
                        message: "J2K MetalDirect repeated store batch count exceeds u32"
                            .to_string(),
                    })?;
                    if matches!(fmt, PixelFormat::Gray8 | PixelFormat::Gray16) {
                        let scale = j2k_scalar_pack_params(u32::from(plan.bit_depth));
                        surfaces.extend(encode_repeated_gray_store_to_surfaces_in_command_buffer(
                            runtime,
                            command_buffer,
                            &input,
                            J2kRepeatedGrayStoreParams {
                                input_width: store.input_rect.width(),
                                input_height: store.input_rect.height(),
                                source_x: store.source_x,
                                source_y: store.source_y,
                                copy_width: store.copy_width,
                                copy_height: store.copy_height,
                                output_width: store.output_width,
                                output_height: store.output_height,
                                output_x: store.output_x,
                                output_y: store.output_y,
                                addend: store.addend,
                                batch_count,
                                max_value: scale.max_value,
                                u8_scale: scale.u8_scale,
                                u16_scale: scale.u16_scale,
                            },
                            plan.dimensions,
                            fmt,
                            count,
                        )?);
                    } else {
                        let per_instance_len =
                            store.output_width as usize * store.output_height as usize;
                        let output = take_f32_scratch_buffer(runtime, per_instance_len * count)?;
                        dispatch_store_component_repeated_in_command_buffer(
                            runtime,
                            command_buffer,
                            &input,
                            0,
                            &output.buffer,
                            J2kRepeatedStoreParams {
                                input_width: store.input_rect.width(),
                                input_height: store.input_rect.height(),
                                input_instance_stride: store
                                    .input_rect
                                    .width()
                                    .checked_mul(store.input_rect.height())
                                    .ok_or_else(|| Error::MetalKernel {
                                        message: "J2K MetalDirect repeated store input stride overflows u32"
                                            .to_string(),
                                    })?,
                                source_x: store.source_x,
                                source_y: store.source_y,
                                copy_width: store.copy_width,
                                copy_height: store.copy_height,
                                output_width: store.output_width,
                                output_height: store.output_height,
                                output_x: store.output_x,
                                output_y: store.output_y,
                                addend: store.addend,
                                batch_count,
                            },
                        );
                        retained_buffers.push(output.buffer.clone());
                        surfaces.extend(encode_repeated_gray_plane_to_surfaces_in_command_buffer(
                            runtime,
                            command_buffer,
                            &output.buffer,
                            plan.dimensions,
                            plan.bit_depth,
                            fmt,
                            count,
                        )?);
                        scratch_buffers.push(output);
                    }
                } else {
                    for bands in &band_sets {
                        let (input, input_offset) =
                            lookup_direct_band_slice(bands, store.input_band_id, store.input_rect)?;
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
                        dispatch_store_component_buffer_in_command_buffer_with_offsets(
                            runtime,
                            command_buffer,
                            &input,
                            input_offset,
                            &output.buffer,
                            0,
                            params,
                        );
                        retained_buffers.push(output.buffer.clone());
                        surfaces.push(encode_gray_plane_to_surface_in_command_buffer_with_offset(
                            runtime,
                            command_buffer,
                            &output.buffer,
                            0,
                            plan.dimensions,
                            plan.bit_depth,
                            fmt,
                        )?);
                        scratch_buffers.push(output);
                    }
                }
            }
        }
        step_idx += 1;
    }

    if surfaces.len() != count {
        return Err(Error::MetalKernel {
            message: format!(
                "J2K MetalDirect repeated grayscale plan produced {} surfaces for count {}",
                surfaces.len(),
                count
            ),
        });
    }

    Ok(surfaces)
}
