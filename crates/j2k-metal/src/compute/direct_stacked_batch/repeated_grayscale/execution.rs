// SPDX-License-Identifier: MIT OR Apache-2.0

use std::mem::size_of;

use j2k_native::J2kDirectStoreStep;

use super::{
    Buffer, CommandBufferRef, DirectScratchBuffer, DirectStatusCheck, Error, MetalRuntime,
    PixelFormat, RepeatedDirectGrayscalePlanRequest, Surface,
};
use crate::compute::direct_stacked_batch::{
    lookup_direct_band_slice, lookup_direct_band_slice_entry,
    lookup_repeated_direct_band_layout_entry, DirectBandSlice,
};
use crate::compute::{
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
    take_f32_scratch_buffer, BandRequiredRegion, DirectIdwtCommandBuffers, IdwtSubBandBuffers,
    J2kRepeatedGrayStoreParams, J2kRepeatedStoreParams, J2kStoreParams, J2kWaveletTransform,
    PreparedClassicSubBand, PreparedClassicSubBandGroup, PreparedDirectGrayscaleStep,
    PreparedDirectIdwt, PreparedHtSubBand, PreparedHtSubBandGroup, PreparedIdwtInputStrides,
    RepeatedIdwtDispatch, SingleIdwtDispatch,
};

struct RepeatedGrayscaleExecution<'a> {
    runtime: &'a MetalRuntime,
    command_buffer: &'a CommandBufferRef,
    fmt: PixelFormat,
    dimensions: (u32, u32),
    bit_depth: u8,
    count: usize,
    retained_buffers: &'a mut Vec<Buffer>,
    status_checks: &'a mut Vec<DirectStatusCheck>,
    scratch_buffers: &'a mut Vec<DirectScratchBuffer>,
    band_sets: Vec<Vec<DirectBandSlice>>,
    surfaces: Vec<Surface>,
    stacked_outputs: bool,
}

impl RepeatedGrayscaleExecution<'_> {
    fn encode_classic_group(&mut self, group: &PreparedClassicSubBandGroup) -> Result<(), Error> {
        let per_instance_len = group.total_coefficients;
        let output = take_f32_scratch_buffer(self.runtime, per_instance_len * self.count)?;
        let (buffers, status_check) =
            encode_repeated_classic_sub_band_group_to_buffer_in_command_buffer(
                self.runtime,
                self.command_buffer,
                group,
                self.count,
                &output.buffer,
                self.scratch_buffers,
            )?;
        self.retained_buffers.extend(buffers);
        self.status_checks.push(status_check);
        let stride_bytes = per_instance_len * size_of::<f32>();
        for (instance_idx, bands) in self.band_sets.iter_mut().enumerate() {
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
        self.scratch_buffers.push(output);
        Ok(())
    }

    fn encode_ht_group(&mut self, group: &PreparedHtSubBandGroup) -> Result<(), Error> {
        let per_instance_len = group.total_coefficients;
        let output = take_f32_scratch_buffer(self.runtime, per_instance_len * self.count)?;
        let (buffers, status_check) =
            encode_repeated_ht_sub_band_group_to_buffer_in_command_buffer(
                self.runtime,
                self.command_buffer,
                group,
                self.count,
                &output.buffer,
            )?;
        self.retained_buffers.extend(buffers);
        self.status_checks.push(status_check);
        let stride_bytes = per_instance_len * size_of::<f32>();
        for (instance_idx, bands) in self.band_sets.iter_mut().enumerate() {
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
        self.scratch_buffers.push(output);
        Ok(())
    }

    fn retain_sub_band(
        &mut self,
        output: DirectScratchBuffer,
        band_id: j2k_native::J2kDirectBandId,
        width: u32,
        height: u32,
    ) {
        let per_instance_len = width as usize * height as usize;
        let stride_bytes = per_instance_len * size_of::<f32>();
        for (instance_idx, bands) in self.band_sets.iter_mut().enumerate() {
            bands.push(DirectBandSlice {
                band_id,
                buffer: output.buffer.clone(),
                offset_bytes: instance_idx * stride_bytes,
                window: BandRequiredRegion::full(width, height),
            });
        }
        self.scratch_buffers.push(output);
    }

    fn encode_classic_sub_band(&mut self, sub_band: &PreparedClassicSubBand) -> Result<(), Error> {
        let per_instance_len = sub_band.width as usize * sub_band.height as usize;
        let output = take_f32_scratch_buffer(self.runtime, per_instance_len * self.count)?;
        let (buffers, status_check) = encode_repeated_classic_sub_band_to_buffer_in_command_buffer(
            self.runtime,
            self.command_buffer,
            sub_band,
            self.count,
            &output.buffer,
            self.scratch_buffers,
        )?;
        self.retained_buffers.extend(buffers);
        self.status_checks.push(status_check);
        self.retain_sub_band(output, sub_band.band_id, sub_band.width, sub_band.height);
        Ok(())
    }

    fn encode_ht_sub_band(&mut self, sub_band: &PreparedHtSubBand) -> Result<(), Error> {
        let per_instance_len = sub_band.width as usize * sub_band.height as usize;
        let output = take_f32_scratch_buffer(self.runtime, per_instance_len * self.count)?;
        let (buffers, status_check) = encode_repeated_ht_sub_band_to_buffer_in_command_buffer(
            self.runtime,
            self.command_buffer,
            sub_band,
            self.count,
            &output.buffer,
        )?;
        self.retained_buffers.extend(buffers);
        self.status_checks.push(status_check);
        self.retain_sub_band(output, sub_band.band_id, sub_band.width, sub_band.height);
        Ok(())
    }

    fn encode_stacked_idwt(&mut self, idwt: &PreparedDirectIdwt) -> Result<(), Error> {
        let (ll, low_low_stride) = lookup_repeated_direct_band_layout_entry(
            &self.band_sets,
            idwt.step.ll_band_id,
            idwt.step.ll,
        )?;
        let (hl, high_low_stride) = lookup_repeated_direct_band_layout_entry(
            &self.band_sets,
            idwt.step.hl_band_id,
            idwt.step.hl,
        )?;
        let (lh, low_high_stride) = lookup_repeated_direct_band_layout_entry(
            &self.band_sets,
            idwt.step.lh_band_id,
            idwt.step.lh,
        )?;
        let (hh, high_high_stride) = lookup_repeated_direct_band_layout_entry(
            &self.band_sets,
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
            "repeated",
        )?;
        let per_instance_len = prepared_idwt_output_len(idwt);
        let output = take_f32_scratch_buffer(self.runtime, per_instance_len * self.count)?;
        dispatch_reversible53_repeated_buffers_in_command_buffer_with_offsets(
            DirectIdwtCommandBuffers::single(self.command_buffer),
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
        let stride_bytes = per_instance_len * size_of::<f32>();
        for (instance_idx, bands) in self.band_sets.iter_mut().enumerate() {
            bands.push(DirectBandSlice {
                band_id: idwt.step.output_band_id,
                buffer: output.buffer.clone(),
                offset_bytes: instance_idx * stride_bytes,
                window: idwt.output_window,
            });
        }
        self.scratch_buffers.push(output);
        Ok(())
    }

    fn encode_per_instance_idwt(&mut self, idwt: &PreparedDirectIdwt) -> Result<(), Error> {
        for bands in &mut self.band_sets {
            let ll = lookup_direct_band_slice_entry(bands, idwt.step.ll_band_id, idwt.step.ll)?;
            let hl = lookup_direct_band_slice_entry(bands, idwt.step.hl_band_id, idwt.step.hl)?;
            let lh = lookup_direct_band_slice_entry(bands, idwt.step.lh_band_id, idwt.step.lh)?;
            let hh = lookup_direct_band_slice_entry(bands, idwt.step.hh_band_id, idwt.step.hh)?;
            let params =
                prepared_idwt_params(idwt, idwt_input_windows_from_slices(&ll, &hl, &lh, &hh));
            let output = take_f32_scratch_buffer(self.runtime, prepared_idwt_output_len(idwt))?;
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
                    dispatch_reversible53_single_decomposition_buffers_in_command_buffer_with_offsets(
                        self.command_buffer,
                        dispatch,
                    );
                }
                J2kWaveletTransform::Irreversible97 => {
                    self.status_checks.push(
                        dispatch_irreversible97_single_decomposition_buffers_in_command_buffer_with_offsets(
                            self.command_buffer,
                            dispatch,
                        ),
                    );
                }
            }
            bands.push(DirectBandSlice {
                band_id: idwt.step.output_band_id,
                buffer: output.buffer.clone(),
                offset_bytes: 0,
                window: idwt.output_window,
            });
            self.scratch_buffers.push(output);
        }
        Ok(())
    }

    fn encode_idwt(&mut self, idwt: &PreparedDirectIdwt) -> Result<(), Error> {
        if idwt.step.transform == J2kWaveletTransform::Reversible53 && self.stacked_outputs {
            self.encode_stacked_idwt(idwt)
        } else {
            self.stacked_outputs = false;
            self.encode_per_instance_idwt(idwt)
        }
    }

    fn encode_stacked_store(&mut self, store: &J2kDirectStoreStep) -> Result<(), Error> {
        let (input, _) =
            lookup_direct_band_slice(&self.band_sets[0], store.input_band_id, store.input_rect)?;
        let batch_count = u32::try_from(self.count).map_err(|_| Error::MetalKernel {
            message: "J2K MetalDirect repeated store batch count exceeds u32".to_string(),
        })?;
        if matches!(self.fmt, PixelFormat::Gray8 | PixelFormat::Gray16) {
            let scale = j2k_scalar_pack_params(u32::from(self.bit_depth));
            self.surfaces
                .extend(encode_repeated_gray_store_to_surfaces_in_command_buffer(
                    self.runtime,
                    self.command_buffer,
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
                    self.dimensions,
                    self.fmt,
                    self.count,
                )?);
        } else {
            let per_instance_len = store.output_width as usize * store.output_height as usize;
            let output = take_f32_scratch_buffer(self.runtime, per_instance_len * self.count)?;
            dispatch_store_component_repeated_in_command_buffer(
                self.runtime,
                self.command_buffer,
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
            self.retained_buffers.push(output.buffer.clone());
            self.surfaces
                .extend(encode_repeated_gray_plane_to_surfaces_in_command_buffer(
                    self.runtime,
                    self.command_buffer,
                    &output.buffer,
                    self.dimensions,
                    self.bit_depth,
                    self.fmt,
                    self.count,
                )?);
            self.scratch_buffers.push(output);
        }
        Ok(())
    }

    fn encode_per_instance_store(&mut self, store: &J2kDirectStoreStep) -> Result<(), Error> {
        for bands in &self.band_sets {
            let (input, input_offset) =
                lookup_direct_band_slice(bands, store.input_band_id, store.input_rect)?;
            let output = take_f32_scratch_buffer(
                self.runtime,
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
                self.runtime,
                self.command_buffer,
                &input,
                input_offset,
                &output.buffer,
                0,
                params,
            );
            self.retained_buffers.push(output.buffer.clone());
            self.surfaces
                .push(encode_gray_plane_to_surface_in_command_buffer_with_offset(
                    self.runtime,
                    self.command_buffer,
                    &output.buffer,
                    0,
                    self.dimensions,
                    self.bit_depth,
                    self.fmt,
                )?);
            self.scratch_buffers.push(output);
        }
        Ok(())
    }

    fn encode_store(&mut self, store: &J2kDirectStoreStep) -> Result<(), Error> {
        if self.stacked_outputs {
            self.encode_stacked_store(store)
        } else {
            self.encode_per_instance_store(store)
        }
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

    fn finish(self) -> Result<Vec<Surface>, Error> {
        if self.surfaces.len() != self.count {
            return Err(Error::MetalKernel {
                message: format!(
                    "J2K MetalDirect repeated grayscale plan produced {} surfaces for count {}",
                    self.surfaces.len(),
                    self.count
                ),
            });
        }
        Ok(self.surfaces)
    }
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn encode_repeated_direct_grayscale_plan_in_command_buffer(
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
    let mut execution = RepeatedGrayscaleExecution {
        runtime,
        command_buffer,
        fmt,
        dimensions: plan.dimensions,
        bit_depth: plan.bit_depth,
        count,
        retained_buffers,
        status_checks,
        scratch_buffers,
        band_sets: vec![Vec::new(); count],
        surfaces: Vec::with_capacity(count),
        stacked_outputs: true,
    };
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
}
