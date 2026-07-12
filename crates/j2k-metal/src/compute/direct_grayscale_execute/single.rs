// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_native::J2kDirectStoreStep;
use metal::ComputeCommandEncoderRef;

use super::{
    dispatch_irreversible97_single_decomposition_buffers_in_encoder_with_offsets,
    dispatch_reversible53_single_decomposition_buffers_in_encoder_with_offsets,
    dispatch_store_component_buffer_in_encoder_with_offsets,
    encode_gray_plane_to_surface_in_encoder, encode_gray_store_to_surface_in_encoder,
    encode_prepared_classic_sub_band_group_to_buffer_in_encoder,
    encode_prepared_classic_sub_band_to_buffer_in_encoder,
    encode_prepared_ht_sub_band_group_to_buffer_in_encoder,
    encode_prepared_ht_sub_band_to_buffer_in_encoder, idwt_input_windows_from_slices,
    j2k_scalar_pack_params, lookup_direct_band_slice, lookup_direct_band_slice_entry,
    new_compute_command_encoder, prepared_idwt_output_len, prepared_idwt_params, size_of,
    take_f32_scratch_buffer, BandRequiredRegion, Buffer, CommandBufferRef, DirectBandSlice,
    DirectScratchBuffer, DirectStatusCheck, Error, IdwtSubBandBuffers, J2kGrayStoreParams,
    J2kStoreParams, J2kWaveletTransform, MetalRuntime, PixelFormat, PreparedDirectGrayscalePlan,
    PreparedDirectGrayscaleStep, SingleIdwtDispatch, Surface,
};
use crate::compute::{
    PreparedClassicSubBand, PreparedClassicSubBandGroup, PreparedDirectIdwt, PreparedHtSubBand,
    PreparedHtSubBandGroup,
};

struct SingleGrayscaleExecution<'a> {
    runtime: &'a MetalRuntime,
    encoder: &'a ComputeCommandEncoderRef,
    fmt: PixelFormat,
    dimensions: (u32, u32),
    bit_depth: u8,
    retained_buffers: &'a mut Vec<Buffer>,
    status_checks: &'a mut Vec<DirectStatusCheck>,
    scratch_buffers: &'a mut Vec<DirectScratchBuffer>,
    bands: Vec<DirectBandSlice>,
    final_surface: Option<Surface>,
}

impl SingleGrayscaleExecution<'_> {
    fn encode_classic_group(&mut self, group: &PreparedClassicSubBandGroup) -> Result<(), Error> {
        let output = take_f32_scratch_buffer(self.runtime, group.total_coefficients)?;
        let (buffers, status_check) = encode_prepared_classic_sub_band_group_to_buffer_in_encoder(
            self.runtime,
            self.encoder,
            group,
            &output.buffer,
            self.scratch_buffers,
        )?;
        self.retained_buffers.extend(buffers);
        self.status_checks.push(status_check);
        for member in &group.members {
            self.bands.push(DirectBandSlice {
                band_id: member.band_id,
                buffer: output.buffer.clone(),
                offset_bytes: member.offset_elements * size_of::<f32>(),
                window: member.window,
            });
        }
        self.scratch_buffers.push(output);
        Ok(())
    }

    fn encode_ht_group(&mut self, group: &PreparedHtSubBandGroup) -> Result<(), Error> {
        let output = take_f32_scratch_buffer(self.runtime, group.total_coefficients)?;
        let (buffers, status_check) = encode_prepared_ht_sub_band_group_to_buffer_in_encoder(
            self.runtime,
            self.encoder,
            group,
            &output.buffer,
        )?;
        self.retained_buffers.extend(buffers);
        self.status_checks.push(status_check);
        for member in &group.members {
            self.bands.push(DirectBandSlice {
                band_id: member.band_id,
                buffer: output.buffer.clone(),
                offset_bytes: member.offset_elements * size_of::<f32>(),
                window: member.window,
            });
        }
        self.scratch_buffers.push(output);
        Ok(())
    }

    fn encode_classic_sub_band(&mut self, sub_band: &PreparedClassicSubBand) -> Result<(), Error> {
        let output = take_f32_scratch_buffer(
            self.runtime,
            sub_band.width as usize * sub_band.height as usize,
        )?;
        let (buffers, status_check) = encode_prepared_classic_sub_band_to_buffer_in_encoder(
            self.runtime,
            self.encoder,
            sub_band,
            &output.buffer,
            self.scratch_buffers,
        )?;
        self.retained_buffers.extend(buffers);
        self.status_checks.push(status_check);
        self.bands.push(DirectBandSlice {
            band_id: sub_band.band_id,
            buffer: output.buffer.clone(),
            offset_bytes: 0,
            window: BandRequiredRegion::full(sub_band.width, sub_band.height),
        });
        self.scratch_buffers.push(output);
        Ok(())
    }

    fn encode_ht_sub_band(&mut self, sub_band: &PreparedHtSubBand) -> Result<(), Error> {
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
        self.bands.push(DirectBandSlice {
            band_id: sub_band.band_id,
            buffer: output.buffer.clone(),
            offset_bytes: 0,
            window: BandRequiredRegion::full(sub_band.width, sub_band.height),
        });
        self.scratch_buffers.push(output);
        Ok(())
    }

    fn encode_idwt(&mut self, idwt: &PreparedDirectIdwt) -> Result<(), Error> {
        let ll = lookup_direct_band_slice_entry(&self.bands, idwt.step.ll_band_id, idwt.step.ll)?;
        let hl = lookup_direct_band_slice_entry(&self.bands, idwt.step.hl_band_id, idwt.step.hl)?;
        let lh = lookup_direct_band_slice_entry(&self.bands, idwt.step.lh_band_id, idwt.step.lh)?;
        let hh = lookup_direct_band_slice_entry(&self.bands, idwt.step.hh_band_id, idwt.step.hh)?;
        let params = prepared_idwt_params(idwt, idwt_input_windows_from_slices(&ll, &hl, &lh, &hh));
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
                    )?,
                );
            }
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
        if matches!(self.fmt, PixelFormat::Gray8 | PixelFormat::Gray16) {
            let scale = j2k_scalar_pack_params(u32::from(self.bit_depth));
            self.final_surface = Some(encode_gray_store_to_surface_in_encoder(
                self.runtime,
                self.encoder,
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
                self.dimensions,
                self.fmt,
            )?);
        } else {
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
            dispatch_store_component_buffer_in_encoder_with_offsets(
                self.runtime,
                self.encoder,
                &input,
                input_offset,
                &output.buffer,
                0,
                params,
            );
            self.retained_buffers.push(output.buffer.clone());
            self.final_surface = Some(encode_gray_plane_to_surface_in_encoder(
                self.runtime,
                self.encoder,
                &output.buffer,
                self.dimensions,
                self.bit_depth,
                self.fmt,
            )?);
            self.scratch_buffers.push(output);
        }
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

    fn finish(self) -> Result<Surface, Error> {
        self.final_surface.ok_or_else(|| Error::MetalKernel {
            message: "J2K MetalDirect prepared grayscale plan did not produce a final stored plane"
                .to_string(),
        })
    }
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn encode_prepared_direct_grayscale_plan_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    plan: &PreparedDirectGrayscalePlan,
    fmt: PixelFormat,
    retained_buffers: &mut Vec<Buffer>,
    status_checks: &mut Vec<DirectStatusCheck>,
    scratch_buffers: &mut Vec<DirectScratchBuffer>,
) -> Result<Surface, Error> {
    let mut budget = crate::batch_allocation::BatchMetadataBudget::new(
        "J2K MetalDirect grayscale band metadata",
    );
    let bands = budget.try_vec(plan.steps.len(), "J2K MetalDirect grayscale band metadata")?;
    let encoder = new_compute_command_encoder(command_buffer)?;
    let mut execution = SingleGrayscaleExecution {
        runtime,
        encoder: &encoder,
        fmt,
        dimensions: plan.dimensions,
        bit_depth: plan.bit_depth,
        retained_buffers,
        status_checks,
        scratch_buffers,
        bands,
        final_surface: None,
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
