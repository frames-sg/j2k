// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_native::J2kDirectStoreStep;

use crate::compute::abi::{J2kRepeatedGrayStoreParams, J2kRepeatedStoreParams, J2kStoreParams};
use crate::compute::decode_dispatch::{
    dispatch_store_component_buffer_in_command_buffer_with_offsets,
    dispatch_store_component_repeated_in_command_buffer,
    encode_repeated_gray_store_to_surfaces_in_command_buffer,
};
use crate::compute::direct_stacked_batch::{
    lookup_direct_band_slice, validation::checked_f32_dimension_span,
};
use crate::compute::{
    encode_gray_plane_to_surface_in_command_buffer_with_offset,
    encode_repeated_gray_plane_to_surfaces_in_command_buffer, j2k_scalar_pack_params,
    take_f32_scratch_buffer, Error, PixelFormat,
};

use super::RepeatedGrayscaleExecution;

impl RepeatedGrayscaleExecution<'_> {
    fn encode_stacked_store(&mut self, store: &J2kDirectStoreStep) -> Result<(), Error> {
        let first_bands = self.band_sets.first().ok_or_else(|| Error::MetalKernel {
            message: "J2K MetalDirect repeated store has no band sets".to_string(),
        })?;
        let (input, _) =
            lookup_direct_band_slice(first_bands, store.input_band_id, store.input_rect)?;
        let span = checked_f32_dimension_span(
            store.output_width,
            store.output_height,
            self.count,
            "J2K MetalDirect repeated stacked store",
        )?;
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
            let output = take_f32_scratch_buffer(self.runtime, span.total_elements)?;
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
            )?;
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
        let span = checked_f32_dimension_span(
            store.output_width,
            store.output_height,
            1,
            "J2K MetalDirect repeated per-instance store",
        )?;
        for bands in &self.band_sets {
            let (input, input_offset) =
                lookup_direct_band_slice(bands, store.input_band_id, store.input_rect)?;
            let output = take_f32_scratch_buffer(self.runtime, span.total_elements)?;
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
            )?;
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

    pub(super) fn encode_store(&mut self, store: &J2kDirectStoreStep) -> Result<(), Error> {
        if self.stacked_outputs {
            self.encode_stacked_store(store)
        } else {
            self.encode_per_instance_store(store)
        }
    }
}
