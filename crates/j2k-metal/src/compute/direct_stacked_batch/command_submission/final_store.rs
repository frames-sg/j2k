// SPDX-License-Identifier: MIT OR Apache-2.0

use std::time::Instant;

use crate::compute::abi::J2kRepeatedStoreParams;
use crate::compute::decode_dispatch::{
    dispatch_store_component_repeated_in_command_buffer,
    dispatch_store_component_repeated_in_encoder,
};
use crate::compute::{elapsed_us, take_f32_scratch_buffer, Error};

use super::super::resources::{lookup_repeated_direct_band_layout_entry, StackedFinalPlane};
use super::super::validation::checked_f32_dimension_span;
use super::SubmissionContext;

impl SubmissionContext<'_, '_, '_> {
    pub(super) fn submit_store(
        &mut self,
        store: &j2k_native::J2kDirectStoreStep,
    ) -> Result<(), Error> {
        let (input, input_instance_stride) = lookup_repeated_direct_band_layout_entry(
            &self.resources.band_sets,
            store.input_band_id,
            store.input_rect,
        )?;
        let dimensions = (store.output_width, store.output_height);
        let span = checked_f32_dimension_span(
            store.output_width,
            store.output_height,
            self.count,
            "J2K MetalDirect stacked store",
        )?;
        let required_bytes = u64::try_from(span.total_bytes).map_err(|_| Error::MetalKernel {
            message: "J2K MetalDirect stacked store byte length exceeds u64".to_string(),
        })?;
        let output = if let Some(output) = self.resources.final_plane.as_ref() {
            if output.dimensions != dimensions || output.len != span.total_elements {
                return Err(Error::MetalStateInvariant {
                    state: "J2K MetalDirect stacked component tile store",
                    reason: "later tile store changed the final component plane shape",
                });
            }
            if output.buffer.length() < required_bytes {
                return Err(Error::MetalStateInvariant {
                    state: "J2K MetalDirect stacked component tile store",
                    reason: "retained final component plane is smaller than the validated store",
                });
            }
            output.buffer.clone()
        } else {
            let output = take_f32_scratch_buffer(self.runtime, span.total_elements)?;
            let buffer = output.buffer.clone();
            self.resources.final_plane = Some(StackedFinalPlane {
                buffer: buffer.clone(),
                dimensions,
                len: span.total_elements,
            });
            self.scratch_buffers.push(output);
            buffer
        };
        let encode_started = self.profile_stages.then(Instant::now);
        let params = J2kRepeatedStoreParams {
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
        };
        if let Some(encoder) = self.compute_encoder {
            dispatch_store_component_repeated_in_encoder(
                self.runtime,
                encoder,
                &input.buffer,
                input.offset_bytes,
                &output,
                params,
            );
            encoder.memory_barrier_with_resources(&[&output]);
        } else {
            dispatch_store_component_repeated_in_command_buffer(
                self.runtime,
                self.command_buffers.store,
                &input.buffer,
                input.offset_bytes,
                &output,
                params,
            )?;
        }
        if let Some(started) = encode_started {
            self.stage_timings.metal_store_encode += elapsed_us(started);
        }
        for bands in &mut self.resources.band_sets {
            bands.clear();
        }
        Ok(())
    }
}
