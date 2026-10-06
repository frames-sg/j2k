// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::metal_types::prelude::*;
use crate::metal_types::Buffer;

use std::time::Instant;

use crate::engine::abi::J2kRepeatedStoreParams;
use crate::engine::decode_dispatch::{
    dispatch_store_component_repeated_in_command_buffer,
    dispatch_store_component_repeated_in_encoder,
};
use crate::engine::{
    direct_preflight_invariant, elapsed_us, take_f32_scratch_buffer, Error,
    PreparedDirectGrayscaleStep,
};

use super::super::resources::{lookup_mapped_repeated_direct_band_layout_entry, StackedFinalPlane};
use super::super::validation::checked_f32_dimension_span;
use super::SubmissionContext;

impl SubmissionContext<'_, '_, '_> {
    pub(super) fn submit_store(
        &mut self,
        step_idx: usize,
        store: &j2k_native::J2kDirectStoreStep,
    ) -> Result<(), Error> {
        let (input, input_instance_stride) = lookup_mapped_repeated_direct_band_layout_entry(
            &self.resources.band_sets,
            self.plans
                .iter()
                .map(|plan| match plan.steps.get(step_idx) {
                    Some(PreparedDirectGrayscaleStep::Store(step)) => {
                        Ok((step.input_band_id, step.input_rect))
                    }
                    _ => Err(direct_preflight_invariant(
                        "Store step mismatch in mapped stacked component batch",
                    )),
                }),
        )?;
        let dimensions = (store.output_width, store.output_height);
        let span = checked_f32_dimension_span(
            store.output_width,
            store.output_height,
            self.count,
            "J2K MetalDirect stacked store",
        )?;
        // A complete identity store can keep the already retained coefficient
        // plane. Tile assembly, cropping, and centered rounding still use Store.
        if self.compute_encoder.is_some()
            && self.resources.final_plane.is_none()
            && step_idx + 1 == self.plans[0].steps.len()
            && store.addend == 0.0
            && !self.round_centered_store
            && store.source_x == 0
            && store.source_y == 0
            && store.output_x == 0
            && store.output_y == 0
            && store.copy_width == store.output_width
            && store.copy_height == store.output_height
            && store.input_rect.width() == store.output_width
            && store.input_rect.height() == store.output_height
            && input.window.width() == store.output_width
            && input.window.height() == store.output_height
            && input.offset_bytes == 0
            && input_instance_stride as usize == span.per_instance_elements
            && input.buffer.length() >= span.total_bytes
        {
            self.resources.final_plane = Some(StackedFinalPlane {
                buffer: input.buffer,
                dimensions,
                len: span.total_elements,
            });
            for bands in &mut self.resources.band_sets {
                bands.clear();
            }
            return Ok(());
        }
        let output = self.final_store_output(dimensions, span.total_elements, span.total_bytes)?;
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
            round_centered: u32::from(self.round_centered_store),
        };
        if let Some(encoder) = self.compute_encoder {
            dispatch_store_component_repeated_in_encoder(
                self.runtime.decode()?,
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

    /// Returns the final component plane a store writes: the one an earlier
    /// tile store created, or a new scratch plane.
    fn final_store_output(
        &mut self,
        dimensions: (u32, u32),
        total_elements: usize,
        total_bytes: usize,
    ) -> Result<Buffer, Error> {
        let required_bytes = u64::try_from(total_bytes).map_err(|_| Error::MetalKernel {
            message: "J2K MetalDirect stacked store byte length exceeds u64".to_string(),
        })?;
        if let Some(output) = self.resources.final_plane.as_ref() {
            if output.dimensions != dimensions || output.len != total_elements {
                return Err(Error::MetalStateInvariant {
                    state: "J2K MetalDirect stacked component tile store",
                    reason: "later tile store changed the final component plane shape",
                });
            }
            if u64::try_from(output.buffer.length()).map_or(true, |len| len < required_bytes) {
                return Err(Error::MetalStateInvariant {
                    state: "J2K MetalDirect stacked component tile store",
                    reason: "retained final component plane is smaller than the validated store",
                });
            }
            return Ok(output.buffer.clone());
        }
        let output = take_f32_scratch_buffer(self.runtime, total_elements)?;
        let buffer = output.buffer.clone();
        self.resources.final_plane = Some(StackedFinalPlane {
            buffer: buffer.clone(),
            dimensions,
            len: total_elements,
        });
        self.scratch_buffers.push(output);
        Ok(buffer)
    }
}
