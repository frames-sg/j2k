// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(target_os = "macos")]
use crate::metal_types::prelude::*;

use std::time::Instant;

use crate::engine::abi::{J2kRepeatedIdwtSingleDecompositionParams, J2kStackedGrayStoreParams};
use crate::engine::decode_dispatch::{
    dispatch_fused_idwt_stacked_gray_store,
    dispatch_irreversible97_repeated_buffers_in_command_buffer_with_offsets,
    dispatch_irreversible97_repeated_buffers_in_encoder_with_offsets,
    dispatch_irreversible97_single_decomposition_buffers_in_command_buffer_with_offsets,
    dispatch_irreversible97_single_decomposition_buffers_in_encoder_with_offsets,
    dispatch_reversible53_repeated_buffers_in_command_buffer_with_offsets,
    dispatch_reversible53_repeated_buffers_in_encoder_with_offsets,
    fused_idwt_stacked_gray_store_pipeline, supports_fused_2d_idwt, FusedStackedGrayStore,
    IdwtSubBandBuffers, RepeatedIdwtDispatch, SingleIdwtDispatch,
};
use crate::engine::direct_roi::{
    idwt_input_windows_from_slices, prepared_idwt_params, repeated_idwt_params,
    PreparedIdwtInputStrides,
};
use crate::engine::{
    direct_preflight_invariant, elapsed_us, take_f32_scratch_buffer, Error, J2kWaveletTransform,
    PreparedDirectGrayscalePlan, PreparedDirectGrayscaleStep, PreparedDirectIdwt,
};

use super::super::resources::{
    lookup_direct_band_slice_entry, lookup_mapped_repeated_direct_band_layout_entry,
    DirectBandSlice,
};
use super::super::validation::{
    checked_f32_dimension_span, checked_f32_instance_offset, CheckedF32BatchSpan,
};
use super::SubmissionContext;

impl SubmissionContext<'_, '_, '_> {
    /// The four sub-band layouts every image's `idwt` at `step_idx` reads, and
    /// the batched kernel parameters.
    fn stacked_idwt_inputs(
        &self,
        step_idx: usize,
        idwt: &PreparedDirectIdwt,
    ) -> Result<
        (
            [DirectBandSlice; 4],
            J2kRepeatedIdwtSingleDecompositionParams,
        ),
        Error,
    > {
        let (ll, low_low_stride) = lookup_mapped_repeated_direct_band_layout_entry(
            &self.resources.band_sets,
            idwt_band_keys(self.plans, step_idx, |step| {
                (step.step.ll_band_id, step.step.ll)
            }),
        )?;
        let (hl, high_low_stride) = lookup_mapped_repeated_direct_band_layout_entry(
            &self.resources.band_sets,
            idwt_band_keys(self.plans, step_idx, |step| {
                (step.step.hl_band_id, step.step.hl)
            }),
        )?;
        let (lh, low_high_stride) = lookup_mapped_repeated_direct_band_layout_entry(
            &self.resources.band_sets,
            idwt_band_keys(self.plans, step_idx, |step| {
                (step.step.lh_band_id, step.step.lh)
            }),
        )?;
        let (hh, high_high_stride) = lookup_mapped_repeated_direct_band_layout_entry(
            &self.resources.band_sets,
            idwt_band_keys(self.plans, step_idx, |step| {
                (step.step.hh_band_id, step.step.hh)
            }),
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
        Ok(([ll, hl, lh, hh], params))
    }

    /// Encodes the last level `idwt` and the final `store` that reads it as
    /// one batched kernel writing the requested grayscale output, so neither
    /// the level's float plane nor the store's final plane is written. Applies
    /// only when every image's store reads the whole level unmoved; returns
    /// `false`, having encoded nothing, otherwise.
    pub(super) fn submit_idwt_into_gray_output(
        &mut self,
        step_idx: usize,
        idwt: &PreparedDirectIdwt,
        store: &j2k_native::J2kDirectStoreStep,
    ) -> Result<bool, Error> {
        let (Some(output), Some(encoder)) = (self.gray_output, self.compute_encoder) else {
            return Ok(false);
        };
        #[cfg(test)]
        if crate::engine::test_counters::per_image_idwt_forced()
            || !crate::engine::test_counters::decode_stage_enabled(
                crate::engine::test_counters::DecodeStageLimit::Full,
            )
        {
            return Ok(false);
        }
        let window = idwt.output_window;
        let reads_level = self.plans.iter().all(|plan| {
            matches!(
                (plan.steps.get(step_idx), plan.steps.get(step_idx + 1)),
                (
                    Some(PreparedDirectGrayscaleStep::Idwt(level)),
                    Some(PreparedDirectGrayscaleStep::Store(final_store)),
                ) if final_store.input_band_id == level.step.output_band_id
            )
        });
        if self.count < 2
            || self.resources.final_plane.is_some()
            || !reads_level
            || (
                store.source_x,
                store.source_y,
                store.output_x,
                store.output_y,
            ) != (0, 0, 0, 0)
            || (store.copy_width, store.copy_height) != (store.output_width, store.output_height)
            || (store.output_width, store.output_height) != (window.width(), window.height())
            || store.input_rect.width() != window.width()
        {
            return Ok(false);
        }
        let kernels = self.runtime.decode()?;
        let Some(pipeline) =
            fused_idwt_stacked_gray_store_pipeline(kernels, idwt.step.transform, output.kind)
        else {
            return Ok(false);
        };
        if !supports_fused_2d_idwt(kernels, pipeline, window.width(), window.height()) {
            return Ok(false);
        }
        let ([ll, hl, lh, hh], params) = self.stacked_idwt_inputs(step_idx, idwt)?;
        let image_stride =
            window
                .width()
                .checked_mul(window.height())
                .ok_or_else(|| Error::MetalKernel {
                    message: "J2K MetalDirect stacked grayscale image size exceeds u32".to_string(),
                })?;
        dispatch_fused_idwt_stacked_gray_store(
            encoder,
            pipeline,
            FusedStackedGrayStore {
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
                transform: idwt.step.transform,
                output: output.buffer,
                output_offset_bytes: output.offset_bytes,
                store: J2kStackedGrayStoreParams {
                    source_x: 0,
                    source_y: 0,
                    copy_width: store.copy_width,
                    copy_height: store.copy_height,
                    output_stride: store.output_width,
                    image_stride,
                    addend: store.addend,
                    round_centered: u32::from(self.round_centered_store),
                    max_value: output.max_value,
                    u8_scale: output.u8_scale,
                    u16_scale: output.u16_scale,
                    reserved: 0,
                },
            },
        );
        #[cfg(test)]
        crate::engine::test_counters::record_fused_idwt_store();
        self.resources.gray_output_written = true;
        for bands in &mut self.resources.band_sets {
            bands.clear();
        }
        Ok(true)
    }

    fn encode_stacked_idwt(
        &mut self,
        step_idx: usize,
        idwt: &PreparedDirectIdwt,
        output: &crate::metal_types::Buffer,
    ) -> Result<(), Error> {
        #[cfg(test)]
        if !crate::engine::test_counters::decode_stage_enabled(
            crate::engine::test_counters::DecodeStageLimit::Idwt,
        ) {
            return Ok(());
        }
        let ([ll, hl, lh, hh], params) = self.stacked_idwt_inputs(step_idx, idwt)?;
        let dispatch = RepeatedIdwtDispatch {
            kernels: self.runtime.decode()?,
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
            decoded: output,
        };
        match (idwt.step.transform, self.compute_encoder) {
            (J2kWaveletTransform::Reversible53, Some(encoder)) => {
                dispatch_reversible53_repeated_buffers_in_encoder_with_offsets(encoder, dispatch);
            }
            (J2kWaveletTransform::Reversible53, None) => {
                dispatch_reversible53_repeated_buffers_in_command_buffer_with_offsets(
                    self.command_buffers.idwt,
                    dispatch,
                )?;
            }
            (J2kWaveletTransform::Irreversible97, Some(encoder)) => {
                dispatch_irreversible97_repeated_buffers_in_encoder_with_offsets(encoder, dispatch);
            }
            (J2kWaveletTransform::Irreversible97, None) => {
                dispatch_irreversible97_repeated_buffers_in_command_buffer_with_offsets(
                    self.command_buffers.idwt.interleave,
                    dispatch,
                )?;
            }
        }
        Ok(())
    }

    fn encode_distinct_irreversible97_idwt(
        &mut self,
        step_idx: usize,
        output: &crate::metal_types::Buffer,
        span: &CheckedF32BatchSpan,
    ) -> Result<(), Error> {
        #[cfg(test)]
        if !crate::engine::test_counters::decode_stage_enabled(
            crate::engine::test_counters::DecodeStageLimit::Idwt,
        ) {
            return Ok(());
        }
        for (instance_idx, bands) in self.resources.band_sets.iter().enumerate() {
            let PreparedDirectGrayscaleStep::Idwt(step) = &self.plans[instance_idx].steps[step_idx]
            else {
                return Err(direct_preflight_invariant(
                    "IDWT step mismatch in stacked component batch",
                ));
            };
            let ll = lookup_direct_band_slice_entry(bands, step.step.ll_band_id, step.step.ll)?;
            let hl = lookup_direct_band_slice_entry(bands, step.step.hl_band_id, step.step.hl)?;
            let lh = lookup_direct_band_slice_entry(bands, step.step.lh_band_id, step.step.lh)?;
            let hh = lookup_direct_band_slice_entry(bands, step.step.hh_band_id, step.step.hh)?;
            let params =
                prepared_idwt_params(step, idwt_input_windows_from_slices(&ll, &hl, &lh, &hh));
            let dispatch = SingleIdwtDispatch {
                kernels: self.runtime.decode()?,
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
                decoded: output,
                decoded_offset: checked_f32_instance_offset(
                    span,
                    instance_idx,
                    "J2K MetalDirect stacked irreversible IDWT",
                )?,
            };
            if let Some(encoder) = self.compute_encoder {
                dispatch_irreversible97_single_decomposition_buffers_in_encoder_with_offsets(
                    encoder, dispatch,
                );
            } else {
                dispatch_irreversible97_single_decomposition_buffers_in_command_buffer_with_offsets(
                    self.command_buffers.idwt.interleave,
                    dispatch,
                )?;
            }
        }
        Ok(())
    }

    pub(super) fn submit_idwt(
        &mut self,
        step_idx: usize,
        idwt: &PreparedDirectIdwt,
    ) -> Result<(), Error> {
        let span = checked_f32_dimension_span(
            idwt.output_window.width(),
            idwt.output_window.height(),
            self.count,
            "J2K MetalDirect stacked IDWT",
        )?;
        let output = take_f32_scratch_buffer(self.runtime, span.total_elements)?;
        let use_single = self.count == 1;
        #[cfg(test)]
        let use_single = use_single || crate::engine::test_counters::per_image_idwt_forced();
        let encode_started = self.profile_stages.then(Instant::now);
        // Batches reconstruct every instance in one dispatch per level. With
        // fused lifting this is about 11% faster in GPU time than per-image
        // reconstruction at 16 x 1024x1024.
        if idwt.step.transform == J2kWaveletTransform::Irreversible97 && use_single {
            self.encode_distinct_irreversible97_idwt(step_idx, &output.buffer, &span)?;
        } else {
            self.encode_stacked_idwt(step_idx, idwt, &output.buffer)?;
        }
        if let Some(encoder) = self.compute_encoder {
            encoder.memory_barrier_with_resources(&[&output.buffer]);
        }
        if let Some(started) = encode_started {
            self.stage_timings.metal_idwt_encode += elapsed_us(started);
        }

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
                offset_bytes: checked_f32_instance_offset(
                    &span,
                    instance_idx,
                    "J2K MetalDirect stacked IDWT output",
                )?,
                window: step.output_window,
            });
        }
        self.scratch_buffers.push(output);
        Ok(())
    }
}

fn idwt_band_keys<'a>(
    plans: &'a [&PreparedDirectGrayscalePlan],
    step_idx: usize,
    select: fn(&PreparedDirectIdwt) -> (crate::engine::J2kDirectBandId, j2k_native::J2kRect),
) -> impl Iterator<Item = Result<(crate::engine::J2kDirectBandId, j2k_native::J2kRect), Error>> + 'a
{
    plans
        .iter()
        .map(move |plan| match plan.steps.get(step_idx) {
            Some(PreparedDirectGrayscaleStep::Idwt(step)) => Ok(select(step)),
            _ => Err(direct_preflight_invariant(
                "IDWT step mismatch in mapped stacked component batch",
            )),
        })
}
