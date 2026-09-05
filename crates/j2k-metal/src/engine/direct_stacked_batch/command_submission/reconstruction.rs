// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(target_os = "macos")]
use crate::metal_types::prelude::*;

use std::time::Instant;

use crate::engine::decode_dispatch::{
    dispatch_irreversible97_repeated_buffers_in_command_buffer_with_offsets,
    dispatch_irreversible97_repeated_buffers_in_encoder_with_offsets,
    dispatch_irreversible97_single_decomposition_buffers_in_command_buffer_with_offsets,
    dispatch_irreversible97_single_decomposition_buffers_in_encoder_with_offsets,
    dispatch_reversible53_repeated_buffers_in_command_buffer_with_offsets,
    dispatch_reversible53_repeated_buffers_in_encoder_with_offsets, IdwtSubBandBuffers,
    RepeatedIdwtDispatch, SingleIdwtDispatch,
};
use crate::engine::direct_roi::{
    idwt_input_windows_from_slices, prepared_idwt_params, repeated_idwt_params,
    PreparedIdwtInputStrides,
};
use crate::engine::{
    direct_preflight_invariant, elapsed_us, take_f32_scratch_buffer, Error, J2kWaveletTransform,
    PreparedDirectGrayscaleStep, PreparedDirectIdwt,
};

use super::super::resources::{
    lookup_direct_band_slice_entry, lookup_repeated_direct_band_layout_entry, DirectBandSlice,
};
use super::super::validation::{
    checked_f32_dimension_span, checked_f32_instance_offset, CheckedF32BatchSpan,
};
use super::SubmissionContext;

impl SubmissionContext<'_, '_, '_> {
    fn encode_stacked_idwt(
        &mut self,
        idwt: &PreparedDirectIdwt,
        output: &crate::metal_types::Buffer,
    ) -> Result<(), Error> {
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
                dispatch_irreversible97_repeated_buffers_in_encoder_with_offsets(
                    encoder, dispatch,
                )?;
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
        let encode_started = self.profile_stages.then(Instant::now);
        // Bound the simultaneously reconstructed planes: the measured 16-image
        // 640x480 stage wins, while a 64 MiB 1024x1024 stage regresses (P20).
        static DISABLE_CHUNKS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let disable_chunks = *DISABLE_CHUNKS
            .get_or_init(|| std::env::var_os("J2K_METAL_DISABLE_IDWT97_CHUNKS").is_some());
        let use_single = self.count == 1 || (disable_chunks && span.total_bytes > 20 * 1024 * 1024);
        if idwt.step.transform == J2kWaveletTransform::Irreversible97 && use_single {
            self.encode_distinct_irreversible97_idwt(step_idx, &output.buffer, &span)?;
        } else {
            self.encode_stacked_idwt(idwt, &output.buffer)?;
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
