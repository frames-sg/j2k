// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::compute::decode_dispatch::{
    dispatch_irreversible97_single_decomposition_buffers_in_command_buffer_with_offsets,
    dispatch_reversible53_repeated_buffers_in_command_buffer_with_offsets,
    dispatch_reversible53_single_decomposition_buffers_in_command_buffer_with_offsets,
    IdwtSubBandBuffers, RepeatedIdwtDispatch, SingleIdwtDispatch,
};
use crate::compute::direct_roi::{
    idwt_input_windows_from_slices, prepared_idwt_params, repeated_idwt_params,
    PreparedIdwtInputStrides,
};
use crate::compute::direct_stacked_batch::validation::{
    checked_f32_dimension_span, checked_f32_instance_offset,
};
use crate::compute::direct_stacked_batch::{
    lookup_direct_band_slice_entry, lookup_repeated_direct_band_layout_entry, DirectBandSlice,
};
use crate::compute::{
    take_f32_scratch_buffer, DirectIdwtCommandBuffers, Error, J2kWaveletTransform,
    PreparedDirectIdwt,
};

use super::RepeatedGrayscaleExecution;

impl RepeatedGrayscaleExecution<'_> {
    pub(super) fn encode_stacked_idwt(&mut self, idwt: &PreparedDirectIdwt) -> Result<(), Error> {
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
        let span = checked_f32_dimension_span(
            idwt.output_window.width(),
            idwt.output_window.height(),
            self.count,
            "J2K MetalDirect repeated stacked IDWT",
        )?;
        let output = take_f32_scratch_buffer(self.runtime, span.total_elements)?;
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
        )?;
        for (instance_idx, bands) in self.band_sets.iter_mut().enumerate() {
            bands.push(DirectBandSlice {
                band_id: idwt.step.output_band_id,
                buffer: output.buffer.clone(),
                offset_bytes: checked_f32_instance_offset(
                    &span,
                    instance_idx,
                    "J2K MetalDirect repeated stacked IDWT output",
                )?,
                window: idwt.output_window,
            });
        }
        self.scratch_buffers.push(output);
        Ok(())
    }

    pub(super) fn encode_per_instance_idwt(
        &mut self,
        idwt: &PreparedDirectIdwt,
    ) -> Result<(), Error> {
        let span = checked_f32_dimension_span(
            idwt.output_window.width(),
            idwt.output_window.height(),
            1,
            "J2K MetalDirect repeated per-instance IDWT",
        )?;
        for bands in &mut self.band_sets {
            let ll = lookup_direct_band_slice_entry(bands, idwt.step.ll_band_id, idwt.step.ll)?;
            let hl = lookup_direct_band_slice_entry(bands, idwt.step.hl_band_id, idwt.step.hl)?;
            let lh = lookup_direct_band_slice_entry(bands, idwt.step.lh_band_id, idwt.step.lh)?;
            let hh = lookup_direct_band_slice_entry(bands, idwt.step.hh_band_id, idwt.step.hh)?;
            let params =
                prepared_idwt_params(idwt, idwt_input_windows_from_slices(&ll, &hl, &lh, &hh));
            let output = take_f32_scratch_buffer(self.runtime, span.total_elements)?;
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
                    )?;
                }
                J2kWaveletTransform::Irreversible97 => {
                    self.status_checks.push(
                        dispatch_irreversible97_single_decomposition_buffers_in_command_buffer_with_offsets(
                            self.command_buffer,
                            dispatch,
                        )?,
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

    pub(super) fn encode_idwt(&mut self, idwt: &PreparedDirectIdwt) -> Result<(), Error> {
        if idwt.step.transform == J2kWaveletTransform::Reversible53 && self.stacked_outputs {
            self.encode_stacked_idwt(idwt)
        } else {
            self.stacked_outputs = false;
            self.encode_per_instance_idwt(idwt)
        }
    }
}
