// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::compute::decode_dispatch::{
    encode_repeated_classic_sub_band_group_to_buffer_in_command_buffer,
    encode_repeated_classic_sub_band_to_buffer_in_command_buffer,
    encode_repeated_ht_sub_band_group_to_buffer_in_command_buffer,
    encode_repeated_ht_sub_band_to_buffer_in_command_buffer,
};
use crate::compute::direct_roi::BandRequiredRegion;
use crate::compute::direct_stacked_batch::validation::{
    checked_f32_batch_span, checked_f32_dimension_span, checked_f32_element_offset,
    checked_f32_instance_offset, CheckedF32BatchSpan,
};
use crate::compute::{
    take_f32_scratch_buffer, Error, PreparedClassicSubBand, PreparedClassicSubBandGroup,
    PreparedHtSubBand, PreparedHtSubBandGroup,
};

use super::{DirectBandSlice, DirectScratchBuffer, RepeatedGrayscaleExecution};

impl RepeatedGrayscaleExecution<'_> {
    pub(super) fn encode_classic_group(
        &mut self,
        group: &PreparedClassicSubBandGroup,
    ) -> Result<(), Error> {
        let span = checked_f32_batch_span(
            group.total_coefficients,
            self.count,
            "J2K MetalDirect repeated classic group",
        )?;
        let output = take_f32_scratch_buffer(self.runtime, span.total_elements)?;
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
        for (instance_idx, bands) in self.band_sets.iter_mut().enumerate() {
            for member in &group.members {
                bands.push(DirectBandSlice {
                    band_id: member.band_id,
                    buffer: output.buffer.clone(),
                    offset_bytes: checked_f32_element_offset(
                        &span,
                        instance_idx,
                        member.offset_elements,
                        "J2K MetalDirect repeated classic group member",
                    )?,
                    window: member.window,
                });
            }
        }
        self.scratch_buffers.push(output);
        Ok(())
    }

    pub(super) fn encode_ht_group(&mut self, group: &PreparedHtSubBandGroup) -> Result<(), Error> {
        let span = checked_f32_batch_span(
            group.total_coefficients,
            self.count,
            "J2K MetalDirect repeated HT group",
        )?;
        let output = take_f32_scratch_buffer(self.runtime, span.total_elements)?;
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
        for (instance_idx, bands) in self.band_sets.iter_mut().enumerate() {
            for member in &group.members {
                bands.push(DirectBandSlice {
                    band_id: member.band_id,
                    buffer: output.buffer.clone(),
                    offset_bytes: checked_f32_element_offset(
                        &span,
                        instance_idx,
                        member.offset_elements,
                        "J2K MetalDirect repeated HT group member",
                    )?,
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
        span: &CheckedF32BatchSpan,
    ) -> Result<(), Error> {
        for (instance_idx, bands) in self.band_sets.iter_mut().enumerate() {
            bands.push(DirectBandSlice {
                band_id,
                buffer: output.buffer.clone(),
                offset_bytes: checked_f32_instance_offset(
                    span,
                    instance_idx,
                    "J2K MetalDirect repeated sub-band",
                )?,
                window: BandRequiredRegion::full(width, height),
            });
        }
        self.scratch_buffers.push(output);
        Ok(())
    }

    pub(super) fn encode_classic_sub_band(
        &mut self,
        sub_band: &PreparedClassicSubBand,
    ) -> Result<(), Error> {
        let span = checked_f32_dimension_span(
            sub_band.width,
            sub_band.height,
            self.count,
            "J2K MetalDirect repeated classic sub-band",
        )?;
        let output = take_f32_scratch_buffer(self.runtime, span.total_elements)?;
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
        self.retain_sub_band(
            output,
            sub_band.band_id,
            sub_band.width,
            sub_band.height,
            &span,
        )
    }

    pub(super) fn encode_ht_sub_band(&mut self, sub_band: &PreparedHtSubBand) -> Result<(), Error> {
        let span = checked_f32_dimension_span(
            sub_band.width,
            sub_band.height,
            self.count,
            "J2K MetalDirect repeated HT sub-band",
        )?;
        let output = take_f32_scratch_buffer(self.runtime, span.total_elements)?;
        let (buffers, status_check) = encode_repeated_ht_sub_band_to_buffer_in_command_buffer(
            self.runtime,
            self.command_buffer,
            sub_band,
            self.count,
            &output.buffer,
        )?;
        self.retained_buffers.extend(buffers);
        self.status_checks.push(status_check);
        self.retain_sub_band(
            output,
            sub_band.band_id,
            sub_band.width,
            sub_band.height,
            &span,
        )
    }
}
