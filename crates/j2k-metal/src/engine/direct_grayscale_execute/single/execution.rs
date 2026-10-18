// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(target_os = "macos")]
use crate::metal_types::prelude::*;

use crate::engine::decode_dispatch::store::GraySourceSigning;
use crate::metal_types::ComputeCommandEncoderRef;
use j2k_metal_support::MetalImageDestination;
use j2k_native::J2kDirectStoreStep;

use super::super::{
    dispatch_irreversible97_single_decomposition_buffers_in_encoder_with_offsets,
    dispatch_reversible53_single_decomposition_buffers_in_encoder_with_offsets,
    dispatch_store_component_buffer_in_encoder_with_offsets,
    encode_gray_plane_to_surface_in_encoder, encode_gray_store_to_destination_in_encoder,
    encode_gray_store_to_surface_in_encoder,
    encode_prepared_classic_sub_band_group_to_buffer_in_encoder,
    encode_prepared_classic_sub_band_to_buffer_in_encoder,
    encode_prepared_ht_sub_band_group_to_buffer_in_encoder,
    encode_prepared_ht_sub_band_to_buffer_in_encoder, extend_preallocated_retained_buffers,
    idwt_input_windows_from_slices, j2k_scalar_pack_params, lookup_direct_band_slice,
    lookup_direct_band_slice_entry, prepared_idwt_output_len, prepared_idwt_params,
    take_f32_scratch_buffer, BandRequiredRegion, Buffer, DirectBandSlice, DirectScratchBuffer,
    DirectStatusCheck, Error, GrayStoreDestinationRequest, IdwtSubBandBuffers, J2kGrayStoreParams,
    J2kStoreParams, J2kWaveletTransform, MetalRuntime, PixelFormat, PreparedDirectGrayscaleStep,
    SingleIdwtDispatch, Surface,
};
use crate::engine::abi::J2kIdwtSingleDecompositionParams;
use crate::engine::decode_dispatch::{
    fused_idwt_gray_store_supported, gray_store_kind, FusedIdwtStoreSource, GrayStoreSource,
};
use crate::engine::{
    direct_roi::checked_f32_span, PreparedClassicSubBand, PreparedClassicSubBandGroup,
    PreparedDirectIdwt, PreparedHtSubBand, PreparedHtSubBandGroup,
};

pub(super) struct SingleGrayscaleExecution<'a> {
    pub(super) runtime: &'a MetalRuntime,
    pub(super) encoder: &'a ComputeCommandEncoderRef,
    pub(super) fmt: PixelFormat,
    pub(super) dimensions: (u32, u32),
    pub(super) bit_depth: u8,
    pub(super) retained_buffers: &'a mut Vec<Buffer>,
    pub(super) status_checks: &'a mut Vec<DirectStatusCheck>,
    pub(super) scratch_buffers: &'a mut Vec<DirectScratchBuffer>,
    pub(super) bands: Vec<DirectBandSlice>,
    pub(super) final_surface: Option<Surface>,
    pub(super) destination: Option<&'a MetalImageDestination>,
    /// Whether a destination receives native-precision samples (batch
    /// contracts) instead of the surface store's format conversion.
    pub(super) native_destination: bool,
    pub(super) destination_item_index: usize,
    pub(super) destination_written: bool,
    pub(super) round_centered_store: bool,
}

impl SingleGrayscaleExecution<'_> {
    pub(super) fn encode_classic_group(
        &mut self,
        group: &PreparedClassicSubBandGroup,
    ) -> Result<(), Error> {
        let output_span = checked_f32_span(
            group.total_coefficients,
            1,
            "classic J2K MetalDirect single grouped coefficients",
        )?;
        let output = take_f32_scratch_buffer(self.runtime, output_span.elements)?;
        let (buffers, status_check) = encode_prepared_classic_sub_band_group_to_buffer_in_encoder(
            self.runtime,
            self.encoder,
            group,
            &output.buffer,
            self.scratch_buffers,
        )?;
        extend_preallocated_retained_buffers(self.retained_buffers, buffers)?;
        self.status_checks.push(status_check);
        self.encoder
            .memory_barrier_with_resources(&[&output.buffer]);
        for member in &group.members {
            let offset_bytes = checked_f32_span(
                member.offset_elements,
                1,
                "classic J2K MetalDirect single grouped member offset",
            )?
            .bytes;
            self.bands.push(DirectBandSlice {
                band_id: member.band_id,
                buffer: output.buffer.clone(),
                offset_bytes,
                window: member.window,
            });
        }
        self.scratch_buffers.push(output);
        Ok(())
    }

    pub(super) fn encode_ht_group(&mut self, group: &PreparedHtSubBandGroup) -> Result<(), Error> {
        let output_span = checked_f32_span(
            group.total_coefficients,
            1,
            "HTJ2K MetalDirect single grouped coefficients",
        )?;
        let output = take_f32_scratch_buffer(self.runtime, output_span.elements)?;
        let (buffers, status_check) = encode_prepared_ht_sub_band_group_to_buffer_in_encoder(
            self.runtime,
            self.encoder,
            group,
            &output.buffer,
        )?;
        extend_preallocated_retained_buffers(self.retained_buffers, buffers)?;
        self.status_checks.push(status_check);
        self.encoder
            .memory_barrier_with_resources(&[&output.buffer]);
        for member in &group.members {
            let offset_bytes = checked_f32_span(
                member.offset_elements,
                1,
                "HTJ2K MetalDirect single grouped member offset",
            )?
            .bytes;
            self.bands.push(DirectBandSlice {
                band_id: member.band_id,
                buffer: output.buffer.clone(),
                offset_bytes,
                window: member.window,
            });
        }
        self.scratch_buffers.push(output);
        Ok(())
    }

    fn encode_classic_sub_band(&mut self, sub_band: &PreparedClassicSubBand) -> Result<(), Error> {
        let output_span = checked_f32_span(
            sub_band.width as usize,
            sub_band.height as usize,
            "classic J2K MetalDirect single sub-band",
        )?;
        let output = take_f32_scratch_buffer(self.runtime, output_span.elements)?;
        let (buffers, status_check) = encode_prepared_classic_sub_band_to_buffer_in_encoder(
            self.runtime,
            self.encoder,
            sub_band,
            &output.buffer,
            self.scratch_buffers,
        )?;
        extend_preallocated_retained_buffers(self.retained_buffers, buffers)?;
        self.status_checks.push(status_check);
        self.encoder
            .memory_barrier_with_resources(&[&output.buffer]);
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
        let output_span = checked_f32_span(
            sub_band.width as usize,
            sub_band.height as usize,
            "HTJ2K MetalDirect single sub-band",
        )?;
        let output = take_f32_scratch_buffer(self.runtime, output_span.elements)?;
        let (buffers, status_check) = encode_prepared_ht_sub_band_to_buffer_in_encoder(
            self.runtime,
            self.encoder,
            sub_band,
            &output.buffer,
        )?;
        extend_preallocated_retained_buffers(self.retained_buffers, buffers)?;
        self.status_checks.push(status_check);
        self.encoder
            .memory_barrier_with_resources(&[&output.buffer]);
        self.bands.push(DirectBandSlice {
            band_id: sub_band.band_id,
            buffer: output.buffer.clone(),
            offset_bytes: 0,
            window: BandRequiredRegion::full(sub_band.width, sub_band.height),
        });
        self.scratch_buffers.push(output);
        Ok(())
    }

    /// The four sub-band slices `idwt` reads, and its kernel parameters.
    fn idwt_inputs(
        &self,
        idwt: &PreparedDirectIdwt,
    ) -> Result<([DirectBandSlice; 4], J2kIdwtSingleDecompositionParams), Error> {
        let ll = lookup_direct_band_slice_entry(&self.bands, idwt.step.ll_band_id, idwt.step.ll)?;
        let hl = lookup_direct_band_slice_entry(&self.bands, idwt.step.hl_band_id, idwt.step.hl)?;
        let lh = lookup_direct_band_slice_entry(&self.bands, idwt.step.lh_band_id, idwt.step.lh)?;
        let hh = lookup_direct_band_slice_entry(&self.bands, idwt.step.hh_band_id, idwt.step.hh)?;
        let params = prepared_idwt_params(idwt, idwt_input_windows_from_slices(&ll, &hl, &lh, &hh));
        Ok(([ll, hl, lh, hh], params))
    }

    fn encode_idwt(&mut self, idwt: &PreparedDirectIdwt) -> Result<(), Error> {
        let ([ll, hl, lh, hh], params) = self.idwt_inputs(idwt)?;
        let output = take_f32_scratch_buffer(self.runtime, prepared_idwt_output_len(idwt)?)?;
        let dispatch = SingleIdwtDispatch {
            kernels: self.runtime.decode()?,
            sub_bands: sub_band_buffers(&ll, &hl, &lh, &hh),
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
                dispatch_irreversible97_single_decomposition_buffers_in_encoder_with_offsets(
                    self.encoder,
                    dispatch,
                );
            }
        }
        self.encoder
            .memory_barrier_with_resources(&[&output.buffer]);
        self.bands.push(DirectBandSlice {
            band_id: idwt.step.output_band_id,
            buffer: output.buffer.clone(),
            offset_bytes: 0,
            window: idwt.output_window,
        });
        self.scratch_buffers.push(output);
        Ok(())
    }

    /// Encodes `idwt` and the grayscale `store` that reads its output as one
    /// kernel that never writes the level's float plane, when the store reads
    /// that whole plane row-aligned, nothing in `later_steps` reads it, and
    /// the tile kernel supports the level. Returns `false`, having encoded
    /// nothing, otherwise.
    pub(super) fn encode_idwt_into_store(
        &mut self,
        idwt: &PreparedDirectIdwt,
        store: &J2kDirectStoreStep,
        later_steps: &[PreparedDirectGrayscaleStep],
    ) -> Result<bool, Error> {
        let band = idwt.step.output_band_id;
        if !is_gray(self.fmt)
            || store.input_band_id != band
            || later_steps.iter().any(|step| step_reads_band(step, band))
        {
            return Ok(false);
        }
        let ([ll, hl, lh, hh], params) = self.idwt_inputs(idwt)?;
        if store.input_rect.width() != params.width
            || store
                .source_x
                .checked_add(store.copy_width)
                .is_none_or(|end| end > params.width)
            || store
                .source_y
                .checked_add(store.copy_height)
                .is_none_or(|end| end > params.height)
        {
            return Ok(false);
        }
        let (store_params, signing) = self.gray_store_params(store);
        let Ok(kind) = gray_store_kind(self.fmt, signing) else {
            return Ok(false);
        };
        let source = FusedIdwtStoreSource {
            sub_bands: sub_band_buffers(&ll, &hl, &lh, &hh),
            params,
            transform: idwt.step.transform,
        };
        if !fused_idwt_gray_store_supported(self.runtime.decode()?, &source, kind) {
            return Ok(false);
        }
        self.encode_gray_store(GrayStoreSource::Idwt(source), store_params, signing)?;
        #[cfg(test)]
        crate::engine::test_counters::record_fused_idwt_store();
        Ok(true)
    }

    fn encode_store(&mut self, store: &J2kDirectStoreStep) -> Result<(), Error> {
        let (input, input_offset) =
            lookup_direct_band_slice(&self.bands, store.input_band_id, store.input_rect)?;
        if is_gray(self.fmt) {
            let (params, signing) = self.gray_store_params(store);
            self.encode_gray_store(
                GrayStoreSource::Plane {
                    buffer: &input,
                    offset_bytes: input_offset,
                },
                params,
                signing,
            )?;
        } else {
            self.encode_store_through_plane(store, &input, input_offset)?;
        }
        Ok(())
    }

    fn gray_store_params(
        &self,
        store: &J2kDirectStoreStep,
    ) -> (J2kGrayStoreParams, GraySourceSigning) {
        let scale = j2k_scalar_pack_params(u32::from(self.bit_depth));
        let signing = GraySourceSigning::for_store(self.fmt, self.bit_depth, store.addend);
        let max_value =
            if self.fmt == PixelFormat::GrayI16 || signing != GraySourceSigning::Unsigned {
                crate::engine::decode_dispatch::store::signed_store_max_value(self.bit_depth)
            } else {
                scale.max_value
            };
        let params = J2kGrayStoreParams {
            input_width: store.input_rect.width(),
            source_x: store.source_x,
            source_y: store.source_y,
            copy_width: store.copy_width,
            copy_height: store.copy_height,
            output_width: store.output_width,
            output_stride: store.output_width,
            output_item_offset: 0,
            output_x: store.output_x,
            output_y: store.output_y,
            addend: store.addend,
            max_value,
            u8_scale: if self.destination.is_some() && self.native_destination {
                1.0
            } else {
                scale.u8_scale
            },
            u16_scale: scale.u16_scale,
        };
        (params, signing)
    }

    /// Stores `source` into the destination, or into a new final surface.
    fn encode_gray_store(
        &mut self,
        source: GrayStoreSource<'_>,
        params: J2kGrayStoreParams,
        signing: GraySourceSigning,
    ) -> Result<(), Error> {
        if let Some(destination) = self.destination {
            encode_gray_store_to_destination_in_encoder(&GrayStoreDestinationRequest {
                runtime: self.runtime,
                encoder: self.encoder,
                source,
                params,
                dims: self.dimensions,
                fmt: self.fmt,
                destination,
                destination_item_index: self.destination_item_index,
                signing,
            })?;
            self.destination_written = true;
        } else {
            self.final_surface = Some(encode_gray_store_to_surface_in_encoder(
                self.runtime,
                self.encoder,
                source,
                params,
                self.dimensions,
                self.fmt,
                signing,
            )?);
        }
        Ok(())
    }

    /// Stores into an f32 plane, then packs it to a non-gray output format.
    fn encode_store_through_plane(
        &mut self,
        store: &J2kDirectStoreStep,
        input: &Buffer,
        input_offset: usize,
    ) -> Result<(), Error> {
        let output_span = checked_f32_span(
            store.output_width as usize,
            store.output_height as usize,
            "J2K MetalDirect single stored component plane",
        )?;
        let output = take_f32_scratch_buffer(self.runtime, output_span.elements)?;
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
            round_centered: u32::from(self.round_centered_store),
        };
        dispatch_store_component_buffer_in_encoder_with_offsets(
            self.runtime.decode()?,
            self.encoder,
            input,
            input_offset,
            &output.buffer,
            0,
            params,
        );
        self.encoder
            .memory_barrier_with_resources(&[&output.buffer]);
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
        Ok(())
    }

    pub(super) fn encode_step(&mut self, step: &PreparedDirectGrayscaleStep) -> Result<(), Error> {
        match step {
            PreparedDirectGrayscaleStep::ClassicSubBand(sub_band) => {
                self.encode_classic_sub_band(sub_band)
            }
            PreparedDirectGrayscaleStep::HtSubBand(sub_band) => self.encode_ht_sub_band(sub_band),
            PreparedDirectGrayscaleStep::Idwt(idwt) => self.encode_idwt(idwt),
            PreparedDirectGrayscaleStep::Store(store) => self.encode_store(store),
        }
    }

    pub(super) fn finish(self) -> Result<Option<Surface>, Error> {
        if self.destination.is_some() {
            return self
                .destination_written
                .then_some(None)
                .ok_or_else(|| Error::MetalKernel {
                    message:
                        "J2K MetalDirect prepared grayscale plan did not write its destination"
                            .to_string(),
                });
        }
        self.final_surface
            .map(Some)
            .ok_or_else(|| Error::MetalKernel {
                message:
                    "J2K MetalDirect prepared grayscale plan did not produce a final stored plane"
                        .to_string(),
            })
    }
}

fn is_gray(fmt: PixelFormat) -> bool {
    matches!(
        fmt,
        PixelFormat::Gray8 | PixelFormat::Gray16 | PixelFormat::GrayI16
    )
}

fn sub_band_buffers<'b>(
    ll: &'b DirectBandSlice,
    hl: &'b DirectBandSlice,
    lh: &'b DirectBandSlice,
    hh: &'b DirectBandSlice,
) -> IdwtSubBandBuffers<'b> {
    IdwtSubBandBuffers {
        ll: &ll.buffer,
        ll_offset: ll.offset_bytes,
        hl: &hl.buffer,
        hl_offset: hl.offset_bytes,
        lh: &lh.buffer,
        lh_offset: lh.offset_bytes,
        hh: &hh.buffer,
        hh_offset: hh.offset_bytes,
    }
}

fn step_reads_band(step: &PreparedDirectGrayscaleStep, band: j2k_native::J2kDirectBandId) -> bool {
    match step {
        PreparedDirectGrayscaleStep::Idwt(idwt) => [
            idwt.step.ll_band_id,
            idwt.step.hl_band_id,
            idwt.step.lh_band_id,
            idwt.step.hh_band_id,
        ]
        .contains(&band),
        PreparedDirectGrayscaleStep::Store(store) => store.input_band_id == band,
        PreparedDirectGrayscaleStep::ClassicSubBand(_)
        | PreparedDirectGrayscaleStep::HtSubBand(_) => false,
    }
}
