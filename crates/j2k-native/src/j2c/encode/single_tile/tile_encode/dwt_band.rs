// SPDX-License-Identifier: MIT OR Apache-2.0

//! Contiguous adaptation of borrowed packed-DWT bands for Tier-1 hooks.

use alloc::vec::Vec;

use crate::j2c::encode::allocation::{
    checked_add_bytes, checked_element_bytes, host_allocation_failed,
};
use crate::j2c::encode::{
    prepare_subband_for_session, BlockCodingMode, ComponentRoiEncodeRegion,
    F32SubbandEncodeRequest, J2kEncodeStageAccelerator, NativeEncodePipelineResult,
    NativeEncodeSession, PreparedEncodeSubband, QuantStepSize, SubBandType, SubbandSamples,
};

use super::super::coefficient_source::DwtBandView;
use crate::j2c::fdwt::PackedSubbandView;

pub(super) struct DwtBandEncodeRequest<'a, 'input> {
    pub(super) band: DwtBandView<'a>,
    pub(super) settings: DwtBandEncodeSettings<'a, 'input>,
}

pub(super) struct DwtBandEncodeSettings<'a, 'input> {
    pub(super) origin: (u32, u32),
    pub(super) step_size: &'a QuantStepSize,
    pub(super) bit_depth: u8,
    pub(super) guard_bits: u8,
    pub(super) reversible: bool,
    pub(super) block_coding_mode: BlockCodingMode,
    pub(super) cb_width: u32,
    pub(super) cb_height: u32,
    pub(super) sub_band_type: SubBandType,
    pub(super) roi_shift: u8,
    pub(super) roi_regions: &'a [ComponentRoiEncodeRegion],
    pub(super) roi_scale: u32,
    pub(super) ht_target_coding_passes: u8,
    pub(super) single_layer: bool,
    pub(super) session: &'a NativeEncodeSession<'input>,
    pub(super) retained_base_bytes: usize,
}

impl<'a, 'input> DwtBandEncodeSettings<'a, 'input> {
    fn with_coefficients<'request>(
        self,
        coefficients: SubbandSamples<'request>,
        width: u32,
        height: u32,
        retained_base_bytes: usize,
    ) -> F32SubbandEncodeRequest<'request, 'input>
    where
        'a: 'request,
    {
        F32SubbandEncodeRequest {
            origin: self.origin,
            coefficients,
            width,
            height,
            step_size: self.step_size,
            bit_depth: self.bit_depth,
            guard_bits: self.guard_bits,
            reversible: self.reversible,
            block_coding_mode: self.block_coding_mode,
            cb_width: self.cb_width,
            cb_height: self.cb_height,
            sub_band_type: self.sub_band_type,
            roi_shift: self.roi_shift,
            roi_regions: self.roi_regions,
            roi_scale: self.roi_scale,
            ht_target_coding_passes: self.ht_target_coding_passes,
            single_layer: self.single_layer,
            session: self.session,
            retained_base_bytes,
        }
    }
}

pub(super) fn prepare_dwt_band_for_session(
    request: DwtBandEncodeRequest<'_, '_>,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> NativeEncodePipelineResult<PreparedEncodeSubband> {
    let DwtBandEncodeRequest { band, settings } = request;
    match band {
        DwtBandView::Contiguous {
            coefficients,
            width,
            height,
        } => {
            let retained_base_bytes = settings.retained_base_bytes;
            let samples = SubbandSamples::Contiguous(coefficients);
            prepare_subband_for_session(
                &settings.with_coefficients(samples, width, height, retained_base_bytes),
                accelerator,
            )
        }
        // No stage hook reads the band, so code blocks quantize from its rows.
        DwtBandView::Packed(view)
            if accelerator.declines_every_stage() && settings.roi_shift == 0 =>
        {
            let (width, height) = (view.width(), view.height());
            let retained_base_bytes = settings.retained_base_bytes;
            let samples = SubbandSamples::Packed(view);
            prepare_subband_for_session(
                &settings.with_coefficients(samples, width, height, retained_base_bytes),
                accelerator,
            )
        }
        DwtBandView::Packed(view) => {
            let coefficient_count = usize::try_from(view.width())
                .map_err(|_| crate::EncodeError::ArithmeticOverflow {
                    what: "packed DWT subband width",
                })?
                .checked_mul(usize::try_from(view.height()).map_err(|_| {
                    crate::EncodeError::ArithmeticOverflow {
                        what: "packed DWT subband height",
                    }
                })?)
                .ok_or(crate::EncodeError::ArithmeticOverflow {
                    what: "packed DWT contiguous subband sample count",
                })?;
            let requested_bytes = checked_element_bytes::<f32>(
                coefficient_count,
                "packed DWT contiguous subband samples",
            )?;
            settings.session.checked_phase(
                checked_add_bytes(
                    settings.retained_base_bytes,
                    requested_bytes,
                    "packed DWT contiguous subband copy",
                )?,
                "packed DWT contiguous subband copy",
            )?;
            let mut coefficients = Vec::new();
            coefficients
                .try_reserve_exact(coefficient_count)
                .map_err(|_| {
                    host_allocation_failed("packed DWT contiguous subband samples", requested_bytes)
                })?;
            let actual_bytes = checked_element_bytes::<f32>(
                coefficients.capacity(),
                "packed DWT contiguous subband samples",
            )?;
            let retained_base_bytes = checked_add_bytes(
                settings.retained_base_bytes,
                actual_bytes,
                "packed DWT contiguous subband copy",
            )?;
            settings
                .session
                .checked_phase(retained_base_bytes, "packed DWT contiguous subband copy")?;
            let parallel = accelerator.prefer_parallel_cpu_code_block_fallback();
            copy_band_rows(view, &mut coefficients, coefficient_count, parallel)?;
            if coefficients.len() != coefficient_count {
                return Err(crate::EncodeError::InternalInvariant {
                    what: "packed DWT contiguous subband copy length mismatch",
                }
                .into());
            }
            prepare_subband_for_session(
                &settings.with_coefficients(
                    SubbandSamples::Contiguous(&coefficients),
                    view.width(),
                    view.height(),
                    retained_base_bytes,
                ),
                accelerator,
            )
        }
    }
}

/// Copy a packed band's rows into `coefficients`, reserved for all of them;
/// `parallel` lets large bands copy their rows on the Rayon pool.
fn copy_band_rows(
    view: PackedSubbandView<'_, f32>,
    coefficients: &mut Vec<f32>,
    coefficient_count: usize,
    parallel: bool,
) -> NativeEncodePipelineResult<()> {
    let missing = || crate::EncodeError::InternalInvariant {
        what: "validated packed DWT subband row is missing",
    };
    #[cfg(feature = "parallel")]
    if parallel && coefficient_count >= 1 << 16 && rayon::current_num_threads() > 1 {
        use rayon::prelude::*;
        coefficients.resize(coefficient_count, 0.0);
        let width = coefficient_count / view.height() as usize;
        return coefficients
            .par_chunks_mut(width)
            .enumerate()
            .try_for_each(|(y, out)| {
                let row = u32::try_from(y)
                    .ok()
                    .and_then(|y| view.row(y))
                    .ok_or_else(missing)?;
                out.copy_from_slice(row);
                Ok(())
            });
    }
    let _ = (parallel, coefficient_count);
    for row_index in 0..view.height() {
        coefficients.extend_from_slice(view.row(row_index).ok_or_else(missing)?);
    }
    Ok(())
}
