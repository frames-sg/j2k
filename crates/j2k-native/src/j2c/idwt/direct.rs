// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec::Vec;

use super::super::build::Decomposition;
use super::super::codestream::WaveletTransform;
use super::super::decode::DecompositionStorage;
use super::super::rect::IntRect;
use super::horizontal::filter_horizontal;
use super::interleave::interleave_samples;
use super::model::{CoefficientSource, IDWTInput, IDWTTempOutput};
use super::roi::interleave_samples_roi;
use super::vertical::filter_vertical;
use crate::error::{bail, DecodingError};
use crate::{
    checked_decode_usize_product2, try_resize_decode_elements, HtCodeBlockDecoder, J2kIdwtBand,
    J2kRect, J2kSingleDecompositionIdwtJob, J2kWaveletTransform, Result,
};

pub(super) fn apply_level(
    input: IDWTInput<'_>,
    target: &mut Vec<f32>,
    decomposition: &Decomposition,
    transform: WaveletTransform,
    storage: &DecompositionStorage<'_>,
    backend: &mut Option<&mut dyn HtCodeBlockDecoder>,
) -> Result<IDWTTempOutput> {
    let handled = if let Some(backend) = backend.as_deref_mut() {
        let required_len = checked_decode_usize_product2(
            decomposition.rect.width() as usize,
            decomposition.rect.height() as usize,
        )?;
        try_resize_decode_elements(target, required_len, 0.0)?;
        let job = single_decomposition_job(input, decomposition, storage, transform);
        backend
            .decode_single_decomposition_idwt(job, target)
            .map_err(|_| DecodingError::CodeBlockDecodeFailure)?
    } else {
        false
    };

    if handled {
        Ok(IDWTTempOutput {
            rect: decomposition.rect,
        })
    } else {
        filter_2d(input, target, decomposition, transform, storage)
    }
}

pub(crate) fn apply_single_decomposition_idwt_job(
    job: J2kSingleDecompositionIdwtJob<'_>,
    target: &mut Vec<f32>,
) -> Result<()> {
    let rect = int_rect_from_public(job.rect);
    validate_direct_band(job.ll)?;
    validate_direct_band(job.hl)?;
    validate_direct_band(job.lh)?;
    validate_direct_band(job.hh)?;

    target.clear();
    let required_len =
        checked_decode_usize_product2(rect.width() as usize, rect.height() as usize)?;
    try_resize_decode_elements(target, required_len, 0.0)?;

    interleave_samples_roi(
        direct_coefficient_source(job.ll),
        direct_coefficient_source(job.hl),
        direct_coefficient_source(job.lh),
        direct_coefficient_source(job.hh),
        target,
        rect,
        rect,
    );
    if rect.width() > 0 && rect.height() > 0 {
        let transform = wavelet_transform_from_public(job.transform);
        filter_horizontal(target, rect, transform);
        filter_vertical(target, rect, transform);
    }
    Ok(())
}

fn validate_direct_band(band: J2kIdwtBand<'_>) -> Result<()> {
    let rect = int_rect_from_public(band.rect);
    let required_len = rect
        .width()
        .checked_mul(rect.height())
        .and_then(|len| usize::try_from(len).ok())
        .ok_or(DecodingError::CodeBlockDecodeFailure)?;
    if band.coefficients.len() < required_len {
        bail!(DecodingError::CodeBlockDecodeFailure);
    }
    Ok(())
}

fn direct_coefficient_source(band: J2kIdwtBand<'_>) -> CoefficientSource<'_> {
    let rect = int_rect_from_public(band.rect);
    CoefficientSource::new(band.coefficients, rect, rect.width())
}

fn int_rect_from_public(rect: J2kRect) -> IntRect {
    IntRect::from_ltrb(rect.x0, rect.y0, rect.x1, rect.y1)
}

fn wavelet_transform_from_public(transform: J2kWaveletTransform) -> WaveletTransform {
    match transform {
        J2kWaveletTransform::Reversible53 => WaveletTransform::Reversible53,
        J2kWaveletTransform::Irreversible97 => WaveletTransform::Irreversible97,
    }
}

fn single_decomposition_job<'a>(
    input: IDWTInput<'a>,
    decomposition: &'a Decomposition,
    storage: &'a DecompositionStorage<'a>,
    transform: WaveletTransform,
) -> J2kSingleDecompositionIdwtJob<'a> {
    let hl = &storage.sub_bands[decomposition.sub_bands[0]];
    let lh = &storage.sub_bands[decomposition.sub_bands[1]];
    let hh = &storage.sub_bands[decomposition.sub_bands[2]];
    J2kSingleDecompositionIdwtJob {
        rect: J2kRect::from(decomposition.rect),
        transform: J2kWaveletTransform::from(transform),
        ll: J2kIdwtBand {
            rect: J2kRect::from(input.rect),
            coefficients: input.coefficients,
        },
        hl: J2kIdwtBand {
            rect: J2kRect::from(hl.rect),
            coefficients: &storage.coefficients[hl.coefficients.clone()],
        },
        lh: J2kIdwtBand {
            rect: J2kRect::from(lh.rect),
            coefficients: &storage.coefficients[lh.coefficients.clone()],
        },
        hh: J2kIdwtBand {
            rect: J2kRect::from(hh.rect),
            coefficients: &storage.coefficients[hh.coefficients.clone()],
        },
    }
}

/// The `2D_SR` procedure illustrated in Figure F.6.
fn filter_2d(
    // The LL sub band of the given decomposition level.
    input: IDWTInput<'_>,
    coefficients: &mut Vec<f32>,
    decomposition: &Decomposition,
    transform: WaveletTransform,
    storage: &DecompositionStorage<'_>,
) -> Result<IDWTTempOutput> {
    // First interleave all sub-bands into a single buffer.
    interleave_samples(input, decomposition, coefficients, storage)?;

    if decomposition.rect.width() > 0 && decomposition.rect.height() > 0 {
        filter_horizontal(coefficients, decomposition.rect, transform);
        filter_vertical(coefficients, decomposition.rect, transform);
    }

    Ok(IDWTTempOutput {
        rect: decomposition.rect,
    })
}
