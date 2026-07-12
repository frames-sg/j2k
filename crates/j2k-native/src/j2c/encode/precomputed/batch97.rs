// SPDX-License-Identifier: MIT OR Apache-2.0

//! Aggregate retained-owner orchestration for precomputed 9/7 batches.

use super::allocation::precomputed_97_images_retained_bytes;
use super::orchestrator::Prepared97PacketPlan;
use super::{EncodeOptions, Vec};
use super::{
    J2kEncodeStageAccelerator, NativeEncodePipelineError, NativeEncodeRetainedInput,
    NativeEncodeSession, PrecomputedHtj2k97Image,
};

mod finalize;
mod prepare;
use self::prepare::{encode_prepared_batch, prepare_batch_plans};

/// Encode multiple borrowed precomputed 9/7 images while sharing one Tier-1
/// batch across all images.
#[doc(hidden)]
pub fn encode_precomputed_htj2k_97_batch_with_accelerator(
    images: &[PrecomputedHtj2k97Image],
    options: &EncodeOptions,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> crate::EncodeResult<Vec<Vec<u8>>> {
    if images.is_empty() {
        return Ok(Vec::new());
    }
    let retained_bytes = precomputed_97_images_retained_bytes(images, images.len())?;
    let session = NativeEncodeSession::try_new(NativeEncodeRetainedInput::from_owner_bytes(
        images,
        retained_bytes,
    ))?;
    prepare_batch_plans(images, options, &session)
        .and_then(|plans| encode_prepared_batch(plans, &session, accelerator))
        .map_err(NativeEncodePipelineError::into_encode_error)
}

/// Owned batch adapter used by transcode paths so the coefficient-image graph
/// can be released after packet preparation and before Tier-1/output growth.
#[doc(hidden)]
pub fn encode_precomputed_htj2k_97_batch_owned_with_accelerator(
    images: Vec<PrecomputedHtj2k97Image>,
    options: &EncodeOptions,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> crate::EncodeResult<Vec<Vec<u8>>> {
    encode_precomputed_htj2k_97_batch_owned_with_accelerator_and_cap(
        images,
        options,
        accelerator,
        crate::DEFAULT_MAX_CODEC_BYTES,
    )
}

pub(super) fn encode_precomputed_htj2k_97_batch_owned_with_accelerator_and_cap(
    images: Vec<PrecomputedHtj2k97Image>,
    options: &EncodeOptions,
    accelerator: &mut impl J2kEncodeStageAccelerator,
    max_host_bytes: usize,
) -> crate::EncodeResult<Vec<Vec<u8>>> {
    if images.is_empty() {
        return Ok(Vec::new());
    }
    let plans = prepare_owned_batch_plans(images, options, max_host_bytes)
        .map_err(NativeEncodePipelineError::into_encode_error)?;
    let session = NativeEncodeSession::try_with_lowered_cap(
        NativeEncodeRetainedInput::none(),
        max_host_bytes,
    )?;
    encode_prepared_batch(plans, &session, accelerator)
        .map_err(NativeEncodePipelineError::into_encode_error)
}

fn prepare_owned_batch_plans(
    images: Vec<PrecomputedHtj2k97Image>,
    options: &EncodeOptions,
    max_host_bytes: usize,
) -> super::NativeEncodePipelineResult<Vec<Prepared97PacketPlan>> {
    let retained_bytes = precomputed_97_images_retained_bytes(&images, images.capacity())?;
    let plans = {
        let input_session = NativeEncodeSession::try_with_lowered_cap(
            NativeEncodeRetainedInput::from_owner_bytes(&images, retained_bytes),
            max_host_bytes,
        )?;
        prepare_batch_plans(&images, options, &input_session)?
    };
    drop(images);
    Ok(plans)
}

#[cfg(test)]
#[path = "batch97/tests.rs"]
mod tests;
