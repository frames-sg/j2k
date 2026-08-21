// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    context::CudaContext,
    error::CudaError,
    j2k_decode::types::CudaJ2kIdwtTarget,
    memory::{CheckedDeviceBufferRanges, CudaBufferPool, CudaDeviceBuffer},
};

fn idwt_bands_belong_to_context<'a>(
    context: &CudaContext,
    bands: impl IntoIterator<Item = &'a CudaDeviceBuffer>,
) -> bool {
    bands.into_iter().all(|buffer| buffer.is_owned_by(context))
}

pub(super) fn validate_idwt_enqueue_context(
    context: &CudaContext,
    targets: &[CudaJ2kIdwtTarget<'_>],
    pool: &CudaBufferPool,
) -> Result<(), CudaError> {
    let buffers_match = targets.iter().all(|target| {
        idwt_bands_belong_to_context(
            context,
            [target.ll, target.hl, target.lh, target.hh, target.output],
        )
    });
    if !pool.is_owned_by(context) || !buffers_match {
        return Err(CudaError::InvalidArgument {
            message: "queued IDWT buffers and pool must belong to the launch context".to_string(),
        });
    }

    let outputs = CheckedDeviceBufferRanges::from_same_context(
        context,
        targets
            .iter()
            .enumerate()
            .map(|(index, target)| (index, target.output)),
    )?;
    let inputs = CheckedDeviceBufferRanges::from_same_context(
        context,
        targets.iter().enumerate().flat_map(|(index, target)| {
            [target.ll, target.hl, target.lh, target.hh]
                .into_iter()
                .map(move |input| (index, input))
        }),
    )?;
    if let Some((target_index, _)) = outputs.first_cross_overlap(&inputs)? {
        return Err(CudaError::InvalidArgument {
            message: format!(
                "queued IDWT output {target_index} overlaps a concurrently read input"
            ),
        });
    }
    if outputs.first_self_overlap().is_some() {
        return Err(CudaError::InvalidArgument {
            message: "queued IDWT outputs must be pairwise disjoint".to_string(),
        });
    }
    Ok(())
}

pub(super) fn validate_idwt_sequence_enqueue_context(
    context: &CudaContext,
    target_batches: &[&[CudaJ2kIdwtTarget<'_>]],
    pool: &CudaBufferPool,
) -> Result<(), CudaError> {
    if target_batches.is_empty() {
        return validate_idwt_enqueue_context(context, &[], pool);
    }
    for targets in target_batches {
        validate_idwt_enqueue_context(context, targets, pool)?;
    }
    Ok(())
}

pub(super) fn idwt_inputs_belong_to_context(
    context: &CudaContext,
    bands: [&CudaDeviceBuffer; 4],
) -> bool {
    idwt_bands_belong_to_context(context, bands)
}
