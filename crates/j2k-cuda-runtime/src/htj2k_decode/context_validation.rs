// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    context::CudaContext,
    error::CudaError,
    memory::{CheckedDeviceBufferRanges, CudaBufferPool, CudaDeviceBuffer},
};

use super::{
    CudaHtj2kCleanupTarget, CudaHtj2kDecodePayload, CudaHtj2kDecodeResources,
    CudaHtj2kDecodeTableResources, CudaHtj2kDequantizeTarget,
};

impl CudaHtj2kDecodePayload {
    pub(super) fn is_owned_by(&self, context: &CudaContext) -> Result<bool, CudaError> {
        Ok(self.buffer()?.is_owned_by(context))
    }
}

impl CudaHtj2kDecodeTableResources {
    pub(super) fn is_owned_by(&self, context: &CudaContext) -> bool {
        self.inner.vlc_table0.is_owned_by(context)
            && self.inner.vlc_table1.is_owned_by(context)
            && self.inner.uvlc_table0.is_owned_by(context)
            && self.inner.uvlc_table1.is_owned_by(context)
    }
}

impl CudaHtj2kDecodeResources {
    pub(crate) fn is_owned_by(&self, context: &CudaContext) -> Result<bool, CudaError> {
        Ok(self.payload.is_owned_by(context)?
            && self
                .tables
                .as_ref()
                .is_none_or(|tables| tables.is_owned_by(context)))
    }
}

fn validate_target_allocations_disjoint<T>(
    context: &CudaContext,
    targets: &[T],
    buffer: impl for<'a> Fn(&'a T) -> &'a CudaDeviceBuffer,
    message: &'static str,
) -> Result<(), CudaError> {
    let ranges = CheckedDeviceBufferRanges::from_same_context(
        context,
        targets
            .iter()
            .enumerate()
            .map(|(index, target)| (index, buffer(target))),
    )?;
    if ranges.first_self_overlap().is_some() {
        return Err(CudaError::InvalidArgument {
            message: message.to_string(),
        });
    }
    Ok(())
}

pub(super) fn validate_cleanup_context(
    context: &CudaContext,
    resources: &CudaHtj2kDecodeResources,
    targets: &[CudaHtj2kCleanupTarget<'_>],
    pool: &CudaBufferPool,
) -> Result<(), CudaError> {
    let targets_match = targets
        .iter()
        .all(|target| target.coefficients.is_owned_by(context));
    if !pool.is_owned_by(context) || !resources.is_owned_by(context)? || !targets_match {
        return Err(CudaError::InvalidArgument {
            message: "HTJ2K cleanup resources, targets, and pool must belong to the launch context"
                .to_string(),
        });
    }
    validate_target_allocations_disjoint(
        context,
        targets,
        |target| target.coefficients,
        "HTJ2K cleanup target allocations must be pairwise disjoint",
    )
}

pub(super) fn validate_dequantize_context(
    context: &CudaContext,
    targets: &[CudaHtj2kDequantizeTarget<'_>],
    pool: &CudaBufferPool,
) -> Result<(), CudaError> {
    if !pool.is_owned_by(context)
        || !targets
            .iter()
            .all(|target| target.coefficients.is_owned_by(context))
    {
        return Err(CudaError::InvalidArgument {
            message: "HTJ2K dequantize targets and pool must belong to the launch context"
                .to_string(),
        });
    }
    validate_target_allocations_disjoint(
        context,
        targets,
        |target| target.coefficients,
        "HTJ2K dequantize target allocations must be pairwise disjoint",
    )
}
