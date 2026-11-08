// SPDX-License-Identifier: MIT OR Apache-2.0

//! Exact operations prepared for accelerators whose prepared routes consume
//! offset plans.

use alloc::sync::Arc;
use core::mem::size_of;

use j2k_core::{BatchInfrastructureError, Rect};

use super::{
    joins_operation_group, preparation_error, prepare_operation, unsupported,
    CpuOperationPreparation, OperationPlans, PreparedImageOperation,
};
use crate::batch::allocation::try_vec_with_capacity;
use crate::batch::worker::BatchWorker;
use crate::owned_batch::{
    prepare::try_reserve_one, BatchDecodeOptions, BatchGroupInfo, BatchItemError,
    IndexedBatchError, PreparedClassicPlan, PreparedCodecPlan, PreparedHtj2kPlan,
    PreparedImageInner, J2K_BATCH_METADATA_ALLOWANCE_BYTES,
};
use crate::{DecodeSettings, J2kCancellationToken, J2kDecodeControl, J2kError, J2kSupportInfo};

/// One exact operation prepared for an accelerator.
///
/// Output geometry, group contract and admission match
/// [`crate::CpuBatchDecoder::decode_operations`]. The offset plan is built
/// whenever the source supports one, because device routes have no native
/// geometry fallback.
#[doc(hidden)]
#[derive(Debug)]
pub struct PreparedDeviceOperation {
    source: Arc<PreparedImageInner>,
    roi: Rect,
    covers_reduced_image: bool,
    preparation: CpuOperationPreparation,
}

impl PreparedDeviceOperation {
    /// Compressed bytes of the prepared source, shared rather than copied.
    #[must_use]
    pub fn bytes(&self) -> &Arc<[u8]> {
        &self.source.bytes
    }

    /// Requested region of the full-resolution image.
    #[must_use]
    pub const fn source_region(&self) -> Rect {
        self.roi
    }

    /// Wavelet resolution levels the operation discards.
    #[must_use]
    pub const fn reduction_levels(&self) -> u8 {
        self.preparation.reduction_levels
    }

    /// Decoded rectangle in reduced-image coordinates.
    #[must_use]
    pub const fn output_rect(&self) -> Rect {
        self.preparation.output_rect
    }

    /// Whether the output is the whole reduced image rather than a region of it.
    #[must_use]
    pub const fn covers_reduced_image(&self) -> bool {
        self.covers_reduced_image
    }

    /// HTJ2K offset plan for this operation's region and reduction.
    #[must_use]
    pub const fn htj2k_plan(&self) -> Option<&PreparedHtj2kPlan> {
        match &self.preparation.codec_plan {
            PreparedCodecPlan::Htj2k(plan) => Some(plan),
            _ => None,
        }
    }

    /// Classic offset plan for this operation's region and reduction.
    #[must_use]
    pub const fn classic_plan(&self) -> Option<&PreparedClassicPlan> {
        match &self.preparation.codec_plan {
            PreparedCodecPlan::Classic(plan) => Some(plan),
            _ => None,
        }
    }

    /// Header-derived support facts of the prepared source.
    #[must_use]
    pub fn support(&self) -> &J2kSupportInfo {
        &self.source.support
    }

    /// Validation policy the source was prepared with.
    #[must_use]
    pub fn decode_settings(&self) -> DecodeSettings {
        self.source.decode_settings
    }

    /// Whether preparing this operation used lenient metadata recovery.
    #[must_use]
    pub const fn used_lenient_metadata_recovery(&self) -> bool {
        self.preparation.used_lenient_metadata_recovery
    }
}

/// Prepared exact operations sharing one output contract.
#[doc(hidden)]
#[derive(Debug)]
pub struct PreparedDeviceOperationGroup {
    info: BatchGroupInfo,
    operations: Vec<PreparedDeviceOperation>,
    source_indices: Vec<usize>,
    /// The one cancellation token every operation of the group carries.
    cancellation: Option<J2kCancellationToken>,
}

impl PreparedDeviceOperationGroup {
    /// Stop before submitting the group when its operations were cancelled.
    ///
    /// Operations share a group only when they share a cancellation token
    /// (or none), so cancelling one operation never fails an unrelated one.
    ///
    /// # Errors
    /// Returns [`J2kError::Cancelled`] once the group's token is cancelled.
    pub fn check_cancelled(&self) -> Result<(), J2kError> {
        if self
            .cancellation
            .as_ref()
            .is_some_and(J2kCancellationToken::is_cancelled)
        {
            return Err(J2kError::Cancelled);
        }
        Ok(())
    }

    /// Output contract shared by every operation in the group.
    #[must_use]
    pub const fn info(&self) -> &BatchGroupInfo {
        &self.info
    }

    /// Prepared operations in group order.
    #[must_use]
    pub fn operations(&self) -> &[PreparedDeviceOperation] {
        &self.operations
    }

    /// Positions of the group's operations in the submitted operation slice.
    #[must_use]
    pub fn source_indices(&self) -> &[usize] {
        &self.source_indices
    }
}

/// Grouped device preparations and the operations that failed to prepare.
#[doc(hidden)]
#[derive(Debug)]
pub struct PreparedDeviceOperations {
    groups: Vec<PreparedDeviceOperationGroup>,
    errors: Vec<IndexedBatchError>,
}

impl PreparedDeviceOperations {
    /// Groups of operations sharing an output contract.
    #[must_use]
    pub fn groups(&self) -> &[PreparedDeviceOperationGroup] {
        &self.groups
    }

    /// Operations that failed to prepare, by position in the submitted slice.
    #[must_use]
    pub fn errors(&self) -> &[IndexedBatchError] {
        &self.errors
    }
}

/// Prepare exact operations for an accelerator, grouped as
/// [`crate::CpuBatchDecoder::decode_operations`] groups them, and only with
/// operations sharing their cancellation token.
///
/// A [`J2kDecodeControl`]'s output limit and cancellation apply to each
/// operation's preparation; [`PreparedDeviceOperationGroup::check_cancelled`]
/// lets the device route stop before submitting a group. A working-memory
/// limit fails the operation as unsupported: device working memory is not
/// accounted against a host allowance.
///
/// # Errors
/// Returns [`BatchInfrastructureError`] when operation metadata cannot be
/// allocated. Per-operation failures are reported in
/// [`PreparedDeviceOperations::errors`].
#[doc(hidden)]
pub fn prepare_device_operations(
    operations: &[PreparedImageOperation<'_>],
    options: BatchDecodeOptions,
) -> Result<PreparedDeviceOperations, BatchInfrastructureError> {
    let item_bytes = size_of::<PreparedDeviceOperation>()
        + size_of::<PreparedDeviceOperationGroup>()
        + size_of::<IndexedBatchError>()
        + size_of::<usize>();
    let metadata_bytes = operations.len().saturating_mul(item_bytes);
    if metadata_bytes > J2K_BATCH_METADATA_ALLOWANCE_BYTES {
        return Err(BatchInfrastructureError::AllocationTooLarge {
            what: "J2K device operation metadata",
            requested: metadata_bytes,
            cap: J2K_BATCH_METADATA_ALLOWANCE_BYTES,
        });
    }
    let mut groups: Vec<PreparedDeviceOperationGroup> =
        try_vec_with_capacity(operations.len(), "J2K device operation groups")?;
    let mut errors = try_vec_with_capacity(operations.len(), "J2K device operation errors")?;
    let mut worker = BatchWorker::new_owned(2);
    let mut live_bytes = 0usize;
    for (index, &operation) in operations.iter().enumerate() {
        let prepared = match prepare_one(operation, options, &mut worker, live_bytes) {
            Ok(prepared) => prepared,
            Err(source) => {
                errors.push(IndexedBatchError { index, source });
                continue;
            }
        };
        live_bytes = live_bytes.saturating_add(prepared.preparation.retained_bytes);
        let cancellation = operation
            .control
            .and_then(|control| control.cancellation.as_ref());
        if let Some(group) = groups.iter_mut().find(|group| {
            group
                .cancellation
                .as_ref()
                .map(J2kCancellationToken::flag_ptr)
                == cancellation.map(J2kCancellationToken::flag_ptr)
                && joins_operation_group(
                    &group.info,
                    group.operations.len(),
                    &prepared.preparation.info,
                )
        }) {
            try_reserve_one(&mut group.operations, "J2K device operations")?;
            try_reserve_one(&mut group.source_indices, "J2K device operation indices")?;
            group.operations.push(prepared);
            group.source_indices.push(index);
        } else {
            let info = prepared.preparation.info.clone();
            let mut grouped = try_vec_with_capacity(1, "J2K device operations")?;
            let mut indices = try_vec_with_capacity(1, "J2K device operation indices")?;
            grouped.push(prepared);
            indices.push(index);
            groups.push(PreparedDeviceOperationGroup {
                info,
                operations: grouped,
                source_indices: indices,
                cancellation: cancellation.cloned(),
            });
        }
    }
    Ok(PreparedDeviceOperations { groups, errors })
}

fn prepare_one(
    operation: PreparedImageOperation<'_>,
    options: BatchDecodeOptions,
    worker: &mut BatchWorker,
    live_bytes: usize,
) -> Result<PreparedDeviceOperation, BatchItemError> {
    if let Some(control) = operation.control {
        if control.max_working_memory_bytes.is_some() {
            return Err(unsupported(
                "working-memory limits are not supported on accelerator exact-operation routes",
            ));
        }
        control.check_cancelled().map_err(preparation_error)?;
    }
    let preparation = prepare_operation(
        operation,
        options,
        worker,
        live_bytes,
        OperationPlans::Device,
    )?;
    if let Some(control) = operation.control {
        control.check_cancelled().map_err(preparation_error)?;
    }
    // Keep every live plan and the worker's retained native workspace within
    // the default decode budget, as the CPU operation cache does.
    let control = J2kDecodeControl::default();
    let workspace = worker
        .retained_native_workspace_bytes()
        .map_err(|error| preparation_error(control.native_error(error)))?;
    control
        .to_native()
        .beside_retained_bytes(
            live_bytes
                .saturating_add(preparation.retained_bytes)
                .saturating_add(workspace),
        )
        .map_err(|error| preparation_error(control.native_error(error)))?;
    let source = &operation.image.inner;
    let covers_reduced_image = source.reference_grid.reduced_region(
        Rect::full(source.support.info.dimensions),
        operation.reduction_levels,
    ) == Some(preparation.output_rect);
    Ok(PreparedDeviceOperation {
        source: Arc::clone(source),
        roi: operation.roi,
        covers_reduced_image,
        preparation,
    })
}
