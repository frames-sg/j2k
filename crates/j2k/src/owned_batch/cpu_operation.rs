// SPDX-License-Identifier: MIT OR Apache-2.0

//! Exact native operations over immutable owned source preparation.

use alloc::sync::{Arc, Weak};
use j2k_core::{BatchInfrastructureError, CacheStats, Rect, Unsupported};

use super::prepare::image::{
    batch_group_info, choose_cpu_execution, prepare_classic_offset_plan, prepare_htj2k_offset_plan,
    reconcile_codec_plan_metadata,
};
use super::{
    decode_cpu_group, BatchCodecRoute, BatchDecodeOptions, BatchErrorStage, BatchGroupInfo,
    BatchItemError, CpuBatchDecodeResult, CpuBatchDecoder, CpuExecution, IndexedBatchError,
    PreparedBatchGroupRef, PreparedClassicPlan, PreparedCodecPlan, PreparedHtj2kPlan,
    PreparedImage, PreparedImageInner,
};
use crate::batch::allocation::try_vec_with_capacity;
use crate::{J2kDecodeControl, J2kError};

mod device;
pub use self::device::{
    prepare_device_operations, PreparedDeviceOperation, PreparedDeviceOperationGroup,
    PreparedDeviceOperations,
};

/// One source-coordinate region and exact JPEG 2000 reduction of a prepared image.
///
/// The source's validation settings and native sample interpretation are preserved.
/// Output layout, admission, and concurrency use the receiving CPU session.
/// The caller-owned source preparation and compressed bytes are outside an
/// operation's working limit; derived plans and execution owners are counted.
#[derive(Debug, Clone, Copy)]
pub struct PreparedImageOperation<'a> {
    image: &'a PreparedImage,
    roi: Rect,
    reduction_levels: u8,
    control: Option<&'a J2kDecodeControl>,
}

impl<'a> PreparedImageOperation<'a> {
    /// Request `roi` relative to the original full-resolution image, discarding
    /// exactly `reduction_levels` wavelet resolution levels.
    #[must_use]
    pub const fn new(image: &'a PreparedImage, roi: Rect, reduction_levels: u8) -> Self {
        Self {
            image,
            roi,
            reduction_levels,
            control: None,
        }
    }

    /// Apply limits and cancellation to this operation only.
    #[must_use]
    pub const fn with_control(mut self, control: &'a J2kDecodeControl) -> Self {
        self.control = Some(control);
        self
    }
}

#[derive(Debug)]
struct CpuOperationPreparation {
    info: BatchGroupInfo,
    output_rect: Rect,
    reduction_levels: u8,
    codec_plan: PreparedCodecPlan,
    region_streams_rows: bool,
    used_lenient_metadata_recovery: bool,
    retained_bytes: usize,
}

struct CpuOperationGroup {
    info: BatchGroupInfo,
    images: Vec<CpuPreparedImage>,
    source_indices: Vec<usize>,
}

/// The common CPU executor's private view. A deep operation never fabricates
/// a legacy `DeviceDecodePlan`, whose scale is limited to three levels.
#[derive(Debug, Clone)]
pub(super) struct CpuPreparedImage {
    source: Arc<PreparedImageInner>,
    operation: Option<Arc<CpuOperationPreparation>>,
    execution: CpuExecution,
    control: Option<J2kDecodeControl>,
    retained_bytes: usize,
}

impl CpuPreparedImage {
    pub(super) fn legacy(
        image: &PreparedImage,
        info: &BatchGroupInfo,
        options: BatchDecodeOptions,
    ) -> Self {
        Self {
            source: Arc::clone(&image.inner),
            operation: None,
            execution: super::prepare::cpu_execution_for(image, info, options),
            control: None,
            retained_bytes: 0,
        }
    }

    pub(super) fn bytes(&self) -> &Arc<[u8]> {
        &self.source.bytes
    }
    pub(super) fn output_rect(&self) -> Rect {
        self.operation.as_ref().map_or_else(
            || self.source.plan.output_rect(),
            |operation| operation.output_rect,
        )
    }
    #[expect(
        clippy::cast_possible_truncation,
        reason = "u32::trailing_zeros returns at most 32, which fits u8"
    )]
    pub(super) fn reduction_levels(&self) -> u8 {
        self.operation.as_ref().map_or_else(
            || self.source.plan.scale().denominator().trailing_zeros() as u8,
            |operation| operation.reduction_levels,
        )
    }
    pub(super) fn settings(&self) -> crate::DecodeSettings {
        self.source.decode_settings
    }
    pub(super) fn control(&self) -> Option<&J2kDecodeControl> {
        self.control.as_ref()
    }
    pub(super) fn retained_bytes(&self) -> usize {
        self.retained_bytes
    }
    pub(super) fn check_owned_output(&self, output_bytes: usize) -> Result<(), J2kError> {
        let Some(control) = self.control() else {
            return Ok(());
        };
        control.check_cancelled()?;
        let cap = control
            .max_working_memory_bytes
            .map_or(
                j2k_native::DEFAULT_MAX_DECODE_BYTES,
                core::num::NonZeroUsize::get,
            )
            .min(j2k_native::DEFAULT_MAX_DECODE_BYTES);
        let requested = self.retained_bytes.saturating_add(output_bytes);
        if requested > cap {
            return Err(J2kError::from_native_decode_error(
                j2k_native::DecodeError::AllocationTooLarge {
                    what: "owned operation output and retained plans",
                    requested,
                    cap,
                },
            ));
        }
        Ok(())
    }
    pub(super) fn codec_plan(&self) -> &PreparedCodecPlan {
        self.operation
            .as_ref()
            .map_or(&self.source.codec_plan, |operation| &operation.codec_plan)
    }
    pub(super) const fn cpu_execution(&self) -> CpuExecution {
        self.execution
    }
    pub(super) fn classic_plan(&self) -> Option<&PreparedClassicPlan> {
        match self.codec_plan() {
            PreparedCodecPlan::Classic(plan) => Some(plan),
            _ => None,
        }
    }
    pub(super) fn htj2k_plan(&self) -> Option<&PreparedHtj2kPlan> {
        match self.codec_plan() {
            PreparedCodecPlan::Htj2k(plan) => Some(plan),
            _ => None,
        }
    }
    pub(super) fn used_lenient_metadata_recovery(&self) -> bool {
        self.operation
            .as_ref()
            .map_or(self.source.used_lenient_metadata_recovery, |operation| {
                operation.used_lenient_metadata_recovery
            })
    }
}

struct CachedOperation {
    source: Weak<PreparedImageInner>,
    roi: Rect,
    reduction_levels: u8,
    preparation: Arc<CpuOperationPreparation>,
}

pub(super) struct CpuOperationCache {
    entry: Option<CachedOperation>,
    limit: usize,
    stats: CacheStats,
}

impl Default for CpuOperationCache {
    fn default() -> Self {
        Self {
            entry: None,
            limit: j2k_native::DEFAULT_MAX_DECODE_BYTES,
            stats: CacheStats::default(),
        }
    }
}

impl CpuOperationCache {
    fn clear(&mut self) {
        if self.entry.take().is_some() {
            self.stats.evictions = self.stats.evictions.saturating_add(1);
        }
    }

    fn prepare(
        &mut self,
        request: PreparedImageOperation<'_>,
        options: BatchDecodeOptions,
        worker: &mut crate::batch::worker::BatchWorker,
        live_bytes: usize,
    ) -> Result<Arc<CpuOperationPreparation>, BatchItemError> {
        if let Some(control) = request.control {
            control.check_cancelled().map_err(preparation_error)?;
        }
        // Controlled execution does not use offset plans. Release the optional
        // cache instead of making its admission depend on an earlier call.
        if let Some(entry) = self.entry.as_ref().filter(|_| request.control.is_none()) {
            if entry.source.ptr_eq(&Arc::downgrade(&request.image.inner))
                && entry.roi == request.roi
                && entry.reduction_levels == request.reduction_levels
            {
                self.stats.hits = self.stats.hits.saturating_add(1);
                return Ok(Arc::clone(&entry.preparation));
            }
        }
        self.clear();
        self.stats.misses = self.stats.misses.saturating_add(1);
        if request.control.is_some() {
            worker.release_owned_allocations();
        }
        let preparation =
            prepare_operation(request, options, worker, live_bytes, OperationPlans::Cpu)?;
        let control = request.control.cloned().unwrap_or_default();
        let owners = live_bytes
            .saturating_add(preparation.retained_bytes)
            .saturating_add(
                worker
                    .retained_native_workspace_bytes()
                    .map_err(|error| preparation_error(control.native_error(error)))?,
            );
        control
            .to_native()
            .beside_retained_bytes(owners)
            .map_err(|error| preparation_error(control.native_error(error)))?;
        let preparation = Arc::new(preparation);
        // A controlled miss needs only ephemeral native geometry. Do not cache
        // its deliberately unbuilt offset plan as a reusable capability fact.
        if request.control.is_none() && preparation.retained_bytes <= self.limit && self.limit != 0
        {
            self.entry = Some(CachedOperation {
                source: Arc::downgrade(&request.image.inner),
                roi: request.roi,
                reduction_levels: request.reduction_levels,
                preparation: Arc::clone(&preparation),
            });
        }
        Ok(preparation)
    }
}

/// Whether an operation with `info` joins an output group of `len` images
/// sharing `group_info`: identical output contracts, with the dense group
/// output still within one host allocation. Every backend groups by this rule.
fn joins_operation_group(group_info: &BatchGroupInfo, len: usize, info: &BatchGroupInfo) -> bool {
    let per_image = info.output_bytes(1).unwrap_or(usize::MAX);
    group_info == info
        && per_image.saturating_mul(len.saturating_add(1))
            <= j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES
}

fn preparation_error(source: J2kError) -> BatchItemError {
    BatchItemError::Codec {
        stage: BatchErrorStage::Prepare,
        source: Arc::new(source),
    }
}

fn unsupported(what: &'static str) -> BatchItemError {
    preparation_error(J2kError::Unsupported(Unsupported { what }))
}

/// Which offset plans an operation's preparation builds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OperationPlans {
    /// Controlled CPU operations decode from native geometry alone; others
    /// build an offset plan when the source supports one.
    Cpu,
    /// Device routes have no native geometry fallback: always build the plan,
    /// under the operation's control when it has one.
    Device,
}

fn prepare_operation(
    request: PreparedImageOperation<'_>,
    mut options: BatchDecodeOptions,
    worker: &mut crate::batch::worker::BatchWorker,
    live_bytes: usize,
    plans: OperationPlans,
) -> Result<CpuOperationPreparation, BatchItemError> {
    let source = &request.image.inner;
    options.settings = source.decode_settings;
    let output = source
        .reference_grid
        .reduced_region(request.roi, request.reduction_levels)
        .ok_or_else(|| unsupported("invalid source region or unrepresentable reduction"))?;
    if output.w == 0 || output.h == 0 {
        return Err(unsupported(
            "region has no samples at the requested reduction",
        ));
    }
    let mut info = batch_group_info(&source.support, (output.w, output.h), options)?;
    if let Some(error) = super::prepare::cpu_preflight_error(CpuExecution::ComponentPlanes, &info) {
        return Err(error);
    }
    if let Some(control) = request.control {
        control
            .check_output(info.output_bytes(1).unwrap_or(usize::MAX))
            .map_err(preparation_error)?;
    }
    let image = crate::backend::image_for_operation(
        &source.bytes,
        source.decode_settings,
        request.reduction_levels,
        live_bytes,
        request.control,
    )
    .map_err(preparation_error)?;
    let expected = source
        .reference_grid
        .reduced_region(
            Rect::full(source.support.info.dimensions),
            request.reduction_levels,
        )
        .ok_or_else(|| unsupported("unrepresentable reduction"))?;
    if (image.width(), image.height()) != (expected.w, expected.h) {
        return Err(unsupported(
            "native backend did not honor the requested reduction level",
        ));
    }
    let codec_plan = if let Some(control) = request.control.filter(|_| plans == OperationPlans::Cpu)
    {
        if request.reduction_levels
            > image
                .max_reduction_levels_with_operation_control(&control.to_native())
                .map_err(|error| control.native_error(error))
                .map_err(preparation_error)?
        {
            return Err(unsupported(
                "requested reduction exceeds a component or tile resolution ladder",
            ));
        }
        PreparedCodecPlan::MetadataOnly
    } else {
        match info.route {
            BatchCodecRoute::Classic => prepare_classic_offset_plan(
                &image,
                &info,
                output,
                worker,
                live_bytes,
                request.control,
            )?
            .map_or(PreparedCodecPlan::MetadataOnly, PreparedCodecPlan::Classic),
            BatchCodecRoute::Htj2k => prepare_htj2k_offset_plan(
                &image,
                &info,
                output,
                worker,
                live_bytes,
                request.control,
            )?
            .map_or(PreparedCodecPlan::MetadataOnly, PreparedCodecPlan::Htj2k),
        }
    };
    reconcile_codec_plan_metadata(&mut info, &codec_plan)?;
    let retained_bytes = codec_plan
        .retained_allocation_bytes()
        .and_then(|bytes| {
            bytes
                .checked_add(super::prepared_plan::arc_owner_bytes::<
                    CpuOperationPreparation,
                >())
                .ok_or(j2k_types::DecodePlanAllocationError)
        })
        .map_err(|_| preparation_error(CpuExecution::decode_bound_error()))?;
    Ok(CpuOperationPreparation {
        info,
        output_rect: output,
        reduction_levels: request.reduction_levels,
        codec_plan,
        region_streams_rows: image.region_decode_streams_rows(),
        used_lenient_metadata_recovery: image.used_lenient_metadata_recovery(),
        retained_bytes,
    })
}

/// What one exact operation decodes to, planned before any sample is decoded.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct PreparedOperationPlan {
    /// The output contract of the group [`CpuBatchDecoder::decode_operations`]
    /// returns for the operation: reduced region dimensions, colour channels,
    /// alpha, native sample type and precision.
    pub info: BatchGroupInfo,
    /// Covered rectangle relative to the reduced image's origin.
    pub output_rect: Rect,
    /// Whether preparation applied a lenient JP2/JPH metadata recovery.
    pub used_lenient_metadata_recovery: bool,
}

impl CpuBatchDecoder {
    /// Plan exact operations as [`Self::decode_operations`] prepares them,
    /// without decoding: each operation's output contract, or the preparation
    /// error its decode would report. Results follow `operations`' order.
    ///
    /// Each operation is planned under its own control, with no sibling plans
    /// live. Planning goes through the session's exact-operation cache, so an
    /// uncontrolled operation decoded next can reuse its preparation.
    ///
    /// # Errors
    /// Returns [`BatchInfrastructureError`] when the session's workers are
    /// poisoned or the result list cannot be allocated.
    pub fn plan_operations(
        &mut self,
        operations: &[PreparedImageOperation<'_>],
    ) -> Result<Vec<Result<PreparedOperationPlan, BatchItemError>>, BatchInfrastructureError> {
        let workers = self
            .workers
            .get_mut()
            .map_err(|_| BatchInfrastructureError::SchedulerPoisoned)?;
        let mut plans = try_vec_with_capacity(operations.len(), "J2K operation plans")?;
        for &operation in operations {
            let plan = self
                .operation_cache
                .prepare(operation, self.options, &mut workers[0], 0)
                .map(|preparation| PreparedOperationPlan {
                    info: preparation.info.clone(),
                    output_rect: preparation.output_rect,
                    used_lenient_metadata_recovery: preparation.used_lenient_metadata_recovery,
                });
            plans.push(plan);
        }
        Ok(plans)
    }

    /// Limit derived exact-operation plan capacities retained between calls.
    /// Zero disables the one-entry cache. Source input remains caller-owned;
    /// execution workspace and in-flight operation plans are separate.
    pub fn set_operation_cache_limit_bytes(&mut self, bytes: usize) {
        self.operation_cache.limit = bytes;
        if self.operation_cache_retained_bytes() > bytes || bytes == 0 {
            self.operation_cache.clear();
        }
    }

    /// Configured exact-operation plan retention limit.
    #[must_use]
    pub const fn operation_cache_limit_bytes(&self) -> usize {
        self.operation_cache.limit
    }

    /// Retained allocator capacities, counting shared plan owners only once.
    /// Includes the preparation and plan Arc records and their vector storage.
    #[must_use]
    pub fn operation_cache_retained_bytes(&self) -> usize {
        self.operation_cache
            .entry
            .as_ref()
            .map_or(0, |entry| entry.preparation.retained_bytes)
    }

    /// Derived operation cache lookups and evictions.
    #[must_use]
    pub fn operation_cache_stats(&self) -> CacheStats {
        CacheStats {
            occupied_slots: u64::from(self.operation_cache.entry.is_some()),
            ..self.operation_cache.stats
        }
    }

    /// Decode regions at exact wavelet reductions from existing prepared sources.
    ///
    /// Samples retain their native integer values and widths. Returned source
    /// indices refer to positions in `operations`, independent of the sources'
    /// original preparation indices. Each session owns its mutable decode scratch.
    /// Controlled operations use individual output groups: their working limit
    /// includes live derived plans and that operation's owned output, excluding
    /// caller-owned source preparation and completed sibling outputs. The ordinary session bound still controls
    /// aggregate planned concurrency.
    #[expect(
        clippy::too_many_lines,
        reason = "preparation, unique live-plan accounting, and execution remain in one ownership scope so every charged owner stays alive"
    )]
    pub fn decode_operations(
        &mut self,
        operations: &[PreparedImageOperation<'_>],
    ) -> Result<CpuBatchDecodeResult, BatchInfrastructureError> {
        // The all-distinct case holds one preparation and result group per
        // operation. Preserve the existing facade's bounded metadata domain.
        let item_bytes = core::mem::size_of::<CpuOperationGroup>()
            + core::mem::size_of::<CpuPreparedImage>()
            + core::mem::size_of::<Arc<CpuOperationPreparation>>()
            + core::mem::size_of::<IndexedBatchError>()
            + core::mem::size_of::<super::CpuBatchGroup>()
            + core::mem::size_of::<Rect>()
            + core::mem::size_of::<Vec<crate::J2kDecodeWarning>>()
            + 2 * core::mem::size_of::<usize>();
        let metadata_bytes = operations.len().saturating_mul(item_bytes);
        if metadata_bytes > super::J2K_BATCH_METADATA_ALLOWANCE_BYTES {
            return Err(BatchInfrastructureError::AllocationTooLarge {
                what: "J2K operation metadata",
                requested: metadata_bytes,
                cap: super::J2K_BATCH_METADATA_ALLOWANCE_BYTES,
            });
        }
        let workers = self
            .workers
            .get_mut()
            .map_err(|_| BatchInfrastructureError::SchedulerPoisoned)?;
        let stream_pool = self
            .stream_pool
            .get_mut()
            .map_err(|_| BatchInfrastructureError::SchedulerPoisoned)?;
        if operations
            .iter()
            .any(|operation| operation.control.is_some())
        {
            for worker in workers.iter_mut() {
                worker.release_owned_allocations();
            }
            self.fast_workspace.release_allocations();
            self.staged_workspace = super::CpuStagedWorkspace::default();
        }
        let mut jobs: Vec<CpuOperationGroup> =
            try_vec_with_capacity(operations.len(), "J2K operation groups")?;
        let mut errors = try_vec_with_capacity(operations.len(), "J2K operation errors")?;
        let mut live_plans: Vec<Arc<CpuOperationPreparation>> =
            try_vec_with_capacity(operations.len(), "J2K live operation plans")?;
        let mut live_bytes = 0usize;
        for (index, &operation) in operations.iter().enumerate() {
            let result =
                self.operation_cache
                    .prepare(operation, self.options, &mut workers[0], live_bytes);
            let preparation = match result {
                Ok(preparation) => preparation,
                Err(source) => {
                    errors.push(IndexedBatchError { index, source });
                    continue;
                }
            };
            // A hit can only return the one cache's most recent preparation.
            // Misses replace that entry, so repeated operations compare once.
            if !live_plans
                .last()
                .is_some_and(|plan| Arc::ptr_eq(plan, &preparation))
            {
                live_bytes = live_bytes.checked_add(preparation.retained_bytes).ok_or(
                    BatchInfrastructureError::AllocationTooLarge {
                        what: "J2K live operation plans",
                        requested: usize::MAX,
                        cap: j2k_native::DEFAULT_MAX_DECODE_BYTES,
                    },
                )?;
                live_plans.push(Arc::clone(&preparation));
            }
            let execution = choose_cpu_execution(
                &preparation.codec_plan,
                &preparation.info,
                preparation.region_streams_rows,
                self.options
                    .planned_decode_bound(j2k_native::DEFAULT_MAX_DECODE_BYTES),
            );
            if let Some(source) = super::prepare::cpu_preflight_error(execution, &preparation.info)
            {
                errors.push(IndexedBatchError { index, source });
                continue;
            }
            let image = CpuPreparedImage {
                source: Arc::clone(&operation.image.inner),
                operation: Some(Arc::clone(&preparation)),
                execution,
                control: operation.control.cloned(),
                retained_bytes: 0,
            };
            if let Some(group) = jobs.iter_mut().find(|group| {
                operation.control.is_none()
                    && group.images[0].control().is_none()
                    && joins_operation_group(&group.info, group.images.len(), &preparation.info)
            }) {
                super::prepare::try_reserve_one(&mut group.images, "J2K operation images")?;
                super::prepare::try_reserve_one(
                    &mut group.source_indices,
                    "J2K operation indices",
                )?;
                group.images.push(image);
                group.source_indices.push(index);
            } else {
                let mut images = try_vec_with_capacity(1, "J2K operation images")?;
                let mut indices = try_vec_with_capacity(1, "J2K operation indices")?;
                images.push(image);
                indices.push(index);
                jobs.push(CpuOperationGroup {
                    info: preparation.info.clone(),
                    images,
                    source_indices: indices,
                });
            }
        }
        let mut groups = try_vec_with_capacity(jobs.len(), "J2K operation results")?;
        for job in &mut jobs {
            for image in &mut job.images {
                image.retained_bytes = live_bytes;
            }
            let group = PreparedBatchGroupRef {
                info: &job.info,
                images: &job.images,
                source_indices: &job.source_indices,
                working_memory_limit: self.options.max_working_memory_bytes,
            };
            let held = self.options.scheduled_workers(workers.len());
            if let Some(group) = decode_cpu_group(
                workers,
                &mut self.fast_workspace,
                &mut self.staged_workspace,
                stream_pool,
                group,
                self.options,
                held,
                &mut errors,
            )? {
                groups.push(group);
            }
        }
        errors.sort_by_key(|error| error.index);
        Ok(CpuBatchDecodeResult { groups, errors })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn controlled_pixels_include_other_live_plans_and_the_owned_output() {
        let pixels: Vec<u8> = (0..=u8::MAX).cycle().take(256 * 256).collect();
        let bytes = j2k_native::encode_htj2k(
            &pixels,
            256,
            256,
            1,
            8,
            false,
            &j2k_native::EncodeOptions {
                reversible: true,
                num_decomposition_levels: 2,
                ..Default::default()
            },
        )
        .unwrap();
        let prepared = super::super::prepare_batch(
            vec![super::super::EncodedImage::full(bytes.into())],
            BatchDecodeOptions::default(),
        )
        .unwrap();
        let source = &prepared.groups()[0].images()[0];
        let info = prepared.groups()[0].info();
        let mut image = CpuPreparedImage::legacy(source, info, BatchDecodeOptions::default());
        image.control = Some(J2kDecodeControl::default());
        // Stand in for other in-flight immutable plans without huge allocations.
        image.retained_bytes = j2k_native::DEFAULT_MAX_DECODE_BYTES - 256 * 1024;
        let output_bytes = info.output_bytes(1).unwrap();
        image.check_owned_output(output_bytes).unwrap();
        crate::backend::image_for_operation(
            image.bytes(),
            image.settings(),
            0,
            image.retained_bytes + output_bytes,
            image.control(),
        )
        .expect("parsing and final output both fit");
        let mut session = CpuBatchDecoder::new(BatchDecodeOptions::default());
        let mut errors = Vec::new();
        let group = PreparedBatchGroupRef {
            info,
            images: core::slice::from_ref(&image),
            source_indices: &[0],
            working_memory_limit: None,
        };
        let result = decode_cpu_group(
            session.workers.get_mut().unwrap(),
            &mut session.fast_workspace,
            &mut session.staged_workspace,
            session.stream_pool.get_mut().unwrap(),
            group,
            session.options,
            core::num::NonZeroUsize::new(1),
            &mut errors,
        )
        .unwrap();
        assert!(
            result.is_none(),
            "pixel scratch must fit beside the retained plans and output"
        );
        assert!(
            matches!(&errors[0].source, BatchItemError::Codec { source, .. } if matches!(source.as_ref(), J2kError::ResourceLimit { .. }))
        );
        image.retained_bytes = 0;
        errors.clear();
        let group = PreparedBatchGroupRef {
            info,
            images: core::slice::from_ref(&image),
            source_indices: &[0],
            working_memory_limit: None,
        };
        let result = decode_cpu_group(
            session.workers.get_mut().unwrap(),
            &mut session.fast_workspace,
            &mut session.staged_workspace,
            session.stream_pool.get_mut().unwrap(),
            group,
            session.options,
            core::num::NonZeroUsize::new(1),
            &mut errors,
        )
        .unwrap()
        .unwrap();
        assert!(errors.is_empty(), "{errors:?}");
        assert_eq!(result.samples(), &super::super::CpuBatchSamples::U8(pixels));
    }
}
