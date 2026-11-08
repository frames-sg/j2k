// SPDX-License-Identifier: MIT OR Apache-2.0

//! Persistent CPU batch session and public native output owners.

use super::{
    decode_cpu_group, fmt, prepare_batch_from_images_for, prepare_batch_with_workers,
    BatchDecodeOptions, BatchDecoder, BatchGroupInfo, BatchInfrastructureError, BatchWorker,
    CpuBatchWorkspaceStats, CpuGroupFastWorkspace, CpuOperationCache, CpuPreparedImage,
    CpuStagedWorkspace, CpuStreamPool, EncodedImage, IndexedBatchError, J2kDecodeWarning, Mutex,
    NativeSampleType, NonZeroUsize, PrepareFor, PreparedBatch, PreparedBatchGroup,
    PreparedBatchGroupRef, PreparedImage, Rect, Vec,
};
use crate::batch::allocation::try_vec_with_capacity;
use crate::batch::scheduler::decode_retained_tiles;
use crate::{DecodeOutcome, PixelFormat, TileBatchError, TileBatchOptions, TileDecodeJob};

/// Native-width contiguous samples returned by the CPU batch decoder.
#[derive(Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum CpuBatchSamples {
    /// Unsigned samples with precision at most eight bits.
    U8(Vec<u8>),
    /// Unsigned samples with precision from nine through sixteen bits.
    U16(Vec<u16>),
    /// Signed samples with precision at most sixteen bits.
    I16(Vec<i16>),
}

impl CpuBatchSamples {
    /// Native sample type stored by this owner.
    #[must_use]
    pub const fn sample_type(&self) -> NativeSampleType {
        match self {
            Self::U8(_) => NativeSampleType::U8,
            Self::U16(_) => NativeSampleType::U16,
            Self::I16(_) => NativeSampleType::I16,
        }
    }

    /// Total number of channel samples across every image.
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::U8(samples) => samples.len(),
            Self::U16(samples) => samples.len(),
            Self::I16(samples) => samples.len(),
        }
    }

    /// Whether no channel samples are stored.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// One successfully decoded homogeneous CPU output group.
#[derive(Debug, PartialEq, Eq)]
pub struct CpuBatchGroup {
    info: BatchGroupInfo,
    source_indices: Vec<usize>,
    decoded_rects: Vec<Rect>,
    warnings: Vec<Vec<J2kDecodeWarning>>,
    samples: CpuBatchSamples,
}

impl CpuBatchGroup {
    pub(super) fn new(
        info: BatchGroupInfo,
        source_indices: Vec<usize>,
        decoded_rects: Vec<Rect>,
        warnings: Vec<Vec<J2kDecodeWarning>>,
        samples: CpuBatchSamples,
    ) -> Self {
        Self {
            info,
            source_indices,
            decoded_rects,
            warnings,
            samples,
        }
    }

    /// Shared output metadata.
    #[must_use]
    pub fn info(&self) -> &BatchGroupInfo {
        &self.info
    }

    /// Original input indices for the dense batch dimension.
    #[must_use]
    pub fn source_indices(&self) -> &[usize] {
        &self.source_indices
    }

    /// Actual decoded rectangle for each output image.
    #[must_use]
    pub fn decoded_rects(&self) -> &[Rect] {
        &self.decoded_rects
    }

    /// Non-fatal warnings for each output image.
    #[must_use]
    pub fn warnings(&self) -> &[Vec<J2kDecodeWarning>] {
        &self.warnings
    }

    /// Contiguous native-width samples.
    #[must_use]
    pub const fn samples(&self) -> &CpuBatchSamples {
        &self.samples
    }

    /// Consume the group into its metadata, indices, rectangles, warnings, and samples.
    #[must_use]
    pub fn into_parts(
        self,
    ) -> (
        BatchGroupInfo,
        Vec<usize>,
        Vec<Rect>,
        Vec<Vec<J2kDecodeWarning>>,
        CpuBatchSamples,
    ) {
        (
            self.info,
            self.source_indices,
            self.decoded_rects,
            self.warnings,
            self.samples,
        )
    }
}

/// CPU batch decode successes and indexed item failures.
#[derive(Debug, PartialEq, Eq)]
pub struct CpuBatchDecodeResult {
    pub(super) groups: Vec<CpuBatchGroup>,
    pub(super) errors: Vec<IndexedBatchError>,
}

impl CpuBatchDecodeResult {
    /// Successfully decoded homogeneous groups.
    #[must_use]
    pub fn groups(&self) -> &[CpuBatchGroup] {
        &self.groups
    }

    /// Preparation and decode failures in original input order.
    #[must_use]
    pub fn errors(&self) -> &[IndexedBatchError] {
        &self.errors
    }

    /// Consume the result into successful groups and indexed errors.
    #[must_use]
    pub fn into_parts(self) -> (Vec<CpuBatchGroup>, Vec<IndexedBatchError>) {
        (self.groups, self.errors)
    }
}

/// Persistent CPU facade for repeated owned/prepared batch decode.
///
/// Prepared metadata is borrowed across calls. Decoder settings and worker
/// policy are retained by the session. Each worker owns a lifetime-free native
/// workspace whose decoded-component, Tier-1, and IDWT allocations move
/// through the borrowing context and return to the worker after every group,
/// including across calls.
/// Supported Gray/RGB/RGBA classic and HTJ2K inputs execute their retained
/// per-tile offset plans directly. Metadata-only inputs retain the general
/// decode fallback.
pub struct CpuBatchDecoder {
    pub(super) options: BatchDecodeOptions,
    pub(super) workers: Mutex<Vec<BatchWorker>>,
    pub(super) fast_workspace: CpuGroupFastWorkspace,
    pub(super) staged_workspace: CpuStagedWorkspace,
    pub(super) stream_pool: Mutex<CpuStreamPool>,
    pub(super) operation_cache: CpuOperationCache,
}

impl fmt::Debug for CpuBatchDecoder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CpuBatchDecoder")
            .field("options", &self.options)
            .field("retained_workers", &self.retained_worker_count())
            .finish_non_exhaustive()
    }
}

impl CpuBatchDecoder {
    /// Create a persistent CPU batch decoder.
    #[must_use]
    pub fn new(options: BatchDecodeOptions) -> Self {
        let available = std::thread::available_parallelism().map_or(1, NonZeroUsize::get);
        let worker_count =
            options.worker_limit(options.workers.map_or(available, NonZeroUsize::get));
        let workers = (0..worker_count)
            .map(|_| BatchWorker::new_owned(worker_count.max(2)))
            .collect();
        Self {
            options,
            workers: Mutex::new(workers),
            fast_workspace: CpuGroupFastWorkspace::default(),
            staged_workspace: CpuStagedWorkspace::default(),
            stream_pool: Mutex::default(),
            operation_cache: CpuOperationCache::default(),
        }
    }

    /// Retained session options.
    #[must_use]
    pub const fn options(&self) -> BatchDecodeOptions {
        self.options
    }

    /// Threads of the session's streamed-group pool, once a group used it.
    #[cfg(test)]
    pub(super) fn stream_pool_threads(&mut self) -> Option<usize> {
        self.stream_pool
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .threads()
    }

    /// Offset-plan executions of this session's workers with nested
    /// parallelism.
    #[cfg(test)]
    pub(super) fn parallel_plan_decodes(&mut self) -> u64 {
        self.workers
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .iter()
            .map(BatchWorker::parallel_plan_decode_calls)
            .sum()
    }

    /// Number of reusable worker context/scratch owners retained by this session.
    #[must_use]
    pub fn retained_worker_count(&self) -> usize {
        self.workers
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .len()
    }

    /// Aggregate native workspace reuse counters for this session.
    #[must_use]
    pub fn workspace_stats(&self) -> CpuBatchWorkspaceStats {
        let mut aggregate = self
            .workers
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .iter()
            .fold(
                CpuBatchWorkspaceStats::default(),
                |mut aggregate, worker| {
                    aggregate.preparation_calls = aggregate
                        .preparation_calls
                        .saturating_add(worker.preparation_calls());
                    aggregate.preparation_worker_reuses = aggregate
                        .preparation_worker_reuses
                        .saturating_add(worker.preparation_worker_reuses());
                    aggregate.prepared_plan_decode_calls = aggregate
                        .prepared_plan_decode_calls
                        .saturating_add(worker.prepared_plan_decode_calls());
                    aggregate.retained_prepared_plan_ht_workspace_bytes = aggregate
                        .retained_prepared_plan_ht_workspace_bytes
                        .saturating_add(worker.prepared_plan_ht_workspace_bytes());
                    aggregate.retained_prepared_plan_classic_workspace_bytes = aggregate
                        .retained_prepared_plan_classic_workspace_bytes
                        .saturating_add(worker.prepared_plan_classic_workspace_bytes());
                    let worker = worker.native_workspace_stats();
                    aggregate.decode_calls =
                        aggregate.decode_calls.saturating_add(worker.decode_calls());
                    aggregate.component_owner_reuses = aggregate
                        .component_owner_reuses
                        .saturating_add(worker.component_owner_reuses());
                    aggregate.tier1_owner_reuses = aggregate
                        .tier1_owner_reuses
                        .saturating_add(worker.tier1_owner_reuses());
                    aggregate.idwt_owner_reuses = aggregate
                        .idwt_owner_reuses
                        .saturating_add(worker.idwt_owner_reuses());
                    aggregate.scratch_capacity_retries = aggregate
                        .scratch_capacity_retries
                        .saturating_add(worker.scratch_capacity_retries());
                    aggregate.retained_component_bytes = aggregate
                        .retained_component_bytes
                        .saturating_add(worker.retained_component_bytes());
                    aggregate.retained_tier1_bytes = aggregate
                        .retained_tier1_bytes
                        .saturating_add(worker.retained_tier1_bytes());
                    aggregate.retained_idwt_bytes = aggregate
                        .retained_idwt_bytes
                        .saturating_add(worker.retained_idwt_bytes());
                    aggregate
                },
            );
        let fast = self.fast_workspace.stats();
        aggregate.flattened_group_plans = fast.flattened_group_plans;
        aggregate.flattened_payload_jobs = fast.flattened_payload_jobs;
        aggregate.flattened_cleanup_jobs = fast.flattened_cleanup_jobs;
        aggregate.flattened_sigprop_jobs = fast.flattened_sigprop_jobs;
        aggregate.flattened_magref_jobs = fast.flattened_magref_jobs;
        aggregate.flattened_classic_jobs = fast.flattened_classic_jobs;
        aggregate.entropy_job_dispatches = fast.entropy_job_dispatches;
        aggregate.cross_image_entropy_windows = fast.cross_image_entropy_windows;
        aggregate.compressed_arena_reuses = fast.compressed_arena_reuses;
        aggregate.retained_compressed_arena_bytes = fast.retained_compressed_arena_bytes;
        aggregate.output_group_allocations = fast.output_group_allocations;
        aggregate.output_compaction_copied_samples = fast.output_compaction_copied_samples;
        aggregate
    }

    /// Decode independent J2K/HTJ2K tiles into caller-owned output buffers.
    ///
    /// Behaves like [`decode_tiles_into`](crate::decode_tiles_into) with this
    /// session's worker count and decoder settings; `layout` does not apply
    /// because `fmt` defines the output. Each retained worker keeps its
    /// decoded-component, Tier-1, and IDWT allocations across tiles and calls,
    /// so repeated batches neither reallocate nor re-fault them.
    ///
    /// # Errors
    /// Same as [`decode_tiles_into`](crate::decode_tiles_into).
    pub fn decode_tiles_into(
        &mut self,
        jobs: &mut [TileDecodeJob<'_, '_>],
        fmt: PixelFormat,
    ) -> Result<Vec<DecodeOutcome<J2kDecodeWarning>>, TileBatchError> {
        let workers = self
            .workers
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let held = self.options.scheduled_workers(workers.len());
        decode_retained_tiles(
            workers,
            jobs,
            fmt,
            TileBatchOptions { workers: held },
            self.options.settings,
        )
    }

    /// Inspect and group owned encoded inputs for repeated decode.
    ///
    /// Inputs whose output allocation or planned execution exceeds a memory
    /// limit are recorded as indexed preparation errors in the returned batch.
    /// Preparation checks individual images. During decoding, a group too
    /// large for one output allocation is split into smaller result groups.
    pub fn prepare(
        &self,
        inputs: Vec<EncodedImage>,
    ) -> Result<PreparedBatch, BatchInfrastructureError> {
        let mut workers = self
            .workers
            .lock()
            .map_err(|_| BatchInfrastructureError::SchedulerPoisoned)?;
        prepare_batch_with_workers(inputs, self.options, &mut workers, PrepareFor::CpuDecoder)
    }

    /// Inspect and group owned encoded inputs as [`Self::prepare`] does, under
    /// one operation's limits and cancellation.
    ///
    /// Inputs are prepared one at a time. The working-memory limit bounds each
    /// input's native parsing and packet planning together with the plans
    /// already retained for earlier inputs of this call; the prepared batch
    /// keeps only those actual plans, never the control's allowance. The output
    /// limit rejects inputs whose planned decoded output exceeds it, and
    /// cancellation stops at the next input or native checkpoint. Inputs that
    /// fail a limit or observe cancellation are indexed preparation errors.
    /// The control applies to this preparation only; decoding the batch later
    /// is a separate operation.
    pub fn prepare_with_control(
        &self,
        inputs: Vec<EncodedImage>,
        control: &crate::J2kDecodeControl,
    ) -> Result<PreparedBatch, BatchInfrastructureError> {
        let mut workers = self
            .workers
            .lock()
            .map_err(|_| BatchInfrastructureError::SchedulerPoisoned)?;
        let mut spare;
        let worker = if let Some(worker) = workers.first_mut() {
            worker
        } else {
            spare = BatchWorker::new_owned(2);
            &mut spare
        };
        let prepared = super::prepare::prepare_batch_with_control(
            inputs,
            self.options,
            worker,
            PrepareFor::CpuDecoder,
            control,
        );
        drop(workers);
        prepared
    }

    /// Regroup caller-supplied prepared images under this session's output policy.
    ///
    /// This applies the same per-image memory checks as [`Self::prepare`].
    pub fn prepare_prepared_images(
        &self,
        images: Vec<PreparedImage>,
    ) -> Result<PreparedBatch, BatchInfrastructureError> {
        prepare_batch_from_images_for(images, self.options, PrepareFor::CpuDecoder)
    }

    /// Prepare and decode one batch.
    pub fn decode(
        &mut self,
        inputs: Vec<EncodedImage>,
    ) -> Result<CpuBatchDecodeResult, BatchInfrastructureError> {
        let prepared = self.prepare(inputs)?;
        self.decode_prepared(&prepared)
    }

    /// Regroup and decode caller-supplied prepared images without reparsing them.
    pub fn decode_prepared_images(
        &mut self,
        images: Vec<PreparedImage>,
    ) -> Result<CpuBatchDecodeResult, BatchInfrastructureError> {
        let prepared = self.prepare_prepared_images(images)?;
        self.decode_prepared(&prepared)
    }

    /// Decode a previously prepared batch without consuming it.
    ///
    /// Retains the batch's validation and output settings while applying this
    /// session's working-memory admission limit and worker policy.
    pub fn decode_prepared(
        &mut self,
        prepared: &PreparedBatch,
    ) -> Result<CpuBatchDecodeResult, BatchInfrastructureError> {
        let workers = self
            .workers
            .get_mut()
            .map_err(|_| BatchInfrastructureError::SchedulerPoisoned)?;
        let stream_pool = self
            .stream_pool
            .get_mut()
            .map_err(|_| BatchInfrastructureError::SchedulerPoisoned)?;
        let input_count = prepared
            .groups()
            .iter()
            .try_fold(prepared.errors().len(), |count, group| {
                count.checked_add(group.images().len())
            })
            .ok_or(BatchInfrastructureError::AllocationTooLarge {
                what: "J2K owned batch result metadata",
                requested: usize::MAX,
                cap: j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
            })?;
        let mut groups =
            try_vec_with_capacity(prepared.groups().len(), "J2K owned batch result groups")?;
        let mut errors = try_vec_with_capacity(input_count, "J2K owned batch indexed errors")?;
        errors.extend_from_slice(prepared.errors());
        for group in prepared.groups() {
            let images_per_group = output_images_per_group(group);
            // One dense buffer for the whole group would exceed the
            // per-allocation cap: decode consecutive images as several
            // groups with the same info, each buffer within the cap.
            for (images, source_indices) in group
                .images
                .chunks(images_per_group)
                .zip(group.source_indices.chunks(images_per_group))
            {
                // Keep immutable preparation shared while reapplying this
                // receiving session's admission and scheduler claims.
                let mut admitted = try_vec_with_capacity(images.len(), "J2K CPU session images")?;
                for image in images {
                    admitted.push(CpuPreparedImage::legacy(image, &group.info, self.options));
                }
                let part = PreparedBatchGroupRef {
                    info: &group.info,
                    images: &admitted,
                    source_indices,
                    working_memory_limit: self.options.max_working_memory_bytes,
                };
                let held = self.options.scheduled_workers(workers.len());
                if let Some(decoded) = decode_cpu_group(
                    workers,
                    &mut self.fast_workspace,
                    &mut self.staged_workspace,
                    stream_pool,
                    part,
                    prepared.options,
                    held,
                    &mut errors,
                )? {
                    push_group(&mut groups, decoded)?;
                }
            }
        }
        errors.sort_by_key(|error| error.index);
        Ok(CpuBatchDecodeResult { groups, errors })
    }
}

/// Images whose dense outputs fit one per-allocation-capped buffer together
/// (at least one). An image whose own output exceeds the cap still fails in
/// its own group, as before.
fn output_images_per_group(group: &PreparedBatchGroup) -> usize {
    group
        .info
        .output_bytes(1)
        .map_or(group.images.len(), |image_bytes| {
            images_within_cap(image_bytes, j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES)
        })
}

/// How many images of `image_bytes` one buffer of at most `cap` bytes holds,
/// never fewer than one.
fn images_within_cap(image_bytes: usize, cap: usize) -> usize {
    cap.checked_div(image_bytes).unwrap_or(usize::MAX).max(1)
}

fn push_group(
    groups: &mut Vec<CpuBatchGroup>,
    group: CpuBatchGroup,
) -> Result<(), BatchInfrastructureError> {
    groups
        .try_reserve(1)
        .map_err(|_| BatchInfrastructureError::HostAllocationFailed {
            what: "J2K owned batch result groups",
            bytes: core::mem::size_of::<CpuBatchGroup>(),
        })?;
    groups.push(group);
    Ok(())
}

impl BatchDecoder for CpuBatchDecoder {
    type Output = CpuBatchDecodeResult;
    type Error = BatchInfrastructureError;

    fn options(&self) -> BatchDecodeOptions {
        self.options
    }

    fn prepare_batch(&self, inputs: Vec<EncodedImage>) -> Result<PreparedBatch, Self::Error> {
        self.prepare(inputs)
    }

    fn prepare_prepared_images(
        &self,
        images: Vec<PreparedImage>,
    ) -> Result<PreparedBatch, Self::Error> {
        Self::prepare_prepared_images(self, images)
    }

    fn decode_prepared(&mut self, prepared: &PreparedBatch) -> Result<Self::Output, Self::Error> {
        Self::decode_prepared(self, prepared)
    }
}

#[cfg(test)]
mod output_group_tests {
    use super::images_within_cap;

    #[test]
    fn groups_above_the_cap_split_into_buffers_within_it() {
        const MIB: usize = 1024 * 1024;
        // Eight 5792² RGB8 images (96 MiB each) under a 512 MiB cap.
        assert_eq!(images_within_cap(96 * MIB, 512 * MIB), 5);
        assert_eq!(images_within_cap(512 * MIB, 512 * MIB), 1);
        // An image above the cap keeps a group of its own.
        assert_eq!(images_within_cap(768 * MIB, 512 * MIB), 1);
        assert_eq!(images_within_cap(3 * MIB, 512 * MIB), 170);
        assert_eq!(images_within_cap(0, 512 * MIB), usize::MAX);
    }
}
