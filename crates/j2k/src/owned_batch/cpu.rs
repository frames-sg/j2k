// SPDX-License-Identifier: MIT OR Apache-2.0

//! Persistent CPU batch session and public native output owners.

use super::{
    decode_cpu_group, fmt, prepare_batch_from_images, prepare_batch_with_workers,
    BatchDecodeOptions, BatchDecoder, BatchGroupInfo, BatchInfrastructureError, BatchWorker,
    CpuBatchWorkspaceStats, CpuGroupFastWorkspace, CpuStagedWorkspace, EncodedImage,
    IndexedBatchError, J2kDecodeWarning, Mutex, NativeSampleType, NonZeroUsize, PreparedBatch,
    PreparedImage, Rect, Vec, MAX_GENERIC_BATCH_WORKERS,
};

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
    groups: Vec<CpuBatchGroup>,
    errors: Vec<IndexedBatchError>,
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
    options: BatchDecodeOptions,
    workers: Mutex<Vec<BatchWorker>>,
    fast_workspace: CpuGroupFastWorkspace,
    staged_workspace: CpuStagedWorkspace,
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
        let worker_count = options
            .workers
            .map_or(available, NonZeroUsize::get)
            .clamp(1, MAX_GENERIC_BATCH_WORKERS);
        let workers = (0..worker_count)
            .map(|_| BatchWorker::new_owned(worker_count.max(2)))
            .collect();
        Self {
            options,
            workers: Mutex::new(workers),
            fast_workspace: CpuGroupFastWorkspace::default(),
            staged_workspace: CpuStagedWorkspace::default(),
        }
    }

    /// Retained session options.
    #[must_use]
    pub const fn options(&self) -> BatchDecodeOptions {
        self.options
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

    /// Inspect and group owned encoded inputs for repeated decode.
    pub fn prepare(
        &self,
        inputs: Vec<EncodedImage>,
    ) -> Result<PreparedBatch, BatchInfrastructureError> {
        let mut workers = self
            .workers
            .lock()
            .map_err(|_| BatchInfrastructureError::SchedulerPoisoned)?;
        prepare_batch_with_workers(inputs, self.options, &mut workers)
    }

    /// Regroup caller-supplied prepared images under this session's output policy.
    pub fn prepare_prepared_images(
        &self,
        images: Vec<PreparedImage>,
    ) -> Result<PreparedBatch, BatchInfrastructureError> {
        prepare_batch_from_images(images, self.options)
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
    pub fn decode_prepared(
        &mut self,
        prepared: &PreparedBatch,
    ) -> Result<CpuBatchDecodeResult, BatchInfrastructureError> {
        let workers = self
            .workers
            .get_mut()
            .map_err(|_| BatchInfrastructureError::SchedulerPoisoned)?;
        let mut groups = Vec::new();
        let mut errors = prepared.errors.to_vec();
        for group in prepared.groups() {
            if let Some(decoded) = decode_cpu_group(
                workers,
                &mut self.fast_workspace,
                &mut self.staged_workspace,
                group,
                prepared.options,
                self.options.workers,
                &mut errors,
            )? {
                groups.push(decoded);
            }
        }
        errors.sort_by_key(|error| error.index);
        Ok(CpuBatchDecodeResult { groups, errors })
    }
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
        CpuBatchDecoder::prepare_prepared_images(self, images)
    }

    fn decode_prepared(&mut self, prepared: &PreparedBatch) -> Result<Self::Output, Self::Error> {
        CpuBatchDecoder::decode_prepared(self, prepared)
    }
}
