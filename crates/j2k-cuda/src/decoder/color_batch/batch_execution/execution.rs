// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::decoder::resident::{
    enqueue_chunked_htj2k_cleanup_dequant, enqueue_component_cleanup_dequant_batches,
    ChunkedHtj2kCleanup,
};
use j2k_cuda_j2k_engine::{
    CudaHtj2kDecodeResources, CudaHtj2kDecodeTableResources, CudaQueuedHtj2kCleanup, J2kCudaEngine,
};
use j2k_cuda_runtime::CudaContext;

use super::completion::batch_store::can_batch_rgb8_mct_color_store;
use super::preparation::{
    prepare_color_cuda_resident_batch_with_cap, PreparedColorCudaResidentBatch,
};
use crate::error::combine_cuda_cleanup_errors;
use j2k_core::{HtGpuJobChunkLimits, PixelFormat};

use super::super::{
    build_color_component_work, cuda_error, final_vertical_fuses_with_store, host_owners, profile,
    run_color_component_idwt_batches, run_component_cleanup_dequant_batches, CudaBufferPool,
    CudaComponentDecodeWork, CudaHtj2kColorDecodePlans, CudaQueuedIdwtBatch, CudaSession, Error,
    HostPhaseBudget,
};

/// Upload chunk size for color entropy decode. Not derived from a measurement.
pub(super) const COLOR_HT_PAYLOAD_CHUNK_BYTES: usize = 8 * 1024 * 1024;
/// Images planned and enqueued together per pipelined entropy group. Not
/// derived from a measurement.
pub(super) const COLOR_ENTROPY_PIPELINE_GROUP_IMAGES: usize = 4;

/// Profiled batches (`collect_stage_timings`) always take the single-group,
/// single-upload route, so their stage timings do not cover the pipelined or
/// chunked routes larger batches use.
pub(super) fn should_pipeline_color_entropy_groups(
    inputs: &[&[u8]],
    collect_stage_timings: bool,
    max_payload_bytes: usize,
) -> bool {
    let chunk_bytes = max_payload_bytes.min(COLOR_HT_PAYLOAD_CHUNK_BYTES);
    !collect_stage_timings
        && inputs.len() > COLOR_ENTROPY_PIPELINE_GROUP_IMAGES
        && inputs
            .iter()
            .fold(0usize, |total, input| total.saturating_add(input.len()))
            > chunk_bytes.saturating_mul(2)
}

pub(super) enum ColorBatchDecodeResources {
    Uploaded(CudaHtj2kDecodeResources),
    Chunked {
        tables: CudaHtj2kDecodeTableResources,
        limits: HtGpuJobChunkLimits,
    },
}

pub(super) enum PendingColorCleanup {
    Single {
        queued: CudaQueuedHtj2kCleanup,
        mapping: ColorCleanupJobMapping,
    },
    Chunked {
        queued: ChunkedHtj2kCleanup,
        source_index_base: usize,
        job_index_base: usize,
    },
}

pub(super) struct ColorCleanupJobMapping {
    source_index_base: usize,
    job_index_base: usize,
    source_job_ends: Vec<usize>,
}

impl PendingColorCleanup {
    fn queued_retained_host_bytes(&self) -> usize {
        match self {
            Self::Single { queued, .. } => queued.retained_host_bytes(),
            Self::Chunked { queued, .. } => queued.retained_host_bytes(),
        }
    }

    fn retained_host_bytes(&self) -> usize {
        self.queued_retained_host_bytes()
            .saturating_add(match self {
                Self::Single { mapping, .. } => {
                    j2k_core::host_capacity_bytes::<usize>(mapping.source_job_ends.capacity())
                }
                Self::Chunked { .. } => 0,
            })
    }

    fn status_count(&self) -> usize {
        match self {
            Self::Single { queued, .. } => queued.status_count(),
            Self::Chunked { queued, .. } => queued.status_count(),
        }
    }

    fn finish(self) -> Result<(), Error> {
        match self {
            Self::Single { queued, mapping } => queued
                .finish()
                .map(|_| ())
                .map_err(|error| mapping.map_error(error)),
            Self::Chunked {
                queued,
                source_index_base,
                job_index_base,
            } => queued.finish().map_err(|error| {
                offset_color_cleanup_error(error, source_index_base, job_index_base)
            }),
        }
    }
}

impl ColorCleanupJobMapping {
    fn map_error(self, error: j2k_cuda_runtime::CudaError) -> Error {
        let Some(local_job_index) = error.kernel_job_index() else {
            return cuda_error(error);
        };
        let local_source_index = self
            .source_job_ends
            .partition_point(|&end| end <= local_job_index);
        if local_source_index >= self.source_job_ends.len() {
            return cuda_error(error);
        }
        Error::CudaTier1JobFailed {
            source_index: self.source_index_base.saturating_add(local_source_index),
            original_job_index: self.job_index_base.saturating_add(local_job_index),
            source: error,
        }
    }
}

fn offset_color_cleanup_error(
    error: Error,
    source_index_base: usize,
    job_index_base: usize,
) -> Error {
    match error {
        Error::CudaTier1JobFailed {
            source_index,
            original_job_index,
            source,
        } => Error::CudaTier1JobFailed {
            source_index: source_index.saturating_add(source_index_base),
            original_job_index: original_job_index.saturating_add(job_index_base),
            source,
        },
        error => error,
    }
}

pub(super) struct ColorEntropyOwner {
    pending_cleanup: Option<PendingColorCleanup>,
    decode_resources: ColorBatchDecodeResources,
}

impl ColorEntropyOwner {
    fn retained_host_bytes(&self) -> usize {
        self.pending_cleanup
            .as_ref()
            .map_or(0, PendingColorCleanup::retained_host_bytes)
    }

    fn finish(self) -> Result<(), Error> {
        let Self {
            pending_cleanup,
            decode_resources,
        } = self;
        let result = pending_cleanup.map_or(Ok(()), PendingColorCleanup::finish);
        drop(decode_resources);
        result
    }

    fn finish_after_error(self, primary: Error) -> Error {
        match self.finish() {
            Ok(()) => primary,
            Err(cleanup) => combine_cuda_cleanup_errors(primary, cleanup),
        }
    }

    fn status_count(&self) -> usize {
        self.pending_cleanup
            .as_ref()
            .map_or(0, PendingColorCleanup::status_count)
    }

    #[cfg(test)]
    fn chunk_count(&self) -> usize {
        match &self.pending_cleanup {
            Some(PendingColorCleanup::Chunked { queued, .. }) => queued.chunk_count(),
            _ => 0,
        }
    }
}

/// Entropy owners for every enqueued group; a non-pipelined batch is one group.
pub(super) struct ColorEntropyOwners(Vec<ColorEntropyOwner>);

impl ColorEntropyOwners {
    pub(super) fn retained_host_bytes(&self) -> usize {
        j2k_core::host_capacity_bytes::<ColorEntropyOwner>(self.0.capacity()).saturating_add(
            self.0
                .iter()
                .map(ColorEntropyOwner::retained_host_bytes)
                .sum::<usize>(),
        )
    }

    pub(super) fn finish(self) -> Result<(), Error> {
        let mut error = None;
        for owner in self.0 {
            if let Err(next) = owner.finish() {
                error = Some(match error {
                    Some(primary) => combine_cuda_cleanup_errors(primary, next),
                    None => next,
                });
            }
        }
        error.map_or(Ok(()), Err)
    }

    fn finish_after_error(self, primary: Error) -> Error {
        match self.finish() {
            Ok(()) => primary,
            Err(cleanup) => combine_cuda_cleanup_errors(primary, cleanup),
        }
    }
}

pub(super) struct EnqueuedColorCudaResidentBatch {
    pub(super) context: CudaContext,
    pub(super) pool: CudaBufferPool,
    pub(super) output_pool: CudaBufferPool,
    pub(super) colors: Vec<CudaHtj2kColorDecodePlans>,
    pub(super) component_work: Vec<CudaComponentDecodeWork>,
    pub(super) pending_idwt_batch: Option<CudaQueuedIdwtBatch>,
    pub(super) entropy_owners: ColorEntropyOwners,
    pub(super) entropy_live_host_bytes: usize,
    /// Decided with `fused_final_vertical` before IDWT is enqueued; the fused
    /// final vertical pass only runs inside the batch store.
    pub(super) use_batch_store: bool,
    pub(super) fused_final_vertical: bool,
    pub(super) table_upload_us: u128,
    pub(super) payload_upload_us: u128,
}

/// Where one entropy group's images and HT jobs sit in the whole batch.
#[derive(Clone, Copy)]
struct ColorGroupPlacement {
    source_index_base: usize,
    job_index_base: usize,
}

struct EnqueuedColorEntropyGroup {
    colors: Vec<CudaHtj2kColorDecodePlans>,
    component_work: Vec<CudaComponentDecodeWork>,
    owner: ColorEntropyOwner,
    ht_job_count: usize,
    table_upload_us: u128,
    payload_upload_us: u128,
}

struct ColorEntropyPipeline {
    host_budget: HostPhaseBudget,
    colors: Vec<CudaHtj2kColorDecodePlans>,
    component_work: Vec<CudaComponentDecodeWork>,
    owners: Vec<ColorEntropyOwner>,
    ht_job_count: usize,
    table_upload_us: u128,
    payload_upload_us: u128,
}

impl ColorEntropyPipeline {
    fn new(input_count: usize) -> Result<Self, Error> {
        let group_count = input_count.div_ceil(COLOR_ENTROPY_PIPELINE_GROUP_IMAGES);
        let mut host_budget =
            HostPhaseBudget::new("j2k CUDA pipelined color retained execution graph");
        Ok(Self {
            colors: host_budget.try_vec_with_capacity(input_count)?,
            component_work: host_budget.try_vec_with_capacity(input_count.saturating_mul(3))?,
            owners: host_budget.try_vec_with_capacity(group_count)?,
            host_budget,
            ht_job_count: 0,
            table_upload_us: 0,
            payload_upload_us: 0,
        })
    }

    fn remaining_host_cap(&self) -> usize {
        j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES.saturating_sub(self.host_budget.live_bytes())
    }

    fn placement(&self, source_index_base: usize) -> ColorGroupPlacement {
        ColorGroupPlacement {
            source_index_base,
            job_index_base: self.ht_job_count,
        }
    }

    fn append_group(&mut self, mut group: EnqueuedColorEntropyGroup) -> Result<(), Error> {
        let mut accounting_result = (|| {
            for color in &group.colors {
                color.account_host_owners(&mut self.host_budget)?;
            }
            for work in &group.component_work {
                host_owners::account_component_work_item(&mut self.host_budget, work)?;
            }
            self.host_budget
                .account_bytes(group.owner.retained_host_bytes())?;
            Ok::<(), Error>(())
        })();
        let next_ht_job_count =
            self.ht_job_count
                .checked_add(group.ht_job_count)
                .ok_or(Error::HostAllocationFailed {
                    bytes: usize::MAX,
                    what: "j2k CUDA pipelined color HT job count",
                });
        match next_ht_job_count {
            Ok(count) => self.ht_job_count = count,
            Err(error) if accounting_result.is_ok() => accounting_result = Err(error),
            Err(_) => {}
        }
        self.table_upload_us = self.table_upload_us.saturating_add(group.table_upload_us);
        self.payload_upload_us = self
            .payload_upload_us
            .saturating_add(group.payload_upload_us);
        self.colors.append(&mut group.colors);
        self.component_work.append(&mut group.component_work);
        self.owners.push(group.owner);
        accounting_result
    }

    fn enqueue_group_now(
        &mut self,
        session: &mut CudaSession,
        context: &CudaContext,
        pool: &CudaBufferPool,
        prepared: PreparedColorCudaResidentBatch<'_>,
        source_index_base: usize,
        collect_stage_timings: bool,
    ) -> Result<(), Error> {
        let group = enqueue_color_entropy_group(
            session,
            context,
            pool,
            prepared,
            self.placement(source_index_base),
            self.host_budget.live_bytes(),
            collect_stage_timings,
        )?;
        self.append_group(group)
    }

    /// Enqueues groups of [`COLOR_ENTROPY_PIPELINE_GROUP_IMAGES`] in input
    /// order, planning the next group while the current one enqueues. After
    /// the first host planning limit, the remaining groups run without overlap
    /// so each planner gets the whole remaining budget.
    fn enqueue_groups(
        &mut self,
        session: &mut CudaSession,
        context: &CudaContext,
        pool: &CudaBufferPool,
        inputs: &[&[u8]],
        fmt: PixelFormat,
    ) -> Result<(), Error> {
        let group_count = inputs.len().div_ceil(COLOR_ENTROPY_PIPELINE_GROUP_IMAGES);
        let group_inputs = |index: usize| {
            let start = index * COLOR_ENTROPY_PIPELINE_GROUP_IMAGES;
            &inputs[start..(start + COLOR_ENTROPY_PIPELINE_GROUP_IMAGES).min(inputs.len())]
        };
        let mut sequential = false;
        let mut prepared_next = None;
        for index in 0..group_count {
            let source_index_base = index * COLOR_ENTROPY_PIPELINE_GROUP_IMAGES;
            let prepared = match prepared_next.take() {
                Some(prepared) => prepared,
                None => prepare_color_cuda_resident_batch_with_cap(
                    group_inputs(index),
                    fmt,
                    self.remaining_host_cap(),
                )?,
            };
            if sequential || index + 1 == group_count {
                self.enqueue_group_now(session, context, pool, prepared, source_index_base, false)?;
                continue;
            }

            let next_inputs = group_inputs(index + 1);
            let next_prepare_cap = self
                .remaining_host_cap()
                .saturating_sub(prepared.retained_host_bytes()?)
                / 2;
            let reserved_external_bytes = self
                .host_budget
                .live_bytes()
                .saturating_add(next_prepare_cap);
            let placement = self.placement(source_index_base);
            let (group_result, next_result) = std::thread::scope(|scope| {
                let next = scope.spawn(|| {
                    prepare_color_cuda_resident_batch_with_cap(next_inputs, fmt, next_prepare_cap)
                });
                let group = enqueue_color_entropy_group(
                    session,
                    context,
                    pool,
                    prepared,
                    placement,
                    reserved_external_bytes,
                    false,
                );
                let next = match next.join() {
                    Ok(next) => next,
                    Err(payload) => std::panic::resume_unwind(payload),
                };
                (group, next)
            });

            match group_result {
                Ok(group) => self.append_group(group)?,
                Err(error) if error.is_host_planning_limit() => {
                    drop(next_result);
                    sequential = true;
                    let prepared = prepare_color_cuda_resident_batch_with_cap(
                        group_inputs(index),
                        fmt,
                        self.remaining_host_cap(),
                    )?;
                    self.enqueue_group_now(
                        session,
                        context,
                        pool,
                        prepared,
                        source_index_base,
                        false,
                    )?;
                    continue;
                }
                Err(error) => return Err(error),
            }
            match next_result {
                Ok(next) => prepared_next = Some(next),
                Err(error) if error.is_host_planning_limit() => sequential = true,
                Err(error) => return Err(error),
            }
        }
        Ok(())
    }

    /// Enqueues the batched IDWT and transfers ownership of every enqueued
    /// group to the completion stage.
    fn into_enqueued_batch(
        mut self,
        pools: ColorBatchPools,
        fmt: PixelFormat,
        collect_stage_timings: bool,
    ) -> Result<EnqueuedColorCudaResidentBatch, Error> {
        let ColorBatchPools {
            context,
            pool,
            output_pool,
        } = pools;
        let preflight = self.owners.iter().try_for_each(|owner| {
            self.host_budget
                .preflight_capacity::<j2k_cuda_j2k_engine::CudaHtj2kStatus>(owner.status_count())
                .map(|_| ())
        });
        if let Err(error) = preflight {
            return Err(self.retire_after_error(error.into()));
        }
        let idwt = match enqueue_color_idwt(
            &context,
            &pool,
            &self.colors,
            &mut self.component_work,
            fmt,
            collect_stage_timings,
            &mut self.host_budget,
        ) {
            Ok(idwt) => idwt,
            Err(error) => return Err(self.retire_after_error(error)),
        };
        let entropy_owners = ColorEntropyOwners(self.owners);
        let entropy_live_host_bytes = entropy_owners.retained_host_bytes();
        Ok(EnqueuedColorCudaResidentBatch {
            context,
            pool,
            output_pool,
            colors: self.colors,
            component_work: self.component_work,
            pending_idwt_batch: idwt.pending,
            entropy_owners,
            entropy_live_host_bytes,
            use_batch_store: idwt.use_batch_store,
            fused_final_vertical: idwt.fused_final_vertical,
            table_upload_us: self.table_upload_us,
            payload_upload_us: self.payload_upload_us,
        })
    }

    #[cfg(test)]
    fn record_chunk_count_for_test(&self, session: &mut CudaSession) {
        session.record_htj2k_decode_chunk_count_for_test(
            self.owners.iter().fold(0usize, |count, owner| {
                count.saturating_add(owner.chunk_count())
            }),
        );
    }

    fn retire_after_error(self, primary: Error) -> Error {
        let Self {
            colors,
            component_work,
            owners,
            ..
        } = self;
        let error = ColorEntropyOwners(owners).finish_after_error(primary);
        drop(component_work);
        drop(colors);
        error
    }
}

struct ColorBatchPools {
    context: CudaContext,
    pool: CudaBufferPool,
    /// Final surfaces can outlive a batch and must not displace its larger
    /// entropy scratch allocations from the bounded batch working-set cache.
    output_pool: CudaBufferPool,
}

impl ColorBatchPools {
    fn from_session(session: &mut CudaSession) -> Result<Self, Error> {
        Ok(Self {
            context: session.cuda_context()?,
            pool: session.decode_batch_buffer_pool()?,
            output_pool: session.decode_buffer_pool()?,
        })
    }
}

pub(super) fn enqueue_color_cuda_resident_batch_pipelined(
    inputs: &[&[u8]],
    session: &mut CudaSession,
    fmt: PixelFormat,
) -> Result<EnqueuedColorCudaResidentBatch, Error> {
    let pools = ColorBatchPools::from_session(session)?;
    let mut pipeline = ColorEntropyPipeline::new(inputs.len())?;
    if let Err(error) = pipeline.enqueue_groups(session, &pools.context, &pools.pool, inputs, fmt) {
        return Err(pipeline.retire_after_error(error));
    }
    #[cfg(test)]
    pipeline.record_chunk_count_for_test(session);
    pipeline.into_enqueued_batch(pools, fmt, false)
}

pub(super) fn enqueue_color_cuda_resident_batch(
    session: &mut CudaSession,
    prepared: PreparedColorCudaResidentBatch<'_>,
    fmt: PixelFormat,
    collect_stage_timings: bool,
) -> Result<EnqueuedColorCudaResidentBatch, Error> {
    let pools = ColorBatchPools::from_session(session)?;
    let mut pipeline = ColorEntropyPipeline::new(prepared.colors.len())?;
    if let Err(error) = pipeline.enqueue_group_now(
        session,
        &pools.context,
        &pools.pool,
        prepared,
        0,
        collect_stage_timings,
    ) {
        return Err(pipeline.retire_after_error(error));
    }
    #[cfg(test)]
    pipeline.record_chunk_count_for_test(session);
    pipeline.into_enqueued_batch(pools, fmt, collect_stage_timings)
}

fn enqueue_color_entropy_group(
    session: &mut CudaSession,
    context: &CudaContext,
    pool: &CudaBufferPool,
    prepared: PreparedColorCudaResidentBatch<'_>,
    placement: ColorGroupPlacement,
    external_live_bytes: usize,
    collect_stage_timings: bool,
) -> Result<EnqueuedColorEntropyGroup, Error> {
    let payload_parts = prepared.payload_parts_with_live_bytes(external_live_bytes)?;
    let mut host_budget =
        HostPhaseBudget::with_live_bytes("j2k CUDA color entropy graph", external_live_bytes)?;
    prepared.account_host_owners(&mut host_budget)?;
    host_budget.account_vec(&payload_parts)?;
    let source_job_ends = color_source_job_ends(&prepared.colors, &mut host_budget)?;
    let ht_job_count = source_job_ends.last().copied().unwrap_or(0);
    let upload = upload_color_batch_resources(
        session,
        context,
        pool,
        &prepared.colors,
        &payload_parts,
        ht_job_count,
        collect_stage_timings,
    )?;
    let mut component_work = build_color_component_work(
        context,
        pool,
        &prepared.colors,
        collect_stage_timings,
        &mut host_budget,
    )?;
    let pending_cleanup = enqueue_color_entropy_cleanup(
        context,
        pool,
        &upload.resources,
        &prepared.colors,
        &payload_parts,
        &mut component_work,
        collect_stage_timings,
        ColorCleanupJobMapping {
            source_index_base: placement.source_index_base,
            job_index_base: placement.job_index_base,
            source_job_ends,
        },
        &mut host_budget,
    )?;
    let owner = ColorEntropyOwner {
        pending_cleanup,
        decode_resources: upload.resources,
    };
    if let Err(error) = host_budget.account_bytes(
        owner
            .pending_cleanup
            .as_ref()
            .map_or(0, PendingColorCleanup::queued_retained_host_bytes),
    ) {
        return Err(owner.finish_after_error(error.into()));
    }
    drop(payload_parts);
    Ok(EnqueuedColorEntropyGroup {
        colors: prepared.colors,
        component_work,
        owner,
        ht_job_count,
        table_upload_us: upload.table_upload_us,
        payload_upload_us: upload.payload_upload_us,
    })
}

/// Running HT job totals at the end of each image, in batch order.
fn color_source_job_ends(
    colors: &[CudaHtj2kColorDecodePlans],
    host_budget: &mut HostPhaseBudget,
) -> Result<Vec<usize>, Error> {
    let mut source_job_ends = host_budget.try_vec_with_capacity(colors.len())?;
    let mut job_count = 0usize;
    for color in colors {
        for component in &color.components {
            job_count = job_count.checked_add(component.code_blocks().len()).ok_or(
                Error::HostAllocationFailed {
                    bytes: usize::MAX,
                    what: "j2k CUDA color entropy job identity ranges",
                },
            )?;
        }
        source_job_ends.push(job_count);
    }
    Ok(source_job_ends)
}

struct UploadedColorResources {
    resources: ColorBatchDecodeResources,
    table_upload_us: u128,
    payload_upload_us: u128,
}

fn upload_color_batch_resources(
    session: &mut CudaSession,
    context: &CudaContext,
    pool: &CudaBufferPool,
    colors: &[CudaHtj2kColorDecodePlans],
    payload_parts: &[&[u8]],
    ht_job_count: usize,
    collect_stage_timings: bool,
) -> Result<UploadedColorResources, Error> {
    let table_upload_start = profile::profile_now(collect_stage_timings);
    let table_resources = if colors.iter().all(|color| {
        color
            .components
            .iter()
            .all(|plan| plan.subbands().is_empty())
    }) {
        None
    } else {
        Some(session.htj2k_decode_table_resources()?)
    };
    let table_upload_us = profile::elapsed_us(table_upload_start);
    let session_limits = session.htj2k_decode_chunk_limits();
    let chunk_bytes = session_limits
        .max_payload_bytes()
        .min(COLOR_HT_PAYLOAD_CHUNK_BYTES);
    let payload_bytes = payload_parts.iter().map(|part| part.len()).sum::<usize>();
    let pure_ht = colors
        .iter()
        .flat_map(|color| &color.components)
        .all(|component| component.classic_subbands().is_empty());
    let descriptor_bytes =
        ht_job_count.saturating_mul(j2k_cuda_j2k_engine::htj2k_cleanup_multi_descriptor_bytes());
    let use_chunks = !collect_stage_timings
        && pure_ht
        && (payload_bytes
            > chunk_bytes
                .saturating_mul(2)
                .min(session_limits.max_payload_bytes())
            || ht_job_count > session_limits.max_jobs().get()
            || descriptor_bytes > session_limits.max_descriptor_bytes());
    if use_chunks {
        if let Some(tables) = table_resources {
            return Ok(UploadedColorResources {
                resources: ColorBatchDecodeResources::Chunked {
                    tables,
                    limits: HtGpuJobChunkLimits::new(
                        session_limits.max_jobs(),
                        chunk_bytes,
                        session_limits.max_descriptor_bytes(),
                    ),
                },
                table_upload_us,
                payload_upload_us: 0,
            });
        }
    }
    let payload_upload_start = profile::profile_now(collect_stage_timings);
    // Preparation rebased block offsets onto this back-to-back layout.
    let engine = J2kCudaEngine::new(context);
    let resources = match table_resources.as_ref() {
        Some(tables) => {
            engine.upload_htj2k_decode_resources_with_tables_and_pool(payload_parts, tables, pool)
        }
        None => engine.upload_j2k_decode_payload_with_pool(payload_parts, pool),
    }
    .map_err(cuda_error)?;
    Ok(UploadedColorResources {
        resources: ColorBatchDecodeResources::Uploaded(resources),
        table_upload_us,
        payload_upload_us: profile::elapsed_us(payload_upload_start),
    })
}

struct EnqueuedColorIdwt {
    pending: Option<CudaQueuedIdwtBatch>,
    use_batch_store: bool,
    fused_final_vertical: bool,
}

fn enqueue_color_idwt(
    context: &CudaContext,
    pool: &CudaBufferPool,
    colors: &[CudaHtj2kColorDecodePlans],
    component_work: &mut [CudaComponentDecodeWork],
    fmt: PixelFormat,
    collect_stage_timings: bool,
    host_budget: &mut HostPhaseBudget,
) -> Result<EnqueuedColorIdwt, Error> {
    let mut batch_components = host_budget.try_vec_with_capacity(component_work.len())?;
    for color in colors {
        batch_components.extend(color.components.iter());
    }
    let use_batch_store = can_batch_rgb8_mct_color_store(fmt, colors, component_work)?;
    let fused_final_vertical =
        use_batch_store && colors.iter().all(final_vertical_fuses_with_store);
    let pending = run_color_component_idwt_batches(
        context,
        &batch_components,
        component_work,
        pool,
        collect_stage_timings,
        host_budget.live_bytes(),
        fused_final_vertical,
    )?;
    Ok(EnqueuedColorIdwt {
        pending,
        use_batch_store,
        fused_final_vertical,
    })
}

#[expect(
    clippy::too_many_arguments,
    reason = "entropy submission keeps resource, source-identity, and host-budget ownership explicit"
)]
fn enqueue_color_entropy_cleanup(
    context: &CudaContext,
    pool: &CudaBufferPool,
    decode_resources: &ColorBatchDecodeResources,
    colors: &[CudaHtj2kColorDecodePlans],
    payload_parts: &[&[u8]],
    component_work: &mut [CudaComponentDecodeWork],
    collect_stage_timings: bool,
    mapping: ColorCleanupJobMapping,
    host_budget: &mut HostPhaseBudget,
) -> Result<Option<PendingColorCleanup>, Error> {
    match decode_resources {
        ColorBatchDecodeResources::Uploaded(resources) => {
            if collect_stage_timings {
                // Profiled cleanup runs synchronously and reports kernel
                // statuses before returning.
                run_component_cleanup_dequant_batches(
                    context,
                    resources,
                    component_work,
                    pool,
                    true,
                    host_budget.live_bytes(),
                )?;
                return Ok(None);
            }
            Ok(enqueue_component_cleanup_dequant_batches(
                context,
                resources,
                component_work,
                pool,
                host_budget.live_bytes(),
            )?
            .map(|queued| PendingColorCleanup::Single { queued, mapping }))
        }
        ColorBatchDecodeResources::Chunked { tables, limits } => {
            let mut source_indices = host_budget.try_vec_with_capacity(component_work.len())?;
            for (index, color) in colors.iter().enumerate() {
                source_indices.extend(std::iter::repeat_n(index, color.components.len()));
            }
            Ok(Some(PendingColorCleanup::Chunked {
                queued: enqueue_chunked_htj2k_cleanup_dequant(
                    context,
                    Some(tables),
                    payload_parts,
                    component_work,
                    &source_indices,
                    pool,
                    *limits,
                    host_budget.live_bytes(),
                )?,
                source_index_base: mapping.source_index_base,
                job_index_base: mapping.job_index_base,
            }))
        }
    }
}
