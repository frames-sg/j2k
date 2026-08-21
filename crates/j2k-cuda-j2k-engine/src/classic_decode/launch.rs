// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{ffi::c_void, sync::Arc, time::Instant};

use super::abi::{
    CudaClassicDecodeStageTimings, CudaClassicDecodeTableResourceInner,
    CudaClassicDecodeTableResources, CudaClassicDecodeTarget, CudaClassicStatus,
    CLASSIC_KERNEL_TABLES,
};
use super::bytes::{
    classic_jobs_as_bytes, classic_segments_as_bytes, classic_statuses_as_bytes_mut,
    classic_tables_as_bytes,
};
use super::prepare::{
    checked_bytes, invalid, prepare_classic_decode, validate_classic_launch_owners,
};
use super::queued::CudaQueuedClassicDecode;
use crate::{
    allocation::HostPhaseBudget, build_flags::ensure_j2k_classic_decode_ptx_built,
    CudaHtj2kDecodeResources, J2kCudaEngine,
};
use j2k_cuda_runtime::{
    cuda_kernel_param, CudaBufferPool, CudaContext, CudaDeviceBuffer, CudaError,
    CudaExecutionStats, CudaKernelSpec, CudaLaunchGeometry, CudaPooledDeviceBuffer,
};

const CLASSIC_KERNEL_NAME: &str = "j2k_decode_classic_codeblocks_multi";
const CLASSIC_KERNEL_ENTRYPOINT: &[u8] = b"j2k_decode_classic_codeblocks_multi\0";
const CLASSIC_DECODE_CODEBLOCK_THREADS: u32 = 32;
const STATUS_RESOURCE_INDEX: usize = 3;

macro_rules! cuda_kernel_params {
    ($($arg:ident),+ $(,)?) => {
        [$(cuda_kernel_param(&mut $arg)),+]
    };
}

impl J2kCudaEngine<'_> {
    /// Upload static classic Tier-1 lookup tables once for session reuse.
    #[doc(hidden)]
    pub fn upload_classic_decode_table_resources(
        &self,
    ) -> Result<CudaClassicDecodeTableResources, CudaError> {
        Ok(CudaClassicDecodeTableResources {
            inner: Arc::new(CudaClassicDecodeTableResourceInner {
                tables: self
                    .context
                    .upload(classic_tables_as_bytes(&CLASSIC_KERNEL_TABLES))?,
            }),
        })
    }

    /// Allocate and clear one classic Tier-1 coefficient plane.
    #[doc(hidden)]
    pub fn allocate_classic_coefficients_with_pool(
        &self,
        output_words: usize,
        pool: &CudaBufferPool,
    ) -> Result<CudaPooledDeviceBuffer, CudaError> {
        if !pool.is_owned_by(self.context) {
            return Err(invalid(
                "classic coefficient pool must belong to the allocation context",
            ));
        }
        let output = pool.take(checked_bytes::<f32>(output_words)?)?;
        self.context
            .memset_d32_async(pooled_device_buffer(&output)?, 0, output_words)?;
        Ok(output)
    }

    /// Decode classic Tier-1 code-blocks into one or more device coefficient planes.
    #[doc(hidden)]
    pub fn decode_classic_codeblocks_multi_with_resources_and_pool(
        &self,
        resources: &CudaHtj2kDecodeResources,
        targets: &[CudaClassicDecodeTarget<'_>],
        pool: &CudaBufferPool,
        live_host_bytes: usize,
    ) -> Result<Vec<CudaClassicStatus>, CudaError> {
        self.decode_classic_codeblocks_multi_with_resources_and_pool_timed(
            resources,
            targets,
            pool,
            live_host_bytes,
            false,
        )
        .map(|(statuses, _)| statuses)
    }

    /// Enqueue classic Tier-1 decoding and defer its single status transfer.
    ///
    /// # Safety
    ///
    /// Payload, table, coefficient, and pool owners must remain live and
    /// unmodified until the returned guard is finished or dropped. Targets
    /// must remain pairwise disjoint and confined to this context's default
    /// stream until completion.
    #[doc(hidden)]
    pub unsafe fn decode_classic_codeblocks_multi_enqueue_with_resources_and_pool(
        &self,
        resources: &CudaHtj2kDecodeResources,
        tables: &CudaClassicDecodeTableResources,
        targets: &[CudaClassicDecodeTarget<'_>],
        pool: &CudaBufferPool,
        live_host_bytes: usize,
    ) -> Result<CudaQueuedClassicDecode, CudaError> {
        validate_classic_launch_owners(self.context, resources, targets, pool)?;
        if !tables.is_owned_by(self.context) {
            return Err(invalid(
                "classic Tier-1 tables must belong to the decode context",
            ));
        }
        let mut host_budget =
            HostPhaseBudget::with_live_bytes("CUDA queued classic Tier-1 owners", live_host_bytes)?;
        let prepared = prepare_classic_decode(resources.payload_len(), targets, &mut host_budget)?;
        if prepared.jobs.is_empty() {
            return Ok(CudaQueuedClassicDecode::empty(self.context));
        }
        let spec = classic_kernel_spec()?;
        let payload = resources.payload_buffer()?;
        let jobs = pool.upload_pinned(classic_jobs_as_bytes(&prepared.jobs))?;
        let segments = pool.upload_pinned(classic_segments_as_bytes(&prepared.segments))?;
        let scratch = pool.take(checked_bytes::<u32>(prepared.scratch_words)?)?;
        let statuses = pool.take(checked_bytes::<CudaClassicStatus>(prepared.jobs.len())?)?;
        let mut queued_resources = host_budget.try_vec_with_capacity(4)?;
        queued_resources.push(jobs);
        queued_resources.push(segments);
        queued_resources.push(scratch);
        queued_resources.push(statuses);
        let mut finish_budget = HostPhaseBudget::with_live_bytes(
            "CUDA queued classic Tier-1 retained metadata",
            live_host_bytes,
        )?;
        finish_budget.account_vec(&queued_resources)?;

        let mut payload_ptr = payload.device_ptr();
        let mut jobs_ptr = pooled_device_buffer(&queued_resources[0])?.device_ptr();
        let mut segments_ptr = pooled_device_buffer(&queued_resources[1])?.device_ptr();
        let mut tables_ptr = tables.inner.tables.device_ptr();
        let mut statuses_ptr =
            pooled_device_buffer(&queued_resources[STATUS_RESOURCE_INDEX])?.device_ptr();
        let mut scratch_ptr = pooled_device_buffer(&queued_resources[2])?.device_ptr();
        let mut params = cuda_kernel_params!(
            payload_ptr,
            jobs_ptr,
            segments_ptr,
            tables_ptr,
            statuses_ptr,
            scratch_ptr
        );
        let geometry = classic_launch_geometry(prepared.jobs.len())?;
        let execution = CudaExecutionStats::new(1, 0, 1, false);
        let queued = self
            .context
            .with_nvtx_range("j2k.classic.decode.tier1.batch", || {
                // SAFETY: ABI values remain live through submission; all device
                // pointers were validated for this context, and pooled owners
                // transfer into the returned completion guard.
                unsafe {
                    self.context.launch_compiled_kernel_queued(
                        spec,
                        geometry,
                        &mut params,
                        pool,
                        queued_resources,
                        execution,
                    )
                }
            })?;
        Ok(CudaQueuedClassicDecode {
            context: self.context.clone(),
            queued: Some(queued),
            status_index: STATUS_RESOURCE_INDEX,
            status_count: prepared.jobs.len(),
            execution,
            timings: CudaClassicDecodeStageTimings::default(),
            finish_host_live_bytes: finish_budget.live_bytes(),
        })
    }

    /// Decode classic Tier-1 code-blocks and return optional stage timings.
    #[doc(hidden)]
    pub fn decode_classic_codeblocks_multi_with_resources_and_pool_timed(
        &self,
        resources: &CudaHtj2kDecodeResources,
        targets: &[CudaClassicDecodeTarget<'_>],
        pool: &CudaBufferPool,
        live_host_bytes: usize,
        collect_stage_timings: bool,
    ) -> Result<(Vec<CudaClassicStatus>, CudaClassicDecodeStageTimings), CudaError> {
        validate_classic_launch_owners(self.context, resources, targets, pool)?;
        let mut host_budget =
            HostPhaseBudget::with_live_bytes("CUDA classic Tier-1 launch owners", live_host_bytes)?;
        let prepared = prepare_classic_decode(resources.payload_len(), targets, &mut host_budget)?;
        if prepared.jobs.is_empty() {
            return Ok((Vec::new(), CudaClassicDecodeStageTimings::default()));
        }
        let spec = classic_kernel_spec()?;
        let payload = resources.payload_buffer()?;
        let job_upload_start = collect_stage_timings.then(Instant::now);
        let jobs = pool.upload_pinned(classic_jobs_as_bytes(&prepared.jobs))?;
        let segments = pool.upload_pinned(classic_segments_as_bytes(&prepared.segments))?;
        let job_upload_us = job_upload_start.map_or(0, |start| start.elapsed().as_micros());
        let table_upload_start = collect_stage_timings.then(Instant::now);
        let tables = pool.upload_pinned(classic_tables_as_bytes(&CLASSIC_KERNEL_TABLES))?;
        let table_upload_us = table_upload_start.map_or(0, |start| start.elapsed().as_micros());
        let statuses = pool.take(checked_bytes::<CudaClassicStatus>(prepared.jobs.len())?)?;
        let scratch = pool.take(checked_bytes::<u32>(prepared.scratch_words)?)?;
        let mut retained = host_budget.try_vec_with_capacity(5)?;
        retained.push(jobs);
        retained.push(segments);
        retained.push(tables);
        retained.push(statuses);
        retained.push(scratch);
        let mut payload_ptr = payload.device_ptr();
        let mut jobs_ptr = pooled_device_buffer(&retained[0])?.device_ptr();
        let mut segments_ptr = pooled_device_buffer(&retained[1])?.device_ptr();
        let mut tables_ptr = pooled_device_buffer(&retained[2])?.device_ptr();
        let mut statuses_ptr = pooled_device_buffer(&retained[STATUS_RESOURCE_INDEX])?.device_ptr();
        let mut scratch_ptr = pooled_device_buffer(&retained[4])?.device_ptr();
        let mut params = cuda_kernel_params!(
            payload_ptr,
            jobs_ptr,
            segments_ptr,
            tables_ptr,
            statuses_ptr,
            scratch_ptr
        );
        let geometry = classic_launch_geometry(prepared.jobs.len())?;
        let execution = CudaExecutionStats::new(1, 0, 1, false);
        // SAFETY: parameter values and external payload/target owners remain
        // live through completion; pooled owners move into the queued guard.
        let (completed_resources, kernel_us) = unsafe {
            launch_classic_to_completion(
                self.context,
                spec,
                geometry,
                &mut params,
                pool,
                retained,
                execution,
                collect_stage_timings,
            )?
        };

        let mut host_statuses =
            host_budget.try_vec_filled(prepared.jobs.len(), CudaClassicStatus::default())?;
        let status_d2h_start = collect_stage_timings.then(Instant::now);
        completed_resources[STATUS_RESOURCE_INDEX]
            .copy_to_host(classic_statuses_as_bytes_mut(&mut host_statuses))?;
        let status_d2h_us = status_d2h_start.map_or(0, |start| start.elapsed().as_micros());
        if let Some((index, status)) = host_statuses
            .iter()
            .copied()
            .enumerate()
            .find(|(_, status)| status.code != 0)
        {
            return Err(CudaError::KernelStatus {
                kernel: CLASSIC_KERNEL_NAME,
                code: status.code,
                detail: ((u32::try_from(index).unwrap_or(u32::MAX)) << 8) | (status.detail & 0xff),
            });
        }
        Ok((
            host_statuses,
            CudaClassicDecodeStageTimings {
                job_upload_us,
                table_upload_us,
                kernel_us,
                status_d2h_us,
            },
        ))
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "the low-level launch boundary keeps its validated ABI inputs explicit"
)]
unsafe fn launch_classic_to_completion(
    context: &CudaContext,
    spec: CudaKernelSpec,
    geometry: CudaLaunchGeometry,
    params: &mut [*mut c_void],
    pool: &CudaBufferPool,
    resources: Vec<CudaPooledDeviceBuffer>,
    execution: CudaExecutionStats,
    collect_stage_timings: bool,
) -> Result<(Vec<CudaPooledDeviceBuffer>, u128), CudaError> {
    let mut resources = Some(resources);
    let mut launch = || {
        let retained = resources.take().ok_or_else(|| CudaError::StatePoisoned {
            message: "classic launch resources were submitted more than once".to_string(),
        })?;
        // SAFETY: inherited from this helper's caller; pooled owners transfer
        // into the returned completion guard.
        unsafe {
            context.launch_compiled_kernel_queued(spec, geometry, params, pool, retained, execution)
        }
    };
    if collect_stage_timings {
        let (queued, elapsed_us) =
            context.time_default_stream_named_us("j2k.classic.decode.tier1.batch", &mut launch)?;
        // SAFETY: the timing helper synchronized its end event after the
        // queued default-stream kernel.
        let (resources, _) = unsafe { queued.finish_with_resources_after_completion()? };
        Ok((resources, elapsed_us))
    } else {
        let queued = context.with_nvtx_range("j2k.classic.decode.tier1.batch", launch)?;
        let (resources, _) = queued.finish_with_resources()?;
        Ok((resources, 0))
    }
}

fn pooled_device_buffer(buffer: &CudaPooledDeviceBuffer) -> Result<&CudaDeviceBuffer, CudaError> {
    buffer
        .as_device_buffer()
        .ok_or_else(|| CudaError::StatePoisoned {
            message: "classic pooled CUDA buffer lost its device allocation".to_string(),
        })
}

pub(super) fn classic_launch_geometry(job_count: usize) -> Result<CudaLaunchGeometry, CudaError> {
    let jobs =
        u32::try_from(job_count).map_err(|_| CudaError::LengthTooLarge { len: job_count })?;
    CudaLaunchGeometry::new((jobs, 1, 1), (CLASSIC_DECODE_CODEBLOCK_THREADS, 1, 1))
        .ok_or(CudaError::LengthTooLarge { len: job_count })
}

fn classic_kernel_spec() -> Result<CudaKernelSpec, CudaError> {
    ensure_j2k_classic_decode_ptx_built()?;
    CudaKernelSpec::new(
        "j2k-classic-decode",
        classic_decode_ptx(),
        CLASSIC_KERNEL_ENTRYPOINT,
    )
}

#[cfg(feature = "cuda-oxide-j2k-classic-decode")]
fn classic_decode_ptx() -> &'static [u8] {
    include_bytes!(concat!(
        env!("OUT_DIR"),
        "/cuda_oxide_j2k_classic_decode.ptx"
    ))
}

#[cfg(not(feature = "cuda-oxide-j2k-classic-decode"))]
fn classic_decode_ptx() -> &'static [u8] {
    b".version 7.0\n.target sm_52\n.address_size 64\n\0"
}
