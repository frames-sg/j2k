// SPDX-License-Identifier: MIT OR Apache-2.0

use std::time::Instant;

use super::abi::{
    CudaClassicDecodeStageTimings, CudaClassicDecodeTarget, CudaClassicStatus,
    CLASSIC_KERNEL_TABLES,
};
use super::prepare::{
    checked_bytes, invalid, prepare_classic_decode, validate_classic_launch_owners,
};
use crate::{
    allocation::HostPhaseBudget,
    bytes::{
        classic_jobs_as_bytes, classic_segments_as_bytes, classic_statuses_as_bytes_mut,
        classic_tables_as_bytes,
    },
    context::CudaContext,
    error::{select_resource_release_error, CudaError},
    execution::cuda_kernel_param,
    htj2k_decode::CudaHtj2kDecodeResources,
    kernels::{j2k_classic_codeblock_launch_geometry, CudaKernel},
    memory::{pooled_device_buffer, CudaBufferPool},
};

const CLASSIC_KERNEL_NAME: &str = "j2k_decode_classic_codeblocks_multi";

impl CudaContext {
    /// Allocate and clear one classic Tier-1 coefficient plane.
    #[doc(hidden)]
    pub fn allocate_classic_coefficients_with_pool(
        &self,
        output_words: usize,
        pool: &CudaBufferPool,
    ) -> Result<crate::memory::CudaPooledDeviceBuffer, CudaError> {
        if !pool.is_owned_by(self) {
            return Err(invalid(
                "classic coefficient pool must belong to the allocation context",
            ));
        }
        let bytes = checked_bytes::<f32>(output_words)?;
        let output = pool.take(bytes)?;
        self.memset_d32(pooled_device_buffer(&output)?, 0, output_words)?;
        self.synchronize()?;
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
        validate_classic_launch_owners(self, resources, targets, pool)?;
        let mut host_budget =
            HostPhaseBudget::with_live_bytes("CUDA classic Tier-1 launch owners", live_host_bytes)?;
        let prepared = prepare_classic_decode(resources.payload_len, targets, &mut host_budget)?;
        if prepared.jobs.is_empty() {
            return Ok((Vec::new(), CudaClassicDecodeStageTimings::default()));
        }
        let payload = resources.payload.buffer()?;
        let job_upload_start = collect_stage_timings.then(Instant::now);
        let jobs = pool.upload_pinned(classic_jobs_as_bytes(&prepared.jobs))?;
        let segments = pool.upload_pinned(classic_segments_as_bytes(&prepared.segments))?;
        let job_upload_us = job_upload_start.map_or(0, |start| start.elapsed().as_micros());
        let table_upload_start = collect_stage_timings.then(Instant::now);
        let tables = pool.upload_pinned(classic_tables_as_bytes(&CLASSIC_KERNEL_TABLES))?;
        let table_upload_us = table_upload_start.map_or(0, |start| start.elapsed().as_micros());
        let statuses = pool.take(checked_bytes::<CudaClassicStatus>(prepared.jobs.len())?)?;
        let scratch = pool.take(checked_bytes::<u32>(prepared.scratch_words)?)?;

        let mut payload_ptr = payload.device_ptr();
        let mut jobs_ptr = pooled_device_buffer(&jobs)?.device_ptr();
        let mut segments_ptr = pooled_device_buffer(&segments)?.device_ptr();
        let mut tables_ptr = pooled_device_buffer(&tables)?.device_ptr();
        let mut statuses_ptr = pooled_device_buffer(&statuses)?.device_ptr();
        let mut scratch_ptr = pooled_device_buffer(&scratch)?.device_ptr();
        let mut params = cuda_kernel_params!(
            payload_ptr,
            jobs_ptr,
            segments_ptr,
            tables_ptr,
            statuses_ptr,
            scratch_ptr
        );
        let geometry = j2k_classic_codeblock_launch_geometry(prepared.jobs.len()).ok_or(
            CudaError::LengthTooLarge {
                len: prepared.jobs.len(),
            },
        )?;
        let function = self.inner.cuda_oxide_j2k_classic_decode_kernel_function(
            CudaKernel::J2kClassicDecodeCodeblocksMulti,
        )?;
        let pool_reuse_guard = pool.defer_reuse()?;
        let kernel_result = if collect_stage_timings {
            self.time_default_stream_named_us("j2k.classic.decode.tier1.batch", || {
                self.launch_kernel(function, geometry, &mut params)
            })
            .map(|((), elapsed_us)| elapsed_us)
        } else {
            self.with_nvtx_range("j2k.classic.decode.tier1.batch", || {
                self.launch_kernel(function, geometry, &mut params)
            })
            .map(|()| 0)
        };
        let kernel_us = match kernel_result {
            Ok(elapsed_us) => elapsed_us,
            Err(error) => return pool_reuse_guard.synchronize_then_error(error),
        };

        let mut host_statuses =
            host_budget.try_vec_filled(prepared.jobs.len(), CudaClassicStatus::default())?;
        let status_d2h_start = collect_stage_timings.then(Instant::now);
        if let Err(error) = statuses.copy_to_host(classic_statuses_as_bytes_mut(&mut host_statuses))
        {
            return pool_reuse_guard.release_after_recoverable_operation_error(error);
        }
        let status_d2h_us = status_d2h_start.map_or(0, |start| start.elapsed().as_micros());
        let release_result = pool_reuse_guard.release();
        let status_error = host_statuses
            .iter()
            .copied()
            .enumerate()
            .find(|(_, status)| status.code != 0)
            .map(|(index, status)| CudaError::KernelStatus {
                kernel: CLASSIC_KERNEL_NAME,
                code: status.code,
                detail: ((u32::try_from(index).unwrap_or(u32::MAX)) << 8) | (status.detail & 0xff),
            });
        match (status_error, release_result) {
            (Some(primary), Err(release)) => Err(select_resource_release_error(primary, release)),
            (Some(error), Ok(())) | (None, Err(error)) => Err(error),
            (None, Ok(())) => Ok((
                host_statuses,
                CudaClassicDecodeStageTimings {
                    job_upload_us,
                    table_upload_us,
                    kernel_us,
                    status_d2h_us,
                },
            )),
        }
    }
}
