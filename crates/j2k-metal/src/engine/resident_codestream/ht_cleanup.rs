// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(target_os = "macos")]
use crate::metal_types::prelude::*;

#[cfg(target_os = "macos")]
use super::{
    checked_buffer_read, checked_buffer_slice, commit_and_wait_metal, copied_slice_buffer,
    decode_ht_status_error, dispatch_single_thread, dispatch_zero_u32_buffer_in_encoder,
    ht_batch_output_word_count, ht_output_word_count, new_command_buffer,
    new_compute_command_encoder, size_of, zeroed_shared_buffer, Buffer, ComputeCommandEncoderRef,
    Error, J2kHtCleanupBatchJob, J2kHtCleanupParams, J2kHtRepeatedBatchParams, J2kHtStatus,
    MetalRuntime, J2K_HT_STATUS_OK,
};
#[cfg(target_os = "macos")]
use crate::engine::abi::J2kHtVlcDispatchParams;
#[cfg(target_os = "macos")]
use crate::engine::decode_dispatch::MetalHtPipelineKind;

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
pub(in crate::engine) struct HtCleanupBatchDispatch<'a> {
    pub(in crate::engine) coded_data: &'a Buffer,
    pub(in crate::engine) jobs: &'a Buffer,
    pub(in crate::engine) job_count: usize,
    pub(in crate::engine) decoded: &'a Buffer,
    pub(in crate::engine) status_buffer: &'a Buffer,
    pub(in crate::engine) status_offset_bytes: u64,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
pub(in crate::engine) struct HtCleanupRepeatedBatchDispatch<'a> {
    pub(in crate::engine) coded_data: &'a Buffer,
    pub(in crate::engine) jobs: &'a Buffer,
    pub(in crate::engine) base_job_count: usize,
    pub(in crate::engine) repeated: J2kHtRepeatedBatchParams,
    pub(in crate::engine) decoded: &'a Buffer,
    pub(in crate::engine) status_buffer: &'a Buffer,
    pub(in crate::engine) status_offset_bytes: u64,
}

/// SIMD width, code blocks per `MagSgn` threadgroup, and VLC threads per
/// threadgroup of the two-kernel cleanup decode in `ht_cleanup_simd.metal`;
/// all three must match the shader constants.
#[cfg(target_os = "macos")]
const HT_SIMD_LANES: usize = 32;
#[cfg(target_os = "macos")]
const HT_SIMD_BLOCKS_PER_GROUP: usize = 4;
#[cfg(target_os = "macos")]
const HT_VLC_THREADS_PER_GROUP: usize = 32;
/// SIMD groups the VLC kernel aims to keep in flight. Each block's MEL/VLC
/// walk is one long serial chain: packing many blocks into a SIMD group adds
/// divergence, while spreading them over many groups makes the walk
/// issue-bound. On a 16-core M4 Pro, 40-150 groups measured best for 64 to
/// 3,072 blocks.
#[cfg(target_os = "macos")]
const HT_VLC_TARGET_SIMD_GROUPS: usize = 96;

#[cfg(target_os = "macos")]
fn ht_vlc_lanes_per_simd(total_jobs: usize) -> usize {
    total_jobs
        .div_ceil(HT_VLC_TARGET_SIMD_GROUPS)
        .next_power_of_two()
        .clamp(1, HT_SIMD_LANES)
}

/// Cleanup-only decode split into a thread-per-block MEL/VLC kernel and a
/// SIMD-group-per-block `MagSgn` kernel.
#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
struct CooperativeHtCleanup<'a> {
    vlc: &'a crate::metal_types::ComputePipelineState,
    magsgn: &'a crate::metal_types::ComputePipelineState,
    magsgn_threads_per_group: usize,
}

#[cfg(target_os = "macos")]
impl<'a> CooperativeHtCleanup<'a> {
    /// `None` when the device's SIMD width is not the one the kernels assume.
    fn new(
        vlc: &'a crate::metal_types::ComputePipelineState,
        magsgn: &'a crate::metal_types::ComputePipelineState,
        job_count: usize,
    ) -> Option<Self> {
        if magsgn.threadExecutionWidth() != HT_SIMD_LANES
            || vlc.maxTotalThreadsPerThreadgroup() < HT_VLC_THREADS_PER_GROUP
        {
            return None;
        }
        let blocks = HT_SIMD_BLOCKS_PER_GROUP
            .min(magsgn.maxTotalThreadsPerThreadgroup() / HT_SIMD_LANES)
            .min(job_count);
        (blocks > 0).then_some(Self {
            vlc,
            magsgn,
            magsgn_threads_per_group: blocks * HT_SIMD_LANES,
        })
    }

    /// Encodes both kernels over buffers the caller has already bound. The
    /// `MagSgn` kernel reads the per-quad VLC words the first kernel leaves in
    /// `decoded`, and may overwrite `status` with a `MagSgn` failure.
    fn encode(
        self,
        encoder: &ComputeCommandEncoderRef,
        job_count: usize,
        batch_count: u32,
        decoded: &Buffer,
        status: &Buffer,
    ) -> Result<(), Error> {
        let total_jobs =
            job_count
                .checked_mul(batch_count as usize)
                .ok_or_else(|| Error::MetalKernel {
                    message: "HTJ2K Metal cooperative cleanup job count overflow".to_string(),
                })?;
        let lanes_per_simd = ht_vlc_lanes_per_simd(total_jobs);
        let dispatch = J2kHtVlcDispatchParams {
            job_count: u32::try_from(job_count).map_err(|_| Error::MetalKernel {
                message: "HTJ2K Metal cooperative cleanup job count exceeds u32".to_string(),
            })?,
            lanes_per_simd: u32::try_from(lanes_per_simd).map_err(|_| Error::MetalKernel {
                message: "HTJ2K Metal cooperative cleanup lane count exceeds u32".to_string(),
            })?,
        };
        let vlc_threads = job_count.div_ceil(lanes_per_simd) * HT_SIMD_LANES;
        encoder.set_bytes::<J2kHtVlcDispatchParams>(9, &dispatch);
        encoder.setComputePipelineState(self.vlc);
        encoder.dispatchThreads_threadsPerThreadgroup(
            j2k_metal_support::mtl_size(vlc_threads as u64, u64::from(batch_count), 1),
            j2k_metal_support::mtl_size(HT_VLC_THREADS_PER_GROUP as u64, 1, 1),
        );
        encoder.memory_barrier_with_resources(&[decoded, status]);
        #[cfg(test)]
        if !crate::engine::test_counters::decode_stage_enabled(
            crate::engine::test_counters::DecodeStageLimit::Tier1,
        ) {
            return Ok(());
        }
        encoder.setComputePipelineState(self.magsgn);
        encoder.dispatchThreads_threadsPerThreadgroup(
            j2k_metal_support::mtl_size(
                (job_count * HT_SIMD_LANES) as u64,
                u64::from(batch_count),
                1,
            ),
            j2k_metal_support::mtl_size(self.magsgn_threads_per_group as u64, 1, 1),
        );
        Ok(())
    }
}

#[cfg(target_os = "macos")]
pub(in crate::engine) fn dispatch_ht_cleanup(
    runtime: &MetalRuntime,
    coded_data: &[u8],
    params: J2kHtCleanupParams,
    decoded: &Buffer,
) -> Result<(), Error> {
    let input = copied_slice_buffer(&runtime.device, coded_data)?;
    let status_buffer = zeroed_shared_buffer(&runtime.device, size_of::<J2kHtStatus>())?;

    let command_buffer = new_command_buffer(&runtime.queue)?;
    let encoder = new_compute_command_encoder(&command_buffer)?;
    dispatch_zero_u32_buffer_in_encoder(
        runtime,
        &encoder,
        decoded,
        ht_output_word_count(
            params.output_offset,
            params.output_stride,
            params.width,
            params.height,
        )?,
    )?;
    encoder.memory_barrier_with_resources(&[decoded]);
    let kernels = runtime.decode()?;
    let pipeline = if params.number_of_coding_passes == 1 {
        &kernels.ht_cleanup_cleanup_only
    } else {
        &kernels.ht_cleanup
    };
    encoder.setComputePipelineState(pipeline);
    encoder.set_buffer(0, Some(&input), 0);
    encoder.set_buffer(1, Some(decoded), 0);
    encoder.set_bytes::<J2kHtCleanupParams>(2, &params);
    encoder.set_buffer(3, Some(&runtime.decode()?.ht_vlc_table0), 0);
    encoder.set_buffer(4, Some(&runtime.decode()?.ht_vlc_table1), 0);
    encoder.set_buffer(5, Some(&runtime.decode()?.ht_uvlc_table0), 0);
    encoder.set_buffer(6, Some(&runtime.decode()?.ht_uvlc_table1), 0);
    encoder.set_buffer(7, Some(&status_buffer), 0);
    dispatch_single_thread(&encoder);
    encoder.endEncoding();
    commit_and_wait_metal(&command_buffer)?;

    let status = checked_buffer_read::<J2kHtStatus>(&status_buffer, "HT cleanup status")?;
    if status.code != J2K_HT_STATUS_OK {
        return Err(decode_ht_status_error(status));
    }

    Ok(())
}

#[cfg(target_os = "macos")]
pub(in crate::engine) fn dispatch_ht_cleanup_batched(
    runtime: &MetalRuntime,
    coded_data: &[u8],
    jobs: &[J2kHtCleanupBatchJob],
    decoded: &Buffer,
) -> Result<(), Error> {
    let input = copied_slice_buffer(&runtime.device, coded_data)?;
    let jobs_buffer = copied_slice_buffer(&runtime.device, jobs)?;
    let status_buffer = zeroed_shared_buffer(
        &runtime.device,
        jobs.len().max(1) * size_of::<J2kHtStatus>(),
    )?;

    let command_buffer = new_command_buffer(&runtime.queue)?;
    let encoder = new_compute_command_encoder(&command_buffer)?;
    dispatch_zero_u32_buffer_in_encoder(
        runtime,
        &encoder,
        decoded,
        ht_batch_output_word_count(jobs)?,
    )?;
    encoder.memory_barrier_with_resources(&[decoded]);
    let kernels = runtime.decode()?;
    let cleanup_only = jobs.iter().all(|job| job.number_of_coding_passes == 1);
    let cooperative = cleanup_only
        .then(|| {
            CooperativeHtCleanup::new(
                &kernels.ht_cleanup_vlc_batched,
                &kernels.ht_cleanup_magsgn_batched,
                jobs.len(),
            )
        })
        .flatten();
    let pipeline = match (cooperative, cleanup_only) {
        (Some(cooperative), _) => cooperative.vlc,
        (None, true) => &kernels.ht_cleanup_batched_cleanup_only,
        (None, false) => &kernels.ht_cleanup_batched,
    };
    encoder.setComputePipelineState(pipeline);
    encoder.set_buffer(0, Some(&input), 0);
    encoder.set_buffer(1, Some(decoded), 0);
    encoder.set_buffer(2, Some(&jobs_buffer), 0);
    encoder.set_buffer(3, Some(&kernels.ht_vlc_table0), 0);
    encoder.set_buffer(4, Some(&kernels.ht_vlc_table1), 0);
    encoder.set_buffer(5, Some(&kernels.ht_uvlc_table0), 0);
    encoder.set_buffer(6, Some(&kernels.ht_uvlc_table1), 0);
    encoder.set_buffer(7, Some(&status_buffer), 0);
    if let Some(cooperative) = cooperative {
        cooperative.encode(&encoder, jobs.len(), 1, decoded, &status_buffer)?;
    } else {
        let width = pipeline.threadExecutionWidth().max(1).min(jobs.len());
        encoder.dispatchThreads_threadsPerThreadgroup(
            j2k_metal_support::mtl_size(jobs.len() as u64, 1, 1),
            j2k_metal_support::mtl_size(width as u64, 1, 1),
        );
    }
    encoder.endEncoding();
    commit_and_wait_metal(&command_buffer)?;

    let statuses =
        checked_buffer_slice::<J2kHtStatus>(&status_buffer, jobs.len(), "HT cleanup statuses")?;
    if let Some(status) = statuses
        .iter()
        .copied()
        .find(|status| status.code != J2K_HT_STATUS_OK)
    {
        return Err(decode_ht_status_error(status));
    }

    Ok(())
}

#[cfg(all(test, target_os = "macos"))]
mod tests;

#[cfg(target_os = "macos")]
pub(in crate::engine) fn dispatch_ht_cleanup_batched_in_encoder_with_status_offset(
    kernels: &crate::engine::runtime::DecodeKernels,
    encoder: &ComputeCommandEncoderRef,
    pipeline_kind: MetalHtPipelineKind,
    dispatch: HtCleanupBatchDispatch<'_>,
) -> Result<(), Error> {
    let cooperative = (pipeline_kind == MetalHtPipelineKind::CleanupOnly)
        .then(|| {
            CooperativeHtCleanup::new(
                &kernels.ht_cleanup_vlc_batched,
                &kernels.ht_cleanup_magsgn_batched,
                dispatch.job_count,
            )
        })
        .flatten();
    let pipeline = match (pipeline_kind, cooperative) {
        (MetalHtPipelineKind::CleanupOnly, Some(cooperative)) => cooperative.vlc,
        (MetalHtPipelineKind::CleanupOnly, None) => &kernels.ht_cleanup_batched_cleanup_only,
        (MetalHtPipelineKind::SigProp, _) => &kernels.ht_cleanup_batched_sigprop,
        (MetalHtPipelineKind::MagRef, _) => &kernels.ht_cleanup_batched_magref,
    };
    encoder.setComputePipelineState(pipeline);
    encoder.set_buffer(0, Some(dispatch.coded_data), 0);
    encoder.set_buffer(1, Some(dispatch.decoded), 0);
    encoder.set_buffer(2, Some(dispatch.jobs), 0);
    encoder.set_buffer(3, Some(&kernels.ht_vlc_table0), 0);
    encoder.set_buffer(4, Some(&kernels.ht_vlc_table1), 0);
    encoder.set_buffer(5, Some(&kernels.ht_uvlc_table0), 0);
    encoder.set_buffer(6, Some(&kernels.ht_uvlc_table1), 0);
    encoder.set_buffer(
        7,
        Some(dispatch.status_buffer),
        dispatch.status_offset_bytes,
    );
    if let Some(cooperative) = cooperative {
        return cooperative.encode(
            encoder,
            dispatch.job_count,
            1,
            dispatch.decoded,
            dispatch.status_buffer,
        );
    }
    let width = pipeline
        .threadExecutionWidth()
        .max(1)
        .min(dispatch.job_count);
    encoder.dispatchThreads_threadsPerThreadgroup(
        j2k_metal_support::mtl_size(dispatch.job_count as u64, 1, 1),
        j2k_metal_support::mtl_size(width as u64, 1, 1),
    );
    Ok(())
}

#[cfg(target_os = "macos")]
pub(in crate::engine) fn dispatch_ht_cleanup_repeated_batched_in_encoder_with_status_offset(
    kernels: &crate::engine::runtime::DecodeKernels,
    encoder: &ComputeCommandEncoderRef,
    pipeline_kind: MetalHtPipelineKind,
    dispatch: HtCleanupRepeatedBatchDispatch<'_>,
) -> Result<(), Error> {
    let cooperative = (pipeline_kind == MetalHtPipelineKind::CleanupOnly)
        .then(|| {
            CooperativeHtCleanup::new(
                &kernels.ht_cleanup_vlc_repeated_batched,
                &kernels.ht_cleanup_magsgn_repeated_batched,
                dispatch.base_job_count,
            )
        })
        .flatten();
    let pipeline = match (pipeline_kind, cooperative) {
        (MetalHtPipelineKind::CleanupOnly, Some(cooperative)) => cooperative.vlc,
        (MetalHtPipelineKind::CleanupOnly, None) => {
            &kernels.ht_cleanup_repeated_batched_cleanup_only
        }
        (MetalHtPipelineKind::SigProp, _) => &kernels.ht_cleanup_repeated_batched_sigprop,
        (MetalHtPipelineKind::MagRef, _) => &kernels.ht_cleanup_repeated_batched_magref,
    };
    encoder.setComputePipelineState(pipeline);
    encoder.set_buffer(0, Some(dispatch.coded_data), 0);
    encoder.set_buffer(1, Some(dispatch.decoded), 0);
    encoder.set_buffer(2, Some(dispatch.jobs), 0);
    encoder.set_bytes::<J2kHtRepeatedBatchParams>(3, &dispatch.repeated);
    encoder.set_buffer(4, Some(&kernels.ht_vlc_table0), 0);
    encoder.set_buffer(5, Some(&kernels.ht_vlc_table1), 0);
    encoder.set_buffer(6, Some(&kernels.ht_uvlc_table0), 0);
    encoder.set_buffer(7, Some(&kernels.ht_uvlc_table1), 0);
    encoder.set_buffer(
        8,
        Some(dispatch.status_buffer),
        dispatch.status_offset_bytes,
    );
    if let Some(cooperative) = cooperative {
        return cooperative.encode(
            encoder,
            dispatch.base_job_count,
            dispatch.repeated.batch_count,
            dispatch.decoded,
            dispatch.status_buffer,
        );
    }
    let width = pipeline
        .threadExecutionWidth()
        .max(1)
        .min(dispatch.base_job_count);
    encoder.dispatchThreads_threadsPerThreadgroup(
        j2k_metal_support::mtl_size(
            dispatch.base_job_count as u64,
            u64::from(dispatch.repeated.batch_count),
            1,
        ),
        j2k_metal_support::mtl_size(width as u64, 1, 1),
    );
    Ok(())
}
