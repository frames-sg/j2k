use crate::{
    bytes::{
        htj2k_cleanup_multi_jobs_as_bytes, htj2k_dequantize_jobs_as_bytes, htj2k_jobs_as_bytes,
        htj2k_statuses_as_bytes_mut, htj2k_statuses_byte_len, u16_slice_as_bytes,
    },
    context::CudaContext,
    error::CudaError,
    execution::{cuda_kernel_param, CudaExecutionStats, CudaLaunchMode},
    kernels::{
        htj2k_codeblock_launch_geometry, htj2k_codeblock_sample_launch_geometry, CudaKernel,
    },
    memory::{pooled_device_buffer, CudaBufferPool, CudaDeviceBuffer, CudaPooledDeviceBuffer},
};
use std::{os::raw::c_uint, sync::Arc, time::Instant};

/// HTJ2K code-block decode job consumed by the CUDA entropy kernel launcher.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CudaHtj2kCodeBlockJob {
    /// Byte offset into the contiguous compressed payload buffer.
    pub payload_offset: u64,
    /// Code-block width in coefficients.
    pub width: u32,
    /// Code-block height in coefficients.
    pub height: u32,
    /// Combined cleanup/refinement byte length.
    pub payload_len: u32,
    /// Cleanup segment length in bytes.
    pub cleanup_length: u32,
    /// Refinement segment length in bytes.
    pub refinement_length: u32,
    /// Missing most-significant bit planes.
    pub missing_bit_planes: u8,
    /// Total coded bitplanes for this code block's sub-band.
    pub num_bitplanes: u8,
    /// Number of HT coding passes present.
    pub number_of_coding_passes: u8,
    /// Output row stride, in coefficients.
    pub output_stride: u32,
    /// Output offset, in coefficients, into the destination plane.
    pub output_offset: u32,
    /// Dequantization multiplier for decoded coefficient values.
    pub dequantization_step: f32,
    /// Vertically causal context mode flag.
    pub stripe_causal: bool,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct CudaHtj2kCodeBlockKernelJob {
    pub(crate) coded_offset: u32,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) coded_len: u32,
    pub(crate) cleanup_length: u32,
    pub(crate) refinement_length: u32,
    pub(crate) missing_msbs: u32,
    pub(crate) num_bitplanes: u32,
    pub(crate) number_of_coding_passes: u32,
    pub(crate) output_stride: u32,
    pub(crate) output_offset: u32,
    pub(crate) dequantization_step: f32,
    pub(crate) stripe_causal: u32,
}

/// One output buffer and its code-block jobs for batched HTJ2K cleanup decode.
#[derive(Clone, Copy, Debug)]
pub struct CudaHtj2kCleanupTarget<'a> {
    /// Device buffer receiving decoded integer coefficient bits.
    pub coefficients: &'a CudaDeviceBuffer,
    /// Code-block jobs that write into `coefficients`.
    pub jobs: &'a [CudaHtj2kCodeBlockJob],
    /// Number of coefficient words available in `coefficients`.
    pub output_words: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct CudaHtj2kCleanupMultiKernelJob {
    pub(crate) output_ptr: u64,
    pub(crate) coded_offset: u32,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) coded_len: u32,
    pub(crate) cleanup_length: u32,
    pub(crate) refinement_length: u32,
    pub(crate) missing_msbs: u32,
    pub(crate) num_bitplanes: u32,
    pub(crate) number_of_coding_passes: u32,
    pub(crate) output_stride: u32,
    pub(crate) output_offset: u32,
    pub(crate) dequantization_step: f32,
    pub(crate) stripe_causal: u32,
}

/// One output buffer and its code-block jobs for batched HTJ2K dequantization.
#[derive(Clone, Copy, Debug)]
pub struct CudaHtj2kDequantizeTarget<'a> {
    /// Device buffer containing decoded integer coefficient bits.
    pub coefficients: &'a CudaDeviceBuffer,
    /// Code-block jobs that write into `coefficients`.
    pub jobs: &'a [CudaHtj2kCodeBlockJob],
    /// Number of coefficient words available in `coefficients`.
    pub output_words: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct CudaHtj2kDequantizeKernelJob {
    pub(crate) output_ptr: u64,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) output_stride: u32,
    pub(crate) output_offset: u32,
    pub(crate) num_bitplanes: u32,
    pub(crate) reserved: u32,
    pub(crate) dequantization_step: f32,
}

/// Static HTJ2K entropy lookup tables uploaded for CUDA code-block decode.
#[derive(Clone, Copy, Debug)]
pub struct CudaHtj2kDecodeTables<'a> {
    /// HT cleanup VLC table for first quad row contexts.
    pub vlc_table0: &'a [u16; 1024],
    /// HT cleanup VLC table for subsequent quad row contexts.
    pub vlc_table1: &'a [u16; 1024],
    /// HT cleanup UVLC table for first quad row contexts.
    pub uvlc_table0: &'a [u16; 320],
    /// HT cleanup UVLC table for subsequent quad row contexts.
    pub uvlc_table1: &'a [u16; 256],
}

/// Status written by the CUDA HTJ2K entropy decoder for one code-block job.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CudaHtj2kStatus {
    /// Zero on success; nonzero values are kernel-defined failures.
    pub code: u32,
    /// Kernel-defined failure detail.
    pub detail: u32,
    /// Reserved for ABI stability.
    pub reserved0: u32,
    /// Reserved for ABI stability.
    pub reserved1: u32,
}

impl CudaHtj2kStatus {
    /// Return true when the CUDA kernel reported success.
    pub fn is_ok(self) -> bool {
        self.code == HTJ2K_STATUS_OK
    }
}

/// CUDA event timings for resident HTJ2K decode stages.
#[allow(clippy::struct_field_names)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CudaHtj2kDecodeStageTimings {
    /// HT cleanup entropy decode dispatch time, in microseconds.
    pub ht_cleanup_us: u128,
    /// HT refinement work time, in microseconds.
    ///
    /// The current CUDA entropy kernel executes cleanup and refinement for a
    /// code-block in one dispatch. When a batch contains refinement segments,
    /// this records that fused dispatch time so higher-level profiles expose
    /// refinement-bearing work instead of silently reporting zero.
    pub ht_refine_us: u128,
    /// Sign/magnitude dequantization time, in microseconds.
    pub dequant_us: u128,
    /// Host-observed status download time, in microseconds.
    pub status_d2h_us: u128,
}

/// Device-resident HTJ2K entropy decode result.
#[derive(Debug)]
pub struct CudaHtj2kDecodeOutput {
    pub(crate) coefficients: CudaDeviceBuffer,
    pub(crate) execution: CudaExecutionStats,
    pub(crate) statuses: Vec<CudaHtj2kStatus>,
    pub(crate) stage_timings: CudaHtj2kDecodeStageTimings,
}

impl CudaHtj2kDecodeOutput {
    /// Device buffer containing decoded f32 coefficients.
    pub fn coefficients(&self) -> &CudaDeviceBuffer {
        &self.coefficients
    }

    /// CUDA execution counters for the decode dispatch.
    pub fn execution(&self) -> CudaExecutionStats {
        self.execution
    }

    /// Per-code-block kernel status rows downloaded after dispatch.
    pub fn statuses(&self) -> &[CudaHtj2kStatus] {
        &self.statuses
    }

    /// CUDA event timings for the decode stages inside this output.
    pub fn stage_timings(&self) -> CudaHtj2kDecodeStageTimings {
        self.stage_timings
    }

    /// Split output into device coefficients, execution counters, and statuses.
    pub fn into_parts(self) -> (CudaDeviceBuffer, CudaExecutionStats, Vec<CudaHtj2kStatus>) {
        (self.coefficients, self.execution, self.statuses)
    }
}

/// Device-resident HTJ2K entropy decode result borrowed from a CUDA buffer pool.
#[derive(Debug)]
pub struct CudaPooledHtj2kDecodeOutput {
    pub(crate) coefficients: CudaPooledDeviceBuffer,
    pub(crate) execution: CudaExecutionStats,
    pub(crate) statuses: Vec<CudaHtj2kStatus>,
    pub(crate) stage_timings: CudaHtj2kDecodeStageTimings,
}

impl CudaPooledHtj2kDecodeOutput {
    /// Device buffer containing decoded f32 coefficients.
    pub fn coefficients(&self) -> Option<&CudaDeviceBuffer> {
        self.coefficients.as_device_buffer()
    }

    /// CUDA execution counters for the decode dispatch.
    pub fn execution(&self) -> CudaExecutionStats {
        self.execution
    }

    /// Per-code-block kernel status rows downloaded after dispatch.
    pub fn statuses(&self) -> &[CudaHtj2kStatus] {
        &self.statuses
    }

    /// CUDA event timings for the decode stages inside this output.
    pub fn stage_timings(&self) -> CudaHtj2kDecodeStageTimings {
        self.stage_timings
    }

    /// Split output into pooled device coefficients, execution counters, and statuses.
    pub fn into_parts(
        self,
    ) -> (
        CudaPooledDeviceBuffer,
        CudaExecutionStats,
        Vec<CudaHtj2kStatus>,
    ) {
        (self.coefficients, self.execution, self.statuses)
    }
}

/// Device-resident static HTJ2K cleanup decode lookup tables.
#[derive(Clone, Debug)]
pub struct CudaHtj2kDecodeTableResources {
    pub(crate) inner: Arc<CudaHtj2kDecodeTableResourceInner>,
}

#[derive(Debug)]
pub(crate) struct CudaHtj2kDecodeTableResourceInner {
    pub(crate) vlc_table0: CudaDeviceBuffer,
    pub(crate) vlc_table1: CudaDeviceBuffer,
    pub(crate) uvlc_table0: CudaDeviceBuffer,
    pub(crate) uvlc_table1: CudaDeviceBuffer,
}

/// Device-resident HTJ2K decode payload plus shared lookup tables reused across sub-band dispatches.
#[derive(Debug)]
pub struct CudaHtj2kDecodeResources {
    pub(crate) payload: CudaHtj2kDecodePayload,
    pub(crate) payload_len: usize,
    pub(crate) tables: CudaHtj2kDecodeTableResources,
}

#[derive(Debug)]
pub(crate) enum CudaHtj2kDecodePayload {
    Owned(CudaDeviceBuffer),
    Pooled(CudaPooledDeviceBuffer),
}

impl CudaHtj2kDecodePayload {
    fn buffer(&self) -> Result<&CudaDeviceBuffer, CudaError> {
        match self {
            Self::Owned(buffer) => Ok(buffer),
            Self::Pooled(buffer) => pooled_device_buffer(buffer),
        }
    }
}

pub(crate) const HTJ2K_STATUS_OK: u32 = 0;

pub(crate) const HTJ2K_STATUS_UNSUPPORTED: u32 = 2;

impl CudaContext {
    /// Decode HTJ2K code blocks into a device-resident f32 coefficient plane.
    #[allow(clippy::similar_names)]
    pub fn decode_htj2k_codeblocks(
        &self,
        payload: &[u8],
        jobs: &[CudaHtj2kCodeBlockJob],
        tables: CudaHtj2kDecodeTables<'_>,
        output_words: usize,
    ) -> Result<CudaHtj2kDecodeOutput, CudaError> {
        if jobs.is_empty() {
            return self.decode_empty_htj2k_codeblocks(jobs, output_words);
        }
        let resources = self.upload_htj2k_decode_resources(payload, tables)?;
        self.decode_htj2k_codeblocks_with_resources(&resources, jobs, output_words)
    }

    /// Decode HTJ2K code blocks without collecting CUDA event timings.
    #[allow(clippy::similar_names)]
    pub fn decode_htj2k_codeblocks_untimed(
        &self,
        payload: &[u8],
        jobs: &[CudaHtj2kCodeBlockJob],
        tables: CudaHtj2kDecodeTables<'_>,
        output_words: usize,
    ) -> Result<CudaHtj2kDecodeOutput, CudaError> {
        if jobs.is_empty() {
            return self.decode_empty_htj2k_codeblocks(jobs, output_words);
        }
        let resources = self.upload_htj2k_decode_resources(payload, tables)?;
        self.decode_htj2k_codeblocks_with_resources_untimed(&resources, jobs, output_words)
    }

    /// Upload HTJ2K decode payload and lookup tables once for reuse by sub-band dispatches.
    pub fn upload_htj2k_decode_resources(
        &self,
        payload: &[u8],
        tables: CudaHtj2kDecodeTables<'_>,
    ) -> Result<CudaHtj2kDecodeResources, CudaError> {
        let tables = self.upload_htj2k_decode_table_resources(tables)?;
        self.upload_htj2k_decode_resources_with_tables(payload, &tables)
    }

    /// Upload static HTJ2K cleanup decode lookup tables once for reuse.
    pub fn upload_htj2k_decode_table_resources(
        &self,
        tables: CudaHtj2kDecodeTables<'_>,
    ) -> Result<CudaHtj2kDecodeTableResources, CudaError> {
        self.inner.set_current()?;
        Ok(CudaHtj2kDecodeTableResources {
            inner: Arc::new(CudaHtj2kDecodeTableResourceInner {
                vlc_table0: self.upload(u16_slice_as_bytes(tables.vlc_table0))?,
                vlc_table1: self.upload(u16_slice_as_bytes(tables.vlc_table1))?,
                uvlc_table0: self.upload(u16_slice_as_bytes(tables.uvlc_table0))?,
                uvlc_table1: self.upload(u16_slice_as_bytes(tables.uvlc_table1))?,
            }),
        })
    }

    /// Upload an HTJ2K decode payload while reusing already resident cleanup tables.
    pub fn upload_htj2k_decode_resources_with_tables(
        &self,
        payload: &[u8],
        tables: &CudaHtj2kDecodeTableResources,
    ) -> Result<CudaHtj2kDecodeResources, CudaError> {
        self.inner.set_current()?;
        Ok(CudaHtj2kDecodeResources {
            payload: CudaHtj2kDecodePayload::Owned(self.upload_pinned(payload)?),
            payload_len: payload.len(),
            tables: tables.clone(),
        })
    }

    /// Upload an HTJ2K decode payload into a pooled buffer while reusing already resident cleanup tables.
    pub fn upload_htj2k_decode_resources_with_tables_and_pool(
        &self,
        payload: &[u8],
        tables: &CudaHtj2kDecodeTableResources,
        pool: &CudaBufferPool,
    ) -> Result<CudaHtj2kDecodeResources, CudaError> {
        self.inner.set_current()?;
        Ok(CudaHtj2kDecodeResources {
            payload: CudaHtj2kDecodePayload::Pooled(pool.upload_pinned(payload)?),
            payload_len: payload.len(),
            tables: tables.clone(),
        })
    }

    /// Decode HTJ2K code blocks using already resident payload and lookup tables.
    pub fn decode_htj2k_codeblocks_with_resources(
        &self,
        resources: &CudaHtj2kDecodeResources,
        jobs: &[CudaHtj2kCodeBlockJob],
        output_words: usize,
    ) -> Result<CudaHtj2kDecodeOutput, CudaError> {
        self.decode_htj2k_codeblocks_with_resources_impl(resources, jobs, output_words, true)
    }

    /// Decode HTJ2K code blocks using resident resources without CUDA event timings.
    pub fn decode_htj2k_codeblocks_with_resources_untimed(
        &self,
        resources: &CudaHtj2kDecodeResources,
        jobs: &[CudaHtj2kCodeBlockJob],
        output_words: usize,
    ) -> Result<CudaHtj2kDecodeOutput, CudaError> {
        self.decode_htj2k_codeblocks_with_resources_impl(resources, jobs, output_words, false)
    }

    /// Decode HTJ2K code blocks using resident resources and caller-owned
    /// transient buffer reuse.
    pub fn decode_htj2k_codeblocks_with_resources_and_pool(
        &self,
        resources: &CudaHtj2kDecodeResources,
        jobs: &[CudaHtj2kCodeBlockJob],
        output_words: usize,
        pool: &CudaBufferPool,
    ) -> Result<CudaPooledHtj2kDecodeOutput, CudaError> {
        self.decode_htj2k_codeblocks_with_resources_and_pool_impl(
            resources,
            jobs,
            output_words,
            pool,
            true,
            true,
        )
    }

    /// Decode HTJ2K code blocks using resident resources and caller-owned
    /// transient buffer reuse, without CUDA event timings.
    pub fn decode_htj2k_codeblocks_with_resources_untimed_and_pool(
        &self,
        resources: &CudaHtj2kDecodeResources,
        jobs: &[CudaHtj2kCodeBlockJob],
        output_words: usize,
        pool: &CudaBufferPool,
    ) -> Result<CudaPooledHtj2kDecodeOutput, CudaError> {
        self.decode_htj2k_codeblocks_with_resources_and_pool_impl(
            resources,
            jobs,
            output_words,
            pool,
            false,
            true,
        )
    }

    /// Decode HTJ2K cleanup passes into resident coefficient buffers using
    /// caller-owned transient buffer reuse. Dequantization is left to a later
    /// dispatch.
    pub fn decode_htj2k_codeblocks_cleanup_with_resources_and_pool(
        &self,
        resources: &CudaHtj2kDecodeResources,
        jobs: &[CudaHtj2kCodeBlockJob],
        output_words: usize,
        pool: &CudaBufferPool,
    ) -> Result<CudaPooledHtj2kDecodeOutput, CudaError> {
        self.decode_htj2k_codeblocks_with_resources_and_pool_impl(
            resources,
            jobs,
            output_words,
            pool,
            true,
            false,
        )
    }

    /// Decode HTJ2K cleanup passes into resident coefficient buffers using
    /// caller-owned transient buffer reuse, without CUDA event timings.
    pub fn decode_htj2k_codeblocks_cleanup_with_resources_untimed_and_pool(
        &self,
        resources: &CudaHtj2kDecodeResources,
        jobs: &[CudaHtj2kCodeBlockJob],
        output_words: usize,
        pool: &CudaBufferPool,
    ) -> Result<CudaPooledHtj2kDecodeOutput, CudaError> {
        self.decode_htj2k_codeblocks_with_resources_and_pool_impl(
            resources,
            jobs,
            output_words,
            pool,
            false,
            false,
        )
    }

    /// Allocate and initialize an HTJ2K coefficient output buffer without
    /// launching entropy cleanup decode. This is used when cleanup work is
    /// batched across multiple output buffers.
    pub fn allocate_htj2k_codeblock_coefficients_with_pool(
        &self,
        jobs: &[CudaHtj2kCodeBlockJob],
        output_words: usize,
        pool: &CudaBufferPool,
    ) -> Result<CudaPooledHtj2kDecodeOutput, CudaError> {
        self.inner.set_current()?;
        let output_bytes = output_words
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or(CudaError::LengthTooLarge { len: output_words })?;
        let coefficients = pool.take(output_bytes)?;
        let coefficient_buffer = pooled_device_buffer(&coefficients)?;
        if htj2k_decode_needs_zero_fill(jobs, output_words)? {
            self.memset_d32(coefficient_buffer, 0, output_words)?;
        }
        Ok(CudaPooledHtj2kDecodeOutput {
            coefficients,
            execution: CudaExecutionStats::default(),
            statuses: Vec::new(),
            stage_timings: CudaHtj2kDecodeStageTimings::default(),
        })
    }

    /// Decode HTJ2K cleanup passes for multiple output buffers with one CUDA
    /// dispatch. Dequantization is left to a later dispatch.
    pub fn decode_htj2k_codeblocks_cleanup_multi_with_resources_and_pool(
        &self,
        resources: &CudaHtj2kDecodeResources,
        targets: &[CudaHtj2kCleanupTarget<'_>],
        pool: &CudaBufferPool,
    ) -> Result<CudaExecutionStats, CudaError> {
        self.decode_htj2k_codeblocks_cleanup_multi_with_resources_and_pool_timed(
            resources, targets, pool, false,
        )
        .map(|(execution, _timings)| execution)
    }

    /// Enqueue HTJ2K cleanup passes for multiple output buffers with one CUDA
    /// dispatch. The returned value must be kept live until `finish` validates
    /// the kernel statuses after the default stream has completed.
    pub fn decode_htj2k_codeblocks_cleanup_multi_enqueue_with_resources_and_pool(
        &self,
        resources: &CudaHtj2kDecodeResources,
        targets: &[CudaHtj2kCleanupTarget<'_>],
        pool: &CudaBufferPool,
    ) -> Result<CudaQueuedHtj2kCleanup, CudaError> {
        self.inner.set_current()?;
        let kernel_jobs = htj2k_cleanup_multi_kernel_jobs(targets, resources.payload_len)?;
        if kernel_jobs.is_empty() {
            return Ok(CudaQueuedHtj2kCleanup {
                resources: Vec::new(),
                status_buffer: None,
                status_count: 0,
                kernel_name: "j2k_htj2k_decode_codeblocks_multi",
                execution: CudaExecutionStats::default(),
            });
        }
        let (decode_kernel, decode_kernel_name) = htj2k_decode_multi_kernel_for_jobs(&kernel_jobs);

        let jobs_buffer = pool.upload(htj2k_cleanup_multi_jobs_as_bytes(&kernel_jobs))?;
        let status_buffer = pool.take(htj2k_statuses_byte_len(kernel_jobs.len())?)?;
        let launch_result = self.launch_htj2k_decode_codeblocks_multi(
            decode_kernel,
            resources.payload.buffer()?,
            pooled_device_buffer(&jobs_buffer)?,
            &resources.tables.inner.vlc_table0,
            &resources.tables.inner.vlc_table1,
            &resources.tables.inner.uvlc_table0,
            &resources.tables.inner.uvlc_table1,
            pooled_device_buffer(&status_buffer)?,
            kernel_jobs.len(),
            CudaLaunchMode::Async,
        );
        if let Err(error) = launch_result {
            let _ = self.synchronize();
            return Err(error);
        }

        Ok(CudaQueuedHtj2kCleanup {
            resources: vec![jobs_buffer],
            status_buffer: Some(status_buffer),
            status_count: kernel_jobs.len(),
            kernel_name: decode_kernel_name,
            execution: CudaExecutionStats {
                kernel_dispatches: 1,
                copy_kernel_dispatches: 0,
                decode_kernel_dispatches: 1,
                hardware_decode: false,
            },
        })
    }

    /// Decode HTJ2K cleanup passes for multiple output buffers with one CUDA
    /// dispatch and return optional host-side timing splits.
    ///
    /// Dequantization is left to a later dispatch. When `collect_stage_timings`
    /// is false, the cleanup kernel launch is left asynchronous and the
    /// mandatory status readback remains the completion point.
    pub fn decode_htj2k_codeblocks_cleanup_multi_with_resources_and_pool_timed(
        &self,
        resources: &CudaHtj2kDecodeResources,
        targets: &[CudaHtj2kCleanupTarget<'_>],
        pool: &CudaBufferPool,
        collect_stage_timings: bool,
    ) -> Result<(CudaExecutionStats, CudaHtj2kDecodeStageTimings), CudaError> {
        self.inner.set_current()?;
        let kernel_jobs = htj2k_cleanup_multi_kernel_jobs(targets, resources.payload_len)?;
        if kernel_jobs.is_empty() {
            return Ok((
                CudaExecutionStats::default(),
                CudaHtj2kDecodeStageTimings::default(),
            ));
        }

        let jobs_buffer = pool.upload(htj2k_cleanup_multi_jobs_as_bytes(&kernel_jobs))?;
        let status_buffer = pool.take(htj2k_statuses_byte_len(kernel_jobs.len())?)?;
        let (decode_kernel, decode_kernel_name) = htj2k_decode_multi_kernel_for_jobs(&kernel_jobs);
        if collect_stage_timings {
            self.launch_htj2k_decode_codeblocks_multi(
                decode_kernel,
                resources.payload.buffer()?,
                pooled_device_buffer(&jobs_buffer)?,
                &resources.tables.inner.vlc_table0,
                &resources.tables.inner.vlc_table1,
                &resources.tables.inner.uvlc_table0,
                &resources.tables.inner.uvlc_table1,
                pooled_device_buffer(&status_buffer)?,
                kernel_jobs.len(),
                CudaLaunchMode::Sync,
            )?;
        } else {
            self.launch_htj2k_decode_codeblocks_multi(
                decode_kernel,
                resources.payload.buffer()?,
                pooled_device_buffer(&jobs_buffer)?,
                &resources.tables.inner.vlc_table0,
                &resources.tables.inner.vlc_table1,
                &resources.tables.inner.uvlc_table0,
                &resources.tables.inner.uvlc_table1,
                pooled_device_buffer(&status_buffer)?,
                kernel_jobs.len(),
                CudaLaunchMode::Async,
            )?;
        }

        let mut statuses = vec![CudaHtj2kStatus::default(); kernel_jobs.len()];
        let status_d2h_start = collect_stage_timings.then(Instant::now);
        status_buffer.copy_to_host(htj2k_statuses_as_bytes_mut(&mut statuses))?;
        let status_d2h_us = status_d2h_start.map_or(0, |start| start.elapsed().as_micros());
        if let Some(status) = statuses.iter().copied().find(|status| !status.is_ok()) {
            return Err(CudaError::KernelStatus {
                kernel: decode_kernel_name,
                code: status.code,
                detail: status.detail,
            });
        }

        Ok((
            CudaExecutionStats {
                kernel_dispatches: 1,
                copy_kernel_dispatches: 0,
                decode_kernel_dispatches: 1,
                hardware_decode: false,
            },
            CudaHtj2kDecodeStageTimings {
                status_d2h_us,
                ..CudaHtj2kDecodeStageTimings::default()
            },
        ))
    }

    /// Decode HTJ2K cleanup-only passes and dequantize their coefficients in
    /// one CUDA dispatch. Targets containing refinement passes are rejected so
    /// callers can fall back to cleanup followed by dequantization.
    pub fn decode_htj2k_codeblocks_cleanup_dequantize_multi_with_resources_and_pool_timed(
        &self,
        resources: &CudaHtj2kDecodeResources,
        targets: &[CudaHtj2kCleanupTarget<'_>],
        pool: &CudaBufferPool,
        collect_stage_timings: bool,
    ) -> Result<(CudaExecutionStats, CudaHtj2kDecodeStageTimings), CudaError> {
        self.inner.set_current()?;
        let kernel_jobs = htj2k_cleanup_multi_kernel_jobs(targets, resources.payload_len)?;
        if kernel_jobs.is_empty() {
            return Ok((
                CudaExecutionStats::default(),
                CudaHtj2kDecodeStageTimings::default(),
            ));
        }
        let Some((decode_kernel, decode_kernel_name)) =
            htj2k_decode_multi_cleanup_dequant_kernel_for_jobs(&kernel_jobs)
        else {
            return Err(CudaError::InvalidArgument {
                message: "fused HTJ2K cleanup/dequantize requires cleanup-only jobs".to_string(),
            });
        };

        let jobs_buffer = pool.upload(htj2k_cleanup_multi_jobs_as_bytes(&kernel_jobs))?;
        let status_buffer = pool.take(htj2k_statuses_byte_len(kernel_jobs.len())?)?;
        if collect_stage_timings {
            self.launch_htj2k_decode_codeblocks_multi(
                decode_kernel,
                resources.payload.buffer()?,
                pooled_device_buffer(&jobs_buffer)?,
                &resources.tables.inner.vlc_table0,
                &resources.tables.inner.vlc_table1,
                &resources.tables.inner.uvlc_table0,
                &resources.tables.inner.uvlc_table1,
                pooled_device_buffer(&status_buffer)?,
                kernel_jobs.len(),
                CudaLaunchMode::Sync,
            )?;
        } else {
            self.launch_htj2k_decode_codeblocks_multi(
                decode_kernel,
                resources.payload.buffer()?,
                pooled_device_buffer(&jobs_buffer)?,
                &resources.tables.inner.vlc_table0,
                &resources.tables.inner.vlc_table1,
                &resources.tables.inner.uvlc_table0,
                &resources.tables.inner.uvlc_table1,
                pooled_device_buffer(&status_buffer)?,
                kernel_jobs.len(),
                CudaLaunchMode::Async,
            )?;
        }

        let mut statuses = vec![CudaHtj2kStatus::default(); kernel_jobs.len()];
        let status_d2h_start = collect_stage_timings.then(Instant::now);
        status_buffer.copy_to_host(htj2k_statuses_as_bytes_mut(&mut statuses))?;
        let status_d2h_us = status_d2h_start.map_or(0, |start| start.elapsed().as_micros());
        if let Some(status) = statuses.iter().copied().find(|status| !status.is_ok()) {
            return Err(CudaError::KernelStatus {
                kernel: decode_kernel_name,
                code: status.code,
                detail: status.detail,
            });
        }

        Ok((
            CudaExecutionStats {
                kernel_dispatches: 1,
                copy_kernel_dispatches: 0,
                decode_kernel_dispatches: 1,
                hardware_decode: false,
            },
            CudaHtj2kDecodeStageTimings {
                status_d2h_us,
                ..CudaHtj2kDecodeStageTimings::default()
            },
        ))
    }

    fn decode_htj2k_codeblocks_with_resources_impl(
        &self,
        resources: &CudaHtj2kDecodeResources,
        jobs: &[CudaHtj2kCodeBlockJob],
        output_words: usize,
        collect_stage_timings: bool,
    ) -> Result<CudaHtj2kDecodeOutput, CudaError> {
        self.inner.set_current()?;
        let output_bytes = output_words
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or(CudaError::LengthTooLarge { len: output_words })?;
        let coefficients = self.allocate(output_bytes)?;
        if htj2k_decode_needs_zero_fill(jobs, output_words)? {
            self.memset_d32(&coefficients, 0, output_words)?;
        }
        if jobs.is_empty() {
            return Ok(CudaHtj2kDecodeOutput {
                coefficients,
                execution: CudaExecutionStats::default(),
                statuses: Vec::new(),
                stage_timings: CudaHtj2kDecodeStageTimings::default(),
            });
        }

        let kernel_jobs = htj2k_kernel_jobs(jobs, resources.payload_len, output_words)?;
        let jobs_buffer = self.upload(htj2k_jobs_as_bytes(&kernel_jobs))?;
        let status_buffer = self.allocate(htj2k_statuses_byte_len(jobs.len())?)?;

        let has_refinement = jobs
            .iter()
            .any(|job| job.refinement_length > 0 || job.number_of_coding_passes > 1);
        let (ht_cleanup_us, dequant_us) = self.submit_htj2k_decode_and_dequantize(
            resources,
            &coefficients,
            &jobs_buffer,
            &status_buffer,
            jobs.len(),
            collect_stage_timings,
        )?;

        let mut statuses = vec![CudaHtj2kStatus::default(); jobs.len()];
        if let Err(error) = status_buffer.copy_to_host(htj2k_statuses_as_bytes_mut(&mut statuses)) {
            if !collect_stage_timings {
                let _ = self.synchronize();
            }
            return Err(error);
        }
        if let Some(status) = statuses.iter().copied().find(|status| !status.is_ok()) {
            return Err(CudaError::KernelStatus {
                kernel: "j2k_htj2k_decode_codeblocks",
                code: status.code,
                detail: status.detail,
            });
        }

        Ok(CudaHtj2kDecodeOutput {
            coefficients,
            execution: CudaExecutionStats {
                kernel_dispatches: 2,
                copy_kernel_dispatches: 0,
                decode_kernel_dispatches: 2,
                hardware_decode: false,
            },
            statuses,
            stage_timings: CudaHtj2kDecodeStageTimings {
                ht_cleanup_us,
                ht_refine_us: if has_refinement { ht_cleanup_us } else { 0 },
                dequant_us,
                ..CudaHtj2kDecodeStageTimings::default()
            },
        })
    }

    fn decode_htj2k_codeblocks_with_resources_and_pool_impl(
        &self,
        resources: &CudaHtj2kDecodeResources,
        jobs: &[CudaHtj2kCodeBlockJob],
        output_words: usize,
        pool: &CudaBufferPool,
        collect_stage_timings: bool,
        dequantize: bool,
    ) -> Result<CudaPooledHtj2kDecodeOutput, CudaError> {
        self.inner.set_current()?;
        let output_bytes = output_words
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or(CudaError::LengthTooLarge { len: output_words })?;
        let coefficients = pool.take(output_bytes)?;
        let coefficient_buffer = pooled_device_buffer(&coefficients)?;
        if htj2k_decode_needs_zero_fill(jobs, output_words)? {
            self.memset_d32(coefficient_buffer, 0, output_words)?;
        }
        if jobs.is_empty() {
            return Ok(CudaPooledHtj2kDecodeOutput {
                coefficients,
                execution: CudaExecutionStats::default(),
                statuses: Vec::new(),
                stage_timings: CudaHtj2kDecodeStageTimings::default(),
            });
        }

        let kernel_jobs = htj2k_kernel_jobs(jobs, resources.payload_len, output_words)?;
        let jobs_buffer = pool.upload(htj2k_jobs_as_bytes(&kernel_jobs))?;
        let status_buffer = pool.take(htj2k_statuses_byte_len(jobs.len())?)?;

        let has_refinement = jobs
            .iter()
            .any(|job| job.refinement_length > 0 || job.number_of_coding_passes > 1);
        let jobs_device = pooled_device_buffer(&jobs_buffer)?;
        let status_device = pooled_device_buffer(&status_buffer)?;
        let (ht_cleanup_us, dequant_us, kernel_dispatches) = if dequantize {
            let (ht_cleanup_us, dequant_us) = self.submit_htj2k_decode_and_dequantize(
                resources,
                coefficient_buffer,
                jobs_device,
                status_device,
                jobs.len(),
                collect_stage_timings,
            )?;
            (ht_cleanup_us, dequant_us, 2)
        } else {
            let ht_cleanup_us = self.submit_htj2k_decode_cleanup(
                resources,
                coefficient_buffer,
                jobs_device,
                status_device,
                jobs.len(),
                collect_stage_timings,
            )?;
            (ht_cleanup_us, 0, 1)
        };

        let mut statuses = vec![CudaHtj2kStatus::default(); jobs.len()];
        if let Err(error) = status_buffer.copy_to_host(htj2k_statuses_as_bytes_mut(&mut statuses)) {
            if !collect_stage_timings {
                let _ = self.synchronize();
            }
            return Err(error);
        }
        if let Some(status) = statuses.iter().copied().find(|status| !status.is_ok()) {
            return Err(CudaError::KernelStatus {
                kernel: "j2k_htj2k_decode_codeblocks",
                code: status.code,
                detail: status.detail,
            });
        }

        Ok(CudaPooledHtj2kDecodeOutput {
            coefficients,
            execution: CudaExecutionStats {
                kernel_dispatches,
                copy_kernel_dispatches: 0,
                decode_kernel_dispatches: kernel_dispatches,
                hardware_decode: false,
            },
            statuses,
            stage_timings: CudaHtj2kDecodeStageTimings {
                ht_cleanup_us,
                ht_refine_us: if has_refinement { ht_cleanup_us } else { 0 },
                dequant_us,
                ..CudaHtj2kDecodeStageTimings::default()
            },
        })
    }

    fn submit_htj2k_decode_and_dequantize(
        &self,
        resources: &CudaHtj2kDecodeResources,
        coefficients: &CudaDeviceBuffer,
        jobs_buffer: &CudaDeviceBuffer,
        status_buffer: &CudaDeviceBuffer,
        job_count: usize,
        collect_stage_timings: bool,
    ) -> Result<(u128, u128), CudaError> {
        let ht_cleanup_us = self.submit_htj2k_decode_cleanup(
            resources,
            coefficients,
            jobs_buffer,
            status_buffer,
            job_count,
            collect_stage_timings,
        )?;
        let dequant_us = self.submit_htj2k_dequantize_htj2k_codeblocks(
            coefficients,
            jobs_buffer,
            job_count,
            collect_stage_timings,
        )?;
        Ok((ht_cleanup_us, dequant_us))
    }

    fn submit_htj2k_decode_cleanup(
        &self,
        resources: &CudaHtj2kDecodeResources,
        coefficients: &CudaDeviceBuffer,
        jobs_buffer: &CudaDeviceBuffer,
        status_buffer: &CudaDeviceBuffer,
        job_count: usize,
        collect_stage_timings: bool,
    ) -> Result<u128, CudaError> {
        let ((), ht_cleanup_us) = self.time_default_stream_named_us_if(
            collect_stage_timings,
            "j2k.htj2k.decode.cleanup",
            || {
                if !collect_stage_timings {
                    return self.launch_htj2k_decode_codeblocks(
                        resources.payload.buffer()?,
                        coefficients,
                        jobs_buffer,
                        &resources.tables.inner.vlc_table0,
                        &resources.tables.inner.vlc_table1,
                        &resources.tables.inner.uvlc_table0,
                        &resources.tables.inner.uvlc_table1,
                        status_buffer,
                        job_count,
                        CudaLaunchMode::Async,
                    );
                }
                self.launch_htj2k_decode_codeblocks(
                    resources.payload.buffer()?,
                    coefficients,
                    jobs_buffer,
                    &resources.tables.inner.vlc_table0,
                    &resources.tables.inner.vlc_table1,
                    &resources.tables.inner.uvlc_table0,
                    &resources.tables.inner.uvlc_table1,
                    status_buffer,
                    job_count,
                    CudaLaunchMode::Sync,
                )
            },
        )?;
        Ok(ht_cleanup_us)
    }

    fn submit_htj2k_dequantize_htj2k_codeblocks(
        &self,
        coefficients: &CudaDeviceBuffer,
        jobs_buffer: &CudaDeviceBuffer,
        job_count: usize,
        collect_stage_timings: bool,
    ) -> Result<u128, CudaError> {
        let ((), dequant_us) = match self.time_default_stream_named_us_if(
            collect_stage_timings,
            "j2k.htj2k.decode.dequantize",
            || {
                if collect_stage_timings {
                    self.launch_j2k_dequantize_htj2k_codeblocks(
                        coefficients,
                        jobs_buffer,
                        job_count,
                        CudaLaunchMode::Sync,
                    )
                } else {
                    self.launch_j2k_dequantize_htj2k_codeblocks(
                        coefficients,
                        jobs_buffer,
                        job_count,
                        CudaLaunchMode::Async,
                    )
                }
            },
        ) {
            Ok(result) => result,
            Err(error) => {
                if !collect_stage_timings {
                    let _ = self.synchronize();
                }
                return Err(error);
            }
        };
        Ok(dequant_us)
    }

    /// Dequantize HTJ2K code-block outputs that live in multiple device buffers
    /// with one CUDA dispatch.
    pub fn j2k_dequantize_htj2k_codeblocks_multi_device(
        &self,
        targets: &[CudaHtj2kDequantizeTarget<'_>],
    ) -> Result<CudaExecutionStats, CudaError> {
        let pool = self.buffer_pool();
        self.j2k_dequantize_htj2k_codeblocks_multi_device_with_pool(targets, &pool)
    }

    /// Dequantize HTJ2K code-block outputs that live in multiple device buffers
    /// with one CUDA dispatch, reusing caller-owned transient storage.
    pub fn j2k_dequantize_htj2k_codeblocks_multi_device_with_pool(
        &self,
        targets: &[CudaHtj2kDequantizeTarget<'_>],
        pool: &CudaBufferPool,
    ) -> Result<CudaExecutionStats, CudaError> {
        self.j2k_dequantize_htj2k_codeblocks_multi_device_with_pool_impl(targets, pool, true)
    }

    /// Dequantize HTJ2K code-block outputs in multiple device buffers without
    /// CUDA event timings. The launch is still synchronized before returning
    /// so the pooled job upload cannot be reused while the kernel reads it.
    pub fn j2k_dequantize_htj2k_codeblocks_multi_device_untimed_with_pool(
        &self,
        targets: &[CudaHtj2kDequantizeTarget<'_>],
        pool: &CudaBufferPool,
    ) -> Result<CudaExecutionStats, CudaError> {
        self.j2k_dequantize_htj2k_codeblocks_multi_device_with_pool_impl(targets, pool, true)
    }

    fn j2k_dequantize_htj2k_codeblocks_multi_device_with_pool_impl(
        &self,
        targets: &[CudaHtj2kDequantizeTarget<'_>],
        pool: &CudaBufferPool,
        synchronize_each_launch: bool,
    ) -> Result<CudaExecutionStats, CudaError> {
        self.inner.set_current()?;
        let kernel_jobs = htj2k_dequantize_kernel_jobs(targets)?;
        if kernel_jobs.is_empty() {
            return Ok(CudaExecutionStats::default());
        }
        let jobs_buffer = pool.upload(htj2k_dequantize_jobs_as_bytes(&kernel_jobs))?;
        self.launch_j2k_dequantize_htj2k_codeblocks_multi(
            pooled_device_buffer(&jobs_buffer)?,
            kernel_jobs.len(),
            CudaLaunchMode::from_synchronize(synchronize_each_launch),
        )?;
        Ok(CudaExecutionStats {
            kernel_dispatches: 1,
            copy_kernel_dispatches: 0,
            decode_kernel_dispatches: 1,
            hardware_decode: false,
        })
    }

    #[allow(clippy::similar_names, clippy::too_many_arguments)]
    fn launch_htj2k_decode_codeblocks(
        &self,
        payload: &CudaDeviceBuffer,
        coefficients: &CudaDeviceBuffer,
        jobs: &CudaDeviceBuffer,
        vlc_table0: &CudaDeviceBuffer,
        vlc_table1: &CudaDeviceBuffer,
        uvlc_table0: &CudaDeviceBuffer,
        uvlc_table1: &CudaDeviceBuffer,
        statuses: &CudaDeviceBuffer,
        job_count: usize,
        mode: CudaLaunchMode,
    ) -> Result<(), CudaError> {
        let mut payload_ptr = payload.device_ptr();
        let mut coefficients_ptr = coefficients.device_ptr();
        let mut jobs_ptr = jobs.device_ptr();
        let mut vlc_table0_ptr = vlc_table0.device_ptr();
        let mut vlc_table1_ptr = vlc_table1.device_ptr();
        let mut uvlc_table0_ptr = uvlc_table0.device_ptr();
        let mut uvlc_table1_ptr = uvlc_table1.device_ptr();
        let mut statuses_ptr = statuses.device_ptr();
        let mut job_count = c_uint::try_from(job_count)
            .map_err(|_| CudaError::LengthTooLarge { len: job_count })?;
        let mut params = cuda_kernel_params!(
            payload_ptr,
            coefficients_ptr,
            jobs_ptr,
            vlc_table0_ptr,
            vlc_table1_ptr,
            uvlc_table0_ptr,
            uvlc_table1_ptr,
            statuses_ptr,
            job_count
        );
        let geometry = htj2k_codeblock_launch_geometry(job_count as usize).ok_or(
            CudaError::LengthTooLarge {
                len: job_count as usize,
            },
        )?;

        self.launch_named_kernel(
            CudaKernel::Htj2kDecodeCodeblocks,
            geometry,
            &mut params,
            mode,
        )
    }

    #[allow(clippy::similar_names, clippy::too_many_arguments)]
    fn launch_htj2k_decode_codeblocks_multi(
        &self,
        kernel: CudaKernel,
        payload: &CudaDeviceBuffer,
        jobs: &CudaDeviceBuffer,
        vlc_table0: &CudaDeviceBuffer,
        vlc_table1: &CudaDeviceBuffer,
        uvlc_table0: &CudaDeviceBuffer,
        uvlc_table1: &CudaDeviceBuffer,
        statuses: &CudaDeviceBuffer,
        job_count: usize,
        mode: CudaLaunchMode,
    ) -> Result<(), CudaError> {
        let mut payload_ptr = payload.device_ptr();
        let mut jobs_ptr = jobs.device_ptr();
        let mut vlc_table0_ptr = vlc_table0.device_ptr();
        let mut vlc_table1_ptr = vlc_table1.device_ptr();
        let mut uvlc_table0_ptr = uvlc_table0.device_ptr();
        let mut uvlc_table1_ptr = uvlc_table1.device_ptr();
        let mut statuses_ptr = statuses.device_ptr();
        let mut job_count = c_uint::try_from(job_count)
            .map_err(|_| CudaError::LengthTooLarge { len: job_count })?;
        let mut params = cuda_kernel_params!(
            payload_ptr,
            jobs_ptr,
            vlc_table0_ptr,
            vlc_table1_ptr,
            uvlc_table0_ptr,
            uvlc_table1_ptr,
            statuses_ptr,
            job_count
        );
        let geometry = htj2k_codeblock_launch_geometry(job_count as usize).ok_or(
            CudaError::LengthTooLarge {
                len: job_count as usize,
            },
        )?;

        self.launch_named_kernel(kernel, geometry, &mut params, mode)
    }

    fn launch_j2k_dequantize_htj2k_codeblocks(
        &self,
        coefficients: &CudaDeviceBuffer,
        jobs: &CudaDeviceBuffer,
        job_count: usize,
        mode: CudaLaunchMode,
    ) -> Result<(), CudaError> {
        let mut coefficients_ptr = coefficients.device_ptr();
        let mut jobs_ptr = jobs.device_ptr();
        let mut params = cuda_kernel_params!(coefficients_ptr, jobs_ptr);
        let geometry = htj2k_codeblock_sample_launch_geometry(job_count)
            .ok_or(CudaError::LengthTooLarge { len: job_count })?;

        self.launch_named_kernel(
            CudaKernel::J2kDequantizeHtj2kCodeblocks,
            geometry,
            &mut params,
            mode,
        )
    }

    fn launch_j2k_dequantize_htj2k_codeblocks_multi(
        &self,
        jobs: &CudaDeviceBuffer,
        job_count: usize,
        mode: CudaLaunchMode,
    ) -> Result<(), CudaError> {
        let mut jobs_ptr = jobs.device_ptr();
        let mut params = cuda_kernel_params!(jobs_ptr);
        let geometry = htj2k_codeblock_sample_launch_geometry(job_count)
            .ok_or(CudaError::LengthTooLarge { len: job_count })?;

        self.launch_named_kernel(
            CudaKernel::J2kDequantizeHtj2kCodeblocksMulti,
            geometry,
            &mut params,
            mode,
        )
    }

    pub(crate) fn launch_j2k_dequantize_htj2k_cleanup_jobs_multi(
        &self,
        jobs: &CudaDeviceBuffer,
        job_count: usize,
        mode: CudaLaunchMode,
    ) -> Result<(), CudaError> {
        let mut jobs_ptr = jobs.device_ptr();
        let mut params = cuda_kernel_params!(jobs_ptr);
        let geometry = htj2k_codeblock_sample_launch_geometry(job_count)
            .ok_or(CudaError::LengthTooLarge { len: job_count })?;

        self.launch_named_kernel(
            CudaKernel::J2kDequantizeHtj2kCleanupJobsMulti,
            geometry,
            &mut params,
            mode,
        )
    }
}

/// Enqueued HTJ2K cleanup work plus pooled resources/statuses that must stay
/// live until `finish` validates kernel completion.
#[derive(Debug)]
pub struct CudaQueuedHtj2kCleanup {
    pub(crate) resources: Vec<CudaPooledDeviceBuffer>,
    pub(crate) status_buffer: Option<CudaPooledDeviceBuffer>,
    pub(crate) status_count: usize,
    pub(crate) kernel_name: &'static str,
    pub(crate) execution: CudaExecutionStats,
}

impl CudaQueuedHtj2kCleanup {
    /// CUDA execution counters for the enqueued cleanup work.
    pub fn execution(&self) -> CudaExecutionStats {
        self.execution
    }

    /// Number of pooled resource buffers held live for the queued cleanup work.
    pub fn resource_count(&self) -> usize {
        self.resources.len() + usize::from(self.status_buffer.is_some())
    }

    /// Synchronize through status download and validate kernel statuses.
    pub fn finish(self) -> Result<CudaExecutionStats, CudaError> {
        let Some(status_buffer) = self.status_buffer else {
            return Ok(self.execution);
        };

        let mut statuses = vec![CudaHtj2kStatus::default(); self.status_count];
        status_buffer.copy_to_host(htj2k_statuses_as_bytes_mut(&mut statuses))?;
        if let Some(status) = statuses.iter().copied().find(|status| !status.is_ok()) {
            return Err(CudaError::KernelStatus {
                kernel: self.kernel_name,
                code: status.code,
                detail: status.detail,
            });
        }

        Ok(self.execution)
    }
}

pub(crate) fn htj2k_kernel_jobs(
    jobs: &[CudaHtj2kCodeBlockJob],
    payload_len: usize,
    output_words: usize,
) -> Result<Vec<CudaHtj2kCodeBlockKernelJob>, CudaError> {
    jobs.iter()
        .map(|job| {
            let payload_offset = usize::try_from(job.payload_offset)
                .map_err(|_| CudaError::LengthTooLarge { len: usize::MAX })?;
            let payload_end = payload_offset
                .checked_add(job.payload_len as usize)
                .ok_or(CudaError::LengthTooLarge { len: payload_len })?;
            let expected_payload_len = job
                .cleanup_length
                .checked_add(job.refinement_length)
                .ok_or(CudaError::LengthTooLarge {
                    len: job.payload_len as usize,
                })?;
            let output_stride = job.output_stride as usize;
            let output_offset = job.output_offset as usize;
            let output_end = if job.height == 0 {
                output_offset
            } else {
                output_offset
                    .checked_add(
                        output_stride
                            .checked_mul(job.height as usize - 1)
                            .ok_or(CudaError::LengthTooLarge { len: output_words })?,
                    )
                    .and_then(|last_row| last_row.checked_add(job.width as usize))
                    .ok_or(CudaError::LengthTooLarge { len: output_words })?
            };
            if payload_end > payload_len
                || expected_payload_len != job.payload_len
                || output_end > output_words
            {
                return Err(CudaError::LengthTooLarge {
                    len: payload_len.max(output_words),
                });
            }
            Ok(CudaHtj2kCodeBlockKernelJob {
                coded_offset: u32::try_from(payload_offset)
                    .map_err(|_| CudaError::LengthTooLarge { len: payload_len })?,
                width: job.width,
                height: job.height,
                coded_len: job.payload_len,
                cleanup_length: job.cleanup_length,
                refinement_length: job.refinement_length,
                missing_msbs: u32::from(job.missing_bit_planes),
                num_bitplanes: u32::from(job.num_bitplanes),
                number_of_coding_passes: u32::from(job.number_of_coding_passes),
                output_stride: job.output_stride,
                output_offset: job.output_offset,
                dequantization_step: job.dequantization_step,
                stripe_causal: u32::from(job.stripe_causal),
            })
        })
        .collect()
}

pub(crate) fn htj2k_dequantize_kernel_jobs(
    targets: &[CudaHtj2kDequantizeTarget<'_>],
) -> Result<Vec<CudaHtj2kDequantizeKernelJob>, CudaError> {
    let total_jobs = targets
        .iter()
        .try_fold(0usize, |count, target| count.checked_add(target.jobs.len()))
        .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
    let mut kernel_jobs = Vec::with_capacity(total_jobs);
    for target in targets {
        let output_bytes = target
            .output_words
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or(CudaError::LengthTooLarge {
                len: target.output_words,
            })?;
        if output_bytes > target.coefficients.byte_len() {
            return Err(CudaError::LengthTooLarge { len: output_bytes });
        }
        for job in target.jobs {
            let output_stride = job.output_stride as usize;
            let output_offset = job.output_offset as usize;
            let output_end = if job.height == 0 {
                output_offset
            } else {
                output_offset
                    .checked_add(output_stride.checked_mul(job.height as usize - 1).ok_or(
                        CudaError::LengthTooLarge {
                            len: target.output_words,
                        },
                    )?)
                    .and_then(|last_row| last_row.checked_add(job.width as usize))
                    .ok_or(CudaError::LengthTooLarge {
                        len: target.output_words,
                    })?
            };
            if output_end > target.output_words {
                return Err(CudaError::LengthTooLarge {
                    len: target.output_words,
                });
            }
            kernel_jobs.push(CudaHtj2kDequantizeKernelJob {
                output_ptr: target.coefficients.device_ptr(),
                width: job.width,
                height: job.height,
                output_stride: job.output_stride,
                output_offset: job.output_offset,
                num_bitplanes: u32::from(job.num_bitplanes),
                reserved: 0,
                dequantization_step: job.dequantization_step,
            });
        }
    }
    Ok(kernel_jobs)
}

pub(crate) fn htj2k_cleanup_multi_kernel_jobs(
    targets: &[CudaHtj2kCleanupTarget<'_>],
    payload_len: usize,
) -> Result<Vec<CudaHtj2kCleanupMultiKernelJob>, CudaError> {
    let total_jobs = targets
        .iter()
        .try_fold(0usize, |count, target| count.checked_add(target.jobs.len()))
        .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
    let mut kernel_jobs = Vec::with_capacity(total_jobs);
    for target in targets {
        let output_bytes = target
            .output_words
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or(CudaError::LengthTooLarge {
                len: target.output_words,
            })?;
        if output_bytes > target.coefficients.byte_len() {
            return Err(CudaError::LengthTooLarge { len: output_bytes });
        }
        for job in htj2k_kernel_jobs(target.jobs, payload_len, target.output_words)? {
            kernel_jobs.push(CudaHtj2kCleanupMultiKernelJob {
                output_ptr: target.coefficients.device_ptr(),
                coded_offset: job.coded_offset,
                width: job.width,
                height: job.height,
                coded_len: job.coded_len,
                cleanup_length: job.cleanup_length,
                refinement_length: job.refinement_length,
                missing_msbs: job.missing_msbs,
                num_bitplanes: job.num_bitplanes,
                number_of_coding_passes: job.number_of_coding_passes,
                output_stride: job.output_stride,
                output_offset: job.output_offset,
                dequantization_step: job.dequantization_step,
                stripe_causal: job.stripe_causal,
            });
        }
    }
    Ok(kernel_jobs)
}

pub(crate) fn htj2k_decode_multi_kernel_for_jobs(
    jobs: &[CudaHtj2kCleanupMultiKernelJob],
) -> (CudaKernel, &'static str) {
    let cleanup_only = jobs
        .iter()
        .all(|job| job.refinement_length == 0 && job.number_of_coding_passes <= 1);
    if cleanup_only {
        (
            CudaKernel::Htj2kDecodeCodeblocksMultiCleanupOnly,
            "j2k_htj2k_decode_codeblocks_multi_cleanup_only",
        )
    } else {
        (
            CudaKernel::Htj2kDecodeCodeblocksMulti,
            "j2k_htj2k_decode_codeblocks_multi",
        )
    }
}

pub(crate) fn htj2k_decode_multi_cleanup_dequant_kernel_for_jobs(
    jobs: &[CudaHtj2kCleanupMultiKernelJob],
) -> Option<(CudaKernel, &'static str)> {
    let cleanup_only = jobs
        .iter()
        .all(|job| job.refinement_length == 0 && job.number_of_coding_passes <= 1);
    cleanup_only.then_some((
        CudaKernel::Htj2kDecodeCodeblocksMultiCleanupDequantize,
        "j2k_htj2k_decode_codeblocks_multi_cleanup_dequantize",
    ))
}

pub(crate) fn htj2k_decode_needs_zero_fill(
    jobs: &[CudaHtj2kCodeBlockJob],
    output_words: usize,
) -> Result<bool, CudaError> {
    let mut covered_words = 0usize;
    for job in jobs {
        let area = (job.width as usize)
            .checked_mul(job.height as usize)
            .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
        covered_words = covered_words
            .checked_add(area)
            .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
    }
    if covered_words > output_words {
        return Err(CudaError::LengthTooLarge { len: covered_words });
    }
    Ok(covered_words != output_words)
}
