use crate::{
    bytes::{
        idwt_job_as_bytes, idwt_multi_jobs_as_bytes, inverse_mct_job_as_bytes,
        store_gray16_job_as_bytes, store_gray8_job_as_bytes, store_rgb16_job_as_bytes,
        store_rgb16_mct_job_as_bytes, store_rgb8_job_as_bytes, store_rgb8_mct_batch_jobs_as_bytes,
    },
    context::{cuda_idwt_trace_enabled, CudaContext},
    driver::CuDevicePtr,
    error::CudaError,
    execution::{
        cuda_kernel_param, elapsed_event_us_ceil, CudaExecutionStats, CudaKernelBatchOutput,
        CudaKernelContiguousBatchOutput, CudaKernelOutput, CudaLaunchMode, CudaPooledKernelOutput,
        CudaQueuedExecution,
    },
    kernels::{
        j2k_dwt53_launch_geometry, j2k_forward_rct_launch_geometry,
        j2k_idwt_multi_1d_launch_geometry, j2k_idwt_multi_coop_axis_launch_geometry,
        j2k_idwt_multi_coop_columns_launch_geometry, j2k_idwt_multi_coop_launch_geometry,
        j2k_store_batch_launch_geometry, CudaKernel,
    },
    memory::{
        checked_image_words, pooled_device_buffer, CudaBufferPool, CudaDeviceBuffer,
        CudaDeviceBufferRange,
    },
};

/// CUDA-side integer rectangle for JPEG 2000 direct-plan kernels.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CudaJ2kRect {
    /// Inclusive minimum x coordinate.
    pub x0: u32,
    /// Inclusive minimum y coordinate.
    pub y0: u32,
    /// Exclusive maximum x coordinate.
    pub x1: u32,
    /// Exclusive maximum y coordinate.
    pub y1: u32,
}

/// One single-decomposition inverse DWT dispatch over device coefficient bands.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaJ2kIdwtJob {
    /// Output rectangle produced by the IDWT stage.
    pub rect: CudaJ2kRect,
    /// LL input band rectangle.
    pub ll_rect: CudaJ2kRect,
    /// HL input band rectangle.
    pub hl_rect: CudaJ2kRect,
    /// LH input band rectangle.
    pub lh_rect: CudaJ2kRect,
    /// HH input band rectangle.
    pub hh_rect: CudaJ2kRect,
    /// Nonzero for irreversible 9/7; zero for reversible 5/3.
    pub irreversible97: u32,
}

/// One output buffer and input band set for batched inverse DWT.
#[derive(Clone, Copy, Debug)]
pub struct CudaJ2kIdwtTarget<'a> {
    /// LL input band.
    pub ll: &'a CudaDeviceBuffer,
    /// HL input band.
    pub hl: &'a CudaDeviceBuffer,
    /// LH input band.
    pub lh: &'a CudaDeviceBuffer,
    /// HH input band.
    pub hh: &'a CudaDeviceBuffer,
    /// Output buffer for the reconstructed band.
    pub output: &'a CudaDeviceBuffer,
    /// IDWT geometry and transform metadata.
    pub job: CudaJ2kIdwtJob,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct CudaJ2kIdwtMultiKernelJob {
    pub(crate) ll_ptr: u64,
    pub(crate) hl_ptr: u64,
    pub(crate) lh_ptr: u64,
    pub(crate) hh_ptr: u64,
    pub(crate) output_ptr: u64,
    pub(crate) job: CudaJ2kIdwtJob,
}

/// Grayscale store dispatch from f32 component samples to tightly packed Gray8.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CudaJ2kStoreGray8Job {
    /// Source component buffer width in samples.
    pub input_width: u32,
    /// Source x offset in samples.
    pub source_x: u32,
    /// Source y offset in samples.
    pub source_y: u32,
    /// Number of samples copied per row.
    pub copy_width: u32,
    /// Number of rows copied.
    pub copy_height: u32,
    /// Destination output width in samples.
    pub output_width: u32,
    /// Destination output height in rows.
    pub output_height: u32,
    /// Destination x offset in samples.
    pub output_x: u32,
    /// Destination y offset in samples.
    pub output_y: u32,
    /// Level-shift addend applied before quantizing to Gray8.
    pub addend: f32,
    /// Source component bit depth.
    pub bit_depth: u32,
}

/// Grayscale store dispatch from f32 component samples to tightly packed Gray16.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CudaJ2kStoreGray16Job {
    /// Source component buffer width in samples.
    pub input_width: u32,
    /// Source x offset in samples.
    pub source_x: u32,
    /// Source y offset in samples.
    pub source_y: u32,
    /// Number of samples copied per row.
    pub copy_width: u32,
    /// Number of rows copied.
    pub copy_height: u32,
    /// Destination output width in samples.
    pub output_width: u32,
    /// Destination output height in rows.
    pub output_height: u32,
    /// Destination x offset in samples.
    pub output_x: u32,
    /// Destination y offset in samples.
    pub output_y: u32,
    /// Level-shift addend applied before quantizing to Gray16.
    pub addend: f32,
    /// Source component bit depth.
    pub bit_depth: u32,
}

/// In-place inverse MCT dispatch over three device f32 component planes.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CudaJ2kInverseMctJob {
    /// Number of samples in each component plane.
    pub len: u32,
    /// Nonzero for irreversible ICT; zero for reversible RCT.
    pub irreversible97: u32,
    /// Addend applied to output channel 0 after inverse MCT.
    pub addend0: f32,
    /// Addend applied to output channel 1 after inverse MCT.
    pub addend1: f32,
    /// Addend applied to output channel 2 after inverse MCT.
    pub addend2: f32,
}

/// RGB/RGBA store dispatch from three f32 component planes to packed 8-bit pixels.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CudaJ2kStoreRgb8Job {
    /// Source width for component 0.
    pub input_width0: u32,
    /// Source width for component 1.
    pub input_width1: u32,
    /// Source width for component 2.
    pub input_width2: u32,
    /// Source x offset for component 0.
    pub source_x0: u32,
    /// Source y offset for component 0.
    pub source_y0: u32,
    /// Source x offset for component 1.
    pub source_x1: u32,
    /// Source y offset for component 1.
    pub source_y1: u32,
    /// Source x offset for component 2.
    pub source_x2: u32,
    /// Source y offset for component 2.
    pub source_y2: u32,
    /// Number of pixels copied per row.
    pub copy_width: u32,
    /// Number of rows copied.
    pub copy_height: u32,
    /// Destination output width in pixels.
    pub output_width: u32,
    /// Destination output height in rows.
    pub output_height: u32,
    /// Destination x offset.
    pub output_x: u32,
    /// Destination y offset.
    pub output_y: u32,
    /// Addend applied to component 0 before quantizing.
    pub addend0: f32,
    /// Addend applied to component 1 before quantizing.
    pub addend1: f32,
    /// Addend applied to component 2 before quantizing.
    pub addend2: f32,
    /// Source bit depth for component 0.
    pub bit_depth0: u32,
    /// Source bit depth for component 1.
    pub bit_depth1: u32,
    /// Source bit depth for component 2.
    pub bit_depth2: u32,
    /// Nonzero to write RGBA8 with opaque alpha; zero writes RGB8.
    pub rgba: u32,
}

/// RGB/RGBA store dispatch from three f32 component planes to packed 16-bit pixels.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CudaJ2kStoreRgb16Job {
    /// Source width for component 0.
    pub input_width0: u32,
    /// Source width for component 1.
    pub input_width1: u32,
    /// Source width for component 2.
    pub input_width2: u32,
    /// Source x offset for component 0.
    pub source_x0: u32,
    /// Source y offset for component 0.
    pub source_y0: u32,
    /// Source x offset for component 1.
    pub source_x1: u32,
    /// Source y offset for component 1.
    pub source_y1: u32,
    /// Source x offset for component 2.
    pub source_x2: u32,
    /// Source y offset for component 2.
    pub source_y2: u32,
    /// Number of pixels copied per row.
    pub copy_width: u32,
    /// Number of rows copied.
    pub copy_height: u32,
    /// Destination output width in pixels.
    pub output_width: u32,
    /// Destination output height in rows.
    pub output_height: u32,
    /// Destination x offset.
    pub output_x: u32,
    /// Destination y offset.
    pub output_y: u32,
    /// Addend applied to component 0 before quantizing.
    pub addend0: f32,
    /// Addend applied to component 1 before quantizing.
    pub addend1: f32,
    /// Addend applied to component 2 before quantizing.
    pub addend2: f32,
    /// Source bit depth for component 0.
    pub bit_depth0: u32,
    /// Source bit depth for component 1.
    pub bit_depth1: u32,
    /// Source bit depth for component 2.
    pub bit_depth2: u32,
    /// Nonzero to write RGBA16 with opaque alpha; zero writes RGB16.
    pub rgba: u32,
}

/// Fused inverse RCT/ICT and packed RGB8/RGBA8 store dispatch.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CudaJ2kStoreRgb8MctJob {
    /// RGB/RGBA store geometry, addends, bit depths, and alpha mode.
    pub store: CudaJ2kStoreRgb8Job,
    /// Nonzero for irreversible ICT; zero for reversible RCT.
    pub irreversible97: u32,
}

/// One fused inverse MCT plus RGB8/RGBA8 store item for a batched dispatch.
#[derive(Clone, Copy, Debug)]
pub struct CudaJ2kStoreRgb8MctTarget<'a> {
    /// Source component plane 0.
    pub plane0: &'a CudaDeviceBuffer,
    /// Source component plane 1.
    pub plane1: &'a CudaDeviceBuffer,
    /// Source component plane 2.
    pub plane2: &'a CudaDeviceBuffer,
    /// Store geometry and inverse MCT parameters.
    pub job: CudaJ2kStoreRgb8MctJob,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct CudaJ2kStoreRgb8MctBatchJob {
    pub(crate) plane0_ptr: CuDevicePtr,
    pub(crate) plane1_ptr: CuDevicePtr,
    pub(crate) plane2_ptr: CuDevicePtr,
    pub(crate) output_ptr: CuDevicePtr,
    pub(crate) job: CudaJ2kStoreRgb8MctJob,
}

/// Fused inverse RCT/ICT and packed RGB16/RGBA16 store dispatch.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CudaJ2kStoreRgb16MctJob {
    /// RGB/RGBA store geometry, addends, bit depths, and alpha mode.
    pub store: CudaJ2kStoreRgb16Job,
    /// Nonzero for irreversible ICT; zero for reversible RCT.
    pub irreversible97: u32,
}

impl CudaContext {
    /// Apply one inverse JPEG 2000 DWT decomposition to device coefficient bands.
    pub fn j2k_inverse_dwt_single_device(
        &self,
        ll: &CudaDeviceBuffer,
        hl: &CudaDeviceBuffer,
        lh: &CudaDeviceBuffer,
        hh: &CudaDeviceBuffer,
        job: CudaJ2kIdwtJob,
    ) -> Result<CudaKernelOutput, CudaError> {
        self.j2k_inverse_dwt_single_device_impl(ll, hl, lh, hh, job, true)
    }

    /// Apply one inverse JPEG 2000 DWT decomposition without per-kernel synchronizes.
    pub fn j2k_inverse_dwt_single_device_untimed(
        &self,
        ll: &CudaDeviceBuffer,
        hl: &CudaDeviceBuffer,
        lh: &CudaDeviceBuffer,
        hh: &CudaDeviceBuffer,
        job: CudaJ2kIdwtJob,
    ) -> Result<CudaKernelOutput, CudaError> {
        self.j2k_inverse_dwt_single_device_impl(ll, hl, lh, hh, job, false)
    }

    /// Apply one inverse JPEG 2000 DWT decomposition with caller-owned
    /// transient buffer reuse.
    pub fn j2k_inverse_dwt_single_device_with_pool(
        &self,
        ll: &CudaDeviceBuffer,
        hl: &CudaDeviceBuffer,
        lh: &CudaDeviceBuffer,
        hh: &CudaDeviceBuffer,
        job: CudaJ2kIdwtJob,
        pool: &CudaBufferPool,
    ) -> Result<CudaPooledKernelOutput, CudaError> {
        self.j2k_inverse_dwt_single_device_with_pool_impl(ll, hl, lh, hh, job, true, pool)
    }

    /// Apply one inverse JPEG 2000 DWT decomposition with caller-owned
    /// transient buffer reuse and without per-kernel synchronizes.
    pub fn j2k_inverse_dwt_single_device_untimed_with_pool(
        &self,
        ll: &CudaDeviceBuffer,
        hl: &CudaDeviceBuffer,
        lh: &CudaDeviceBuffer,
        hh: &CudaDeviceBuffer,
        job: CudaJ2kIdwtJob,
        pool: &CudaBufferPool,
    ) -> Result<CudaPooledKernelOutput, CudaError> {
        self.j2k_inverse_dwt_single_device_with_pool_impl(ll, hl, lh, hh, job, false, pool)
    }

    /// Apply inverse JPEG 2000 DWT decompositions for multiple independent
    /// targets using one dispatch per parallel stage.
    pub fn j2k_inverse_dwt_batch_device_with_pool(
        &self,
        targets: &[CudaJ2kIdwtTarget<'_>],
        pool: &CudaBufferPool,
    ) -> Result<CudaExecutionStats, CudaError> {
        self.j2k_inverse_dwt_batch_device_with_pool_impl(targets, pool, true)
    }

    /// Apply inverse JPEG 2000 DWT decompositions for multiple independent
    /// targets without per-stage synchronizes.
    pub fn j2k_inverse_dwt_batch_device_untimed_with_pool(
        &self,
        targets: &[CudaJ2kIdwtTarget<'_>],
        pool: &CudaBufferPool,
    ) -> Result<CudaExecutionStats, CudaError> {
        self.j2k_inverse_dwt_batch_device_with_pool_impl(targets, pool, false)
    }

    /// Enqueue batched inverse JPEG 2000 DWT decompositions without
    /// synchronizing. The returned value must be kept live until the default
    /// stream has been synchronized by the caller.
    pub fn j2k_inverse_dwt_batch_device_enqueue_with_pool(
        &self,
        targets: &[CudaJ2kIdwtTarget<'_>],
        pool: &CudaBufferPool,
    ) -> Result<CudaQueuedExecution, CudaError> {
        self.inner.set_current()?;
        let kernel_jobs = j2k_idwt_multi_kernel_jobs(targets)?;
        if kernel_jobs.is_empty() {
            return Ok(CudaQueuedExecution {
                resources: Vec::new(),
                execution: CudaExecutionStats::default(),
            });
        }
        let jobs_buffer = pool.upload(idwt_multi_jobs_as_bytes(&kernel_jobs))?;
        let jobs_device = pooled_device_buffer(&jobs_buffer)?;
        let max_width = kernel_jobs
            .iter()
            .map(|job| job.job.rect.x1.saturating_sub(job.job.rect.x0))
            .max()
            .unwrap_or(0);
        let max_height = kernel_jobs
            .iter()
            .map(|job| job.job.rect.y1.saturating_sub(job.job.rect.y0))
            .max()
            .unwrap_or(0);
        let kernel_mode = idwt_batch_kernel_mode(&kernel_jobs, max_width, max_height);
        let interleave_horizontal_result = match kernel_mode {
            CudaJ2kIdwtBatchKernelMode::Cooperative53 => self
                .launch_j2k_idwt_interleave_horizontal_53_multi(
                    jobs_device,
                    max_height as usize,
                    kernel_jobs.len(),
                    false,
                ),
            CudaJ2kIdwtBatchKernelMode::Cooperative97 => self
                .launch_j2k_idwt_interleave_horizontal_97_multi_ptr(
                    jobs_device.device_ptr(),
                    max_width as usize,
                    max_height as usize,
                    kernel_jobs.len(),
                    false,
                ),
            CudaJ2kIdwtBatchKernelMode::Generic => self
                .launch_j2k_idwt_interleave_horizontal_multi(
                    jobs_device,
                    max_height as usize,
                    kernel_jobs.len(),
                    false,
                ),
        };
        if let Err(error) = interleave_horizontal_result {
            let _ = self.synchronize();
            return Err(error);
        }
        let vertical_result = match kernel_mode {
            CudaJ2kIdwtBatchKernelMode::Cooperative53 => self.launch_j2k_idwt_vertical_53_multi(
                jobs_device,
                max_width as usize,
                kernel_jobs.len(),
                false,
            ),
            CudaJ2kIdwtBatchKernelMode::Cooperative97 => self
                .launch_j2k_idwt_vertical_97_multi_ptr(
                    jobs_device.device_ptr(),
                    max_width as usize,
                    max_height as usize,
                    kernel_jobs.len(),
                    false,
                ),
            CudaJ2kIdwtBatchKernelMode::Generic => self.launch_j2k_idwt_vertical_multi(
                jobs_device,
                max_width as usize,
                kernel_jobs.len(),
                false,
            ),
        };
        if let Err(error) = vertical_result {
            let _ = self.synchronize();
            return Err(error);
        }

        Ok(CudaQueuedExecution {
            resources: vec![jobs_buffer],
            execution: CudaExecutionStats {
                kernel_dispatches: 2,
                copy_kernel_dispatches: 0,
                decode_kernel_dispatches: 2,
                hardware_decode: false,
            },
        })
    }

    /// Enqueue a sequence of batched inverse JPEG 2000 DWT stages while
    /// uploading all stage job metadata in one device buffer. The returned
    /// value must be kept live until the default stream has been synchronized
    /// by the caller.
    #[allow(clippy::too_many_lines)]
    pub fn j2k_inverse_dwt_batch_sequence_enqueue_with_pool(
        &self,
        target_batches: &[&[CudaJ2kIdwtTarget<'_>]],
        pool: &CudaBufferPool,
    ) -> Result<CudaQueuedExecution, CudaError> {
        self.inner.set_current()?;
        let mut all_jobs = Vec::new();
        let mut batches = Vec::new();
        for targets in target_batches {
            let kernel_jobs = j2k_idwt_multi_kernel_jobs(targets)?;
            if kernel_jobs.is_empty() {
                continue;
            }
            let start = all_jobs.len();
            let count = kernel_jobs.len();
            let max_width = kernel_jobs
                .iter()
                .map(|job| job.job.rect.x1.saturating_sub(job.job.rect.x0))
                .max()
                .unwrap_or(0);
            let max_height = kernel_jobs
                .iter()
                .map(|job| job.job.rect.y1.saturating_sub(job.job.rect.y0))
                .max()
                .unwrap_or(0);
            let kernel_mode = idwt_batch_kernel_mode(&kernel_jobs, max_width, max_height);
            all_jobs.extend(kernel_jobs);
            batches.push((start, count, max_width, max_height, kernel_mode));
        }
        if all_jobs.is_empty() {
            return Ok(CudaQueuedExecution {
                resources: Vec::new(),
                execution: CudaExecutionStats::default(),
            });
        }

        let jobs_buffer = pool.upload(idwt_multi_jobs_as_bytes(&all_jobs))?;
        let jobs_base = pooled_device_buffer(&jobs_buffer)?.device_ptr();
        let job_size = std::mem::size_of::<CudaJ2kIdwtMultiKernelJob>();
        let mut kernel_dispatches = 0usize;
        let trace_enabled = cuda_idwt_trace_enabled();
        for (stage_index, (start, count, max_width, max_height, kernel_mode)) in
            batches.into_iter().enumerate()
        {
            let byte_offset = start
                .checked_mul(job_size)
                .ok_or(CudaError::LengthTooLarge { len: start })?;
            let jobs_ptr = jobs_base
                .checked_add(byte_offset as u64)
                .ok_or(CudaError::LengthTooLarge { len: byte_offset })?;
            let trace_start = if trace_enabled {
                let event = self.create_event()?;
                event.record_default_stream()?;
                Some(event)
            } else {
                None
            };
            let interleave_horizontal_result = match kernel_mode {
                CudaJ2kIdwtBatchKernelMode::Cooperative53 => self
                    .launch_j2k_idwt_interleave_horizontal_53_multi_ptr(
                        jobs_ptr,
                        max_height as usize,
                        count,
                        false,
                    ),
                CudaJ2kIdwtBatchKernelMode::Cooperative97 => self
                    .launch_j2k_idwt_interleave_horizontal_97_multi_ptr(
                        jobs_ptr,
                        max_width as usize,
                        max_height as usize,
                        count,
                        false,
                    ),
                CudaJ2kIdwtBatchKernelMode::Generic => self
                    .launch_j2k_idwt_interleave_horizontal_multi_ptr(
                        jobs_ptr,
                        max_height as usize,
                        count,
                        false,
                    ),
            };
            if let Err(error) = interleave_horizontal_result {
                let _ = self.synchronize();
                return Err(error);
            }
            kernel_dispatches = kernel_dispatches.saturating_add(1);

            let vertical_result = match kernel_mode {
                CudaJ2kIdwtBatchKernelMode::Cooperative53 => self
                    .launch_j2k_idwt_vertical_53_multi_ptr(
                        jobs_ptr,
                        max_width as usize,
                        count,
                        false,
                    ),
                CudaJ2kIdwtBatchKernelMode::Cooperative97 => self
                    .launch_j2k_idwt_vertical_97_multi_ptr(
                        jobs_ptr,
                        max_width as usize,
                        max_height as usize,
                        count,
                        false,
                    ),
                CudaJ2kIdwtBatchKernelMode::Generic => self.launch_j2k_idwt_vertical_multi_ptr(
                    jobs_ptr,
                    max_width as usize,
                    count,
                    false,
                ),
            };
            if let Err(error) = vertical_result {
                let _ = self.synchronize();
                return Err(error);
            }
            kernel_dispatches = kernel_dispatches.saturating_add(1);
            if let Some(trace_start) = trace_start {
                let trace_end = self.create_event()?;
                trace_end.record_default_stream()?;
                trace_end.synchronize()?;
                let elapsed_us = elapsed_event_us_ceil(&trace_start, &trace_end)?;
                let end = start.saturating_add(count);
                let row = idwt_batch_trace_row(
                    stage_index,
                    &all_jobs[start..end],
                    max_width,
                    max_height,
                    kernel_mode,
                    elapsed_us,
                );
                eprintln!("{}", format_idwt_batch_trace_row(row));
            }
        }

        Ok(CudaQueuedExecution {
            resources: vec![jobs_buffer],
            execution: CudaExecutionStats {
                kernel_dispatches,
                copy_kernel_dispatches: 0,
                decode_kernel_dispatches: kernel_dispatches,
                hardware_decode: false,
            },
        })
    }

    fn j2k_inverse_dwt_batch_device_with_pool_impl(
        &self,
        targets: &[CudaJ2kIdwtTarget<'_>],
        pool: &CudaBufferPool,
        synchronize_each_launch: bool,
    ) -> Result<CudaExecutionStats, CudaError> {
        self.inner.set_current()?;
        let kernel_jobs = j2k_idwt_multi_kernel_jobs(targets)?;
        if kernel_jobs.is_empty() {
            return Ok(CudaExecutionStats::default());
        }
        let jobs_buffer = pool.upload(idwt_multi_jobs_as_bytes(&kernel_jobs))?;
        let jobs_device = pooled_device_buffer(&jobs_buffer)?;
        let max_width = kernel_jobs
            .iter()
            .map(|job| job.job.rect.x1.saturating_sub(job.job.rect.x0))
            .max()
            .unwrap_or(0);
        let max_height = kernel_jobs
            .iter()
            .map(|job| job.job.rect.y1.saturating_sub(job.job.rect.y0))
            .max()
            .unwrap_or(0);
        let kernel_mode = idwt_batch_kernel_mode(&kernel_jobs, max_width, max_height);
        let interleave_horizontal_result = match kernel_mode {
            CudaJ2kIdwtBatchKernelMode::Cooperative53 => self
                .launch_j2k_idwt_interleave_horizontal_53_multi(
                    jobs_device,
                    max_height as usize,
                    kernel_jobs.len(),
                    synchronize_each_launch,
                ),
            CudaJ2kIdwtBatchKernelMode::Cooperative97 => self
                .launch_j2k_idwt_interleave_horizontal_97_multi_ptr(
                    jobs_device.device_ptr(),
                    max_width as usize,
                    max_height as usize,
                    kernel_jobs.len(),
                    synchronize_each_launch,
                ),
            CudaJ2kIdwtBatchKernelMode::Generic => self
                .launch_j2k_idwt_interleave_horizontal_multi(
                    jobs_device,
                    max_height as usize,
                    kernel_jobs.len(),
                    synchronize_each_launch,
                ),
        };
        if let Err(error) = interleave_horizontal_result {
            if !synchronize_each_launch {
                let _ = self.synchronize();
            }
            return Err(error);
        }
        let vertical_result = match kernel_mode {
            CudaJ2kIdwtBatchKernelMode::Cooperative53 => self.launch_j2k_idwt_vertical_53_multi(
                jobs_device,
                max_width as usize,
                kernel_jobs.len(),
                synchronize_each_launch,
            ),
            CudaJ2kIdwtBatchKernelMode::Cooperative97 => self
                .launch_j2k_idwt_vertical_97_multi_ptr(
                    jobs_device.device_ptr(),
                    max_width as usize,
                    max_height as usize,
                    kernel_jobs.len(),
                    synchronize_each_launch,
                ),
            CudaJ2kIdwtBatchKernelMode::Generic => self.launch_j2k_idwt_vertical_multi(
                jobs_device,
                max_width as usize,
                kernel_jobs.len(),
                synchronize_each_launch,
            ),
        };
        if let Err(error) = vertical_result {
            if !synchronize_each_launch {
                let _ = self.synchronize();
            }
            return Err(error);
        }
        if !synchronize_each_launch {
            self.synchronize()?;
        }

        Ok(CudaExecutionStats {
            kernel_dispatches: 2,
            copy_kernel_dispatches: 0,
            decode_kernel_dispatches: 2,
            hardware_decode: false,
        })
    }

    fn j2k_inverse_dwt_single_device_impl(
        &self,
        ll: &CudaDeviceBuffer,
        hl: &CudaDeviceBuffer,
        lh: &CudaDeviceBuffer,
        hh: &CudaDeviceBuffer,
        job: CudaJ2kIdwtJob,
        synchronize_each_launch: bool,
    ) -> Result<CudaKernelOutput, CudaError> {
        let width = job.rect.x1.saturating_sub(job.rect.x0);
        let height = job.rect.y1.saturating_sub(job.rect.y0);
        let output_words = checked_image_words(width, height, 1)?;
        let output = self.allocate(checked_f32_words_byte_len(output_words)?)?;
        if output_words == 0 {
            return Ok(CudaKernelOutput {
                buffer: output,
                execution: CudaExecutionStats::default(),
            });
        }

        let job_buffer = self.upload(idwt_job_as_bytes(&job))?;
        let (horizontal_kernel, vertical_kernel) = if job.irreversible97 == 0 {
            (
                CudaKernel::J2kIdwtHorizontal53,
                CudaKernel::J2kIdwtVertical53,
            )
        } else {
            (
                CudaKernel::J2kIdwtHorizontal97,
                CudaKernel::J2kIdwtVertical97,
            )
        };
        if synchronize_each_launch {
            self.launch_j2k_idwt_interleave(
                [ll, hl, lh, hh],
                &output,
                &job_buffer,
                width,
                height,
                CudaLaunchMode::Sync,
            )?;
            self.launch_j2k_idwt_horizontal(
                horizontal_kernel,
                &output,
                &job_buffer,
                height as usize,
                CudaLaunchMode::Sync,
            )?;
            self.launch_j2k_idwt_vertical(
                vertical_kernel,
                &output,
                &job_buffer,
                width as usize,
                CudaLaunchMode::Sync,
            )?;
        } else {
            self.launch_j2k_idwt_interleave(
                [ll, hl, lh, hh],
                &output,
                &job_buffer,
                width,
                height,
                CudaLaunchMode::Async,
            )?;
            if let Err(error) = self.launch_j2k_idwt_horizontal(
                horizontal_kernel,
                &output,
                &job_buffer,
                height as usize,
                CudaLaunchMode::Async,
            ) {
                let _ = self.synchronize();
                return Err(error);
            }
            if let Err(error) = self.launch_j2k_idwt_vertical(
                vertical_kernel,
                &output,
                &job_buffer,
                width as usize,
                CudaLaunchMode::Async,
            ) {
                let _ = self.synchronize();
                return Err(error);
            }
            self.synchronize()?;
        }
        Ok(CudaKernelOutput {
            buffer: output,
            execution: CudaExecutionStats {
                kernel_dispatches: 3,
                copy_kernel_dispatches: 0,
                decode_kernel_dispatches: 3,
                hardware_decode: false,
            },
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn j2k_inverse_dwt_single_device_with_pool_impl(
        &self,
        ll: &CudaDeviceBuffer,
        hl: &CudaDeviceBuffer,
        lh: &CudaDeviceBuffer,
        hh: &CudaDeviceBuffer,
        job: CudaJ2kIdwtJob,
        synchronize_each_launch: bool,
        pool: &CudaBufferPool,
    ) -> Result<CudaPooledKernelOutput, CudaError> {
        let width = job.rect.x1.saturating_sub(job.rect.x0);
        let height = job.rect.y1.saturating_sub(job.rect.y0);
        let output_words = checked_image_words(width, height, 1)?;
        let output = pool.take(checked_f32_words_byte_len(output_words)?)?;
        let output_buffer = pooled_device_buffer(&output)?;
        if output_words == 0 {
            return Ok(CudaPooledKernelOutput {
                buffer: output,
                execution: CudaExecutionStats::default(),
            });
        }

        let job_buffer = pool.upload(idwt_job_as_bytes(&job))?;
        let job_device_buffer = pooled_device_buffer(&job_buffer)?;
        let (horizontal_kernel, vertical_kernel) = if job.irreversible97 == 0 {
            (
                CudaKernel::J2kIdwtHorizontal53,
                CudaKernel::J2kIdwtVertical53,
            )
        } else {
            (
                CudaKernel::J2kIdwtHorizontal97,
                CudaKernel::J2kIdwtVertical97,
            )
        };
        if synchronize_each_launch {
            self.launch_j2k_idwt_interleave(
                [ll, hl, lh, hh],
                output_buffer,
                job_device_buffer,
                width,
                height,
                CudaLaunchMode::Sync,
            )?;
            self.launch_j2k_idwt_horizontal(
                horizontal_kernel,
                output_buffer,
                job_device_buffer,
                height as usize,
                CudaLaunchMode::Sync,
            )?;
            self.launch_j2k_idwt_vertical(
                vertical_kernel,
                output_buffer,
                job_device_buffer,
                width as usize,
                CudaLaunchMode::Sync,
            )?;
        } else {
            self.launch_j2k_idwt_interleave(
                [ll, hl, lh, hh],
                output_buffer,
                job_device_buffer,
                width,
                height,
                CudaLaunchMode::Async,
            )?;
            if let Err(error) = self.launch_j2k_idwt_horizontal(
                horizontal_kernel,
                output_buffer,
                job_device_buffer,
                height as usize,
                CudaLaunchMode::Async,
            ) {
                let _ = self.synchronize();
                return Err(error);
            }
            if let Err(error) = self.launch_j2k_idwt_vertical(
                vertical_kernel,
                output_buffer,
                job_device_buffer,
                width as usize,
                CudaLaunchMode::Async,
            ) {
                let _ = self.synchronize();
                return Err(error);
            }
            self.synchronize()?;
        }
        Ok(CudaPooledKernelOutput {
            buffer: output,
            execution: CudaExecutionStats {
                kernel_dispatches: 3,
                copy_kernel_dispatches: 0,
                decode_kernel_dispatches: 3,
                hardware_decode: false,
            },
        })
    }

    /// Store a device f32 component plane as tightly packed Gray8 pixels.
    pub fn j2k_store_gray8_device(
        &self,
        input: &CudaDeviceBuffer,
        job: CudaJ2kStoreGray8Job,
    ) -> Result<CudaKernelOutput, CudaError> {
        let output_words = checked_image_words(job.output_width, job.output_height, 1)?;
        let output = self.allocate(output_words)?;
        if output_words == 0 {
            return Ok(CudaKernelOutput {
                buffer: output,
                execution: CudaExecutionStats::default(),
            });
        }
        let pixels = checked_image_words(job.copy_width, job.copy_height, 1)?;
        if pixels == 0 {
            return Ok(CudaKernelOutput {
                buffer: output,
                execution: CudaExecutionStats::default(),
            });
        }
        validate_store_rgb8_plane(
            input,
            job.input_width,
            job.source_x,
            job.source_y,
            job.copy_width,
            job.copy_height,
        )?;

        let job_buffer = self.upload(store_gray8_job_as_bytes(&job))?;
        self.launch_j2k_store_gray8(input, &output, &job_buffer, pixels)?;
        Ok(CudaKernelOutput {
            buffer: output,
            execution: CudaExecutionStats {
                kernel_dispatches: 1,
                copy_kernel_dispatches: 0,
                decode_kernel_dispatches: 1,
                hardware_decode: false,
            },
        })
    }

    /// Store a device f32 component plane as tightly packed Gray16 pixels.
    pub fn j2k_store_gray16_device(
        &self,
        input: &CudaDeviceBuffer,
        job: CudaJ2kStoreGray16Job,
    ) -> Result<CudaKernelOutput, CudaError> {
        let output_words = checked_image_words(job.output_width, job.output_height, 1)?;
        let output = self.allocate(
            output_words
                .checked_mul(std::mem::size_of::<u16>())
                .ok_or(CudaError::LengthTooLarge { len: output_words })?,
        )?;
        if output_words == 0 {
            return Ok(CudaKernelOutput {
                buffer: output,
                execution: CudaExecutionStats::default(),
            });
        }
        let pixels = checked_image_words(job.copy_width, job.copy_height, 1)?;
        if pixels == 0 {
            return Ok(CudaKernelOutput {
                buffer: output,
                execution: CudaExecutionStats::default(),
            });
        }
        validate_store_rgb8_plane(
            input,
            job.input_width,
            job.source_x,
            job.source_y,
            job.copy_width,
            job.copy_height,
        )?;

        let job_buffer = self.upload(store_gray16_job_as_bytes(&job))?;
        self.launch_j2k_store_gray16(input, &output, &job_buffer, pixels)?;
        Ok(CudaKernelOutput {
            buffer: output,
            execution: CudaExecutionStats {
                kernel_dispatches: 1,
                copy_kernel_dispatches: 0,
                decode_kernel_dispatches: 1,
                hardware_decode: false,
            },
        })
    }

    /// Apply inverse RCT/ICT in place on three device f32 component planes.
    pub fn j2k_inverse_mct_device(
        &self,
        plane0: &CudaDeviceBuffer,
        plane1: &CudaDeviceBuffer,
        plane2: &CudaDeviceBuffer,
        job: CudaJ2kInverseMctJob,
    ) -> Result<CudaExecutionStats, CudaError> {
        let bytes = (job.len as usize)
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
        if bytes > plane0.byte_len() || bytes > plane1.byte_len() || bytes > plane2.byte_len() {
            return Err(CudaError::LengthTooLarge { len: bytes });
        }
        if job.len == 0 {
            return Ok(CudaExecutionStats::default());
        }

        let job_buffer = self.upload(inverse_mct_job_as_bytes(&job))?;
        self.launch_j2k_inverse_mct(plane0, plane1, plane2, &job_buffer, job.len as usize)?;
        Ok(CudaExecutionStats {
            kernel_dispatches: 1,
            copy_kernel_dispatches: 0,
            decode_kernel_dispatches: 1,
            hardware_decode: false,
        })
    }

    /// Store three device f32 component planes as tightly packed RGB8/RGBA8.
    pub fn j2k_store_rgb8_device(
        &self,
        plane0: &CudaDeviceBuffer,
        plane1: &CudaDeviceBuffer,
        plane2: &CudaDeviceBuffer,
        job: CudaJ2kStoreRgb8Job,
    ) -> Result<CudaKernelOutput, CudaError> {
        let channels = if job.rgba == 0 { 3 } else { 4 };
        let output_bytes = checked_image_words(job.output_width, job.output_height, channels)?;
        let output = self.allocate(output_bytes)?;
        let pixels = checked_image_words(job.copy_width, job.copy_height, 1)?;
        if output_bytes == 0 || pixels == 0 {
            return Ok(CudaKernelOutput {
                buffer: output,
                execution: CudaExecutionStats::default(),
            });
        }
        validate_store_rgb8_plane(
            plane0,
            job.input_width0,
            job.source_x0,
            job.source_y0,
            job.copy_width,
            job.copy_height,
        )?;
        validate_store_rgb8_plane(
            plane1,
            job.input_width1,
            job.source_x1,
            job.source_y1,
            job.copy_width,
            job.copy_height,
        )?;
        validate_store_rgb8_plane(
            plane2,
            job.input_width2,
            job.source_x2,
            job.source_y2,
            job.copy_width,
            job.copy_height,
        )?;
        let dst_end = (job.output_x as usize)
            .checked_add(job.copy_width as usize)
            .zip((job.output_y as usize).checked_add(job.copy_height as usize))
            .ok_or(CudaError::LengthTooLarge { len: output_bytes })?;
        if dst_end.0 > job.output_width as usize || dst_end.1 > job.output_height as usize {
            return Err(CudaError::LengthTooLarge { len: output_bytes });
        }

        let job_buffer = self.upload(store_rgb8_job_as_bytes(&job))?;
        self.launch_j2k_store_rgb8(plane0, plane1, plane2, &output, &job_buffer, pixels)?;
        Ok(CudaKernelOutput {
            buffer: output,
            execution: CudaExecutionStats {
                kernel_dispatches: 1,
                copy_kernel_dispatches: 0,
                decode_kernel_dispatches: 1,
                hardware_decode: false,
            },
        })
    }

    /// Store three device f32 component planes as tightly packed RGB16/RGBA16.
    pub fn j2k_store_rgb16_device(
        &self,
        plane0: &CudaDeviceBuffer,
        plane1: &CudaDeviceBuffer,
        plane2: &CudaDeviceBuffer,
        job: CudaJ2kStoreRgb16Job,
    ) -> Result<CudaKernelOutput, CudaError> {
        let channels = if job.rgba == 0 { 3 } else { 4 };
        let output_samples = checked_image_words(job.output_width, job.output_height, channels)?;
        let output_bytes = output_samples
            .checked_mul(std::mem::size_of::<u16>())
            .ok_or(CudaError::LengthTooLarge {
                len: output_samples,
            })?;
        let output = self.allocate(output_bytes)?;
        let pixels = checked_image_words(job.copy_width, job.copy_height, 1)?;
        if output_bytes == 0 || pixels == 0 {
            return Ok(CudaKernelOutput {
                buffer: output,
                execution: CudaExecutionStats::default(),
            });
        }
        validate_store_rgb8_plane(
            plane0,
            job.input_width0,
            job.source_x0,
            job.source_y0,
            job.copy_width,
            job.copy_height,
        )?;
        validate_store_rgb8_plane(
            plane1,
            job.input_width1,
            job.source_x1,
            job.source_y1,
            job.copy_width,
            job.copy_height,
        )?;
        validate_store_rgb8_plane(
            plane2,
            job.input_width2,
            job.source_x2,
            job.source_y2,
            job.copy_width,
            job.copy_height,
        )?;
        let dst_end = (job.output_x as usize)
            .checked_add(job.copy_width as usize)
            .zip((job.output_y as usize).checked_add(job.copy_height as usize))
            .ok_or(CudaError::LengthTooLarge { len: output_bytes })?;
        if dst_end.0 > job.output_width as usize || dst_end.1 > job.output_height as usize {
            return Err(CudaError::LengthTooLarge { len: output_bytes });
        }

        let job_buffer = self.upload(store_rgb16_job_as_bytes(&job))?;
        self.launch_j2k_store_rgb16(plane0, plane1, plane2, &output, &job_buffer, pixels)?;
        Ok(CudaKernelOutput {
            buffer: output,
            execution: CudaExecutionStats {
                kernel_dispatches: 1,
                copy_kernel_dispatches: 0,
                decode_kernel_dispatches: 1,
                hardware_decode: false,
            },
        })
    }

    /// Apply inverse RCT/ICT and store tightly packed RGB8/RGBA8 in one dispatch.
    pub fn j2k_store_rgb8_mct_device(
        &self,
        plane0: &CudaDeviceBuffer,
        plane1: &CudaDeviceBuffer,
        plane2: &CudaDeviceBuffer,
        job: CudaJ2kStoreRgb8MctJob,
    ) -> Result<CudaKernelOutput, CudaError> {
        let batch = self.j2k_store_rgb8_mct_batch_device(&[CudaJ2kStoreRgb8MctTarget {
            plane0,
            plane1,
            plane2,
            job,
        }])?;
        let (mut outputs, execution) = batch.into_parts();
        let buffer = outputs.pop().ok_or_else(|| CudaError::InvalidArgument {
            message: "single RGB8 MCT batch store returned no output".to_string(),
        })?;
        Ok(CudaKernelOutput { buffer, execution })
    }

    /// Apply inverse RCT/ICT and store multiple tightly packed RGB8/RGBA8 images
    /// in one dispatch.
    pub fn j2k_store_rgb8_mct_batch_device(
        &self,
        targets: &[CudaJ2kStoreRgb8MctTarget<'_>],
    ) -> Result<CudaKernelBatchOutput, CudaError> {
        if targets.is_empty() {
            return Ok(CudaKernelBatchOutput {
                outputs: Vec::new(),
                execution: CudaExecutionStats::default(),
            });
        }

        let mut outputs = Vec::with_capacity(targets.len());
        let mut kernel_jobs = Vec::with_capacity(targets.len());
        let mut max_pixels = 0usize;
        for target in targets {
            let store = target.job.store;
            let channels = if store.rgba == 0 { 3 } else { 4 };
            let output_bytes =
                checked_image_words(store.output_width, store.output_height, channels)?;
            let output = self.allocate(output_bytes)?;
            let pixels = checked_image_words(store.copy_width, store.copy_height, 1)?;
            if output_bytes != 0 && pixels != 0 {
                validate_store_rgb8_plane(
                    target.plane0,
                    store.input_width0,
                    store.source_x0,
                    store.source_y0,
                    store.copy_width,
                    store.copy_height,
                )?;
                validate_store_rgb8_plane(
                    target.plane1,
                    store.input_width1,
                    store.source_x1,
                    store.source_y1,
                    store.copy_width,
                    store.copy_height,
                )?;
                validate_store_rgb8_plane(
                    target.plane2,
                    store.input_width2,
                    store.source_x2,
                    store.source_y2,
                    store.copy_width,
                    store.copy_height,
                )?;
                let dst_end = (store.output_x as usize)
                    .checked_add(store.copy_width as usize)
                    .zip((store.output_y as usize).checked_add(store.copy_height as usize))
                    .ok_or(CudaError::LengthTooLarge { len: output_bytes })?;
                if dst_end.0 > store.output_width as usize
                    || dst_end.1 > store.output_height as usize
                {
                    return Err(CudaError::LengthTooLarge { len: output_bytes });
                }
                max_pixels = max_pixels.max(pixels);
            }
            kernel_jobs.push(CudaJ2kStoreRgb8MctBatchJob {
                plane0_ptr: target.plane0.device_ptr(),
                plane1_ptr: target.plane1.device_ptr(),
                plane2_ptr: target.plane2.device_ptr(),
                output_ptr: output.device_ptr(),
                job: target.job,
            });
            outputs.push(output);
        }
        if max_pixels == 0 {
            return Ok(CudaKernelBatchOutput {
                outputs,
                execution: CudaExecutionStats::default(),
            });
        }

        let jobs_buffer = self.upload(store_rgb8_mct_batch_jobs_as_bytes(&kernel_jobs))?;
        self.launch_j2k_store_rgb8_mct_batch(&jobs_buffer, max_pixels, kernel_jobs.len())?;
        Ok(CudaKernelBatchOutput {
            outputs,
            execution: CudaExecutionStats {
                kernel_dispatches: 1,
                copy_kernel_dispatches: 0,
                decode_kernel_dispatches: 1,
                hardware_decode: false,
            },
        })
    }

    /// Apply inverse RCT/ICT and store multiple tightly packed RGB8/RGBA8 images
    /// into one contiguous device allocation in one dispatch.
    pub fn j2k_store_rgb8_mct_batch_contiguous_device(
        &self,
        targets: &[CudaJ2kStoreRgb8MctTarget<'_>],
    ) -> Result<CudaKernelContiguousBatchOutput, CudaError> {
        let mut ranges = Vec::with_capacity(targets.len());
        let mut total_bytes = 0usize;
        let mut max_pixels = 0usize;
        for target in targets {
            let store = target.job.store;
            let channels = if store.rgba == 0 { 3 } else { 4 };
            let output_bytes =
                checked_image_words(store.output_width, store.output_height, channels)?;
            let pixels = checked_image_words(store.copy_width, store.copy_height, 1)?;
            if output_bytes != 0 && pixels != 0 {
                validate_store_rgb8_plane(
                    target.plane0,
                    store.input_width0,
                    store.source_x0,
                    store.source_y0,
                    store.copy_width,
                    store.copy_height,
                )?;
                validate_store_rgb8_plane(
                    target.plane1,
                    store.input_width1,
                    store.source_x1,
                    store.source_y1,
                    store.copy_width,
                    store.copy_height,
                )?;
                validate_store_rgb8_plane(
                    target.plane2,
                    store.input_width2,
                    store.source_x2,
                    store.source_y2,
                    store.copy_width,
                    store.copy_height,
                )?;
                let dst_end = (store.output_x as usize)
                    .checked_add(store.copy_width as usize)
                    .zip((store.output_y as usize).checked_add(store.copy_height as usize))
                    .ok_or(CudaError::LengthTooLarge { len: output_bytes })?;
                if dst_end.0 > store.output_width as usize
                    || dst_end.1 > store.output_height as usize
                {
                    return Err(CudaError::LengthTooLarge { len: output_bytes });
                }
                max_pixels = max_pixels.max(pixels);
            }
            let offset = total_bytes;
            total_bytes = total_bytes
                .checked_add(output_bytes)
                .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
            ranges.push(CudaDeviceBufferRange {
                offset,
                len: output_bytes,
            });
        }

        let output = self.allocate(total_bytes)?;
        if targets.is_empty() || max_pixels == 0 {
            return Ok(CudaKernelContiguousBatchOutput {
                output,
                ranges,
                execution: CudaExecutionStats::default(),
            });
        }

        let base_ptr = output.device_ptr();
        let kernel_jobs = targets
            .iter()
            .zip(ranges.iter())
            .map(|(target, range)| {
                let output_ptr = base_ptr
                    .checked_add(
                        u64::try_from(range.offset)
                            .map_err(|_| CudaError::LengthTooLarge { len: range.offset })?,
                    )
                    .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
                Ok(CudaJ2kStoreRgb8MctBatchJob {
                    plane0_ptr: target.plane0.device_ptr(),
                    plane1_ptr: target.plane1.device_ptr(),
                    plane2_ptr: target.plane2.device_ptr(),
                    output_ptr,
                    job: target.job,
                })
            })
            .collect::<Result<Vec<_>, CudaError>>()?;
        let jobs_buffer = self.upload(store_rgb8_mct_batch_jobs_as_bytes(&kernel_jobs))?;
        self.launch_j2k_store_rgb8_mct_batch(&jobs_buffer, max_pixels, kernel_jobs.len())?;
        Ok(CudaKernelContiguousBatchOutput {
            output,
            ranges,
            execution: CudaExecutionStats {
                kernel_dispatches: 1,
                copy_kernel_dispatches: 0,
                decode_kernel_dispatches: 1,
                hardware_decode: false,
            },
        })
    }

    /// Apply inverse RCT/ICT and store tightly packed RGB16/RGBA16 in one dispatch.
    pub fn j2k_store_rgb16_mct_device(
        &self,
        plane0: &CudaDeviceBuffer,
        plane1: &CudaDeviceBuffer,
        plane2: &CudaDeviceBuffer,
        job: CudaJ2kStoreRgb16MctJob,
    ) -> Result<CudaKernelOutput, CudaError> {
        let store = job.store;
        let channels = if store.rgba == 0 { 3 } else { 4 };
        let output_samples =
            checked_image_words(store.output_width, store.output_height, channels)?;
        let output_bytes = output_samples
            .checked_mul(std::mem::size_of::<u16>())
            .ok_or(CudaError::LengthTooLarge {
                len: output_samples,
            })?;
        let output = self.allocate(output_bytes)?;
        let pixels = checked_image_words(store.copy_width, store.copy_height, 1)?;
        if output_bytes == 0 || pixels == 0 {
            return Ok(CudaKernelOutput {
                buffer: output,
                execution: CudaExecutionStats::default(),
            });
        }
        validate_store_rgb8_plane(
            plane0,
            store.input_width0,
            store.source_x0,
            store.source_y0,
            store.copy_width,
            store.copy_height,
        )?;
        validate_store_rgb8_plane(
            plane1,
            store.input_width1,
            store.source_x1,
            store.source_y1,
            store.copy_width,
            store.copy_height,
        )?;
        validate_store_rgb8_plane(
            plane2,
            store.input_width2,
            store.source_x2,
            store.source_y2,
            store.copy_width,
            store.copy_height,
        )?;
        let dst_end = (store.output_x as usize)
            .checked_add(store.copy_width as usize)
            .zip((store.output_y as usize).checked_add(store.copy_height as usize))
            .ok_or(CudaError::LengthTooLarge { len: output_bytes })?;
        if dst_end.0 > store.output_width as usize || dst_end.1 > store.output_height as usize {
            return Err(CudaError::LengthTooLarge { len: output_bytes });
        }

        let job_buffer = self.upload(store_rgb16_mct_job_as_bytes(&job))?;
        self.launch_j2k_store_rgb16_mct(plane0, plane1, plane2, &output, &job_buffer, pixels)?;
        Ok(CudaKernelOutput {
            buffer: output,
            execution: CudaExecutionStats {
                kernel_dispatches: 1,
                copy_kernel_dispatches: 0,
                decode_kernel_dispatches: 1,
                hardware_decode: false,
            },
        })
    }

    fn launch_j2k_idwt_interleave(
        &self,
        bands: [&CudaDeviceBuffer; 4],
        output: &CudaDeviceBuffer,
        job: &CudaDeviceBuffer,
        width: u32,
        height: u32,
        mode: CudaLaunchMode,
    ) -> Result<(), CudaError> {
        let [ll, hl, lh, hh] = bands;
        let mut low_low_ptr = ll.device_ptr();
        let mut high_low_ptr = hl.device_ptr();
        let mut low_high_ptr = lh.device_ptr();
        let mut high_high_ptr = hh.device_ptr();
        let mut output_ptr = output.device_ptr();
        let mut job_ptr = job.device_ptr();
        let mut params = cuda_kernel_params!(
            low_low_ptr,
            high_low_ptr,
            low_high_ptr,
            high_high_ptr,
            output_ptr,
            job_ptr
        );
        let geometry =
            j2k_dwt53_launch_geometry(width, height).ok_or(CudaError::ImageTooLarge {
                width,
                height,
                channels: 1,
            })?;
        self.launch_named_kernel(CudaKernel::J2kIdwtInterleave, geometry, &mut params, mode)
    }

    fn launch_j2k_idwt_interleave_horizontal_multi(
        &self,
        jobs: &CudaDeviceBuffer,
        max_rows: usize,
        job_count: usize,
        synchronize: bool,
    ) -> Result<(), CudaError> {
        self.launch_j2k_idwt_interleave_horizontal_multi_ptr(
            jobs.device_ptr(),
            max_rows,
            job_count,
            synchronize,
        )
    }

    fn launch_j2k_idwt_interleave_horizontal_multi_ptr(
        &self,
        jobs_ptr: CuDevicePtr,
        max_rows: usize,
        job_count: usize,
        synchronize: bool,
    ) -> Result<(), CudaError> {
        let mut jobs_ptr = jobs_ptr;
        let mut params = cuda_kernel_params!(jobs_ptr);
        let geometry = j2k_idwt_multi_1d_launch_geometry(max_rows, job_count)
            .ok_or(CudaError::LengthTooLarge { len: job_count })?;
        self.launch_named_kernel(
            CudaKernel::J2kIdwtInterleaveHorizontalMulti,
            geometry,
            &mut params,
            CudaLaunchMode::from_synchronize(synchronize),
        )
    }

    fn launch_j2k_idwt_interleave_horizontal_53_multi(
        &self,
        jobs: &CudaDeviceBuffer,
        max_rows: usize,
        job_count: usize,
        synchronize: bool,
    ) -> Result<(), CudaError> {
        self.launch_j2k_idwt_interleave_horizontal_53_multi_ptr(
            jobs.device_ptr(),
            max_rows,
            job_count,
            synchronize,
        )
    }

    fn launch_j2k_idwt_interleave_horizontal_53_multi_ptr(
        &self,
        jobs_ptr: CuDevicePtr,
        max_rows: usize,
        job_count: usize,
        synchronize: bool,
    ) -> Result<(), CudaError> {
        let mut jobs_ptr = jobs_ptr;
        let mut params = cuda_kernel_params!(jobs_ptr);
        let geometry = j2k_idwt_multi_coop_launch_geometry(max_rows, job_count)
            .ok_or(CudaError::LengthTooLarge { len: job_count })?;
        self.launch_named_kernel(
            CudaKernel::J2kIdwtInterleaveHorizontal53Multi,
            geometry,
            &mut params,
            CudaLaunchMode::from_synchronize(synchronize),
        )
    }

    fn launch_j2k_idwt_interleave_horizontal_97_multi_ptr(
        &self,
        jobs_ptr: CuDevicePtr,
        max_width: usize,
        max_rows: usize,
        job_count: usize,
        synchronize: bool,
    ) -> Result<(), CudaError> {
        let mut jobs_ptr = jobs_ptr;
        let mut params = cuda_kernel_params!(jobs_ptr);
        let geometry = j2k_idwt_multi_coop_axis_launch_geometry(max_rows, max_width, job_count)
            .ok_or(CudaError::LengthTooLarge { len: job_count })?;
        self.launch_named_kernel(
            CudaKernel::J2kIdwtInterleaveHorizontal97Multi,
            geometry,
            &mut params,
            CudaLaunchMode::from_synchronize(synchronize),
        )
    }

    fn launch_j2k_idwt_horizontal(
        &self,
        kernel: CudaKernel,
        output: &CudaDeviceBuffer,
        job: &CudaDeviceBuffer,
        rows: usize,
        mode: CudaLaunchMode,
    ) -> Result<(), CudaError> {
        let mut output_ptr = output.device_ptr();
        let mut job_ptr = job.device_ptr();
        let mut params = cuda_kernel_params!(output_ptr, job_ptr);
        let geometry =
            j2k_forward_rct_launch_geometry(rows).ok_or(CudaError::LengthTooLarge { len: rows })?;
        self.launch_named_kernel(kernel, geometry, &mut params, mode)
    }

    fn launch_j2k_idwt_vertical(
        &self,
        kernel: CudaKernel,
        output: &CudaDeviceBuffer,
        job: &CudaDeviceBuffer,
        columns: usize,
        mode: CudaLaunchMode,
    ) -> Result<(), CudaError> {
        let mut output_ptr = output.device_ptr();
        let mut job_ptr = job.device_ptr();
        let mut params = cuda_kernel_params!(output_ptr, job_ptr);
        let geometry = j2k_forward_rct_launch_geometry(columns)
            .ok_or(CudaError::LengthTooLarge { len: columns })?;
        self.launch_named_kernel(kernel, geometry, &mut params, mode)
    }

    fn launch_j2k_idwt_vertical_multi(
        &self,
        jobs: &CudaDeviceBuffer,
        max_columns: usize,
        job_count: usize,
        synchronize: bool,
    ) -> Result<(), CudaError> {
        self.launch_j2k_idwt_vertical_multi_ptr(
            jobs.device_ptr(),
            max_columns,
            job_count,
            synchronize,
        )
    }

    fn launch_j2k_idwt_vertical_multi_ptr(
        &self,
        jobs_ptr: CuDevicePtr,
        max_columns: usize,
        job_count: usize,
        synchronize: bool,
    ) -> Result<(), CudaError> {
        let mut jobs_ptr = jobs_ptr;
        let mut params = cuda_kernel_params!(jobs_ptr);
        let geometry = j2k_idwt_multi_1d_launch_geometry(max_columns, job_count)
            .ok_or(CudaError::LengthTooLarge { len: job_count })?;
        self.launch_named_kernel(
            CudaKernel::J2kIdwtVerticalMulti,
            geometry,
            &mut params,
            CudaLaunchMode::from_synchronize(synchronize),
        )
    }

    fn launch_j2k_idwt_vertical_53_multi(
        &self,
        jobs: &CudaDeviceBuffer,
        max_columns: usize,
        job_count: usize,
        synchronize: bool,
    ) -> Result<(), CudaError> {
        self.launch_j2k_idwt_vertical_53_multi_ptr(
            jobs.device_ptr(),
            max_columns,
            job_count,
            synchronize,
        )
    }

    fn launch_j2k_idwt_vertical_53_multi_ptr(
        &self,
        jobs_ptr: CuDevicePtr,
        max_columns: usize,
        job_count: usize,
        synchronize: bool,
    ) -> Result<(), CudaError> {
        let mut jobs_ptr = jobs_ptr;
        let mut params = cuda_kernel_params!(jobs_ptr);
        let geometry = j2k_idwt_multi_coop_launch_geometry(max_columns, job_count)
            .ok_or(CudaError::LengthTooLarge { len: job_count })?;
        self.launch_named_kernel(
            CudaKernel::J2kIdwtVertical53Multi,
            geometry,
            &mut params,
            CudaLaunchMode::from_synchronize(synchronize),
        )
    }

    fn launch_j2k_idwt_vertical_97_multi_ptr(
        &self,
        jobs_ptr: CuDevicePtr,
        max_columns: usize,
        max_height: usize,
        job_count: usize,
        synchronize: bool,
    ) -> Result<(), CudaError> {
        const COLUMNS_PER_BLOCK: usize = 4;
        const MIN_COLS4_JOBS: usize = 64;
        let (kernel, geometry) = if job_count >= MIN_COLS4_JOBS && max_height <= 256 {
            let geometry = j2k_idwt_multi_coop_columns_launch_geometry(
                max_columns,
                max_height,
                job_count,
                COLUMNS_PER_BLOCK,
            )
            .ok_or(CudaError::LengthTooLarge { len: job_count })?;
            (CudaKernel::J2kIdwtVertical97MultiCols4, geometry)
        } else {
            let geometry =
                j2k_idwt_multi_coop_axis_launch_geometry(max_columns, max_height, job_count)
                    .ok_or(CudaError::LengthTooLarge { len: job_count })?;
            (CudaKernel::J2kIdwtVertical97Multi, geometry)
        };
        let mut jobs_ptr = jobs_ptr;
        let mut params = cuda_kernel_params!(jobs_ptr);
        self.launch_named_kernel(
            kernel,
            geometry,
            &mut params,
            CudaLaunchMode::from_synchronize(synchronize),
        )
    }

    fn launch_j2k_store_gray8(
        &self,
        input: &CudaDeviceBuffer,
        output: &CudaDeviceBuffer,
        job: &CudaDeviceBuffer,
        pixels: usize,
    ) -> Result<(), CudaError> {
        let function = self.inner.kernel_function(CudaKernel::J2kStoreGray8)?;
        let mut input_ptr = input.device_ptr();
        let mut output_ptr = output.device_ptr();
        let mut job_ptr = job.device_ptr();
        let mut params = cuda_kernel_params!(input_ptr, output_ptr, job_ptr);
        let geometry = j2k_forward_rct_launch_geometry(pixels)
            .ok_or(CudaError::LengthTooLarge { len: pixels })?;
        self.launch_kernel(function, geometry, &mut params)
    }

    fn launch_j2k_store_gray16(
        &self,
        input: &CudaDeviceBuffer,
        output: &CudaDeviceBuffer,
        job: &CudaDeviceBuffer,
        pixels: usize,
    ) -> Result<(), CudaError> {
        let function = self.inner.kernel_function(CudaKernel::J2kStoreGray16)?;
        let mut input_ptr = input.device_ptr();
        let mut output_ptr = output.device_ptr();
        let mut job_ptr = job.device_ptr();
        let mut params = cuda_kernel_params!(input_ptr, output_ptr, job_ptr);
        let geometry = j2k_forward_rct_launch_geometry(pixels)
            .ok_or(CudaError::LengthTooLarge { len: pixels })?;
        self.launch_kernel(function, geometry, &mut params)
    }

    fn launch_j2k_inverse_mct(
        &self,
        plane0: &CudaDeviceBuffer,
        plane1: &CudaDeviceBuffer,
        plane2: &CudaDeviceBuffer,
        job: &CudaDeviceBuffer,
        len: usize,
    ) -> Result<(), CudaError> {
        let function = self.inner.kernel_function(CudaKernel::J2kInverseMct)?;
        let mut plane0_ptr = plane0.device_ptr();
        let mut plane1_ptr = plane1.device_ptr();
        let mut plane2_ptr = plane2.device_ptr();
        let mut job_ptr = job.device_ptr();
        let mut params = cuda_kernel_params!(plane0_ptr, plane1_ptr, plane2_ptr, job_ptr);
        let geometry =
            j2k_forward_rct_launch_geometry(len).ok_or(CudaError::LengthTooLarge { len })?;
        self.launch_kernel(function, geometry, &mut params)
    }

    fn launch_j2k_store_rgb8(
        &self,
        plane0: &CudaDeviceBuffer,
        plane1: &CudaDeviceBuffer,
        plane2: &CudaDeviceBuffer,
        output: &CudaDeviceBuffer,
        job: &CudaDeviceBuffer,
        pixels: usize,
    ) -> Result<(), CudaError> {
        let function = self.inner.kernel_function(CudaKernel::J2kStoreRgb8)?;
        let mut plane0_ptr = plane0.device_ptr();
        let mut plane1_ptr = plane1.device_ptr();
        let mut plane2_ptr = plane2.device_ptr();
        let mut output_ptr = output.device_ptr();
        let mut job_ptr = job.device_ptr();
        let mut params =
            cuda_kernel_params!(plane0_ptr, plane1_ptr, plane2_ptr, output_ptr, job_ptr);
        let geometry = j2k_forward_rct_launch_geometry(pixels)
            .ok_or(CudaError::LengthTooLarge { len: pixels })?;
        self.launch_kernel(function, geometry, &mut params)
    }

    fn launch_j2k_store_rgb16(
        &self,
        plane0: &CudaDeviceBuffer,
        plane1: &CudaDeviceBuffer,
        plane2: &CudaDeviceBuffer,
        output: &CudaDeviceBuffer,
        job: &CudaDeviceBuffer,
        pixels: usize,
    ) -> Result<(), CudaError> {
        let function = self.inner.kernel_function(CudaKernel::J2kStoreRgb16)?;
        let mut plane0_ptr = plane0.device_ptr();
        let mut plane1_ptr = plane1.device_ptr();
        let mut plane2_ptr = plane2.device_ptr();
        let mut output_ptr = output.device_ptr();
        let mut job_ptr = job.device_ptr();
        let mut params =
            cuda_kernel_params!(plane0_ptr, plane1_ptr, plane2_ptr, output_ptr, job_ptr);
        let geometry = j2k_forward_rct_launch_geometry(pixels)
            .ok_or(CudaError::LengthTooLarge { len: pixels })?;
        self.launch_kernel(function, geometry, &mut params)
    }

    fn launch_j2k_store_rgb8_mct_batch(
        &self,
        jobs: &CudaDeviceBuffer,
        max_pixels: usize,
        job_count: usize,
    ) -> Result<(), CudaError> {
        let function = self
            .inner
            .kernel_function(CudaKernel::J2kStoreRgb8MctBatch)?;
        let mut jobs_ptr = jobs.device_ptr();
        let mut params = cuda_kernel_params!(jobs_ptr);
        let geometry = j2k_store_batch_launch_geometry(max_pixels, job_count)
            .ok_or(CudaError::LengthTooLarge { len: max_pixels })?;
        self.launch_kernel(function, geometry, &mut params)
    }

    fn launch_j2k_store_rgb16_mct(
        &self,
        plane0: &CudaDeviceBuffer,
        plane1: &CudaDeviceBuffer,
        plane2: &CudaDeviceBuffer,
        output: &CudaDeviceBuffer,
        job: &CudaDeviceBuffer,
        pixels: usize,
    ) -> Result<(), CudaError> {
        let function = self.inner.kernel_function(CudaKernel::J2kStoreRgb16Mct)?;
        let mut plane0_ptr = plane0.device_ptr();
        let mut plane1_ptr = plane1.device_ptr();
        let mut plane2_ptr = plane2.device_ptr();
        let mut output_ptr = output.device_ptr();
        let mut job_ptr = job.device_ptr();
        let mut params =
            cuda_kernel_params!(plane0_ptr, plane1_ptr, plane2_ptr, output_ptr, job_ptr);
        let geometry = j2k_forward_rct_launch_geometry(pixels)
            .ok_or(CudaError::LengthTooLarge { len: pixels })?;
        self.launch_kernel(function, geometry, &mut params)
    }
}

/// Device-resident interleaved JPEG 2000 input pixels with row stride metadata.
#[derive(Clone, Copy, Debug)]
pub struct CudaJ2kStridedInterleavedPixels<'a> {
    /// Backing CUDA device byte buffer.
    pub buffer: &'a CudaDeviceBuffer,
    /// Byte offset to the first pixel in `buffer`.
    pub byte_offset: usize,
    /// Active input width in pixels.
    pub width: u32,
    /// Active input height in pixels.
    pub height: u32,
    /// Bytes between the start of consecutive rows.
    pub pitch_bytes: usize,
    /// Number of interleaved components per pixel.
    pub num_components: u8,
    /// Integer sample precision.
    pub bit_depth: u8,
    /// Whether integer samples are signed.
    pub signed: bool,
}

pub(crate) fn active_dwt53_buffers<'a>(
    buffer_a: &'a CudaDeviceBuffer,
    buffer_b: &'a CudaDeviceBuffer,
    active_is_a: bool,
) -> (&'a CudaDeviceBuffer, &'a CudaDeviceBuffer) {
    if active_is_a {
        (buffer_a, buffer_b)
    } else {
        (buffer_b, buffer_a)
    }
}

pub(crate) fn j2k_idwt_multi_kernel_jobs(
    targets: &[CudaJ2kIdwtTarget<'_>],
) -> Result<Vec<CudaJ2kIdwtMultiKernelJob>, CudaError> {
    let mut kernel_jobs = Vec::with_capacity(targets.len());
    for target in targets {
        let width = target.job.rect.x1.saturating_sub(target.job.rect.x0);
        let height = target.job.rect.y1.saturating_sub(target.job.rect.y0);
        if width == 0 || height == 0 {
            continue;
        }
        ensure_idwt_buffer_len(target.output, target.job.rect)?;
        ensure_idwt_buffer_len(target.ll, target.job.ll_rect)?;
        ensure_idwt_buffer_len(target.hl, target.job.hl_rect)?;
        ensure_idwt_buffer_len(target.lh, target.job.lh_rect)?;
        ensure_idwt_buffer_len(target.hh, target.job.hh_rect)?;
        kernel_jobs.push(CudaJ2kIdwtMultiKernelJob {
            ll_ptr: target.ll.device_ptr(),
            hl_ptr: target.hl.device_ptr(),
            lh_ptr: target.lh.device_ptr(),
            hh_ptr: target.hh.device_ptr(),
            output_ptr: target.output.device_ptr(),
            job: target.job,
        });
    }
    Ok(kernel_jobs)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CudaJ2kIdwtBatchKernelMode {
    Generic,
    Cooperative53,
    Cooperative97,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct CudaJ2kIdwtBatchTraceRow {
    pub(crate) stage_index: usize,
    pub(crate) mode: CudaJ2kIdwtBatchKernelMode,
    pub(crate) job_count: usize,
    pub(crate) max_width: u32,
    pub(crate) max_height: u32,
    pub(crate) min_width: u32,
    pub(crate) min_height: u32,
    pub(crate) total_pixels: u64,
    pub(crate) irreversible_jobs: usize,
    pub(crate) elapsed_us: u128,
}

pub(crate) fn idwt_batch_kernel_mode(
    kernel_jobs: &[CudaJ2kIdwtMultiKernelJob],
    max_width: u32,
    max_height: u32,
) -> CudaJ2kIdwtBatchKernelMode {
    const MAX_COOPERATIVE_DIMENSION: u32 = 512;
    const MIN_COOPERATIVE_53_DIMENSION: u32 = 128;
    const MIN_COOPERATIVE_97_DIMENSION: u32 = 64;
    let bounded_cooperative_shape =
        max_width <= MAX_COOPERATIVE_DIMENSION && max_height <= MAX_COOPERATIVE_DIMENSION;
    if !bounded_cooperative_shape {
        return CudaJ2kIdwtBatchKernelMode::Generic;
    }
    if kernel_jobs.iter().all(|job| job.job.irreversible97 == 0) {
        if max_width >= MIN_COOPERATIVE_53_DIMENSION && max_height >= MIN_COOPERATIVE_53_DIMENSION {
            CudaJ2kIdwtBatchKernelMode::Cooperative53
        } else {
            CudaJ2kIdwtBatchKernelMode::Generic
        }
    } else if kernel_jobs.iter().all(|job| job.job.irreversible97 != 0) {
        if max_width >= MIN_COOPERATIVE_97_DIMENSION && max_height >= MIN_COOPERATIVE_97_DIMENSION {
            CudaJ2kIdwtBatchKernelMode::Cooperative97
        } else {
            CudaJ2kIdwtBatchKernelMode::Generic
        }
    } else {
        CudaJ2kIdwtBatchKernelMode::Generic
    }
}

pub(crate) fn idwt_batch_trace_row(
    stage_index: usize,
    kernel_jobs: &[CudaJ2kIdwtMultiKernelJob],
    max_width: u32,
    max_height: u32,
    mode: CudaJ2kIdwtBatchKernelMode,
    elapsed_us: u128,
) -> CudaJ2kIdwtBatchTraceRow {
    let mut min_width = u32::MAX;
    let mut min_height = u32::MAX;
    let mut total_pixels = 0u64;
    let mut irreversible_jobs = 0usize;
    for kernel_job in kernel_jobs {
        let width = kernel_job
            .job
            .rect
            .x1
            .saturating_sub(kernel_job.job.rect.x0);
        let height = kernel_job
            .job
            .rect
            .y1
            .saturating_sub(kernel_job.job.rect.y0);
        min_width = min_width.min(width);
        min_height = min_height.min(height);
        total_pixels =
            total_pixels.saturating_add(u64::from(width).saturating_mul(u64::from(height)));
        if kernel_job.job.irreversible97 != 0 {
            irreversible_jobs = irreversible_jobs.saturating_add(1);
        }
    }
    if kernel_jobs.is_empty() {
        min_width = 0;
        min_height = 0;
    }
    CudaJ2kIdwtBatchTraceRow {
        stage_index,
        mode,
        job_count: kernel_jobs.len(),
        max_width,
        max_height,
        min_width,
        min_height,
        total_pixels,
        irreversible_jobs,
        elapsed_us,
    }
}

pub(crate) fn format_idwt_batch_trace_row(row: CudaJ2kIdwtBatchTraceRow) -> String {
    format!(
        "j2k_profile codec=j2k op=cuda_idwt_batch path=decode \
         stage_index={} mode={:?} job_count={} max_width={} max_height={} \
         min_width={} min_height={} total_pixels={} irreversible_jobs={} elapsed_us={}",
        row.stage_index,
        row.mode,
        row.job_count,
        row.max_width,
        row.max_height,
        row.min_width,
        row.min_height,
        row.total_pixels,
        row.irreversible_jobs,
        row.elapsed_us
    )
}

#[cfg(test)]
pub(crate) fn idwt_batch_uses_cooperative_53(
    kernel_jobs: &[CudaJ2kIdwtMultiKernelJob],
    max_width: u32,
    max_height: u32,
) -> bool {
    idwt_batch_kernel_mode(kernel_jobs, max_width, max_height)
        == CudaJ2kIdwtBatchKernelMode::Cooperative53
}

pub(crate) fn ensure_idwt_buffer_len(
    buffer: &CudaDeviceBuffer,
    rect: CudaJ2kRect,
) -> Result<(), CudaError> {
    let width = rect.x1.saturating_sub(rect.x0);
    let height = rect.y1.saturating_sub(rect.y0);
    let words = checked_image_words(width, height, 1)?;
    let bytes = checked_f32_words_byte_len(words)?;
    if bytes > buffer.byte_len() {
        return Err(CudaError::OutputTooSmall {
            required: bytes,
            have: buffer.byte_len(),
        });
    }
    Ok(())
}

pub(crate) fn checked_f32_words_byte_len(words: usize) -> Result<usize, CudaError> {
    words
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or(CudaError::LengthTooLarge { len: words })
}

pub(crate) fn validate_store_rgb8_plane(
    plane: &CudaDeviceBuffer,
    input_width: u32,
    source_x: u32,
    source_y: u32,
    copy_width: u32,
    copy_height: u32,
) -> Result<(), CudaError> {
    if source_x
        .checked_add(copy_width)
        .is_none_or(|end_x| end_x > input_width)
    {
        return Err(CudaError::LengthTooLarge {
            len: plane.byte_len(),
        });
    }
    let last_sample = if copy_height == 0 {
        0
    } else {
        (source_y as usize)
            .checked_add(copy_height as usize - 1)
            .and_then(|row| row.checked_mul(input_width as usize))
            .and_then(|row| row.checked_add(source_x as usize))
            .and_then(|row| row.checked_add(copy_width as usize))
            .ok_or(CudaError::LengthTooLarge {
                len: plane.byte_len(),
            })?
    };
    let required_bytes =
        last_sample
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or(CudaError::LengthTooLarge {
                len: plane.byte_len(),
            })?;
    if required_bytes > plane.byte_len() {
        return Err(CudaError::LengthTooLarge {
            len: required_bytes,
        });
    }
    Ok(())
}
