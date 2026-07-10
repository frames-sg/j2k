// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    bytes::{
        inverse_mct_job_as_bytes, store_gray16_job_as_bytes, store_gray8_job_as_bytes,
        store_rgb16_job_as_bytes, store_rgb16_mct_job_as_bytes, store_rgb8_job_as_bytes,
        store_rgb8_mct_batch_jobs_as_bytes,
    },
    context::CudaContext,
    error::CudaError,
    execution::{
        CudaExecutionStats, CudaKernelBatchOutput, CudaKernelContiguousBatchOutput,
        CudaKernelOutput,
    },
    memory::{checked_image_words, CudaDeviceBuffer, CudaDeviceBufferRange},
};

use super::{
    types::{
        CudaJ2kInverseMctJob, CudaJ2kStoreGray16Job, CudaJ2kStoreGray8Job, CudaJ2kStoreRgb16Job,
        CudaJ2kStoreRgb16MctJob, CudaJ2kStoreRgb8Job, CudaJ2kStoreRgb8MctBatchJob,
        CudaJ2kStoreRgb8MctJob, CudaJ2kStoreRgb8MctTarget,
    },
    validate_store_rgb8_plane,
};

impl CudaContext {
    /// Store a device f32 component plane as tightly packed Gray8 pixels.
    #[doc(hidden)]
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
    #[doc(hidden)]
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
    #[doc(hidden)]
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
    #[doc(hidden)]
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
    #[doc(hidden)]
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
    #[doc(hidden)]
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
    #[doc(hidden)]
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
    #[doc(hidden)]
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
    #[doc(hidden)]
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
}
