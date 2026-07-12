// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    bytes::{
        inverse_mct_job_as_bytes, store_gray16_job_as_bytes, store_gray8_job_as_bytes,
        store_rgb16_job_as_bytes, store_rgb16_mct_job_as_bytes, store_rgb8_job_as_bytes,
    },
    context::CudaContext,
    error::CudaError,
    execution::{CudaExecutionStats, CudaKernelOutput},
    memory::{checked_image_words, CudaDeviceBuffer},
};

use super::types::{
    CudaJ2kInverseMctJob, CudaJ2kStoreGray16Job, CudaJ2kStoreGray8Job, CudaJ2kStoreRgb16Job,
    CudaJ2kStoreRgb16MctJob, CudaJ2kStoreRgb8Job, CudaJ2kStoreRgb8MctJob,
    CudaJ2kStoreRgb8MctTarget,
};

mod batch;
mod destination;
mod validation;

use destination::{validate_store_destination, zero_unwritten_store_output};
use validation::{
    validate_inverse_mct_planes_disjoint, validate_store_buffer_context, validate_store_plane,
};

impl CudaContext {
    /// Store a device f32 component plane as tightly packed Gray8 pixels.
    #[doc(hidden)]
    pub fn j2k_store_gray8_device(
        &self,
        input: &CudaDeviceBuffer,
        job: CudaJ2kStoreGray8Job,
    ) -> Result<CudaKernelOutput, CudaError> {
        validate_store_buffer_context(self, [input])?;
        let output_bytes = checked_image_words(job.output_width, job.output_height, 1)?;
        let pixels = checked_image_words(job.copy_width, job.copy_height, 1)?;
        let full_coverage = validate_store_destination(
            job.output_width,
            job.output_height,
            job.output_x,
            job.output_y,
            job.copy_width,
            job.copy_height,
            1,
        )?;
        if pixels != 0 {
            validate_store_plane(
                input,
                job.input_width,
                job.source_x,
                job.source_y,
                job.copy_width,
                job.copy_height,
            )?;
        }
        let output = self.allocate(output_bytes)?;
        if output_bytes == 0 || pixels == 0 {
            let zero_fill_enqueued =
                zero_unwritten_store_output(self, &output, output_bytes, full_coverage)?;
            if zero_fill_enqueued {
                self.synchronize()?;
            }
            return Ok(CudaKernelOutput {
                buffer: output,
                execution: CudaExecutionStats::default(),
            });
        }

        let job_buffer = self.upload(store_gray8_job_as_bytes(&job))?;
        if zero_unwritten_store_output(self, &output, output_bytes, full_coverage)? {
            self.synchronize()?;
        }
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
        validate_store_buffer_context(self, [input])?;
        let output_samples = checked_image_words(job.output_width, job.output_height, 1)?;
        let output_bytes = output_samples
            .checked_mul(std::mem::size_of::<u16>())
            .ok_or(CudaError::LengthTooLarge {
                len: output_samples,
            })?;
        let pixels = checked_image_words(job.copy_width, job.copy_height, 1)?;
        let full_coverage = validate_store_destination(
            job.output_width,
            job.output_height,
            job.output_x,
            job.output_y,
            job.copy_width,
            job.copy_height,
            1,
        )?;
        if pixels != 0 {
            validate_store_plane(
                input,
                job.input_width,
                job.source_x,
                job.source_y,
                job.copy_width,
                job.copy_height,
            )?;
        }
        let output = self.allocate(output_bytes)?;
        if output_bytes == 0 || pixels == 0 {
            let zero_fill_enqueued =
                zero_unwritten_store_output(self, &output, output_bytes, full_coverage)?;
            if zero_fill_enqueued {
                self.synchronize()?;
            }
            return Ok(CudaKernelOutput {
                buffer: output,
                execution: CudaExecutionStats::default(),
            });
        }

        let job_buffer = self.upload(store_gray16_job_as_bytes(&job))?;
        if zero_unwritten_store_output(self, &output, output_bytes, full_coverage)? {
            self.synchronize()?;
        }
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

    /// Apply inverse RCT/ICT in place on three pairwise-disjoint device f32 component planes.
    #[doc(hidden)]
    pub fn j2k_inverse_mct_device(
        &self,
        plane0: &CudaDeviceBuffer,
        plane1: &CudaDeviceBuffer,
        plane2: &CudaDeviceBuffer,
        job: CudaJ2kInverseMctJob,
    ) -> Result<CudaExecutionStats, CudaError> {
        validate_store_buffer_context(self, [plane0, plane1, plane2])?;
        validate_inverse_mct_planes_disjoint([plane0, plane1, plane2])?;
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
        validate_store_buffer_context(self, [plane0, plane1, plane2])?;
        let channels = if job.rgba == 0 { 3u8 } else { 4u8 };
        let output_bytes =
            checked_image_words(job.output_width, job.output_height, usize::from(channels))?;
        let pixels = checked_image_words(job.copy_width, job.copy_height, 1)?;
        let full_coverage = validate_store_destination(
            job.output_width,
            job.output_height,
            job.output_x,
            job.output_y,
            job.copy_width,
            job.copy_height,
            u32::from(channels),
        )?;
        if pixels != 0 {
            validate_store_plane(
                plane0,
                job.input_width0,
                job.source_x0,
                job.source_y0,
                job.copy_width,
                job.copy_height,
            )?;
            validate_store_plane(
                plane1,
                job.input_width1,
                job.source_x1,
                job.source_y1,
                job.copy_width,
                job.copy_height,
            )?;
            validate_store_plane(
                plane2,
                job.input_width2,
                job.source_x2,
                job.source_y2,
                job.copy_width,
                job.copy_height,
            )?;
        }
        let output = self.allocate(output_bytes)?;
        if output_bytes == 0 || pixels == 0 {
            let zero_fill_enqueued =
                zero_unwritten_store_output(self, &output, output_bytes, full_coverage)?;
            if zero_fill_enqueued {
                self.synchronize()?;
            }
            return Ok(CudaKernelOutput {
                buffer: output,
                execution: CudaExecutionStats::default(),
            });
        }

        let job_buffer = self.upload(store_rgb8_job_as_bytes(&job))?;
        if zero_unwritten_store_output(self, &output, output_bytes, full_coverage)? {
            self.synchronize()?;
        }
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
        validate_store_buffer_context(self, [plane0, plane1, plane2])?;
        let channels = if job.rgba == 0 { 3u8 } else { 4u8 };
        let output_samples =
            checked_image_words(job.output_width, job.output_height, usize::from(channels))?;
        let output_bytes = output_samples
            .checked_mul(std::mem::size_of::<u16>())
            .ok_or(CudaError::LengthTooLarge {
                len: output_samples,
            })?;
        let pixels = checked_image_words(job.copy_width, job.copy_height, 1)?;
        let full_coverage = validate_store_destination(
            job.output_width,
            job.output_height,
            job.output_x,
            job.output_y,
            job.copy_width,
            job.copy_height,
            u32::from(channels),
        )?;
        if pixels != 0 {
            validate_store_plane(
                plane0,
                job.input_width0,
                job.source_x0,
                job.source_y0,
                job.copy_width,
                job.copy_height,
            )?;
            validate_store_plane(
                plane1,
                job.input_width1,
                job.source_x1,
                job.source_y1,
                job.copy_width,
                job.copy_height,
            )?;
            validate_store_plane(
                plane2,
                job.input_width2,
                job.source_x2,
                job.source_y2,
                job.copy_width,
                job.copy_height,
            )?;
        }
        let output = self.allocate(output_bytes)?;
        if output_bytes == 0 || pixels == 0 {
            let zero_fill_enqueued =
                zero_unwritten_store_output(self, &output, output_bytes, full_coverage)?;
            if zero_fill_enqueued {
                self.synchronize()?;
            }
            return Ok(CudaKernelOutput {
                buffer: output,
                execution: CudaExecutionStats::default(),
            });
        }

        let job_buffer = self.upload(store_rgb16_job_as_bytes(&job))?;
        if zero_unwritten_store_output(self, &output, output_bytes, full_coverage)? {
            self.synchronize()?;
        }
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

    /// Apply inverse RCT/ICT and store tightly packed RGB16/RGBA16 in one dispatch.
    #[doc(hidden)]
    pub fn j2k_store_rgb16_mct_device(
        &self,
        plane0: &CudaDeviceBuffer,
        plane1: &CudaDeviceBuffer,
        plane2: &CudaDeviceBuffer,
        job: CudaJ2kStoreRgb16MctJob,
    ) -> Result<CudaKernelOutput, CudaError> {
        validate_store_buffer_context(self, [plane0, plane1, plane2])?;
        let store = job.store;
        let channels = if store.rgba == 0 { 3u8 } else { 4u8 };
        let output_samples = checked_image_words(
            store.output_width,
            store.output_height,
            usize::from(channels),
        )?;
        let output_bytes = output_samples
            .checked_mul(std::mem::size_of::<u16>())
            .ok_or(CudaError::LengthTooLarge {
                len: output_samples,
            })?;
        let pixels = checked_image_words(store.copy_width, store.copy_height, 1)?;
        let full_coverage = validate_store_destination(
            store.output_width,
            store.output_height,
            store.output_x,
            store.output_y,
            store.copy_width,
            store.copy_height,
            u32::from(channels),
        )?;
        if pixels != 0 {
            validate_store_plane(
                plane0,
                store.input_width0,
                store.source_x0,
                store.source_y0,
                store.copy_width,
                store.copy_height,
            )?;
            validate_store_plane(
                plane1,
                store.input_width1,
                store.source_x1,
                store.source_y1,
                store.copy_width,
                store.copy_height,
            )?;
            validate_store_plane(
                plane2,
                store.input_width2,
                store.source_x2,
                store.source_y2,
                store.copy_width,
                store.copy_height,
            )?;
        }
        let output = self.allocate(output_bytes)?;
        if output_bytes == 0 || pixels == 0 {
            let zero_fill_enqueued =
                zero_unwritten_store_output(self, &output, output_bytes, full_coverage)?;
            if zero_fill_enqueued {
                self.synchronize()?;
            }
            return Ok(CudaKernelOutput {
                buffer: output,
                execution: CudaExecutionStats::default(),
            });
        }

        let job_buffer = self.upload(store_rgb16_mct_job_as_bytes(&job))?;
        if zero_unwritten_store_output(self, &output, output_bytes, full_coverage)? {
            self.synchronize()?;
        }
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

#[cfg(test)]
mod tests;
