// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    bytes::f32_slice_as_bytes,
    error::CudaError,
    execution::CudaExecutionStats,
    memory::{checked_image_words, CudaBufferPool, CudaDeviceBuffer},
};

use super::{
    validate_encode_buffer_context, validate_quantize_region,
    validation::validate_encode_context_matches, CudaJ2kQuantizeJob,
    CudaJ2kQuantizeSubbandRegionJob, CudaJ2kQuantizedSubband, CudaJ2kResidentQuantizedSubband,
};

impl crate::J2kCudaEngine<'_> {
    /// Quantize one JPEG 2000 sub-band on the device.
    #[doc(hidden)]
    pub fn j2k_quantize_subband(
        &self,
        samples: &[f32],
        job: CudaJ2kQuantizeJob,
    ) -> Result<CudaJ2kQuantizedSubband, CudaError> {
        let sample_buffer = self.upload(f32_slice_as_bytes(samples))?;
        let resident = self.j2k_quantize_subband_resident(&sample_buffer, samples.len(), job)?;
        let coefficients = resident.download_coefficients()?;
        Ok(CudaJ2kQuantizedSubband {
            coefficients,
            execution: resident.execution(),
        })
    }

    /// Quantize a resident contiguous JPEG 2000 sub-band into resident `i32` coefficients.
    #[doc(hidden)]
    pub fn j2k_quantize_subband_resident(
        &self,
        samples: &CudaDeviceBuffer,
        sample_count: usize,
        job: CudaJ2kQuantizeJob,
    ) -> Result<CudaJ2kResidentQuantizedSubband, CudaError> {
        validate_encode_buffer_context(self.context, [samples])?;
        if sample_count == 0 {
            return Ok(CudaJ2kResidentQuantizedSubband {
                coefficients: self.allocate(0)?,
                coefficient_count: 0,
                execution: CudaExecutionStats::default(),
            });
        }

        let available_samples = samples.typed_view::<f32>()?.len();
        if available_samples < sample_count {
            return Err(CudaError::OutputTooSmall {
                required: sample_count
                    .checked_mul(std::mem::size_of::<f32>())
                    .ok_or(CudaError::LengthTooLarge { len: sample_count })?,
                have: samples.byte_len(),
            });
        }

        self.prepare_operation()?;
        let coefficient_buffer = self.allocate(
            sample_count
                .checked_mul(std::mem::size_of::<i32>())
                .ok_or(CudaError::LengthTooLarge { len: sample_count })?,
        )?;
        self.launch_j2k_quantize_subband(samples, &coefficient_buffer, sample_count, job)?;

        Ok(CudaJ2kResidentQuantizedSubband {
            coefficients: coefficient_buffer,
            coefficient_count: sample_count,
            execution: CudaExecutionStats::new(1, 0, 0, false),
        })
    }

    /// Quantize resident strided DWT sub-band rectangles into resident `i32`
    /// coefficients, stored back to back in job order in one buffer taken from
    /// `pool`; callers may recycle it into `pool` once work reading it has
    /// completed.
    #[doc(hidden)]
    pub fn j2k_quantize_subband_regions_resident(
        &self,
        samples: &CudaDeviceBuffer,
        jobs: &[CudaJ2kQuantizeSubbandRegionJob],
        pool: &CudaBufferPool,
    ) -> Result<CudaJ2kResidentQuantizedSubband, CudaError> {
        validate_encode_context_matches([
            samples.is_owned_by(self.context),
            pool.is_owned_by(self.context),
        ])?;
        let available_samples = samples.typed_view::<f32>()?.len();
        let mut coefficient_count = 0usize;
        for job in jobs {
            validate_quantize_region(*job, available_samples)?;
            coefficient_count = coefficient_count
                .checked_add(checked_image_words(job.width, job.height, 1)?)
                .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
        }
        if coefficient_count == 0 {
            return Ok(CudaJ2kResidentQuantizedSubband {
                coefficients: self.allocate(0)?,
                coefficient_count: 0,
                execution: CudaExecutionStats::default(),
            });
        }

        self.prepare_operation()?;
        let coefficient_buffer = pool
            .take(
                coefficient_count
                    .checked_mul(std::mem::size_of::<i32>())
                    .ok_or(CudaError::LengthTooLarge {
                        len: coefficient_count,
                    })?,
            )?
            .into_device_buffer()?;
        let mut offset = 0usize;
        let mut dispatches = 0usize;
        for job in jobs {
            let count = checked_image_words(job.width, job.height, 1)?;
            if count != 0 {
                self.launch_j2k_quantize_subband_region(
                    samples,
                    &coefficient_buffer,
                    offset,
                    *job,
                )?;
                offset += count;
                dispatches += 1;
            }
        }

        Ok(CudaJ2kResidentQuantizedSubband {
            coefficients: coefficient_buffer,
            coefficient_count,
            execution: CudaExecutionStats::new(dispatches, 0, 0, false),
        })
    }
}
