// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    bytes::f32_slice_as_bytes,
    context::CudaContext,
    error::CudaError,
    execution::CudaExecutionStats,
    memory::{checked_image_words, CudaDeviceBuffer},
};

use super::{
    validate_encode_buffer_context, validate_quantize_region, CudaJ2kQuantizeJob,
    CudaJ2kQuantizeSubbandRegionJob, CudaJ2kQuantizedSubband, CudaJ2kResidentQuantizedSubband,
};

impl CudaContext {
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
        validate_encode_buffer_context(self, [samples])?;
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

        self.inner.set_current()?;
        let coefficient_buffer = self.allocate(
            sample_count
                .checked_mul(std::mem::size_of::<i32>())
                .ok_or(CudaError::LengthTooLarge { len: sample_count })?,
        )?;
        self.launch_j2k_quantize_subband(samples, &coefficient_buffer, sample_count, job)?;

        Ok(CudaJ2kResidentQuantizedSubband {
            coefficients: coefficient_buffer,
            coefficient_count: sample_count,
            execution: CudaExecutionStats {
                kernel_dispatches: 1,
                copy_kernel_dispatches: 0,
                decode_kernel_dispatches: 0,
                hardware_decode: false,
            },
        })
    }

    /// Quantize a resident strided DWT sub-band rectangle into resident `i32` coefficients.
    #[doc(hidden)]
    pub fn j2k_quantize_subband_region_resident(
        &self,
        samples: &CudaDeviceBuffer,
        job: CudaJ2kQuantizeSubbandRegionJob,
    ) -> Result<CudaJ2kResidentQuantizedSubband, CudaError> {
        validate_encode_buffer_context(self, [samples])?;
        let coefficient_count = checked_image_words(job.width, job.height, 1)?;
        if coefficient_count == 0 {
            return Ok(CudaJ2kResidentQuantizedSubband {
                coefficients: self.allocate(0)?,
                coefficient_count: 0,
                execution: CudaExecutionStats::default(),
            });
        }

        let available_samples = samples.typed_view::<f32>()?.len();
        validate_quantize_region(job, available_samples)?;
        self.inner.set_current()?;
        let coefficient_buffer = self.allocate(
            coefficient_count
                .checked_mul(std::mem::size_of::<i32>())
                .ok_or(CudaError::LengthTooLarge {
                    len: coefficient_count,
                })?,
        )?;
        self.launch_j2k_quantize_subband_region(samples, &coefficient_buffer, job)?;

        Ok(CudaJ2kResidentQuantizedSubband {
            coefficients: coefficient_buffer,
            coefficient_count,
            execution: CudaExecutionStats {
                kernel_dispatches: 1,
                copy_kernel_dispatches: 0,
                decode_kernel_dispatches: 0,
                hardware_decode: false,
            },
        })
    }
}
