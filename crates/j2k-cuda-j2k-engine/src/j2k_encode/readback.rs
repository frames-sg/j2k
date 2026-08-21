// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    allocation::{try_vec_filled, HostPhaseBudget},
    bytes::{f32_slice_as_bytes_mut, i32_slice_as_bytes_mut},
    driver::CuDevicePtr,
    error::CudaError,
    execution::CudaExecutionStats,
    memory::CudaDeviceBuffer,
};

use super::{
    CudaDwt53LevelShape, CudaDwt53Output, CudaDwt97Output, CudaJ2kDeinterleavedComponents,
    CudaJ2kQuantizedSubband, CudaJ2kResidentComponents, CudaJ2kResidentQuantizedSubband,
    CudaResidentDwt53Output, CudaResidentDwt97Output,
};

impl CudaJ2kResidentComponents {
    /// Contiguous component-major f32 device buffer.
    pub fn buffer(&self) -> &CudaDeviceBuffer {
        &self.buffer
    }

    /// Number of pixels in each component plane.
    pub fn num_pixels(&self) -> usize {
        self.num_pixels
    }

    /// Number of resident component planes.
    pub fn num_components(&self) -> u8 {
        self.num_components
    }

    /// CUDA execution counters for the producing dispatch.
    pub fn execution(&self) -> CudaExecutionStats {
        self.execution
    }

    /// Download component planes into host memory for verification or host APIs.
    pub fn download_components(&self) -> Result<Vec<Vec<f32>>, CudaError> {
        if self.num_pixels == 0 {
            return try_vec_filled(usize::from(self.num_components), Vec::new());
        }
        let sample_count = self
            .num_pixels
            .checked_mul(usize::from(self.num_components))
            .ok_or(CudaError::LengthTooLarge {
                len: self.num_pixels,
            })?;
        let mut host_budget = HostPhaseBudget::new("CUDA component readback");
        let mut flattened = host_budget.try_vec_filled(sample_count, 0.0f32)?;
        self.buffer
            .copy_to_host(f32_slice_as_bytes_mut(&mut flattened))?;
        let mut components = host_budget.try_vec_with_capacity(usize::from(self.num_components))?;
        for component in flattened.chunks_exact(self.num_pixels) {
            components.push(host_budget.try_vec_from_slice(component)?);
        }
        Ok(components)
    }

    pub(super) fn component_plane_device_ptr(
        &self,
        component: u8,
    ) -> Result<CuDevicePtr, CudaError> {
        let range = self.component_plane_byte_range(component)?;
        let offset = u64::try_from(range.start)
            .map_err(|_| CudaError::LengthTooLarge { len: range.start })?;
        self.buffer
            .device_ptr()
            .checked_add(offset)
            .ok_or(CudaError::LengthTooLarge {
                len: self.buffer.byte_len(),
            })
    }

    pub(super) fn component_plane_byte_range(
        &self,
        component: u8,
    ) -> Result<std::ops::Range<usize>, CudaError> {
        if component >= self.num_components {
            return Err(CudaError::InvalidArgument {
                message: "component plane index is out of range".to_string(),
            });
        }
        let plane_bytes = self
            .num_pixels
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or(CudaError::LengthTooLarge {
                len: self.num_pixels,
            })?;
        let offset = plane_bytes
            .checked_mul(usize::from(component))
            .ok_or(CudaError::LengthTooLarge { len: plane_bytes })?;
        let end = offset
            .checked_add(plane_bytes)
            .ok_or(CudaError::LengthTooLarge { len: offset })?;
        if end > self.buffer.byte_len() {
            return Err(CudaError::OutputTooSmall {
                required: end,
                have: self.buffer.byte_len(),
            });
        }
        Ok(offset..end)
    }
}

impl CudaJ2kDeinterleavedComponents {
    /// Per-component f32 sample planes in component order.
    pub fn components(&self) -> &[Vec<f32>] {
        &self.components
    }

    /// CUDA execution counters for the deinterleave dispatch.
    pub fn execution(&self) -> CudaExecutionStats {
        self.execution
    }

    /// Consume the output and return owned component planes.
    pub fn into_components(self) -> Vec<Vec<f32>> {
        self.components
    }
}

impl CudaDwt53Output {
    /// Transformed coefficients downloaded to host memory.
    pub fn transformed(&self) -> &[f32] {
        &self.transformed
    }

    /// Per-level DWT shapes.
    pub fn levels(&self) -> &[CudaDwt53LevelShape] {
        &self.levels
    }

    /// Dimensions of the final low-low band.
    pub fn ll_dimensions(&self) -> (u32, u32) {
        (self.ll_width, self.ll_height)
    }

    /// CUDA execution counters for the transform.
    pub fn execution(&self) -> CudaExecutionStats {
        self.execution
    }
}

impl CudaResidentDwt53Output {
    /// Resident component-major transformed coefficient buffer.
    pub fn buffer(&self) -> &CudaDeviceBuffer {
        &self.buffer
    }

    /// Transformed coefficient count.
    pub fn sample_count(&self) -> usize {
        self.sample_count
    }

    /// Download transformed coefficients into host memory.
    pub fn download_transformed(&self) -> Result<Vec<f32>, CudaError> {
        let mut host_budget = HostPhaseBudget::new("CUDA forward 5/3 DWT readback");
        self.download_transformed_with_budget(&mut host_budget)
    }

    pub(super) fn download_transformed_with_budget(
        &self,
        host_budget: &mut HostPhaseBudget,
    ) -> Result<Vec<f32>, CudaError> {
        let mut transformed = host_budget.try_vec_filled(self.sample_count, 0f32)?;
        self.buffer
            .copy_to_host(f32_slice_as_bytes_mut(&mut transformed))?;
        Ok(transformed)
    }

    /// Per-level DWT shapes.
    pub fn levels(&self) -> &[CudaDwt53LevelShape] {
        &self.levels
    }

    /// Dimensions of the final low-low band.
    pub fn ll_dimensions(&self) -> (u32, u32) {
        (self.ll_width, self.ll_height)
    }

    /// CUDA execution counters for the transform.
    pub fn execution(&self) -> CudaExecutionStats {
        self.execution
    }
}

impl CudaDwt97Output {
    /// Transformed coefficients downloaded to host memory.
    pub fn transformed(&self) -> &[f32] {
        &self.transformed
    }

    /// Per-level DWT shapes.
    pub fn levels(&self) -> &[CudaDwt53LevelShape] {
        &self.levels
    }

    /// Dimensions of the final low-low band.
    pub fn ll_dimensions(&self) -> (u32, u32) {
        (self.ll_width, self.ll_height)
    }

    /// CUDA execution counters for the transform.
    pub fn execution(&self) -> CudaExecutionStats {
        self.execution
    }
}

impl CudaResidentDwt97Output {
    /// Resident component-major transformed coefficient buffer.
    pub fn buffer(&self) -> &CudaDeviceBuffer {
        &self.buffer
    }

    /// Transformed coefficient count.
    pub fn sample_count(&self) -> usize {
        self.sample_count
    }

    /// Download transformed coefficients into host memory.
    pub fn download_transformed(&self) -> Result<Vec<f32>, CudaError> {
        let mut host_budget = HostPhaseBudget::new("CUDA forward 9/7 DWT readback");
        self.download_transformed_with_budget(&mut host_budget)
    }

    pub(super) fn download_transformed_with_budget(
        &self,
        host_budget: &mut HostPhaseBudget,
    ) -> Result<Vec<f32>, CudaError> {
        let mut transformed = host_budget.try_vec_filled(self.sample_count, 0f32)?;
        self.buffer
            .copy_to_host(f32_slice_as_bytes_mut(&mut transformed))?;
        Ok(transformed)
    }

    /// Per-level DWT shapes.
    pub fn levels(&self) -> &[CudaDwt53LevelShape] {
        &self.levels
    }

    /// Dimensions of the final low-low band.
    pub fn ll_dimensions(&self) -> (u32, u32) {
        (self.ll_width, self.ll_height)
    }

    /// CUDA execution counters for the transform.
    pub fn execution(&self) -> CudaExecutionStats {
        self.execution
    }
}

impl CudaJ2kQuantizedSubband {
    /// Quantized sub-band coefficients downloaded to host memory.
    pub fn coefficients(&self) -> &[i32] {
        &self.coefficients
    }

    /// Consume the output and return its quantized coefficients.
    pub fn into_coefficients(self) -> Vec<i32> {
        self.coefficients
    }

    /// CUDA execution counters for the quantization stage.
    pub fn execution(&self) -> CudaExecutionStats {
        self.execution
    }
}

impl CudaJ2kResidentQuantizedSubband {
    /// Device buffer containing row-major `i32` coefficients.
    pub fn buffer(&self) -> &CudaDeviceBuffer {
        &self.coefficients
    }

    /// Number of `i32` coefficients in the resident buffer.
    pub fn coefficient_count(&self) -> usize {
        self.coefficient_count
    }

    /// Copy quantized coefficients to host memory.
    pub fn download_coefficients(&self) -> Result<Vec<i32>, CudaError> {
        let mut coefficients = try_vec_filled(self.coefficient_count, 0i32)?;
        self.coefficients
            .copy_to_host(i32_slice_as_bytes_mut(&mut coefficients))?;
        Ok(coefficients)
    }

    /// CUDA execution counters for the quantization stage.
    pub fn execution(&self) -> CudaExecutionStats {
        self.execution
    }
}
