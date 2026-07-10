// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    bytes::f32_slice_as_bytes, context::CudaContext, error::CudaError,
    execution::CudaExecutionStats, j2k_decode::active_dwt53_buffers, kernels::CudaKernel,
    memory::CudaDeviceBuffer,
};

use super::{
    CudaDwt53LevelPass, CudaDwt53LevelShape, CudaDwt53Output, CudaDwt53Pass, CudaDwt97Output,
    CudaJ2kResidentComponents, CudaResidentDwt53Output, CudaResidentDwt97Output,
};

impl CudaContext {
    /// Run the reversible 5/3 forward DWT stage on one component plane.
    #[doc(hidden)]
    pub fn j2k_forward_dwt53(
        &self,
        samples: &[f32],
        width: u32,
        height: u32,
        num_levels: u8,
    ) -> Result<CudaDwt53Output, CudaError> {
        let expected_len =
            (width as usize)
                .checked_mul(height as usize)
                .ok_or(CudaError::ImageTooLarge {
                    width,
                    height,
                    channels: 1,
                })?;
        if expected_len != samples.len() {
            return Err(CudaError::ImageTooLarge {
                width,
                height,
                channels: 1,
            });
        }
        if samples.is_empty() || num_levels == 0 {
            return Ok(CudaDwt53Output {
                transformed: samples.to_vec(),
                levels: Vec::new(),
                ll_width: width,
                ll_height: height,
                execution: CudaExecutionStats::default(),
            });
        }

        self.inner.set_current()?;
        let buffer_a = self.upload(f32_slice_as_bytes(samples))?;
        let resident = self.j2k_forward_dwt53_resident_buffer(
            buffer_a,
            samples.len(),
            width,
            height,
            num_levels,
            0,
        )?;
        let transformed = resident.download_transformed()?;
        Ok(CudaDwt53Output {
            transformed,
            levels: resident.levels().to_vec(),
            ll_width: resident.ll_dimensions().0,
            ll_height: resident.ll_dimensions().1,
            execution: resident.execution(),
        })
    }

    /// Run the reversible 5/3 forward DWT on one resident component plane.
    #[doc(hidden)]
    pub fn j2k_forward_dwt53_resident_component(
        &self,
        components: &CudaJ2kResidentComponents,
        component: u8,
        width: u32,
        height: u32,
        num_levels: u8,
    ) -> Result<CudaResidentDwt53Output, CudaError> {
        let expected_len =
            (width as usize)
                .checked_mul(height as usize)
                .ok_or(CudaError::ImageTooLarge {
                    width,
                    height,
                    channels: 1,
                })?;
        if expected_len != components.num_pixels {
            return Err(CudaError::ImageTooLarge {
                width,
                height,
                channels: 1,
            });
        }

        self.inner.set_current()?;
        let plane_ptr = components.component_plane_device_ptr(component)?;
        let byte_len = expected_len
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or(CudaError::LengthTooLarge { len: expected_len })?;
        let buffer_a = self.copy_device_ptr_to_device_with_kernel(plane_ptr, byte_len)?;
        let copy_dispatches = usize::from(byte_len != 0);
        self.j2k_forward_dwt53_resident_buffer(
            buffer_a,
            expected_len,
            width,
            height,
            num_levels,
            copy_dispatches,
        )
    }

    fn j2k_forward_dwt53_resident_buffer(
        &self,
        buffer_a: CudaDeviceBuffer,
        sample_count: usize,
        width: u32,
        height: u32,
        num_levels: u8,
        initial_copy_dispatches: usize,
    ) -> Result<CudaResidentDwt53Output, CudaError> {
        if sample_count == 0 || num_levels == 0 {
            return Ok(CudaResidentDwt53Output {
                buffer: buffer_a,
                sample_count,
                levels: Vec::new(),
                ll_width: width,
                ll_height: height,
                execution: CudaExecutionStats {
                    kernel_dispatches: initial_copy_dispatches,
                    copy_kernel_dispatches: initial_copy_dispatches,
                    decode_kernel_dispatches: 0,
                    hardware_decode: false,
                },
            });
        }

        let buffer_b = self.allocate(
            sample_count
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or(CudaError::LengthTooLarge { len: sample_count })?,
        )?;
        let mut current_width = width;
        let mut current_height = height;
        let mut levels = Vec::new();
        let mut dispatches = 0usize;
        let mut active_is_a = true;

        for _ in 0..num_levels {
            if current_width < 2 && current_height < 2 {
                break;
            }
            let (level_dispatches, level_shape) = self.launch_j2k_forward_dwt53_level(
                &buffer_a,
                &buffer_b,
                &mut active_is_a,
                CudaDwt53LevelPass {
                    full_width: width,
                    current_width,
                    current_height,
                },
            )?;
            dispatches = dispatches.saturating_add(level_dispatches);
            levels.push(level_shape);
            current_width = level_shape.low_width;
            current_height = level_shape.low_height;
        }

        let buffer = if active_is_a { buffer_a } else { buffer_b };
        Ok(CudaResidentDwt53Output {
            buffer,
            sample_count,
            levels,
            ll_width: current_width,
            ll_height: current_height,
            execution: CudaExecutionStats {
                kernel_dispatches: initial_copy_dispatches.saturating_add(dispatches),
                copy_kernel_dispatches: initial_copy_dispatches,
                decode_kernel_dispatches: 0,
                hardware_decode: false,
            },
        })
    }

    /// Run the irreversible 9/7 forward DWT stage on one component plane.
    #[doc(hidden)]
    pub fn j2k_forward_dwt97(
        &self,
        samples: &[f32],
        width: u32,
        height: u32,
        num_levels: u8,
    ) -> Result<CudaDwt97Output, CudaError> {
        let expected_len =
            (width as usize)
                .checked_mul(height as usize)
                .ok_or(CudaError::ImageTooLarge {
                    width,
                    height,
                    channels: 1,
                })?;
        if expected_len != samples.len() {
            return Err(CudaError::ImageTooLarge {
                width,
                height,
                channels: 1,
            });
        }
        if samples.is_empty() || num_levels == 0 {
            return Ok(CudaDwt97Output {
                transformed: samples.to_vec(),
                levels: Vec::new(),
                ll_width: width,
                ll_height: height,
                execution: CudaExecutionStats::default(),
            });
        }

        self.inner.set_current()?;
        let buffer_a = self.upload(f32_slice_as_bytes(samples))?;
        let resident = self.j2k_forward_dwt97_resident_buffer(
            buffer_a,
            samples.len(),
            width,
            height,
            num_levels,
            0,
        )?;
        let transformed = resident.download_transformed()?;
        Ok(CudaDwt97Output {
            transformed,
            levels: resident.levels().to_vec(),
            ll_width: resident.ll_dimensions().0,
            ll_height: resident.ll_dimensions().1,
            execution: resident.execution(),
        })
    }

    /// Run the irreversible 9/7 forward DWT on one resident component plane.
    #[doc(hidden)]
    pub fn j2k_forward_dwt97_resident_component(
        &self,
        components: &CudaJ2kResidentComponents,
        component: u8,
        width: u32,
        height: u32,
        num_levels: u8,
    ) -> Result<CudaResidentDwt97Output, CudaError> {
        let expected_len =
            (width as usize)
                .checked_mul(height as usize)
                .ok_or(CudaError::ImageTooLarge {
                    width,
                    height,
                    channels: 1,
                })?;
        if expected_len != components.num_pixels {
            return Err(CudaError::ImageTooLarge {
                width,
                height,
                channels: 1,
            });
        }

        self.inner.set_current()?;
        let plane_ptr = components.component_plane_device_ptr(component)?;
        let byte_len = expected_len
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or(CudaError::LengthTooLarge { len: expected_len })?;
        let buffer_a = self.copy_device_ptr_to_device_with_kernel(plane_ptr, byte_len)?;
        let copy_dispatches = usize::from(byte_len != 0);
        self.j2k_forward_dwt97_resident_buffer(
            buffer_a,
            expected_len,
            width,
            height,
            num_levels,
            copy_dispatches,
        )
    }

    fn j2k_forward_dwt97_resident_buffer(
        &self,
        buffer_a: CudaDeviceBuffer,
        sample_count: usize,
        width: u32,
        height: u32,
        num_levels: u8,
        initial_copy_dispatches: usize,
    ) -> Result<CudaResidentDwt97Output, CudaError> {
        if sample_count == 0 || num_levels == 0 {
            return Ok(CudaResidentDwt97Output {
                buffer: buffer_a,
                sample_count,
                levels: Vec::new(),
                ll_width: width,
                ll_height: height,
                execution: CudaExecutionStats {
                    kernel_dispatches: initial_copy_dispatches,
                    copy_kernel_dispatches: initial_copy_dispatches,
                    decode_kernel_dispatches: 0,
                    hardware_decode: false,
                },
            });
        }

        let buffer_b = self.allocate(
            sample_count
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or(CudaError::LengthTooLarge { len: sample_count })?,
        )?;
        let mut current_width = width;
        let mut current_height = height;
        let mut levels = Vec::new();
        let mut dispatches = 0usize;
        let mut active_is_a = true;

        for _ in 0..num_levels {
            if current_width < 2 && current_height < 2 {
                break;
            }
            let (level_dispatches, level_shape) = self.launch_j2k_forward_dwt97_level(
                &buffer_a,
                &buffer_b,
                &mut active_is_a,
                CudaDwt53LevelPass {
                    full_width: width,
                    current_width,
                    current_height,
                },
            )?;
            dispatches = dispatches.saturating_add(level_dispatches);
            levels.push(level_shape);
            current_width = level_shape.low_width;
            current_height = level_shape.low_height;
        }

        let buffer = if active_is_a { buffer_a } else { buffer_b };
        Ok(CudaResidentDwt97Output {
            buffer,
            sample_count,
            levels,
            ll_width: current_width,
            ll_height: current_height,
            execution: CudaExecutionStats {
                kernel_dispatches: initial_copy_dispatches.saturating_add(dispatches),
                copy_kernel_dispatches: initial_copy_dispatches,
                decode_kernel_dispatches: 0,
                hardware_decode: false,
            },
        })
    }

    fn launch_j2k_forward_dwt53_level(
        &self,
        buffer_a: &CudaDeviceBuffer,
        buffer_b: &CudaDeviceBuffer,
        active_is_a: &mut bool,
        pass: CudaDwt53LevelPass,
    ) -> Result<(usize, CudaDwt53LevelShape), CudaError> {
        let low_width = pass.current_width.div_ceil(2);
        let low_height = pass.current_height.div_ceil(2);
        let mut dispatches = 0usize;

        if pass.current_height >= 2 {
            let (input, output) = active_dwt53_buffers(buffer_a, buffer_b, *active_is_a);
            self.launch_j2k_forward_dwt53_pass(
                CudaKernel::J2kForwardDwt53Vertical,
                input,
                output,
                CudaDwt53Pass {
                    full_width: pass.full_width,
                    current_width: pass.current_width,
                    current_height: pass.current_height,
                    low_extent: low_height,
                },
            )?;
            *active_is_a = !*active_is_a;
            dispatches = dispatches.saturating_add(1);
        }

        if pass.current_width >= 2 {
            let (input, output) = active_dwt53_buffers(buffer_a, buffer_b, *active_is_a);
            self.launch_j2k_forward_dwt53_pass(
                CudaKernel::J2kForwardDwt53Horizontal,
                input,
                output,
                CudaDwt53Pass {
                    full_width: pass.full_width,
                    current_width: pass.current_width,
                    current_height: pass.current_height,
                    low_extent: low_width,
                },
            )?;
            *active_is_a = !*active_is_a;
            dispatches = dispatches.saturating_add(1);
        }

        Ok((
            dispatches,
            CudaDwt53LevelShape {
                width: pass.current_width,
                height: pass.current_height,
                low_width,
                low_height,
                high_width: pass.current_width / 2,
                high_height: pass.current_height / 2,
            },
        ))
    }

    fn launch_j2k_forward_dwt97_level(
        &self,
        buffer_a: &CudaDeviceBuffer,
        buffer_b: &CudaDeviceBuffer,
        active_is_a: &mut bool,
        pass: CudaDwt53LevelPass,
    ) -> Result<(usize, CudaDwt53LevelShape), CudaError> {
        let low_width = pass.current_width.div_ceil(2);
        let low_height = pass.current_height.div_ceil(2);
        let mut dispatches = 0usize;

        if pass.current_height >= 2 {
            let (input, output) = active_dwt53_buffers(buffer_a, buffer_b, *active_is_a);
            self.launch_j2k_forward_dwt53_pass(
                CudaKernel::J2kForwardDwt97Vertical,
                input,
                output,
                CudaDwt53Pass {
                    full_width: pass.full_width,
                    current_width: pass.current_width,
                    current_height: pass.current_height,
                    low_extent: low_height,
                },
            )?;
            *active_is_a = !*active_is_a;
            dispatches = dispatches.saturating_add(1);
        }

        if pass.current_width >= 2 {
            let (input, output) = active_dwt53_buffers(buffer_a, buffer_b, *active_is_a);
            self.launch_j2k_forward_dwt53_pass(
                CudaKernel::J2kForwardDwt97Horizontal,
                input,
                output,
                CudaDwt53Pass {
                    full_width: pass.full_width,
                    current_width: pass.current_width,
                    current_height: pass.current_height,
                    low_extent: low_width,
                },
            )?;
            *active_is_a = !*active_is_a;
            dispatches = dispatches.saturating_add(1);
        }

        Ok((
            dispatches,
            CudaDwt53LevelShape {
                width: pass.current_width,
                height: pass.current_height,
                low_width,
                low_height,
                high_width: pass.current_width / 2,
                high_height: pass.current_height / 2,
            },
        ))
    }
}
