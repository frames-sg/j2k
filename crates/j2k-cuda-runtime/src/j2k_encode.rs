use crate::{
    bytes::{f32_slice_as_bytes, f32_slice_as_bytes_mut, i32_slice_as_bytes_mut},
    context::CudaContext,
    driver::{CuDevicePtr, CuFunction},
    error::CudaError,
    execution::{cuda_kernel_param, CudaExecutionStats},
    j2k_decode::{active_dwt53_buffers, CudaJ2kStridedInterleavedPixels},
    kernels::{j2k_dwt53_launch_geometry, j2k_forward_rct_launch_geometry, CudaKernel},
    memory::{checked_image_words, CudaDeviceBuffer},
};

#[derive(Clone, Copy)]
struct J2kStridedDeinterleaveLaunch<'a> {
    pixels: &'a CudaDeviceBuffer,
    output: &'a CudaDeviceBuffer,
    width: u32,
    height: u32,
    byte_offset: usize,
    pitch_bytes: usize,
    num_components: u8,
    bit_depth: u8,
    signed: bool,
}

impl CudaContext {
    /// Deinterleave interleaved pixel bytes into f32 component planes.
    #[doc(hidden)]
    pub fn j2k_deinterleave_to_f32(
        &self,
        pixels: &[u8],
        num_pixels: usize,
        num_components: u8,
        bit_depth: u8,
        signed: bool,
    ) -> Result<CudaJ2kDeinterleavedComponents, CudaError> {
        let resident = self.j2k_deinterleave_to_f32_resident(
            pixels,
            num_pixels,
            num_components,
            bit_depth,
            signed,
        )?;
        let execution = resident.execution();
        let components = resident.download_components()?;
        Ok(CudaJ2kDeinterleavedComponents {
            components,
            execution,
        })
    }

    /// Deinterleave interleaved pixel bytes into resident f32 component planes.
    #[doc(hidden)]
    pub fn j2k_deinterleave_to_f32_resident(
        &self,
        pixels: &[u8],
        num_pixels: usize,
        num_components: u8,
        bit_depth: u8,
        signed: bool,
    ) -> Result<CudaJ2kResidentComponents, CudaError> {
        if num_components == 0 || num_components > 4 {
            return Err(CudaError::InvalidArgument {
                message: "component count must be between 1 and 4".to_string(),
            });
        }
        if bit_depth == 0 || bit_depth > 16 {
            return Err(CudaError::InvalidArgument {
                message: "bit depth must be between 1 and 16".to_string(),
            });
        }
        let bytes_per_sample = if bit_depth <= 8 { 1usize } else { 2usize };
        let expected_len = num_pixels
            .checked_mul(usize::from(num_components))
            .and_then(|len| len.checked_mul(bytes_per_sample))
            .ok_or(CudaError::LengthTooLarge { len: num_pixels })?;
        if pixels.len() < expected_len {
            return Err(CudaError::InvalidArgument {
                message: "pixel buffer is shorter than the requested image".to_string(),
            });
        }

        self.inner.set_current()?;
        let sample_count = num_pixels
            .checked_mul(usize::from(num_components))
            .ok_or(CudaError::LengthTooLarge { len: num_pixels })?;
        let output_bytes = sample_count
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or(CudaError::LengthTooLarge { len: sample_count })?;
        let output = self.allocate(output_bytes)?;
        if num_pixels == 0 {
            return Ok(CudaJ2kResidentComponents {
                buffer: output,
                num_pixels,
                num_components,
                execution: CudaExecutionStats::default(),
            });
        }

        let pixels = self.upload(&pixels[..expected_len])?;
        self.launch_j2k_deinterleave_to_f32(
            &pixels,
            &output,
            num_pixels,
            num_components,
            bit_depth,
            signed,
        )?;

        Ok(CudaJ2kResidentComponents {
            buffer: output,
            num_pixels,
            num_components,
            execution: CudaExecutionStats {
                kernel_dispatches: 1,
                copy_kernel_dispatches: 0,
                decode_kernel_dispatches: 0,
                hardware_decode: false,
            },
        })
    }

    /// Deinterleave strided device-resident pixel bytes into resident f32 component planes.
    #[doc(hidden)]
    pub fn j2k_deinterleave_strided_to_f32_resident(
        &self,
        image: CudaJ2kStridedInterleavedPixels<'_>,
    ) -> Result<CudaJ2kResidentComponents, CudaError> {
        let CudaJ2kStridedInterleavedPixels {
            buffer: pixels,
            byte_offset,
            width,
            height,
            pitch_bytes,
            num_components,
            bit_depth,
            signed,
        } = image;
        if width == 0 || height == 0 {
            return Err(CudaError::InvalidArgument {
                message: "image dimensions must be nonzero".to_string(),
            });
        }
        if num_components == 0 || num_components > 4 {
            return Err(CudaError::InvalidArgument {
                message: "component count must be between 1 and 4".to_string(),
            });
        }
        if bit_depth == 0 || bit_depth > 16 {
            return Err(CudaError::InvalidArgument {
                message: "bit depth must be between 1 and 16".to_string(),
            });
        }
        let bytes_per_sample = if bit_depth <= 8 { 1usize } else { 2usize };
        let bytes_per_pixel = usize::from(num_components)
            .checked_mul(bytes_per_sample)
            .ok_or(CudaError::LengthTooLarge {
                len: usize::from(num_components),
            })?;
        let row_bytes =
            (width as usize)
                .checked_mul(bytes_per_pixel)
                .ok_or(CudaError::ImageTooLarge {
                    width,
                    height,
                    channels: usize::from(num_components),
                })?;
        if pitch_bytes < row_bytes {
            return Err(CudaError::InvalidArgument {
                message: "pitch is shorter than one row".to_string(),
            });
        }
        let required_end = byte_offset
            .checked_add(
                pitch_bytes
                    .checked_mul(height.saturating_sub(1) as usize)
                    .and_then(|prefix| prefix.checked_add(row_bytes))
                    .ok_or(CudaError::LengthTooLarge { len: pitch_bytes })?,
            )
            .ok_or(CudaError::LengthTooLarge { len: byte_offset })?;
        if required_end > pixels.byte_len() {
            return Err(CudaError::OutputTooSmall {
                required: required_end,
                have: pixels.byte_len(),
            });
        }

        self.inner.set_current()?;
        let num_pixels =
            (width as usize)
                .checked_mul(height as usize)
                .ok_or(CudaError::ImageTooLarge {
                    width,
                    height,
                    channels: usize::from(num_components),
                })?;
        let sample_count = num_pixels
            .checked_mul(usize::from(num_components))
            .ok_or(CudaError::LengthTooLarge { len: num_pixels })?;
        let output_bytes = sample_count
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or(CudaError::LengthTooLarge { len: sample_count })?;
        let output = self.allocate(output_bytes)?;
        self.launch_j2k_deinterleave_strided_to_f32(J2kStridedDeinterleaveLaunch {
            pixels,
            output: &output,
            width,
            height,
            byte_offset,
            pitch_bytes,
            num_components,
            bit_depth,
            signed,
        })?;

        Ok(CudaJ2kResidentComponents {
            buffer: output,
            num_pixels,
            num_components,
            execution: CudaExecutionStats {
                kernel_dispatches: 1,
                copy_kernel_dispatches: 0,
                decode_kernel_dispatches: 0,
                hardware_decode: false,
            },
        })
    }

    /// Run the reversible color transform in place on resident component planes.
    ///
    /// The transform is applied to the first three planes (R, G, B → Y, Cb, Cr).
    /// Any additional plane (e.g. a 4th alpha/auxiliary component) is left
    /// untouched, matching the native reference which applies RCT to the first
    /// three of `&mut [Vec<f32>]` and passes the remainder through unchanged.
    #[doc(hidden)]
    pub fn j2k_forward_rct_resident(
        &self,
        components: &mut CudaJ2kResidentComponents,
    ) -> Result<CudaExecutionStats, CudaError> {
        if components.num_components < 3 {
            return Err(CudaError::InvalidArgument {
                message: "forward RCT requires at least three resident component planes"
                    .to_string(),
            });
        }
        if components.num_pixels == 0 {
            return Ok(CudaExecutionStats::default());
        }

        self.inner.set_current()?;
        let plane0 = components.component_plane_device_ptr(0)?;
        let plane1 = components.component_plane_device_ptr(1)?;
        let plane2 = components.component_plane_device_ptr(2)?;
        self.launch_j2k_forward_rct_ptrs(plane0, plane1, plane2, components.num_pixels)?;

        Ok(CudaExecutionStats {
            kernel_dispatches: 1,
            copy_kernel_dispatches: 0,
            decode_kernel_dispatches: 0,
            hardware_decode: false,
        })
    }

    /// Run the irreversible color transform in place on resident component planes.
    ///
    /// The transform is applied to the first three planes (R, G, B → Y, Cb, Cr).
    /// Any additional plane is left untouched, matching the native reference
    /// which applies ICT to the first three of `&mut [Vec<f32>]` and passes the
    /// remainder through unchanged.
    #[doc(hidden)]
    pub fn j2k_forward_ict_resident(
        &self,
        components: &mut CudaJ2kResidentComponents,
    ) -> Result<CudaExecutionStats, CudaError> {
        if components.num_components < 3 {
            return Err(CudaError::InvalidArgument {
                message: "forward ICT requires at least three resident component planes"
                    .to_string(),
            });
        }
        if components.num_pixels == 0 {
            return Ok(CudaExecutionStats::default());
        }

        self.inner.set_current()?;
        let plane0 = components.component_plane_device_ptr(0)?;
        let plane1 = components.component_plane_device_ptr(1)?;
        let plane2 = components.component_plane_device_ptr(2)?;
        self.launch_j2k_forward_ict_ptrs(plane0, plane1, plane2, components.num_pixels)?;

        Ok(CudaExecutionStats {
            kernel_dispatches: 1,
            copy_kernel_dispatches: 0,
            decode_kernel_dispatches: 0,
            hardware_decode: false,
        })
    }

    /// Run the reversible color transform stage on three component planes.
    #[doc(hidden)]
    pub fn j2k_forward_rct(
        &self,
        plane0: &mut [f32],
        plane1: &mut [f32],
        plane2: &mut [f32],
    ) -> Result<CudaExecutionStats, CudaError> {
        if plane0.len() != plane1.len() || plane0.len() != plane2.len() {
            return Err(CudaError::ImageTooLarge {
                width: u32::try_from(plane0.len()).unwrap_or(u32::MAX),
                height: 1,
                channels: 3,
            });
        }
        if plane0.is_empty() {
            return Ok(CudaExecutionStats::default());
        }

        self.inner.set_current()?;
        let buffer0 = self.upload(f32_slice_as_bytes(plane0))?;
        let buffer1 = self.upload(f32_slice_as_bytes(plane1))?;
        let buffer2 = self.upload(f32_slice_as_bytes(plane2))?;
        self.launch_j2k_forward_rct_buffers(&buffer0, &buffer1, &buffer2, plane0.len())?;
        buffer0.copy_to_host(f32_slice_as_bytes_mut(plane0))?;
        buffer1.copy_to_host(f32_slice_as_bytes_mut(plane1))?;
        buffer2.copy_to_host(f32_slice_as_bytes_mut(plane2))?;

        Ok(CudaExecutionStats {
            kernel_dispatches: 1,
            copy_kernel_dispatches: 0,
            decode_kernel_dispatches: 0,
            hardware_decode: false,
        })
    }

    /// Run the irreversible color transform stage on three component planes.
    #[doc(hidden)]
    pub fn j2k_forward_ict(
        &self,
        plane0: &mut [f32],
        plane1: &mut [f32],
        plane2: &mut [f32],
    ) -> Result<CudaExecutionStats, CudaError> {
        if plane0.len() != plane1.len() || plane0.len() != plane2.len() {
            return Err(CudaError::ImageTooLarge {
                width: u32::try_from(plane0.len()).unwrap_or(u32::MAX),
                height: 1,
                channels: 3,
            });
        }
        if plane0.is_empty() {
            return Ok(CudaExecutionStats::default());
        }

        self.inner.set_current()?;
        let buffer0 = self.upload(f32_slice_as_bytes(plane0))?;
        let buffer1 = self.upload(f32_slice_as_bytes(plane1))?;
        let buffer2 = self.upload(f32_slice_as_bytes(plane2))?;
        self.launch_j2k_forward_ict_buffers(&buffer0, &buffer1, &buffer2, plane0.len())?;
        buffer0.copy_to_host(f32_slice_as_bytes_mut(plane0))?;
        buffer1.copy_to_host(f32_slice_as_bytes_mut(plane1))?;
        buffer2.copy_to_host(f32_slice_as_bytes_mut(plane2))?;

        Ok(CudaExecutionStats {
            kernel_dispatches: 1,
            copy_kernel_dispatches: 0,
            decode_kernel_dispatches: 0,
            hardware_decode: false,
        })
    }

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

    fn launch_j2k_forward_rct_buffers(
        &self,
        plane0: &CudaDeviceBuffer,
        plane1: &CudaDeviceBuffer,
        plane2: &CudaDeviceBuffer,
        len: usize,
    ) -> Result<(), CudaError> {
        self.launch_j2k_forward_rct_ptrs(
            plane0.device_ptr(),
            plane1.device_ptr(),
            plane2.device_ptr(),
            len,
        )
    }

    fn j2k_encode_kernel_function(&self, kernel: CudaKernel) -> Result<CuFunction, CudaError> {
        self.inner.cuda_oxide_j2k_encode_kernel_function(kernel)
    }

    fn launch_j2k_forward_rct_ptrs(
        &self,
        plane0: CuDevicePtr,
        plane1: CuDevicePtr,
        plane2: CuDevicePtr,
        len: usize,
    ) -> Result<(), CudaError> {
        let function = self.j2k_encode_kernel_function(CudaKernel::J2kForwardRct)?;
        let mut plane0_ptr = plane0;
        let mut plane1_ptr = plane1;
        let mut plane2_ptr = plane2;
        let mut len_u64 = u64::try_from(len).map_err(|_| CudaError::LengthTooLarge { len })?;
        let mut params = cuda_kernel_params!(plane0_ptr, plane1_ptr, plane2_ptr, len_u64);
        let geometry =
            j2k_forward_rct_launch_geometry(len).ok_or(CudaError::LengthTooLarge { len })?;

        self.launch_kernel(function, geometry, &mut params)
    }

    fn launch_j2k_deinterleave_to_f32(
        &self,
        pixels: &CudaDeviceBuffer,
        output: &CudaDeviceBuffer,
        num_pixels: usize,
        num_components: u8,
        bit_depth: u8,
        signed: bool,
    ) -> Result<(), CudaError> {
        let function = self.j2k_encode_kernel_function(CudaKernel::J2kDeinterleaveToF32)?;
        let mut pixels_ptr = pixels.device_ptr();
        let mut output_ptr = output.device_ptr();
        let mut num_pixels_u64 =
            u64::try_from(num_pixels).map_err(|_| CudaError::LengthTooLarge { len: num_pixels })?;
        let mut num_components_u32 = u32::from(num_components);
        let mut bit_depth_u32 = u32::from(bit_depth);
        let mut signed_u32 = u32::from(signed);
        let mut params = cuda_kernel_params!(
            pixels_ptr,
            output_ptr,
            num_pixels_u64,
            num_components_u32,
            bit_depth_u32,
            signed_u32
        );
        let geometry = j2k_forward_rct_launch_geometry(num_pixels)
            .ok_or(CudaError::LengthTooLarge { len: num_pixels })?;

        self.launch_kernel(function, geometry, &mut params)
    }

    fn launch_j2k_deinterleave_strided_to_f32(
        &self,
        request: J2kStridedDeinterleaveLaunch<'_>,
    ) -> Result<(), CudaError> {
        let function = self.j2k_encode_kernel_function(CudaKernel::J2kDeinterleaveStridedToF32)?;
        let mut pixels_ptr = request.pixels.device_ptr();
        let mut output_ptr = request.output.device_ptr();
        let mut width_u64 = u64::from(request.width);
        let mut height_u64 = u64::from(request.height);
        let mut byte_offset_u64 =
            u64::try_from(request.byte_offset).map_err(|_| CudaError::LengthTooLarge {
                len: request.byte_offset,
            })?;
        let mut pitch_bytes_u64 =
            u64::try_from(request.pitch_bytes).map_err(|_| CudaError::LengthTooLarge {
                len: request.pitch_bytes,
            })?;
        let mut num_components_u32 = u32::from(request.num_components);
        let mut bit_depth_u32 = u32::from(request.bit_depth);
        let mut signed_u32 = u32::from(request.signed);
        let mut params = cuda_kernel_params!(
            pixels_ptr,
            output_ptr,
            width_u64,
            height_u64,
            byte_offset_u64,
            pitch_bytes_u64,
            num_components_u32,
            bit_depth_u32,
            signed_u32
        );
        let num_pixels = (request.width as usize)
            .checked_mul(request.height as usize)
            .ok_or(CudaError::ImageTooLarge {
                width: request.width,
                height: request.height,
                channels: usize::from(request.num_components),
            })?;
        let geometry = j2k_forward_rct_launch_geometry(num_pixels)
            .ok_or(CudaError::LengthTooLarge { len: num_pixels })?;

        self.launch_kernel(function, geometry, &mut params)
    }

    fn launch_j2k_forward_ict_buffers(
        &self,
        plane0: &CudaDeviceBuffer,
        plane1: &CudaDeviceBuffer,
        plane2: &CudaDeviceBuffer,
        len: usize,
    ) -> Result<(), CudaError> {
        self.launch_j2k_forward_ict_ptrs(
            plane0.device_ptr(),
            plane1.device_ptr(),
            plane2.device_ptr(),
            len,
        )
    }

    fn launch_j2k_forward_ict_ptrs(
        &self,
        plane0: CuDevicePtr,
        plane1: CuDevicePtr,
        plane2: CuDevicePtr,
        len: usize,
    ) -> Result<(), CudaError> {
        let function = self.j2k_encode_kernel_function(CudaKernel::J2kForwardIct)?;
        let mut plane0_ptr = plane0;
        let mut plane1_ptr = plane1;
        let mut plane2_ptr = plane2;
        let mut len_u64 = u64::try_from(len).map_err(|_| CudaError::LengthTooLarge { len })?;
        let mut params = cuda_kernel_params!(plane0_ptr, plane1_ptr, plane2_ptr, len_u64);
        let geometry =
            j2k_forward_rct_launch_geometry(len).ok_or(CudaError::LengthTooLarge { len })?;

        self.launch_kernel(function, geometry, &mut params)
    }

    fn launch_j2k_forward_dwt53_pass(
        &self,
        kernel: CudaKernel,
        input: &CudaDeviceBuffer,
        output: &CudaDeviceBuffer,
        pass: CudaDwt53Pass,
    ) -> Result<(), CudaError> {
        let function = self.j2k_encode_kernel_function(kernel)?;
        let mut input_ptr = input.device_ptr();
        let mut output_ptr = output.device_ptr();
        let mut full_width = pass.full_width;
        let mut current_width = pass.current_width;
        let mut current_height = pass.current_height;
        let mut low_extent = pass.low_extent;
        let mut params = cuda_kernel_params!(
            input_ptr,
            output_ptr,
            full_width,
            current_width,
            current_height,
            low_extent
        );
        let geometry = j2k_dwt53_launch_geometry(current_width, current_height).ok_or(
            CudaError::ImageTooLarge {
                width: pass.current_width,
                height: pass.current_height,
                channels: 1,
            },
        )?;
        self.launch_kernel(function, geometry, &mut params)
    }

    fn launch_j2k_quantize_subband(
        &self,
        samples: &CudaDeviceBuffer,
        coefficients: &CudaDeviceBuffer,
        len: usize,
        job: CudaJ2kQuantizeJob,
    ) -> Result<(), CudaError> {
        let function = self.j2k_encode_kernel_function(CudaKernel::J2kQuantizeSubband)?;
        let mut samples_ptr = samples.device_ptr();
        let mut coefficients_ptr = coefficients.device_ptr();
        let mut len_u64 = u64::try_from(len).map_err(|_| CudaError::LengthTooLarge { len })?;
        let mut step_exponent = u32::from(job.step_exponent);
        let mut step_mantissa = u32::from(job.step_mantissa);
        let mut range_bits = u32::from(job.range_bits);
        let mut reversible = u32::from(job.reversible);
        let mut params = cuda_kernel_params!(
            samples_ptr,
            coefficients_ptr,
            len_u64,
            step_exponent,
            step_mantissa,
            range_bits,
            reversible
        );
        let geometry =
            j2k_forward_rct_launch_geometry(len).ok_or(CudaError::LengthTooLarge { len })?;

        self.launch_kernel(function, geometry, &mut params)
    }

    fn launch_j2k_quantize_subband_region(
        &self,
        samples: &CudaDeviceBuffer,
        coefficients: &CudaDeviceBuffer,
        job: CudaJ2kQuantizeSubbandRegionJob,
    ) -> Result<(), CudaError> {
        let function = self.j2k_encode_kernel_function(CudaKernel::J2kQuantizeSubbandStrided)?;
        let mut samples_ptr = samples.device_ptr();
        let mut coefficients_ptr = coefficients.device_ptr();
        let mut x0 = job.x0;
        let mut y0 = job.y0;
        let mut width = job.width;
        let mut height = job.height;
        let mut stride = job.stride;
        let mut step_exponent = u32::from(job.quantization.step_exponent);
        let mut step_mantissa = u32::from(job.quantization.step_mantissa);
        let mut range_bits = u32::from(job.quantization.range_bits);
        let mut reversible = u32::from(job.quantization.reversible);
        let mut params = cuda_kernel_params!(
            samples_ptr,
            coefficients_ptr,
            x0,
            y0,
            width,
            height,
            stride,
            step_exponent,
            step_mantissa,
            range_bits,
            reversible
        );
        let geometry =
            j2k_dwt53_launch_geometry(job.width, job.height).ok_or(CudaError::ImageTooLarge {
                width: job.width,
                height: job.height,
                channels: 1,
            })?;

        self.launch_kernel(function, geometry, &mut params)
    }
}

/// Resident f32 component planes produced by CUDA JPEG 2000 encode preparation.
#[derive(Debug)]
#[doc(hidden)]
pub struct CudaJ2kResidentComponents {
    pub(crate) buffer: CudaDeviceBuffer,
    pub(crate) num_pixels: usize,
    pub(crate) num_components: u8,
    pub(crate) execution: CudaExecutionStats,
}

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
            return Ok(vec![Vec::new(); usize::from(self.num_components)]);
        }
        let sample_count = self
            .num_pixels
            .checked_mul(usize::from(self.num_components))
            .ok_or(CudaError::LengthTooLarge {
                len: self.num_pixels,
            })?;
        let mut flattened = vec![0.0f32; sample_count];
        self.buffer
            .copy_to_host(f32_slice_as_bytes_mut(&mut flattened))?;
        Ok(flattened
            .chunks_exact(self.num_pixels)
            .map(<[f32]>::to_vec)
            .collect())
    }

    fn component_plane_device_ptr(&self, component: u8) -> Result<CuDevicePtr, CudaError> {
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
        let offset =
            u64::try_from(offset).map_err(|_| CudaError::LengthTooLarge { len: offset })?;
        self.buffer
            .device_ptr()
            .checked_add(offset)
            .ok_or(CudaError::LengthTooLarge {
                len: self.buffer.byte_len(),
            })
    }
}

/// Host-visible component planes produced by CUDA pixel deinterleave.
#[derive(Debug)]
#[doc(hidden)]
pub struct CudaJ2kDeinterleavedComponents {
    pub(crate) components: Vec<Vec<f32>>,
    pub(crate) execution: CudaExecutionStats,
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

/// Forward 5/3 DWT output and level metadata.
#[derive(Debug)]
#[doc(hidden)]
pub struct CudaDwt53Output {
    pub(crate) transformed: Vec<f32>,
    pub(crate) levels: Vec<CudaDwt53LevelShape>,
    pub(crate) ll_width: u32,
    pub(crate) ll_height: u32,
    pub(crate) execution: CudaExecutionStats,
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

/// Resident forward 5/3 DWT output and level metadata.
#[derive(Debug)]
#[doc(hidden)]
pub struct CudaResidentDwt53Output {
    pub(crate) buffer: CudaDeviceBuffer,
    pub(crate) sample_count: usize,
    pub(crate) levels: Vec<CudaDwt53LevelShape>,
    pub(crate) ll_width: u32,
    pub(crate) ll_height: u32,
    pub(crate) execution: CudaExecutionStats,
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
        let mut transformed = vec![0f32; self.sample_count];
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

/// Forward 9/7 DWT output and level metadata.
#[derive(Debug)]
#[doc(hidden)]
pub struct CudaDwt97Output {
    pub(crate) transformed: Vec<f32>,
    pub(crate) levels: Vec<CudaDwt53LevelShape>,
    pub(crate) ll_width: u32,
    pub(crate) ll_height: u32,
    pub(crate) execution: CudaExecutionStats,
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

/// Resident forward 9/7 DWT output and level metadata.
#[derive(Debug)]
#[doc(hidden)]
pub struct CudaResidentDwt97Output {
    pub(crate) buffer: CudaDeviceBuffer,
    pub(crate) sample_count: usize,
    pub(crate) levels: Vec<CudaDwt53LevelShape>,
    pub(crate) ll_width: u32,
    pub(crate) ll_height: u32,
    pub(crate) execution: CudaExecutionStats,
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
        let mut transformed = vec![0f32; self.sample_count];
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

/// JPEG 2000 sub-band quantization parameters.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[doc(hidden)]
pub struct CudaJ2kQuantizeJob {
    /// Quantization step-size exponent.
    pub step_exponent: u16,
    /// Quantization step-size mantissa.
    pub step_mantissa: u16,
    /// Nominal range bits for this sub-band.
    pub range_bits: u8,
    /// Whether to use reversible integer quantization.
    pub reversible: bool,
}

/// Resident strided sub-band rectangle and quantization parameters.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[doc(hidden)]
pub struct CudaJ2kQuantizeSubbandRegionJob {
    /// X offset, in f32 samples, of the sub-band rectangle inside the resident plane.
    pub x0: u32,
    /// Y offset, in f32 samples, of the sub-band rectangle inside the resident plane.
    pub y0: u32,
    /// Sub-band rectangle width in samples.
    pub width: u32,
    /// Sub-band rectangle height in samples.
    pub height: u32,
    /// Resident source row stride in f32 samples.
    pub stride: u32,
    /// Quantization parameters applied to every source sample.
    pub quantization: CudaJ2kQuantizeJob,
}

/// Quantized JPEG 2000 sub-band coefficients and execution metadata.
#[derive(Debug)]
#[doc(hidden)]
pub struct CudaJ2kQuantizedSubband {
    pub(crate) coefficients: Vec<i32>,
    pub(crate) execution: CudaExecutionStats,
}

impl CudaJ2kQuantizedSubband {
    /// Quantized sub-band coefficients downloaded to host memory.
    pub fn coefficients(&self) -> &[i32] {
        &self.coefficients
    }

    /// CUDA execution counters for the quantization stage.
    pub fn execution(&self) -> CudaExecutionStats {
        self.execution
    }
}

/// Device-resident quantized JPEG 2000 sub-band coefficients and execution metadata.
#[derive(Debug)]
#[doc(hidden)]
pub struct CudaJ2kResidentQuantizedSubband {
    pub(crate) coefficients: CudaDeviceBuffer,
    pub(crate) coefficient_count: usize,
    pub(crate) execution: CudaExecutionStats,
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
        let mut coefficients = vec![0i32; self.coefficient_count];
        self.coefficients
            .copy_to_host(i32_slice_as_bytes_mut(&mut coefficients))?;
        Ok(coefficients)
    }

    /// CUDA execution counters for the quantization stage.
    pub fn execution(&self) -> CudaExecutionStats {
        self.execution
    }
}

/// Shape metadata for one forward 5/3 DWT level.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[doc(hidden)]
pub struct CudaDwt53LevelShape {
    /// Input level width.
    pub width: u32,
    /// Input level height.
    pub height: u32,
    /// Low-pass width.
    pub low_width: u32,
    /// Low-pass height.
    pub low_height: u32,
    /// High-pass width.
    pub high_width: u32,
    /// High-pass height.
    pub high_height: u32,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct CudaDwt53Pass {
    pub(crate) full_width: u32,
    pub(crate) current_width: u32,
    pub(crate) current_height: u32,
    pub(crate) low_extent: u32,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct CudaDwt53LevelPass {
    pub(crate) full_width: u32,
    pub(crate) current_width: u32,
    pub(crate) current_height: u32,
}

/// Backend stage timings for a same-geometry 9/7 (or fused code-block) batch.
///
/// Mirrors `j2k-transcode`'s `Dwt97BatchStageTimings`; kept local because
/// `j2k-cuda-runtime` does not depend on `j2k-transcode`. The dispatch
/// layer maps this onto the transcode type.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[doc(hidden)]
pub struct CudaDwt97BatchStageTimings {
    /// Buffer allocation plus host-to-device block upload time, microseconds.
    pub pack_upload_us: u128,
    /// IDCT plus horizontal 9/7 row-lift stage time, microseconds.
    pub idct_row_lift_us: u128,
    /// Vertical 9/7 column-lift stage time, microseconds.
    pub column_lift_us: u128,
    /// Code-block quantization stage time, microseconds (0 for the band path).
    pub quantize_codeblock_us: u128,
    /// Resident HT code-block encode time, microseconds.
    pub ht_encode_us: u128,
    /// Resident HT code-block encode dispatches.
    pub ht_codeblock_dispatches: usize,
    /// Device-to-host readback and unpack time, microseconds.
    pub readback_us: u128,
}

pub(crate) fn validate_quantize_region(
    job: CudaJ2kQuantizeSubbandRegionJob,
    available_samples: usize,
) -> Result<(), CudaError> {
    if job.width == 0 || job.height == 0 {
        return Ok(());
    }
    if job.stride == 0
        || job
            .x0
            .checked_add(job.width)
            .is_none_or(|end_x| end_x > job.stride)
    {
        return Err(CudaError::LengthTooLarge {
            len: available_samples,
        });
    }

    let last_sample = (job.y0 as usize)
        .checked_add(job.height as usize - 1)
        .and_then(|row| row.checked_mul(job.stride as usize))
        .and_then(|row| row.checked_add(job.x0 as usize))
        .and_then(|row| row.checked_add(job.width as usize))
        .ok_or(CudaError::LengthTooLarge {
            len: available_samples,
        })?;
    if last_sample > available_samples {
        return Err(CudaError::OutputTooSmall {
            required: last_sample
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or(CudaError::LengthTooLarge { len: last_sample })?,
            have: available_samples
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or(CudaError::LengthTooLarge {
                    len: available_samples,
                })?,
        });
    }
    Ok(())
}
