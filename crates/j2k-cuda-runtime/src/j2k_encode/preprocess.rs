// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    bytes::{f32_slice_as_bytes, f32_slice_as_bytes_mut},
    context::CudaContext,
    error::CudaError,
    execution::CudaExecutionStats,
    j2k_decode::CudaJ2kStridedInterleavedPixels,
};

use super::{
    validate_encode_buffer_context, CudaJ2kDeinterleavedComponents, CudaJ2kResidentComponents,
    J2kStridedDeinterleaveLaunch,
};

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
        validate_encode_buffer_context(self, [pixels])?;
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
        validate_encode_buffer_context(self, [&components.buffer])?;
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
        validate_encode_buffer_context(self, [&components.buffer])?;
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
}
