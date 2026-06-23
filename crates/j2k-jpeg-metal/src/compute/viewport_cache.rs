// SPDX-License-Identifier: MIT OR Apache-2.0

use std::mem::size_of;

use j2k_core::{PixelFormat, Rect};
use j2k_jpeg::{ColorSpace as JpegColorSpace, ComponentRowWriter};
use j2k_metal_support::dispatch_2d_pipeline;
use metal::{Buffer, Device, MTLResourceOptions};

use crate::buffers::new_private_buffer;
use crate::{Error, Surface};

use super::{
    batch_output_buffer_or_new, bind_three_plane_pack, validate_rgba_texture_batch_output,
    JpegPackParams, JpegRgb8ToRgbaTextureParams, MetalRuntime, MODE_GRAY, MODE_RGB, MODE_YCBCR,
    OUT_GRAY, OUT_RGB, OUT_RGBA,
};

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum PlaneMode {
    Gray,
    YCbCr,
    Rgb,
}

#[cfg(target_os = "macos")]
pub(super) struct PlaneStage {
    pub(super) dims: (u32, u32),
    pub(super) mode: PlaneMode,
    pub(super) plane0: Buffer,
    pub(super) plane1: Option<Buffer>,
    pub(super) plane2: Option<Buffer>,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
enum PlaneStageResidency {
    CpuStagedMetalUpload,
    MetalResidentDecode,
}

#[cfg(target_os = "macos")]
pub(super) struct ViewportPlaneWriter<'a> {
    pub(super) stage: &'a mut PlaneStage,
    pub(super) dest: Rect,
}

#[cfg(target_os = "macos")]
pub(super) struct CachedViewportPlanes {
    pub(super) dims: (u32, u32),
    pub(super) mode: PlaneMode,
    pub(super) plane0: Buffer,
    pub(super) plane1: Option<Buffer>,
    pub(super) plane2: Option<Buffer>,
}

#[cfg(target_os = "macos")]
impl PlaneStage {
    pub(super) fn new(
        device: &Device,
        color_space: JpegColorSpace,
        dims: (u32, u32),
    ) -> Result<Self, Error> {
        let len = dims.0 as usize * dims.1 as usize;
        let plane0 = device.new_buffer(len as u64, MTLResourceOptions::StorageModeShared);
        let (mode, plane1, plane2) = match color_space {
            JpegColorSpace::Grayscale => (PlaneMode::Gray, None, None),
            JpegColorSpace::YCbCr => (
                PlaneMode::YCbCr,
                Some(device.new_buffer(len as u64, MTLResourceOptions::StorageModeShared)),
                Some(device.new_buffer(len as u64, MTLResourceOptions::StorageModeShared)),
            ),
            JpegColorSpace::Rgb => (
                PlaneMode::Rgb,
                Some(device.new_buffer(len as u64, MTLResourceOptions::StorageModeShared)),
                Some(device.new_buffer(len as u64, MTLResourceOptions::StorageModeShared)),
            ),
            JpegColorSpace::Cmyk | JpegColorSpace::Ycck => {
                return Err(Error::MetalKernel {
                    message: "Metal compute path does not support CMYK/YCCK JPEG output"
                        .to_string(),
                })
            }
        };

        Ok(Self {
            dims,
            mode,
            plane0,
            plane1,
            plane2,
        })
    }

    pub(super) fn finish_with_runtime(
        self,
        runtime: &MetalRuntime,
        fmt: PixelFormat,
    ) -> Result<Surface, Error> {
        self.finish_with_runtime_and_residency(
            runtime,
            fmt,
            PlaneStageResidency::CpuStagedMetalUpload,
        )
    }

    pub(super) fn finish_resident_with_runtime(
        self,
        runtime: &MetalRuntime,
        fmt: PixelFormat,
    ) -> Result<Surface, Error> {
        self.finish_with_runtime_and_residency(
            runtime,
            fmt,
            PlaneStageResidency::MetalResidentDecode,
        )
    }

    fn finish_with_runtime_and_residency(
        self,
        runtime: &MetalRuntime,
        fmt: PixelFormat,
        residency: PlaneStageResidency,
    ) -> Result<Surface, Error> {
        match (self.mode, fmt) {
            (PlaneMode::Gray | PlaneMode::YCbCr, PixelFormat::Gray8) => Ok(
                surface_from_plane_buffer(self.plane0, self.dims, fmt, residency),
            ),
            (
                PlaneMode::Gray | PlaneMode::YCbCr | PlaneMode::Rgb,
                PixelFormat::Rgb8 | PixelFormat::Rgba8,
            )
            | (PlaneMode::Rgb, PixelFormat::Gray8) => {
                Ok(self.dispatch_with_runtime(runtime, fmt, residency))
            }
            _ => Err(Error::MetalKernel {
                message: format!("unsupported JPEG Metal pixel format {fmt:?}"),
            }),
        }
    }

    fn dispatch_with_runtime(
        self,
        runtime: &MetalRuntime,
        fmt: PixelFormat,
        residency: PlaneStageResidency,
    ) -> Surface {
        let pitch_bytes = self.dims.0 as usize * fmt.bytes_per_pixel();
        let out_buffer = runtime.device.new_buffer(
            (pitch_bytes * self.dims.1 as usize) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let params = JpegPackParams {
            width: self.dims.0,
            height: self.dims.1,
            out_stride: u32::try_from(pitch_bytes).expect("JPEG Metal output stride fits in u32"),
            alpha: u32::from(u8::MAX),
            mode: match self.mode {
                PlaneMode::Gray => MODE_GRAY,
                PlaneMode::YCbCr => MODE_YCBCR,
                PlaneMode::Rgb => MODE_RGB,
            },
            out_format: match fmt {
                PixelFormat::Gray8 => OUT_GRAY,
                PixelFormat::Rgb8 => OUT_RGB,
                PixelFormat::Rgba8 => OUT_RGBA,
                _ => unreachable!("validated by finish"),
            },
        };

        let command_buffer = runtime.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&runtime.pack_pipeline);
        bind_three_plane_pack::<JpegPackParams>(
            encoder,
            [
                Some(&self.plane0),
                self.plane1.as_ref(),
                self.plane2.as_ref(),
            ],
            &out_buffer,
            &params,
        );
        dispatch_2d_pipeline(encoder, &runtime.pack_pipeline, self.dims);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        surface_from_plane_buffer(out_buffer, self.dims, fmt, residency)
    }

    pub(super) fn finish_rgb8_into_output_with_runtime(
        self,
        runtime: &MetalRuntime,
        output: &crate::MetalBatchOutputBuffer,
    ) -> Result<Surface, Error> {
        let fmt = PixelFormat::Rgb8;
        let pitch_bytes = self.dims.0 as usize * fmt.bytes_per_pixel();
        let tile_len = pitch_bytes * self.dims.1 as usize;
        let out_buffer =
            batch_output_buffer_or_new(runtime, Some(output), self.dims, 1, pitch_bytes, tile_len)?;
        let params = JpegPackParams {
            width: self.dims.0,
            height: self.dims.1,
            out_stride: u32::try_from(pitch_bytes).expect("JPEG Metal output stride fits in u32"),
            alpha: u32::from(u8::MAX),
            mode: match self.mode {
                PlaneMode::Gray => MODE_GRAY,
                PlaneMode::YCbCr => MODE_YCBCR,
                PlaneMode::Rgb => MODE_RGB,
            },
            out_format: OUT_RGB,
        };

        let command_buffer = runtime.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&runtime.pack_pipeline);
        bind_three_plane_pack::<JpegPackParams>(
            encoder,
            [
                Some(&self.plane0),
                self.plane1.as_ref(),
                self.plane2.as_ref(),
            ],
            &out_buffer,
            &params,
        );
        dispatch_2d_pipeline(encoder, &runtime.pack_pipeline, self.dims);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(Surface::from_metal_buffer_offset(
            out_buffer, self.dims, fmt, 0,
        ))
    }

    pub(super) fn finish_rgba8_into_texture_output_with_runtime(
        self,
        runtime: &MetalRuntime,
        output: &crate::MetalBatchTextureOutput,
    ) -> Result<crate::MetalTextureTile, Error> {
        let rgb_pitch_bytes = self.dims.0 as usize * PixelFormat::Rgb8.bytes_per_pixel();
        let rgb_tile_len = rgb_pitch_bytes * self.dims.1 as usize;
        let rgba_tile_len =
            self.dims.0 as usize * self.dims.1 as usize * PixelFormat::Rgba8.bytes_per_pixel();
        validate_rgba_texture_batch_output(output, self.dims, 1, rgba_tile_len)?;
        let out_buffer = {
            let mut batch_scratch = runtime.batch_scratch()?;
            batch_scratch.private_buffer(
                &runtime.device,
                "viewport_sparse_rgba_texture_rgb8",
                rgb_tile_len,
            )
        };
        let texture = output.texture(0).ok_or_else(|| Error::MetalKernel {
            message: "JPEG Metal batch texture output slot was missing".to_string(),
        })?;
        let pack_params = JpegPackParams {
            width: self.dims.0,
            height: self.dims.1,
            out_stride: u32::try_from(rgb_pitch_bytes)
                .expect("JPEG Metal output stride fits in u32"),
            alpha: u32::from(u8::MAX),
            mode: match self.mode {
                PlaneMode::Gray => MODE_GRAY,
                PlaneMode::YCbCr => MODE_YCBCR,
                PlaneMode::Rgb => MODE_RGB,
            },
            out_format: OUT_RGB,
        };
        let texture_params = JpegRgb8ToRgbaTextureParams {
            width: self.dims.0,
            height: self.dims.1,
            in_stride: u32::try_from(rgb_pitch_bytes)
                .expect("JPEG Metal RGB texture input stride fits in u32"),
            alpha: u32::from(u8::MAX),
        };

        let command_buffer = runtime.queue.new_command_buffer();
        let pack_encoder = command_buffer.new_compute_command_encoder();
        pack_encoder.set_compute_pipeline_state(&runtime.pack_pipeline);
        bind_three_plane_pack::<JpegPackParams>(
            pack_encoder,
            [
                Some(&self.plane0),
                self.plane1.as_ref(),
                self.plane2.as_ref(),
            ],
            &out_buffer,
            &pack_params,
        );
        dispatch_2d_pipeline(pack_encoder, &runtime.pack_pipeline, self.dims);
        pack_encoder.end_encoding();

        let texture_encoder = command_buffer.new_compute_command_encoder();
        texture_encoder.set_compute_pipeline_state(&runtime.rgb8_to_rgba_texture_pipeline);
        texture_encoder.set_buffer(0, Some(&out_buffer), 0);
        texture_encoder.set_bytes(
            1,
            size_of::<JpegRgb8ToRgbaTextureParams>() as u64,
            (&raw const texture_params).cast(),
        );
        texture_encoder.set_texture(0, Some(texture));
        dispatch_2d_pipeline(
            texture_encoder,
            &runtime.rgb8_to_rgba_texture_pipeline,
            self.dims,
        );
        texture_encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let texture = output.clone_texture(0).ok_or_else(|| Error::MetalKernel {
            message: "JPEG Metal batch texture output slot was missing".to_string(),
        })?;
        Ok(crate::MetalTextureTile::new(
            texture,
            self.dims,
            PixelFormat::Rgba8,
        ))
    }

    pub(super) fn dispatch_private_rgb8_with_runtime(
        self,
        runtime: &MetalRuntime,
        status_buffer: Buffer,
    ) -> crate::ResidentPrivateJpegTile {
        let fmt = PixelFormat::Rgb8;
        let pitch_bytes = self.dims.0 as usize * fmt.bytes_per_pixel();
        let out_buffer = new_private_buffer(&runtime.device, pitch_bytes * self.dims.1 as usize);
        let params = JpegPackParams {
            width: self.dims.0,
            height: self.dims.1,
            out_stride: u32::try_from(pitch_bytes).expect("JPEG Metal output stride fits in u32"),
            alpha: u32::from(u8::MAX),
            mode: match self.mode {
                PlaneMode::Gray => MODE_GRAY,
                PlaneMode::YCbCr => MODE_YCBCR,
                PlaneMode::Rgb => MODE_RGB,
            },
            out_format: OUT_RGB,
        };

        let command_buffer = runtime.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&runtime.pack_pipeline);
        bind_three_plane_pack::<JpegPackParams>(
            encoder,
            [
                Some(&self.plane0),
                self.plane1.as_ref(),
                self.plane2.as_ref(),
            ],
            &out_buffer,
            &params,
        );
        dispatch_2d_pipeline(encoder, &runtime.pack_pipeline, self.dims);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        let command_buffer = command_buffer.to_owned();

        crate::ResidentPrivateJpegTile {
            buffer: out_buffer,
            byte_offset: 0,
            dimensions: self.dims,
            pixel_format: fmt,
            pitch_bytes,
            status_buffer,
            command_buffer,
        }
    }
}

#[cfg(target_os = "macos")]
fn surface_from_plane_buffer(
    buffer: Buffer,
    dims: (u32, u32),
    fmt: PixelFormat,
    residency: PlaneStageResidency,
) -> Surface {
    match residency {
        PlaneStageResidency::CpuStagedMetalUpload => {
            Surface::from_cpu_staged_metal_buffer(buffer, dims, fmt)
        }
        PlaneStageResidency::MetalResidentDecode => Surface::from_metal_buffer(buffer, dims, fmt),
    }
}

#[cfg(target_os = "macos")]
impl ComponentRowWriter for PlaneStage {
    fn write_gray_row(&mut self, y: u32, gray_row: &[u8]) -> Result<(), j2k_jpeg::JpegError> {
        let width = self.dims.0 as usize;
        write_row_u8(&self.plane0, y, width, gray_row);
        Ok(())
    }

    fn write_ycbcr_row(
        &mut self,
        y: u32,
        y_row: &[u8],
        chroma_blue_row: &[u8],
        chroma_red_row: &[u8],
    ) -> Result<(), j2k_jpeg::JpegError> {
        let width = self.dims.0 as usize;
        write_row_u8(&self.plane0, y, width, y_row);
        write_row_u8(
            self.plane1.as_ref().expect("Cb plane"),
            y,
            width,
            chroma_blue_row,
        );
        write_row_u8(
            self.plane2.as_ref().expect("Cr plane"),
            y,
            width,
            chroma_red_row,
        );
        Ok(())
    }

    fn write_rgb_row(
        &mut self,
        y: u32,
        r_row: &[u8],
        g_row: &[u8],
        b_row: &[u8],
    ) -> Result<(), j2k_jpeg::JpegError> {
        let width = self.dims.0 as usize;
        write_row_u8(&self.plane0, y, width, r_row);
        write_row_u8(self.plane1.as_ref().expect("G plane"), y, width, g_row);
        write_row_u8(self.plane2.as_ref().expect("B plane"), y, width, b_row);
        Ok(())
    }
}

#[cfg(target_os = "macos")]
fn write_row_u8(buffer: &Buffer, y: u32, width: usize, src: &[u8]) {
    let row_start = y as usize * width;
    let row_end = row_start + width;
    let len = width * (y as usize + 1);
    // SAFETY: Metal buffer access follows validated sizes and synchronized command completion.
    let dst = unsafe {
        core::slice::from_raw_parts_mut(buffer.contents().cast::<u8>(), len.max(row_end))
    };
    dst[row_start..row_end].copy_from_slice(&src[..width]);
}

#[cfg(target_os = "macos")]
fn write_row_u8_at(buffer: &Buffer, y: u32, x: u32, full_width: usize, src: &[u8]) {
    let row_start = y as usize * full_width + x as usize;
    let row_end = row_start + src.len();
    let len = full_width * (y as usize + 1);
    // SAFETY: Metal buffer access follows validated sizes and synchronized command completion.
    let dst = unsafe {
        core::slice::from_raw_parts_mut(buffer.contents().cast::<u8>(), len.max(row_end))
    };
    dst[row_start..row_end].copy_from_slice(src);
}

#[cfg(target_os = "macos")]
fn plane_mode_for_color_space(color_space: JpegColorSpace) -> Result<PlaneMode, Error> {
    match color_space {
        JpegColorSpace::Grayscale => Ok(PlaneMode::Gray),
        JpegColorSpace::YCbCr => Ok(PlaneMode::YCbCr),
        JpegColorSpace::Rgb => Ok(PlaneMode::Rgb),
        JpegColorSpace::Cmyk | JpegColorSpace::Ycck => Err(Error::MetalKernel {
            message: "Metal compute path does not support CMYK/YCCK JPEG output".to_string(),
        }),
    }
}

#[cfg(target_os = "macos")]
fn clear_buffer(buffer: &Buffer, len: usize) {
    fill_buffer(buffer, len, 0);
}

#[cfg(target_os = "macos")]
fn fill_buffer(buffer: &Buffer, len: usize, value: u8) {
    // SAFETY: Metal buffer access follows validated sizes and synchronized command completion.
    unsafe {
        core::ptr::write_bytes(buffer.contents().cast::<u8>(), value, len);
    }
}

#[cfg(target_os = "macos")]
pub(super) fn cached_plane_stage(
    runtime: &MetalRuntime,
    color_space: JpegColorSpace,
    dims: (u32, u32),
) -> Result<PlaneStage, Error> {
    let mode = plane_mode_for_color_space(color_space)?;
    let mut slot = runtime.viewport_plane_cache()?;
    let len = dims.0 as usize * dims.1 as usize;
    let refresh = slot
        .as_ref()
        .is_none_or(|cached| cached.dims != dims || cached.mode != mode);
    if refresh {
        let plane0 = runtime
            .device
            .new_buffer(len as u64, MTLResourceOptions::StorageModeShared);
        let (plane1, plane2) = match mode {
            PlaneMode::Gray => (None, None),
            PlaneMode::YCbCr | PlaneMode::Rgb => (
                Some(
                    runtime
                        .device
                        .new_buffer(len as u64, MTLResourceOptions::StorageModeShared),
                ),
                Some(
                    runtime
                        .device
                        .new_buffer(len as u64, MTLResourceOptions::StorageModeShared),
                ),
            ),
        };
        *slot = Some(CachedViewportPlanes {
            dims,
            mode,
            plane0,
            plane1,
            plane2,
        });
    }

    let cached = slot.as_ref().expect("viewport plane cache");
    let stage = PlaneStage {
        dims,
        mode,
        plane0: cached.plane0.clone(),
        plane1: cached.plane1.clone(),
        plane2: cached.plane2.clone(),
    };
    drop(slot);

    clear_buffer(&stage.plane0, len);
    match stage.mode {
        PlaneMode::Gray => {}
        PlaneMode::YCbCr => {
            if let Some(plane1) = &stage.plane1 {
                fill_buffer(plane1, len, 128);
            }
            if let Some(plane2) = &stage.plane2 {
                fill_buffer(plane2, len, 128);
            }
        }
        PlaneMode::Rgb => {
            if let Some(plane1) = &stage.plane1 {
                clear_buffer(plane1, len);
            }
            if let Some(plane2) = &stage.plane2 {
                clear_buffer(plane2, len);
            }
        }
    }
    Ok(stage)
}

#[cfg(target_os = "macos")]
impl ComponentRowWriter for ViewportPlaneWriter<'_> {
    fn write_gray_row(&mut self, y: u32, gray_row: &[u8]) -> Result<(), j2k_jpeg::JpegError> {
        write_row_u8_at(
            &self.stage.plane0,
            self.dest.y + y,
            self.dest.x,
            self.stage.dims.0 as usize,
            gray_row,
        );
        Ok(())
    }

    fn write_ycbcr_row(
        &mut self,
        y: u32,
        y_row: &[u8],
        chroma_blue_row: &[u8],
        chroma_red_row: &[u8],
    ) -> Result<(), j2k_jpeg::JpegError> {
        let width = self.stage.dims.0 as usize;
        write_row_u8_at(
            &self.stage.plane0,
            self.dest.y + y,
            self.dest.x,
            width,
            y_row,
        );
        write_row_u8_at(
            self.stage.plane1.as_ref().expect("Cb plane"),
            self.dest.y + y,
            self.dest.x,
            width,
            chroma_blue_row,
        );
        write_row_u8_at(
            self.stage.plane2.as_ref().expect("Cr plane"),
            self.dest.y + y,
            self.dest.x,
            width,
            chroma_red_row,
        );
        Ok(())
    }

    fn write_rgb_row(
        &mut self,
        y: u32,
        r_row: &[u8],
        g_row: &[u8],
        b_row: &[u8],
    ) -> Result<(), j2k_jpeg::JpegError> {
        let width = self.stage.dims.0 as usize;
        write_row_u8_at(
            &self.stage.plane0,
            self.dest.y + y,
            self.dest.x,
            width,
            r_row,
        );
        write_row_u8_at(
            self.stage.plane1.as_ref().expect("G plane"),
            self.dest.y + y,
            self.dest.x,
            width,
            g_row,
        );
        write_row_u8_at(
            self.stage.plane2.as_ref().expect("B plane"),
            self.dest.y + y,
            self.dest.x,
            width,
            b_row,
        );
        Ok(())
    }
}
