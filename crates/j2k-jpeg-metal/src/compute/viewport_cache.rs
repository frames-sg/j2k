// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(test)]
use std::mem::size_of;
#[cfg(target_os = "macos")]
use std::sync::{Arc, Condvar, Mutex};

use j2k_core::{PixelFormat, Rect};
use j2k_jpeg::{ColorSpace as JpegColorSpace, ComponentRowWriter, JpegError};
use j2k_metal_support::dispatch_2d_pipeline;
use metal::{Buffer, Device, MTLResourceOptions};

use crate::buffers::{checked_copy_bytes_to_buffer_at, checked_fill_buffer_u8, new_private_buffer};
use crate::{Error, Surface};

#[cfg(test)]
use super::{
    batch_output_buffer_or_new, validate_rgba_texture_batch_output, JpegRgb8ToRgbaTextureParams,
};
use super::{
    bind_three_plane_pack, commit_and_wait_jpeg, JpegPackParams, MetalRuntime, MODE_GRAY, MODE_RGB,
    MODE_YCBCR, OUT_GRAY, OUT_RGB, OUT_RGBA,
};

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum PlaneMode {
    Gray,
    YCbCr,
    Rgb,
}

#[cfg(target_os = "macos")]
pub(super) struct ViewportPlaneCacheGate {
    leased: Mutex<bool>,
    available: Condvar,
}

#[cfg(target_os = "macos")]
impl ViewportPlaneCacheGate {
    pub(super) fn new() -> Arc<Self> {
        Arc::new(Self {
            leased: Mutex::new(false),
            available: Condvar::new(),
        })
    }

    pub(super) fn acquire(self: &Arc<Self>) -> Result<ViewportPlaneCacheLease, Error> {
        let mut leased = self.leased.lock().map_err(|_| Error::MetalStatePoisoned {
            state: "JPEG Metal viewport plane cache lease",
        })?;
        while *leased {
            leased = self
                .available
                .wait(leased)
                .map_err(|_| Error::MetalStatePoisoned {
                    state: "JPEG Metal viewport plane cache lease",
                })?;
        }
        *leased = true;
        drop(leased);
        Ok(ViewportPlaneCacheLease {
            gate: Arc::clone(self),
        })
    }
}

#[cfg(target_os = "macos")]
pub(super) struct ViewportPlaneCacheLease {
    gate: Arc<ViewportPlaneCacheGate>,
}

#[cfg(target_os = "macos")]
impl Drop for ViewportPlaneCacheLease {
    fn drop(&mut self) {
        let mut leased = self
            .gate
            .leased
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        *leased = false;
        self.gate.available.notify_one();
    }
}

#[cfg(target_os = "macos")]
pub(super) struct PlaneStage {
    pub(super) dims: (u32, u32),
    pub(super) mode: PlaneMode,
    pub(super) plane0: Buffer,
    pub(super) plane1: Option<Buffer>,
    pub(super) plane2: Option<Buffer>,
    pub(super) cache_lease: Option<ViewportPlaneCacheLease>,
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
struct PlaneRowTarget<'a> {
    plane0: &'a Buffer,
    plane1: Option<&'a Buffer>,
    plane2: Option<&'a Buffer>,
    full_width: usize,
    origin_x: u32,
    origin_y: u32,
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
            cache_lease: None,
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
        let cache_owned = self.cache_lease.is_some();
        match (self.mode, fmt) {
            (PlaneMode::Gray | PlaneMode::YCbCr, PixelFormat::Gray8) if !cache_owned => Ok(
                surface_from_plane_buffer(self.plane0, self.dims, fmt, residency),
            ),
            (
                PlaneMode::Gray | PlaneMode::YCbCr | PlaneMode::Rgb,
                PixelFormat::Gray8 | PixelFormat::Rgb8 | PixelFormat::Rgba8,
            ) => self.dispatch_with_runtime(runtime, fmt, residency),
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
    ) -> Result<Surface, Error> {
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
        commit_and_wait_jpeg(command_buffer)?;

        Ok(surface_from_plane_buffer(
            out_buffer, self.dims, fmt, residency,
        ))
    }

    #[cfg(test)]
    pub(super) fn finish_rgb8_into_output_with_runtime(
        self,
        runtime: &MetalRuntime,
        output: &crate::MetalBatchOutputBuffer,
    ) -> Result<Surface, Error> {
        let _output_access = output.lock_for_safe_access()?;
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
        commit_and_wait_jpeg(command_buffer)?;

        Ok(Surface::from_batch_output_buffer_offset(
            output, self.dims, fmt, 0,
        ))
    }

    #[cfg(test)]
    #[expect(
        clippy::similar_names,
        reason = "RGB and RGBA identify distinct public pixel formats"
    )]
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
        let texture = output
            .texture_trusted(0)
            .ok_or_else(|| Error::MetalKernel {
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
        commit_and_wait_jpeg(command_buffer)?;

        let texture = output
            .clone_texture_trusted(0)
            .ok_or_else(|| Error::MetalKernel {
                message: "JPEG Metal batch texture output slot was missing".to_string(),
            })?;
        Ok(crate::MetalTextureTile::new(
            texture,
            output.clone_access_gate(),
            self.dims,
            PixelFormat::Rgba8,
        ))
    }

    pub(super) fn dispatch_private_rgb8_with_runtime(
        self,
        runtime: &MetalRuntime,
        status_buffer: Buffer,
    ) -> Result<crate::ResidentPrivateJpegTile, Error> {
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
        commit_and_wait_jpeg(command_buffer)?;
        let command_buffer = command_buffer.to_owned();

        Ok(crate::ResidentPrivateJpegTile::new(
            out_buffer,
            0,
            self.dims,
            fmt,
            pitch_bytes,
            status_buffer,
            command_buffer,
        ))
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
        self.row_target()
            .write_gray_row(y, gray_row)
            .map_err(jpeg_plane_write_error)
    }

    fn write_ycbcr_row(
        &mut self,
        y: u32,
        y_row: &[u8],
        chroma_blue_row: &[u8],
        chroma_red_row: &[u8],
    ) -> Result<(), j2k_jpeg::JpegError> {
        self.row_target()
            .write_ycbcr_row(y, y_row, chroma_blue_row, chroma_red_row)
            .map_err(jpeg_plane_write_error)
    }

    fn write_rgb_row(
        &mut self,
        y: u32,
        r_row: &[u8],
        g_row: &[u8],
        b_row: &[u8],
    ) -> Result<(), j2k_jpeg::JpegError> {
        self.row_target()
            .write_rgb_row(y, r_row, g_row, b_row)
            .map_err(jpeg_plane_write_error)
    }
}

#[cfg(target_os = "macos")]
impl PlaneStage {
    fn row_target(&self) -> PlaneRowTarget<'_> {
        PlaneRowTarget {
            plane0: &self.plane0,
            plane1: self.plane1.as_ref(),
            plane2: self.plane2.as_ref(),
            full_width: self.dims.0 as usize,
            origin_x: 0,
            origin_y: 0,
        }
    }
}

#[cfg(target_os = "macos")]
impl ViewportPlaneWriter<'_> {
    fn row_target(&self) -> PlaneRowTarget<'_> {
        PlaneRowTarget {
            plane0: &self.stage.plane0,
            plane1: self.stage.plane1.as_ref(),
            plane2: self.stage.plane2.as_ref(),
            full_width: self.stage.dims.0 as usize,
            origin_x: self.dest.x,
            origin_y: self.dest.y,
        }
    }
}

#[cfg(target_os = "macos")]
impl PlaneRowTarget<'_> {
    fn write_gray_row(&self, y: u32, gray_row: &[u8]) -> Result<(), Error> {
        self.write_plane_row(self.plane0, y, gray_row)
    }

    fn write_ycbcr_row(
        &self,
        y: u32,
        y_row: &[u8],
        chroma_blue_row: &[u8],
        chroma_red_row: &[u8],
    ) -> Result<(), Error> {
        self.write_plane_row(self.plane0, y, y_row)?;
        self.write_plane_row(self.plane1.expect("Cb plane"), y, chroma_blue_row)?;
        self.write_plane_row(self.plane2.expect("Cr plane"), y, chroma_red_row)
    }

    fn write_rgb_row(&self, y: u32, r_row: &[u8], g_row: &[u8], b_row: &[u8]) -> Result<(), Error> {
        self.write_plane_row(self.plane0, y, r_row)?;
        self.write_plane_row(self.plane1.expect("G plane"), y, g_row)?;
        self.write_plane_row(self.plane2.expect("B plane"), y, b_row)
    }

    fn write_plane_row(&self, buffer: &Buffer, y: u32, src: &[u8]) -> Result<(), Error> {
        let target_y = self
            .origin_y
            .checked_add(y)
            .ok_or_else(|| Error::MetalKernel {
                message: "JPEG Metal viewport plane row y offset overflow".to_string(),
            })?;
        checked_write_row_u8_at(buffer, target_y, self.origin_x, self.full_width, src)
    }
}

#[cfg(target_os = "macos")]
fn checked_write_row_u8_at(
    buffer: &Buffer,
    y: u32,
    x: u32,
    full_width: usize,
    src: &[u8],
) -> Result<(), Error> {
    let row_start = (y as usize)
        .checked_mul(full_width)
        .and_then(|offset| offset.checked_add(x as usize))
        .ok_or_else(|| Error::MetalKernel {
            message: "JPEG Metal viewport plane row offset overflow".to_string(),
        })?;
    checked_copy_bytes_to_buffer_at(buffer, row_start, src, "viewport plane row write")
}

#[cfg(target_os = "macos")]
fn jpeg_plane_write_error(_: Error) -> JpegError {
    JpegError::InternalInvariant {
        reason: "JPEG Metal viewport plane buffer write failed",
    }
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
fn clear_buffer(buffer: &Buffer, len: usize) -> Result<(), Error> {
    fill_buffer(buffer, len, 0)
}

#[cfg(target_os = "macos")]
fn fill_buffer(buffer: &Buffer, len: usize, value: u8) -> Result<(), Error> {
    checked_fill_buffer_u8(buffer, len, value, "viewport plane fill")
}

#[cfg(target_os = "macos")]
pub(super) fn cached_plane_stage(
    runtime: &MetalRuntime,
    color_space: JpegColorSpace,
    dims: (u32, u32),
) -> Result<PlaneStage, Error> {
    let mode = plane_mode_for_color_space(color_space)?;
    let cache_lease = runtime.viewport_plane_cache_lease()?;
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
        cache_lease: Some(cache_lease),
    };
    drop(slot);

    clear_buffer(&stage.plane0, len)?;
    match stage.mode {
        PlaneMode::Gray => {}
        PlaneMode::YCbCr => {
            if let Some(plane1) = &stage.plane1 {
                fill_buffer(plane1, len, 128)?;
            }
            if let Some(plane2) = &stage.plane2 {
                fill_buffer(plane2, len, 128)?;
            }
        }
        PlaneMode::Rgb => {
            if let Some(plane1) = &stage.plane1 {
                clear_buffer(plane1, len)?;
            }
            if let Some(plane2) = &stage.plane2 {
                clear_buffer(plane2, len)?;
            }
        }
    }
    Ok(stage)
}

#[cfg(target_os = "macos")]
impl ComponentRowWriter for ViewportPlaneWriter<'_> {
    fn write_gray_row(&mut self, y: u32, gray_row: &[u8]) -> Result<(), j2k_jpeg::JpegError> {
        self.row_target()
            .write_gray_row(y, gray_row)
            .map_err(jpeg_plane_write_error)
    }

    fn write_ycbcr_row(
        &mut self,
        y: u32,
        y_row: &[u8],
        chroma_blue_row: &[u8],
        chroma_red_row: &[u8],
    ) -> Result<(), j2k_jpeg::JpegError> {
        self.row_target()
            .write_ycbcr_row(y, y_row, chroma_blue_row, chroma_red_row)
            .map_err(jpeg_plane_write_error)
    }

    fn write_rgb_row(
        &mut self,
        y: u32,
        r_row: &[u8],
        g_row: &[u8],
        b_row: &[u8],
    ) -> Result<(), j2k_jpeg::JpegError> {
        self.row_target()
            .write_rgb_row(y, r_row, g_row, b_row)
            .map_err(jpeg_plane_write_error)
    }
}
