// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(test)]
use std::mem::size_of;
#[cfg(target_os = "macos")]
use std::sync::{Arc, Condvar, Mutex};

use j2k_core::{PixelFormat, Rect};
use j2k_jpeg::{ColorSpace as JpegColorSpace, ComponentRowWriter, JpegError};
use j2k_metal_support::dispatch_2d_pipeline;
use metal::{Buffer, Device};

use crate::buffers::{
    checked_copy_bytes_to_buffer_at, checked_fill_buffer_u8, new_private_buffer, new_shared_buffer,
};
use crate::{Error, Surface};

#[cfg(test)]
use super::{
    batch_output_buffer_or_new, validate_rgba_texture_batch_output, JpegRgb8ToRgbaTextureParams,
};
use super::{
    bind_three_plane_pack, commit_and_wait_jpeg, new_command_buffer, new_compute_command_encoder,
    JpegPackParams, MetalRuntime, MODE_GRAY, MODE_RGB, MODE_YCBCR, OUT_GRAY, OUT_RGB, OUT_RGBA,
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
fn checked_output_stride(pitch_bytes: usize, context: &'static str) -> Result<u32, Error> {
    u32::try_from(pitch_bytes).map_err(|_| Error::MetalKernel {
        message: format!("{context} ({pitch_bytes} bytes) exceeds the Metal u32 stride ABI"),
    })
}

#[cfg(target_os = "macos")]
fn checked_pack_output_format(fmt: PixelFormat) -> Result<u32, Error> {
    match fmt {
        PixelFormat::Gray8 => Ok(OUT_GRAY),
        PixelFormat::Rgb8 => Ok(OUT_RGB),
        PixelFormat::Rgba8 => Ok(OUT_RGBA),
        _ => Err(Error::MetalKernel {
            message: format!("unsupported JPEG Metal viewport pack format {fmt:?}"),
        }),
    }
}

#[cfg(target_os = "macos")]
fn required_plane<'a>(
    plane: Option<&'a Buffer>,
    component: &'static str,
) -> Result<&'a Buffer, Error> {
    plane.ok_or_else(|| Error::MetalKernel {
        message: format!("JPEG Metal viewport {component} plane is missing"),
    })
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
        external_live_bytes: usize,
    ) -> Result<Self, Error> {
        let mode = plane_mode_for_color_space(color_space)?;
        let len = viewport_plane_len(dims)?;
        let (plane0, plane1, plane2) =
            allocate_viewport_planes(device, mode, len, external_live_bytes)?;

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
        let pitch_bytes = crate::batch_allocation::checked_count_product(
            self.dims.0 as usize,
            fmt.bytes_per_pixel(),
            "JPEG Metal viewport output row bytes",
        )?;
        let out_len = crate::batch_allocation::checked_count_product(
            pitch_bytes,
            self.dims.1 as usize,
            "JPEG Metal viewport output bytes",
        )?;
        let out_buffer = new_shared_buffer(&runtime.device, out_len)?;
        let params = JpegPackParams {
            width: self.dims.0,
            height: self.dims.1,
            out_stride: checked_output_stride(pitch_bytes, "JPEG Metal viewport output stride")?,
            alpha: u32::from(u8::MAX),
            mode: match self.mode {
                PlaneMode::Gray => MODE_GRAY,
                PlaneMode::YCbCr => MODE_YCBCR,
                PlaneMode::Rgb => MODE_RGB,
            },
            out_format: checked_pack_output_format(fmt)?,
        };

        let command_buffer = new_command_buffer(&runtime.queue)?;
        let encoder = new_compute_command_encoder(&command_buffer)?;
        encoder.set_compute_pipeline_state(&runtime.pack_pipeline);
        bind_three_plane_pack::<JpegPackParams>(
            &encoder,
            [
                Some(&self.plane0),
                self.plane1.as_ref(),
                self.plane2.as_ref(),
            ],
            &out_buffer,
            &params,
        );
        dispatch_2d_pipeline(&encoder, &runtime.pack_pipeline, self.dims);
        encoder.end_encoding();
        commit_and_wait_jpeg(&command_buffer)?;

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
            out_stride: checked_output_stride(pitch_bytes, "JPEG Metal viewport output stride")?,
            alpha: u32::from(u8::MAX),
            mode: match self.mode {
                PlaneMode::Gray => MODE_GRAY,
                PlaneMode::YCbCr => MODE_YCBCR,
                PlaneMode::Rgb => MODE_RGB,
            },
            out_format: OUT_RGB,
        };

        let command_buffer = new_command_buffer(&runtime.queue)?;
        let encoder = new_compute_command_encoder(&command_buffer)?;
        encoder.set_compute_pipeline_state(&runtime.pack_pipeline);
        bind_three_plane_pack::<JpegPackParams>(
            &encoder,
            [
                Some(&self.plane0),
                self.plane1.as_ref(),
                self.plane2.as_ref(),
            ],
            &out_buffer,
            &params,
        );
        dispatch_2d_pipeline(&encoder, &runtime.pack_pipeline, self.dims);
        encoder.end_encoding();
        commit_and_wait_jpeg(&command_buffer)?;

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
            )?
        };
        let texture = output
            .texture_trusted(0)
            .ok_or_else(|| Error::MetalKernel {
                message: "JPEG Metal batch texture output slot was missing".to_string(),
            })?;
        let pack_params = JpegPackParams {
            width: self.dims.0,
            height: self.dims.1,
            out_stride: checked_output_stride(
                rgb_pitch_bytes,
                "JPEG Metal viewport RGB output stride",
            )?,
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
            in_stride: checked_output_stride(
                rgb_pitch_bytes,
                "JPEG Metal viewport RGB texture input stride",
            )?,
            alpha: u32::from(u8::MAX),
        };

        let command_buffer = new_command_buffer(&runtime.queue)?;
        let pack_encoder = new_compute_command_encoder(&command_buffer)?;
        pack_encoder.set_compute_pipeline_state(&runtime.pack_pipeline);
        bind_three_plane_pack::<JpegPackParams>(
            &pack_encoder,
            [
                Some(&self.plane0),
                self.plane1.as_ref(),
                self.plane2.as_ref(),
            ],
            &out_buffer,
            &pack_params,
        );
        dispatch_2d_pipeline(&pack_encoder, &runtime.pack_pipeline, self.dims);
        pack_encoder.end_encoding();

        let texture_encoder = new_compute_command_encoder(&command_buffer)?;
        texture_encoder.set_compute_pipeline_state(&runtime.rgb8_to_rgba_texture_pipeline);
        texture_encoder.set_buffer(0, Some(&out_buffer), 0);
        texture_encoder.set_bytes(
            1,
            size_of::<JpegRgb8ToRgbaTextureParams>() as u64,
            (&raw const texture_params).cast(),
        );
        texture_encoder.set_texture(0, Some(texture));
        dispatch_2d_pipeline(
            &texture_encoder,
            &runtime.rgb8_to_rgba_texture_pipeline,
            self.dims,
        );
        texture_encoder.end_encoding();
        commit_and_wait_jpeg(&command_buffer)?;

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
        let out_buffer = new_private_buffer(&runtime.device, pitch_bytes * self.dims.1 as usize)?;
        let params = JpegPackParams {
            width: self.dims.0,
            height: self.dims.1,
            out_stride: checked_output_stride(pitch_bytes, "JPEG Metal viewport output stride")?,
            alpha: u32::from(u8::MAX),
            mode: match self.mode {
                PlaneMode::Gray => MODE_GRAY,
                PlaneMode::YCbCr => MODE_YCBCR,
                PlaneMode::Rgb => MODE_RGB,
            },
            out_format: OUT_RGB,
        };

        let command_buffer = new_command_buffer(&runtime.queue)?;
        let encoder = new_compute_command_encoder(&command_buffer)?;
        encoder.set_compute_pipeline_state(&runtime.pack_pipeline);
        bind_three_plane_pack::<JpegPackParams>(
            &encoder,
            [
                Some(&self.plane0),
                self.plane1.as_ref(),
                self.plane2.as_ref(),
            ],
            &out_buffer,
            &params,
        );
        dispatch_2d_pipeline(&encoder, &runtime.pack_pipeline, self.dims);
        encoder.end_encoding();
        commit_and_wait_jpeg(&command_buffer)?;
        let command_buffer = command_buffer.clone();

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
        self.write_plane_row(required_plane(self.plane1, "Cb")?, y, chroma_blue_row)?;
        self.write_plane_row(required_plane(self.plane2, "Cr")?, y, chroma_red_row)
    }

    fn write_rgb_row(&self, y: u32, r_row: &[u8], g_row: &[u8], b_row: &[u8]) -> Result<(), Error> {
        self.write_plane_row(self.plane0, y, r_row)?;
        self.write_plane_row(required_plane(self.plane1, "G")?, y, g_row)?;
        self.write_plane_row(required_plane(self.plane2, "B")?, y, b_row)
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
fn viewport_plane_len(dims: (u32, u32)) -> Result<usize, Error> {
    crate::batch_allocation::checked_count_product(
        dims.0 as usize,
        dims.1 as usize,
        "JPEG Metal viewport plane bytes",
    )
    .map_err(Error::from)
}

#[cfg(target_os = "macos")]
fn allocate_viewport_planes(
    device: &Device,
    mode: PlaneMode,
    len: usize,
    external_live_bytes: usize,
) -> Result<(Buffer, Option<Buffer>, Option<Buffer>), Error> {
    let plane_count = if mode == PlaneMode::Gray { 1 } else { 3 };
    let total_bytes = crate::batch_allocation::checked_count_product(
        len,
        plane_count,
        "JPEG Metal viewport plane live allocation",
    )?;
    crate::batch_allocation::BatchMetadataBudget::with_external_live(
        "JPEG Metal viewport plane live allocation",
        external_live_bytes,
    )
    .preflight(&[crate::batch_allocation::BatchMetadataRequest::of::<u8>(
        total_bytes,
    )])?;
    let plane0 = new_shared_buffer(device, len)?;
    let (plane1, plane2) = if mode == PlaneMode::Gray {
        (None, None)
    } else {
        (
            Some(new_shared_buffer(device, len)?),
            Some(new_shared_buffer(device, len)?),
        )
    };
    Ok((plane0, plane1, plane2))
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
    external_live_bytes: usize,
) -> Result<PlaneStage, Error> {
    let mode = plane_mode_for_color_space(color_space)?;
    let cache_lease = runtime.viewport_plane_cache_lease()?;
    let mut slot = runtime.viewport_plane_cache()?;
    let len = viewport_plane_len(dims)?;
    let refresh = slot
        .as_ref()
        .is_none_or(|cached| cached.dims != dims || cached.mode != mode);
    if refresh {
        let (plane0, plane1, plane2) =
            allocate_viewport_planes(&runtime.device, mode, len, external_live_bytes)?;
        *slot = Some(CachedViewportPlanes {
            dims,
            mode,
            plane0,
            plane1,
            plane2,
        });
    }

    let cached = slot.as_ref().ok_or_else(|| Error::MetalKernel {
        message: "JPEG Metal viewport plane cache is missing after refresh".to_string(),
    })?;
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

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;

    #[test]
    fn viewport_cache_helpers_surface_invalid_state_without_panicking() {
        let too_wide = usize::try_from(u64::from(u32::MAX) + 1).expect("macOS usize is 64-bit");
        assert!(matches!(
            checked_output_stride(too_wide, "test stride"),
            Err(Error::MetalKernel { message })
                if message.contains("test stride") && message.contains("u32 stride ABI")
        ));
        assert!(matches!(
            checked_pack_output_format(PixelFormat::Gray16),
            Err(Error::MetalKernel { message })
                if message.contains("unsupported JPEG Metal viewport pack format")
        ));
        let missing: Option<&Buffer> = None;
        assert!(matches!(
            required_plane(missing, "Cb"),
            Err(Error::MetalKernel { message }) if message.contains("Cb plane is missing")
        ));
    }
}
