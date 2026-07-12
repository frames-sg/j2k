// SPDX-License-Identifier: MIT OR Apache-2.0

use std::mem::size_of;

use crate::buffers::new_shared_buffer;

use super::super::{
    dispatch_2d_pipeline, new_compute_command_encoder, Buffer, CommandBufferRef, Error,
    JpegPackParams, MetalRuntime, PixelFormat, PlaneMode, Surface, MODE_GRAY, MODE_RGB, MODE_YCBCR,
    OUT_GRAY, OUT_RGB, OUT_RGBA,
};

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
pub(in crate::compute) struct JpegPackSurfaceRequest<'a> {
    pub(in crate::compute) plane0: &'a Buffer,
    pub(in crate::compute) plane1: Option<&'a Buffer>,
    pub(in crate::compute) plane2: Option<&'a Buffer>,
    pub(in crate::compute) dims: (u32, u32),
    pub(in crate::compute) mode: PlaneMode,
    pub(in crate::compute) fmt: PixelFormat,
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn encode_jpeg_pack_to_surface_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    request: JpegPackSurfaceRequest<'_>,
) -> Result<Surface, Error> {
    let JpegPackSurfaceRequest {
        plane0,
        plane1,
        plane2,
        dims,
        mode,
        fmt,
    } = request;
    match (mode, fmt) {
        (PlaneMode::Gray | PlaneMode::YCbCr, PixelFormat::Gray8) => {
            return Ok(Surface::from_metal_buffer(plane0.clone(), dims, fmt));
        }
        (
            PlaneMode::Gray | PlaneMode::YCbCr | PlaneMode::Rgb,
            PixelFormat::Rgb8 | PixelFormat::Rgba8,
        )
        | (PlaneMode::Rgb, PixelFormat::Gray8) => {}
        _ => {
            return Err(Error::MetalKernel {
                message: format!("unsupported JPEG Metal pixel format {fmt:?}"),
            });
        }
    }

    let pitch_bytes = dims.0 as usize * fmt.bytes_per_pixel();
    let out_len = crate::batch_allocation::checked_count_product(
        pitch_bytes,
        dims.1 as usize,
        "JPEG Metal packed surface output bytes",
    )?;
    let out_buffer = new_shared_buffer(&runtime.device, out_len)?;
    let params = JpegPackParams {
        width: dims.0,
        height: dims.1,
        out_stride: u32::try_from(pitch_bytes).expect("JPEG Metal output stride fits in u32"),
        alpha: u32::from(u8::MAX),
        mode: match mode {
            PlaneMode::Gray => MODE_GRAY,
            PlaneMode::YCbCr => MODE_YCBCR,
            PlaneMode::Rgb => MODE_RGB,
        },
        out_format: match fmt {
            PixelFormat::Gray8 => OUT_GRAY,
            PixelFormat::Rgb8 => OUT_RGB,
            PixelFormat::Rgba8 => OUT_RGBA,
            _ => unreachable!("validated by caller"),
        },
    };

    let encoder = new_compute_command_encoder(command_buffer)?;
    encoder.set_compute_pipeline_state(&runtime.pack_pipeline);
    encoder.set_buffer(0, Some(plane0), 0);
    encoder.set_buffer(1, plane1.map(std::convert::AsRef::as_ref), 0);
    encoder.set_buffer(2, plane2.map(std::convert::AsRef::as_ref), 0);
    encoder.set_buffer(3, Some(&out_buffer), 0);
    encoder.set_bytes(
        4,
        size_of::<JpegPackParams>() as u64,
        (&raw const params).cast(),
    );
    dispatch_2d_pipeline(&encoder, &runtime.pack_pipeline, dims);
    encoder.end_encoding();

    Ok(Surface::from_metal_buffer(out_buffer, dims, fmt))
}
