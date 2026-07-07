// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{Buffer, Rect, MetalRuntime, ComputeCommandEncoderRef, PixelFormat, Surface, Error, CommandBufferRef, checked_metal_surface_len, checked_metal_buffer_len_u64, MTLResourceOptions, NativeColorSpace, j2k_pack_scale_arrays, J2kPackParams, j2k_u32_param, size_of, dispatch_2d_pipeline, j2k_scalar_pack_params, J2kRepeatedGrayPackParams, dispatch_3d_pipeline, ComputePipelineState};

#[cfg(target_os = "macos")]
pub(super) fn copy_plane_samples(
    buffer: &mut Buffer,
    samples: &[f32],
    image_width: usize,
    roi: Rect,
) -> Result<(), Error> {
    let row_width = roi.w as usize;
    let row_count = roi.h as usize;
    let dst_len = row_width.checked_mul(row_count).ok_or_else(|| Error::MetalKernel {
        message: "J2K MetalDirect plane upload sample count overflow".to_string(),
    })?;
    let dst = j2k_metal_support::checked_buffer_contents_slice_mut::<f32>(buffer, 0, dst_len)
        .map_err(|error| Error::MetalKernel {
            message: format!("J2K MetalDirect plane upload buffer view invalid: {error}"),
        })?;

    for row in 0..row_count {
        let src_y = roi.y as usize + row;
        let src_start = src_y
            .checked_mul(image_width)
            .and_then(|offset| offset.checked_add(roi.x as usize))
            .ok_or_else(|| Error::MetalKernel {
                message: "J2K MetalDirect plane upload source offset overflow".to_string(),
            })?;
        let src_end = src_start
            .checked_add(row_width)
            .ok_or_else(|| Error::MetalKernel {
                message: "J2K MetalDirect plane upload source range overflow".to_string(),
            })?;
        if src_end > samples.len() {
            return Err(Error::MetalKernel {
                message: "J2K MetalDirect plane upload source range exceeds plane".to_string(),
            });
        }
        let dst_start = row * row_width;
        dst[dst_start..dst_start + row_width].copy_from_slice(&samples[src_start..src_end]);
    }

    Ok(())
}

#[cfg(target_os = "macos")]
pub(super) fn encode_gray_plane_to_surface_in_encoder(
    runtime: &MetalRuntime,
    encoder: &ComputeCommandEncoderRef,
    plane: &Buffer,
    dims: (u32, u32),
    bit_depth: u8,
    fmt: PixelFormat,
) -> Result<Surface, Error> {
    encode_gray_plane_to_surface_in_encoder_with_offset(
        runtime, encoder, plane, 0, dims, bit_depth, fmt,
    )
}

#[cfg(target_os = "macos")]
pub(super) fn encode_gray_plane_to_surface_in_command_buffer_with_offset(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    plane: &Buffer,
    plane_offset_bytes: usize,
    dims: (u32, u32),
    bit_depth: u8,
    fmt: PixelFormat,
) -> Result<Surface, Error> {
    let encoder = command_buffer.new_compute_command_encoder();
    let result = encode_gray_plane_to_surface_in_encoder_with_offset(
        runtime,
        encoder,
        plane,
        plane_offset_bytes,
        dims,
        bit_depth,
        fmt,
    );
    encoder.end_encoding();
    result
}

#[cfg(target_os = "macos")]
pub(super) fn encode_gray_plane_to_surface_in_encoder_with_offset(
    runtime: &MetalRuntime,
    encoder: &ComputeCommandEncoderRef,
    plane: &Buffer,
    plane_offset_bytes: usize,
    dims: (u32, u32),
    bit_depth: u8,
    fmt: PixelFormat,
) -> Result<Surface, Error> {
    let (pitch_bytes, surface_bytes) = checked_metal_surface_len(
        dims,
        fmt.bytes_per_pixel(),
        "J2K Metal repeated grayscale output size overflow",
    )?;
    let out_buffer = runtime.device.new_buffer(
        checked_metal_buffer_len_u64(
            surface_bytes,
            "J2K Metal repeated grayscale output size exceeds u64",
        )?,
        MTLResourceOptions::StorageModeShared,
    );
    let (output_channels, opaque_alpha, pipeline) =
        output_shape_for(&NativeColorSpace::Gray, false, 1, fmt, runtime)?;
    let mut bit_depths = [0u32; 4];
    bit_depths[0] = u32::from(bit_depth);
    let (max_values, u8_scales, u16_scales) = j2k_pack_scale_arrays(bit_depths);
    let params = J2kPackParams {
        width: dims.0,
        height: dims.1,
        out_stride: j2k_u32_param(pitch_bytes, "J2K Metal output stride exceeds u32")?,
        output_channels,
        opaque_alpha,
        max_values,
        u8_scales,
        u16_scales,
    };

    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(plane), plane_offset_bytes as u64);
    encoder.set_buffer(1, None, 0);
    encoder.set_buffer(2, None, 0);
    encoder.set_buffer(3, None, 0);
    encoder.set_buffer(4, Some(&out_buffer), 0);
    encoder.set_bytes(
        5,
        size_of::<J2kPackParams>() as u64,
        (&raw const params).cast(),
    );
    dispatch_2d_pipeline(encoder, pipeline, dims);

    Ok(Surface::from_metal_buffer(out_buffer, dims, fmt))
}

#[cfg(target_os = "macos")]
pub(super) fn encode_repeated_gray_plane_to_surfaces_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    plane: &Buffer,
    dims: (u32, u32),
    bit_depth: u8,
    fmt: PixelFormat,
    count: usize,
) -> Result<Vec<Surface>, Error> {
    let count_u32 = u32::try_from(count).map_err(|_| Error::MetalKernel {
        message: "J2K Metal repeated grayscale surface count exceeds u32".to_string(),
    })?;
    let (pitch_bytes, surface_bytes) = checked_metal_surface_len(
        dims,
        fmt.bytes_per_pixel(),
        "J2K Metal repeated grayscale surface size overflow",
    )?;
    let total_bytes = surface_bytes
        .checked_mul(count)
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K Metal repeated grayscale output size overflow".to_string(),
        })?;
    let output_len = checked_metal_buffer_len_u64(
        total_bytes,
        "J2K Metal repeated grayscale output size exceeds u64",
    )?;
    let out_buffer = runtime
        .device
        .new_buffer(output_len, MTLResourceOptions::StorageModeShared);
    let scale = j2k_scalar_pack_params(u32::from(bit_depth));
    let params = J2kRepeatedGrayPackParams {
        width: dims.0,
        height: dims.1,
        out_stride: j2k_u32_param(pitch_bytes, "J2K Metal output stride exceeds u32")?,
        batch_count: count_u32,
        max_value: scale.max_value,
        u8_scale: scale.u8_scale,
        u16_scale: scale.u16_scale,
    };
    let pipeline = match fmt {
        PixelFormat::Gray8 => &runtime.pack_u8_repeated_gray,
        PixelFormat::Gray16 => &runtime.pack_u16_repeated_gray,
        _ => {
            return Err(Error::MetalKernel {
                message: format!("J2K Metal repeated grayscale pack does not support {fmt:?}"),
            })
        }
    };

    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(plane), 0);
    encoder.set_buffer(1, Some(&out_buffer), 0);
    encoder.set_bytes(
        2,
        size_of::<J2kRepeatedGrayPackParams>() as u64,
        (&raw const params).cast(),
    );
    dispatch_3d_pipeline(encoder, pipeline, (dims.0, dims.1, count_u32));
    encoder.end_encoding();

    let mut surfaces = Vec::with_capacity(count);
    for instance_idx in 0..count {
        surfaces.push(Surface::from_metal_buffer_with_offset(
            out_buffer.clone(),
            dims,
            fmt,
            instance_idx * surface_bytes,
        ));
    }
    Ok(surfaces)
}

#[cfg(target_os = "macos")]
pub(super) fn j2k_pack_kernel_name_for(
    color_space: &NativeColorSpace,
    has_alpha: bool,
    plane_count: usize,
    fmt: PixelFormat,
) -> Option<&'static str> {
    match (color_space, has_alpha, plane_count, fmt) {
        (NativeColorSpace::Gray, false, 1, PixelFormat::Gray8) => Some("j2k_pack_gray8"),
        (NativeColorSpace::RGB, false, 3, PixelFormat::Rgb8)
        | (NativeColorSpace::RGB, true, 4, PixelFormat::Rgb8) => Some("j2k_pack_rgb8"),
        (NativeColorSpace::RGB, false, 3, PixelFormat::Rgba8) => Some("j2k_pack_rgb_opaque_rgba8"),
        (NativeColorSpace::RGB, true, 4, PixelFormat::Rgba8) => Some("j2k_pack_rgba8"),
        (NativeColorSpace::Gray, false, 1, PixelFormat::Gray16) => Some("j2k_pack_gray16"),
        (NativeColorSpace::RGB, false, 3, PixelFormat::Rgb16) => Some("j2k_pack_rgb16"),
        _ => None,
    }
}

#[cfg(target_os = "macos")]
pub(super) fn j2k_pack_pipeline_for<'a>(
    runtime: &'a MetalRuntime,
    kernel_name: &str,
) -> Result<&'a ComputePipelineState, Error> {
    let pipeline = match kernel_name {
        "j2k_pack_gray8" => &runtime.pack_gray8,
        "j2k_pack_rgb8" => &runtime.pack_rgb8,
        "j2k_pack_rgb_opaque_rgba8" => &runtime.pack_rgb_opaque_rgba8,
        "j2k_pack_rgba8" => &runtime.pack_rgba8,
        "j2k_pack_gray16" => &runtime.pack_gray16,
        "j2k_pack_rgb16" => &runtime.pack_rgb16,
        _ => {
            return Err(Error::MetalKernel {
                message: format!("unsupported validated J2K Metal pack kernel `{kernel_name}`"),
            });
        }
    };
    Ok(pipeline)
}

#[cfg(target_os = "macos")]
pub(super) fn output_shape_for<'a>(
    color_space: &NativeColorSpace,
    has_alpha: bool,
    plane_count: usize,
    fmt: PixelFormat,
    runtime: &'a MetalRuntime,
) -> Result<(u32, u32, &'a ComputePipelineState), Error> {
    let Some(kernel_name) = j2k_pack_kernel_name_for(color_space, has_alpha, plane_count, fmt)
    else {
        return Err(Error::MetalKernel {
            message: format!(
                "unsupported J2K Metal mapping for {color_space:?}, alpha={has_alpha}, planes={plane_count}, fmt={fmt:?}"
            ),
        });
    };
    let (output_channels, opaque_alpha) = match (color_space, has_alpha, plane_count, fmt) {
        (NativeColorSpace::Gray, false, 1, PixelFormat::Gray8 | PixelFormat::Gray16) => (1, 0),
        (NativeColorSpace::RGB, false, 3, PixelFormat::Rgb8 | PixelFormat::Rgb16)
        | (NativeColorSpace::RGB, true, 4, PixelFormat::Rgb8) => (3, 0),
        (NativeColorSpace::RGB, false, 3, PixelFormat::Rgba8) => (4, 1),
        (NativeColorSpace::RGB, true, 4, PixelFormat::Rgba8) => (4, 0),
        _ => {
            return Err(Error::MetalKernel {
                message: format!(
                    "unsupported validated J2K Metal pack shape for {color_space:?}, alpha={has_alpha}, planes={plane_count}, fmt={fmt:?}"
                ),
            });
        }
    };
    Ok((
        output_channels,
        opaque_alpha,
        j2k_pack_pipeline_for(runtime, kernel_name)?,
    ))
}
