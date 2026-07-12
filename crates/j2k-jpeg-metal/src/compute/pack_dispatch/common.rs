// SPDX-License-Identifier: MIT OR Apache-2.0

use std::mem::size_of;

use crate::buffers::new_shared_buffer;

use super::super::{
    batch, commit_and_wait_jpeg, dispatch_2d_pipeline, new_blit_command_encoder,
    new_command_buffer, new_compute_command_encoder, BatchDeviceBufferCache, Buffer, BufferError,
    CommandBufferRef, ComputePipelineState, Error, FastSubsampledMetal, JpegFast444PacketV1,
    JpegPackParams, JpegRgb8ToRgbaTextureParams, JpegTexturePackBatchParams,
    JpegWindowedTexturePackBatchParams, MTLPixelFormat, MetalRuntime, PixelFormat, PlaneMode, Rect,
    Surface, MODE_GRAY, MODE_RGB, MODE_YCBCR, OUT_GRAY, OUT_RGB, OUT_RGBA,
};
#[cfg(test)]
use super::super::{
    dispatch_1d_pipeline, dispatch_3d_pipeline, JpegFast420BatchParams, PreparedHuffmanHost,
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
pub(in crate::compute) struct FastSubsampledScaledRegionBatchItemRequest<'a, P: FastSubsampledMetal>
{
    pub(in crate::compute) runtime: &'a MetalRuntime,
    pub(in crate::compute) command_buffer: &'a CommandBufferRef,
    pub(in crate::compute) device_buffer_cache: &'a mut BatchDeviceBufferCache,
    pub(in crate::compute) packet: &'a P,
    pub(in crate::compute) fmt: PixelFormat,
    pub(in crate::compute) roi: Rect,
    pub(in crate::compute) scale: j2k_core::Downscale,
}

#[cfg(target_os = "macos")]
pub(in crate::compute) struct FastSubsampledOpBatchItemRequest<'a, P: FastSubsampledMetal> {
    pub(in crate::compute) runtime: &'a MetalRuntime,
    pub(in crate::compute) command_buffer: &'a CommandBufferRef,
    pub(in crate::compute) device_buffer_cache: &'a mut BatchDeviceBufferCache,
    pub(in crate::compute) packet: &'a P,
    pub(in crate::compute) fmt: PixelFormat,
    pub(in crate::compute) op: batch::BatchOp,
}

#[cfg(target_os = "macos")]
pub(in crate::compute) struct Fast444ScaledRegionBatchItemRequest<'a> {
    pub(in crate::compute) runtime: &'a MetalRuntime,
    pub(in crate::compute) command_buffer: &'a CommandBufferRef,
    pub(in crate::compute) device_buffer_cache: &'a mut BatchDeviceBufferCache,
    pub(in crate::compute) packet: &'a JpegFast444PacketV1,
    pub(in crate::compute) mode: PlaneMode,
    pub(in crate::compute) fmt: PixelFormat,
    pub(in crate::compute) roi: Rect,
    pub(in crate::compute) scale: j2k_core::Downscale,
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

#[cfg(target_os = "macos")]
pub(in crate::compute) fn checked_u32(value: usize, label: &str) -> Result<u32, Error> {
    u32::try_from(value).map_err(|_| Error::MetalKernel {
        message: format!("JPEG Metal {label} does not fit in u32"),
    })
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn batch_output_buffer_or_new(
    runtime: &MetalRuntime,
    output: Option<&crate::MetalBatchOutputBuffer>,
    dimensions: (u32, u32),
    tile_count: usize,
    out_stride: usize,
    out_tile_len: usize,
) -> Result<Buffer, Error> {
    let Some(output) = output else {
        let byte_len = out_tile_len
            .checked_mul(tile_count)
            .ok_or(BufferError::SizeOverflow {
                what: "JPEG Metal batch output bytes",
            })?;
        return new_shared_buffer(&runtime.device, byte_len);
    };

    if output.dimensions() != dimensions
        || output.pixel_format() != PixelFormat::Rgb8
        || output.pitch_bytes() != out_stride
        || output.tile_stride_bytes() < out_tile_len
    {
        return Err(Error::UnsupportedMetalRequest {
            reason: "JPEG Metal batch output buffer shape does not match requested RGB8 tiles",
        });
    }
    if output.tile_capacity() < tile_count {
        return Err(BufferError::OutputTooSmall {
            required: output.tile_stride_bytes().checked_mul(tile_count).ok_or(
                BufferError::SizeOverflow {
                    what: "JPEG Metal batch output bytes",
                },
            )?,
            have: output.byte_len(),
        }
        .into());
    }

    Ok(output.clone_buffer())
}

#[cfg(target_os = "macos")]
pub(in crate::compute) type GroupedSurfaceResult = (usize, Result<Surface, Error>);

#[cfg(target_os = "macos")]
pub(in crate::compute) type GroupedTextureResult = (usize, Result<crate::MetalTextureTile, Error>);

#[cfg(target_os = "macos")]
type Rgb8TextureCopyRecord = (usize, Buffer, usize);

#[cfg(target_os = "macos")]
type Rgb8TextureCopyPlan = (Vec<Rgb8TextureCopyRecord>, Vec<GroupedTextureResult>);

#[cfg(target_os = "macos")]
pub(in crate::compute) fn copy_grouped_surfaces_to_output(
    runtime: &MetalRuntime,
    output: &crate::MetalBatchOutputBuffer,
    dimensions: (u32, u32),
    out_tile_len: usize,
    group_indices: &[usize],
    group_results: Vec<Result<Surface, Error>>,
    external_live_bytes: usize,
) -> Result<Vec<GroupedSurfaceResult>, Error> {
    if group_results.len() != group_indices.len() {
        return Err(Error::MetalKernel {
            message: "JPEG Metal grouped buffer result count mismatch".to_string(),
        });
    }

    let output_buffer = output.clone_buffer();
    let mut budget = crate::batch_allocation::BatchMetadataBudget::with_external_live(
        "JPEG Metal grouped surface output copy",
        external_live_bytes,
    );
    budget.preflight(&[
        crate::batch_allocation::BatchMetadataRequest::of::<(Buffer, usize, usize)>(
            group_indices.len(),
        ),
        crate::batch_allocation::BatchMetadataRequest::of::<GroupedSurfaceResult>(
            group_indices.len(),
        ),
    ])?;
    let mut copies = budget.try_vec(
        group_indices.len(),
        "JPEG Metal grouped surface copy records",
    )?;
    let mut mapped_results =
        budget.try_vec(group_indices.len(), "JPEG Metal grouped surface results")?;
    for (original_index, result) in group_indices.iter().copied().zip(group_results) {
        match result {
            Ok(surface) => {
                let (source, source_offset) =
                    surface
                        .metal_buffer_trusted()
                        .ok_or_else(|| Error::MetalKernel {
                            message: "JPEG Metal grouped buffer source was not Metal-backed"
                                .to_string(),
                        })?;
                let destination_offset = original_index
                    .checked_mul(output.tile_stride_bytes())
                    .ok_or_else(|| Error::MetalKernel {
                        message: "JPEG Metal grouped buffer destination offset overflowed"
                            .to_string(),
                    })?;
                copies.push((source.clone(), source_offset, destination_offset));
                mapped_results.push((
                    original_index,
                    Ok(Surface::from_batch_output_buffer_offset(
                        output,
                        dimensions,
                        PixelFormat::Rgb8,
                        destination_offset,
                    )),
                ));
            }
            Err(error) => mapped_results.push((original_index, Err(error))),
        }
    }

    if !copies.is_empty() {
        let command_buffer = new_command_buffer(&runtime.queue)?;
        let blit = new_blit_command_encoder(&command_buffer)?;
        for (source, source_offset, destination_offset) in copies {
            blit.copy_from_buffer(
                &source,
                u64::try_from(source_offset).map_err(|_| Error::MetalKernel {
                    message: "JPEG Metal grouped buffer source offset exceeds u64".to_string(),
                })?,
                &output_buffer,
                u64::try_from(destination_offset).map_err(|_| Error::MetalKernel {
                    message: "JPEG Metal grouped buffer destination offset exceeds u64".to_string(),
                })?,
                u64::try_from(out_tile_len).map_err(|_| Error::MetalKernel {
                    message: "JPEG Metal grouped buffer copy size exceeds u64".to_string(),
                })?,
            );
        }
        blit.end_encoding();
        commit_and_wait_jpeg(&command_buffer)?;
    }

    Ok(mapped_results)
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn validate_rgba_texture_batch_output(
    output: &crate::MetalBatchTextureOutput,
    dimensions: (u32, u32),
    tile_count: usize,
    out_tile_len: usize,
) -> Result<(), Error> {
    if output.dimensions() != dimensions
        || output.pixel_format() != PixelFormat::Rgba8
        || output.metal_pixel_format() != MTLPixelFormat::RGBA8Unorm
    {
        return Err(Error::UnsupportedMetalRequest {
            reason: "JPEG Metal batch texture output shape does not match requested RGBA8 tiles",
        });
    }
    if output.tile_capacity() < tile_count {
        return Err(BufferError::OutputTooSmall {
            required: out_tile_len
                .checked_mul(tile_count)
                .ok_or(BufferError::SizeOverflow {
                    what: "JPEG Metal batch texture output bytes",
                })?,
            have: out_tile_len.checked_mul(output.tile_capacity()).ok_or(
                BufferError::SizeOverflow {
                    what: "JPEG Metal batch texture output bytes",
                },
            )?,
        }
        .into());
    }

    for index in 0..tile_count {
        let Some(texture) = output.texture_trusted(index) else {
            return Err(Error::MetalKernel {
                message: "JPEG Metal batch texture output slot was missing".to_string(),
            });
        };
        if texture.width() != u64::from(dimensions.0)
            || texture.height() != u64::from(dimensions.1)
            || texture.pixel_format() != MTLPixelFormat::RGBA8Unorm
        {
            return Err(Error::UnsupportedMetalRequest {
                reason:
                    "JPEG Metal batch texture output texture does not match requested RGBA8 tiles",
            });
        }
    }

    Ok(())
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn texture_batch_success_results(
    requests: &[batch::QueuedRequest],
    output: &crate::MetalBatchTextureOutput,
    dimensions: (u32, u32),
    tile_count: usize,
) -> Result<Vec<Result<crate::MetalTextureTile, Error>>, Error> {
    let mut budget = crate::plan_owner_ledger::batch_execution_budget(
        "JPEG Metal texture batch success results",
        requests,
    )?;
    let mut results =
        budget.try_vec(tile_count, "JPEG Metal texture batch success result slots")?;
    for index in 0..tile_count {
        let texture = output
            .clone_texture_trusted(index)
            .ok_or_else(|| Error::MetalKernel {
                message: "JPEG Metal batch texture output slot was missing".to_string(),
            })?;
        results.push(Ok(crate::MetalTextureTile::new(
            texture,
            output.clone_access_gate(),
            dimensions,
            PixelFormat::Rgba8,
        )));
    }
    Ok(results)
}

#[cfg(target_os = "macos")]
fn collect_rgb8_texture_copy_results(
    output: &crate::MetalBatchTextureOutput,
    dimensions: (u32, u32),
    group_indices: &[usize],
    group_results: Vec<Result<Surface, Error>>,
    external_live_bytes: usize,
) -> Result<Rgb8TextureCopyPlan, Error> {
    let mut budget = crate::batch_allocation::BatchMetadataBudget::with_external_live(
        "JPEG Metal grouped texture output copy",
        external_live_bytes,
    );
    budget.preflight(&[
        crate::batch_allocation::BatchMetadataRequest::of::<Rgb8TextureCopyRecord>(
            group_indices.len(),
        ),
        crate::batch_allocation::BatchMetadataRequest::of::<GroupedTextureResult>(
            group_indices.len(),
        ),
    ])?;
    let mut copies = budget.try_vec(
        group_indices.len(),
        "JPEG Metal grouped texture copy records",
    )?;
    let mut mapped_results =
        budget.try_vec(group_indices.len(), "JPEG Metal grouped texture results")?;
    for (original_index, result) in group_indices.iter().copied().zip(group_results) {
        match result {
            Ok(surface) => {
                if surface.dimensions != dimensions || surface.fmt != PixelFormat::Rgb8 {
                    return Err(Error::MetalKernel {
                        message: "JPEG Metal texture copy source shape mismatch".to_string(),
                    });
                }
                let (source, source_offset) =
                    surface
                        .metal_buffer_trusted()
                        .ok_or_else(|| Error::MetalKernel {
                            message: "JPEG Metal texture copy source was not Metal-backed"
                                .to_string(),
                        })?;
                let texture = output
                    .clone_texture_trusted(original_index)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "JPEG Metal batch texture output slot was missing".to_string(),
                    })?;
                copies.push((original_index, source.clone(), source_offset));
                mapped_results.push((
                    original_index,
                    Ok(crate::MetalTextureTile::new(
                        texture,
                        output.clone_access_gate(),
                        dimensions,
                        PixelFormat::Rgba8,
                    )),
                ));
            }
            Err(error) => mapped_results.push((original_index, Err(error))),
        }
    }
    Ok((copies, mapped_results))
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn copy_rgb8_surfaces_to_rgba_textures(
    runtime: &MetalRuntime,
    output: &crate::MetalBatchTextureOutput,
    dimensions: (u32, u32),
    tile_count: usize,
    group_indices: &[usize],
    group_results: Vec<Result<Surface, Error>>,
    external_live_bytes: usize,
) -> Result<Vec<GroupedTextureResult>, Error> {
    if group_results.len() != group_indices.len() {
        return Err(Error::MetalKernel {
            message: "JPEG Metal grouped texture result count mismatch".to_string(),
        });
    }
    let out_tile_len = dimensions
        .0
        .checked_mul(dimensions.1)
        .and_then(|pixels| {
            pixels.checked_mul(u32::try_from(PixelFormat::Rgba8.bytes_per_pixel()).ok()?)
        })
        .ok_or(BufferError::SizeOverflow {
            what: "JPEG Metal batch texture output bytes",
        })? as usize;
    validate_rgba_texture_batch_output(output, dimensions, tile_count, out_tile_len)?;

    let in_stride = dimensions
        .0
        .checked_mul(
            u32::try_from(PixelFormat::Rgb8.bytes_per_pixel()).map_err(|_| {
                BufferError::SizeOverflow {
                    what: "JPEG Metal RGB texture copy input stride",
                }
            })?,
        )
        .ok_or(BufferError::SizeOverflow {
            what: "JPEG Metal RGB texture copy input stride",
        })?;
    let params = JpegRgb8ToRgbaTextureParams {
        width: dimensions.0,
        height: dimensions.1,
        in_stride,
        alpha: u32::from(u8::MAX),
    };
    let (copies, mapped_results) = collect_rgb8_texture_copy_results(
        output,
        dimensions,
        group_indices,
        group_results,
        external_live_bytes,
    )?;

    if !copies.is_empty() {
        let command_buffer = new_command_buffer(&runtime.queue)?;
        let encoder = new_compute_command_encoder(&command_buffer)?;
        encoder.set_compute_pipeline_state(&runtime.rgb8_to_rgba_texture_pipeline);
        for (original_index, source, source_offset) in copies {
            let texture =
                output
                    .texture_trusted(original_index)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "JPEG Metal batch texture output slot was missing".to_string(),
                    })?;
            encoder.set_buffer(
                0,
                Some(&source),
                u64::try_from(source_offset).map_err(|_| Error::MetalKernel {
                    message: "JPEG Metal texture copy source offset exceeds u64".to_string(),
                })?,
            );
            encoder.set_bytes(
                1,
                size_of::<JpegRgb8ToRgbaTextureParams>() as u64,
                (&raw const params).cast(),
            );
            encoder.set_texture(0, Some(texture));
            dispatch_2d_pipeline(&encoder, &runtime.rgb8_to_rgba_texture_pipeline, dimensions);
        }
        encoder.end_encoding();
        commit_and_wait_jpeg(&command_buffer)?;
    }

    Ok(mapped_results)
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn dispatch_rgba_texture_pack(
    command_buffer: &CommandBufferRef,
    pipeline: &ComputePipelineState,
    planes: (&Buffer, &Buffer, &Buffer),
    output: &crate::MetalBatchTextureOutput,
    params: JpegTexturePackBatchParams,
    tile_count: usize,
    dispatch_dims: (u32, u32),
) -> Result<(), Error> {
    let pack_encoder = new_compute_command_encoder(command_buffer)?;
    pack_encoder.set_compute_pipeline_state(pipeline);
    pack_encoder.set_buffer(0, Some(planes.0), 0);
    pack_encoder.set_buffer(1, Some(planes.1), 0);
    pack_encoder.set_buffer(2, Some(planes.2), 0);
    for index in 0..tile_count {
        let texture = output
            .texture_trusted(index)
            .ok_or_else(|| Error::MetalKernel {
                message: "JPEG Metal batch texture output slot was missing".to_string(),
            })?;
        let mut params = params;
        params.tile_index = checked_u32(index, "texture batch tile index")?;
        pack_encoder.set_texture(0, Some(texture));
        pack_encoder.set_bytes(
            3,
            size_of::<JpegTexturePackBatchParams>() as u64,
            (&raw const params).cast(),
        );
        dispatch_2d_pipeline(&pack_encoder, pipeline, dispatch_dims);
    }
    pack_encoder.end_encoding();
    Ok(())
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn dispatch_windowed_rgba_texture_pack(
    command_buffer: &CommandBufferRef,
    pipeline: &ComputePipelineState,
    planes: (&Buffer, &Buffer, &Buffer),
    output: &crate::MetalBatchTextureOutput,
    params: JpegWindowedTexturePackBatchParams,
    tile_count: usize,
    dispatch_dims: (u32, u32),
) -> Result<(), Error> {
    let pack_encoder = new_compute_command_encoder(command_buffer)?;
    pack_encoder.set_compute_pipeline_state(pipeline);
    pack_encoder.set_buffer(0, Some(planes.0), 0);
    pack_encoder.set_buffer(1, Some(planes.1), 0);
    pack_encoder.set_buffer(2, Some(planes.2), 0);
    for index in 0..tile_count {
        let texture = output
            .texture_trusted(index)
            .ok_or_else(|| Error::MetalKernel {
                message: "JPEG Metal batch texture output slot was missing".to_string(),
            })?;
        let mut params = params;
        params.tile_index = checked_u32(index, "windowed texture batch tile index")?;
        pack_encoder.set_texture(0, Some(texture));
        pack_encoder.set_bytes(
            3,
            size_of::<JpegWindowedTexturePackBatchParams>() as u64,
            (&raw const params).cast(),
        );
        dispatch_2d_pipeline(&pack_encoder, pipeline, dispatch_dims);
    }
    pack_encoder.end_encoding();
    Ok(())
}

/// Encode the split coeff-decode + IDCT-deposit passes shared by the surfaces
/// and texture drivers' `SplitCoeffIdct` debug mode.
#[cfg(all(target_os = "macos", test))]
#[derive(Clone, Copy)]
pub(in crate::compute) struct SplitCoeffIdctPasses<'a> {
    pub(in crate::compute) command_buffer: &'a CommandBufferRef,
    pub(in crate::compute) pipelines: (&'a ComputePipelineState, &'a ComputePipelineState),
    pub(in crate::compute) params: &'a JpegFast420BatchParams,
    pub(in crate::compute) quants: [&'a [u16; 64]; 3],
    pub(in crate::compute) dc_tables: &'a [PreparedHuffmanHost; 3],
    pub(in crate::compute) ac_tables: &'a [PreparedHuffmanHost; 3],
    pub(in crate::compute) entropy: (&'a Buffer, &'a Buffer, &'a Buffer, &'a Buffer),
    pub(in crate::compute) status_buffer: &'a Buffer,
    pub(in crate::compute) planes: [&'a Buffer; 3],
    pub(in crate::compute) scratch: (&'a Buffer, &'a Buffer),
    pub(in crate::compute) total_decode_threads: u32,
    pub(in crate::compute) idct_grid: (u32, u32, u32),
}

#[cfg(all(target_os = "macos", test))]
pub(in crate::compute) fn encode_split_coeff_idct_passes(
    request: SplitCoeffIdctPasses<'_>,
) -> Result<(), Error> {
    let SplitCoeffIdctPasses {
        command_buffer,
        pipelines,
        params,
        quants,
        dc_tables,
        ac_tables,
        entropy,
        status_buffer,
        planes,
        scratch,
        total_decode_threads,
        idct_grid,
    } = request;
    let (coeffs_pipeline, idct_pipeline) = pipelines;
    let (entropy_payload, entropy_offsets, entropy_lens, entropy_checkpoints) = entropy;
    let (coeff_blocks, dc_only_flags) = scratch;

    let coeff_encoder = new_compute_command_encoder(command_buffer)?;
    coeff_encoder.set_compute_pipeline_state(coeffs_pipeline);
    coeff_encoder.set_buffer(0, Some(entropy_payload), 0);
    coeff_encoder.set_buffer(1, Some(coeff_blocks), 0);
    coeff_encoder.set_buffer(2, Some(dc_only_flags), 0);
    coeff_encoder.set_bytes(
        4,
        size_of::<JpegFast420BatchParams>() as u64,
        (&raw const *params).cast(),
    );
    coeff_encoder.set_bytes(5, size_of::<[u16; 64]>() as u64, quants[0].as_ptr().cast());
    coeff_encoder.set_bytes(6, size_of::<[u16; 64]>() as u64, quants[1].as_ptr().cast());
    coeff_encoder.set_bytes(7, size_of::<[u16; 64]>() as u64, quants[2].as_ptr().cast());
    coeff_encoder.set_bytes(
        8,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const dc_tables[0]).cast(),
    );
    coeff_encoder.set_bytes(
        9,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const ac_tables[0]).cast(),
    );
    coeff_encoder.set_bytes(
        10,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const dc_tables[1]).cast(),
    );
    coeff_encoder.set_bytes(
        11,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const ac_tables[1]).cast(),
    );
    coeff_encoder.set_bytes(
        12,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const dc_tables[2]).cast(),
    );
    coeff_encoder.set_bytes(
        13,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const ac_tables[2]).cast(),
    );
    coeff_encoder.set_buffer(14, Some(entropy_offsets), 0);
    coeff_encoder.set_buffer(15, Some(entropy_lens), 0);
    coeff_encoder.set_buffer(16, Some(status_buffer), 0);
    coeff_encoder.set_buffer(17, Some(entropy_checkpoints), 0);
    dispatch_1d_pipeline(&coeff_encoder, coeffs_pipeline, total_decode_threads);
    coeff_encoder.end_encoding();

    let idct_encoder = new_compute_command_encoder(command_buffer)?;
    idct_encoder.set_compute_pipeline_state(idct_pipeline);
    idct_encoder.set_buffer(0, Some(coeff_blocks), 0);
    idct_encoder.set_buffer(1, Some(dc_only_flags), 0);
    idct_encoder.set_buffer(2, Some(planes[0]), 0);
    idct_encoder.set_buffer(3, Some(planes[1]), 0);
    idct_encoder.set_buffer(4, Some(planes[2]), 0);
    idct_encoder.set_bytes(
        5,
        size_of::<JpegFast420BatchParams>() as u64,
        (&raw const *params).cast(),
    );
    dispatch_3d_pipeline(&idct_encoder, idct_pipeline, idct_grid);
    idct_encoder.end_encoding();
    Ok(())
}
