// SPDX-License-Identifier: MIT OR Apache-2.0

use std::mem::size_of;

use super::super::{
    batch, commit_and_wait_jpeg, dispatch_2d_pipeline, new_command_buffer,
    new_compute_command_encoder, Buffer, BufferError, Error, JpegRgb8ToRgbaTextureParams,
    MTLPixelFormat, MetalRuntime, PixelFormat, Surface,
};

#[cfg(target_os = "macos")]
pub(in crate::compute) type GroupedTextureResult = (usize, Result<crate::MetalTextureTile, Error>);

#[cfg(target_os = "macos")]
type Rgb8TextureCopyRecord = (usize, Buffer, usize);

#[cfg(target_os = "macos")]
type Rgb8TextureCopyPlan = (Vec<Rgb8TextureCopyRecord>, Vec<GroupedTextureResult>);

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
