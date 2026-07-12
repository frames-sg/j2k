// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::buffers::new_shared_buffer;

use super::super::{
    commit_and_wait_jpeg, new_blit_command_encoder, new_command_buffer, Buffer, BufferError, Error,
    MetalRuntime, PixelFormat, Surface,
};

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
