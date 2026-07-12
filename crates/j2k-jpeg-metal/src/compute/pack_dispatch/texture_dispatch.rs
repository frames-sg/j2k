// SPDX-License-Identifier: MIT OR Apache-2.0

use std::mem::size_of;

use super::super::{
    dispatch_2d_pipeline, new_compute_command_encoder, Buffer, CommandBufferRef,
    ComputePipelineState, Error, JpegTexturePackBatchParams, JpegWindowedTexturePackBatchParams,
};
use super::conversion::checked_u32;

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
