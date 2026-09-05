// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::metal_types::prelude::*;

use super::irreversible::dispatch_irreversible97_stages;
use super::{
    dispatch_3d_pipeline, label_compute_encoder, new_compute_command_encoder, CommandBufferRef,
    ComputeCommandEncoderRef, Error, J2kIdwtSingleDecompositionParams,
    J2kRepeatedIdwtSingleDecompositionParams, RepeatedIdwtDispatch,
};

pub(in crate::engine) fn dispatch_irreversible97_repeated_buffers_in_command_buffer_with_offsets(
    command_buffer: &CommandBufferRef,
    dispatch: RepeatedIdwtDispatch<'_>,
) -> Result<(), Error> {
    let encoder = new_compute_command_encoder(command_buffer)?;
    label_compute_encoder(&encoder, "J2K decode batched irreversible97 IDWT");
    dispatch_irreversible97_repeated_buffers_in_encoder_with_offsets(&encoder, dispatch)?;
    encoder.endEncoding();
    Ok(())
}

pub(in crate::engine) fn dispatch_irreversible97_repeated_buffers_in_encoder_with_offsets(
    encoder: &ComputeCommandEncoderRef,
    dispatch: RepeatedIdwtDispatch<'_>,
) -> Result<(), Error> {
    let params = dispatch.params;
    let plane_bytes = usize::try_from(params.width)
        .ok()
        .and_then(|width| width.checked_mul(params.height as usize))
        .and_then(|samples| samples.checked_mul(size_of::<f32>()))
        .filter(|bytes| *bytes != 0)
        .ok_or_else(|| chunk_offset_error())?;
    let total_bytes = plane_bytes
        .checked_mul(params.batch_count as usize)
        .ok_or_else(|| chunk_offset_error())?;
    let chunk_count = if total_bytes > 20 * 1024 * 1024 {
        (16 * 1024 * 1024 / plane_bytes).max(1)
    } else {
        (params.batch_count as usize).max(1)
    };
    for start in (0..params.batch_count as usize).step_by(chunk_count) {
        let mut chunk = dispatch;
        chunk.params.batch_count =
            u32::try_from(chunk_count.min(params.batch_count as usize - start))
                .map_err(|_| chunk_offset_error())?;
        for (offset, stride) in [
            (&mut chunk.sub_bands.ll_offset, params.ll_instance_stride),
            (&mut chunk.sub_bands.hl_offset, params.hl_instance_stride),
            (&mut chunk.sub_bands.lh_offset, params.lh_instance_stride),
            (&mut chunk.sub_bands.hh_offset, params.hh_instance_stride),
        ] {
            *offset = start
                .checked_mul(stride as usize)
                .and_then(|samples| samples.checked_mul(size_of::<f32>()))
                .and_then(|bytes| bytes.checked_add(*offset))
                .ok_or_else(|| chunk_offset_error())?;
        }
        let decoded_offset = start
            .checked_mul(plane_bytes)
            .ok_or_else(|| chunk_offset_error())?;
        dispatch_chunk(encoder, chunk, decoded_offset);
    }
    Ok(())
}

fn chunk_offset_error() -> Error {
    Error::MetalKernel {
        message: "J2K Metal batched IDWT chunk dimensions or offsets overflow".to_owned(),
    }
}

fn dispatch_chunk(
    encoder: &ComputeCommandEncoderRef,
    dispatch: RepeatedIdwtDispatch<'_>,
    decoded_offset: usize,
) {
    let RepeatedIdwtDispatch {
        kernels,
        sub_bands,
        params,
        decoded,
    } = dispatch;
    encoder.setComputePipelineState(&kernels.idwt_interleave_batched);
    for (index, buffer, offset) in [
        (0, sub_bands.ll, sub_bands.ll_offset),
        (1, sub_bands.hl, sub_bands.hl_offset),
        (2, sub_bands.lh, sub_bands.lh_offset),
        (3, sub_bands.hh, sub_bands.hh_offset),
    ] {
        encoder.set_buffer(index, Some(buffer), offset as u64);
    }
    encoder.set_buffer(4, Some(decoded), decoded_offset as u64);
    encoder.set_bytes::<J2kRepeatedIdwtSingleDecompositionParams>(5, &params);
    dispatch_3d_pipeline(
        encoder,
        &kernels.idwt_interleave_batched,
        (params.width, params.height, params.batch_count),
    );
    encoder.memory_barrier_with_resources(&[decoded]);

    // The stacked-plan preflight guarantees identical geometry and origin
    // parity. Only the plane offset varies along the third grid dimension.
    dispatch_irreversible97_stages(
        encoder,
        kernels,
        decoded,
        decoded_offset,
        single_params(params),
        j2k_codec_math::dwt::IDWT97_OPENJPEG_TWO_INV_KAPPA_F32 * 0.5,
        params.batch_count,
    );
}

fn single_params(
    params: J2kRepeatedIdwtSingleDecompositionParams,
) -> J2kIdwtSingleDecompositionParams {
    J2kIdwtSingleDecompositionParams {
        x0: params.x0,
        y0: params.y0,
        output_x: params.output_x,
        output_y: params.output_y,
        width: params.width,
        height: params.height,
        ll_x: params.ll_x,
        ll_y: params.ll_y,
        ll_width: params.ll_width,
        ll_height: params.ll_height,
        hl_x: params.hl_x,
        hl_y: params.hl_y,
        hl_width: params.hl_width,
        hl_height: params.hl_height,
        lh_x: params.lh_x,
        lh_y: params.lh_y,
        lh_width: params.lh_width,
        lh_height: params.lh_height,
        hh_x: params.hh_x,
        hh_y: params.hh_y,
        hh_width: params.hh_width,
        hh_height: params.hh_height,
    }
}
