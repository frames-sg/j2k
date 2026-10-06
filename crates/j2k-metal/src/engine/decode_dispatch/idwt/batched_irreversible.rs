// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::metal_types::prelude::*;

use super::irreversible::{
    dispatch_irreversible97_stages_after_horizontal_scale,
    dispatch_irreversible97_vertical_after_fused_horizontal, irreversible97_horizontal_lift_steps,
    supports_irreversible97_interleave_horizontal_fused, IDWT97_ROWS_PER_GROUP, IDWT97_ROW_THREADS,
    IDWT97_ROW_TILE,
};
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
    dispatch_irreversible97_repeated_buffers_in_encoder_with_offsets(&encoder, dispatch);
    encoder.endEncoding();
    Ok(())
}

pub(in crate::engine) fn dispatch_irreversible97_repeated_buffers_in_encoder_with_offsets(
    encoder: &ComputeCommandEncoderRef,
    dispatch: RepeatedIdwtDispatch<'_>,
) {
    let high_pass = j2k_codec_math::dwt::IDWT97_OPENJPEG_TWO_INV_KAPPA_F32 * 0.5;
    if supports_irreversible97_interleave_horizontal_fused(
        dispatch.kernels,
        &dispatch
            .kernels
            .idwt_irreversible97_interleave_horizontal_fused_batched,
        dispatch.params.width,
        dispatch.params.height,
    ) {
        dispatch_irreversible97_repeated_interleave_horizontal_fused(encoder, dispatch, high_pass);
        dispatch_irreversible97_vertical_after_fused_horizontal(
            encoder,
            dispatch.kernels,
            dispatch.decoded,
            0,
            single_params(dispatch.params),
            high_pass,
            dispatch.params.batch_count,
        );
    } else {
        dispatch_irreversible97_repeated_interleave_horizontal_scale(encoder, dispatch, high_pass);
        dispatch_irreversible97_stages_after_horizontal_scale(
            encoder,
            dispatch.kernels,
            dispatch.decoded,
            0,
            single_params(dispatch.params),
            high_pass,
            dispatch.params.batch_count,
        );
    }
}

pub(super) fn dispatch_irreversible97_repeated_interleave_horizontal_fused(
    encoder: &ComputeCommandEncoderRef,
    dispatch: RepeatedIdwtDispatch<'_>,
    high_pass: f32,
) {
    let RepeatedIdwtDispatch {
        kernels,
        sub_bands,
        params,
        decoded,
    } = dispatch;
    let pipeline = &kernels.idwt_irreversible97_interleave_horizontal_fused_batched;
    encoder.setComputePipelineState(pipeline);
    for (index, buffer, offset) in [
        (0, sub_bands.ll, sub_bands.ll_offset),
        (1, sub_bands.hl, sub_bands.hl_offset),
        (2, sub_bands.lh, sub_bands.lh_offset),
        (3, sub_bands.hh, sub_bands.hh_offset),
    ] {
        encoder.set_buffer(index, Some(buffer), offset as u64);
    }
    encoder.set_buffer(4, Some(decoded), 0);
    encoder.set_bytes::<J2kRepeatedIdwtSingleDecompositionParams>(5, &params);
    encoder.set_bytes(
        6,
        &irreversible97_horizontal_lift_steps(params.x0 + params.output_x, high_pass),
    );
    encoder.dispatchThreadgroups_threadsPerThreadgroup(
        j2k_metal_support::mtl_size(
            u64::from(params.width.div_ceil(IDWT97_ROW_TILE)),
            u64::from(params.height.div_ceil(IDWT97_ROWS_PER_GROUP)),
            u64::from(params.batch_count),
        ),
        j2k_metal_support::mtl_size(
            u64::from(IDWT97_ROW_THREADS),
            u64::from(IDWT97_ROWS_PER_GROUP),
            1,
        ),
    );
    #[cfg(test)]
    crate::engine::test_counters::record_idwt97_logical_dispatch((
        params.width,
        params.height,
        params.batch_count,
    ));
    encoder.memory_barrier_with_resources(&[decoded]);
}

pub(super) fn dispatch_irreversible97_repeated_interleave_horizontal_scale(
    encoder: &ComputeCommandEncoderRef,
    dispatch: RepeatedIdwtDispatch<'_>,
    high_pass: f32,
) {
    let RepeatedIdwtDispatch {
        kernels,
        sub_bands,
        params,
        decoded,
    } = dispatch;
    encoder
        .setComputePipelineState(&kernels.idwt_irreversible97_interleave_horizontal_scale_batched);
    for (index, buffer, offset) in [
        (0, sub_bands.ll, sub_bands.ll_offset),
        (1, sub_bands.hl, sub_bands.hl_offset),
        (2, sub_bands.lh, sub_bands.lh_offset),
        (3, sub_bands.hh, sub_bands.hh_offset),
    ] {
        encoder.set_buffer(index, Some(buffer), offset as u64);
    }
    encoder.set_buffer(4, Some(decoded), 0);
    encoder.set_bytes::<J2kRepeatedIdwtSingleDecompositionParams>(5, &params);
    encoder.set_bytes::<f32>(6, &high_pass);
    dispatch_3d_pipeline(
        encoder,
        &kernels.idwt_irreversible97_interleave_horizontal_scale_batched,
        (params.width, params.height, params.batch_count),
    );
    #[cfg(test)]
    crate::engine::test_counters::record_idwt97_logical_dispatch((
        params.width,
        params.height,
        params.batch_count,
    ));
    encoder.memory_barrier_with_resources(&[decoded]);
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
