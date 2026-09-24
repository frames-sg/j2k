// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(target_os = "macos")]
use crate::metal_types::prelude::*;

#[cfg(test)]
use super::super::checked_buffer_slice;
#[cfg(test)]
use super::super::dispatch_3d_pipeline;
use super::super::{
    checked_buffer_copy_into, commit_and_wait_metal, copied_slice_buffer, dispatch_2d_pipeline,
    hybrid_stage_signpost, label_compute_encoder, new_command_buffer, new_compute_command_encoder,
    new_shared_buffer, with_runtime, Buffer, CommandBufferRef, ComputeCommandEncoderRef, Error,
    J2kIdwt97LiftSteps, J2kIdwtSingleDecompositionParams, J2kSingleDecompositionIdwtJob,
    SIGNPOST_DECODE_HYBRID_IDWT_COMMAND_ENCODE,
};
use super::{checked_host_output_layout, IdwtSubBandBuffers, SingleIdwtDispatch};
use j2k_codec_math::dwt;

pub(crate) fn decode_irreversible97_single_decomposition_idwt(
    job: J2kSingleDecompositionIdwtJob<'_>,
    output: &mut [f32],
) -> Result<(), Error> {
    decode_irreversible97_staged_single_decomposition_idwt(job, output)
}

pub(crate) fn decode_openjpeg_irreversible97_single_decomposition_idwt(
    job: J2kSingleDecompositionIdwtJob<'_>,
    output: &mut [f32],
) -> Result<(), Error> {
    decode_irreversible97_staged_single_decomposition_idwt_with_high_pass(
        job,
        output,
        dwt::IDWT97_OPENJPEG_TWO_INV_KAPPA_F32 * 0.5,
    )
}

pub(crate) fn decode_irreversible97_staged_single_decomposition_idwt(
    job: J2kSingleDecompositionIdwtJob<'_>,
    output: &mut [f32],
) -> Result<(), Error> {
    decode_irreversible97_staged_single_decomposition_idwt_with_high_pass(
        job,
        output,
        dwt::DWT97_INV_KAPPA_F32,
    )
}

fn decode_irreversible97_staged_single_decomposition_idwt_with_high_pass(
    job: J2kSingleDecompositionIdwtJob<'_>,
    output: &mut [f32],
    high_pass: f32,
) -> Result<(), Error> {
    with_runtime(|runtime| {
        let (required_len, required_bytes) =
            checked_host_output_layout(job.rect.width(), job.rect.height(), output.len())?;

        let params = J2kIdwtSingleDecompositionParams {
            x0: job.rect.x0,
            y0: job.rect.y0,
            output_x: 0,
            output_y: 0,
            width: job.rect.width(),
            height: job.rect.height(),
            ll_x: 0,
            ll_y: 0,
            ll_width: job.ll.rect.width(),
            ll_height: job.ll.rect.height(),
            hl_x: 0,
            hl_y: 0,
            hl_width: job.hl.rect.width(),
            hl_height: job.hl.rect.height(),
            lh_x: 0,
            lh_y: 0,
            lh_width: job.lh.rect.width(),
            lh_height: job.lh.rect.height(),
            hh_x: 0,
            hh_y: 0,
            hh_width: job.hh.rect.width(),
            hh_height: job.hh.rect.height(),
        };

        let decoded = new_shared_buffer(&runtime.device, required_bytes)?;
        let ll = copied_slice_buffer(&runtime.device, job.ll.coefficients)?;
        let hl = copied_slice_buffer(&runtime.device, job.hl.coefficients)?;
        let lh = copied_slice_buffer(&runtime.device, job.lh.coefficients)?;
        let hh = copied_slice_buffer(&runtime.device, job.hh.coefficients)?;
        let command_buffer = new_command_buffer(&runtime.queue)?;
        let encoder = new_compute_command_encoder(&command_buffer)?;
        dispatch_irreversible97_single_decomposition_buffers_in_encoder_with_high_pass(
            &encoder,
            SingleIdwtDispatch {
                kernels: runtime.decode()?,
                sub_bands: IdwtSubBandBuffers {
                    ll: &ll,
                    ll_offset: 0,
                    hl: &hl,
                    hl_offset: 0,
                    lh: &lh,
                    lh_offset: 0,
                    hh: &hh,
                    hh_offset: 0,
                },
                params,
                decoded: &decoded,
                decoded_offset: 0,
            },
            high_pass,
        );
        encoder.endEncoding();
        commit_and_wait_metal(&command_buffer)?;

        checked_buffer_copy_into(&decoded, 0, &mut output[..required_len], "IDWT output")?;
        Ok(())
    })
}

pub(in crate::engine) fn dispatch_irreversible97_single_decomposition_buffers_in_command_buffer_with_offsets(
    command_buffer: &CommandBufferRef,
    dispatch: SingleIdwtDispatch<'_>,
) -> Result<(), Error> {
    let _signpost = hybrid_stage_signpost(SIGNPOST_DECODE_HYBRID_IDWT_COMMAND_ENCODE);
    let encoder = new_compute_command_encoder(command_buffer)?;
    label_compute_encoder(&encoder, "J2K decode hybrid irreversible97 IDWT");
    dispatch_irreversible97_single_decomposition_buffers_in_encoder_with_offsets(
        &encoder, dispatch,
    );
    encoder.endEncoding();
    Ok(())
}

pub(in crate::engine) fn dispatch_irreversible97_single_decomposition_buffers_in_encoder_with_offsets(
    encoder: &ComputeCommandEncoderRef,
    dispatch: SingleIdwtDispatch<'_>,
) {
    dispatch_irreversible97_single_decomposition_buffers_in_encoder_with_high_pass(
        encoder,
        dispatch,
        dwt::IDWT97_OPENJPEG_TWO_INV_KAPPA_F32 * 0.5,
    );
}

fn dispatch_irreversible97_single_decomposition_buffers_in_encoder_with_high_pass(
    encoder: &ComputeCommandEncoderRef,
    dispatch: SingleIdwtDispatch<'_>,
    high_pass: f32,
) {
    dispatch_irreversible97_interleave_horizontal_scale(encoder, dispatch, high_pass);
    dispatch_irreversible97_stages_after_horizontal_scale(
        encoder,
        dispatch.kernels,
        dispatch.decoded,
        dispatch.decoded_offset,
        dispatch.params,
        high_pass,
        1,
    );
}

pub(super) fn dispatch_irreversible97_interleave_horizontal_scale(
    encoder: &ComputeCommandEncoderRef,
    dispatch: SingleIdwtDispatch<'_>,
    high_pass: f32,
) {
    let SingleIdwtDispatch {
        kernels,
        sub_bands,
        params,
        decoded,
        decoded_offset,
    } = dispatch;
    let IdwtSubBandBuffers {
        ll,
        ll_offset,
        hl,
        hl_offset,
        lh,
        lh_offset,
        hh,
        hh_offset,
    } = sub_bands;
    encoder.setComputePipelineState(&kernels.idwt_irreversible97_interleave_horizontal_scale);
    encoder.set_buffer(0, Some(ll), ll_offset as u64);
    encoder.set_buffer(1, Some(hl), hl_offset as u64);
    encoder.set_buffer(2, Some(lh), lh_offset as u64);
    encoder.set_buffer(3, Some(hh), hh_offset as u64);
    encoder.set_buffer(4, Some(decoded), decoded_offset as u64);
    encoder.set_bytes::<J2kIdwtSingleDecompositionParams>(5, &params);
    encoder.set_bytes::<f32>(6, &high_pass);
    dispatch_2d_pipeline(
        encoder,
        &kernels.idwt_irreversible97_interleave_horizontal_scale,
        (params.width, params.height),
    );
    #[cfg(test)]
    crate::engine::test_counters::record_idwt97_logical_dispatch((params.width, params.height, 1));
    encoder.memory_barrier_with_resources(&[decoded]);
}

#[cfg(test)]
pub(super) fn dispatch_irreversible97_stages(
    encoder: &ComputeCommandEncoderRef,
    kernels: &crate::engine::runtime::DecodeKernels,
    decoded: &Buffer,
    decoded_offset: usize,
    params: J2kIdwtSingleDecompositionParams,
    high_pass: f32,
    batch_count: u32,
) {
    dispatch_irreversible97_horizontal_scale(
        encoder,
        kernels,
        decoded,
        decoded_offset,
        params,
        high_pass,
        batch_count,
    );
    dispatch_irreversible97_stages_after_horizontal_scale(
        encoder,
        kernels,
        decoded,
        decoded_offset,
        params,
        high_pass,
        batch_count,
    );
}

#[cfg(test)]
pub(super) fn dispatch_irreversible97_horizontal_scale(
    encoder: &ComputeCommandEncoderRef,
    kernels: &crate::engine::runtime::DecodeKernels,
    decoded: &Buffer,
    decoded_offset: usize,
    params: J2kIdwtSingleDecompositionParams,
    high_pass: f32,
    batch_count: u32,
) {
    encoder.setComputePipelineState(&kernels.idwt_irreversible97_horizontal_scale);
    encoder.set_buffer(0, Some(decoded), decoded_offset as u64);
    encoder.set_bytes::<J2kIdwtSingleDecompositionParams>(1, &params);
    encoder.set_bytes::<f32>(2, &high_pass);
    let horizontal_scale_grid = (params.width, params.height, batch_count);
    #[cfg(test)]
    crate::engine::test_counters::record_idwt97_logical_dispatch(horizontal_scale_grid);
    dispatch_3d_pipeline(
        encoder,
        &kernels.idwt_irreversible97_horizontal_scale,
        horizontal_scale_grid,
    );
    encoder.memory_barrier_with_resources(&[decoded]);
}

/// Threadgroup geometry of the fused lifting kernels; must match
/// `J2K_IDWT97_*` in `idwt.metal`. Each horizontal group owns whole rows and
/// each vertical group a whole strip of columns.
const IDWT97_ROWS_PER_GROUP: u32 = 4;
const IDWT97_ROW_THREADS: u32 = 64;
const IDWT97_COL_TILE: u32 = 32;
const IDWT97_COL_ROW_THREADS: u32 = 8;

const IDWT97_LIFT_COEFFICIENTS: [f32; 4] = [
    dwt::IDWT97_NEG_DELTA_F32,
    dwt::IDWT97_NEG_GAMMA_F32,
    dwt::IDWT97_NEG_BETA_F32,
    dwt::IDWT97_NEG_ALPHA_F32,
];

/// Encodes the horizontal lifting steps, vertical scale, and vertical lifting
/// steps as two fused tile dispatches.
pub(super) fn dispatch_irreversible97_stages_after_horizontal_scale(
    encoder: &ComputeCommandEncoderRef,
    kernels: &crate::engine::runtime::DecodeKernels,
    decoded: &Buffer,
    decoded_offset: usize,
    params: J2kIdwtSingleDecompositionParams,
    high_pass: f32,
    batch_count: u32,
) {
    #[cfg(test)]
    crate::engine::test_counters::record_idwt97_stage_sequence();

    if params.width == 0 || params.height == 0 || batch_count == 0 {
        return;
    }
    let size = j2k_metal_support::mtl_size;
    encoder.set_buffer(0, Some(decoded), decoded_offset as u64);
    encoder.set_bytes::<J2kIdwtSingleDecompositionParams>(1, &params);

    if params.width > 1 {
        let horizontal = J2kIdwt97LiftSteps {
            coefficients: IDWT97_LIFT_COEFFICIENTS,
            first_parity: (params.x0 + params.output_x) & 1,
            high_pass_bits: high_pass.to_bits(),
            _reserved0: 0,
            _reserved1: 0,
        };
        let groups = (
            1_u32,
            params.height.div_ceil(IDWT97_ROWS_PER_GROUP),
            batch_count,
        );
        #[cfg(test)]
        crate::engine::test_counters::record_idwt97_logical_dispatch((
            params.width,
            params.height,
            batch_count,
        ));
        encoder.setComputePipelineState(&kernels.idwt_irreversible97_horizontal_lift_fused);
        encoder.set_bytes::<J2kIdwt97LiftSteps>(2, &horizontal);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            size(
                u64::from(groups.0),
                u64::from(groups.1),
                u64::from(groups.2),
            ),
            size(
                u64::from(IDWT97_ROW_THREADS),
                u64::from(IDWT97_ROWS_PER_GROUP),
                1,
            ),
        );
        encoder.memory_barrier_with_resources(&[decoded]);
    }

    let vertical = J2kIdwt97LiftSteps {
        coefficients: IDWT97_LIFT_COEFFICIENTS,
        first_parity: (params.y0 + params.output_y) & 1,
        high_pass_bits: high_pass.to_bits(),
        _reserved0: 0,
        _reserved1: 0,
    };
    let groups = (params.width.div_ceil(IDWT97_COL_TILE), 1_u32, batch_count);
    #[cfg(test)]
    crate::engine::test_counters::record_idwt97_logical_dispatch((
        params.width,
        params.height,
        batch_count,
    ));
    encoder.setComputePipelineState(&kernels.idwt_irreversible97_vertical_fused);
    encoder.set_bytes::<J2kIdwt97LiftSteps>(2, &vertical);
    encoder.dispatchThreadgroups_threadsPerThreadgroup(
        size(
            u64::from(groups.0),
            u64::from(groups.1),
            u64::from(groups.2),
        ),
        size(
            u64::from(IDWT97_COL_TILE),
            u64::from(IDWT97_COL_ROW_THREADS),
            1,
        ),
    );
    encoder.memory_barrier_with_resources(&[decoded]);
}

#[cfg(test)]
mod parity_tests;
#[cfg(test)]
mod performance;
