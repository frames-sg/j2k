// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::{
    checked_buffer_read, checked_buffer_slice, commit_and_wait_metal, copied_slice_buffer,
    decode_idwt_status_error, dispatch_2d_pipeline, hybrid_stage_signpost, label_compute_encoder,
    new_command_buffer, new_compute_command_encoder, size_of, with_runtime, zeroed_shared_buffer,
    Buffer, CommandBufferRef, ComputeCommandEncoderRef, DirectStatusCheck, Error,
    J2kIdwt97StepParams, J2kIdwtSingleDecompositionParams, J2kIdwtStatus,
    J2kSingleDecompositionIdwtJob, MetalRuntime, J2K_IDWT_STATUS_OK,
    SIGNPOST_DECODE_HYBRID_IDWT_COMMAND_ENCODE,
};
use super::{IdwtSubBandBuffers, SingleIdwtDispatch};
use j2k_codec_math::dwt;

pub(crate) fn decode_irreversible97_single_decomposition_idwt(
    job: J2kSingleDecompositionIdwtJob<'_>,
    output: &mut [f32],
) -> Result<(), Error> {
    decode_irreversible97_staged_single_decomposition_idwt(job, output)
}

pub(crate) fn decode_irreversible97_staged_single_decomposition_idwt(
    job: J2kSingleDecompositionIdwtJob<'_>,
    output: &mut [f32],
) -> Result<(), Error> {
    with_runtime(|runtime| {
        let required_len = job.rect.width() as usize * job.rect.height() as usize;
        if output.len() < required_len {
            return Err(Error::MetalKernel {
                message: "J2K Metal IDWT output slice is too small".to_string(),
            });
        }

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

        let ll = copied_slice_buffer(&runtime.device, job.ll.coefficients)?;
        let hl = copied_slice_buffer(&runtime.device, job.hl.coefficients)?;
        let lh = copied_slice_buffer(&runtime.device, job.lh.coefficients)?;
        let hh = copied_slice_buffer(&runtime.device, job.hh.coefficients)?;
        let decoded = copied_slice_buffer(&runtime.device, output)?;
        let status_buffer = zeroed_shared_buffer(&runtime.device, size_of::<J2kIdwtStatus>())?;

        let command_buffer = new_command_buffer(&runtime.queue)?;
        let encoder = new_compute_command_encoder(&command_buffer)?;
        dispatch_irreversible97_single_decomposition_buffers_in_encoder_with_status(
            &encoder,
            SingleIdwtDispatch {
                runtime,
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
            &status_buffer,
        );
        encoder.end_encoding();
        commit_and_wait_metal(&command_buffer)?;

        let status = checked_buffer_read::<J2kIdwtStatus>(&status_buffer, "IDWT status")?;
        if status.code != J2K_IDWT_STATUS_OK {
            return Err(decode_idwt_status_error(status));
        }
        let decoded_host = checked_buffer_slice::<f32>(&decoded, output.len(), "IDWT output")?;
        output.copy_from_slice(&decoded_host);
        Ok(())
    })
}

pub(in crate::compute) fn dispatch_irreversible97_single_decomposition_buffers_in_command_buffer_with_offsets(
    command_buffer: &CommandBufferRef,
    dispatch: SingleIdwtDispatch<'_>,
) -> Result<DirectStatusCheck, Error> {
    let _signpost = hybrid_stage_signpost(SIGNPOST_DECODE_HYBRID_IDWT_COMMAND_ENCODE);
    let status_buffer = zeroed_shared_buffer(&dispatch.runtime.device, size_of::<J2kIdwtStatus>())?;

    let encoder = new_compute_command_encoder(command_buffer)?;
    label_compute_encoder(&encoder, "J2K decode hybrid irreversible97 IDWT");
    dispatch_irreversible97_single_decomposition_buffers_in_encoder_with_status(
        &encoder,
        dispatch,
        &status_buffer,
    );
    encoder.end_encoding();

    Ok(DirectStatusCheck::Idwt(status_buffer))
}

pub(in crate::compute) fn dispatch_irreversible97_single_decomposition_buffers_in_encoder_with_offsets(
    encoder: &ComputeCommandEncoderRef,
    dispatch: SingleIdwtDispatch<'_>,
) -> Result<DirectStatusCheck, Error> {
    let status_buffer = zeroed_shared_buffer(&dispatch.runtime.device, size_of::<J2kIdwtStatus>())?;
    dispatch_irreversible97_single_decomposition_buffers_in_encoder_with_status(
        encoder,
        dispatch,
        &status_buffer,
    );

    Ok(DirectStatusCheck::Idwt(status_buffer))
}

fn dispatch_irreversible97_single_decomposition_buffers_in_encoder_with_status(
    encoder: &ComputeCommandEncoderRef,
    dispatch: SingleIdwtDispatch<'_>,
    _status_buffer: &Buffer,
) {
    let SingleIdwtDispatch {
        runtime,
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
    encoder.set_compute_pipeline_state(&runtime.idwt_interleave);
    encoder.set_buffer(0, Some(ll), ll_offset as u64);
    encoder.set_buffer(1, Some(hl), hl_offset as u64);
    encoder.set_buffer(2, Some(lh), lh_offset as u64);
    encoder.set_buffer(3, Some(hh), hh_offset as u64);
    encoder.set_buffer(4, Some(decoded), decoded_offset as u64);
    encoder.set_bytes(
        5,
        size_of::<J2kIdwtSingleDecompositionParams>() as u64,
        (&raw const params).cast(),
    );
    dispatch_2d_pipeline(
        encoder,
        &runtime.idwt_interleave,
        (params.width, params.height),
    );
    encoder.memory_barrier_with_resources(&[decoded]);

    dispatch_irreversible97_stages(encoder, runtime, decoded, decoded_offset, params);
}

fn dispatch_irreversible97_stages(
    encoder: &ComputeCommandEncoderRef,
    runtime: &MetalRuntime,
    decoded: &Buffer,
    decoded_offset: usize,
    params: J2kIdwtSingleDecompositionParams,
) {
    encoder.set_compute_pipeline_state(&runtime.idwt_irreversible97_horizontal_scale);
    encoder.set_buffer(0, Some(decoded), decoded_offset as u64);
    encoder.set_bytes(
        1,
        size_of::<J2kIdwtSingleDecompositionParams>() as u64,
        (&raw const params).cast(),
    );
    dispatch_2d_pipeline(
        encoder,
        &runtime.idwt_irreversible97_horizontal_scale,
        (params.width, params.height),
    );
    encoder.memory_barrier_with_resources(&[decoded]);

    let first_even_x = (params.x0 + params.output_x) & 1;
    let first_odd_x = 1 - first_even_x;
    encoder.set_compute_pipeline_state(&runtime.idwt_irreversible97_horizontal_step);
    encoder.set_buffer(0, Some(decoded), decoded_offset as u64);
    encoder.set_bytes(
        1,
        size_of::<J2kIdwtSingleDecompositionParams>() as u64,
        (&raw const params).cast(),
    );
    for (coefficient, parity) in [
        (dwt::IDWT97_NEG_DELTA_F32, first_even_x),
        (dwt::IDWT97_NEG_GAMMA_F32, first_odd_x),
        (dwt::IDWT97_NEG_BETA_F32, first_even_x),
        (dwt::IDWT97_NEG_ALPHA_F32, first_odd_x),
    ] {
        let step = J2kIdwt97StepParams {
            coefficient,
            parity,
            _reserved0: 0,
            _reserved1: 0,
        };
        encoder.set_bytes(
            2,
            size_of::<J2kIdwt97StepParams>() as u64,
            (&raw const step).cast(),
        );
        dispatch_2d_pipeline(
            encoder,
            &runtime.idwt_irreversible97_horizontal_step,
            (params.width, params.height),
        );
        encoder.memory_barrier_with_resources(&[decoded]);
    }

    encoder.set_compute_pipeline_state(&runtime.idwt_irreversible97_vertical_scale);
    encoder.set_buffer(0, Some(decoded), decoded_offset as u64);
    encoder.set_bytes(
        1,
        size_of::<J2kIdwtSingleDecompositionParams>() as u64,
        (&raw const params).cast(),
    );
    dispatch_2d_pipeline(
        encoder,
        &runtime.idwt_irreversible97_vertical_scale,
        (params.width, params.height),
    );
    encoder.memory_barrier_with_resources(&[decoded]);

    let first_even_y = (params.y0 + params.output_y) & 1;
    let first_odd_y = 1 - first_even_y;
    encoder.set_compute_pipeline_state(&runtime.idwt_irreversible97_vertical_step);
    encoder.set_buffer(0, Some(decoded), decoded_offset as u64);
    encoder.set_bytes(
        1,
        size_of::<J2kIdwtSingleDecompositionParams>() as u64,
        (&raw const params).cast(),
    );
    for (coefficient, parity) in [
        (dwt::IDWT97_NEG_DELTA_F32, first_even_y),
        (dwt::IDWT97_NEG_GAMMA_F32, first_odd_y),
        (dwt::IDWT97_NEG_BETA_F32, first_even_y),
        (dwt::IDWT97_NEG_ALPHA_F32, first_odd_y),
    ] {
        let step = J2kIdwt97StepParams {
            coefficient,
            parity,
            _reserved0: 0,
            _reserved1: 0,
        };
        encoder.set_bytes(
            2,
            size_of::<J2kIdwt97StepParams>() as u64,
            (&raw const step).cast(),
        );
        dispatch_2d_pipeline(
            encoder,
            &runtime.idwt_irreversible97_vertical_step,
            (params.width, params.height),
        );
        encoder.memory_barrier_with_resources(&[decoded]);
    }
}
