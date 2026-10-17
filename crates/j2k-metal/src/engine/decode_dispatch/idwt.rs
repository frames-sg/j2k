// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(target_os = "macos")]
use crate::metal_types::prelude::*;
#[cfg(target_os = "macos")]
use crate::metal_types::ComputePipelineState;

use super::{
    checked_buffer_copy_into, commit_and_wait_metal, copied_slice_buffer, dispatch_2d_pipeline,
    dispatch_3d_pipeline, hybrid_stage_signpost, label_compute_encoder, new_command_buffer,
    new_compute_command_encoder, new_shared_buffer, with_runtime, Buffer, CommandBufferRef,
    ComputeCommandEncoderRef, DirectIdwtCommandBuffers, Error, J2kIdwtSingleDecompositionParams,
    J2kRepeatedIdwtSingleDecompositionParams, J2kSingleDecompositionIdwtJob,
    SIGNPOST_DECODE_HYBRID_IDWT_COMMAND_ENCODE,
};

#[cfg(target_os = "macos")]
fn checked_host_output_layout(
    width: u32,
    height: u32,
    output_len: usize,
) -> Result<(usize, usize), Error> {
    let required_len = (width as usize)
        .checked_mul(height as usize)
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K Metal IDWT output length overflow".to_string(),
        })?;
    let required_bytes = required_len
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K Metal IDWT output byte length overflow".to_string(),
        })?;
    if output_len < required_len {
        return Err(Error::MetalKernel {
            message: "J2K Metal IDWT output slice is too small".to_string(),
        });
    }
    Ok((required_len, required_bytes))
}
#[cfg(target_os = "macos")]
mod batched_irreversible;
#[cfg(target_os = "macos")]
mod irreversible;
#[cfg(target_os = "macos")]
pub(in crate::engine) use batched_irreversible::{
    dispatch_irreversible97_repeated_buffers_in_command_buffer_with_offsets,
    dispatch_irreversible97_repeated_buffers_in_encoder_with_offsets,
};

#[cfg(target_os = "macos")]
const IDWT53_ROW_TILE: u32 = 128;
#[cfg(target_os = "macos")]
const IDWT53_ROWS_PER_GROUP: u32 = 4;
#[cfg(target_os = "macos")]
const IDWT53_ROW_THREADS: u32 = 64;
#[cfg(target_os = "macos")]
const IDWT53_COLS_PER_GROUP: u32 = 16;
#[cfg(target_os = "macos")]
const IDWT53_COL_TILE: u32 = 128;
#[cfg(target_os = "macos")]
const IDWT53_COL_ROW_THREADS: u32 = 16;

#[cfg(target_os = "macos")]
fn supports_reversible53_interleave_horizontal_fused(
    kernels: &crate::engine::runtime::DecodeKernels,
    pipeline: &ComputePipelineState,
    width: u32,
    height: u32,
) -> bool {
    width > IDWT53_ROW_TILE
        && height > 0
        && pipeline.maxTotalThreadsPerThreadgroup()
            >= (IDWT53_ROW_THREADS * IDWT53_ROWS_PER_GROUP) as usize
        && pipeline.staticThreadgroupMemoryLength() <= kernels.max_threadgroup_memory_length
}

/// Output tile of `j2k_idwt_reversible53_fused_2d`, and its threadgroup of
/// `IDWT53_2D_LANES` x `IDWT53_2D_ROWS` threads.
const IDWT53_2D_TILE_W: u32 = 60;
const IDWT53_2D_TILE_H: u32 = 60;
pub(super) const IDWT53_2D_LANES: u64 = 32;
pub(super) const IDWT53_2D_ROWS: u64 = 8;

/// Whether a single-pass tile kernel can run this level: both dimensions at
/// least 2, which the kernels' boundary handling assumes, and a pipeline that
/// fits its threadgroup.
#[cfg(target_os = "macos")]
pub(in crate::engine) fn supports_fused_2d_idwt(
    kernels: &crate::engine::runtime::DecodeKernels,
    pipeline: &ComputePipelineState,
    width: u32,
    height: u32,
) -> bool {
    #[cfg(test)]
    if !crate::engine::test_counters::fused_2d_idwt_for_test() {
        return false;
    }
    width >= 2
        && height >= 2
        && pipeline.threadExecutionWidth() as u64 == IDWT53_2D_LANES
        && pipeline.maxTotalThreadsPerThreadgroup() as u64 >= IDWT53_2D_LANES * IDWT53_2D_ROWS
        && pipeline.staticThreadgroupMemoryLength() <= kernels.max_threadgroup_memory_length
}

#[cfg(target_os = "macos")]
fn dispatch_reversible53_fused_2d_repeated(
    encoder: &ComputeCommandEncoderRef,
    dispatch: RepeatedIdwtDispatch<'_>,
) {
    let params = dispatch.params;
    encoder.setComputePipelineState(&dispatch.kernels.idwt_reversible53_fused_2d_batched);
    for (index, buffer, offset) in [
        (0, dispatch.sub_bands.ll, dispatch.sub_bands.ll_offset),
        (1, dispatch.sub_bands.hl, dispatch.sub_bands.hl_offset),
        (2, dispatch.sub_bands.lh, dispatch.sub_bands.lh_offset),
        (3, dispatch.sub_bands.hh, dispatch.sub_bands.hh_offset),
    ] {
        encoder.set_buffer(index, Some(buffer), offset as u64);
    }
    encoder.set_buffer(4, Some(dispatch.decoded), 0);
    encoder.set_bytes::<J2kRepeatedIdwtSingleDecompositionParams>(5, &params);
    encoder.dispatchThreadgroups_threadsPerThreadgroup(
        j2k_metal_support::mtl_size(
            u64::from(params.width.div_ceil(IDWT53_2D_TILE_W)),
            u64::from(params.height.div_ceil(IDWT53_2D_TILE_H)),
            u64::from(params.batch_count),
        ),
        j2k_metal_support::mtl_size(IDWT53_2D_LANES, IDWT53_2D_ROWS, 1),
    );
}

#[cfg(target_os = "macos")]
fn dispatch_reversible53_fused_2d_single(
    encoder: &ComputeCommandEncoderRef,
    dispatch: SingleIdwtDispatch<'_>,
) {
    let pipeline = &dispatch.kernels.idwt_reversible53_fused_2d;
    encoder.setComputePipelineState(pipeline);
    for (index, buffer, offset) in [
        (0, dispatch.sub_bands.ll, dispatch.sub_bands.ll_offset),
        (1, dispatch.sub_bands.hl, dispatch.sub_bands.hl_offset),
        (2, dispatch.sub_bands.lh, dispatch.sub_bands.lh_offset),
        (3, dispatch.sub_bands.hh, dispatch.sub_bands.hh_offset),
    ] {
        encoder.set_buffer(index, Some(buffer), offset as u64);
    }
    encoder.set_buffer(4, Some(dispatch.decoded), dispatch.decoded_offset as u64);
    encoder.set_bytes::<J2kIdwtSingleDecompositionParams>(5, &dispatch.params);
    encoder.dispatchThreadgroups_threadsPerThreadgroup(
        j2k_metal_support::mtl_size(
            u64::from(dispatch.params.width.div_ceil(IDWT53_2D_TILE_W)),
            u64::from(dispatch.params.height.div_ceil(IDWT53_2D_TILE_H)),
            1,
        ),
        j2k_metal_support::mtl_size(IDWT53_2D_LANES, IDWT53_2D_ROWS, 1),
    );
}

#[cfg(target_os = "macos")]
fn supports_reversible53_vertical_fused(
    kernels: &crate::engine::runtime::DecodeKernels,
    pipeline: &ComputePipelineState,
    width: u32,
    height: u32,
) -> bool {
    height > IDWT53_COL_TILE
        && width > 0
        && pipeline.maxTotalThreadsPerThreadgroup()
            >= (IDWT53_COLS_PER_GROUP * IDWT53_COL_ROW_THREADS) as usize
        && pipeline.staticThreadgroupMemoryLength() <= kernels.max_threadgroup_memory_length
}

#[cfg(target_os = "macos")]
fn dispatch_reversible53_horizontal_single(
    encoder: &ComputeCommandEncoderRef,
    kernels: &crate::engine::runtime::DecodeKernels,
    decoded: &Buffer,
    decoded_offset: usize,
    params: J2kIdwtSingleDecompositionParams,
) {
    let serial = &kernels.idwt_reversible53_horizontal;
    encoder.setComputePipelineState(serial);
    encoder.set_buffer(0, Some(decoded), decoded_offset as u64);
    encoder.set_bytes::<J2kIdwtSingleDecompositionParams>(1, &params);
    let threads = serial.threadExecutionWidth().max(1);
    encoder.dispatchThreads_threadsPerThreadgroup(
        j2k_metal_support::mtl_size(u64::from(params.height), 1, 1),
        j2k_metal_support::mtl_size(threads as u64, 1, 1),
    );
}

#[cfg(target_os = "macos")]
fn dispatch_reversible53_horizontal_repeated(
    encoder: &ComputeCommandEncoderRef,
    kernels: &crate::engine::runtime::DecodeKernels,
    decoded: &Buffer,
    params: J2kRepeatedIdwtSingleDecompositionParams,
) {
    let serial = &kernels.idwt_reversible53_horizontal_batched;
    encoder.setComputePipelineState(serial);
    encoder.set_buffer(0, Some(decoded), 0);
    encoder.set_bytes::<J2kRepeatedIdwtSingleDecompositionParams>(1, &params);
    let threads = serial.threadExecutionWidth().max(1);
    encoder.dispatchThreads_threadsPerThreadgroup(
        j2k_metal_support::mtl_size(u64::from(params.height), u64::from(params.batch_count), 1),
        j2k_metal_support::mtl_size(threads as u64, 1, 1),
    );
}

#[cfg(target_os = "macos")]
fn dispatch_reversible53_vertical_single(
    encoder: &ComputeCommandEncoderRef,
    kernels: &crate::engine::runtime::DecodeKernels,
    decoded: &Buffer,
    decoded_offset: usize,
    params: J2kIdwtSingleDecompositionParams,
) {
    let fused = &kernels.idwt_reversible53_vertical_fused;
    encoder.set_buffer(0, Some(decoded), decoded_offset as u64);
    encoder.set_bytes::<J2kIdwtSingleDecompositionParams>(1, &params);
    if supports_reversible53_vertical_fused(kernels, fused, params.width, params.height) {
        encoder.setComputePipelineState(fused);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            j2k_metal_support::mtl_size(
                u64::from(params.width.div_ceil(IDWT53_COLS_PER_GROUP)),
                1,
                1,
            ),
            j2k_metal_support::mtl_size(
                u64::from(IDWT53_COLS_PER_GROUP),
                u64::from(IDWT53_COL_ROW_THREADS),
                1,
            ),
        );
    } else {
        let serial = &kernels.idwt_reversible53_vertical;
        encoder.setComputePipelineState(serial);
        let threads = serial.threadExecutionWidth().max(1);
        encoder.dispatchThreads_threadsPerThreadgroup(
            j2k_metal_support::mtl_size(u64::from(params.width), 1, 1),
            j2k_metal_support::mtl_size(threads as u64, 1, 1),
        );
    }
}

#[cfg(target_os = "macos")]
fn dispatch_reversible53_vertical_repeated(
    encoder: &ComputeCommandEncoderRef,
    kernels: &crate::engine::runtime::DecodeKernels,
    decoded: &Buffer,
    params: J2kRepeatedIdwtSingleDecompositionParams,
) {
    let fused = &kernels.idwt_reversible53_vertical_fused_batched;
    encoder.set_buffer(0, Some(decoded), 0);
    encoder.set_bytes::<J2kRepeatedIdwtSingleDecompositionParams>(1, &params);
    if supports_reversible53_vertical_fused(kernels, fused, params.width, params.height) {
        encoder.setComputePipelineState(fused);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            j2k_metal_support::mtl_size(
                u64::from(params.width.div_ceil(IDWT53_COLS_PER_GROUP)),
                1,
                u64::from(params.batch_count),
            ),
            j2k_metal_support::mtl_size(
                u64::from(IDWT53_COLS_PER_GROUP),
                u64::from(IDWT53_COL_ROW_THREADS),
                1,
            ),
        );
    } else {
        let serial = &kernels.idwt_reversible53_vertical_batched;
        encoder.setComputePipelineState(serial);
        let threads = serial.threadExecutionWidth().max(1);
        encoder.dispatchThreads_threadsPerThreadgroup(
            j2k_metal_support::mtl_size(u64::from(params.width), u64::from(params.batch_count), 1),
            j2k_metal_support::mtl_size(threads as u64, 1, 1),
        );
    }
}

#[cfg(target_os = "macos")]
fn dispatch_reversible53_interleave_horizontal_fused_single(
    encoder: &ComputeCommandEncoderRef,
    dispatch: SingleIdwtDispatch<'_>,
) {
    let pipeline = &dispatch
        .kernels
        .idwt_reversible53_interleave_horizontal_fused;
    encoder.setComputePipelineState(pipeline);
    for (index, buffer, offset) in [
        (0, dispatch.sub_bands.ll, dispatch.sub_bands.ll_offset),
        (1, dispatch.sub_bands.hl, dispatch.sub_bands.hl_offset),
        (2, dispatch.sub_bands.lh, dispatch.sub_bands.lh_offset),
        (3, dispatch.sub_bands.hh, dispatch.sub_bands.hh_offset),
    ] {
        encoder.set_buffer(index, Some(buffer), offset as u64);
    }
    encoder.set_buffer(4, Some(dispatch.decoded), dispatch.decoded_offset as u64);
    encoder.set_bytes::<J2kIdwtSingleDecompositionParams>(5, &dispatch.params);
    encoder.dispatchThreadgroups_threadsPerThreadgroup(
        j2k_metal_support::mtl_size(
            u64::from(dispatch.params.width.div_ceil(IDWT53_ROW_TILE)),
            u64::from(dispatch.params.height.div_ceil(IDWT53_ROWS_PER_GROUP)),
            1,
        ),
        j2k_metal_support::mtl_size(
            u64::from(IDWT53_ROW_THREADS),
            u64::from(IDWT53_ROWS_PER_GROUP),
            1,
        ),
    );
}

#[cfg(target_os = "macos")]
fn dispatch_reversible53_interleave_horizontal_fused_repeated(
    encoder: &ComputeCommandEncoderRef,
    dispatch: RepeatedIdwtDispatch<'_>,
) {
    let pipeline = &dispatch
        .kernels
        .idwt_reversible53_interleave_horizontal_fused_batched;
    encoder.setComputePipelineState(pipeline);
    for (index, buffer, offset) in [
        (0, dispatch.sub_bands.ll, dispatch.sub_bands.ll_offset),
        (1, dispatch.sub_bands.hl, dispatch.sub_bands.hl_offset),
        (2, dispatch.sub_bands.lh, dispatch.sub_bands.lh_offset),
        (3, dispatch.sub_bands.hh, dispatch.sub_bands.hh_offset),
    ] {
        encoder.set_buffer(index, Some(buffer), offset as u64);
    }
    encoder.set_buffer(4, Some(dispatch.decoded), 0);
    encoder.set_bytes::<J2kRepeatedIdwtSingleDecompositionParams>(5, &dispatch.params);
    encoder.dispatchThreadgroups_threadsPerThreadgroup(
        j2k_metal_support::mtl_size(
            u64::from(dispatch.params.width.div_ceil(IDWT53_ROW_TILE)),
            u64::from(dispatch.params.height.div_ceil(IDWT53_ROWS_PER_GROUP)),
            u64::from(dispatch.params.batch_count),
        ),
        j2k_metal_support::mtl_size(
            u64::from(IDWT53_ROW_THREADS),
            u64::from(IDWT53_ROWS_PER_GROUP),
            1,
        ),
    );
}
#[cfg(all(target_os = "macos", test))]
pub(crate) use irreversible::decode_irreversible97_staged_single_decomposition_idwt;
#[cfg(target_os = "macos")]
pub(crate) use irreversible::{
    decode_irreversible97_single_decomposition_idwt,
    decode_openjpeg_irreversible97_single_decomposition_idwt,
};
#[cfg(target_os = "macos")]
pub(in crate::engine) use irreversible::{
    dispatch_irreversible97_single_decomposition_buffers_in_command_buffer_with_offsets,
    dispatch_irreversible97_single_decomposition_buffers_in_encoder_with_offsets,
};

#[cfg(target_os = "macos")]
pub(crate) fn decode_reversible53_single_decomposition_idwt(
    job: J2kSingleDecompositionIdwtJob<'_>,
    output: &mut [f32],
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
        let dispatch = SingleIdwtDispatch {
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
        };
        let use_fused = supports_reversible53_interleave_horizontal_fused(
            dispatch.kernels,
            &dispatch
                .kernels
                .idwt_reversible53_interleave_horizontal_fused,
            params.width,
            params.height,
        );

        let encoder = new_compute_command_encoder(&command_buffer)?;
        if use_fused {
            dispatch_reversible53_interleave_horizontal_fused_single(&encoder, dispatch);
        } else {
            encoder.setComputePipelineState(&runtime.decode()?.idwt_interleave);
            encoder.set_buffer(0, Some(&ll), 0);
            encoder.set_buffer(1, Some(&hl), 0);
            encoder.set_buffer(2, Some(&lh), 0);
            encoder.set_buffer(3, Some(&hh), 0);
            encoder.set_buffer(4, Some(&decoded), 0);
            encoder.set_bytes::<J2kIdwtSingleDecompositionParams>(5, &params);
            dispatch_2d_pipeline(
                &encoder,
                &runtime.decode()?.idwt_interleave,
                (params.width, params.height),
            );
        }
        encoder.endEncoding();

        if !use_fused {
            let encoder = new_compute_command_encoder(&command_buffer)?;
            dispatch_reversible53_horizontal_single(
                &encoder,
                runtime.decode()?,
                &decoded,
                0,
                params,
            );
            encoder.endEncoding();
        }

        let encoder = new_compute_command_encoder(&command_buffer)?;
        dispatch_reversible53_vertical_single(&encoder, runtime.decode()?, &decoded, 0, params);
        encoder.endEncoding();
        commit_and_wait_metal(&command_buffer)?;
        checked_buffer_copy_into(&decoded, 0, &mut output[..required_len], "IDWT output")?;
        Ok(())
    })
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
pub(in crate::engine) struct IdwtSubBandBuffers<'a> {
    pub(in crate::engine) ll: &'a Buffer,
    pub(in crate::engine) ll_offset: usize,
    pub(in crate::engine) hl: &'a Buffer,
    pub(in crate::engine) hl_offset: usize,
    pub(in crate::engine) lh: &'a Buffer,
    pub(in crate::engine) lh_offset: usize,
    pub(in crate::engine) hh: &'a Buffer,
    pub(in crate::engine) hh_offset: usize,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
pub(in crate::engine) struct SingleIdwtDispatch<'a> {
    pub(in crate::engine) kernels: &'a crate::engine::runtime::DecodeKernels,
    pub(in crate::engine) sub_bands: IdwtSubBandBuffers<'a>,
    pub(in crate::engine) params: J2kIdwtSingleDecompositionParams,
    pub(in crate::engine) decoded: &'a Buffer,
    pub(in crate::engine) decoded_offset: usize,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
pub(in crate::engine) struct RepeatedIdwtDispatch<'a> {
    pub(in crate::engine) kernels: &'a crate::engine::runtime::DecodeKernels,
    pub(in crate::engine) sub_bands: IdwtSubBandBuffers<'a>,
    pub(in crate::engine) params: J2kRepeatedIdwtSingleDecompositionParams,
    pub(in crate::engine) decoded: &'a Buffer,
}

#[cfg(target_os = "macos")]
pub(in crate::engine) fn dispatch_reversible53_single_decomposition_buffers_in_command_buffer_with_offsets(
    command_buffer: &CommandBufferRef,
    dispatch: SingleIdwtDispatch<'_>,
) -> Result<(), Error> {
    let _signpost = hybrid_stage_signpost(SIGNPOST_DECODE_HYBRID_IDWT_COMMAND_ENCODE);
    let encoder = new_compute_command_encoder(command_buffer)?;
    label_compute_encoder(&encoder, "J2K decode hybrid reversible53 IDWT");
    dispatch_reversible53_single_decomposition_buffers_in_encoder_with_offsets(&encoder, dispatch);
    encoder.endEncoding();
    Ok(())
}

#[cfg(target_os = "macos")]
pub(in crate::engine) fn dispatch_reversible53_single_decomposition_buffers_in_encoder_with_offsets(
    encoder: &ComputeCommandEncoderRef,
    dispatch: SingleIdwtDispatch<'_>,
) {
    if supports_fused_2d_idwt(
        dispatch.kernels,
        &dispatch.kernels.idwt_reversible53_fused_2d,
        dispatch.params.width,
        dispatch.params.height,
    ) {
        dispatch_reversible53_fused_2d_single(encoder, dispatch);
        encoder.memory_barrier_with_resources(&[dispatch.decoded]);
        return;
    }
    let use_fused = supports_reversible53_interleave_horizontal_fused(
        dispatch.kernels,
        &dispatch
            .kernels
            .idwt_reversible53_interleave_horizontal_fused,
        dispatch.params.width,
        dispatch.params.height,
    );
    if use_fused {
        dispatch_reversible53_interleave_horizontal_fused_single(encoder, dispatch);
    }
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
    if !use_fused {
        encoder.setComputePipelineState(&kernels.idwt_interleave);
        encoder.set_buffer(0, Some(ll), ll_offset as u64);
        encoder.set_buffer(1, Some(hl), hl_offset as u64);
        encoder.set_buffer(2, Some(lh), lh_offset as u64);
        encoder.set_buffer(3, Some(hh), hh_offset as u64);
        encoder.set_buffer(4, Some(decoded), decoded_offset as u64);
        encoder.set_bytes::<J2kIdwtSingleDecompositionParams>(5, &params);
        dispatch_2d_pipeline(
            encoder,
            &kernels.idwt_interleave,
            (params.width, params.height),
        );
        encoder.memory_barrier_with_resources(&[decoded]);

        dispatch_reversible53_horizontal_single(encoder, kernels, decoded, decoded_offset, params);
    }
    encoder.memory_barrier_with_resources(&[decoded]);

    dispatch_reversible53_vertical_single(encoder, kernels, decoded, decoded_offset, params);
}

#[cfg(target_os = "macos")]
pub(in crate::engine) fn dispatch_reversible53_repeated_buffers_in_command_buffer_with_offsets(
    command_buffers: DirectIdwtCommandBuffers<'_>,
    dispatch: RepeatedIdwtDispatch<'_>,
) -> Result<(), Error> {
    let use_fused = supports_reversible53_interleave_horizontal_fused(
        dispatch.kernels,
        &dispatch
            .kernels
            .idwt_reversible53_interleave_horizontal_fused_batched,
        dispatch.params.width,
        dispatch.params.height,
    );
    let fused_dispatch = dispatch;
    let RepeatedIdwtDispatch {
        kernels,
        sub_bands,
        params,
        decoded,
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
    let _signpost = hybrid_stage_signpost(SIGNPOST_DECODE_HYBRID_IDWT_COMMAND_ENCODE);
    let encoder = new_compute_command_encoder(command_buffers.interleave)?;
    if use_fused {
        label_compute_encoder(&encoder, "J2K decode hybrid repeated IDWT fused horizontal");
        dispatch_reversible53_interleave_horizontal_fused_repeated(&encoder, fused_dispatch);
    } else {
        label_compute_encoder(&encoder, "J2K decode hybrid repeated IDWT interleave");
        encoder.setComputePipelineState(&kernels.idwt_interleave_batched);
        encoder.set_buffer(0, Some(ll), ll_offset as u64);
        encoder.set_buffer(1, Some(hl), hl_offset as u64);
        encoder.set_buffer(2, Some(lh), lh_offset as u64);
        encoder.set_buffer(3, Some(hh), hh_offset as u64);
        encoder.set_buffer(4, Some(decoded), 0);
        encoder.set_bytes::<J2kRepeatedIdwtSingleDecompositionParams>(5, &params);
        dispatch_3d_pipeline(
            &encoder,
            &kernels.idwt_interleave_batched,
            (params.width, params.height, params.batch_count),
        );
    }
    encoder.endEncoding();

    if !use_fused {
        let encoder = new_compute_command_encoder(command_buffers.horizontal)?;
        label_compute_encoder(&encoder, "J2K decode hybrid repeated IDWT horizontal");
        dispatch_reversible53_horizontal_repeated(&encoder, kernels, decoded, params);
        encoder.endEncoding();
    }

    let encoder = new_compute_command_encoder(command_buffers.vertical)?;
    label_compute_encoder(&encoder, "J2K decode hybrid repeated IDWT vertical");
    dispatch_reversible53_vertical_repeated(&encoder, kernels, decoded, params);
    encoder.endEncoding();
    Ok(())
}

#[cfg(target_os = "macos")]
pub(in crate::engine) fn dispatch_reversible53_repeated_buffers_in_encoder_with_offsets(
    encoder: &ComputeCommandEncoderRef,
    dispatch: RepeatedIdwtDispatch<'_>,
) {
    if dispatch.params.batch_count == 0 {
        return;
    }
    if supports_fused_2d_idwt(
        dispatch.kernels,
        &dispatch.kernels.idwt_reversible53_fused_2d_batched,
        dispatch.params.width,
        dispatch.params.height,
    ) {
        dispatch_reversible53_fused_2d_repeated(encoder, dispatch);
        encoder.memory_barrier_with_resources(&[dispatch.decoded]);
        return;
    }
    let use_fused = supports_reversible53_interleave_horizontal_fused(
        dispatch.kernels,
        &dispatch
            .kernels
            .idwt_reversible53_interleave_horizontal_fused_batched,
        dispatch.params.width,
        dispatch.params.height,
    );
    if use_fused {
        dispatch_reversible53_interleave_horizontal_fused_repeated(encoder, dispatch);
    }
    let RepeatedIdwtDispatch {
        kernels,
        sub_bands,
        params,
        decoded,
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
    if !use_fused {
        encoder.setComputePipelineState(&kernels.idwt_interleave_batched);
        encoder.set_buffer(0, Some(ll), ll_offset as u64);
        encoder.set_buffer(1, Some(hl), hl_offset as u64);
        encoder.set_buffer(2, Some(lh), lh_offset as u64);
        encoder.set_buffer(3, Some(hh), hh_offset as u64);
        encoder.set_buffer(4, Some(decoded), 0);
        encoder.set_bytes::<J2kRepeatedIdwtSingleDecompositionParams>(5, &params);
        dispatch_3d_pipeline(
            encoder,
            &kernels.idwt_interleave_batched,
            (params.width, params.height, params.batch_count),
        );
        encoder.memory_barrier_with_resources(&[decoded]);

        dispatch_reversible53_horizontal_repeated(encoder, kernels, decoded, params);
    }
    encoder.memory_barrier_with_resources(&[decoded]);

    dispatch_reversible53_vertical_repeated(encoder, kernels, decoded, params);
}

/// The inputs of a last inverse wavelet level whose output a grayscale store
/// consumes directly.
#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
pub(in crate::engine) struct FusedIdwtStoreSource<'a> {
    pub(in crate::engine) sub_bands: IdwtSubBandBuffers<'a>,
    pub(in crate::engine) params: J2kIdwtSingleDecompositionParams,
    pub(in crate::engine) transform: super::J2kWaveletTransform,
}

#[cfg(target_os = "macos")]
fn fused_idwt_gray_store_pipeline(
    kernels: &crate::engine::runtime::DecodeKernels,
    transform: super::J2kWaveletTransform,
    kind: super::store::GrayStoreKind,
) -> &ComputePipelineState {
    match transform {
        super::J2kWaveletTransform::Reversible53 => &kernels.idwt53_gray_store[kind as usize],
        super::J2kWaveletTransform::Irreversible97 => &kernels.idwt97_gray_store[kind as usize],
    }
}

/// Whether `source` can run as one tile kernel writing a `kind` store.
#[cfg(target_os = "macos")]
pub(in crate::engine) fn fused_idwt_gray_store_supported(
    kernels: &crate::engine::runtime::DecodeKernels,
    source: &FusedIdwtStoreSource<'_>,
    kind: super::store::GrayStoreKind,
) -> bool {
    supports_fused_2d_idwt(
        kernels,
        fused_idwt_gray_store_pipeline(kernels, source.transform, kind),
        source.params.width,
        source.params.height,
    )
}

/// Encodes the level of `source` with each sample in the store window written
/// to `output` as the `kind` grayscale store would write it.
#[cfg(target_os = "macos")]
pub(in crate::engine) fn dispatch_fused_idwt_gray_store(
    encoder: &ComputeCommandEncoderRef,
    kernels: &crate::engine::runtime::DecodeKernels,
    source: FusedIdwtStoreSource<'_>,
    output: &crate::metal_types::BufferRef,
    output_offset_bytes: usize,
    store: &super::J2kGrayStoreParams,
    kind: super::store::GrayStoreKind,
) {
    let params = source.params;
    encoder.setComputePipelineState(fused_idwt_gray_store_pipeline(
        kernels,
        source.transform,
        kind,
    ));
    for (index, buffer, offset) in [
        (0, source.sub_bands.ll, source.sub_bands.ll_offset),
        (1, source.sub_bands.hl, source.sub_bands.hl_offset),
        (2, source.sub_bands.lh, source.sub_bands.lh_offset),
        (3, source.sub_bands.hh, source.sub_bands.hh_offset),
    ] {
        encoder.set_buffer(index, Some(buffer), offset as u64);
    }
    encoder.set_buffer(4, Some(output), output_offset_bytes as u64);
    encoder.set_bytes::<J2kIdwtSingleDecompositionParams>(5, &params);
    let tile = match source.transform {
        super::J2kWaveletTransform::Reversible53 => {
            encoder.set_bytes::<super::J2kGrayStoreParams>(6, store);
            (IDWT53_2D_TILE_W, IDWT53_2D_TILE_H)
        }
        super::J2kWaveletTransform::Irreversible97 => {
            #[cfg(test)]
            {
                // One stage sequence for the level, and one launch covering
                // each of its samples once.
                crate::engine::test_counters::record_idwt97_stage_sequence();
                crate::engine::test_counters::record_idwt97_logical_dispatch((
                    params.width,
                    params.height,
                    1,
                ));
            }
            // The single-image 9/7 route's high-pass gain.
            let high_pass = j2k_codec_math::dwt::IDWT97_OPENJPEG_TWO_INV_KAPPA_F32 * 0.5;
            encoder.set_bytes::<super::J2kIdwt97LiftSteps>(
                6,
                &irreversible::irreversible97_horizontal_lift_steps(
                    params.x0 + params.output_x,
                    high_pass,
                ),
            );
            encoder.set_bytes::<super::J2kIdwt97LiftSteps>(
                7,
                &irreversible::irreversible97_horizontal_lift_steps(
                    params.y0 + params.output_y,
                    high_pass,
                ),
            );
            encoder.set_bytes::<super::J2kGrayStoreParams>(8, store);
            (
                irreversible::IDWT97_2D_TILE_W,
                irreversible::IDWT97_2D_TILE_H,
            )
        }
    };
    encoder.dispatchThreadgroups_threadsPerThreadgroup(
        j2k_metal_support::mtl_size(
            u64::from(params.width.div_ceil(tile.0)),
            u64::from(params.height.div_ceil(tile.1)),
            1,
        ),
        j2k_metal_support::mtl_size(IDWT53_2D_LANES, IDWT53_2D_ROWS, 1),
    );
}

/// The batched last-level kernel writing a stacked grayscale group of `kind`,
/// or `None` for a kind stacked groups do not produce.
#[cfg(target_os = "macos")]
pub(in crate::engine) fn fused_idwt_stacked_gray_store_pipeline(
    kernels: &crate::engine::runtime::DecodeKernels,
    transform: super::J2kWaveletTransform,
    kind: super::store::GrayStoreKind,
) -> Option<&ComputePipelineState> {
    let pipelines = match transform {
        super::J2kWaveletTransform::Reversible53 => &kernels.idwt53_stacked_gray_store,
        super::J2kWaveletTransform::Irreversible97 => &kernels.idwt97_stacked_gray_store,
    };
    pipelines.get(kind as usize)
}

/// One level of every image of a stacked batch, stored by the batched
/// grayscale kernel into `output`.
#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
pub(in crate::engine) struct FusedStackedGrayStore<'a> {
    pub(in crate::engine) sub_bands: IdwtSubBandBuffers<'a>,
    pub(in crate::engine) params: J2kRepeatedIdwtSingleDecompositionParams,
    pub(in crate::engine) transform: super::J2kWaveletTransform,
    pub(in crate::engine) output: &'a crate::metal_types::BufferRef,
    pub(in crate::engine) output_offset_bytes: usize,
    pub(in crate::engine) store: super::J2kStackedGrayStoreParams,
}

/// Encodes one level for every image of a stacked batch with each sample
/// written to that image's range of the output as the group's component store
/// and grayscale pack would write it.
#[cfg(target_os = "macos")]
pub(in crate::engine) fn dispatch_fused_idwt_stacked_gray_store(
    encoder: &ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    request: FusedStackedGrayStore<'_>,
) {
    let FusedStackedGrayStore {
        sub_bands,
        params,
        transform,
        output,
        output_offset_bytes,
        store,
    } = request;
    encoder.setComputePipelineState(pipeline);
    for (index, buffer, offset) in [
        (0, sub_bands.ll, sub_bands.ll_offset),
        (1, sub_bands.hl, sub_bands.hl_offset),
        (2, sub_bands.lh, sub_bands.lh_offset),
        (3, sub_bands.hh, sub_bands.hh_offset),
    ] {
        encoder.set_buffer(index, Some(buffer), offset as u64);
    }
    encoder.set_buffer(4, Some(output), output_offset_bytes as u64);
    encoder.set_bytes::<J2kRepeatedIdwtSingleDecompositionParams>(5, &params);
    let tile = match transform {
        super::J2kWaveletTransform::Reversible53 => {
            encoder.set_bytes::<super::J2kStackedGrayStoreParams>(6, &store);
            (IDWT53_2D_TILE_W, IDWT53_2D_TILE_H)
        }
        super::J2kWaveletTransform::Irreversible97 => {
            #[cfg(test)]
            {
                crate::engine::test_counters::record_idwt97_stage_sequence();
                crate::engine::test_counters::record_idwt97_logical_dispatch((
                    params.width,
                    params.height,
                    params.batch_count,
                ));
            }
            // The batched 9/7 route's high-pass gain.
            let high_pass = j2k_codec_math::dwt::IDWT97_OPENJPEG_TWO_INV_KAPPA_F32 * 0.5;
            encoder.set_bytes::<super::J2kIdwt97LiftSteps>(
                6,
                &irreversible::irreversible97_horizontal_lift_steps(
                    params.x0 + params.output_x,
                    high_pass,
                ),
            );
            encoder.set_bytes::<super::J2kIdwt97LiftSteps>(
                7,
                &irreversible::irreversible97_horizontal_lift_steps(
                    params.y0 + params.output_y,
                    high_pass,
                ),
            );
            encoder.set_bytes::<super::J2kStackedGrayStoreParams>(8, &store);
            (
                irreversible::IDWT97_2D_TILE_W,
                irreversible::IDWT97_2D_TILE_H,
            )
        }
    };
    encoder.dispatchThreadgroups_threadsPerThreadgroup(
        j2k_metal_support::mtl_size(
            u64::from(params.width.div_ceil(tile.0)),
            u64::from(params.height.div_ceil(tile.1)),
            u64::from(params.batch_count),
        ),
        j2k_metal_support::mtl_size(IDWT53_2D_LANES, IDWT53_2D_ROWS, 1),
    );
}
