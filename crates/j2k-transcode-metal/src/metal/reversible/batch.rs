// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::{
    buffer_with_slice, checked_batch_len, checked_command_buffer, checked_compute_command_encoder,
    commit_and_wait, dispatch_reversible_band, output_i32_buffer, read_i32_buffer_at,
    reversible_band_geometry, try_transcode_vec_with_capacity, u32_param, Buffer, MetalRuntime,
    MetalTranscodeError, ReversibleBatchKernelGeometry, ReversibleDwt53FirstLevel,
    METAL_REVERSIBLE_DCT53_UNSUPPORTED_GRID,
};

pub(super) fn dispatch_with_runtime(
    runtime: &MetalRuntime,
    block_samples: &[[i32; 64]],
    batch_count: usize,
    block_cols: usize,
    width: usize,
    height: usize,
) -> Result<Vec<ReversibleDwt53FirstLevel>, MetalTranscodeError> {
    if batch_count == 0 {
        return Ok(Vec::new());
    }
    if !block_samples.len().is_multiple_of(batch_count) {
        return Err(MetalTranscodeError::UnsupportedJob(
            METAL_REVERSIBLE_DCT53_UNSUPPORTED_GRID,
        ));
    }

    let (kernel_geometry, output_shape) = reversible_batch_shapes(
        block_samples.len() / batch_count,
        batch_count,
        block_cols,
        width,
        height,
    )?;
    let blocks = buffer_with_slice(&runtime.device, block_samples)?;
    let output_buffers = reversible_output_buffers(runtime, output_shape)?;
    dispatch_reversible_projection(
        runtime,
        &blocks,
        &output_buffers,
        kernel_geometry,
        output_shape,
    )?;
    read_reversible_batch_outputs(output_buffers.as_refs(), output_shape)
}

#[derive(Clone, Copy)]
struct ReversibleBatchOutputShape {
    low_width: usize,
    low_height: usize,
    high_width: usize,
    high_height: usize,
    ll_len: usize,
    hl_len: usize,
    lh_len: usize,
    hh_len: usize,
    batch_count: usize,
}

fn reversible_batch_shapes(
    blocks_per_item: usize,
    batch_count: usize,
    block_cols: usize,
    width: usize,
    height: usize,
) -> Result<(ReversibleBatchKernelGeometry, ReversibleBatchOutputShape), MetalTranscodeError> {
    let kernel = ReversibleBatchKernelGeometry {
        width: u32_param(width, METAL_REVERSIBLE_DCT53_UNSUPPORTED_GRID)?,
        height: u32_param(height, METAL_REVERSIBLE_DCT53_UNSUPPORTED_GRID)?,
        block_cols: u32_param(block_cols, METAL_REVERSIBLE_DCT53_UNSUPPORTED_GRID)?,
        blocks_per_item: u32_param(blocks_per_item, METAL_REVERSIBLE_DCT53_UNSUPPORTED_GRID)?,
        batch_count: u32_param(batch_count, METAL_REVERSIBLE_DCT53_UNSUPPORTED_GRID)?,
    };
    let low_width = width.div_ceil(2);
    let high_width = width / 2;
    let low_height = height.div_ceil(2);
    let high_height = height / 2;
    let band_len = |band_width: usize, band_height: usize| {
        band_width
            .checked_mul(band_height)
            .ok_or(MetalTranscodeError::UnsupportedJob(
                METAL_REVERSIBLE_DCT53_UNSUPPORTED_GRID,
            ))
    };
    Ok((
        kernel,
        ReversibleBatchOutputShape {
            low_width,
            low_height,
            high_width,
            high_height,
            ll_len: band_len(low_width, low_height)?,
            hl_len: band_len(high_width, low_height)?,
            lh_len: band_len(low_width, high_height)?,
            hh_len: band_len(high_width, high_height)?,
            batch_count,
        },
    ))
}

struct ReversibleOwnedOutputBuffers {
    ll: Buffer,
    hl: Buffer,
    lh: Buffer,
    hh: Buffer,
}

impl ReversibleOwnedOutputBuffers {
    fn as_refs(&self) -> ReversibleOutputBuffers<'_> {
        ReversibleOutputBuffers {
            ll: &self.ll,
            hl: &self.hl,
            lh: &self.lh,
            hh: &self.hh,
        }
    }
}

fn reversible_output_buffers(
    runtime: &MetalRuntime,
    shape: ReversibleBatchOutputShape,
) -> Result<ReversibleOwnedOutputBuffers, MetalTranscodeError> {
    let allocate = |value_count| {
        output_i32_buffer(
            &runtime.device,
            checked_batch_len(
                value_count,
                shape.batch_count,
                METAL_REVERSIBLE_DCT53_UNSUPPORTED_GRID,
            )?,
        )
    };
    Ok(ReversibleOwnedOutputBuffers {
        ll: allocate(shape.ll_len)?,
        hl: allocate(shape.hl_len)?,
        lh: allocate(shape.lh_len)?,
        hh: allocate(shape.hh_len)?,
    })
}

fn dispatch_reversible_projection(
    runtime: &MetalRuntime,
    blocks: &Buffer,
    buffers: &ReversibleOwnedOutputBuffers,
    kernel: ReversibleBatchKernelGeometry,
    shape: ReversibleBatchOutputShape,
) -> Result<(), MetalTranscodeError> {
    let command_buffer = checked_command_buffer(&runtime.queue).map_err(|error| {
        MetalTranscodeError::support("Metal reversible 5/3 command buffer creation", error)
    })?;
    command_buffer.set_label("j2k-transcode-metal reversible dct53 projection");
    let encoder = checked_compute_command_encoder(&command_buffer).map_err(|error| {
        MetalTranscodeError::support("Metal reversible 5/3 compute encoder creation", error)
    })?;
    encoder.set_compute_pipeline_state(&runtime.reversible53_project_band);
    encoder.set_buffer(0, Some(blocks), 0);

    let dispatch = |output: &Buffer,
                    width,
                    height,
                    stride,
                    vertical_low,
                    horizontal_low|
     -> Result<(), MetalTranscodeError> {
        dispatch_reversible_band(
            &encoder,
            output,
            reversible_band_geometry(kernel, width, height, stride, vertical_low, horizontal_low)?,
        );
        Ok(())
    };
    dispatch(
        &buffers.ll,
        shape.low_width,
        shape.low_height,
        shape.ll_len,
        true,
        true,
    )?;
    dispatch(
        &buffers.hl,
        shape.high_width,
        shape.low_height,
        shape.hl_len,
        true,
        false,
    )?;
    dispatch(
        &buffers.lh,
        shape.low_width,
        shape.high_height,
        shape.lh_len,
        false,
        true,
    )?;
    dispatch(
        &buffers.hh,
        shape.high_width,
        shape.high_height,
        shape.hh_len,
        false,
        false,
    )?;
    encoder.end_encoding();
    commit_and_wait(&command_buffer)
        .map_err(|error| MetalTranscodeError::support("Metal reversible 5/3 command buffer", error))
}

#[derive(Clone, Copy)]
struct ReversibleOutputBuffers<'a> {
    ll: &'a Buffer,
    hl: &'a Buffer,
    lh: &'a Buffer,
    hh: &'a Buffer,
}

fn read_reversible_batch_outputs(
    buffers: ReversibleOutputBuffers<'_>,
    shape: ReversibleBatchOutputShape,
) -> Result<Vec<ReversibleDwt53FirstLevel>, MetalTranscodeError> {
    let mut outputs =
        try_transcode_vec_with_capacity(shape.batch_count, "reversible 5/3 batch output metadata")?;
    for idx in 0..shape.batch_count {
        outputs.push(ReversibleDwt53FirstLevel {
            ll: read_i32_buffer_at(
                buffers.ll,
                checked_batch_len(shape.ll_len, idx, METAL_REVERSIBLE_DCT53_UNSUPPORTED_GRID)?,
                shape.ll_len,
            )?,
            hl: read_i32_buffer_at(
                buffers.hl,
                checked_batch_len(shape.hl_len, idx, METAL_REVERSIBLE_DCT53_UNSUPPORTED_GRID)?,
                shape.hl_len,
            )?,
            lh: read_i32_buffer_at(
                buffers.lh,
                checked_batch_len(shape.lh_len, idx, METAL_REVERSIBLE_DCT53_UNSUPPORTED_GRID)?,
                shape.lh_len,
            )?,
            hh: read_i32_buffer_at(
                buffers.hh,
                checked_batch_len(shape.hh_len, idx, METAL_REVERSIBLE_DCT53_UNSUPPORTED_GRID)?,
                shape.hh_len,
            )?,
            low_width: shape.low_width,
            low_height: shape.low_height,
            high_width: shape.high_width,
            high_height: shape.high_height,
        });
    }
    Ok(outputs)
}
