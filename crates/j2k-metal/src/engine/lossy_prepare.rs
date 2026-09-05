// SPDX-License-Identifier: MIT OR Apache-2.0

use super::abi::{
    J2kFusedInputMctParams, J2kHtEncodeBatchJob, J2kHtEncodeStatus, J2kLosslessDeinterleaveParams,
    J2kLossyCoefficientJob, J2kQuantizeSubbandParams,
};
use super::{
    checked_buffer_slice, commit_and_wait_metal, copied_slice_buffer, ht_encode_output_capacity,
    new_command_buffer, new_compute_command_encoder, new_shared_buffer,
    take_recyclable_private_buffer, with_runtime, zeroed_shared_buffer, Buffer, Error,
    J2kLosslessDeviceCodeBlock, MetalRuntime,
};
use crate::metal_types::prelude::*;
use j2k_metal_support::{dispatch_1d_pipeline, dispatch_2d_pipeline, dispatch_3d_pipeline};
use j2k_native::{EncodedHtJ2kCodeBlock, J2kHtj2kTileEncodeJob, J2kSubBandType};

struct LossyJobs {
    quantize: Vec<J2kLossyCoefficientJob>,
    ht: Vec<J2kHtEncodeBatchJob>,
    coefficient_count: usize,
    output_bytes: usize,
    decomposition_levels: Vec<u8>,
}

fn plan_jobs(
    job: J2kHtj2kTileEncodeJob<'_>,
    blocks: &[J2kLosslessDeviceCodeBlock],
    steps: &[(u16, u16, u8)],
) -> Result<LossyJobs, Error> {
    if blocks.len() != steps.len() {
        return Err(layout_error());
    }
    let mut budget = crate::batch_allocation::BatchMetadataBudget::new("Metal resident lossy jobs");
    let mut quantize = budget.try_vec(blocks.len(), "Metal lossy quantization descriptors")?;
    let mut ht = budget.try_vec(blocks.len(), "Metal lossy HT descriptors")?;
    let mut coefficient_count = 0usize;
    let mut output_bytes = 0usize;
    let mut decomposition_levels =
        budget.try_vec(blocks.len(), "Metal lossy decomposition levels")?;
    for (block, &(exponent, mantissa, level)) in blocks.iter().zip(steps) {
        decomposition_levels.push(level);
        let width = block.width;
        let height = block.height;
        let source_x = block
            .subband_x
            .checked_add(block.block_x)
            .ok_or_else(layout_error)?;
        let source_y = block
            .subband_y
            .checked_add(block.block_y)
            .ok_or_else(layout_error)?;
        if block.component >= u32::from(job.num_components)
            || width == 0
            || height == 0
            || source_x
                .checked_add(width)
                .is_none_or(|end| end > job.width)
            || source_y
                .checked_add(height)
                .is_none_or(|end| end > job.height)
        {
            return Err(layout_error());
        }
        let count = width.checked_mul(height).ok_or_else(layout_error)?;
        coefficient_count = coefficient_count.max(
            (block.coefficient_offset as usize)
                .checked_add(count as usize)
                .ok_or_else(layout_error)?,
        );
        let gain = match block.sub_band_type {
            J2kSubBandType::LowLow => 0,
            J2kSubBandType::LowHigh | J2kSubBandType::HighLow => 1,
            J2kSubBandType::HighHigh => 2,
        };
        quantize.push(J2kLossyCoefficientJob {
            coefficient_offset: block.coefficient_offset,
            component: block.component,
            source_x,
            source_y,
            width,
            height,
            full_width: job.width,
            quantize: J2kQuantizeSubbandParams {
                _len: count,
                _step_exponent: u32::from(exponent),
                _step_mantissa: u32::from(mantissa),
                _range_bits: u32::from(job.bit_depth) + gain,
                _reversible: 0,
                _reserved0: 0,
                _reserved1: 0,
                _reserved2: 0,
            },
        });
        let capacity = ht_encode_output_capacity(width, height)?;
        ht.push(J2kHtEncodeBatchJob {
            coefficient_offset: block.coefficient_offset,
            output_offset: u32::try_from(output_bytes).map_err(|_| layout_error())?,
            width,
            height,
            total_bitplanes: u32::from(block.total_bitplanes),
            cleanup_bitplane: 0,
            target_coding_passes: 1,
            output_capacity: u32::try_from(capacity).map_err(|_| layout_error())?,
        });
        output_bytes = output_bytes
            .checked_add(capacity)
            .ok_or_else(layout_error)?;
    }
    Ok(LossyJobs {
        quantize,
        ht,
        coefficient_count,
        output_bytes,
        decomposition_levels,
    })
}

fn layout_error() -> Error {
    Error::MetalKernel {
        message: "Metal resident lossy geometry or allocation overflow".to_owned(),
    }
}

pub(crate) fn encode_resident_lossy_ht_blocks(
    job: J2kHtj2kTileEncodeJob<'_>,
    blocks: &[J2kLosslessDeviceCodeBlock],
    steps: &[(u16, u16, u8)],
) -> Result<(Vec<EncodedHtJ2kCodeBlock>, u8), Error> {
    let jobs = plan_jobs(job, blocks, steps)?;
    with_runtime(|runtime| execute(runtime, job, &jobs))
}

fn execute(
    runtime: &MetalRuntime,
    job: J2kHtj2kTileEncodeJob<'_>,
    jobs: &LossyJobs,
) -> Result<(Vec<EncodedHtJ2kCodeBlock>, u8), Error> {
    let plane_bytes = (job.width as usize)
        .checked_mul(job.height as usize)
        .and_then(|n| n.checked_mul(size_of::<f32>()))
        .ok_or_else(layout_error)?;
    let mut retained = Vec::new();
    let mut planes = Vec::new();
    let mut scratches = Vec::new();
    for _ in 0..job.num_components {
        planes.push(take_recyclable_private_buffer(
            runtime,
            plane_bytes,
            &mut retained,
        )?);
        scratches.push(take_recyclable_private_buffer(
            runtime,
            plane_bytes,
            &mut retained,
        )?);
    }
    let coefficients = take_recyclable_private_buffer(
        runtime,
        jobs.coefficient_count
            .checked_mul(size_of::<i32>())
            .ok_or_else(layout_error)?
            .max(1),
        &mut retained,
    )?;
    let input = copied_slice_buffer(&runtime.device, job.pixels)?;
    let quantize_jobs = copied_slice_buffer(&runtime.device, &jobs.quantize)?;
    let ht_jobs = copied_slice_buffer(&runtime.device, &jobs.ht)?;
    let output = new_shared_buffer(&runtime.device, jobs.output_bytes.max(1))?;
    let status_buffer = zeroed_shared_buffer(
        &runtime.device,
        jobs.ht
            .len()
            .checked_mul(size_of::<J2kHtEncodeStatus>())
            .ok_or_else(layout_error)?
            .max(1),
    )?;
    let command = new_command_buffer(&runtime.queue)?;
    encode_input(runtime, &command, &input, &planes, job)?;
    let mut transformed = Vec::new();
    for (plane, scratch) in planes.iter().zip(&scratches) {
        let layout = super::forward_transform::encode_forward_dwt97_commands(
            runtime,
            &command,
            (plane, scratch),
            (job.width, job.height),
            job.num_decomposition_levels,
        )?;
        transformed.push(if layout.active_is_a {
            plane.clone()
        } else {
            scratch.clone()
        });
    }
    let count = u32::try_from(jobs.ht.len()).map_err(|_| layout_error())?;
    let encoder = new_compute_command_encoder(&command)?;
    let kernel = &runtime.encode()?.lossy_extract_quantized_coefficients;
    encoder.setComputePipelineState(kernel);
    for index in 0usize..3 {
        encoder.set_buffer(
            index as u64,
            Some(transformed.get(index).unwrap_or(&transformed[0])),
            0,
        );
    }
    encoder.set_buffer(3, Some(&coefficients), 0);
    encoder.set_buffer(4, Some(&quantize_jobs), 0);
    encoder.set_bytes(5, &count);
    dispatch_3d_pipeline(
        &encoder,
        kernel,
        (job.code_block_width, job.code_block_height, count),
    );
    encoder.endEncoding();
    let encoder = new_compute_command_encoder(&command)?;
    let kernels = runtime.encode()?;
    encoder.setComputePipelineState(&kernels.ht_encode_code_blocks);
    for (index, buffer) in [
        &coefficients,
        &output,
        &ht_jobs,
        &kernels.ht_vlc_encode_table0,
        &kernels.ht_vlc_encode_table1,
        &kernels.ht_uvlc_encode_table,
        &status_buffer,
    ]
    .into_iter()
    .enumerate()
    {
        encoder.set_buffer(index as u64, Some(buffer), 0);
    }
    encoder.set_bytes(7, &count);
    dispatch_1d_pipeline(&encoder, &kernels.ht_encode_code_blocks, u64::from(count));
    encoder.endEncoding();
    // All input, output and pooled intermediate owners remain live until this
    // completion boundary, including the error path of commit_and_wait_metal.
    commit_and_wait_metal(&command)?;
    read_lossy_blocks(jobs, &status_buffer, &output)
}

fn read_lossy_blocks(
    jobs: &LossyJobs,
    status_buffer: &Buffer,
    output: &Buffer,
) -> Result<(Vec<EncodedHtJ2kCodeBlock>, u8), Error> {
    let statuses = checked_buffer_slice::<J2kHtEncodeStatus>(
        status_buffer,
        jobs.ht.len(),
        "Metal lossy HT statuses",
    )?;
    let mut encoded =
        crate::batch_allocation::try_vec(jobs.ht.len(), "Metal lossy HT output blocks")?;
    let mut bound = 8;
    for ((job, status), level) in jobs
        .ht
        .iter()
        .zip(statuses.iter().copied())
        .zip(jobs.decomposition_levels.iter().copied())
    {
        encoded.push(super::tier1_encode::read_ht_encoded_code_block(
            status,
            output,
            job.output_offset as usize,
            job.output_capacity as usize,
        )?);
        bound = bound.max(j2k_native::htj2k_required_magnitude_bound(
            u64::from(status.detail),
            false,
            level,
        ));
    }
    Ok((encoded, bound.min(74)))
}

fn encode_input(
    runtime: &MetalRuntime,
    command: &super::CommandBufferRef,
    input: &Buffer,
    planes: &[Buffer],
    job: J2kHtj2kTileEncodeJob<'_>,
) -> Result<(), Error> {
    let encoder = new_compute_command_encoder(command)?;
    encoder.set_buffer(0, Some(input), 0);
    for index in 0usize..3 {
        encoder.set_buffer(
            index as u64 + 1,
            Some(planes.get(index).unwrap_or(&planes[0])),
            0,
        );
    }
    if job.use_mct {
        let params = J2kFusedInputMctParams {
            len: job.width * job.height,
            bytes_per_sample: 1,
            bit_depth: 8,
            sample_offset: 128,
            signed_samples: 0,
            reversible: 0,
        };
        encoder.setComputePipelineState(&runtime.encode()?.encode_deinterleave_mct);
        encoder.set_bytes(4, &params);
        dispatch_1d_pipeline(
            &encoder,
            &runtime.encode()?.encode_deinterleave_mct,
            u64::from(params.len),
        );
    } else {
        let params = J2kLosslessDeinterleaveParams {
            src_width: job.width,
            src_height: job.height,
            src_stride: job.width,
            dst_width: job.width,
            dst_height: job.height,
            components: 1,
            bytes_per_sample: 1,
            bit_depth: 8,
            sample_offset: 128,
            signed_samples: 0,
        };
        encoder.setComputePipelineState(&runtime.encode()?.lossless_deinterleave_to_planes);
        encoder.set_buffer(5, Some(&planes[0]), 0);
        encoder.set_bytes(4, &params);
        dispatch_2d_pipeline(
            &encoder,
            &runtime.encode()?.lossless_deinterleave_to_planes,
            (job.width, job.height),
        );
    }
    encoder.endEncoding();
    Ok(())
}
