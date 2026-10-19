// SPDX-License-Identifier: MIT OR Apache-2.0

//! Metal lossless (SOF3) JPEG encode.
//!
//! Metal classifies every sample, prefix-sums code lengths, and packs the
//! bit stream (`shaders_lossless_encode.metal`); the host builds the
//! Huffman table from the device histogram between the two submissions,
//! then copies the packed stream out with byte stuffing and restart markers.
//! Output equals CPU lossless encode byte for byte.

use j2k_jpeg::adapter::{
    assemble_jpeg_lossless_frame, finish_jpeg_lossless_gpu_entropy, jpeg_lossless_huffman_table,
    JpegLosslessGpuEncodeParams, JpegLosslessGpuEncodePlan,
};
use j2k_jpeg::{EncodedJpeg, JpegBackend, JpegEncodeError};

use super::{
    commit_and_wait_jpeg, dispatch_1d_pipeline, new_command_buffer, new_compute_command_encoder,
    runtime::MetalRuntime, with_runtime_for_session,
};
use crate::abi::{JpegLosslessEncodeParams, JpegLosslessEncodeStatus, JpegLosslessEncodeTable};
use crate::buffers::{
    checked_buffer_borrow_slice_at, checked_buffer_read, checked_buffer_slice, MetalBatchScratch,
};
use crate::metal_types::prelude::*;
use crate::metal_types::{
    Buffer, BufferRef, ComputeCommandEncoderRef, ComputePipelineState, DeviceRef,
};
use crate::Error;

/// Difference categories 0..=16.
const CATEGORIES: usize = 17;

/// Encode `input` as planned by `plan`, which the caller validated against
/// the buffer.
pub(crate) fn encode_jpeg_lossless_with_session(
    session: &crate::MetalBackendSession,
    input: &BufferRef,
    plan: &JpegLosslessGpuEncodePlan,
) -> Result<EncodedJpeg, Error> {
    with_runtime_for_session(session, |runtime| encode(runtime, input, plan))
}

fn encode(
    runtime: &MetalRuntime,
    input: &BufferRef,
    plan: &JpegLosslessGpuEncodePlan,
) -> Result<EncodedJpeg, Error> {
    let params = abi_params(&plan.params);
    let device = &runtime.device;
    let mut scratch = runtime.batch_scratch()?;
    let tokens = scratch.private_buffer(
        device,
        "jpeg_lossless_encode_tokens",
        words(params.sample_count)?,
    )?;
    let status = scratch.shared_buffer_with_slice(
        device,
        "jpeg_lossless_encode_status",
        &[JpegLosslessEncodeStatus {
            bad_index: u32::MAX,
            ..JpegLosslessEncodeStatus::default()
        }],
    )?;
    let category_counts = classify(
        runtime,
        &mut scratch,
        input,
        [&tokens, &status],
        &params,
        plan,
    )?;
    let table = jpeg_lossless_huffman_table(&category_counts)?;
    let codes = table.codes();
    let mut device_table = JpegLosslessEncodeTable::default();
    for (category, entry) in device_table.entries.iter_mut().enumerate() {
        *entry = (u32::from(codes.codes[category]) << 8) | u32::from(codes.lens[category]);
    }
    let stages = PackBuffers::new(&mut scratch, device, &params)?;
    pack(runtime, &stages, [&tokens, &status], &params, &device_table)?;

    let packed_bytes =
        checked_buffer_read::<JpegLosslessEncodeStatus>(&status, "lossless encode status")?
            .packed_bytes as usize;
    if packed_bytes > params.packed_words as usize * 4 {
        return Err(Error::MetalKernel {
            message: "JPEG lossless Metal encode reported more bytes than it packed".to_string(),
        });
    }
    let lengths = checked_buffer_slice::<u32>(
        &stages.interval_bytes,
        params.intervals as usize,
        "lossless encode interval lengths",
    )?;
    // SAFETY: the pack submission completed in `commit_and_wait_jpeg`, and no
    // CPU or GPU writer touches the leased packed buffer while it is borrowed.
    let packed = unsafe {
        checked_buffer_borrow_slice_at(&stages.packed, 0, packed_bytes, "lossless packed stream")
    }?;
    let entropy = finish_jpeg_lossless_gpu_entropy(packed, &lengths)?;
    Ok(assemble_jpeg_lossless_frame(
        &plan.plan,
        &table,
        &entropy,
        JpegBackend::Metal,
    )?)
}

/// Stage 1: tokens and the category histogram, or the CPU encoder's error
/// for the first out-of-range sample.
fn classify(
    runtime: &MetalRuntime,
    scratch: &mut MetalBatchScratch,
    input: &BufferRef,
    [tokens, status]: [&BufferRef; 2],
    params: &JpegLosslessEncodeParams,
    plan: &JpegLosslessGpuEncodePlan,
) -> Result<[u64; CATEGORIES], Error> {
    let histogram = scratch.shared_zeroed_buffer(
        &runtime.device,
        "jpeg_lossless_encode_histogram",
        CATEGORIES * 4,
    )?;
    let command_buffer = new_command_buffer(&runtime.queue)?;
    let encoder = new_compute_command_encoder(&command_buffer)?;
    dispatch(
        &encoder,
        &runtime.pipelines.lossless_encode.classify,
        params.sample_count,
        |e| {
            e.bind_buffer(0, Some(input), 0);
            e.bind_buffer(1, Some(tokens), 0);
            e.bind_buffer(2, Some(&histogram), 0);
            e.bind_buffer(3, Some(status), 0);
            e.bind_bytes(4, params);
        },
    );
    encoder.endEncoding();
    commit_and_wait_jpeg(&command_buffer)?;
    let bad_index =
        checked_buffer_read::<JpegLosslessEncodeStatus>(status, "lossless encode status")?
            .bad_index;
    if bad_index != u32::MAX {
        return Err(out_of_range_error(
            runtime, input, status, params, plan, bad_index,
        ));
    }
    let counts = checked_buffer_slice::<u32>(&histogram, CATEGORIES, "lossless encode histogram")?;
    let mut category_counts = [0_u64; CATEGORIES];
    for (count, &device_count) in category_counts.iter_mut().zip(&counts) {
        *count = u64::from(device_count);
    }
    Ok(category_counts)
}

/// Stages 3-5 in one submission; serial dispatches, so each stage reads the
/// previous stage's output.
fn pack(
    runtime: &MetalRuntime,
    stages: &PackBuffers,
    [tokens, status]: [&BufferRef; 2],
    params: &JpegLosslessEncodeParams,
    table: &JpegLosslessEncodeTable,
) -> Result<(), Error> {
    let pipelines = &runtime.pipelines.lossless_encode;
    let chunks = params.chunks_per_row * params.height;
    let command_buffer = new_command_buffer(&runtime.queue)?;
    let encoder = new_compute_command_encoder(&command_buffer)?;
    dispatch(&encoder, &pipelines.chunk_bits, chunks, |e| {
        e.bind_buffer(0, Some(tokens), 0);
        e.bind_buffer(1, Some(&stages.chunk_bits), 0);
        e.bind_bytes(2, params);
        e.bind_bytes(3, table);
    });
    dispatch(&encoder, &pipelines.row_offsets, params.height, |e| {
        e.bind_buffer(0, Some(&stages.chunk_bits), 0);
        e.bind_buffer(1, Some(&stages.chunk_offsets), 0);
        e.bind_buffer(2, Some(&stages.row_bits), 0);
        e.bind_bytes(3, params);
    });
    dispatch(
        &encoder,
        &pipelines.interval_offsets,
        params.intervals,
        |e| {
            e.bind_buffer(0, Some(&stages.row_bits), 0);
            e.bind_buffer(1, Some(&stages.row_offsets), 0);
            e.bind_buffer(2, Some(&stages.interval_bits), 0);
            e.bind_bytes(3, params);
        },
    );
    dispatch(&encoder, &pipelines.interval_bases, 1, |e| {
        e.bind_buffer(0, Some(&stages.interval_bits), 0);
        e.bind_buffer(1, Some(&stages.interval_bases), 0);
        e.bind_buffer(2, Some(&stages.interval_bytes), 0);
        e.bind_buffer(3, Some(status), 0);
        e.bind_bytes(4, params);
    });
    dispatch(&encoder, &pipelines.clear, params.packed_words, |e| {
        e.bind_buffer(0, Some(&stages.packed), 0);
        e.bind_buffer(1, Some(status), 0);
        e.bind_bytes(2, params);
    });
    dispatch(&encoder, &pipelines.pack, chunks, |e| {
        e.bind_buffer(0, Some(tokens), 0);
        e.bind_buffer(1, Some(&stages.packed), 0);
        e.bind_buffer(2, Some(&stages.chunk_offsets), 0);
        e.bind_buffer(3, Some(&stages.row_offsets), 0);
        e.bind_buffer(4, Some(&stages.interval_bases), 0);
        e.bind_buffer(5, Some(&stages.interval_bits), 0);
        e.bind_bytes(6, params);
        e.bind_bytes(7, table);
    });
    encoder.endEncoding();
    commit_and_wait_jpeg(&command_buffer)
}

/// Device buffers of the offset and pack stages.
struct PackBuffers {
    chunk_bits: Buffer,
    chunk_offsets: Buffer,
    row_bits: Buffer,
    row_offsets: Buffer,
    interval_bits: Buffer,
    interval_bases: Buffer,
    interval_bytes: Buffer,
    packed: Buffer,
}

impl PackBuffers {
    fn new(
        scratch: &mut MetalBatchScratch,
        device: &DeviceRef,
        params: &JpegLosslessEncodeParams,
    ) -> Result<Self, Error> {
        let chunks = words(params.chunks_per_row * params.height)?;
        let rows = words(params.height)?;
        let intervals = words(params.intervals)?;
        Ok(Self {
            chunk_bits: scratch.private_buffer(
                device,
                "jpeg_lossless_encode_chunk_bits",
                chunks,
            )?,
            chunk_offsets: scratch.private_buffer(
                device,
                "jpeg_lossless_encode_chunk_offsets",
                chunks,
            )?,
            row_bits: scratch.private_buffer(device, "jpeg_lossless_encode_row_bits", rows)?,
            row_offsets: scratch.private_buffer(
                device,
                "jpeg_lossless_encode_row_offsets",
                rows,
            )?,
            interval_bits: scratch.private_buffer(
                device,
                "jpeg_lossless_encode_interval_bits",
                intervals,
            )?,
            interval_bases: scratch.private_buffer(
                device,
                "jpeg_lossless_encode_interval_bases",
                intervals,
            )?,
            interval_bytes: scratch.shared_buffer(
                device,
                "jpeg_lossless_encode_interval_bytes",
                intervals,
            )?,
            packed: scratch.shared_buffer(
                device,
                "jpeg_lossless_encode_packed",
                words(params.packed_words)?,
            )?,
        })
    }
}

fn dispatch(
    encoder: &ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    threads: u32,
    bind: impl FnOnce(&ComputeCommandEncoderRef),
) {
    encoder.setComputePipelineState(pipeline);
    bind(encoder);
    dispatch_1d_pipeline(encoder, pipeline, threads);
}

/// Read the first out-of-range sample's value for the CPU encoder's error.
fn out_of_range_error(
    runtime: &MetalRuntime,
    input: &BufferRef,
    status: &BufferRef,
    params: &JpegLosslessEncodeParams,
    plan: &JpegLosslessGpuEncodePlan,
    index: u32,
) -> Error {
    let read = || -> Result<u32, Error> {
        let command_buffer = new_command_buffer(&runtime.queue)?;
        let encoder = new_compute_command_encoder(&command_buffer)?;
        dispatch(
            &encoder,
            &runtime.pipelines.lossless_encode.read_sample,
            1,
            |e| {
                e.bind_buffer(0, Some(input), 0);
                e.bind_buffer(1, Some(status), 0);
                e.bind_bytes(2, params);
            },
        );
        encoder.endEncoding();
        commit_and_wait_jpeg(&command_buffer)?;
        Ok(
            checked_buffer_read::<JpegLosslessEncodeStatus>(status, "lossless encode status")?
                .bad_value,
        )
    };
    match read() {
        Ok(value) => JpegEncodeError::SampleOutOfRange {
            index: index as usize,
            value: u16::try_from(value).unwrap_or(u16::MAX),
            precision: plan.plan.precision,
        }
        .into(),
        Err(error) => error,
    }
}

fn words(count: u32) -> Result<usize, Error> {
    (count as usize)
        .checked_mul(4)
        .ok_or_else(|| Error::MetalKernel {
            message: "JPEG lossless Metal encode buffer size overflowed".to_string(),
        })
}

fn abi_params(params: &JpegLosslessGpuEncodeParams) -> JpegLosslessEncodeParams {
    JpegLosslessEncodeParams {
        width: params.width,
        height: params.height,
        components: params.components,
        sample_bytes: params.sample_bytes,
        pitch_bytes: params.pitch_bytes,
        input_offset_bytes: params.input_offset_bytes,
        predictor: params.predictor,
        point_transform: params.point_transform,
        initial_predictor: params.initial_predictor,
        max_sample: params.max_sample,
        interval_rows: params.interval_rows,
        chunk_pixels: params.chunk_pixels,
        chunks_per_row: params.chunks_per_row,
        intervals: params.intervals,
        sample_count: params.sample_count,
        packed_words: params.packed_words,
    }
}
