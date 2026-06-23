// SPDX-License-Identifier: MIT OR Apache-2.0

use std::mem::size_of;

use j2k_core::PixelFormat;
use metal::{Buffer, ComputePipelineState, MTLSize};

use super::{
    PlaneMode, PreparedHuffmanHost, MODE_GRAY, MODE_RGB, MODE_YCBCR, OUT_GRAY, OUT_RGB, OUT_RGBA,
};

#[cfg(target_os = "macos")]
#[cfg(target_os = "macos")]
/// Bind the shared fast-decode entropy kernel inputs at slots 0-16: entropy
/// bytes, the three component planes, the family params struct, the three
/// quantization tables, the per-component DC/AC Huffman table pairs, restart
/// offsets, decode status, and entropy checkpoints.
#[allow(clippy::too_many_arguments)]
pub(super) fn bind_fast_decode_entropy_inputs<P>(
    encoder: &metal::ComputeCommandEncoderRef,
    entropy_buffer: &Buffer,
    planes: [&Buffer; 3],
    params: &P,
    quants: [&[u16; 64]; 3],
    dc_tables: &[PreparedHuffmanHost; 3],
    ac_tables: &[PreparedHuffmanHost; 3],
    restart_offsets_buffer: &Buffer,
    status_buffer: &Buffer,
    entropy_checkpoints_buffer: &Buffer,
) {
    encoder.set_buffer(0, Some(entropy_buffer), 0);
    encoder.set_buffer(1, Some(planes[0]), 0);
    encoder.set_buffer(2, Some(planes[1]), 0);
    encoder.set_buffer(3, Some(planes[2]), 0);
    encoder.set_bytes(4, size_of::<P>() as u64, (&raw const *params).cast());
    for (slot, quant) in (5u64..).zip(quants) {
        encoder.set_bytes(slot, size_of::<[u16; 64]>() as u64, quant.as_ptr().cast());
    }
    for (index, (dc, ac)) in dc_tables.iter().zip(ac_tables.iter()).enumerate() {
        let slot = 8 + 2 * index as u64;
        encoder.set_bytes(
            slot,
            size_of::<PreparedHuffmanHost>() as u64,
            (&raw const *dc).cast(),
        );
        encoder.set_bytes(
            slot + 1,
            size_of::<PreparedHuffmanHost>() as u64,
            (&raw const *ac).cast(),
        );
    }
    encoder.set_buffer(14, Some(restart_offsets_buffer), 0);
    encoder.set_buffer(15, Some(status_buffer), 0);
    encoder.set_buffer(16, Some(entropy_checkpoints_buffer), 0);
}

/// Bind the shared three-plane pack kernel layout at slots 0-4: the component
/// planes, the packed output buffer, and the pack params struct.
pub(super) fn bind_three_plane_pack<P>(
    encoder: &metal::ComputeCommandEncoderRef,
    planes: [Option<&Buffer>; 3],
    out_buffer: &Buffer,
    params: &P,
) {
    encoder.set_buffer(0, planes[0].map(std::convert::AsRef::as_ref), 0);
    encoder.set_buffer(1, planes[1].map(std::convert::AsRef::as_ref), 0);
    encoder.set_buffer(2, planes[2].map(std::convert::AsRef::as_ref), 0);
    encoder.set_buffer(3, Some(out_buffer), 0);
    encoder.set_bytes(4, size_of::<P>() as u64, (&raw const *params).cast());
}

pub(super) fn dispatch_2d_pipeline(
    encoder: &metal::ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    dims: (u32, u32),
) {
    let width = pipeline.thread_execution_width().max(1);
    let max_threads = pipeline.max_total_threads_per_threadgroup().max(width);
    let height = (max_threads / width).max(1);
    encoder.dispatch_threads(
        MTLSize {
            width: u64::from(dims.0),
            height: u64::from(dims.1),
            depth: 1,
        },
        MTLSize {
            width,
            height,
            depth: 1,
        },
    );
}

#[cfg(target_os = "macos")]
pub(super) fn dispatch_3d_pipeline(
    encoder: &metal::ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    dims: (u32, u32, u32),
) {
    let width = pipeline.thread_execution_width().max(1);
    let max_threads = pipeline.max_total_threads_per_threadgroup().max(width);
    let height = (max_threads / width).max(1);
    encoder.dispatch_threads(
        MTLSize {
            width: u64::from(dims.0),
            height: u64::from(dims.1),
            depth: u64::from(dims.2),
        },
        MTLSize {
            width,
            height,
            depth: 1,
        },
    );
}

#[cfg(target_os = "macos")]
pub(super) fn packed_pair_extent(value: u32) -> u32 {
    value.div_ceil(2).max(1)
}

#[cfg(target_os = "macos")]
pub(super) fn dispatch_1d_pipeline(
    encoder: &metal::ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    threads: u32,
) {
    let threadgroup_width = choose_1d_threadgroup_width(
        pipeline.thread_execution_width(),
        pipeline.max_total_threads_per_threadgroup(),
        threads,
    );
    encoder.dispatch_threads(
        MTLSize {
            width: u64::from(threads.max(1)),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: threadgroup_width,
            height: 1,
            depth: 1,
        },
    );
}

#[cfg(target_os = "macos")]
pub(super) fn choose_1d_threadgroup_width(simd_width: u64, max_threads: u64, threads: u32) -> u64 {
    let simd_width = simd_width.max(1);
    let max_threads = max_threads.max(simd_width);
    let requested = u64::from(threads.max(1));
    let rounded = requested.div_ceil(simd_width) * simd_width;
    rounded.clamp(simd_width, max_threads.min(256).max(simd_width))
}

#[cfg(target_os = "macos")]
pub(super) fn pixel_format_to_out_format(fmt: PixelFormat) -> Option<u32> {
    match fmt {
        PixelFormat::Gray8 => Some(OUT_GRAY),
        PixelFormat::Rgb8 => Some(OUT_RGB),
        PixelFormat::Rgba8 => Some(OUT_RGBA),
        _ => None,
    }
}

#[cfg(target_os = "macos")]
pub(super) fn plane_mode_to_u32(mode: PlaneMode) -> u32 {
    match mode {
        PlaneMode::Gray => MODE_GRAY,
        PlaneMode::YCbCr => MODE_YCBCR,
        PlaneMode::Rgb => MODE_RGB,
    }
}
