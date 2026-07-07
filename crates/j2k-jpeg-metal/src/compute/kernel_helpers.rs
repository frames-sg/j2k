// SPDX-License-Identifier: MIT OR Apache-2.0

use std::mem::size_of;

use j2k_core::PixelFormat;
use metal::{Buffer, ComputePipelineState, MTLSize};

use super::{
    FastSubsampledPacket, PlaneMode, PreparedHuffmanHost, MODE_GRAY, MODE_RGB, MODE_YCBCR,
    OUT_GRAY, OUT_RGB, OUT_RGBA,
};

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
pub(super) struct FastDecodeEntropyInputs<'a, P> {
    pub(super) entropy_buffer: &'a Buffer,
    pub(super) planes: [&'a Buffer; 3],
    pub(super) params: &'a P,
    pub(super) quants: [&'a [u16; 64]; 3],
    pub(super) dc_tables: &'a [PreparedHuffmanHost; 3],
    pub(super) ac_tables: &'a [PreparedHuffmanHost; 3],
    /// Buffer bound at Metal slot 14. Single-decode kernels use restart
    /// offsets; batch kernels use entropy offsets.
    pub(super) slot14_buffer: &'a Buffer,
    /// Buffer bound at Metal slot 15. Single-decode kernels use decode status;
    /// batch kernels use entropy lengths.
    pub(super) slot15_buffer: &'a Buffer,
    /// Buffer bound at Metal slot 16. Single-decode kernels use entropy
    /// checkpoints; batch kernels use decode status.
    pub(super) slot16_buffer: &'a Buffer,
}

#[cfg(target_os = "macos")]
/// Bind the shared fast-decode entropy kernel inputs at slots 0-16: entropy
/// bytes, the three component planes, the family params struct, the three
/// quantization tables, the per-component DC/AC Huffman table pairs, and the
/// three layout-specific auxiliary buffers for slots 14-16.
pub(super) fn bind_fast_decode_entropy_inputs<P>(
    encoder: &metal::ComputeCommandEncoderRef,
    inputs: &FastDecodeEntropyInputs<'_, P>,
) {
    encoder.set_buffer(0, Some(inputs.entropy_buffer), 0);
    encoder.set_buffer(1, Some(inputs.planes[0]), 0);
    encoder.set_buffer(2, Some(inputs.planes[1]), 0);
    encoder.set_buffer(3, Some(inputs.planes[2]), 0);
    encoder.set_bytes(4, size_of::<P>() as u64, (&raw const *inputs.params).cast());
    for (slot, quant) in (5u64..).zip(inputs.quants) {
        encoder.set_bytes(slot, size_of::<[u16; 64]>() as u64, quant.as_ptr().cast());
    }
    for (index, (dc, ac)) in inputs
        .dc_tables
        .iter()
        .zip(inputs.ac_tables.iter())
        .enumerate()
    {
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
    encoder.set_buffer(14, Some(inputs.slot14_buffer), 0);
    encoder.set_buffer(15, Some(inputs.slot15_buffer), 0);
    encoder.set_buffer(16, Some(inputs.slot16_buffer), 0);
}

pub(super) fn fast_packet_huffman_tables<P: FastSubsampledPacket>(
    packet: &P,
) -> ([PreparedHuffmanHost; 3], [PreparedHuffmanHost; 3]) {
    (
        [
            PreparedHuffmanHost::from(packet.y_dc_table()),
            PreparedHuffmanHost::from(packet.cb_dc_table()),
            PreparedHuffmanHost::from(packet.cr_dc_table()),
        ],
        [
            PreparedHuffmanHost::from(packet.y_ac_table()),
            PreparedHuffmanHost::from(packet.cb_ac_table()),
            PreparedHuffmanHost::from(packet.cr_ac_table()),
        ],
    )
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
