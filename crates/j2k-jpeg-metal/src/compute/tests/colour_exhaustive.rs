// SPDX-License-Identifier: MIT OR Apache-2.0

//! The shaders' colour conversions match the CPU's scalar reference for every
//! 8-bit colour, all 2^24.

use crate::compute::pipeline_registry::SHADER_SOURCE;
use crate::metal_types::prelude::*;
use j2k_codec_math::jpeg::ycbcr;
use j2k_metal_support::{
    checked_buffer_read_vec, checked_command_buffer, checked_command_queue,
    checked_compute_command_encoder, checked_shared_buffer_for_len, commit_and_wait,
    dispatch_1d_pipeline, system_default_device, MetalPipelineLoader,
};

const COLOURS: usize = 1 << 24;

const ENCODE_KERNEL: &str = r"
kernel void test_rgb_to_ycbcr_every_colour(
    device uchar *out [[buffer(0)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= (1u << 24)) {
        return;
    }
    const uchar3 rgb = uchar3(uchar(gid >> 16), uchar(gid >> 8), uchar(gid));
    out[gid * 3u] = rgb_to_ycbcr_component(rgb, 0u);
    out[gid * 3u + 1u] = rgb_to_ycbcr_component(rgb, 1u);
    out[gid * 3u + 2u] = rgb_to_ycbcr_component(rgb, 2u);
}
";

/// Runs `kernel` from `source` over every colour, one thread per colour, and
/// reads back the three bytes per colour of each of its `outputs` buffers.
fn run_every_colour(source: &str, kernel: &str, outputs: usize) -> Vec<Vec<[u8; 3]>> {
    let device = system_default_device().expect("Metal device");
    let loader = MetalPipelineLoader::new(&device, &format!("{SHADER_SOURCE}\n{source}"))
        .expect("compile colour test kernel");
    let pipeline = loader.pipeline(kernel).expect("colour test pipeline");
    let queue = checked_command_queue(&device).expect("command queue");
    let buffers = (0..outputs)
        .map(|_| checked_shared_buffer_for_len::<[u8; 3]>(&device, COLOURS).expect("output"))
        .collect::<Vec<_>>();
    let command_buffer = checked_command_buffer(&queue).expect("command buffer");
    let encoder = checked_compute_command_encoder(&command_buffer).expect("encoder");
    encoder.setComputePipelineState(&pipeline);
    for (index, buffer) in (0u64..).zip(&buffers) {
        encoder.bind_buffer(index, Some(buffer), 0);
    }
    dispatch_1d_pipeline(&encoder, &pipeline, 1 << 24);
    encoder.endEncoding();
    commit_and_wait(&command_buffer).expect("colour test kernel");
    buffers
        .iter()
        .map(|buffer| {
            // SAFETY: the command buffer completed, so the GPU no longer
            // writes `buffer`, and the read covers exactly its colours.
            unsafe { checked_buffer_read_vec::<[u8; 3]>(buffer, 0, COLOURS) }.expect("read")
        })
        .collect()
}

fn clamp_u8(value: i32) -> u8 {
    u8::try_from(value.clamp(0, 255)).expect("clamped sample fits u8")
}

/// The CPU encoder's scalar transform, `j2k_codec_math::jpeg::ycbcr`.
fn cpu_rgb_to_ycbcr(colour: usize) -> [u8; 3] {
    let [_, red, green, blue] = u32::try_from(colour)
        .expect("colour index fits u32")
        .to_be_bytes()
        .map(i32::from);
    [
        clamp_u8(ycbcr::rgb_to_y(red, green, blue)),
        clamp_u8(ycbcr::rgb_to_cb(red, green, blue, 128)),
        clamp_u8(ycbcr::rgb_to_cr(red, green, blue, 128)),
    ]
}

#[test]
fn encode_rgb_to_ycbcr_matches_the_cpu_for_every_colour() {
    if !super::should_run_metal_runtime() {
        return;
    }
    let outputs = run_every_colour(ENCODE_KERNEL, "test_rgb_to_ycbcr_every_colour", 1);
    if let Some(colour) =
        (0..COLOURS).find(|&colour| outputs[0][colour] != cpu_rgb_to_ycbcr(colour))
    {
        panic!(
            "RGB {colour:06x} gives {:?}, the CPU {:?}",
            outputs[0][colour],
            cpu_rgb_to_ycbcr(colour)
        );
    }
}
