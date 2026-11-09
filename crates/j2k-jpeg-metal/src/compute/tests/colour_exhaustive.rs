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

const DECODE_KERNEL: &str = r"
kernel void test_ycbcr_to_rgb_every_colour(
    device uchar *hybrid [[buffer(0)]],
    device uchar *store [[buffer(1)]],
    device uchar *float_rgba [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= (1u << 24)) {
        return;
    }
    const uchar cb = uchar(gid >> 16);
    const uchar cr = uchar(gid >> 8);
    const uchar y = uchar(gid);
    const uint3 rgb = hybrid_ycbcr8_to_rgb(int(y), int(cb), int(cr));
    hybrid[gid * 3u] = uchar(rgb.x);
    hybrid[gid * 3u + 1u] = uchar(rgb.y);
    hybrid[gid * 3u + 2u] = uchar(rgb.z);
    store_rgb_ycbcr(store, gid * 3u, y, cb, cr);
    const float4 rgba = rgba_float_ycbcr(y, cb, cr, 255u);
    float_rgba[gid * 3u] = uchar(rint(rgba.x * 255.0f));
    float_rgba[gid * 3u + 1u] = uchar(rint(rgba.y * 255.0f));
    float_rgba[gid * 3u + 2u] = uchar(rint(rgba.z * 255.0f));
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

/// The CPU decoder's YCbCr to RGB conversion of every `(Y, Cb, Cr)`, indexed
/// `cb << 16 | cr << 8 | y`.
///
/// For each Cb it decodes a 4:4:4 frame of 256 x 256 blocks: block
/// `(Y, Cr)` carries only a DC of `8 * (sample - 128)` per component under
/// quantizers of one, whose inverse DCT is exactly the flat sample, so the
/// block's pixels are the conversion of `(Y, Cb, Cr)`.
fn cpu_ycbcr_to_rgb_every_colour() -> Vec<[u8; 3]> {
    use j2k_core::PixelFormat;
    use j2k_jpeg::transcode::{encode_baseline_dct_image, extract_dct_blocks, DctExtractOptions};
    use j2k_jpeg::{
        DecodeRequest, Decoder, JpegBackend, JpegEncodeOptions, JpegSamples, JpegSubsampling,
    };

    const SIDE: u32 = 256 * 8;
    let grey = vec![128u8; SIDE as usize * SIDE as usize * 3];
    let template = j2k_jpeg::encode_jpeg_baseline(
        JpegSamples::Rgb8 {
            data: &grey,
            width: SIDE,
            height: SIDE,
        },
        JpegEncodeOptions {
            quality: 100,
            subsampling: JpegSubsampling::Ybr444,
            restart_interval: None,
            backend: JpegBackend::Cpu,
        },
    )
    .expect("template encode");
    let mut image =
        extract_dct_blocks(&template.data, DctExtractOptions::default()).expect("template blocks");
    let mut expected = vec![[0u8; 3]; COLOURS];
    for cb in 0..=255u8 {
        for component in &mut image.components {
            assert!(component
                .quant_table
                .iter()
                .all(|&quantizer| quantizer == 1));
            let columns = component.block_cols as usize;
            let blocks = component
                .quantized_blocks
                .iter_mut()
                .zip(component.dequantized_blocks.iter_mut());
            for (index, (quantized, dequantized)) in blocks.enumerate() {
                let value = match component.component_index {
                    0 => index % columns,
                    1 => usize::from(cb),
                    _ => index / columns,
                };
                let value = i16::try_from(value).expect("sample fits i16");
                *quantized = [0; 64];
                quantized[0] = 8 * (value - 128);
                *dequantized = *quantized;
            }
        }
        let jpeg = encode_baseline_dct_image(&image).expect("flat-block frame");
        let (rgb, _) = Decoder::new(&jpeg)
            .expect("CPU decoder")
            .decode_request(DecodeRequest::full(PixelFormat::Rgb8))
            .expect("CPU decode");
        let pixels = rgb.as_chunks::<3>().0;
        for cr in 0..256usize {
            for y in 0..256usize {
                expected[(usize::from(cb) << 16) | (cr << 8) | y] =
                    pixels[cr * 8 * SIDE as usize + y * 8];
            }
        }
    }
    expected
}

#[test]
fn decode_ycbcr_to_rgb_matches_the_cpu_for_every_colour() {
    if !super::should_run_metal_runtime() {
        return;
    }
    let outputs = run_every_colour(DECODE_KERNEL, "test_ycbcr_to_rgb_every_colour", 3);
    let expected = cpu_ycbcr_to_rgb_every_colour();
    for (name, actual) in [
        "hybrid_ycbcr8_to_rgb",
        "store_rgb_ycbcr",
        "rgba_float_ycbcr",
    ]
    .into_iter()
    .zip(&outputs)
    {
        if let Some(colour) = (0..COLOURS).find(|&colour| actual[colour] != expected[colour]) {
            panic!(
                "{name}: (Cb, Cr, Y) {colour:06x} gives {:?}, the CPU {:?}",
                actual[colour], expected[colour]
            );
        }
    }
}
