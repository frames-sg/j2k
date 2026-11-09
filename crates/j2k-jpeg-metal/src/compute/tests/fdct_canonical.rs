// SPDX-License-Identifier: MIT OR Apache-2.0

//! The baseline encoder's forward DCT and quantization produce exactly the
//! coefficients of `j2k_codec_math::jpeg::fdct`, the definition every encoder
//! backend shares. Codec-math proves that definition against the exact DCT;
//! this test pins the shader to it on the GPU it runs on.

use crate::compute::pipeline_registry::SHADER_SOURCE;
use crate::metal_types::prelude::*;
use j2k_codec_math::jpeg::fdct::fdct_quantize;
use j2k_jpeg::{JpegBackend, JpegEncodeOptions, JpegSubsampling};
use j2k_metal_support::{
    checked_buffer_read_vec, checked_command_buffer, checked_command_queue,
    checked_compute_command_encoder, checked_shared_buffer_for_len,
    checked_shared_buffer_with_slice, commit_and_wait, dispatch_1d_pipeline, system_default_device,
    MetalPipelineLoader,
};

const FDCT_KERNEL: &str = r"
kernel void test_fdct_quantize_blocks(
    device const uchar *blocks [[buffer(0)]],
    constant uchar *quant [[buffer(1)]],
    device int *coefficients [[buffer(2)]],
    constant uint &count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) {
        return;
    }
    thread uchar block[64];
    thread int coeffs[64];
    for (uint i = 0u; i < 64u; i++) {
        block[i] = blocks[gid * 64u + i];
    }
    jpeg_encode_fdct_quantize(block, quant, coeffs);
    for (uint i = 0u; i < 64u; i++) {
        coefficients[gid * 64u + i] = coeffs[i];
    }
}
";

const BLOCKS_PER_FAMILY: usize = 2048;

struct Lcg(u64);

impl Lcg {
    fn next_byte(&mut self) -> u8 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0.to_be_bytes()[0]
    }
}

fn uniform_block(rng: &mut Lcg) -> [u8; 64] {
    core::array::from_fn(|_| rng.next_byte())
}

fn smooth_block(rng: &mut Lcg) -> [u8; 64] {
    let base = i64::from(rng.next_byte());
    let gx = i64::from(rng.next_byte() % 17) - 8;
    let gy = i64::from(rng.next_byte() % 17) - 8;
    core::array::from_fn(|index| {
        let x = i64::try_from(index % 8).expect("column fits i64");
        let y = i64::try_from(index / 8).expect("row fits i64");
        let jitter = i64::from(rng.next_byte() % 5) - 2;
        let sample = (base + gx * x + gy * y + jitter).clamp(0, 255);
        u8::try_from(sample).expect("clamped sample fits u8")
    })
}

/// Two-level blocks produce exact rounding ties at many frequencies.
fn two_level_block(rng: &mut Lcg) -> [u8; 64] {
    let (high, low) = (rng.next_byte(), rng.next_byte());
    core::array::from_fn(|_| if rng.next_byte() & 1 == 1 { high } else { low })
}

/// Uniform noise, smooth gradients, two-level blocks and the extremes that
/// bound every fold.
fn corpus() -> Vec<[u8; 64]> {
    let mut rng = Lcg(0x5eed_f00d);
    let families: [fn(&mut Lcg) -> [u8; 64]; 3] = [uniform_block, smooth_block, two_level_block];
    let mut blocks: Vec<[u8; 64]> = families
        .into_iter()
        .flat_map(|family| {
            (0..BLOCKS_PER_FAMILY)
                .map(|_| family(&mut rng))
                .collect::<Vec<_>>()
        })
        .collect();
    let checker = |even: u8, odd: u8| -> [u8; 64] {
        core::array::from_fn(|index| {
            if (index % 8 + index / 8) % 2 == 0 {
                even
            } else {
                odd
            }
        })
    };
    blocks.extend([
        [0; 64],
        [255; 64],
        [128; 64],
        checker(0, 255),
        checker(255, 0),
    ]);
    blocks
}

fn quantization_tables() -> [[u8; 64]; 5] {
    let tables = |quality| {
        j2k_jpeg::adapter::baseline_encode_tables(JpegEncodeOptions {
            quality,
            subsampling: JpegSubsampling::Ybr420,
            restart_interval: None,
            backend: JpegBackend::Metal,
        })
        .expect("standard baseline tables")
    };
    let mixed = core::array::from_fn(|index| {
        u8::try_from(1 + (index * 37) % 255).expect("quantizer fits u8")
    });
    [
        [1; 64],
        tables(90).q_luma,
        tables(50).q_chroma,
        mixed,
        [255; 64],
    ]
}

#[test]
fn fdct_quantize_matches_the_shared_codec_math_definition() {
    if !super::should_run_metal_runtime() {
        return;
    }
    let device = system_default_device().expect("Metal device");
    let loader = MetalPipelineLoader::new(&device, &format!("{SHADER_SOURCE}\n{FDCT_KERNEL}"))
        .expect("compile FDCT test kernel");
    let pipeline = loader
        .pipeline("test_fdct_quantize_blocks")
        .expect("FDCT test pipeline");
    let queue = checked_command_queue(&device).expect("command queue");
    let blocks = corpus();
    let count = u32::try_from(blocks.len()).expect("block count fits u32");
    let block_buffer = checked_shared_buffer_with_slice(&device, &blocks).expect("blocks");
    let count_buffer =
        checked_shared_buffer_with_slice(&device, std::slice::from_ref(&count)).expect("count");

    for quant in quantization_tables() {
        let quant_buffer = checked_shared_buffer_with_slice(&device, &quant).expect("quant");
        let output = checked_shared_buffer_for_len::<[i32; 64]>(&device, blocks.len())
            .expect("coefficients");
        let command_buffer = checked_command_buffer(&queue).expect("command buffer");
        let encoder = checked_compute_command_encoder(&command_buffer).expect("encoder");
        encoder.setComputePipelineState(&pipeline);
        encoder.bind_buffer(0, Some(&block_buffer), 0);
        encoder.bind_buffer(1, Some(&quant_buffer), 0);
        encoder.bind_buffer(2, Some(&output), 0);
        encoder.bind_buffer(3, Some(&count_buffer), 0);
        dispatch_1d_pipeline(&encoder, &pipeline, u64::from(count));
        encoder.endEncoding();
        commit_and_wait(&command_buffer).expect("FDCT test kernel");
        // SAFETY: the command buffer completed, so the GPU no longer writes
        // `output`, and the read covers exactly its allocated blocks.
        let coefficients =
            unsafe { checked_buffer_read_vec::<[i32; 64]>(&output, 0, blocks.len()) }
                .expect("read");

        for (block_index, (block, actual)) in blocks.iter().zip(&coefficients).enumerate() {
            let expected = fdct_quantize(block, &quant).map(i32::from);
            assert_eq!(
                actual, &expected,
                "block {block_index} with quantizers {quant:?}"
            );
        }
    }
}
