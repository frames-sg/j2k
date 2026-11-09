// SPDX-License-Identifier: MIT OR Apache-2.0

//! The J2K-owned CUDA decode clamps dequantized coefficients to `i16` before
//! the inverse DCT, as the CPU and Metal decoders do.

use j2k_codec_math::jpeg::idct::islow_block;
use j2k_core::{BackendRequest, DeviceSurface, ImageDecodeDevice, PixelFormat};
use j2k_jpeg::transcode::{encode_baseline_dct_image, extract_dct_blocks, DctExtractOptions};
use j2k_jpeg::{DecodeRequest, JpegBackend, JpegEncodeOptions, JpegSamples, JpegSubsampling};
use j2k_jpeg_cuda::Decoder;
use j2k_test_support::cuda_jpeg_hardware_decode_gate;

const SIDE: u32 = 16;

/// The extreme coefficients each block carries, in natural order: a maximal
/// DC and a minimal first AC, whose products with the quality-1 quantizers
/// leave the `i16` range.
fn extreme_block(quant_table: &[u16; 64]) -> ([i16; 64], [i32; 64]) {
    let mut quantized = [0i16; 64];
    quantized[0] = 2_047;
    quantized[1] = -1_023;
    let mut dequantized = [0i32; 64];
    // The quantizers are stored in zigzag order; natural 0 and 1 are zigzag 0
    // and 1.
    dequantized[0] = i32::from(quantized[0]) * i32::from(quant_table[0]);
    dequantized[1] = i32::from(quantized[1]) * i32::from(quant_table[1]);
    (quantized, dequantized)
}

fn clamped(dequantized: &[i32; 64]) -> [i16; 64] {
    dequantized.map(|value| {
        i16::try_from(value.clamp(i32::from(i16::MIN), i32::from(i16::MAX)))
            .expect("clamped coefficient fits i16")
    })
}

#[test]
fn owned_cuda_decode_clamps_dequantized_coefficients_like_the_cpu_when_required() {
    if !cuda_jpeg_hardware_decode_gate(module_path!()) {
        return;
    }
    let rgb = vec![128u8; SIDE as usize * SIDE as usize * 3];
    let template = j2k_jpeg::encode_jpeg_baseline(
        JpegSamples::Rgb8 {
            data: &rgb,
            width: SIDE,
            height: SIDE,
        },
        JpegEncodeOptions {
            quality: 1,
            subsampling: JpegSubsampling::Ybr444,
            restart_interval: None,
            backend: JpegBackend::Cpu,
        },
    )
    .expect("template encode");
    let mut image =
        extract_dct_blocks(&template.data, DctExtractOptions::default()).expect("template blocks");
    for component in &mut image.components {
        let (quantized, dequantized) = extreme_block(&component.quant_table);
        assert!(
            dequantized[0] > i32::from(i16::MAX) && dequantized[1] < i32::from(i16::MIN),
            "both products leave the i16 range"
        );
        // The clamped block decodes to both extremes, unlike the saturated
        // white of the unclamped DC.
        let mut decoded = [0u8; 64];
        islow_block(&clamped(&dequantized), &mut decoded);
        assert!(decoded.contains(&0) && decoded.contains(&255));
        for (block, dequantized_block) in component
            .quantized_blocks
            .iter_mut()
            .zip(component.dequantized_blocks.iter_mut())
        {
            *block = quantized;
            *dequantized_block = clamped(&dequantized);
        }
    }
    let jpeg = encode_baseline_dct_image(&image).expect("extreme-coefficient frame");

    let mut decoder = Decoder::new(&jpeg).expect("decoder");
    let surface = decoder
        .decode_to_device(PixelFormat::Rgb8, BackendRequest::Cuda)
        .expect("cuda surface");
    let stats = surface.cuda_surface().expect("cuda surface").stats();
    assert!(stats.used_owned_cuda_decode() && !stats.used_hardware_decode());
    let mut cuda = vec![0u8; surface.byte_len()];
    surface
        .download_into(&mut cuda, surface.pitch_bytes())
        .expect("download cuda surface");
    let (cpu, _) = j2k_jpeg::Decoder::new(&jpeg)
        .expect("host decoder")
        .decode_request(DecodeRequest::full(PixelFormat::Rgb8))
        .expect("host decode");
    assert!(cuda == cpu, "CUDA and CPU dequantization differ");
}
