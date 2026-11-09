// SPDX-License-Identifier: MIT OR Apache-2.0

//! The J2K-owned CUDA decode converts every `(Y, Cb, Cr)`, all 2^24, to the
//! RGB the CPU decoder produces.
//!
//! Each triple fills one 8x8 block of a 4:4:4 frame whose blocks carry only a
//! DC coefficient of `8 * (sample - 128)` under a quantizer of one. The inverse
//! DCT of such a block is exactly the flat sample, so the frames differ from
//! the CPU's only if the colour conversions do.

use j2k_core::{BackendRequest, DeviceSurface, ImageDecodeDevice, PixelFormat};
use j2k_jpeg::transcode::{encode_baseline_dct_image, extract_dct_blocks, DctExtractOptions};
use j2k_jpeg::{DecodeRequest, JpegBackend, JpegEncodeOptions, JpegSamples, JpegSubsampling};
use j2k_jpeg_cuda::Decoder;
use j2k_test_support::cuda_jpeg_hardware_decode_gate;

/// Blocks per frame side: one block per Y (columns) and Cr (rows) value.
const SIDE: u32 = 256 * 8;

/// A 4:4:4 quality-100 frame whose blocks the test rewrites.
fn template() -> j2k_jpeg::transcode::JpegDctImage {
    let rgb = vec![128u8; SIDE as usize * SIDE as usize * 3];
    let encoded = j2k_jpeg::encode_jpeg_baseline(
        JpegSamples::Rgb8 {
            data: &rgb,
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
    extract_dct_blocks(&encoded.data, DctExtractOptions::default()).expect("template blocks")
}

/// Sets every block of `image` to the flat triple `(Y, Cb, Cr)` with Y its
/// column, Cr its row and the given Cb.
fn fill_flat_blocks(image: &mut j2k_jpeg::transcode::JpegDctImage, cb: u8) {
    for component in &mut image.components {
        assert!(
            component
                .quant_table
                .iter()
                .all(|&quantizer| quantizer == 1),
            "quality 100 quantizes by one"
        );
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
}

#[test]
fn owned_cuda_decode_converts_every_colour_as_the_cpu_does_when_required() {
    if !cuda_jpeg_hardware_decode_gate(module_path!()) {
        return;
    }
    let mut image = template();
    for cb in 0..=255u8 {
        fill_flat_blocks(&mut image, cb);
        let jpeg = encode_baseline_dct_image(&image).expect("flat-block frame");
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
        assert!(cuda == cpu, "CUDA and CPU conversions differ for Cb {cb}");
    }
}
