// SPDX-License-Identifier: MIT OR Apache-2.0

//! CUDA region decodes that reconstruct only windows of each band return the
//! CPU decoder's samples, wherever the region sits in the image.

#![cfg(feature = "cuda-runtime")]

use j2k_core::{BackendRequest, Downscale, ImageDecode, ImageDecodeDevice, PixelFormat, Rect};
use j2k_cuda::{J2kDecoder, J2kScratchPool, SurfaceResidency};
use j2k_native::{encode, encode_htj2k, EncodeOptions};
use j2k_test_support::cuda_runtime_and_strict_oxide_gate;

const WIDTH: u32 = 600;
const HEIGHT: u32 = 440;

fn codestream(components: u16, reversible: bool, ht: bool) -> Vec<u8> {
    let mut pixels = Vec::new();
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            for c in 0..u32::from(components) {
                pixels.push((x * 7 + y * 13 + c * 50 + (x * y) / 97).to_le_bytes()[0]);
            }
        }
    }
    let options = EncodeOptions {
        reversible,
        num_decomposition_levels: 4,
        code_block_width_exp: 3,
        code_block_height_exp: 3,
        ..EncodeOptions::default()
    };
    let encoder = if ht { encode_htj2k } else { encode };
    encoder(&pixels, WIDTH, HEIGHT, components, 8, false, &options).expect("encode source")
}

/// CPU and CUDA decodes of `roi` at `scale`, and the CUDA surface residency.
fn decode_both(
    bytes: &[u8],
    format: PixelFormat,
    channels: usize,
    roi: Rect,
    scale: Downscale,
) -> (Vec<u8>, Vec<u8>, SurfaceResidency) {
    let output = if scale == Downscale::None {
        roi
    } else {
        roi.scaled_covering(scale)
    };
    let stride = output.w as usize * channels;
    let mut cpu = vec![0u8; stride * output.h as usize];
    let mut host = J2kDecoder::new(bytes).expect("host decoder");
    let mut pool = J2kScratchPool::new();
    let mut device = J2kDecoder::new(bytes).expect("decoder");
    let surface = if scale == Downscale::None {
        host.decode_region_into(&mut pool, &mut cpu, stride, format, roi)
            .expect("CPU region decode");
        device.decode_region_to_device(format, roi, BackendRequest::Cuda)
    } else {
        host.decode_region_scaled_into(&mut pool, &mut cpu, stride, format, roi, scale)
            .expect("CPU scaled region decode");
        device.decode_region_scaled_to_device(format, roi, scale, BackendRequest::Cuda)
    }
    .expect("CUDA region decode");
    let mut cuda = vec![0u8; cpu.len()];
    surface
        .download_into(&mut cuda, stride)
        .expect("download region");
    (cpu, cuda, surface.residency())
}

#[test]
fn cuda_region_decodes_match_cpu_wherever_the_region_sits() {
    if !cuda_runtime_and_strict_oxide_gate(module_path!()) {
        return;
    }

    let rect = |x, y, w, h| Rect { x, y, w, h };
    let regions = [
        (rect(0, 0, 64, 48), Downscale::None),
        (rect(257, 131, 101, 77), Downscale::None),
        (rect(471, 333, 129, 107), Downscale::None),
        (rect(300, 0, 1, 440), Downscale::None),
        (rect(0, 219, 600, 3), Downscale::None),
        (rect(75, 55, 61, 47), Downscale::Half),
    ];
    let mut mismatches = Vec::new();
    for (components, format) in [(1, PixelFormat::Gray8), (3, PixelFormat::Rgb8)] {
        for reversible in [true, false] {
            for ht in [false, true] {
                let bytes = codestream(components, reversible, ht);
                for (roi, scale) in regions {
                    let (cpu, cuda, residency) =
                        decode_both(&bytes, format, usize::from(components), roi, scale);
                    if cuda != cpu || residency != SurfaceResidency::CudaResidentDecode {
                        let differing = cpu.iter().zip(&cuda).filter(|(a, b)| a != b).count();
                        mismatches.push(format!(
                            "{format:?} reversible={reversible} ht={ht} {roi:?} {scale:?}: \
                             {residency:?}, {differing} bytes differ"
                        ));
                    }
                }
            }
        }
    }
    assert!(mismatches.is_empty(), "{}", mismatches.join("\n"));
}
