// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(feature = "cuda-runtime")]
use super::{
    encode_with_cuda_test_accelerator, CudaEncodeStageAccelerator, CudaTestEncodeRequest,
    DecodeSettings, EncodeOptions, Image,
};

#[cfg(feature = "cuda-runtime")]
#[test]
fn cuda_encode_uses_resident_tile_body_when_runtime_required() {
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return;
    }

    let pixels: Vec<u8> = (0u16..32 * 32)
        .map(|i| u8::try_from((i * 23 + 11) & 0xFF).expect("masked value fits in u8"))
        .collect();
    let options = EncodeOptions {
        reversible: true,
        use_ht_block_coding: true,
        num_decomposition_levels: 0,
        code_block_width_exp: 2,
        code_block_height_exp: 2,
        ..EncodeOptions::default()
    };
    let mut accelerator = CudaEncodeStageAccelerator::default();

    let codestream = encode_with_cuda_test_accelerator(CudaTestEncodeRequest {
        pixels: &pixels,
        width: 32,
        height: 32,
        components: 1,
        bit_depth: 8,
        signed: false,
        options: &options,
        accelerator: &mut accelerator,
    })
    .expect("encode HTJ2K through CUDA tile-body hook");
    let decoded = Image::new(&codestream, &DecodeSettings::default())
        .expect("codestream parses")
        .decode_native()
        .expect("codestream decodes");

    assert_eq!(decoded.data, pixels);
    assert_eq!(accelerator.htj2k_tile_attempts(), 1);
    assert_eq!(accelerator.htj2k_tile_dispatches(), 1);
    assert_eq!(accelerator.ht_subband_attempts(), 0);
    assert_eq!(accelerator.ht_subband_dispatches(), 0);
    assert_eq!(accelerator.deinterleave_dispatches(), 1);
    assert_eq!(accelerator.quantize_subband_attempts(), 1);
    assert_eq!(accelerator.quantize_subband_dispatches(), 1);
    assert_eq!(accelerator.ht_code_block_attempts(), 4);
    assert_eq!(accelerator.ht_code_block_dispatches(), 1);
    assert_eq!(accelerator.packetization_attempts(), 1);
    assert_eq!(accelerator.packetization_dispatches(), 1);
}

#[cfg(feature = "cuda-runtime")]
#[test]
fn cuda_encode_uses_resident_dwt_tile_body_when_runtime_required() {
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return;
    }

    let pixels: Vec<u8> = (0u16..32 * 32)
        .map(|i| u8::try_from((i * 29 + 5) & 0xFF).expect("masked value fits in u8"))
        .collect();
    let options = EncodeOptions {
        reversible: true,
        use_ht_block_coding: true,
        num_decomposition_levels: 1,
        code_block_width_exp: 2,
        code_block_height_exp: 2,
        ..EncodeOptions::default()
    };
    let mut accelerator = CudaEncodeStageAccelerator::default();

    let codestream = encode_with_cuda_test_accelerator(CudaTestEncodeRequest {
        pixels: &pixels,
        width: 32,
        height: 32,
        components: 1,
        bit_depth: 8,
        signed: false,
        options: &options,
        accelerator: &mut accelerator,
    })
    .expect("encode HTJ2K DWT through CUDA tile-body hook");
    let decoded = Image::new(&codestream, &DecodeSettings::default())
        .expect("codestream parses")
        .decode_native()
        .expect("codestream decodes");

    assert_eq!(decoded.data, pixels);
    assert_eq!(accelerator.htj2k_tile_attempts(), 1);
    assert_eq!(accelerator.htj2k_tile_dispatches(), 1);
    assert_eq!(accelerator.ht_subband_attempts(), 0);
    assert_eq!(accelerator.ht_subband_dispatches(), 0);
    assert_eq!(accelerator.forward_dwt53_attempts(), 1);
    assert!(accelerator.forward_dwt53_dispatches() > 0);
    assert_eq!(accelerator.quantize_subband_attempts(), 4);
    assert_eq!(accelerator.quantize_subband_dispatches(), 4);
    assert_eq!(accelerator.ht_code_block_attempts(), 4);
    assert_eq!(accelerator.ht_code_block_dispatches(), 1);
    assert_eq!(accelerator.packetization_attempts(), 1);
    assert_eq!(accelerator.packetization_dispatches(), 1);
}

#[cfg(feature = "cuda-runtime")]
#[test]
fn cuda_encode_uses_resident_mct_dwt_tile_body_when_runtime_required() {
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return;
    }

    let pixels: Vec<u8> = (0u16..32 * 32 * 3)
        .map(|i| u8::try_from((i * 19 + 17) & 0xFF).expect("masked value fits in u8"))
        .collect();
    let options = EncodeOptions {
        reversible: true,
        use_mct: true,
        use_ht_block_coding: true,
        num_decomposition_levels: 1,
        code_block_width_exp: 2,
        code_block_height_exp: 2,
        ..EncodeOptions::default()
    };
    let mut accelerator = CudaEncodeStageAccelerator::default();

    let codestream = encode_with_cuda_test_accelerator(CudaTestEncodeRequest {
        pixels: &pixels,
        width: 32,
        height: 32,
        components: 3,
        bit_depth: 8,
        signed: false,
        options: &options,
        accelerator: &mut accelerator,
    })
    .expect("encode HTJ2K RGB DWT through CUDA tile-body hook");
    let decoded = Image::new(&codestream, &DecodeSettings::default())
        .expect("codestream parses")
        .decode_native()
        .expect("codestream decodes");

    assert_eq!(decoded.data, pixels);
    assert_eq!(accelerator.htj2k_tile_attempts(), 1);
    assert_eq!(accelerator.htj2k_tile_dispatches(), 1);
    assert_eq!(accelerator.ht_subband_attempts(), 0);
    assert_eq!(accelerator.forward_rct_attempts(), 1);
    assert_eq!(accelerator.forward_rct_dispatches(), 1);
    assert_eq!(accelerator.forward_dwt53_attempts(), 3);
    assert!(accelerator.forward_dwt53_dispatches() > 0);
    assert_eq!(accelerator.quantize_subband_attempts(), 12);
    assert_eq!(accelerator.quantize_subband_dispatches(), 12);
    assert_eq!(accelerator.ht_code_block_attempts(), 12);
    assert_eq!(accelerator.ht_code_block_dispatches(), 1);
    assert_eq!(accelerator.packetization_attempts(), 1);
    assert_eq!(accelerator.packetization_dispatches(), 1);
}

#[cfg(feature = "cuda-runtime")]
#[test]
fn cuda_encode_uses_resident_dwt97_tile_body_when_runtime_required() {
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return;
    }

    let pixels: Vec<u8> = (0u16..32 * 32)
        .map(|i| u8::try_from((i * 31 + 7) & 0xFF).expect("masked value fits in u8"))
        .collect();
    let options = EncodeOptions {
        reversible: false,
        use_ht_block_coding: true,
        num_decomposition_levels: 1,
        code_block_width_exp: 2,
        code_block_height_exp: 2,
        ..EncodeOptions::default()
    };
    let mut accelerator = CudaEncodeStageAccelerator::default();

    let codestream = encode_with_cuda_test_accelerator(CudaTestEncodeRequest {
        pixels: &pixels,
        width: 32,
        height: 32,
        components: 1,
        bit_depth: 8,
        signed: false,
        options: &options,
        accelerator: &mut accelerator,
    })
    .expect("encode irreversible HTJ2K DWT through CUDA tile-body hook");
    let decoded = Image::new(&codestream, &DecodeSettings::default())
        .expect("codestream parses")
        .decode_native()
        .expect("codestream decodes");

    assert_eq!(decoded.width, 32);
    assert_eq!(decoded.height, 32);
    assert_eq!(decoded.num_components, 1);
    assert_eq!(accelerator.htj2k_tile_attempts(), 1);
    assert_eq!(accelerator.htj2k_tile_dispatches(), 1);
    assert_eq!(accelerator.ht_subband_attempts(), 0);
    assert_eq!(accelerator.forward_dwt97_attempts(), 1);
    assert!(accelerator.forward_dwt97_dispatches() > 0);
    assert_eq!(accelerator.quantize_subband_attempts(), 4);
    assert_eq!(accelerator.quantize_subband_dispatches(), 4);
    assert_eq!(accelerator.ht_code_block_attempts(), 4);
    assert_eq!(accelerator.ht_code_block_dispatches(), 1);
    assert_eq!(accelerator.packetization_attempts(), 1);
    assert_eq!(accelerator.packetization_dispatches(), 1);
}

#[cfg(feature = "cuda-runtime")]
#[test]
fn cuda_whole_tile_htj2k_encodes_match_cpu_codestreams() {
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return;
    }

    // A small ramp, and frames whose packets hold more code blocks than a
    // thread block, half of them flat, with no coding passes.
    let mut mismatches = Vec::new();
    for (width, height, components) in [
        (37u32, 29u32, 1u8),
        (37, 29, 3),
        (160, 128, 1),
        (160, 128, 3),
    ] {
        let pixels = tile_test_pixels(width, height, components);
        for reversible in [true, false] {
            for levels in [0u8, 2] {
                let options = EncodeOptions {
                    reversible,
                    use_ht_block_coding: true,
                    num_decomposition_levels: levels,
                    code_block_width_exp: 2,
                    code_block_height_exp: 2,
                    ..EncodeOptions::default()
                };
                let cpu = j2k_native::encode(
                    &pixels,
                    width,
                    height,
                    u16::from(components),
                    8,
                    false,
                    &options,
                )
                .expect("CPU encode");
                let mut accelerator = CudaEncodeStageAccelerator::default();
                let cuda = encode_with_cuda_test_accelerator(CudaTestEncodeRequest {
                    pixels: &pixels,
                    width,
                    height,
                    components,
                    bit_depth: 8,
                    signed: false,
                    options: &options,
                    accelerator: &mut accelerator,
                })
                .expect("CUDA encode");
                let tile_dispatches = accelerator.htj2k_tile_dispatches();
                if cuda != cpu || tile_dispatches != 1 {
                    mismatches.push(format!(
                        "{width}x{height} components={components} reversible={reversible} levels={levels}: \
                         {tile_dispatches} tile dispatches, codestreams equal: {}",
                        cuda == cpu
                    ));
                }
            }
        }
    }
    assert!(mismatches.is_empty(), "{}", mismatches.join("\n"));
}

/// A ramp, or for frames wider than 64 a flat left half and a right half whose
/// texture amplitude varies by row.
#[cfg(feature = "cuda-runtime")]
fn tile_test_pixels(width: u32, height: u32, components: u8) -> Vec<u8> {
    let mut pixels = Vec::new();
    for y in 0..height {
        for x in 0..width {
            for c in 0..u32::from(components) {
                let texture = ((x * 31 + y * 7 + c * 50) ^ (x * y / 5)) >> (y % 5);
                let flat = width > 64 && x < width / 2;
                pixels.push(if flat { 128 } else { texture.to_le_bytes()[0] });
            }
        }
    }
    pixels
}

#[cfg(feature = "cuda-runtime")]
#[test]
fn cuda_large_tiles_with_small_blocks_match_cpu_codestreams() {
    if !j2k_test_support::cuda_runtime_and_strict_oxide_gate(module_path!()) {
        return;
    }

    for (width, height, components, levels, block_exp, precincts) in [
        (2048u32, 2048u32, 1u8, 1u8, 0, true),
        (513, 515, 3, 2, 0, true),
        (513, 515, 4, 2, 1, false),
    ] {
        let pixels: Vec<u8> = (0..width * height * u32::from(components))
            .map(|i| u8::try_from((i * 17 + i / 31) & 255).expect("masked sample"))
            .collect();
        let options = EncodeOptions {
            reversible: true,
            use_ht_block_coding: true,
            use_mct: components >= 3,
            num_decomposition_levels: levels,
            code_block_width_exp: block_exp,
            code_block_height_exp: block_exp,
            // Explicit precincts keep large packet tag trees within the device limit.
            // The resident case crosses a chunk boundary across its four components.
            precinct_exponents: if precincts {
                vec![(7, 7); usize::from(levels) + 1]
            } else {
                Vec::new()
            },
            ..EncodeOptions::default()
        };
        let expected = j2k_native::encode_htj2k(
            &pixels,
            width,
            height,
            u16::from(components),
            8,
            false,
            &options,
        )
        .expect("CPU encode with small blocks");
        let mut accelerator = CudaEncodeStageAccelerator::default();
        let actual = encode_with_cuda_test_accelerator(CudaTestEncodeRequest {
            pixels: &pixels,
            width,
            height,
            components,
            bit_depth: 8,
            signed: false,
            options: &options,
            accelerator: &mut accelerator,
        });
        let device = accelerator
            .cuda_context()
            .expect("CUDA context")
            .expect("resident encoder")
            .diagnostics()
            .expect("device memory diagnostics");
        eprintln!("{width}x{height} components={components} device_peak_bytes={} device_retained_bytes={}",
            device.peak_live_device_bytes, device.live_device_bytes);
        let actual = actual.expect("CUDA encode with bounded code-block scratch");
        assert!(
            actual == expected,
            "{width}x{height}, {components} components"
        );
        assert_eq!(accelerator.htj2k_tile_dispatches(), usize::from(!precincts));
        assert!(accelerator.ht_code_block_dispatches() > 1);
        if precincts {
            assert!(accelerator.ht_subband_dispatches() > 0);
        }
    }
}
