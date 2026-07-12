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
    assert_eq!(accelerator.ht_code_block_dispatches(), 4);
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
    assert_eq!(accelerator.ht_code_block_dispatches(), 12);
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
    assert_eq!(accelerator.ht_code_block_dispatches(), 4);
    assert_eq!(accelerator.packetization_attempts(), 1);
    assert_eq!(accelerator.packetization_dispatches(), 1);
}
