// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(feature = "cuda-runtime")]
use super::{
    cuda_dwt53_output_to_j2k, cuda_dwt97_output_to_j2k, forward_dwt53_reference,
    forward_dwt97_reference, forward_ict_reference, try_deinterleave_reference, CudaContext,
    J2kDeinterleaveToF32Job,
};
#[cfg(feature = "cuda-runtime")]
use super::{
    encode_with_cuda_test_accelerator, CudaEncodeStageAccelerator, CudaTestEncodeRequest,
    DecodeSettings, EncodeOptions, Image, J2kEncodeStageAccelerator,
};

#[cfg(feature = "cuda-runtime")]
#[test]
fn cuda_deinterleave_stage_dispatches_when_runtime_required() {
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return;
    }

    let pixels = [0u8, 128, 255, 64, 32, 16];
    let mut accelerator = CudaEncodeStageAccelerator::default();
    let components = accelerator
        .encode_deinterleave(J2kDeinterleaveToF32Job {
            pixels: &pixels,
            num_pixels: 2,
            num_components: 3,
            bit_depth: 8,
            signed: false,
        })
        .expect("CUDA deinterleave hook")
        .expect("CUDA deinterleave dispatch");

    assert_eq!(accelerator.deinterleave_dispatches(), 1);
    assert_eq!(
        components,
        vec![vec![-128.0, -64.0], vec![0.0, -96.0], vec![127.0, -112.0]]
    );
}

#[cfg(feature = "cuda-runtime")]
#[test]
fn cuda_deinterleave_declines_more_than_four_components_for_cpu_fallback_when_runtime_required() {
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return;
    }

    let mut accelerator = CudaEncodeStageAccelerator::default();
    let components = accelerator
        .encode_deinterleave(J2kDeinterleaveToF32Job {
            pixels: &[0; 5],
            num_pixels: 1,
            num_components: 5,
            bit_depth: 8,
            signed: false,
        })
        .expect("unsupported CUDA component count should decline to the CPU stage");

    assert!(components.is_none());
    assert_eq!(accelerator.deinterleave_dispatches(), 0);
}

#[cfg(feature = "cuda-runtime")]
#[test]
fn cuda_forward_rct_dispatches_when_runtime_required() {
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return;
    }

    let pixels: Vec<u8> = (0u16..7 * 5 * 3)
        .map(|i| u8::try_from((i * 17) & 0xFF).expect("masked value fits in u8"))
        .collect();
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 0,
        ..EncodeOptions::default()
    };
    let mut accelerator = CudaEncodeStageAccelerator::default();

    let codestream = encode_with_cuda_test_accelerator(CudaTestEncodeRequest {
        pixels: &pixels,
        width: 7,
        height: 5,
        components: 3,
        bit_depth: 8,
        signed: false,
        options: &options,
        accelerator: &mut accelerator,
    })
    .expect("encode with CUDA forward RCT");
    let decoded = Image::new(&codestream, &DecodeSettings::default())
        .expect("codestream parses")
        .decode_native()
        .expect("codestream decodes");

    assert_eq!(decoded.data, pixels);
    assert_eq!(accelerator.forward_rct_attempts(), 1);
    assert_eq!(accelerator.forward_rct_dispatches(), 1);
}

#[cfg(feature = "cuda-runtime")]
#[test]
fn cuda_forward_ict_dispatches_when_runtime_required() {
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return;
    }

    let pixels: Vec<u8> = (0u32..32 * 32 * 3)
        .map(|i| u8::try_from((i * 23 + 19) & 0xFF).expect("masked value fits in u8"))
        .collect();
    let options = EncodeOptions {
        reversible: false,
        use_ht_block_coding: true,
        num_decomposition_levels: 1,
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
    .expect("encode irreversible RGB with CUDA forward ICT");
    let decoded = Image::new(&codestream, &DecodeSettings::default())
        .expect("codestream parses")
        .decode_native()
        .expect("codestream decodes");

    assert_eq!(decoded.data.len(), pixels.len());
    assert_eq!(accelerator.forward_ict_attempts(), 1);
    assert_eq!(accelerator.forward_ict_dispatches(), 1);
}

#[cfg(feature = "cuda-runtime")]
#[test]
fn cuda_forward_ict_matches_native_for_external_parity_fixture_when_required() {
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return;
    }
    let Some(path) = std::env::var_os("J2K_CUDA_LOSSY_PARITY_PNM") else {
        return;
    };
    let image = j2k_test_support::read_pnm_image(path).expect("read parity PNM fixture");
    assert_eq!(image.channels, 3, "ICT parity fixture must be RGB");
    let num_pixels = usize::try_from(image.width)
        .expect("fixture width fits usize")
        .checked_mul(usize::try_from(image.height).expect("fixture height fits usize"))
        .expect("fixture sample count");
    let native_planes = try_deinterleave_reference(&image.pixels, num_pixels, 3, 8, false)
        .expect("native deinterleave");
    let expected = forward_ict_reference(native_planes.clone());
    let mut actual = native_planes;
    let context = CudaContext::system_default().expect("CUDA context");
    let (plane0, rest) = actual.split_at_mut(1);
    let (plane1, plane2) = rest.split_at_mut(1);
    context
        .j2k_forward_ict(&mut plane0[0], &mut plane1[0], &mut plane2[0])
        .expect("CUDA forward ICT");

    for (component, (actual_plane, expected_plane)) in actual.iter().zip(&expected).enumerate() {
        if let Some(index) = actual_plane
            .iter()
            .zip(expected_plane)
            .position(|(actual, expected)| actual.to_bits() != expected.to_bits())
        {
            let pixel = &image.pixels[index * 3..index * 3 + 3];
            panic!(
                "forward ICT differs at component {component}, sample {index}, RGB={pixel:?}: CPU={:?} ({:x?}), CUDA={:?} ({:x?})",
                expected.iter().map(|plane| plane[index]).collect::<Vec<_>>(),
                expected
                    .iter()
                    .map(|plane| plane[index].to_bits())
                    .collect::<Vec<_>>(),
                actual.iter().map(|plane| plane[index]).collect::<Vec<_>>(),
                actual
                    .iter()
                    .map(|plane| plane[index].to_bits())
                    .collect::<Vec<_>>(),
            );
        }
    }
}

#[cfg(feature = "cuda-runtime")]
#[test]
fn cuda_forward_dwt97_matches_native_for_external_parity_fixture_when_required() {
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return;
    }
    let Some(path) = std::env::var_os("J2K_CUDA_LOSSY_PARITY_PNM") else {
        return;
    };
    let image = j2k_test_support::read_pnm_image(path).expect("read parity PNM fixture");
    assert_eq!(image.channels, 3, "DWT parity fixture must be RGB");
    let num_pixels = usize::try_from(image.width)
        .expect("fixture width fits usize")
        .checked_mul(usize::try_from(image.height).expect("fixture height fits usize"))
        .expect("fixture sample count");
    let native_planes = try_deinterleave_reference(&image.pixels, num_pixels, 3, 8, false)
        .expect("native deinterleave");
    let transformed = forward_ict_reference(native_planes);
    let context = CudaContext::system_default().expect("CUDA context");

    for (component, plane) in transformed.iter().enumerate() {
        let expected = forward_dwt97_reference(plane, image.width, image.height, 3)
            .expect("native forward DWT 9/7 reference");
        let cuda = context
            .j2k_forward_dwt97(plane, image.width, image.height, 3)
            .expect("CUDA forward DWT 9/7");
        let actual = cuda_dwt97_output_to_j2k(&cuda).expect("reshape CUDA forward DWT 9/7");

        assert_f32_bits_equal(component, "LL", &actual.ll, &expected.ll);
        assert_eq!(actual.levels.len(), expected.levels.len());
        for (level, (actual, expected)) in actual.levels.iter().zip(&expected.levels).enumerate() {
            assert_f32_bits_equal(
                component,
                &format!("level {level} HL"),
                &actual.hl,
                &expected.hl,
            );
            assert_f32_bits_equal(
                component,
                &format!("level {level} LH"),
                &actual.lh,
                &expected.lh,
            );
            assert_f32_bits_equal(
                component,
                &format!("level {level} HH"),
                &actual.hh,
                &expected.hh,
            );
        }
    }
}

#[cfg(feature = "cuda-runtime")]
fn assert_f32_bits_equal(component: usize, band: &str, actual: &[f32], expected: &[f32]) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "component {component} {band} length"
    );
    if let Some(index) = actual
        .iter()
        .zip(expected)
        .position(|(actual, expected)| actual.to_bits() != expected.to_bits())
    {
        panic!(
            "forward DWT 9/7 differs at component {component} {band} sample {index}: CPU={} ({:#010x}), CUDA={} ({:#010x})",
            expected[index],
            expected[index].to_bits(),
            actual[index],
            actual[index].to_bits(),
        );
    }
}

#[cfg(feature = "cuda-runtime")]
#[test]
fn cuda_forward_dwt53_dispatches_when_runtime_required() {
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return;
    }

    let pixels: Vec<u8> = (0u16..8 * 8)
        .map(|i| u8::try_from((i * 5) & 0xFF).expect("masked value fits in u8"))
        .collect();
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 1,
        ..EncodeOptions::default()
    };
    let mut accelerator = CudaEncodeStageAccelerator::default();

    let codestream = encode_with_cuda_test_accelerator(CudaTestEncodeRequest {
        pixels: &pixels,
        width: 8,
        height: 8,
        components: 1,
        bit_depth: 8,
        signed: false,
        options: &options,
        accelerator: &mut accelerator,
    })
    .expect("encode with CUDA forward DWT 5/3");
    let decoded = Image::new(&codestream, &DecodeSettings::default())
        .expect("codestream parses")
        .decode_native()
        .expect("codestream decodes");

    assert_eq!(decoded.data, pixels);
    assert_eq!(accelerator.forward_dwt53_attempts(), 1);
    assert_eq!(accelerator.forward_dwt53_dispatches(), 2);
}

#[cfg(feature = "cuda-runtime")]
fn assert_cuda_forward_dwt53_reshape_matches_native(width: u32, height: u32, num_levels: u8) {
    let samples: Vec<f32> = (0u32..width * height)
        .map(|i| {
            let value = i16::try_from((i * 7 + 3) % 256).expect("sample fits in i16") - 128;
            f32::from(value)
        })
        .collect();

    let native = forward_dwt53_reference(&samples, width, height, num_levels)
        .expect("native forward DWT 5/3 reference");
    let context = CudaContext::system_default().expect("CUDA context");
    let cuda_output = context
        .j2k_forward_dwt53(&samples, width, height, num_levels)
        .expect("CUDA forward DWT 5/3");
    let cuda_as_native = cuda_dwt53_output_to_j2k(&cuda_output)
        .expect("CUDA DWT output reshapes to native subbands");

    assert_eq!(
        cuda_as_native.levels.len(),
        native.levels.len(),
        "reshaped level count (levels={num_levels})"
    );
    assert_eq!(
        (cuda_as_native.ll_width, cuda_as_native.ll_height),
        (native.ll_width, native.ll_height),
        "reshaped LL dimensions (levels={num_levels})"
    );
    for (level_idx, (cuda_level, native_level)) in cuda_as_native
        .levels
        .iter()
        .zip(native.levels.iter())
        .enumerate()
    {
        assert_eq!(
            cuda_level.hl, native_level.hl,
            "levels={num_levels} level {level_idx} HL mismatch"
        );
        assert_eq!(
            cuda_level.lh, native_level.lh,
            "levels={num_levels} level {level_idx} LH mismatch"
        );
        assert_eq!(
            cuda_level.hh, native_level.hh,
            "levels={num_levels} level {level_idx} HH mismatch"
        );
    }
    assert_eq!(
        cuda_as_native.ll, native.ll,
        "levels={num_levels} final LL mismatch"
    );
}

#[cfg(feature = "cuda-runtime")]
#[test]
fn cuda_forward_dwt53_private_reshape_matches_native_reference_when_required() {
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return;
    }

    assert_cuda_forward_dwt53_reshape_matches_native(40, 24, 1);
    assert_cuda_forward_dwt53_reshape_matches_native(40, 24, 2);
    assert_cuda_forward_dwt53_reshape_matches_native(40, 24, 3);
}

#[cfg(feature = "cuda-runtime")]
#[test]
fn cuda_forward_dwt97_dispatches_when_runtime_required() {
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return;
    }

    let pixels: Vec<u8> = (0u16..32 * 32)
        .map(|i| u8::try_from((i * 7 + 13) & 0xFF).expect("masked value fits in u8"))
        .collect();
    let options = EncodeOptions {
        reversible: false,
        use_ht_block_coding: true,
        num_decomposition_levels: 1,
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
    .expect("encode with CUDA forward DWT 9/7");
    let decoded = Image::new(&codestream, &DecodeSettings::default())
        .expect("codestream parses")
        .decode_native()
        .expect("codestream decodes");

    assert_eq!(decoded.data.len(), pixels.len());
    assert_eq!(accelerator.forward_dwt97_attempts(), 1);
    assert_eq!(accelerator.forward_dwt97_dispatches(), 3);
}

#[cfg(feature = "cuda-runtime")]
#[test]
fn cuda_quantize_subband_dispatches_when_runtime_required() {
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return;
    }

    let pixels: Vec<u8> = (0u16..32 * 32)
        .map(|i| u8::try_from((i * 19 + 5) & 0xFF).expect("masked value fits in u8"))
        .collect();
    let options = EncodeOptions {
        reversible: false,
        use_ht_block_coding: true,
        num_decomposition_levels: 1,
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
    .expect("encode with CUDA quantization");
    let decoded = Image::new(&codestream, &DecodeSettings::default())
        .expect("codestream parses")
        .decode_native()
        .expect("codestream decodes");

    assert_eq!(decoded.data.len(), pixels.len());
    assert_eq!(accelerator.quantize_subband_attempts(), 4);
    assert_eq!(accelerator.quantize_subband_dispatches(), 4);
}
