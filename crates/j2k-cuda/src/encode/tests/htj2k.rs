// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(feature = "cuda-runtime")]
use super::{
    cuda_htj2k_encode_tables, CudaContext, CudaHtj2kEncodeCodeBlockJob,
    CudaHtj2kEncodeCodeBlockRegionJob, CudaJ2kQuantizeJob, J2kHtCodeBlockEncodeJob,
};
#[cfg(feature = "cuda-runtime")]
use super::{
    encode_with_cuda_test_accelerator, CudaEncodeStageAccelerator, CudaTestEncodeRequest,
    DecodeSettings, EncodeOptions, Image, J2kEncodeStageAccelerator,
};

#[cfg(feature = "cuda-runtime")]
#[test]
fn cuda_htj2k_codeblock_dispatches_when_runtime_required() {
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return;
    }

    let pixels: Vec<u8> = (0u16..8 * 8)
        .map(|i| u8::try_from((i * 11 + 3) & 0xFF).expect("masked value fits in u8"))
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
        width: 8,
        height: 8,
        components: 1,
        bit_depth: 8,
        signed: false,
        options: &options,
        accelerator: &mut accelerator,
    })
    .expect("encode HTJ2K with CUDA HT codeblock kernel");
    let decoded = Image::new(&codestream, &DecodeSettings::default())
        .expect("codestream parses")
        .decode_native()
        .expect("codestream decodes");

    assert_eq!(decoded.data, pixels);
    assert!(accelerator.ht_code_block_attempts() > 0);
    assert!(accelerator.ht_code_block_dispatches() > 0);
    assert!(accelerator.ht_code_block_dispatches() <= accelerator.ht_code_block_attempts());
    assert_eq!(
        accelerator.dispatch_report().ht_code_block,
        accelerator.ht_code_block_dispatches()
    );
}

#[cfg(feature = "cuda-runtime")]
#[test]
fn cuda_htj2k_codeblock_preserves_requested_refinement_passes_when_runtime_required() {
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return;
    }

    let coefficients = [0, 3, -5, 3, 5, 0, -3, 3, 7, -3, 0, 3, 0, 0, 5, -5];
    let mut accelerator = CudaEncodeStageAccelerator::default();

    let encoded = accelerator
        .encode_ht_code_block(J2kHtCodeBlockEncodeJob {
            coefficients: &coefficients,
            width: 4,
            height: 4,
            total_bitplanes: 4,
            target_coding_passes: 2,
        })
        .expect("CUDA HTJ2K code-block encode hook")
        .expect("CUDA HTJ2K code-block encode output");

    assert_eq!(encoded.num_coding_passes, 2);
    assert_eq!(encoded.num_zero_bitplanes, 2);
    assert_eq!(encoded.refinement_length, 1);
    assert_eq!(
        encoded.cleanup_length + encoded.refinement_length,
        u32::try_from(encoded.data.len()).expect("test payload length fits u32")
    );
    assert_eq!(accelerator.ht_code_block_dispatches(), 1);
}

#[cfg(feature = "cuda-runtime")]
#[test]
fn cuda_htj2k_codeblock_batch_uses_single_dispatch_when_runtime_required() {
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return;
    }

    let pixels: Vec<u8> = (0u16..32 * 32)
        .map(|i| u8::try_from((i * 17 + 9) & 0xFF).expect("masked value fits in u8"))
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
    .expect("encode HTJ2K with CUDA HT batch codeblock kernel");
    let decoded = Image::new(&codestream, &DecodeSettings::default())
        .expect("codestream parses")
        .decode_native()
        .expect("codestream decodes");

    assert_eq!(decoded.data, pixels);
    assert!(accelerator.ht_code_block_attempts() > 1);
    assert_eq!(accelerator.ht_code_block_dispatches(), 1);
    assert!(
        accelerator.ht_code_block_dispatches() < accelerator.ht_code_block_attempts(),
        "batch encode must not launch one kernel per codeblock"
    );
    assert_eq!(
        accelerator.dispatch_report().ht_code_block,
        accelerator.ht_code_block_dispatches()
    );
}

#[cfg(feature = "cuda-runtime")]
#[test]
fn cuda_resident_quantized_subband_feeds_resident_ht_batch_when_runtime_required() {
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return;
    }

    let samples = [-3.6f32, -2.5, -0.4, 0.0, 0.49, 1.5, 3.2, 9.9];
    let context = CudaContext::system_default().expect("CUDA context");
    let sample_buffer = context.upload_f32(&samples).expect("resident samples");
    let quantization = CudaJ2kQuantizeJob {
        step_exponent: 8,
        step_mantissa: 0,
        range_bits: 8,
        reversible: true,
    };
    let resident_quantized = context
        .j2k_quantize_subband_resident(&sample_buffer, samples.len(), quantization)
        .expect("resident quantization");
    let host_quantized = context
        .j2k_quantize_subband(&samples, quantization)
        .expect("host-staged quantization");
    let jobs = [CudaHtj2kEncodeCodeBlockJob {
        coefficient_offset: 0,
        width: 4,
        height: 2,
        total_bitplanes: 5,
        target_coding_passes: 1,
    }];

    let resident_encoded = context
        .encode_htj2k_codeblocks_resident(
            resident_quantized.buffer(),
            resident_quantized.coefficient_count(),
            &jobs,
            cuda_htj2k_encode_tables(),
        )
        .expect("resident HTJ2K encode");
    let staged_encoded = context
        .encode_htj2k_codeblocks(
            host_quantized.coefficients(),
            &jobs,
            cuda_htj2k_encode_tables(),
        )
        .expect("host-staged HTJ2K encode");

    assert_eq!(resident_quantized.coefficient_count(), samples.len());
    assert_eq!(resident_encoded.execution().kernel_dispatches(), 1);
    assert_eq!(
        resident_encoded.code_blocks().len(),
        staged_encoded.code_blocks().len()
    );
    for (resident, staged) in resident_encoded
        .code_blocks()
        .iter()
        .zip(staged_encoded.code_blocks())
    {
        assert_eq!(resident.data(), staged.data());
        assert_eq!(resident.cleanup_length(), staged.cleanup_length());
        assert_eq!(resident.refinement_length(), staged.refinement_length());
        assert_eq!(resident.num_coding_passes(), staged.num_coding_passes());
        assert_eq!(resident.num_zero_bitplanes(), staged.num_zero_bitplanes());
    }
}

#[cfg(feature = "cuda-runtime")]
#[test]
fn cuda_resident_strided_codeblock_region_matches_host_gather_when_runtime_required() {
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return;
    }

    let samples: Vec<f32> = (0u16..16).map(|value| f32::from(value) - 8.0).collect();
    let context = CudaContext::system_default().expect("CUDA context");
    let sample_buffer = context.upload_f32(&samples).expect("resident samples");
    let quantization = CudaJ2kQuantizeJob {
        step_exponent: 8,
        step_mantissa: 0,
        range_bits: 8,
        reversible: true,
    };
    let resident_quantized = context
        .j2k_quantize_subband_resident(&sample_buffer, samples.len(), quantization)
        .expect("resident quantization");
    let quantized = resident_quantized
        .download_coefficients()
        .expect("download quantized coefficients");
    let gathered_codeblock = vec![quantized[5], quantized[6], quantized[9], quantized[10]];
    let region_jobs = [CudaHtj2kEncodeCodeBlockRegionJob {
        coefficient_offset: 5,
        coefficient_stride: 4,
        width: 2,
        height: 2,
        total_bitplanes: 5,
        target_coding_passes: 1,
    }];
    let contiguous_jobs = [CudaHtj2kEncodeCodeBlockJob {
        coefficient_offset: 0,
        width: 2,
        height: 2,
        total_bitplanes: 5,
        target_coding_passes: 1,
    }];

    let resident_encoded = context
        .encode_htj2k_codeblock_regions_resident(
            resident_quantized.buffer(),
            resident_quantized.coefficient_count(),
            &region_jobs,
            cuda_htj2k_encode_tables(),
        )
        .expect("resident strided HTJ2K encode");
    let staged_encoded = context
        .encode_htj2k_codeblocks(
            &gathered_codeblock,
            &contiguous_jobs,
            cuda_htj2k_encode_tables(),
        )
        .expect("host-gathered HTJ2K encode");

    assert_eq!(resident_encoded.execution().kernel_dispatches(), 1);
    assert_eq!(resident_encoded.code_blocks().len(), 1);
    assert_eq!(
        resident_encoded.code_blocks()[0].data(),
        staged_encoded.code_blocks()[0].data()
    );
    assert_eq!(
        resident_encoded.code_blocks()[0].num_zero_bitplanes(),
        staged_encoded.code_blocks()[0].num_zero_bitplanes()
    );
}
