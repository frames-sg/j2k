// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    assert_strict_cuda_classic_tier1_error, cuda_packetization_plan_fallback_reason,
    encode_j2k_lossless_with_cuda, encode_j2k_lossless_with_cuda_and_profile,
    encode_with_cuda_test_accelerator, CudaEncodeStageAccelerator, CudaHtj2kPacketizationPlanError,
    CudaTestEncodeRequest, DecodeSettings, EncodeBackendPreference, EncodeOptions, Image,
    J2kBlockCodingMode, J2kEncodeStageAccelerator, J2kEncodeStageError, J2kEncodeValidation,
    J2kHtSubbandEncodeJob, J2kLosslessEncodeOptions, J2kLosslessSamples, J2kPacketizationEncodeJob,
    J2kPacketizationProgressionOrder, J2kQuantizeSubbandJob,
};
#[cfg(feature = "cuda-runtime")]
use super::{
    cuda_resident_input_error, encode_j2k_lossy_with_accelerator, BackendKind,
    J2kLossyEncodeOptions, J2kLossySamples, J2kResidentEncodeInputError,
};

#[cfg(feature = "cuda-runtime")]
#[test]
fn typed_resident_input_failures_map_to_stable_cuda_rejections() {
    let cases = [
        (
            J2kResidentEncodeInputError::EmptyGeometry {
                width: 0,
                height: 8,
            },
            "resident encode input dimensions must be non-zero",
        ),
        (
            J2kResidentEncodeInputError::ComponentCountOutOfRange { num_components: 0 },
            "resident encode input component count must be in 1..=16384",
        ),
        (
            J2kResidentEncodeInputError::PrecisionOutOfRange { bit_depth: 0 },
            "resident encode input bit depth must be in 1..=38",
        ),
        (
            J2kResidentEncodeInputError::AddressSpaceOverflow,
            "resident encode input dimensions overflow address space",
        ),
    ];

    for (input_error, expected_reason) in cases {
        let mapped = cuda_resident_input_error(input_error);
        assert!(matches!(
            mapped,
            crate::Error::UnsupportedCudaRequest { reason } if reason == expected_reason
        ));
    }
}

#[test]
fn cuda_lossless_encode_auto_errors_for_unsupported_classic_tier1() {
    let pixels: Vec<u8> = (0u32..128 * 128)
        .map(|value| u8::try_from((value * 17 + 5) & 0xFF).expect("masked value fits in u8"))
        .collect();
    let samples =
        J2kLosslessSamples::new(&pixels, 128, 128, 1, 8, false).expect("valid gray8 samples");
    let options = J2kLosslessEncodeOptions::default()
        .with_backend(EncodeBackendPreference::Auto)
        .with_block_coding_mode(J2kBlockCodingMode::Classic)
        .with_max_decomposition_levels(Some(0))
        .with_validation(J2kEncodeValidation::CpuRoundTrip);

    let err = encode_j2k_lossless_with_cuda(samples, &options)
        .expect_err("CUDA-named encode must not silently return CPU fallback");

    assert_strict_cuda_classic_tier1_error(&err, "strict CUDA encode");
}

#[test]
fn cuda_lossless_encode_profile_auto_errors_for_unsupported_classic_tier1() {
    let pixels: Vec<u8> = (0u32..128 * 128)
        .map(|value| u8::try_from((value * 19 + 7) & 0xFF).expect("masked value fits in u8"))
        .collect();
    let samples =
        J2kLosslessSamples::new(&pixels, 128, 128, 1, 8, false).expect("valid gray8 samples");
    let options = J2kLosslessEncodeOptions::default()
        .with_backend(EncodeBackendPreference::Auto)
        .with_block_coding_mode(J2kBlockCodingMode::Classic)
        .with_max_decomposition_levels(Some(0))
        .with_validation(J2kEncodeValidation::External);

    let err = encode_j2k_lossless_with_cuda_and_profile(samples, &options)
        .expect_err("profiled CUDA encode must not silently return CPU fallback");

    assert_strict_cuda_classic_tier1_error(&err, "profiled strict CUDA encode");
}

#[test]
fn cuda_lossless_encode_require_device_errors_for_unsupported_classic_tier1() {
    let pixels: Vec<u8> = (0u32..128 * 128)
        .map(|value| u8::try_from((value * 29 + 11) & 0xFF).expect("masked value fits in u8"))
        .collect();
    let samples =
        J2kLosslessSamples::new(&pixels, 128, 128, 1, 8, false).expect("valid gray8 samples");
    let options = J2kLosslessEncodeOptions::default()
        .with_backend(EncodeBackendPreference::RequireDevice)
        .with_block_coding_mode(J2kBlockCodingMode::Classic)
        .with_max_decomposition_levels(Some(0))
        .with_validation(J2kEncodeValidation::External);

    let err = encode_j2k_lossless_with_cuda(samples, &options)
        .expect_err("strict CUDA encode must not silently fall back to CPU");

    assert_strict_cuda_classic_tier1_error(&err, "strict CUDA encode");
}

#[test]
fn prefer_cpu_ht_subband_declines_fused_subband_but_counts_attempts() {
    let mut accelerator = CudaEncodeStageAccelerator::default()
        .prefer_cpu_ht_subband(true)
        .prefer_cpu_quantize_subband(true);
    let output = accelerator
        .encode_ht_subband(J2kHtSubbandEncodeJob {
            coefficients: &[0.0; 16],
            width: 4,
            height: 4,
            step_exponent: 8,
            step_mantissa: 0,
            range_bits: 8,
            reversible: false,
            code_block_width: 4,
            code_block_height: 4,
            total_bitplanes: 9,
        })
        .expect("subband hook can decline");

    assert!(output.is_none());
    assert_eq!(accelerator.ht_subband_attempts(), 1);
    assert_eq!(accelerator.quantize_subband_attempts(), 1);
    assert_eq!(accelerator.ht_code_block_attempts(), 1);
    assert_eq!(accelerator.dispatch_report().total(), 0);

    let quantized = accelerator
        .encode_quantize_subband(J2kQuantizeSubbandJob {
            coefficients: &[0.0; 16],
            step_exponent: 8,
            step_mantissa: 0,
            range_bits: 8,
            reversible: false,
        })
        .expect("quantize hook can decline");
    assert!(quantized.is_none());
    assert_eq!(accelerator.quantize_subband_attempts(), 2);
    assert_eq!(accelerator.dispatch_report().total(), 0);
}

#[cfg(feature = "cuda-runtime")]
#[test]
fn cuda_lossy_htj2k_facade_require_device_dispatches_supported_stages_when_runtime_required() {
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return;
    }

    let pixels: Vec<u8> = (0u32..64 * 64)
        .map(|value| u8::try_from((value * 41 + 17) & 0xFF).expect("masked value fits in u8"))
        .collect();
    let samples = J2kLossySamples::new(&pixels, 64, 64, 1, 8, false).expect("valid gray8 samples");
    let options = J2kLossyEncodeOptions::default()
        .with_backend(EncodeBackendPreference::RequireDevice)
        .with_block_coding_mode(J2kBlockCodingMode::HighThroughput)
        .with_max_decomposition_levels(Some(1))
        .with_validation(J2kEncodeValidation::CpuRoundTrip);
    let mut accelerator = CudaEncodeStageAccelerator::default();

    let encoded =
        encode_j2k_lossy_with_accelerator(samples, &options, BackendKind::Cuda, &mut accelerator)
            .expect("strict CUDA HTJ2K lossy facade encode should dispatch supported stages");
    let decoded = Image::new(&encoded.codestream, &DecodeSettings::default())
        .expect("codestream parses")
        .decode_native()
        .expect("codestream decodes");

    assert_eq!(encoded.backend, BackendKind::Cuda);
    assert_eq!(decoded.width, 64);
    assert_eq!(decoded.height, 64);
    assert_eq!(decoded.num_components, 1);
    assert_eq!(accelerator.deinterleave_dispatches(), 1);
    assert!(accelerator.forward_dwt97_dispatches() > 0);
    assert_eq!(accelerator.quantize_subband_dispatches(), 4);
    assert_eq!(accelerator.ht_code_block_dispatches(), 4);
    assert_eq!(accelerator.packetization_dispatches(), 1);
}

#[test]
fn cuda_encode_stage_accelerator_preserves_cpu_codestream_validity() {
    let pixels: Vec<u8> = (0u8..192).collect();
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
        components: 3,
        bit_depth: 8,
        signed: false,
        options: &options,
        accelerator: &mut accelerator,
    })
    .expect("encode with CUDA stage accelerator");
    let decoded = Image::new(&codestream, &DecodeSettings::default())
        .expect("codestream parses")
        .decode_native()
        .expect("codestream decodes");

    assert_eq!(decoded.width, 8);
    assert_eq!(decoded.height, 8);
    assert_eq!(decoded.num_components, 3);
    assert_eq!(decoded.bit_depth, 8);
    assert_eq!(accelerator.forward_rct_attempts(), 1);
    assert_eq!(accelerator.forward_dwt53_attempts(), 3);
    assert!(accelerator.tier1_code_block_attempts() > 0);
    assert_eq!(accelerator.packetization_attempts(), 1);
}

#[test]
fn cuda_auto_host_output_declines_packetization_before_flattening() {
    let mut accelerator = CudaEncodeStageAccelerator::for_auto_host_output();
    let invalid_for_cuda_flattening = J2kPacketizationEncodeJob {
        resolution_count: 1,
        num_layers: 1,
        num_components: 3,
        code_block_count: 0,
        progression_order: J2kPacketizationProgressionOrder::Lrcp,
        packet_descriptors: &[],
        resolutions: &[],
    };

    let encoded = J2kEncodeStageAccelerator::encode_packetization(
        &mut accelerator,
        invalid_for_cuda_flattening,
    )
    .expect("Auto host-output CUDA packetization should decline to CPU");

    assert!(encoded.is_none());
    assert_eq!(accelerator.packetization_attempts(), 1);
    assert_eq!(accelerator.packetization_dispatches(), 0);
}

#[test]
fn cuda_invalid_packetization_plan_falls_back_after_classification() {
    let mut accelerator = CudaEncodeStageAccelerator::default();
    let invalid_for_cuda_flattening = J2kPacketizationEncodeJob {
        resolution_count: 1,
        num_layers: 1,
        num_components: 1,
        code_block_count: 0,
        progression_order: J2kPacketizationProgressionOrder::Lrcp,
        packet_descriptors: &[],
        resolutions: &[],
    };

    let encoded = J2kEncodeStageAccelerator::encode_packetization(
        &mut accelerator,
        invalid_for_cuda_flattening,
    )
    .expect("an invalid CUDA packetization plan should decline to the CPU route");

    assert!(encoded.is_none());
    assert_eq!(accelerator.packetization_attempts(), 1);
    assert_eq!(accelerator.packetization_dispatches(), 0);
}

#[test]
fn cuda_packetization_host_allocation_is_a_hard_stage_error() {
    let invalid_reason = "invalid packetization plan";
    assert_eq!(
        cuda_packetization_plan_fallback_reason(CudaHtj2kPacketizationPlanError::Invalid(
            invalid_reason,
        )),
        Ok(invalid_reason)
    );

    let allocation_error = CudaHtj2kPacketizationPlanError::HostAllocation {
        what: "CUDA packetization test plan",
        bytes: 4096,
    };
    assert!(matches!(
        cuda_packetization_plan_fallback_reason(allocation_error),
        Err(J2kEncodeStageError::HostAllocationFailed {
            what: "CUDA packetization test plan",
            bytes: 4096,
        })
    ));
}
