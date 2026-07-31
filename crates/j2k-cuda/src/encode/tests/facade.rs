// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(not(feature = "cuda-runtime"))]
use super::CudaEncodeFallbackReason;
use super::{
    assert_strict_cuda_classic_tier1_error, BackendKind, CudaLosslessEncoder,
    EncodeBackendPreference, J2kBlockCodingMode, J2kEncodeValidation, J2kLosslessEncodeOptions,
    J2kLosslessSamples,
};

#[cfg(not(feature = "cuda-runtime"))]
#[test]
fn cuda_encoder_auto_reports_unavailable_fallback_and_reuses_after_success() {
    let pixels = vec![17u8; 16 * 16];
    let samples =
        J2kLosslessSamples::new(&pixels, 16, 16, 1, 8, false).expect("valid gray8 samples");
    let options = J2kLosslessEncodeOptions::default()
        .with_backend(EncodeBackendPreference::Auto)
        .with_block_coding_mode(J2kBlockCodingMode::HighThroughput)
        .with_max_decomposition_levels(Some(0))
        .with_validation(J2kEncodeValidation::CpuRoundTrip);
    let mut encoder = CudaLosslessEncoder::new();

    for _ in 0..2 {
        let result = encoder
            .encode(samples, &options)
            .expect("Auto must fall back when CUDA support is not compiled");

        assert_eq!(result.requested_backend(), EncodeBackendPreference::Auto);
        assert_eq!(result.actual_backend(), BackendKind::Cpu);
        assert_eq!(
            result.fallback_reason(),
            Some(CudaEncodeFallbackReason::DeviceUnavailable)
        );
        assert_eq!(
            result.dispatch_report(),
            j2k::J2kEncodeDispatchReport::default()
        );
        assert!(!result.encoded().codestream.is_empty());
    }
}

#[test]
fn cuda_encoder_cpu_only_reports_no_fallback() {
    let pixels = vec![23u8; 16 * 16];
    let samples =
        J2kLosslessSamples::new(&pixels, 16, 16, 1, 8, false).expect("valid gray8 samples");
    let options = J2kLosslessEncodeOptions::default()
        .with_backend(EncodeBackendPreference::CpuOnly)
        .with_max_decomposition_levels(Some(0))
        .with_validation(J2kEncodeValidation::CpuRoundTrip);
    let mut encoder = CudaLosslessEncoder::new();

    let result = encoder
        .encode(samples, &options)
        .expect("CPU-only encode must not require CUDA");

    assert_eq!(result.requested_backend(), EncodeBackendPreference::CpuOnly);
    assert_eq!(result.actual_backend(), BackendKind::Cpu);
    assert_eq!(result.fallback_reason(), None);
    assert_eq!(
        result.dispatch_report(),
        j2k::J2kEncodeDispatchReport::default()
    );
}

#[test]
fn cuda_encoder_reuses_after_strict_route_failure() {
    let pixels = vec![31u8; 16 * 16];
    let samples =
        J2kLosslessSamples::new(&pixels, 16, 16, 1, 8, false).expect("valid gray8 samples");
    let cpu_template = J2kLosslessEncodeOptions::default()
        .with_backend(EncodeBackendPreference::CpuOnly)
        .with_block_coding_mode(J2kBlockCodingMode::Classic)
        .with_max_decomposition_levels(Some(0))
        .with_validation(J2kEncodeValidation::External);
    let cpu_options = cpu_template.with_validation(J2kEncodeValidation::CpuRoundTrip);
    let mut encoder = CudaLosslessEncoder::new();

    let error = encoder
        .encode_strict_cuda(samples, &cpu_template)
        .expect_err("strict method must override the CPU-only template");
    assert_strict_cuda_classic_tier1_error(&error, "reusable strict CUDA encoder");

    let result = encoder
        .encode(samples, &cpu_options)
        .expect("a route failure must not poison later jobs");
    assert_eq!(result.requested_backend(), EncodeBackendPreference::CpuOnly);
    assert_eq!(result.actual_backend(), BackendKind::Cpu);
    assert_eq!(result.fallback_reason(), None);
}

#[test]
fn cuda_lossless_encoder_is_send() {
    fn assert_send<T: Send>() {}

    assert_send::<CudaLosslessEncoder>();
}
