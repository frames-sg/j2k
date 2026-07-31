// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k::{
    EncodeBackendPreference, J2kEncodeDispatchReport, J2kEncodeValidation,
    J2kLosslessEncodeOptions, J2kLosslessSamples,
};
use j2k_core::BackendKind;
use j2k_cuda::CudaLosslessEncoder;

#[test]
fn external_consumer_can_use_the_preference_honoring_encoder() {
    fn assert_send<T: Send>() {}
    assert_send::<CudaLosslessEncoder>();

    let pixels = [41_u8; 16 * 16];
    let samples =
        J2kLosslessSamples::new(&pixels, 16, 16, 1, 8, false).expect("valid gray8 samples");
    let options = J2kLosslessEncodeOptions::default()
        .with_backend(EncodeBackendPreference::CpuOnly)
        .with_max_decomposition_levels(Some(0))
        .with_validation(J2kEncodeValidation::CpuRoundTrip);
    let mut encoder = CudaLosslessEncoder::new();

    let result = encoder
        .encode(samples, &options)
        .expect("CPU-only route must not require CUDA");

    assert_eq!(result.requested_backend(), EncodeBackendPreference::CpuOnly);
    assert_eq!(result.actual_backend(), BackendKind::Cpu);
    assert_eq!(result.fallback_reason(), None);
    assert_eq!(result.dispatch_report(), J2kEncodeDispatchReport::default());
    assert!(result.encoded().codestream.starts_with(&[0xff, 0x4f]));
    assert!(!result.into_encoded().codestream.is_empty());
}
