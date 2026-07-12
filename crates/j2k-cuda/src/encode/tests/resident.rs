// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(feature = "cuda-runtime")]
use super::{
    encode_j2k_lossless_with_cuda, encode_j2k_lossless_with_cuda_and_profile, DecodeSettings,
    EncodeBackendPreference, Image, J2kBlockCodingMode, J2kEncodeValidation,
    J2kLosslessEncodeOptions, J2kLosslessSamples,
};
#[cfg(feature = "cuda-runtime")]
use super::{
    encode_lossless_from_cuda_buffer_to_cuda_buffer_with_report,
    strict_cuda_resident_lossless_options, BackendKind, CudaLosslessEncodeTile, CudaSession,
    PixelFormat,
};

#[cfg(feature = "cuda-runtime")]
#[test]
fn cuda_lossless_encode_require_device_dispatches_cleanup_packetization_when_runtime_required() {
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return;
    }

    let pixels: Vec<u8> = (0u16..8 * 8)
        .map(|value| u8::try_from((value * 31 + 7) & 0xFF).expect("masked value fits in u8"))
        .collect();
    let samples = J2kLosslessSamples::new(&pixels, 8, 8, 1, 8, false).expect("valid gray8 samples");
    let options = J2kLosslessEncodeOptions::default()
        .with_backend(EncodeBackendPreference::RequireDevice)
        .with_block_coding_mode(J2kBlockCodingMode::HighThroughput)
        .with_max_decomposition_levels(Some(0))
        .with_validation(J2kEncodeValidation::CpuRoundTrip);

    let encoded = encode_j2k_lossless_with_cuda(samples, &options)
        .expect("strict CUDA single-pass HT encode should dispatch all required stages");
    let decoded = Image::new(&encoded.codestream, &DecodeSettings::default())
        .expect("codestream parses")
        .decode_native()
        .expect("codestream decodes");

    assert_eq!(encoded.backend, BackendKind::Cuda);
    assert_eq!(decoded.data, pixels);
}

#[cfg(feature = "cuda-runtime")]
#[test]
fn cuda_lossless_buffer_encode_returns_resident_codestream_when_runtime_required() {
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return;
    }

    let width = 64;
    let height = 64;
    let pixels: Vec<u8> = (0u32..width * height)
        .map(|value| u8::try_from((value * 23 + 11) & 0xFF).expect("masked value fits in u8"))
        .collect();
    let mut session = CudaSession::default();
    let context = session.cuda_context().expect("CUDA context");
    let buffer = context.upload(&pixels).expect("resident source pixels");
    let tile = CudaLosslessEncodeTile {
        buffer: &buffer,
        byte_offset: 0,
        width,
        height,
        pitch_bytes: width as usize,
        output_width: width,
        output_height: height,
        format: PixelFormat::Gray8,
    };

    let outcome = encode_lossless_from_cuda_buffer_to_cuda_buffer_with_report(
        tile,
        &strict_cuda_resident_lossless_options(),
        &mut session,
    )
    .expect("strict CUDA resident codestream encode");
    let downloaded = outcome
        .encoded
        .codestream
        .download()
        .expect("download resident codestream");
    let decoded = Image::new(&downloaded, &DecodeSettings::default())
        .expect("resident codestream parses")
        .decode_native()
        .expect("resident codestream decodes");

    assert_eq!(outcome.encoded.metadata.backend, BackendKind::Cuda);
    assert_eq!(outcome.encoded.codestream.byte_len(), downloaded.len());
    assert!(!outcome.resident.codestream_assembly_used);
    assert_eq!(decoded.data, pixels);
}

#[cfg(feature = "cuda-runtime")]
#[test]
fn cuda_lossless_encode_require_device_dispatches_multi_block_cleanup_when_runtime_required() {
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return;
    }

    let pixels: Vec<u8> = (0u32..128 * 128)
        .map(|value| u8::try_from((value * 19 + 23) & 0xFF).expect("masked value fits in u8"))
        .collect();
    let samples =
        J2kLosslessSamples::new(&pixels, 128, 128, 1, 8, false).expect("valid gray8 samples");
    let options = J2kLosslessEncodeOptions::default()
        .with_backend(EncodeBackendPreference::RequireDevice)
        .with_block_coding_mode(J2kBlockCodingMode::HighThroughput)
        .with_max_decomposition_levels(Some(0))
        .with_validation(J2kEncodeValidation::CpuRoundTrip);

    let encoded = encode_j2k_lossless_with_cuda(samples, &options)
        .expect("strict CUDA multi-block cleanup encode should dispatch all required stages");
    let decoded = Image::new(&encoded.codestream, &DecodeSettings::default())
        .expect("codestream parses")
        .decode_native()
        .expect("codestream decodes");

    assert_eq!(encoded.backend, BackendKind::Cuda);
    assert_eq!(decoded.data, pixels);
}

#[cfg(feature = "cuda-runtime")]
#[test]
fn cuda_lossless_encode_require_device_dispatches_dwt53_cleanup_when_runtime_required() {
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return;
    }

    let pixels: Vec<u8> = (0u32..128 * 128)
        .map(|value| u8::try_from((value * 37 + 41) & 0xFF).expect("masked value fits in u8"))
        .collect();
    let samples =
        J2kLosslessSamples::new(&pixels, 128, 128, 1, 8, false).expect("valid gray8 samples");
    let options = J2kLosslessEncodeOptions::default()
        .with_backend(EncodeBackendPreference::RequireDevice)
        .with_block_coding_mode(J2kBlockCodingMode::HighThroughput)
        .with_max_decomposition_levels(Some(1))
        .with_validation(J2kEncodeValidation::CpuRoundTrip);

    let encoded = encode_j2k_lossless_with_cuda(samples, &options)
        .expect("strict CUDA DWT cleanup encode should dispatch all required stages");
    let decoded = Image::new(&encoded.codestream, &DecodeSettings::default())
        .expect("codestream parses")
        .decode_native()
        .expect("codestream decodes");

    assert_eq!(encoded.backend, BackendKind::Cuda);
    assert_eq!(decoded.data, pixels);
}

#[cfg(feature = "cuda-runtime")]
#[test]
fn cuda_lossless_encode_profile_reports_resident_stage_timings_when_runtime_required() {
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return;
    }

    let pixels: Vec<u8> = (0u32..128 * 128)
        .map(|value| u8::try_from((value * 43 + 29) & 0xFF).expect("masked value fits in u8"))
        .collect();
    let samples =
        J2kLosslessSamples::new(&pixels, 128, 128, 1, 8, false).expect("valid gray8 samples");
    let options = J2kLosslessEncodeOptions::default()
        .with_backend(EncodeBackendPreference::RequireDevice)
        .with_block_coding_mode(J2kBlockCodingMode::HighThroughput)
        .with_max_decomposition_levels(Some(1))
        .with_validation(J2kEncodeValidation::CpuRoundTrip);

    let (encoded, report) = encode_j2k_lossless_with_cuda_and_profile(samples, &options)
        .expect("strict CUDA profiled DWT cleanup encode should dispatch all required stages");
    let decoded = Image::new(&encoded.codestream, &DecodeSettings::default())
        .expect("codestream parses")
        .decode_native()
        .expect("codestream decodes");

    assert_eq!(encoded.backend, BackendKind::Cuda);
    assert_eq!(decoded.data, pixels);
    assert_eq!(report.backend, BackendKind::Cuda);
    assert_eq!(report.input_bytes, pixels.len());
    assert_eq!(report.codestream_bytes, encoded.codestream.len());
    assert!(report.dispatch_count > 0);
    assert!(report.block_count > 0);
    assert!(report.deinterleave_us > 0);
    assert_eq!(report.mct_us, 0);
    assert!(report.dwt_us > 0);
    assert!(report.quantize_us > 0);
    assert!(report.ht_encode_us > 0);
    assert!(report.packetize_us > 0);
    assert!(report.total_us > 0);
}

#[cfg(feature = "cuda-runtime")]
#[test]
fn cuda_lossless_encode_require_device_dispatches_rgb_rct_cleanup_when_runtime_required() {
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return;
    }

    let pixels: Vec<u8> = (0u32..128 * 128 * 3)
        .map(|value| u8::try_from((value * 13 + 71) & 0xFF).expect("masked value fits in u8"))
        .collect();
    let samples =
        J2kLosslessSamples::new(&pixels, 128, 128, 3, 8, false).expect("valid rgb8 samples");
    let options = J2kLosslessEncodeOptions::default()
        .with_backend(EncodeBackendPreference::RequireDevice)
        .with_block_coding_mode(J2kBlockCodingMode::HighThroughput)
        .with_max_decomposition_levels(Some(1))
        .with_validation(J2kEncodeValidation::CpuRoundTrip);

    let encoded = encode_j2k_lossless_with_cuda(samples, &options)
        .expect("strict CUDA RGB cleanup encode should dispatch all required stages");
    let decoded = Image::new(&encoded.codestream, &DecodeSettings::default())
        .expect("codestream parses")
        .decode_native()
        .expect("codestream decodes");

    assert_eq!(encoded.backend, BackendKind::Cuda);
    assert_eq!(decoded.data, pixels);
}
