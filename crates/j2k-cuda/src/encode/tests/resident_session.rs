// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(feature = "cuda-runtime")]
use super::{
    encode_lossless_from_cuda_buffer_to_cuda_buffer_with_report,
    encode_lossless_from_cuda_buffers_to_cuda_buffers_with_report,
    strict_cuda_resident_lossless_options, BackendKind, CudaContext, CudaLosslessEncodeTile,
    CudaSession, DecodeSettings, Image, PixelFormat,
};
#[cfg(feature = "cuda-runtime")]
use j2k_cuda_runtime::CudaDeviceBuffer;

#[cfg(feature = "cuda-runtime")]
fn gray8_tile(buffer: &CudaDeviceBuffer, width: u32, height: u32) -> CudaLosslessEncodeTile<'_> {
    CudaLosslessEncodeTile {
        buffer,
        byte_offset: 0,
        width,
        height,
        pitch_bytes: width as usize,
        output_width: width,
        output_height: height,
        format: PixelFormat::Gray8,
    }
}

#[cfg(feature = "cuda-runtime")]
#[test]
fn resident_encode_binds_external_context_and_clones_reuse_resources_when_required() {
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return;
    }

    let width = 8;
    let height = 8;
    let pixels = vec![23u8; width as usize * height as usize];
    let context = CudaContext::system_default().expect("external CUDA context");
    let buffer = context.upload(&pixels).expect("resident source pixels");
    let tile = gray8_tile(&buffer, width, height);
    let mut session = CudaSession::default();
    assert!(!session.is_runtime_initialized());

    let first = encode_lossless_from_cuda_buffer_to_cuda_buffer_with_report(
        tile,
        &strict_cuda_resident_lossless_options(),
        &mut session,
    )
    .expect("first resident encode binds the external context");
    assert_eq!(first.encoded.metadata.backend, BackendKind::Cuda);
    assert!(session
        .cuda_context()
        .expect("bound session context")
        .is_same_context(&context));
    assert_eq!(session.htj2k_encode_resource_uploads_for_test(), 1);
    assert_eq!(session.submissions(), 1);

    let mut cloned = session.clone();
    let second = encode_lossless_from_cuda_buffer_to_cuda_buffer_with_report(
        tile,
        &strict_cuda_resident_lossless_options(),
        &mut cloned,
    )
    .expect("cloned session reuses the context-bound resource");
    assert_eq!(second.encoded.metadata.backend, BackendKind::Cuda);
    assert_eq!(cloned.htj2k_encode_resource_uploads_for_test(), 1);
    assert_eq!(cloned.submissions(), 2);
    assert_eq!(session.submissions(), 1);
}

#[cfg(feature = "cuda-runtime")]
#[test]
fn cuda_lossless_buffer_batch_encode_returns_resident_codestreams_in_order_when_runtime_required() {
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return;
    }

    let width = 32;
    let height = 32;
    let inputs = [
        (0u32..width * height)
            .map(|value| u8::try_from((value * 17 + 3) & 0xFF).expect("masked value fits in u8"))
            .collect::<Vec<_>>(),
        (0u32..width * height)
            .map(|value| u8::try_from((value * 31 + 97) & 0xFF).expect("masked value fits in u8"))
            .collect::<Vec<_>>(),
    ];
    let mut session = CudaSession::default();
    let context = session.cuda_context().expect("CUDA context");
    let buffers = inputs
        .iter()
        .map(|pixels| context.upload(pixels).expect("resident source pixels"))
        .collect::<Vec<_>>();
    let tiles = buffers
        .iter()
        .map(|buffer| gray8_tile(buffer, width, height))
        .collect::<Vec<_>>();

    let outcomes = encode_lossless_from_cuda_buffers_to_cuda_buffers_with_report(
        &tiles,
        &strict_cuda_resident_lossless_options(),
        &mut session,
    )
    .expect("strict CUDA resident codestream batch encode");

    assert_eq!(outcomes.len(), inputs.len());
    assert_eq!(session.htj2k_encode_resource_uploads_for_test(), 1);
    assert_eq!(session.submissions(), inputs.len() as u64);
    for (outcome, expected_pixels) in outcomes.iter().zip(inputs.iter()) {
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
        assert_eq!(decoded.data.as_slice(), expected_pixels.as_slice());
    }
}

#[cfg(feature = "cuda-runtime")]
#[test]
fn resident_encode_rejects_session_context_mismatch_before_resource_upload_when_required() {
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return;
    }

    let width = 8;
    let height = 8;
    let pixels = vec![17u8; width as usize * height as usize];
    let mut session = CudaSession::default();
    let session_context = session.cuda_context().expect("session CUDA context");
    let other_context = CudaContext::system_default().expect("independent CUDA context");
    assert!(!session_context.is_same_context(&other_context));
    let session_buffer = session_context
        .upload(&pixels)
        .expect("session-context resident pixels");
    let other_buffer = other_context
        .upload(&pixels)
        .expect("other-context resident pixels");

    encode_lossless_from_cuda_buffer_to_cuda_buffer_with_report(
        gray8_tile(&session_buffer, width, height),
        &strict_cuda_resident_lossless_options(),
        &mut session,
    )
    .expect("encode in the session context");
    let mismatched = encode_lossless_from_cuda_buffer_to_cuda_buffer_with_report(
        gray8_tile(&other_buffer, width, height),
        &strict_cuda_resident_lossless_options(),
        &mut session,
    );
    let Err(error) = mismatched else {
        panic!("a tile from another CUDA context must be rejected");
    };

    assert!(matches!(
        error,
        crate::Error::UnsupportedCudaRequest { reason }
            if reason == "J2K CUDA encode tile belongs to a different context than the session"
    ));
    assert_eq!(session.htj2k_encode_resource_uploads_for_test(), 1);
    assert_eq!(session.submissions(), 2);
}
