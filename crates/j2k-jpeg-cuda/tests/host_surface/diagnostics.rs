// SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg(feature = "cuda-runtime")]

use j2k_core::CodecError;
use j2k_jpeg_cuda::{Codec, CudaSession, Error};
use j2k_test_support::cuda_runtime_gate;

use super::support::generated_rgb_jpeg;

#[test]
fn generated_420_chunked_entropy_diagnostic_runs_when_cuda_runtime_required() {
    if !cuda_runtime_gate(module_path!()) {
        return;
    }

    let input = generated_rgb_jpeg(j2k_jpeg::JpegSubsampling::Ybr420, 256, 256);
    let mut session = CudaSession::default();
    let report = Codec::diagnose_tile_rgb8_chunked_entropy_with_session(
        &input,
        j2k_cuda_runtime::CudaJpegChunkedEntropyConfig {
            subsequence_words: 64,
            sequence_len: 32,
            max_overflow_subsequences: 4,
        },
        &mut session,
    )
    .expect("chunked entropy diagnostic");

    assert!(report.subsequence_count() > 0);
    assert_eq!(report.failed_state_count(), 0);
}

#[test]
fn generated_422_chunked_entropy_diagnostic_returns_diagnostic_420_only_error() {
    let input = generated_rgb_jpeg(j2k_jpeg::JpegSubsampling::Ybr422, 256, 256);
    let mut session = CudaSession::default();
    let error = Codec::diagnose_tile_rgb8_chunked_entropy_with_session(
        &input,
        j2k_cuda_runtime::CudaJpegChunkedEntropyConfig {
            subsequence_words: 64,
            sequence_len: 32,
            max_overflow_subsequences: 4,
        },
        &mut session,
    )
    .expect_err("4:2:2 input should be rejected before diagnostic runtime");

    assert!(error.is_unsupported());
    match error {
        Error::UnsupportedCudaRequest { reason } => {
            assert!(reason.contains("chunked entropy diagnostic"));
            assert!(reason.contains("4:2:0"));
        }
        other => panic!("expected unsupported CUDA diagnostic error, got {other:?}"),
    }
}

#[test]
fn generated_420_chunked_entropy_diagnostic_rejects_invalid_config_before_runtime() {
    let input = generated_rgb_jpeg(j2k_jpeg::JpegSubsampling::Ybr420, 256, 256);
    let mut session = CudaSession::default();
    let error = Codec::diagnose_tile_rgb8_chunked_entropy_with_session(
        &input,
        j2k_cuda_runtime::CudaJpegChunkedEntropyConfig {
            subsequence_words: 0,
            sequence_len: 32,
            max_overflow_subsequences: 4,
        },
        &mut session,
    )
    .expect_err("invalid diagnostic config should be rejected before runtime");

    assert!(error.is_unsupported());
    match error {
        Error::UnsupportedCudaRequest { reason } => {
            assert!(reason.contains("chunked entropy diagnostic"));
            assert!(reason.contains("config"));
        }
        other => panic!("expected unsupported CUDA diagnostic config error, got {other:?}"),
    }
}
