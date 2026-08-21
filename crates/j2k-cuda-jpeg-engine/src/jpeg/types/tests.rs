// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    CudaJpegChunkedEntropyConfig, CudaJpegChunkedEntropyPlan, CudaJpegChunkedEntropyReport,
    CudaJpegEntropyOverflowState, CudaJpegEntropySyncState, CudaJpegHuffmanTable,
};
use crate::jpeg::jpeg_entropy_overflow_count;
use crate::{CudaContext, JpegCudaEngine};
use j2k_cuda_runtime::CudaExecutionStats;

#[test]
fn chunked_entropy_config_counts_bit_subsequences() {
    let config = CudaJpegChunkedEntropyConfig {
        subsequence_words: 4,
        sequence_len: 8,
        max_overflow_subsequences: 2,
    };
    assert_eq!(config.subsequence_bits(), 128);
    assert_eq!(config.subsequence_count_for_entropy_bytes(0).unwrap(), 0);
    assert_eq!(config.subsequence_count_for_entropy_bytes(1).unwrap(), 1);
    assert_eq!(config.subsequence_count_for_entropy_bytes(16).unwrap(), 1);
    assert_eq!(config.subsequence_count_for_entropy_bytes(17).unwrap(), 2);
}

#[test]
fn chunked_entropy_report_has_one_less_overflow_than_subsequence_count() {
    let config = CudaJpegChunkedEntropyConfig {
        subsequence_words: 1,
        sequence_len: 8,
        max_overflow_subsequences: 2,
    };
    let subsequences = config.subsequence_count_for_entropy_bytes(16).unwrap();
    assert_eq!(subsequences, 4);
    assert_eq!(jpeg_entropy_overflow_count(subsequences), 3);
    assert_eq!(jpeg_entropy_overflow_count(0), 0);
}

#[test]
fn chunked_entropy_config_rejects_zero_and_bit_overflow() {
    let zero_words = CudaJpegChunkedEntropyConfig {
        subsequence_words: 0,
        ..CudaJpegChunkedEntropyConfig::default()
    };
    let zero_sequence = CudaJpegChunkedEntropyConfig {
        sequence_len: 0,
        ..CudaJpegChunkedEntropyConfig::default()
    };
    let overflow = CudaJpegChunkedEntropyConfig {
        subsequence_words: (u32::MAX / 32) + 1,
        ..CudaJpegChunkedEntropyConfig::default()
    };
    assert!(zero_words.validate().is_err());
    assert!(zero_sequence.validate().is_err());
    assert!(overflow.validate().is_err());
    assert!(overflow.subsequence_count_for_entropy_bytes(1).is_err());
}

#[test]
fn chunked_entropy_report_summarizes_sync_quality() {
    let report = CudaJpegChunkedEntropyReport {
        config: CudaJpegChunkedEntropyConfig {
            subsequence_words: 4,
            sequence_len: 8,
            max_overflow_subsequences: 2,
        },
        entropy_bytes: 4096,
        states: vec![
            CudaJpegEntropySyncState {
                code: 0,
                start_bit: 0,
                end_bit: 128,
                bit_pos: 128,
                symbol_count: 10,
                block_phase: 0,
                zigzag_index: 0,
                reserved: 0,
            },
            CudaJpegEntropySyncState {
                code: 0,
                start_bit: 128,
                end_bit: 256,
                bit_pos: 256,
                symbol_count: 9,
                block_phase: 3,
                zigzag_index: 12,
                reserved: 0,
            },
        ],
        overflows: vec![CudaJpegEntropyOverflowState {
            code: 0,
            from_subsequence: 0,
            to_subsequence: 1,
            overflow_bits: 96,
            synchronized: 1,
            reserved: [0; 3],
        }],
        execution: CudaExecutionStats::new(2, 0, 0, false),
    };
    assert_eq!(report.subsequence_count(), 2);
    assert_eq!(report.synchronized_overflow_count(), 1);
    assert_eq!(report.max_overflow_bits(), Some(96));
    assert_eq!(report.failed_state_count(), 0);
}

#[test]
fn entropy_self_sync_returns_empty_report_for_empty_entropy_when_runtime_required() {
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return;
    }
    let context = CudaContext::system_default().expect("cuda context");
    let table = CudaJpegHuffmanTable::from_jpeg_bits_values([0; 16], 0, [0; 256])
        .expect("empty huffman table");
    let plan = CudaJpegChunkedEntropyPlan {
        config: CudaJpegChunkedEntropyConfig::default(),
        entropy_bytes: &[],
        y_dc_table: table,
        y_ac_table: table,
        cb_dc_table: table,
        cb_ac_table: table,
        cr_dc_table: table,
        cr_ac_table: table,
    };
    let report = JpegCudaEngine::new(&context)
        .diagnose_jpeg_420_entropy_self_sync(&plan)
        .expect("empty diagnostic report");
    assert_eq!(report.subsequence_count(), 0);
    assert!(report.overflows.is_empty());
}

#[cfg(all(
    feature = "cuda-oxide-jpeg-decode",
    not(j2k_cuda_oxide_jpeg_decode_built)
))]
#[test]
fn missing_decode_build_error_mentions_strict_gate() {
    let error = crate::build_flags::ensure_jpeg_decode_ptx_built()
        .expect_err("missing JPEG Oxide PTX should be reported");
    let message = error.to_string();
    assert!(message.contains("cuda-oxide JPEG decode PTX was not built"));
    assert!(message.contains("J2K_REQUIRE_CUDA_OXIDE_BUILD"));
}

#[cfg(all(feature = "cuda-oxide-jpeg-decode", j2k_cuda_oxide_jpeg_decode_built))]
#[test]
fn entropy_self_sync_decodes_zero_stream_when_required() {
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return;
    }
    let mut bits = [0u8; 16];
    bits[0] = 1;
    let table = CudaJpegHuffmanTable::from_jpeg_bits_values(bits, 1, [0; 256]).expect("zero table");
    let entropy = [0u8; 2];
    let context = CudaContext::system_default().expect("cuda context");
    let plan = CudaJpegChunkedEntropyPlan {
        config: CudaJpegChunkedEntropyConfig {
            subsequence_words: 1,
            sequence_len: 8,
            max_overflow_subsequences: 1,
        },
        entropy_bytes: &entropy,
        y_dc_table: table,
        y_ac_table: table,
        cb_dc_table: table,
        cb_ac_table: table,
        cr_dc_table: table,
        cr_ac_table: table,
    };
    let report = JpegCudaEngine::new(&context)
        .diagnose_jpeg_420_entropy_self_sync(&plan)
        .expect("cuda-oxide JPEG entropy self-sync");
    assert_eq!(report.subsequence_count(), 1);
    assert!(report.overflows.is_empty());
    assert_eq!(report.execution.kernel_dispatches(), 1);
    assert_eq!(report.states[0].code, 0);
    assert_eq!(report.states[0].start_bit, 0);
    assert_eq!(report.states[0].end_bit, 16);
    assert_eq!(report.states[0].bit_pos, 16);
}
