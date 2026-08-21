mod grayscale_external;
mod htj2k_encode;
mod pipeline;

use crate::htj2k_encode::{
    htj2k_encode_compact_jobs, htj2k_encode_compact_jobs_multi_input, CudaHtj2kEncodeCompactJob,
    CudaHtj2kEncodeKernelJob, CudaHtj2kEncodeMultiInputKernelJob, HTJ2K_ENCODE_OUTPUT_CAPACITY,
    HTJ2K_STATUS_OK, HTJ2K_STATUS_UNSUPPORTED, HTJ2K_UVLC_ENCODE_TABLE_BYTES,
};
use crate::j2k_decode::{
    checked_f32_words_byte_len, format_idwt_batch_trace_row, idwt_batch_kernel_mode,
    idwt_batch_trace_row, idwt_batch_uses_cooperative_53, CudaJ2kIdwtBatchKernelMode,
    CudaJ2kIdwtBatchStageProfile, CudaJ2kIdwtMultiKernelJob,
};
use crate::{
    CudaHtj2kEncodeCodeBlockJob, CudaHtj2kEncodeCodeBlockRegionJob, CudaHtj2kEncodeResidentTarget,
    CudaHtj2kEncodeStageTimings, CudaHtj2kEncodeStatus, CudaHtj2kEncodeTables,
    CudaHtj2kEncodedCodeBlock, CudaJ2kIdwtJob, CudaJ2kIdwtNormalization, CudaJ2kIdwtTarget,
    CudaJ2kQuantizeJob, CudaJ2kQuantizeSubbandRegionJob, CudaJ2kRect,
};
use j2k_cuda_runtime::{CudaContext, CudaError, CudaExecutionStats};

fn cuda_runtime_gate() -> bool {
    j2k_test_support::cuda_runtime_gate(module_path!())
}

#[test]
fn checked_f32_words_byte_len_rejects_multiplication_overflow() {
    assert_eq!(checked_f32_words_byte_len(2).expect("byte len"), 8);
    assert!(matches!(
        checked_f32_words_byte_len(usize::MAX),
        Err(j2k_cuda_runtime::CudaError::LengthTooLarge { len }) if len == usize::MAX
    ));
}
