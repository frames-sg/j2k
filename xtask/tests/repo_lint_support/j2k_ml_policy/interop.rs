// SPDX-License-Identifier: MIT OR Apache-2.0

use super::read;
use crate::repo_lint_support::{assert_pattern_checks, PatternCheck};

#[test]
fn j2k_ml_accelerator_upload_contracts_are_source_enforced() {
    let cuda_batch = read("crates/j2k-ml/src/cuda/batch.rs");
    let metal_batch = read("crates/j2k-ml/src/metal/batch.rs");
    let staging = read("crates/j2k-ml/src/staging.rs");
    let cuda_upload = format!("{cuda_batch}\n{staging}");
    let metal_upload = format!("{metal_batch}\n{staging}");

    assert_pattern_checks(&[
        PatternCheck::new("CUDA staged upload", &cuda_upload)
            .required(&[
                "pub struct CudaUploadBurnDecoder",
                "SubmittedCudaResidentBatch",
                "copy_to_host(",
                "Tensor::from_data(",
            ])
            .forbidden(&[
                "external_write_stream",
                "CudaExternalDeviceBufferViewMut",
                "submit_batch_into",
                "register_tensor_handle",
            ]),
        PatternCheck::new("Metal staged upload", &metal_upload)
            .required(&[
                "pub struct MetalUploadBurnDecoder",
                "SubmittedMetalPreparedBatch",
                "Tensor::from_data(",
            ])
            .forbidden(&[
                "retained_raw_handle",
                "mark_external_write_initialized",
                "MetalImageDestination",
                "register_tensor_handle",
            ]),
    ]);
}
