// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;

use super::super::super::{assert_pattern_checks, repo_root, PatternCheck};

#[test]
fn gpu_jpeg_encode_counts_actual_host_capacities_and_preserves_typed_cuda_errors() {
    let root = repo_root();
    let read = |relative: &str| {
        fs::read_to_string(root.join(relative))
            .unwrap_or_else(|error| panic!("read {relative}: {error}"))
    };
    let metal_allocation = read("crates/j2k-jpeg-metal/src/encode/allocation.rs");
    let metal_adapter = read("crates/j2k-jpeg-metal/src/encode/adapter.rs");
    let metal_compute = format!(
        "{}\n{}",
        read("crates/j2k-jpeg-metal/src/compute.rs"),
        read("crates/j2k-jpeg-metal/src/compute/encode.rs")
    );
    let cuda_adapter = read("crates/j2k-jpeg-cuda/src/encode.rs");
    let cuda_mapping = read("crates/j2k-jpeg-cuda/src/runtime.rs");
    let cuda_runtime = format!(
        "{}\n{}",
        read("crates/j2k-cuda-runtime/src/jpeg/encode.rs"),
        read("crates/j2k-cuda-runtime/src/jpeg/encode_batch.rs")
    );
    let cuda_validation = read("crates/j2k-cuda-runtime/src/jpeg/encode_validation.rs");
    let cuda_tables = read("crates/j2k-cuda-runtime/src/jpeg/encode_validation/tables.rs");

    assert_pattern_checks(&[
        PatternCheck::new("Metal JPEG exact host phases", &metal_allocation).required(&[
            "checked_single_output_bytes",
            "checked_batch_conversion_bytes",
            "checked_batch_runtime_bytes",
            "status_capacity",
            "output_outer_capacity",
            "output_payload_capacity",
            "Device buffers are intentionally excluded",
        ]),
        PatternCheck::new("Metal JPEG post-allocation checks", &metal_adapter)
            .required(&["neutral_param_capacity", "params.capacity()"]),
        PatternCheck::new("Metal JPEG actual readback owners", &metal_compute).required(&[
            "entropy.capacity()",
            "statuses.capacity()",
            "status_slice.capacity()",
            "out.capacity()",
            "saturating_add(chunk.capacity())",
        ]),
        PatternCheck::new("CUDA adapter converted capacity", &cuda_adapter)
            .required(&["neutral_param_capacity", "params.capacity()"]),
        PatternCheck::new("CUDA runtime errors stay nested", &cuda_mapping)
            .required(&["Error::CudaRuntime { source: error }"])
            .forbidden(&["CudaError::HostAllocationFailed { bytes } =>"]),
        PatternCheck::new("CUDA JPEG actual result owners", &cuda_runtime).required(&[
            "statuses.capacity()",
            "out.capacity()",
            "saturating_add(chunk.capacity())",
            "checked_single_private_host_bytes(external_live_bytes, out.capacity())",
        ]),
        PatternCheck::new("CUDA range validation actual capacity", &cuda_validation).required(&[
            "retained_host_bytes",
            "HostPhaseBudget::new(\"JPEG baseline encode range validation\")",
            "host_budget.account_bytes(retained_host_bytes)?",
            "host_budget.try_vec_with_capacity(params.len())?",
        ]),
        PatternCheck::new("CUDA Huffman validation actual capacity", &cuda_tables).required(&[
            "retained_host_bytes",
            "HostPhaseBudget::new(\"JPEG baseline Huffman table validation\")",
            "host_budget.account_bytes(retained_host_bytes)?",
            "host_budget.try_vec_with_capacity(table.lens.len())?",
            "JPEG baseline Huffman table validation",
        ]),
    ]);
}
