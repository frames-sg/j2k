// SPDX-License-Identifier: MIT OR Apache-2.0

use super::read;
use crate::repo_lint_support::{assert_pattern_checks, PatternCheck};

#[test]
fn j2k_ml_accelerator_zero_copy_contracts_are_source_enforced() {
    let cuda_batch = read("crates/j2k-ml/src/cuda/batch.rs");
    let cuda_interop = read("crates/j2k-ml/src/cuda/interop.rs");
    let cuda_owners = format!("{cuda_batch}\n{cuda_interop}");
    let metal_batch = read("crates/j2k-ml/src/metal/batch.rs");
    let metal_interop = read("crates/j2k-ml/src/metal/interop.rs");
    let metal_owners = format!("{metal_batch}\n{metal_interop}");

    assert_pattern_checks(&[
        PatternCheck::new("CUDA Burn-owned destination", &cuda_owners)
            .required(&[
                "empty_device_contiguous_dtype",
                "external_write_stream",
                "with_primary_stream_ordering",
                "CudaExternalDeviceBufferViewMut::from_raw_parts(",
                "submit_batch_into",
                "register_tensor_handle(handle)",
            ])
            .forbidden(&[
                ".sync()",
                "TensorData",
                "copy_to_host(",
                "copy_range_to_host(",
                "Tensor::from_data(",
                "j2k_ml_convert_into_external",
            ]),
        PatternCheck::new("Metal Burn-owned destination", &metal_owners)
            .required(&[
                "checked_next_multiple_of(4)",
                "client.empty(tracked_len)",
                "CubeTensor::new_contiguous",
                "tracked_external_write_range(",
                "mark_external_write_initialized(initialized_range)",
                ".as_hal::<wgpu_hal::api::Metal>()",
                ".retained_raw_handle()",
                "MetalImageDestination::from_exclusive_buffer",
                "MetalBackendSession::with_command_queue",
                "submit_prepared_group_into_for_consumer_queue(",
                "register_tensor_handle(handle)",
            ])
            .forbidden(&[
                "download_surfaces_packed",
                "TensorData",
                "Tensor::from_data(",
                "integer_tensor_4_from_bytes",
                "empty_device_contiguous_dtype",
                ".enqueue_consumer_wait(",
            ]),
    ]);
}
