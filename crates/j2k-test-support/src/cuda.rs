// SPDX-License-Identifier: MIT OR Apache-2.0

//! Environment gates used by CUDA tests and benches.

use crate::{gpu_device_unavailable_is_skip, gpu_test_gate};

const CUDA_RUNTIME_GATE: &str = "J2K_REQUIRE_CUDA_RUNTIME";
const CUDA_OXIDE_BUILD_GATE: &str = "J2K_REQUIRE_CUDA_OXIDE_BUILD";
const CUDA_JPEG_HARDWARE_DECODE_GATE: &str = "J2K_REQUIRE_CUDA_JPEG_HARDWARE_DECODE";

/// Returns true when tests should require a working CUDA runtime.
pub fn cuda_runtime_required() -> bool {
    std::env::var_os(CUDA_RUNTIME_GATE).is_some()
}

/// Returns true when CUDA tests should require strict CUDA Oxide PTX generation.
pub fn cuda_strict_oxide_required() -> bool {
    std::env::var_os(CUDA_OXIDE_BUILD_GATE).is_some()
}

/// Returns true when JPEG CUDA tests should require hardware JPEG decode.
pub fn cuda_jpeg_hardware_decode_required() -> bool {
    std::env::var_os(CUDA_JPEG_HARDWARE_DECODE_GATE).is_some()
}

/// Returns true when a CUDA-runtime-gated test should run.
pub fn cuda_runtime_gate(context: &str) -> bool {
    gpu_test_gate(cuda_runtime_required(), CUDA_RUNTIME_GATE, context)
}

/// Returns true when a CUDA-Oxide-build-gated test should run.
pub fn cuda_strict_oxide_gate(context: &str) -> bool {
    gpu_test_gate(cuda_strict_oxide_required(), CUDA_OXIDE_BUILD_GATE, context)
}

/// Returns true when a JPEG CUDA hardware-decode-gated test should run.
pub fn cuda_jpeg_hardware_decode_gate(context: &str) -> bool {
    gpu_test_gate(
        cuda_jpeg_hardware_decode_required(),
        CUDA_JPEG_HARDWARE_DECODE_GATE,
        context,
    )
}

/// Returns true when a CUDA test should skip after observing device/runtime absence.
pub fn cuda_device_unavailable_is_skip(context: &str) -> bool {
    gpu_device_unavailable_is_skip(cuda_runtime_required(), CUDA_RUNTIME_GATE, context)
}

/// Returns true when a test requiring both CUDA runtime and strict Oxide should run.
pub fn cuda_runtime_and_strict_oxide_gate(context: &str) -> bool {
    cuda_runtime_gate(context) && cuda_strict_oxide_gate(context)
}
