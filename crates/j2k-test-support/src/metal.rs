// SPDX-License-Identifier: MIT OR Apache-2.0

//! Environment gates used by Metal tests and validation.

use crate::{gpu_device_unavailable_is_skip, gpu_test_gate};

const METAL_RUNTIME_GATE: &str = "J2K_REQUIRE_METAL_RUNTIME";

/// Returns true when tests should require a working Metal runtime.
pub fn metal_runtime_required() -> bool {
    std::env::var_os(METAL_RUNTIME_GATE).is_some()
}

/// Returns true when a Metal-gated test should run.
pub fn metal_runtime_gate(context: &str) -> bool {
    gpu_test_gate(metal_runtime_required(), METAL_RUNTIME_GATE, context)
}

/// Returns true when a Metal test should skip after observing device/runtime absence.
pub fn metal_device_unavailable_is_skip(context: &str) -> bool {
    gpu_device_unavailable_is_skip(metal_runtime_required(), METAL_RUNTIME_GATE, context)
}
