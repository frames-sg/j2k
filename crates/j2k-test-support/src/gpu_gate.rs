// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared GPU test gates.

/// Marker emitted when an optional GPU test skips because its require gate is unset.
pub const GPU_TEST_SKIP_MARKER: &str = "J2K_GPU_TEST_SKIPPED";

/// Returns true when a GPU-gated test should run.
///
/// If `required` is false, the test should return after calling this helper.
/// The skip message is intentionally stable so CI can count skipped GPU tests.
pub fn gpu_test_gate(required: bool, gate: &'static str, context: &str) -> bool {
    if required {
        return true;
    }
    eprintln!("{GPU_TEST_SKIP_MARKER} gate={gate} context={context}");
    false
}

/// Handles a missing GPU device/runtime for a test that already attempted setup.
///
/// Returns true when the caller should skip. Panics when the matching require
/// gate is set, making self-hosted validation fail closed.
///
/// # Panics
///
/// Panics when `required` is true, because a required GPU runtime was missing.
pub fn gpu_device_unavailable_is_skip(required: bool, gate: &'static str, context: &str) -> bool {
    assert!(
        !required,
        "{gate} is set but GPU device/runtime is unavailable for {context}"
    );
    eprintln!("{GPU_TEST_SKIP_MARKER} gate={gate} context={context} reason=device-unavailable");
    true
}
