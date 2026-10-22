// SPDX-License-Identifier: MIT OR Apache-2.0

//! Environment gates used by CUDA tests and benches.

use crate::{gpu_device_unavailable_is_skip, gpu_test_gate, GPU_TEST_SKIP_MARKER};

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

/// Returns true when a CUDA test should skip because `error` reports PTX that
/// cuda-oxide did not build, as on a CUDA host whose build could not run it.
///
/// # Panics
///
/// Panics when `J2K_REQUIRE_CUDA_OXIDE_BUILD` is set, so self-hosted
/// validation fails instead of skipping.
pub fn cuda_oxide_ptx_missing_is_skip<E: core::fmt::Debug + ?Sized>(
    error: &E,
    context: &str,
) -> bool {
    oxide_ptx_missing_is_skip(cuda_strict_oxide_required(), error, context)
}

fn oxide_ptx_missing_is_skip<E: core::fmt::Debug + ?Sized>(
    required: bool,
    error: &E,
    context: &str,
) -> bool {
    // The runtime and engine PTX guards name the gate in this exact phrase.
    let error = format!("{error:?}");
    if !error.contains(&format!("PTX was not built; set {CUDA_OXIDE_BUILD_GATE}")) {
        return false;
    }
    assert!(
        !required,
        "{CUDA_OXIDE_BUILD_GATE} is set but cuda-oxide PTX was not built for {context}: {error}"
    );
    eprintln!(
        "{GPU_TEST_SKIP_MARKER} gate={CUDA_OXIDE_BUILD_GATE} context={context} reason=oxide-ptx-not-built"
    );
    true
}

#[cfg(test)]
mod tests {
    use super::oxide_ptx_missing_is_skip;

    /// What a CUDA backend returns when cuda-oxide could not build its PTX.
    const PTX_NOT_BUILT: &str = "cuda-oxide HTJ2K decode PTX was not built; set J2K_REQUIRE_CUDA_OXIDE_BUILD on a Linux cuda-oxide host to require it";

    #[test]
    fn unbuilt_oxide_ptx_skips_unless_the_build_is_required() {
        assert!(oxide_ptx_missing_is_skip(false, PTX_NOT_BUILT, "test"));
    }

    #[test]
    #[should_panic(expected = "J2K_REQUIRE_CUDA_OXIDE_BUILD is set")]
    fn unbuilt_oxide_ptx_fails_when_the_build_is_required() {
        oxide_ptx_missing_is_skip(true, PTX_NOT_BUILT, "test");
    }

    #[test]
    fn other_backend_errors_never_skip() {
        // A disabled kernel feature is a wiring bug, not a host without cuda-oxide.
        let feature_disabled = "CUDA Oxide PTX was not built for HTJ2K decode; enable j2k-cuda-j2k-engine/cuda-oxide-htj2k-decode or an adapter feature that implies it";
        let invalid_argument = "CUDA invalid argument: destination pitch is too small";
        for required in [false, true] {
            assert!(!oxide_ptx_missing_is_skip(
                required,
                feature_disabled,
                "test"
            ));
            assert!(!oxide_ptx_missing_is_skip(
                required,
                invalid_argument,
                "test"
            ));
        }
    }
}
