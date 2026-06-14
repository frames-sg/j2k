// SPDX-License-Identifier: Apache-2.0

//! Environment gates used by CUDA tests and benches.

/// Returns true when tests should require a working CUDA runtime.
pub fn cuda_runtime_required() -> bool {
    std::env::var_os("SIGNINUM_REQUIRE_CUDA_RUNTIME").is_some()
}

/// Returns true when HTJ2K CUDA tests should require strict GPU execution.
pub fn cuda_htj2k_strict_required() -> bool {
    std::env::var_os("SIGNINUM_REQUIRE_CUDA_HTJ2K_STRICT").is_some()
}

/// Returns true when JPEG CUDA tests should require hardware JPEG decode.
pub fn cuda_jpeg_hardware_decode_required() -> bool {
    std::env::var_os("SIGNINUM_REQUIRE_CUDA_JPEG_HARDWARE_DECODE").is_some()
}

/// Returns true when CUDA benches should require the runtime instead of skipping.
pub fn cuda_bench_required() -> bool {
    std::env::var_os("SIGNINUM_REQUIRE_CUDA_BENCH").is_some()
        || std::env::var_os("SIGNINUM_REQUIRE_CUDA_RUNTIME").is_some()
}
