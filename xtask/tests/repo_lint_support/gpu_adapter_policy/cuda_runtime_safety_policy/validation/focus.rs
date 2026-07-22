// SPDX-License-Identifier: MIT OR Apache-2.0

use super::read_runtime;

#[test]
fn cuda_runtime_validation_leaves_stay_focused() {
    for (relative, max_lines) in [
        ("htj2k_decode/queued.rs", 150usize),
        ("htj2k_decode/queued/drop_guard.rs", 50),
        ("htj2k_decode/queued/lifecycle.rs", 80),
        ("htj2k_decode/status.rs", 100),
        ("htj2k_decode/status/tests.rs", 100),
        ("htj2k_encode/context_validation.rs", 100),
        ("j2k_decode/idwt/job_validation.rs", 200),
        ("j2k_decode/idwt/job_validation/tests.rs", 225),
        ("j2k_decode/store/batch.rs", 350),
        ("j2k_decode/store/destination.rs", 100),
        ("j2k_decode/store/validation.rs", 150),
    ] {
        let source = read_runtime(relative);
        let line_count = source.lines().count();
        assert!(
            line_count < max_lines,
            "crates/j2k-cuda-runtime/src/{relative} has {line_count} lines; split it before reaching {max_lines}"
        );
    }
}
