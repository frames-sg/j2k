// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;

use super::super::super::{assert_pattern_checks, repo_root, PatternCheck};

mod focus;
mod htj2k_output;
mod j2k_idwt;
mod j2k_store;
mod jpeg_decode;
mod jpeg_decode_device;
mod jpeg_decode_regressions;
mod jpeg_encode;
mod queued;

fn read_repo(relative: &str) -> String {
    let path = repo_root().join(relative);
    fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()))
}

fn read_runtime(relative: &str) -> String {
    read_repo(&format!("crates/j2k-cuda-runtime/src/{relative}"))
}

#[test]
fn cuda_runtime_validation_policy_modules_stay_focused() {
    for (relative, source, max_lines) in [
        ("focus.rs", include_str!("validation/focus.rs"), 100usize),
        (
            "htj2k_output.rs",
            include_str!("validation/htj2k_output.rs"),
            175,
        ),
        (
            "htj2k_output/planning.rs",
            include_str!("validation/htj2k_output/planning.rs"),
            75,
        ),
        (
            "htj2k_output/sources.rs",
            include_str!("validation/htj2k_output/sources.rs"),
            75,
        ),
        (
            "jpeg_decode.rs",
            include_str!("validation/jpeg_decode.rs"),
            250,
        ),
        (
            "jpeg_decode/preflight.rs",
            include_str!("validation/jpeg_decode/preflight.rs"),
            225,
        ),
        (
            "jpeg_decode/preflight/ordering.rs",
            include_str!("validation/jpeg_decode/preflight/ordering.rs"),
            175,
        ),
        (
            "jpeg_decode_device.rs",
            include_str!("validation/jpeg_decode_device.rs"),
            125,
        ),
        (
            "jpeg_decode_regressions.rs",
            include_str!("validation/jpeg_decode_regressions.rs"),
            100,
        ),
        (
            "jpeg_encode.rs",
            include_str!("validation/jpeg_encode.rs"),
            250,
        ),
        ("j2k_idwt.rs", include_str!("validation/j2k_idwt.rs"), 150),
        ("j2k_store.rs", include_str!("validation/j2k_store.rs"), 150),
        ("queued.rs", include_str!("validation/queued.rs"), 125),
        (
            "queued/ordering.rs",
            include_str!("validation/queued/ordering.rs"),
            50,
        ),
    ] {
        let line_count = source.lines().count();
        assert!(
            line_count < max_lines,
            "CUDA runtime validation policy {relative} has {line_count} lines; split it before reaching {max_lines}"
        );
    }
}
