// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;

use super::super::super::{assert_pattern_checks, repo_root, PatternCheck};

#[test]
fn submit_only_cuda_work_has_an_explicit_unsafe_allowlist() {
    let root = repo_root();
    let read = |relative: &str| {
        fs::read_to_string(root.join(relative))
            .unwrap_or_else(|error| panic!("read {relative}: {error}"))
    };
    let events = read("crates/j2k-cuda-runtime/src/execution/events.rs");
    let htj2k_decode = [
        read("crates/j2k-cuda-runtime/src/htj2k_decode/completion.rs"),
        read("crates/j2k-cuda-runtime/src/htj2k_decode/completion/dequant.rs"),
    ]
    .concat();
    let adapter_cleanup = read("crates/j2k-cuda/src/decoder/resident/cleanup_dequant.rs");
    let adapter_idwt = read("crates/j2k-cuda/src/decoder/resident/idwt.rs");
    let adapter_resident = format!("{adapter_cleanup}\n{adapter_idwt}");

    assert_pattern_checks(&[
        PatternCheck::new("CUDA submit-only contract", &events).required(&[
            "pub unsafe fn submit_default_stream_named<T>(",
            "A successful return proves only submission",
            "Prefer a typed `#[must_use]` queued guard",
        ]),
        PatternCheck::new("CUDA runtime submit-only allowlist", &htj2k_decode).required(&[
            "submit_default_stream_named(\"j2k.htj2k.decode.cleanup\"",
            "submit_default_stream_named(\"j2k.htj2k.decode.dequantize\"",
        ]),
        PatternCheck::new("CUDA adapter submit-only allowlist", &adapter_resident).required(&[
            "submit_default_stream_named(\"j2k.htj2k.decode.cleanup.batch\"",
            "submit_default_stream_named(\"j2k.htj2k.decode.idwt.batch\"",
        ]),
    ]);
    assert_eq!(
        htj2k_decode.matches("submit_default_stream_named(").count(),
        2
    );
    assert_eq!(
        adapter_resident
            .matches("submit_default_stream_named(")
            .count(),
        2
    );
    assert!(!htj2k_decode.contains("time_default_stream_named_us_if(false"));
    assert!(!adapter_resident.contains("time_default_stream_named_us_if(false"));
}
