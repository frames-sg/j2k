// SPDX-License-Identifier: MIT OR Apache-2.0

//! Typed-error preservation for replicated JPEG Metal batch failures.

use std::fs;

use super::super::{assert_pattern_checks, repo_root, PatternCheck};

#[test]
fn jpeg_metal_batch_failures_clone_typed_errors_instead_of_rendering_them() {
    let root = repo_root();
    let jpeg_encoder = fs::read_to_string(root.join("crates/j2k-jpeg/src/encoder.rs"))
        .expect("read JPEG encoder errors");
    let metal_error = fs::read_to_string(root.join("crates/j2k-jpeg-metal/src/error.rs"))
        .expect("read JPEG Metal errors");
    let metal_batch = [
        "crates/j2k-jpeg-metal/src/batch.rs",
        "crates/j2k-jpeg-metal/src/batch/flush.rs",
    ]
    .into_iter()
    .map(|path| fs::read_to_string(root.join(path)).expect("read JPEG Metal batch owner"))
    .collect::<Vec<_>>()
    .join("\n");

    assert_pattern_checks(&[
        PatternCheck::new("cloneable JPEG encode errors", &jpeg_encoder)
            .required(&["#[derive(Clone, Debug, Error)]", "pub enum JpegEncodeError"]),
        PatternCheck::new("cloneable JPEG Metal errors", &metal_error).required(&[
            "#[derive(Clone, Debug, thiserror::Error)]",
            "pub enum Error",
        ]),
        PatternCheck::new("typed JPEG Metal batch error replication", &metal_batch)
            .required(&["store_completion_or_error(", "Err(error.clone())"])
            .forbidden(&[
                "fn batched_decode_error(",
                "batched JPEG Metal decode failed",
            ]),
    ]);
}
