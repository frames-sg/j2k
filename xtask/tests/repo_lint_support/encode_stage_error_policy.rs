// SPDX-License-Identifier: MIT OR Apache-2.0

//! Typed, source-preserving error ratchets for the shared encode-stage SPI.

use std::fs;

use super::{assert_pattern_checks, read_source_files, repo_root, PatternCheck};

fn read(relative: &str) -> String {
    fs::read_to_string(repo_root().join(relative))
        .unwrap_or_else(|error| panic!("read {relative}: {error}"))
}

#[test]
fn every_fallible_encode_stage_hook_uses_the_shared_typed_result() {
    let types = read("crates/j2k-types/src/lib.rs");
    let trait_source = types
        .split_once("pub trait J2kEncodeStageAccelerator")
        .and_then(|(_, tail)| {
            tail.split_once("impl J2kEncodeStageAccelerator for CpuOnlyJ2kEncodeStageAccelerator")
        })
        .map_or_else(
            || panic!("encode-stage trait boundaries moved"),
            |(body, _)| body,
        );

    assert_eq!(
        trait_source.matches("-> J2kEncodeStageResult").count(),
        14,
        "all 14 fallible encode-stage hooks must use J2kEncodeStageResult"
    );
    assert!(
        !trait_source.contains("&'static str"),
        "the public encode-stage SPI must not expose static-string errors"
    );
}

#[test]
fn stage_error_taxonomy_remains_no_std_typed_and_source_preserving() {
    let error = read("crates/j2k-types/src/stage_error.rs");
    assert_pattern_checks(&[PatternCheck::new("encode-stage error taxonomy", &error)
        .required(&[
            "#[non_exhaustive]\npub enum J2kEncodeStageError",
            "InvalidRequest {",
            "Unsupported {",
            "ArithmeticOverflow {",
            "MemoryCapExceeded {",
            "HostAllocationFailed {",
            "Backend {",
            "InternalInvariant {",
            "source: Box<dyn Error + Send + Sync + 'static>",
            "Self::Backend { source, .. } => Some(source.as_ref())",
            "backend_failure_retains_concrete_source",
        ])
        .forbidden(&[
            "impl From<&'static str> for J2kEncodeStageError",
            "impl From<&str> for J2kEncodeStageError",
            "impl From<String> for J2kEncodeStageError",
            "Backend(String)",
            "source.to_string()",
        ])]);
}

#[test]
fn native_and_facade_boundaries_retain_stage_operation_and_source() {
    let source = read_source_files(
        repo_root(),
        &[
            "crates/j2k-native/src/error.rs",
            "crates/j2k-native/src/j2c/encode/single_tile/accelerator.rs",
            "crates/j2k-native/src/j2c/encode/tier1_driver.rs",
            "crates/j2k/src/encode/native.rs",
            "crates/j2k/src/encode/resident/tests.rs",
            "crates/j2k/tests/encode_lossless.rs",
        ],
    );

    assert_pattern_checks(&[
        PatternCheck::new("encode-stage source propagation", &source)
            .required(&[
                "operation: &'static str,",
                "source: J2kEncodeStageError,",
                "Self::Accelerator { source, .. } => Some(source)",
                ".map_err(|source| crate::EncodeError::Accelerator {",
                "ResidentHtj2kEncodeError::Accelerator(source)",
                "accelerator_facade_preserves_native_stage_error",
                "resident_encode_decline_and_accelerator_error_are_explicit",
            ])
            .forbidden(&["Err(\"facade accelerator fixture\")"]),
    ]);
}
