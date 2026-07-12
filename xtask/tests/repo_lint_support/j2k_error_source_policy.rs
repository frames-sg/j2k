// SPDX-License-Identifier: MIT OR Apache-2.0

//! Facade error crossings retain typed sources and explicit classifications.

use std::fs;

use super::{assert_pattern_checks, repo_root, PatternCheck};

fn read(relative: &str) -> String {
    fs::read_to_string(repo_root().join(relative))
        .unwrap_or_else(|error| panic!("read {relative}: {error}"))
}

#[test]
fn resident_encode_and_recode_validation_do_not_use_generic_backend_strings() {
    let error = read("crates/j2k/src/error.rs");
    let native_source = read("crates/j2k/src/error/native_source.rs");
    let resident = read("crates/j2k/src/encode/native.rs");
    let recode = read("crates/j2k/src/recode/validation.rs");

    assert_pattern_checks(&[
        PatternCheck::new("facade typed resident fallback", &error)
            .required(&[
                "NativeResidentEncode {",
                "source: NativeBackendError",
                "future_resident_encode_fallback_retains_its_typed_source",
                "assert_native_source_chain",
            ])
            .forbidden(&["pub(crate) fn backend(", "pub(crate) fn native(message:"]),
        PatternCheck::new("facade-owned opaque native source", &native_source)
            .required(&[
                "pub struct NativeBackendError",
                "source: NativeBackendErrorSource",
                "impl core::error::Error for NativeBackendError",
            ])
            .forbidden(&["pub source:", "message: String"]),
        PatternCheck::new("resident encode mapping", &resident)
            .required(&[
                "_ => J2kError::NativeResidentEncode {",
                "source: crate::NativeBackendError::resident_encode(err)",
            ])
            .forbidden(&[
                "J2kError::backend(",
                "resident lossless encode failed: {err}",
            ]),
        PatternCheck::new("recode validation classification", &recode)
            .required(&[
                "J2kError::Backend(BackendError::new(",
                "BackendErrorKind::Validation",
                "failed decoded-sample validation",
                "decoded_sample_mismatch_is_a_validation_backend_error",
            ])
            .forbidden(&["J2kError::backend("]),
    ]);
}
