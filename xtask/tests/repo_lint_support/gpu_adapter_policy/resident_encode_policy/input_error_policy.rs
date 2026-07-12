// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;

use crate::repo_lint_support::{assert_pattern_checks, repo_root, PatternCheck};

#[test]
fn resident_input_constructor_uses_a_typed_validation_contract() {
    let root = repo_root();
    let input_root = fs::read_to_string(root.join("crates/j2k-types/src/resident.rs"))
        .expect("read resident input contract");
    let input_error = fs::read_to_string(root.join("crates/j2k-types/src/resident/input_error.rs"))
        .expect("read resident input error contract");
    let input_tests = fs::read_to_string(root.join("crates/j2k-types/src/resident/tests.rs"))
        .expect("read resident input contract tests");
    let input = format!("{input_root}\n{input_error}\n{input_tests}");
    let cuda = fs::read_to_string(root.join("crates/j2k-cuda/src/encode.rs"))
        .expect("read CUDA resident encode boundary");
    let cuda_tests = fs::read_to_string(root.join("crates/j2k-cuda/src/encode/tests/routing.rs"))
        .expect("read CUDA resident input routing tests");
    let native = fs::read_to_string(
        root.join("crates/j2k-native/src/j2c/encode/single_tile/accelerator.rs"),
    )
    .expect("read native accelerator boundary");
    let native_pipeline =
        fs::read_to_string(root.join("crates/j2k-native/src/j2c/encode/retained_input.rs"))
            .expect("read native encode pipeline error boundary");

    assert_pattern_checks(&[
        PatternCheck::new("typed resident input validation error", &input)
            .required(&[
                "#[non_exhaustive]",
                "pub enum J2kResidentEncodeInputError",
                "EmptyGeometry",
                "ComponentCountOutOfRange",
                "PrecisionOutOfRange",
                "AddressSpaceOverflow",
                "pub const fn reason(&self) -> &'static str",
                "impl core::fmt::Display for J2kResidentEncodeInputError",
                "impl core::error::Error for J2kResidentEncodeInputError",
                ") -> Result<Self, J2kResidentEncodeInputError>",
                "usize::try_from(width)",
                "usize::try_from(height)",
                "resident_input_errors_keep_stable_reasons_and_typed_context",
            ])
            .forbidden(&[") -> Result<Self, &'static str>"]),
        PatternCheck::new("explicit CUDA resident input error mapping", &cuda).required(&[
            ".map_err(cuda_resident_input_error)?",
            "fn cuda_resident_input_error(error: J2kResidentEncodeInputError)",
            "J2kResidentEncodeInputError::EmptyGeometry { .. }",
            "J2kResidentEncodeInputError::AddressSpaceOverflow",
        ]),
        PatternCheck::new("typed CUDA resident input mapping coverage", &cuda_tests)
            .required(&["typed_resident_input_failures_map_to_stable_cuda_rejections"]),
        PatternCheck::new("typed native stage boundary", &native)
            .required(&[
                "NativeEncodePipelineResult<Option<(Vec<u8>, u128)>>",
                "NativeEncodePipelineError::internal_invariant(error.reason())",
            ])
            .forbidden(&[
                "Result<Option<(Vec<u8>, u128)>, &'static str>",
                ".map_err(|error| error.reason())?",
            ]),
        PatternCheck::new("typed native pipeline error ownership", &native_pipeline)
            .required(&[
                "pub(crate) type NativeEncodePipelineResult<T>",
                "pub(crate) enum NativeEncodePipelineError",
                "InvalidInput(&'static str)",
                "Unsupported(&'static str)",
                "ArithmeticOverflow(&'static str)",
                "InternalInvariant(&'static str)",
                "Typed(EncodeError)",
                "Self::InvalidInput(what) => EncodeError::InvalidInput { what }",
                "Self::InternalInvariant(what) => EncodeError::InternalInvariant { what }",
                "impl From<EncodeError> for NativeEncodePipelineError",
            ])
            .forbidden(&["Legacy(&'static str)"]),
    ]);
}
