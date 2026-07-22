// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::CudaDecoderSources;
use crate::repo_lint_support::{
    assert_pattern_checks, rust_function_policy::FunctionCalls, PatternCheck,
};

#[test]
fn grayscale_batch_pipeline_has_focused_phase_ownership() {
    let sources = CudaDecoderSources::read();
    assert_pattern_checks(&[
        PatternCheck::new(
            "CUDA grayscale batch preparation",
            &sources.grayscale_batch_preparation,
        )
        .required(&["fn prepare_grayscale_input", "fn append_grayscale_input("])
        .forbidden(&["clippy::too_many_lines"]),
        PatternCheck::new(
            "CUDA grayscale batch execution phases",
            &sources.grayscale_batch_execution,
        )
        .required(&[
            "fn upload_grayscale_decode_resources(",
            "fn build_grayscale_component_work(",
            "fn enqueue_grayscale_entropy(",
            "fn enqueue_grayscale_idwt(",
            "fn finish_grayscale_components_and_store(",
        ])
        .forbidden(&["clippy::too_many_lines"]),
        PatternCheck::new(
            "CUDA grayscale completion ownership",
            &sources.grayscale_batch_completion,
        )
        .required(&[
            "fn finish_submitted_grayscale_batch(",
            "fn finish_synchronous_grayscale_batch(",
        ]),
    ]);
    FunctionCalls::parse(
        "CUDA grayscale batch orchestration",
        &sources.grayscale_batch_execution,
        "decode_grayscale_cuda_batch_with_profile",
    )
    .assert_ordered(
        "CUDA grayscale batch phases",
        &[
            "prepare_grayscale_batch",
            "upload_grayscale_decode_resources",
            "build_grayscale_component_work",
            "enqueue_grayscale_entropy",
            "enqueue_grayscale_idwt",
            "finish_grayscale_components_and_store",
        ],
    );
}
