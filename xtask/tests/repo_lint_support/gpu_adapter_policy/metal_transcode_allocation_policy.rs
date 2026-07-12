// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;

use super::super::rust_function_policy::FunctionCalls;
use super::super::{assert_pattern_checks, repo_root, PatternCheck};

mod float_projection;

fn read(relative: &str) -> String {
    fs::read_to_string(repo_root().join(relative))
        .unwrap_or_else(|error| panic!("read {relative}: {error}"))
}

fn calls(label: &str, source: &str, function: &str) -> FunctionCalls {
    FunctionCalls::parse(label, source, function)
}

#[test]
fn metal_codeblock_host_workspace_is_preflighted_at_entry_and_readback() {
    let irreversible = read("crates/j2k-transcode-metal/src/metal/irreversible.rs");
    let allocation = read("crates/j2k-transcode-metal/src/metal/geometry/allocation.rs");
    let output = read("crates/j2k-transcode-metal/src/metal/codeblock_output.rs");

    let entry = calls(
        "Metal HTJ2K code-block batch entry",
        &irreversible,
        "dispatch_dct_grid_to_htj2k97_codeblock_batch",
    );
    entry.assert_ordered(
        "Metal HTJ2K entry preflight before runtime work",
        &[
            "validate_dwt97_codeblock_batch_geometry",
            "validate_htj2k97_codeblock_options",
            "checked_mul",
            "validate_codeblock_projection_allocations",
            "with_runtime",
        ],
    );
    entry.assert_propagated(
        "Metal HTJ2K entry preflight propagation",
        &[
            "validate_dwt97_codeblock_batch_geometry",
            "validate_htj2k97_codeblock_options",
            "validate_codeblock_projection_allocations",
        ],
    );

    calls(
        "Metal HTJ2K allocation preflight",
        &allocation,
        "validate_codeblock_projection_allocations",
    )
    .assert_ordered(
        "Metal shared host preflight before device sizing",
        &[
            "validate_codeblock_output_host_workspace",
            "checked_device_element_count",
            "checked_device_workspace_bytes",
        ],
    );

    let readback = calls(
        "Metal HTJ2K code-block readback",
        &output,
        "read_prequantized_97_codeblock_outputs",
    );
    readback.assert_ordered(
        "Metal readback repeats host preflight before metadata reservation",
        &[
            "validate_codeblock_output_host_workspace",
            "try_transcode_vec_with_capacity",
            "read_component",
        ],
    );
    readback.assert_propagated(
        "Metal readback allocation propagation",
        &[
            "validate_codeblock_output_host_workspace",
            "try_transcode_vec_with_capacity",
            "read_component",
        ],
    );
}

#[test]
fn metal_codeblock_host_preflight_accounts_for_all_live_output_metadata() {
    let output = read("crates/j2k-transcode-metal/src/metal/codeblock_output.rs");
    let preflight = calls(
        "Metal HTJ2K host workspace",
        &output,
        "validate_codeblock_output_host_workspace",
    );
    preflight.assert_contains(
        "Metal HTJ2K aggregate host metadata accounting",
        &[
            "checked_host_element_count",
            "codeblocks_per_item",
            "checked_host_workspace_bytes",
        ],
    );
    preflight.assert_propagated(
        "Metal HTJ2K aggregate host metadata propagation",
        &[
            "checked_host_element_count",
            "codeblocks_per_item",
            "checked_host_workspace_bytes",
        ],
    );
    assert_pattern_checks(&[
        PatternCheck::new("Metal code-block host preflight", &output).required(&[
            "prequantized HTJ2K coefficients",
            "prequantized HTJ2K item readback",
            "prequantized HTJ2K code-block metadata",
            "prequantized HTJ2K component metadata",
            "prequantized HTJ2K resolution metadata",
            "prequantized HTJ2K subband metadata",
            "prequantized HTJ2K host workspace",
            "aggregate_output_metadata_rejects_combined_cap_excess",
        ]),
    ]);
}

#[test]
fn metal_codeblock_fixed_array_outputs_are_constructed_fallibly() {
    let output = read("crates/j2k-transcode-metal/src/metal/codeblock_output.rs");
    let component = calls(
        "Metal prequantized component construction",
        &output,
        "read_component",
    );
    component.assert_contains(
        "Metal fallible resolution array conversion",
        &["resolution_from_subbands", "try_vec_from_array"],
    );
    component.assert_propagated(
        "Metal fallible resolution array propagation",
        &["resolution_from_subbands", "try_vec_from_array"],
    );

    let resolution = calls(
        "Metal prequantized resolution construction",
        &output,
        "resolution_from_subbands",
    );
    resolution.assert_propagated(
        "Metal fallible subband array propagation",
        &["try_vec_from_array"],
    );
    let array = calls(
        "Metal fixed-array vector conversion",
        &output,
        "try_vec_from_array",
    );
    array.assert_ordered(
        "Metal fixed-array reserve before infallible move",
        &["try_transcode_vec_with_capacity", "extend"],
    );
    array.assert_propagated(
        "Metal fixed-array reserve propagation",
        &["try_transcode_vec_with_capacity"],
    );
    for function in [component, resolution, array] {
        function.assert_absent(
            "Metal fixed-array output construction",
            &[
                "Vec::from",
                "Vec::with_capacity",
                "vec",
                "collect",
                "to_vec",
            ],
        );
    }
}

#[test]
fn metal_dwt53_symbolic_rows_use_shared_codec_math() {
    let facade = read("crates/j2k-transcode-metal/src/weights.rs");
    let symbolic = read("crates/j2k-transcode-metal/src/weights/symbolic.rs");
    let shared_row = calls(
        "Metal shared 5/3 symbolic row",
        &symbolic,
        "write_dwt53_row",
    );
    shared_row.assert_ordered(
        "Metal shared codec-math 5/3 construction",
        &["linearized_dwt53_row", "taps", "push_tap"],
    );
    shared_row.assert_propagated("Metal shared codec-math tap propagation", &["push_tap"]);
    shared_row.assert_absent(
        "Metal shared codec-math 5/3 construction",
        &["vec", "Vec::with_capacity", "collect", "to_vec"],
    );
    assert_pattern_checks(&[
        PatternCheck::new("Metal weight facade", &facade).forbidden(&[
            "fn linearized_53_from_sample_slice(",
            "let mut basis = vec![0.0; sample_len]",
            "fn sparse_rows_from_dense(",
        ]),
        PatternCheck::new("Metal symbolic 5/3 implementation", &symbolic).required(&[
            "use j2k_codec_math::dwt::{linearized_dwt53_row, Dwt53Band};",
            "return write_dwt53_row(output, sample_len, output_index, band);",
        ]),
    ]);
}

#[test]
fn metal_transcode_allocation_policy_stays_focused() {
    assert!(
        include_str!("metal_transcode_allocation_policy.rs")
            .lines()
            .count()
            < 250,
        "Metal transcode allocation policy must stay focused"
    );
}
