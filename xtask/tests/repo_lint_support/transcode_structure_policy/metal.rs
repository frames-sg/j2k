// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::{assert_pattern_checks, PatternCheck};
use super::{assert_line_budget, read};

#[test]
fn metal_transcode_facades_and_extracted_children_stay_focused() {
    let focused_modules = [
        ("lib.rs", 150),
        ("route.rs", 275),
        ("accelerator.rs", 350),
        ("accelerator/dispatch.rs", 250),
        ("metal/reversible/batch.rs", 275),
        ("metal/irreversible/staging.rs", 150),
        ("metal/geometry/allocation.rs", 125),
        ("metal/buffers/storage.rs", 100),
        ("metal/buffers/basis.rs", 50),
        ("metal/codeblock_output.rs", 375),
        ("weights.rs", 50),
        ("weights/budget.rs", 425),
        ("weights/dense.rs", 425),
        ("weights/error.rs", 425),
        ("weights/shared.rs", 425),
        ("weights/sparse.rs", 425),
        ("weights/symbolic.rs", 425),
        ("weights/tests.rs", 425),
        ("weights/transform.rs", 425),
    ];
    for (relative, max_lines) in focused_modules {
        let path = format!("crates/j2k-transcode-metal/src/{relative}");
        let source = read(&path);
        assert_line_budget(&path, &source, max_lines);
    }

    let library = read("crates/j2k-transcode-metal/src/lib.rs");
    let accelerator = read("crates/j2k-transcode-metal/src/accelerator.rs");
    let reversible = read("crates/j2k-transcode-metal/src/metal/reversible.rs");
    let irreversible = read("crates/j2k-transcode-metal/src/metal/irreversible.rs");
    let geometry = read("crates/j2k-transcode-metal/src/metal/geometry.rs");
    let buffers = read("crates/j2k-transcode-metal/src/metal/buffers.rs");
    let weights = read("crates/j2k-transcode-metal/src/weights.rs");
    assert_pattern_checks(&[
        PatternCheck::new("Metal transcode crate facade", &library)
            .required(&["mod accelerator;", "mod error;", "mod route;"])
            .forbidden(&["impl DctToWaveletStageAccelerator for"]),
        PatternCheck::new("Metal accelerator facade", &accelerator)
            .required(&["mod dispatch;"])
            .forbidden(&["impl DctToWaveletStageAccelerator for"]),
        PatternCheck::new("Metal reversible facade", &reversible).required(&["mod batch;"]),
        PatternCheck::new("Metal irreversible facade", &irreversible).required(&["mod staging;"]),
        PatternCheck::new("Metal geometry facade", &geometry).required(&["mod allocation;"]),
        PatternCheck::new("Metal buffer facade", &buffers)
            .required(&["mod basis;", "mod storage;"]),
        PatternCheck::new("Metal weight facade", &weights)
            .required(&[
                "mod budget;",
                "mod dense;",
                "mod error;",
                "mod shared;",
                "mod sparse;",
                "mod symbolic;",
                "mod transform;",
            ])
            .forbidden(&[
                "fn linearized_53_from_sample_slice(",
                "fn sparse_rows_from_dense(",
            ]),
    ]);
}

#[test]
fn metal_transcode_extracted_children_own_their_focused_stages() {
    let route = read("crates/j2k-transcode-metal/src/route.rs");
    let dispatch = read("crates/j2k-transcode-metal/src/accelerator/dispatch.rs");
    let reversible_batch = read("crates/j2k-transcode-metal/src/metal/reversible/batch.rs");
    let irreversible_staging = read("crates/j2k-transcode-metal/src/metal/irreversible/staging.rs");
    let geometry_allocation = read("crates/j2k-transcode-metal/src/metal/geometry/allocation.rs");
    let buffer_storage = read("crates/j2k-transcode-metal/src/metal/buffers/storage.rs");
    let buffer_basis = read("crates/j2k-transcode-metal/src/metal/buffers/basis.rs");
    let codeblock_output = read("crates/j2k-transcode-metal/src/metal/codeblock_output.rs");
    assert_pattern_checks(&[
        PatternCheck::new("Metal route ownership", &route).required(&[
            "pub fn jpeg_to_htj2k_with_metal_route(",
            "pub fn jpeg_to_htj2k_batch_with_metal_route(",
            "fn fallback_reason(",
        ]),
        PatternCheck::new("Metal accelerator dispatch ownership", &dispatch).required(&[
            "impl DctToWaveletStageAccelerator for MetalDctToWaveletStageAccelerator",
            "fn dct_grid_to_dwt53(",
            "fn dct_grid_to_htj2k97_codeblock_batch(",
        ]),
        PatternCheck::new("Metal reversible batch ownership", &reversible_batch).required(&[
            "fn dispatch_with_runtime(",
            "fn reversible_batch_shapes(",
            "fn read_reversible_batch_outputs(",
        ]),
        PatternCheck::new(
            "Metal irreversible staging ownership",
            &irreversible_staging,
        )
        .required(&[
            "fn dwt97_staged_batch_shape(",
            "fn dwt97_codeblock_batch_shape(",
            "fn dwt97_staged_row_buffers(",
        ]),
        PatternCheck::new("Metal geometry allocation ownership", &geometry_allocation).required(&[
            "fn validate_float_projection_allocations(",
            "fn validate_codeblock_projection_allocations(",
        ]),
        PatternCheck::new("Metal buffer storage ownership", &buffer_storage).required(&[
            "fn buffer_with_slice<",
            "fn dwt97_blocks_buffer(",
            "fn output_i32_buffer(",
        ]),
        PatternCheck::new("Metal basis ownership", &buffer_basis)
            .required(&["fn idct8_basis_table(", "fn idct8_basis("]),
        PatternCheck::new("Metal code-block readback ownership", &codeblock_output).required(&[
            "fn validate_codeblock_output_host_workspace(",
            "fn read_prequantized_97_codeblock_outputs(",
            "fn try_vec_from_array<",
        ]),
    ]);
}
