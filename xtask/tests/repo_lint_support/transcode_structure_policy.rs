// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;

use super::{assert_pattern_checks, repo_root, PatternCheck};

fn read(relative_path: &str) -> String {
    fs::read_to_string(repo_root().join(relative_path))
        .unwrap_or_else(|error| panic!("read {relative_path}: {error}"))
}

fn assert_line_budget(relative_path: &str, source: &str, max_lines: usize) {
    let line_count = source.lines().count();
    assert!(
        line_count < max_lines,
        "{relative_path} has {line_count} lines; expected fewer than {max_lines}"
    );
}

mod cpu;
mod metal;

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "CUDA and Metal residency-stage ownership is one cross-backend structural policy"
)]
fn cuda_and_metal_transcode_backends_stay_split_by_residency_stage() {
    let cuda = read("crates/j2k-transcode-cuda/src/cuda.rs");
    let cuda_transform = read("crates/j2k-transcode-cuda/src/cuda/transform.rs");
    let cuda_transform_components =
        read("crates/j2k-transcode-cuda/src/cuda/transform/components.rs");
    let cuda_transform_staging = read("crates/j2k-transcode-cuda/src/cuda/transform/staging.rs");
    let cuda_resident_dispatch = read("crates/j2k-transcode-cuda/src/cuda/resident_dispatch.rs");
    let cuda_resident_dispatch_grouped =
        read("crates/j2k-transcode-cuda/src/cuda/resident_dispatch/grouped.rs");
    let cuda_resident_encode = read("crates/j2k-transcode-cuda/src/cuda/resident_encode.rs");
    let cuda_resident_encode_orchestration =
        read("crates/j2k-transcode-cuda/src/cuda/resident_encode/orchestration.rs");
    let cuda_resident_encode_output =
        read("crates/j2k-transcode-cuda/src/cuda/resident_encode/output.rs");
    let cuda_resident_encode_planning =
        read("crates/j2k-transcode-cuda/src/cuda/resident_encode/planning.rs");
    let metal = read("crates/j2k-transcode-metal/src/metal.rs");
    let metal_runtime = read("crates/j2k-transcode-metal/src/metal/runtime.rs");
    let metal_reversible = read("crates/j2k-transcode-metal/src/metal/reversible.rs");
    let metal_irreversible = read("crates/j2k-transcode-metal/src/metal/irreversible.rs");
    let metal_projection = read("crates/j2k-transcode-metal/src/metal/projection.rs");
    let metal_resident = read("crates/j2k-transcode-metal/src/metal/resident.rs");
    let metal_geometry = read("crates/j2k-transcode-metal/src/metal/geometry.rs");
    let metal_buffers = read("crates/j2k-transcode-metal/src/metal/buffers.rs");

    for (path, source, max_lines) in [
        ("j2k-transcode-cuda/src/cuda.rs", cuda.as_str(), 250),
        (
            "j2k-transcode-cuda/src/cuda/transform.rs",
            cuda_transform.as_str(),
            625,
        ),
        (
            "j2k-transcode-cuda/src/cuda/transform/components.rs",
            cuda_transform_components.as_str(),
            425,
        ),
        (
            "j2k-transcode-cuda/src/cuda/transform/staging.rs",
            cuda_transform_staging.as_str(),
            425,
        ),
        (
            "j2k-transcode-cuda/src/cuda/resident_dispatch.rs",
            cuda_resident_dispatch.as_str(),
            425,
        ),
        (
            "j2k-transcode-cuda/src/cuda/resident_dispatch/grouped.rs",
            cuda_resident_dispatch_grouped.as_str(),
            425,
        ),
        (
            "j2k-transcode-cuda/src/cuda/resident_encode.rs",
            cuda_resident_encode.as_str(),
            125,
        ),
        (
            "j2k-transcode-cuda/src/cuda/resident_encode/orchestration.rs",
            cuda_resident_encode_orchestration.as_str(),
            425,
        ),
        (
            "j2k-transcode-cuda/src/cuda/resident_encode/output.rs",
            cuda_resident_encode_output.as_str(),
            425,
        ),
        (
            "j2k-transcode-cuda/src/cuda/resident_encode/planning.rs",
            cuda_resident_encode_planning.as_str(),
            425,
        ),
        ("j2k-transcode-metal/src/metal.rs", metal.as_str(), 125),
        (
            "j2k-transcode-metal/src/metal/runtime.rs",
            metal_runtime.as_str(),
            275,
        ),
        (
            "j2k-transcode-metal/src/metal/reversible.rs",
            metal_reversible.as_str(),
            300,
        ),
        (
            "j2k-transcode-metal/src/metal/irreversible.rs",
            metal_irreversible.as_str(),
            700,
        ),
        (
            "j2k-transcode-metal/src/metal/projection.rs",
            metal_projection.as_str(),
            300,
        ),
        (
            "j2k-transcode-metal/src/metal/resident.rs",
            metal_resident.as_str(),
            825,
        ),
        (
            "j2k-transcode-metal/src/metal/geometry.rs",
            metal_geometry.as_str(),
            425,
        ),
        (
            "j2k-transcode-metal/src/metal/buffers.rs",
            metal_buffers.as_str(),
            225,
        ),
    ] {
        assert_line_budget(path, source, max_lines);
    }

    assert_pattern_checks(&[
        PatternCheck::new("CUDA transcode facade", &cuda)
            .required(&[
                "mod transform;",
                "mod resident_dispatch;",
                "mod resident_encode;",
            ])
            .forbidden(&["fn run_dwt97(", "fn encode_resident_subbands("]),
        PatternCheck::new("CUDA transform dispatch ownership", &cuda_transform).required(&[
            "mod components;",
            "mod staging;",
            "fn dispatch_reversible_dwt53_batch(",
            "fn dispatch_dwt97_batch(",
            "fn dispatch_htj2k97_preencoded_batch(",
        ]),
        PatternCheck::new(
            "CUDA transform component ownership",
            &cuda_transform_components,
        )
        .required(&[
            "fn preflight_component_allocation_budget(",
            "fn codeblock_bands_to_components(",
            "fn component_from_subbands(",
            "fn account_codeblock_bands(",
        ]),
        PatternCheck::new("CUDA transform staging ownership", &cuda_transform_staging).required(&[
            "fn validate_staging_and_readback_workspace(",
            "fn preflight_dwt97_conversion_budget(",
            "fn flatten_f64_blocks_to_f32(",
        ]),
        PatternCheck::new("CUDA resident dispatch ownership", &cuda_resident_dispatch).required(&[
            "mod grouped;",
            "fn dispatch_htj2k97_preencoded_i16_batch_with_sink",
            "fn device_bands_to_preencoded_components",
        ]),
        PatternCheck::new(
            "CUDA grouped resident dispatch ownership",
            &cuda_resident_dispatch_grouped,
        )
        .required(&[
            "fn live_staging_budget<",
            "fn dispatch_with_sink<",
            "fn dispatch_htj2k97_compact_preencoded_i16_batch_groups",
        ]),
        PatternCheck::new("CUDA resident encode facade", &cuda_resident_encode)
            .required(&["mod orchestration;", "mod output;", "mod planning;"])
            .forbidden(&[
                "fn encode_resident_subbands(",
                "fn split_resident_subband_blocks(",
            ]),
        PatternCheck::new(
            "CUDA resident encode orchestration ownership",
            &cuda_resident_encode_orchestration,
        )
        .required(&[
            "fn encode_resident_subbands(",
            "fn encode_resident_compact_subbands(",
            "fn device_band_groups_to_preencoded_components",
        ]),
        PatternCheck::new(
            "CUDA resident output ownership",
            &cuda_resident_encode_output,
        )
        .required(&[
            "fn assemble_compact_preencoded_components",
            "fn split_resident_subband_blocks(",
            "fn split_resident_compact_subband_blocks(",
        ]),
        PatternCheck::new(
            "CUDA resident planning ownership",
            &cuda_resident_encode_planning,
        )
        .required(&[
            "type ResidentMetadataBudget = HostPhaseBudget",
            "fn reserve_component_assembly_budget",
            "fn resident_group_targets",
            "fn resident_subband_encode_plan",
        ]),
        PatternCheck::new("Metal transcode facade", &metal)
            .required(&[
                "mod runtime;",
                "mod reversible;",
                "mod irreversible;",
                "mod projection;",
                "mod resident;",
                "mod codeblock_output;",
                "mod geometry;",
                "mod buffers;",
            ])
            .forbidden(&[
                "fn shader_source(",
                "fn dispatch_dct_grid_to_reversible_dwt53(",
                "fn dispatch_dct_grid_to_dwt97(",
            ]),
        PatternCheck::new("Metal runtime ownership", &metal_runtime).required(&[
            "fn shader_source(",
            "struct MetalRuntime",
            "pub struct MetalTranscodeSession",
        ]),
        PatternCheck::new("Metal reversible ownership", &metal_reversible).required(&[
            "fn dispatch_dct_grid_to_reversible_dwt53(",
            "fn dispatch_reversible_dwt53_batch_with_runtime(",
        ]),
        PatternCheck::new("Metal irreversible ownership", &metal_irreversible).required(&[
            "fn dispatch_dct_grid_to_dwt97(",
            "fn encode_dwt97_quantize_codeblocks(",
            "fn dispatch_dct_grid_to_dwt97_batch_staged_with_runtime(",
        ]),
        PatternCheck::new("Metal projection ownership", &metal_projection).required(&[
            "fn dispatch_projection_threads(",
            "struct ProjectionBatchJob",
            "fn dispatch_projected_bands_batch_with_runtime(",
        ]),
        PatternCheck::new("Metal resident handoff ownership", &metal_resident).required(&[
            "fn validate_resident_dct_handoffs(",
            "fn validate_resident_dwt_handoffs(",
            "fn projection_batch_output_buffers(",
        ]),
        PatternCheck::new("Metal geometry ownership", &metal_geometry).required(&[
            "struct BandGeometry",
            "fn validate_reversible_batch_geometry(",
            "fn metal_sparse_rows(",
        ]),
        PatternCheck::new("Metal buffer ownership", &metal_buffers).required(&[
            "fn dwt97_batch_blocks_buffer(",
            "fn read_f32_buffer(",
            "fn idct8_basis_table(",
        ]),
    ]);
}
