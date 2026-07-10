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

#[test]
fn transcode_accelerator_contracts_use_a_real_module() {
    let root = read("crates/j2k-transcode/src/lib.rs");
    assert_line_budget("j2k-transcode/src/lib.rs", &root, 150);
    assert_pattern_checks(&[PatternCheck::new("transcode crate root", &root)
        .required(&[
            "#[doc(hidden)]\npub mod accelerator;",
            "pub use self::accelerator::{",
        ])
        .forbidden(&[
            "include!(\"accelerator.rs\")",
            "#[path = \"accelerator.rs\"]",
        ])]);
}

#[test]
fn jpeg_to_htj2k_core_and_batch_stay_split_by_stage() {
    let core = read("crates/j2k-transcode/src/jpeg_to_htj2k.rs");
    let validation = read("crates/j2k-transcode/src/jpeg_to_htj2k/validation.rs");
    let component_plan = read("crates/j2k-transcode/src/jpeg_to_htj2k/component_plan.rs");
    let float_reference = read("crates/j2k-transcode/src/jpeg_to_htj2k/float_reference.rs");
    let integer_reference = read("crates/j2k-transcode/src/jpeg_to_htj2k/integer_reference.rs");
    let single_tile_encode = read("crates/j2k-transcode/src/jpeg_to_htj2k/single_tile_encode.rs");
    let batch = read("crates/j2k-transcode/src/jpeg_to_htj2k/batch.rs");
    let batch_prepare = read("crates/j2k-transcode/src/jpeg_to_htj2k/batch/prepare.rs");
    let batch_transform = read("crates/j2k-transcode/src/jpeg_to_htj2k/batch/transform.rs");
    let batch_accelerated_storage =
        read("crates/j2k-transcode/src/jpeg_to_htj2k/batch/accelerated_storage.rs");
    let batch_storage = read("crates/j2k-transcode/src/jpeg_to_htj2k/batch/storage.rs");
    let batch_encode = read("crates/j2k-transcode/src/jpeg_to_htj2k/batch/encode.rs");
    for (path, source, max_lines) in [
        ("jpeg_to_htj2k.rs", core.as_str(), 400),
        ("jpeg_to_htj2k/validation.rs", validation.as_str(), 175),
        (
            "jpeg_to_htj2k/component_plan.rs",
            component_plan.as_str(),
            500,
        ),
        (
            "jpeg_to_htj2k/float_reference.rs",
            float_reference.as_str(),
            450,
        ),
        (
            "jpeg_to_htj2k/integer_reference.rs",
            integer_reference.as_str(),
            475,
        ),
        (
            "jpeg_to_htj2k/single_tile_encode.rs",
            single_tile_encode.as_str(),
            100,
        ),
        ("jpeg_to_htj2k/batch.rs", batch.as_str(), 350),
        (
            "jpeg_to_htj2k/batch/prepare.rs",
            batch_prepare.as_str(),
            325,
        ),
        (
            "jpeg_to_htj2k/batch/transform.rs",
            batch_transform.as_str(),
            425,
        ),
        (
            "jpeg_to_htj2k/batch/accelerated_storage.rs",
            batch_accelerated_storage.as_str(),
            475,
        ),
        (
            "jpeg_to_htj2k/batch/storage.rs",
            batch_storage.as_str(),
            475,
        ),
        ("jpeg_to_htj2k/batch/encode.rs", batch_encode.as_str(), 600),
    ] {
        assert_line_budget(path, source, max_lines);
    }

    assert_pattern_checks(&[
        PatternCheck::new("JPEG-to-HTJ2K facade", &core)
            .required(&[
                "mod validation;",
                "mod component_plan;",
                "mod float_reference;",
                "mod integer_reference;",
                "mod single_tile_encode;",
            ])
            .forbidden(&[
                "fn transcode_component_batch(",
                "fn integer_direct_wavelet_from_component(",
                "fn float_direct_wavelet_from_component(",
            ]),
        PatternCheck::new("JPEG-to-HTJ2K validation ownership", &validation).required(&[
            "fn validate_transcode_options(",
            "fn validate_component_block_grid(",
            "fn decomposition_levels_for_components(",
        ]),
        PatternCheck::new(
            "JPEG-to-HTJ2K component planning ownership",
            &component_plan,
        )
        .required(&[
            "fn transcode_component_batch(",
            "struct ComponentTranscodePlan",
            "fn component_to_precomputed_htj2k(",
        ]),
        PatternCheck::new("JPEG-to-HTJ2K float reference ownership", &float_reference).required(&[
            "struct ComponentWavelet97",
            "fn float_direct_97_wavelet_from_component(",
            "fn float97_reference_coefficients(",
        ]),
        PatternCheck::new(
            "JPEG-to-HTJ2K integer reference ownership",
            &integer_reference,
        )
        .required(&[
            "struct IntegerWavelet",
            "fn integer_direct_wavelet_from_component(",
            "fn integer_reference_coefficients(",
        ]),
        PatternCheck::new(
            "JPEG-to-HTJ2K single-tile encode ownership",
            &single_tile_encode,
        )
        .required(&[
            "fn encode_component_batch",
            "record_encode_dispatch_delta(",
            "timings.htj2k_encode_us = encode_us;",
        ]),
        PatternCheck::new("JPEG-to-HTJ2K batch facade", &batch)
            .required(&[
                "mod prepare;",
                "mod transform;",
                "mod accelerated_storage;",
                "mod storage;",
                "mod encode;",
            ])
            .forbidden(&[
                "struct IntegerBatchTile",
                "fn transform_float97_batch_tiles(",
                "fn store_integer_batch_wavelet(",
                "fn encode_float97_batch_tile(",
            ]),
        PatternCheck::new("JPEG-to-HTJ2K batch preparation ownership", &batch_prepare).required(&[
            "struct IntegerBatchTile",
            "fn prepare_float97_batch_tile(",
            "fn batch_component_groups(",
        ]),
        PatternCheck::new("JPEG-to-HTJ2K batch transform ownership", &batch_transform).required(&[
            "fn transform_integer_batch_tiles",
            "fn float97_wavelets_for_batch_group",
            "fn record_cpu_fallback",
        ]),
        PatternCheck::new("accelerated batch storage", &batch_accelerated_storage).required(&[
            "fn store_compact_preencoded_component(",
            "fn try_store_grouped_i16_preencoded_float97_batches",
            "fn try_store_prequantized_float97_batch_group",
        ]),
        PatternCheck::new("wavelet batch storage", &batch_storage).required(&[
            "fn store_integer_batch_wavelet(",
            "fn store_float97_batch_wavelet(",
        ]),
        PatternCheck::new("JPEG-to-HTJ2K batch encode ownership", &batch_encode).required(&[
            "fn record_encode_dispatch_delta(",
            "fn encode_float97_precomputed_tiles_batch",
            "fn encode_float97_batch_tile",
        ]),
    ]);
}

#[test]
fn cuda_and_metal_transcode_backends_stay_split_by_residency_stage() {
    let cuda = read("crates/j2k-transcode-cuda/src/cuda.rs");
    let cuda_transform = read("crates/j2k-transcode-cuda/src/cuda/transform.rs");
    let cuda_resident_dispatch = read("crates/j2k-transcode-cuda/src/cuda/resident_dispatch.rs");
    let cuda_resident_encode = read("crates/j2k-transcode-cuda/src/cuda/resident_encode.rs");
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
            "j2k-transcode-cuda/src/cuda/resident_dispatch.rs",
            cuda_resident_dispatch.as_str(),
            425,
        ),
        (
            "j2k-transcode-cuda/src/cuda/resident_encode.rs",
            cuda_resident_encode.as_str(),
            850,
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
            "fn dispatch_reversible_dwt53_batch(",
            "fn dispatch_dwt97_batch(",
            "fn dispatch_htj2k97_preencoded_batch(",
        ]),
        PatternCheck::new("CUDA resident dispatch ownership", &cuda_resident_dispatch).required(&[
            "fn dispatch_htj2k97_preencoded_i16_batch_with_sink",
            "fn dispatch_htj2k97_compact_preencoded_i16_batch_groups",
            "fn device_bands_to_preencoded_components",
        ]),
        PatternCheck::new("CUDA resident encode ownership", &cuda_resident_encode).required(&[
            "fn encode_resident_subbands(",
            "fn assemble_compact_preencoded_components",
        ]),
        PatternCheck::new("Metal transcode facade", &metal)
            .required(&[
                "mod runtime;",
                "mod reversible;",
                "mod irreversible;",
                "mod projection;",
                "mod resident;",
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
