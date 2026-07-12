// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::{assert_pattern_checks, PatternCheck};
use super::{assert_line_budget, read};

mod batch;

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "core and batch transcode ownership is enforced as one cohesive stage matrix"
)]
fn jpeg_to_htj2k_core_and_batch_stay_split_by_stage() {
    let core = read("crates/j2k-transcode/src/jpeg_to_htj2k.rs");
    let validation = read("crates/j2k-transcode/src/jpeg_to_htj2k/validation.rs");
    let scratch = read("crates/j2k-transcode/src/jpeg_to_htj2k/scratch.rs");
    let output = read("crates/j2k-transcode/src/jpeg_to_htj2k/output.rs");
    let component_plan = read("crates/j2k-transcode/src/jpeg_to_htj2k/component_plan.rs");
    let component_groups = read("crates/j2k-transcode/src/jpeg_to_htj2k/component_groups.rs");
    let float_reference = read("crates/j2k-transcode/src/jpeg_to_htj2k/float_reference.rs");
    let float_output = read("crates/j2k-transcode/src/jpeg_to_htj2k/float_output.rs");
    let integer_reference = read("crates/j2k-transcode/src/jpeg_to_htj2k/integer_reference.rs");
    let integer_storage = read("crates/j2k-transcode/src/jpeg_to_htj2k/integer_storage.rs");
    let single = read("crates/j2k-transcode/src/jpeg_to_htj2k/single.rs");
    let single_tile_encode = read("crates/j2k-transcode/src/jpeg_to_htj2k/single_tile_encode.rs");
    for (path, source, max_lines) in [
        ("jpeg_to_htj2k.rs", core.as_str(), 400),
        ("jpeg_to_htj2k/validation.rs", validation.as_str(), 175),
        ("jpeg_to_htj2k/scratch.rs", scratch.as_str(), 75),
        ("jpeg_to_htj2k/output.rs", output.as_str(), 75),
        (
            "jpeg_to_htj2k/component_plan.rs",
            component_plan.as_str(),
            500,
        ),
        (
            "jpeg_to_htj2k/component_groups.rs",
            component_groups.as_str(),
            75,
        ),
        (
            "jpeg_to_htj2k/float_reference.rs",
            float_reference.as_str(),
            450,
        ),
        ("jpeg_to_htj2k/float_output.rs", float_output.as_str(), 250),
        (
            "jpeg_to_htj2k/integer_reference.rs",
            integer_reference.as_str(),
            475,
        ),
        (
            "jpeg_to_htj2k/integer_storage.rs",
            integer_storage.as_str(),
            100,
        ),
        ("jpeg_to_htj2k/single.rs", single.as_str(), 325),
        (
            "jpeg_to_htj2k/single_tile_encode.rs",
            single_tile_encode.as_str(),
            100,
        ),
    ] {
        assert_line_budget(path, source, max_lines);
    }

    assert_pattern_checks(&[
        PatternCheck::new("JPEG-to-HTJ2K facade", &core)
            .required(&[
                "mod validation;",
                "mod component_plan;",
                "mod component_groups;",
                "mod float_reference;",
                "mod float_output;",
                "mod integer_reference;",
                "mod integer_storage;",
                "mod single;",
                "mod single_tile_encode;",
            ])
            .forbidden(&[
                "fn transcode_component_batch(",
                "fn same_geometry_component_groups(",
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
        ])
        .forbidden(&["fn same_geometry_component_groups("]),
        PatternCheck::new(
            "JPEG-to-HTJ2K component grouping ownership",
            &component_groups,
        )
        .required(&[
            "fn same_geometry_component_groups(",
            "fn same_component_geometry(",
            "try_vec_filled(",
            "try_vec_with_capacity(",
            "try_vec_reserve_len(",
        ]),
        PatternCheck::new("JPEG-to-HTJ2K float reference ownership", &float_reference).required(&[
            "struct ComponentWavelet97",
            "fn float_direct_97_wavelet_from_component(",
            "fn float97_reference_coefficients(",
        ]),
        PatternCheck::new("JPEG-to-HTJ2K float output ownership", &float_output).required(&[
            "fn j2k_dwt_from_wavelet(",
            "fn rounded_wavelet97_i32(",
            "fn wavelet_coefficient_count(",
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
        PatternCheck::new("JPEG-to-HTJ2K integer storage ownership", &integer_storage).required(&[
            "fn idct_component_samples_i32(",
            "fn checked_product(",
            "fn validate_band_len(",
        ]),
        PatternCheck::new("JPEG-to-HTJ2K single orchestration ownership", &single)
            .required(&[
                "fn prepare_single_transcode(",
                "fn jpeg_to_htj2k_with_scratch<",
                "fn finish_single_transcode(",
                "struct PreparedSingleTranscode",
                "struct CompletedSingleTranscode",
            ])
            .forbidden(&["include!(", "use super::*;"]),
        PatternCheck::new(
            "JPEG-to-HTJ2K single-tile encode ownership",
            &single_tile_encode,
        )
        .required(&[
            "fn encode_component_batch",
            "record_encode_dispatch_delta(",
            "timings.htj2k_encode_us = encode_us;",
        ]),
    ]);
    batch::assert_batch_structure();
}
