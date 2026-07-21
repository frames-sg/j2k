// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{assert_pattern_checks, PatternCheck};

mod planning;
mod sources;

use sources::Htj2kOutputSources;

#[test]
fn htj2k_shared_output_modules_stay_focused() {
    let sources = Htj2kOutputSources::read();
    for (relative, source, max_lines) in [
        ("htj2k_decode.rs", sources.decode_root.as_str(), 80usize),
        ("htj2k_decode/api.rs", sources.decode_api.as_str(), 200),
        (
            "htj2k_decode/completion.rs",
            sources.decode_completion.as_str(),
            525,
        ),
        (
            "htj2k_decode/completion/dequant.rs",
            sources.decode_dequant.as_str(),
            125,
        ),
        (
            "htj2k_decode/planning.rs",
            sources.decode_planning.as_str(),
            220,
        ),
        ("htj2k_decode/types.rs", sources.decode_types.as_str(), 390),
        (
            "htj2k_decode/context_validation.rs",
            sources.context_validation.as_str(),
            150,
        ),
        (
            "htj2k_decode/output_regions.rs",
            sources.output_regions.as_str(),
            200,
        ),
        (
            "htj2k_decode/output_regions/sweep.rs",
            sources.output_region_sweep.as_str(),
            175,
        ),
        (
            "htj2k_decode/output_regions/sweep/cross_stride.rs",
            sources.output_region_cross_stride.as_str(),
            125,
        ),
        (
            "htj2k_decode/output_regions/tests.rs",
            sources.output_region_tests.as_str(),
            150,
        ),
    ] {
        assert!(
            source.lines().count() < max_lines,
            "CUDA {relative} must stay below its {max_lines}-line focus ratchet"
        );
    }
}

#[test]
fn htj2k_shared_outputs_require_disjoint_validated_regions() {
    let sources = Htj2kOutputSources::read();
    assert_pattern_checks(&[
        PatternCheck::new("CUDA HTJ2K output-region integration", &sources.decode).required(&[
            "mod context_validation;",
            "mod output_regions;",
            "mod queued;",
            "context_validation::validate_cleanup_context",
            "context_validation::validate_dequantize_context",
            "output_regions::validate_htj2k_output_layout",
            "output_regions::ValidatedHtj2kOutputLayout",
            "pub use self::queued::CudaQueuedHtj2kCleanup;",
            "if output_layout.needs_zero_fill {",
            "self.memset_d32_async(coefficient_buffer, 0, output_words)?;",
        ]),
        PatternCheck::new("CUDA empty HTJ2K output completion", &sources.context).required(&[
            "if htj2k_decode_needs_zero_fill(jobs, output_words)? {",
            "self.memset_d32(&coefficients, 0, output_words)?;",
            "self.synchronize()?;",
        ]),
        PatternCheck::new(
            "CUDA HTJ2K disjoint output-region validation",
            &sources.output_regions,
        )
        .required(&[
            "fn output_rect(",
            "fn validate_disjoint_htj2k_job_outputs_with_live_bytes(",
            "fn validate_htj2k_output_layout(",
            "mod sweep;",
        ]),
        PatternCheck::new(
            "CUDA HTJ2K output-region sweeps",
            &sources.output_region_sweep,
        )
        .required(&[
            "fn validate_same_stride_rects(",
            "fn validate_disjoint_output_regions(",
            "mod cross_stride;",
            "BinaryHeap",
            "try_vec_with_capacity",
            "binary_search_by_key",
            "partition_point",
            "sort_unstable_by_key",
            "active column intervals mutually",
            "jobs sharing one output must write disjoint regions",
        ])
        .forbidden(&["BTreeMap"]),
        PatternCheck::new(
            "CUDA HTJ2K cross-stride output-region sweep",
            &sources.output_region_cross_stride,
        )
        .required(&[
            "fn validate_cross_stride_spans(",
            "HostPhaseBudget::with_live_bytes(",
            "BinaryHeap",
            "sort_unstable_by_key",
            "different-stride HTJ2K output spans must be disjoint",
        ])
        .forbidden(&["BTreeMap"]),
        PatternCheck::new(
            "CUDA HTJ2K output-region adversarial coverage",
            &sources.output_region_tests,
        )
        .required(&[
            "accepts_disjoint_rectangles_in_one_output_plane",
            "rejects_overlapping_rectangles_and_row_wrap",
            "accepts_disjoint_mixed_strides_and_rejects_overlapping_spans",
            "accepts_large_disjoint_grid_without_quadratic_pair_scanning",
            "rejects_overlap_hidden_after_many_disjoint_rectangles",
            "repeated_column_start_after_expiry_keeps_new_interval_indexed",
            "coverage_planning_rejects_overlap_even_when_areas_sum_to_output",
            "coverage_planning_distinguishes_full_output_from_gaps",
        ]),
    ]);
}
