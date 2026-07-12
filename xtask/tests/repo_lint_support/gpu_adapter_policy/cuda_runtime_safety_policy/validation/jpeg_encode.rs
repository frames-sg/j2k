// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{assert_pattern_checks, read_repo, read_runtime, PatternCheck};

struct JpegEncodeSources {
    encode: String,
    batch: String,
    launch: String,
    validation: String,
    layout: String,
    tables: String,
    tests: String,
    boundary_tests: String,
    huffman_tests: String,
}

impl JpegEncodeSources {
    fn read() -> Self {
        Self {
            encode: read_runtime("jpeg/encode.rs"),
            batch: read_runtime("jpeg/encode_batch.rs"),
            launch: read_runtime("jpeg/encode_launch.rs"),
            validation: read_runtime("jpeg/encode_validation.rs"),
            layout: read_runtime("jpeg/encode_validation/layout.rs"),
            tables: read_runtime("jpeg/encode_validation/tables.rs"),
            tests: read_runtime("jpeg/encode_validation/tests.rs"),
            boundary_tests: read_runtime("jpeg/encode_validation/tests/boundaries.rs"),
            huffman_tests: read_runtime("jpeg/encode_validation/tests/huffman.rs"),
        }
    }
}

#[test]
fn cuda_jpeg_encode_modules_stay_focused() {
    let sources = JpegEncodeSources::read();
    for (relative, source, max_lines) in [
        ("encode.rs", sources.encode.as_str(), 325usize),
        ("encode_batch.rs", sources.batch.as_str(), 250),
        ("encode_launch.rs", sources.launch.as_str(), 200),
        ("encode_validation.rs", sources.validation.as_str(), 250),
        ("encode_validation/layout.rs", sources.layout.as_str(), 225),
        ("encode_validation/tables.rs", sources.tables.as_str(), 150),
        ("encode_validation/tests.rs", sources.tests.as_str(), 350),
        (
            "encode_validation/tests/boundaries.rs",
            sources.boundary_tests.as_str(),
            125,
        ),
        (
            "encode_validation/tests/huffman.rs",
            sources.huffman_tests.as_str(),
            150,
        ),
    ] {
        assert!(
            source.lines().count() < max_lines,
            "CUDA JPEG {relative} must stay below its {max_lines}-line focus ratchet"
        );
    }
}

#[test]
fn jpeg_encode_safe_apis_require_full_bounds_and_alias_validation() {
    let sources = JpegEncodeSources::read();
    let cuda_encode_tests = read_repo("crates/j2k-jpeg-cuda/tests/encode.rs");
    assert_pattern_checks(&[
        PatternCheck::new("CUDA JPEG encode validation integration", &sources.encode).required(&[
            "encode_launch::{",
            "validate_jpeg_baseline_encode_request(",
            "job.input.device_ptr()",
            "job.input.byte_len()",
            "validated.first_tile.input_ptr",
            "validated.first_tile.entropy_offset",
            "let batch_geometry = validate_jpeg_encode_batch_launch(validated.tile_count)?;",
            "self.execute_jpeg_baseline_entropy_batch(",
        ]),
        PatternCheck::new("CUDA JPEG validated batch handoff", &sources.batch).required(&[
            "validated: CudaJpegBaselineEncodeValidation",
            "geometry: CudaLaunchGeometry",
            "fn launch_jpeg_baseline_entropy_batch_job(",
            "self.inner.set_current()?;",
            "tile_count: validated.tile_count",
            "geometry,",
        ]),
        PatternCheck::new("CUDA JPEG encode focused launch leaf", &sources.launch)
            .required(&[
                "struct CudaJpegBaselineEntropyLaunch",
                "struct CudaJpegBaselineEntropyBatchLaunch",
                "fn launch_jpeg_encode_baseline_entropy(",
                "fn launch_jpeg_encode_baseline_entropy_batch(",
                "fn jpeg_encode_kernel_function(",
                "fn validate_jpeg_encode_status(",
                "cuda_kernel_params!(",
                "CudaKernel::JpegEncodeBaselineEntropyBatch",
            ])
            .normalized_required(&[
                "cuda_kernel_params!( input_ptr, entropy_ptr, status_ptr, params, q_luma_ptr, q_chroma_ptr, huff_dc_luma_ptr, huff_ac_luma_ptr, huff_dc_chroma_ptr, huff_ac_chroma_ptr )",
                "cuda_kernel_params!( input_ptr, entropy_ptr, status_ptr, params_ptr, q_luma_ptr, q_chroma_ptr, huff_dc_luma_ptr, huff_ac_luma_ptr, huff_dc_chroma_ptr, huff_ac_chroma_ptr, tile_count )",
            ])
            .forbidden(&["pub fn encode_jpeg_baseline_entropy("]),
        PatternCheck::new("CUDA JPEG encode range and alias validation", &sources.validation)
            .required(&[
                "try_vec_with_capacity(params.len())?",
                "ranges.sort_unstable_by_key",
                ".windows(2)",
                "entropy ranges for tiles",
                "input offset and row footprint overflow usize",
                "entropy range is not addressable by u32 byte indexes",
            ]),
        PatternCheck::new("CUDA JPEG encode kernel-index validation", &sources.layout).required(&[
            "fn validate_kernel_index_products(",
            "checked_mul(params.pitch_bytes)",
            "checked_mul(bytes_per_pixel)",
            "fn validate_last_sample_axis(",
            "last MCU {axis} origin overflows u32",
            "component averaging sum overflows u32",
        ]),
        PatternCheck::new("CUDA JPEG encode canonical table validation", &sources.tables)
            .required(&[
                "fn validate_huffman_table(",
                "entries.sort_unstable()",
                "j2k_codec_math::jpeg::derive_canonical_huffman",
                "JPEG-prohibited all-ones code",
                "canonical prefix-free code progression",
                "validate_quant_table(\"luma\"",
            ]),
        PatternCheck::new("CUDA JPEG encode adversarial validation", &sources.tests).required(&[
            "rejects_each_format_component_and_sampling_field",
            "rejects_pitch_and_every_input_range_failure_before_launch",
            "accepts_adjacent_entropy_ranges_and_rejects_aliases_by_original_index",
            "large_entropy_range_sweep_finds_overlap_without_quadratic_pair_scanning",
            "later_invalid_batch_tile_fails_the_whole_preflight",
        ]),
        PatternCheck::new("CUDA JPEG encode exact-boundary validation", &sources.boundary_tests)
            .required(&[
                "accepts_exact_input_and_entropy_ends_then_rejects_one_byte_short",
                "accepts_exact_u32_index_end_and_rejects_the_next_byte",
                "accepts_last_u32_input_index_and_rejects_row_wrap",
                "rejects_zero_out_of_allocation_and_oversized_entropy_ranges",
            ]),
        PatternCheck::new("CUDA JPEG encode Huffman adversarial validation", &sources.huffman_tests)
            .required(&[
                "accepts_empty_missing_and_canonical_prefix_free_huffman_entries",
                "rejects_duplicate_prefix_conflicting_noncanonical_and_all_ones_codes",
            ]),
        PatternCheck::new("CUDA JPEG encode bound-offset integration", &cuda_encode_tests)
            .required(&[
                "cuda_resident_single_encode_honors_prefixed_buffer_offset_when_required",
                "let mut prefixed = vec![0xa5; 37];",
                "byte_offset,",
                "assert_rgb_close(&decoded, &pixels, 40);",
            ]),
    ]);
}

#[test]
fn jpeg_encode_validation_precedes_every_driver_operation() {
    let encode = read_runtime("jpeg/encode.rs");
    let encode_batch = read_runtime("jpeg/encode_batch.rs");
    let batch_start = encode
        .find("pub fn encode_jpeg_baseline_entropy_batch(")
        .expect("find CUDA JPEG batch entry point");
    let single = &encode[..batch_start];
    let batch = &encode[batch_start..];
    let single_validation = single
        .rfind("validate_jpeg_baseline_encode_request(")
        .expect("find single CUDA JPEG validation call");
    let single_driver_work = [
        "self.inner.set_current()",
        "self.allocate(",
        "self.upload(",
        "self.launch_jpeg_encode_baseline_entropy",
    ]
    .into_iter()
    .filter_map(|operation| single.find(operation))
    .min()
    .expect("find single CUDA JPEG first driver operation");
    assert!(
        single_validation < single_driver_work,
        "CUDA JPEG single validation must finish before the first driver operation"
    );

    let batch_validation = batch
        .rfind("validate_jpeg_baseline_encode_request(")
        .expect("find batch CUDA JPEG validation call");
    let geometry_validation = batch
        .find("validate_jpeg_encode_batch_launch(validated.tile_count)?")
        .expect("find batch CUDA JPEG geometry validation");
    let validated_handoff = batch
        .find("self.execute_jpeg_baseline_entropy_batch(")
        .expect("find validated CUDA JPEG batch handoff");
    assert!(
        batch_validation < geometry_validation && geometry_validation < validated_handoff,
        "CUDA JPEG batch request and geometry validation must precede the execution handoff"
    );

    let launch_start = encode_batch
        .find("fn launch_jpeg_baseline_entropy_batch_job(")
        .expect("find validated CUDA JPEG batch launch owner");
    let launch = &encode_batch[launch_start..];
    let typed_plan = launch
        .find("validated: CudaJpegBaselineEncodeValidation")
        .expect("find typed CUDA JPEG batch validation input");
    let typed_geometry = launch
        .find("geometry: CudaLaunchGeometry")
        .expect("find typed CUDA JPEG batch geometry input");
    let first_driver_work = launch
        .find("self.inner.set_current()?")
        .expect("find CUDA JPEG batch first driver operation");
    assert!(
        typed_plan < first_driver_work && typed_geometry < first_driver_work,
        "CUDA JPEG batch driver work must require the validated request and geometry handoff"
    );
}
