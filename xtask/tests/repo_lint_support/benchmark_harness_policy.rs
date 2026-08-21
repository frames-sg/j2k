// SPDX-License-Identifier: MIT OR Apache-2.0

//! Required stage benchmark surfaces for planned GPU performance work.

use super::{assert_file_pattern_checks, repo_root, FilePatternCheck};
use std::path::Path;

#[test]
fn required_gpu_stage_benchmarks_are_registered_and_observable() {
    assert_file_pattern_checks(
        repo_root(),
        &[
            FilePatternCheck::new("crates/j2k-metal/Cargo.toml")
                .required(&["name = \"decode_stages\""]),
            FilePatternCheck::new("crates/j2k-metal/benches/decode_stages.rs").required(&[
                "entropy_tier1",
                "dequantization",
                "idwt",
                "inverse_mct",
                "final_store",
                "readback",
                "dispatch_report",
            ]),
            FilePatternCheck::new("crates/j2k-transcode-cuda/Cargo.toml")
                .required(&["name = \"dwt97\""]),
            FilePatternCheck::new("crates/j2k-transcode-cuda/benches/dwt97.rs")
                .required(&[
                    "staged_column_lift_quantize",
                    "pack_upload_us",
                    "column_lift_us",
                    "quantize_codeblock_us",
                    "readback_us",
                    "temporary_float_band_traffic_bytes",
                    "output_sha256",
                    "P13 product tile codestream decodes",
                ])
                .forbidden(&[
                    "J2K_CUDA_DISABLE_DWT97_FUSED_COLUMN_QUANTIZE",
                    "unfused_column_lift_quantize",
                    "fused_column_lift_quantize",
                ]),
        ],
    );
}

#[test]
fn rejected_cuda_wide_idwt_has_no_candidate_or_active_switch() {
    let root = repo_root();
    for candidate in [
        "crates/j2k-cuda-j2k-engine/src/j2k_decode/idwt_launch/tiled.rs",
        "crates/j2k-cuda-j2k-engine/src/cuda_oxide_j2k_idwt/simt/src/tiled.rs",
    ] {
        assert!(
            !root.join(candidate).exists(),
            "rejected P14 candidate file must be removed: {candidate}"
        );
    }
    assert_file_pattern_checks(
        root,
        &[
            FilePatternCheck::new("crates/j2k-cuda-j2k-engine/Cargo.toml")
                .required(&[
                    "name = \"idwt_wide\"",
                    "\"benches/**\"",
                    "required-features = [\"cuda-oxide-j2k-idwt\"]",
                ])
                .forbidden(&["benchmark-internals"]),
            FilePatternCheck::new("crates/j2k-cuda-j2k-engine/benches/idwt_wide.rs")
                .required(&[
                    "reversible53",
                    "irreversible97",
                    "narrow_512x512",
                    "wide_2592x1944",
                    "const BATCH_SIZES: &[usize] = &[1, 16]",
                    "x0: 1",
                    "y0: 1",
                    "input_sha256",
                    "output_sha256",
                    "exact_parity=true",
                    "route=",
                    "dispatch_count=",
                    "stage_wall_time_ns=",
                    "j2k_cuda_idwt_stage",
                    "priority_end_to_end_decode=j2k_cuda_htj2k_tile_batch_decode",
                ])
                .forbidden(&[
                    "J2K_CUDA_IDWT_WIDE_TILED",
                    "tiled_cooperative",
                    "treatment_dispatches",
                    "set_var(",
                    "remove_var(",
                ]),
            FilePatternCheck::new("crates/j2k-cuda/benches/htj2k_decode.rs").required(&[
                "const TILE_DIM: u32 = 512",
                "const BATCH_SIZES: &[usize] = &[8, 16, 32, 64]",
                "j2k_cuda_htj2k_tile_batch_decode",
                "J2kDecoder::decode_batch_to_device_with_session",
            ]),
            FilePatternCheck::new("crates/j2k-cuda-j2k-engine/src/context.rs")
                .forbidden(&["J2K_CUDA_IDWT_WIDE_TILED", "cuda_idwt_wide_tiled_enabled"]),
            FilePatternCheck::new("crates/j2k-cuda-j2k-engine/src/j2k_decode.rs")
                .forbidden(&["TiledCooperative53", "TiledCooperative97"]),
            FilePatternCheck::new("crates/j2k-cuda-j2k-engine/src/j2k_decode/trace.rs").forbidden(
                &[
                    "J2K_CUDA_IDWT_WIDE_TILED",
                    "benchmark_force_tiled",
                    "TiledCooperative",
                ],
            ),
            FilePatternCheck::new("crates/j2k-cuda-j2k-engine/src/kernels.rs").forbidden(&[
                "J2kIdwtInterleaveTiledMulti",
                "J2kIdwtHorizontal53TiledMulti",
                "J2kIdwtHorizontal97TiledMulti",
                "J2kIdwtVertical53TiledMulti",
                "J2kIdwtVertical97TiledMulti",
                "j2k_idwt_multi_tiled_axis_launch_geometry",
            ]),
            FilePatternCheck::new(
                "crates/j2k-cuda-j2k-engine/src/cuda_oxide_j2k_idwt/simt/src/main.rs",
            )
            .forbidden(&["IDWT_TILED", "_tiled_multi", "mod tiled;"]),
            FilePatternCheck::new("docs/env-vars.md")
                .required(&["J2K_CUDA_IDWT_WIDE_TILED", "Historical P14"]),
        ],
    );
}

#[test]
fn cuda_final_store_preprofile_is_matrix_complete_and_has_no_candidate() {
    assert_file_pattern_checks(
        repo_root(),
        &[
            FilePatternCheck::new("crates/j2k-cuda/benches/htj2k_decode.rs")
                .required(&[
                    "j2k_cuda_p17_final_store_profile",
                    "classic_reversible53",
                    "classic_irreversible97",
                    "ht_reversible53",
                    "ht_irreversible97",
                    "const P17_BATCH_SIZES: &[usize] = &[1, 16]",
                    "width=512 height=512 components=3 sampling=4:4:4 output=rgb8 operation=full",
                    "input_sha256=",
                    "output_sha256=",
                    "exact_parity=true",
                    "deterministic=true",
                    "J2K_REQUIRE_CUDA_BENCH",
                    "idwt_us=",
                    "idwt_final_interleave_horizontal_us=",
                    "idwt_final_vertical_us=",
                    "fused_mct_store_us=",
                    "resident_wall_total_us=",
                    "J2kDecoder::decode_batch_to_device_with_session_and_profile",
                ])
                .forbidden(&[
                    "J2K_CUDA_DISABLE_FINAL_IDWT_MCT_STORE",
                    "fused_final_idwt",
                    "set_var(",
                    "remove_var(",
                ]),
            FilePatternCheck::new("crates/j2k-cuda-j2k-engine/src/j2k_decode/trace.rs")
                .required(&["final_stage", "interleave_horizontal_us", "vertical_us"]),
            FilePatternCheck::new("crates/j2k-cuda-j2k-engine/src/j2k_decode/idwt_launch/batch.rs")
                .required(&[
                    "launch_j2k_idwt_batch_interleave_horizontal_ptr",
                    "launch_j2k_idwt_batch_vertical_ptr",
                ]),
            FilePatternCheck::new(
                "crates/j2k-cuda-j2k-engine/src/j2k_decode/idwt_launch/profiling.rs",
            )
            .required(&[
                "profile_j2k_idwt_batch_mode_ptr",
                "elapsed_event_us_ceil(&start, &horizontal_end)",
                "elapsed_event_us_ceil(&horizontal_end, &end)",
            ]),
            FilePatternCheck::new("crates/j2k-cuda-j2k-engine/src/j2k_decode/idwt/sequence.rs")
                .required(&[
                    "launch.collect_stage_profile && final_stage",
                    "final_stage_profile = stage_profile",
                ]),
        ],
    );
}

#[test]
fn rejected_cuda_fdwt97_shared_staging_retains_only_the_generic_baseline_benchmark() {
    assert_file_pattern_checks(
        repo_root(),
        &[
            FilePatternCheck::new("crates/j2k-cuda/Cargo.toml").required(&[
                "name = \"fdwt97_baseline\"",
                "required-features = [\"cuda-runtime\"]",
            ]),
            FilePatternCheck::new("crates/j2k-cuda/benches/fdwt97_baseline.rs")
                .required(&[
                    "generic_baseline",
                    "small_512x512",
                    "representative_1024x1024",
                    "large_2592x1944",
                    "const BATCH_SIZES: &[usize] = &[1, 16]",
                    "product_htj2k_rgb_512x512_batch16",
                    "encode_j2k_lossy_with_accelerator",
                    "input_sha256",
                    "output_sha256",
                    "exact_parity=true",
                    "route=generic_baseline",
                    "dispatch_count=",
                    "static_global_load_bytes=",
                    "P15 product codestream decodes",
                    "j2k_cuda_fdwt97_stage",
                    "j2k_cuda_p15_product_encode",
                ])
                .forbidden(&[
                    "J2K_CUDA_DISABLE_SHARED_FDWT97",
                    "J2K_CUDA_FDWT97_TRACE",
                    "shared_tiled",
                    "shared_bytes_",
                    "set_var(",
                    "remove_var(",
                ]),
            FilePatternCheck::new("crates/j2k-cuda-j2k-engine/src/kernels.rs").forbidden(&[
                "J2kForwardDwt97HorizontalSharedTiled",
                "J2kForwardDwt97VerticalSharedTiled",
                "j2k_fdwt97_shared_horizontal_launch_geometry",
                "j2k_fdwt97_shared_vertical_launch_geometry",
            ]),
            FilePatternCheck::new(
                "crates/j2k-cuda-j2k-engine/src/cuda_oxide_j2k_encode/simt/src/exports.rs",
            )
            .forbidden(&[
                "j2k_forward_dwt97_horizontal_shared_tiled",
                "j2k_forward_dwt97_vertical_shared_tiled",
            ]),
            FilePatternCheck::new(
                "crates/j2k-cuda-j2k-engine/src/cuda_oxide_j2k_encode/simt/src/dwt97.rs",
            )
            .forbidden(&["FDWT97_SHARED", "Fdwt97SharedLine", "_shared"]),
            FilePatternCheck::new("crates/j2k-cuda-j2k-engine/src/j2k_encode/dwt.rs")
                .forbidden(&["fdwt97_staging", "CudaFdwt97Axis"]),
            FilePatternCheck::new("docs/env-vars.md").required(&[
                "J2K_CUDA_DISABLE_SHARED_FDWT97",
                "J2K_CUDA_FDWT97_TRACE",
                "Historical P15",
            ]),
        ],
    );
}

#[test]
fn rejected_cuda_input_fusion_retains_only_the_separate_route_benchmark() {
    assert_file_pattern_checks(
        repo_root(),
        &[
            FilePatternCheck::new("crates/j2k-cuda/Cargo.toml").required(&[
                "name = \"input_fusion\"",
                "required-features = [\"cuda-runtime\"]",
            ]),
            FilePatternCheck::new("crates/j2k-cuda/benches/input_fusion.rs")
                .required(&[
                    "BenchmarkId::new(\"rct\"",
                    "BenchmarkId::new(\"ict\"",
                    "route=separate_baseline",
                    "j2k_cuda_p16_product_encode",
                    "BenchmarkId::new(\"rct_lossless_htj2k\"",
                    "BenchmarkId::new(\"ict_lossy_htj2k\"",
                    "encode_j2k_lossless_with_accelerator",
                    "encode_j2k_lossy_with_accelerator",
                    "EncodeBackendPreference::RequireDevice",
                    "J2kBlockCodingMode::HighThroughput",
                    "J2kEncodeValidation::External",
                    "Image::new(codestream, &DecodeSettings::strict())",
                    "P16 lossless product decoded pixels",
                    "P16 lossy product PSNR",
                    "codestream_sha256=",
                    "product_route=",
                    "product_deinterleave_dispatches=",
                    "product_mct_dispatches=",
                    "product_physical_input_dispatches=",
                    "product_total_dispatches=",
                    "input_sha256=",
                    "output_sha256=",
                    "exact_parity=true",
                    "deterministic=true",
                    "width=512",
                    "height=512",
                    "bit_depth=8",
                    "signed=false",
                    "deinterleave_dispatches=",
                    "mct_dispatches=",
                    "physical_dispatches=",
                ])
                .forbidden(&[
                    "J2K_CUDA_DISABLE_FUSED_INPUT_MCT",
                    "fused",
                    "combined_dispatches",
                    "BenchmarkId::new(format!",
                    "BenchmarkId::new(variant",
                    "set_var(",
                    "remove_var(",
                ]),
            FilePatternCheck::new("crates/j2k-cuda/src/encode.rs")
                .forbidden(&["mod input_fusion;"]),
            FilePatternCheck::new("crates/j2k-cuda/src/encode/stage.rs").forbidden(&[
                "combined_input_mct_attempts",
                "combined_input_mct_dispatches",
                "fused_input_mct",
            ]),
            FilePatternCheck::new("crates/j2k-cuda/src/encode/htj2k/resident.rs")
                .forbidden(&["fused_input_mct", "j2k_deinterleave_mct"]),
            FilePatternCheck::new("crates/j2k-cuda-j2k-engine/src/kernels.rs").forbidden(&[
                "J2kDeinterleaveMctToF32",
                "J2kDeinterleaveMctStridedToF32",
                "j2k_deinterleave_mct_to_f32",
                "j2k_deinterleave_mct_strided_to_f32",
            ]),
            FilePatternCheck::new(
                "crates/j2k-cuda-j2k-engine/src/cuda_oxide_j2k_encode/simt/src/exports.rs",
            )
            .forbidden(&[
                "j2k_deinterleave_mct_to_f32",
                "j2k_deinterleave_mct_strided_to_f32",
                "store_forward_mct_rgb",
            ]),
            FilePatternCheck::new("docs/env-vars.md").required(&[
                "J2K_CUDA_DISABLE_FUSED_INPUT_MCT",
                "Historical P16",
                "no effect",
            ]),
        ],
    );
}

#[test]
fn promoted_cuda_jpeg_staged_encode_has_no_serial_fallback_or_active_switch() {
    assert_file_pattern_checks(
        repo_root(),
        &[
            FilePatternCheck::new("crates/j2k-jpeg-cuda/Cargo.toml").required(&[
                "name = \"encode_baseline\"",
                "required-features = [\"cuda-runtime\"]",
                "sha2 = { workspace = true }",
                "jpeg-decoder = { workspace = true }",
            ]),
            FilePatternCheck::new("crates/j2k-jpeg-cuda/benches/encode_baseline.rs")
                .required(&[
                    "j2k_cuda_p18_staged_encode",
                    "rgb8_422_512x512_batch8_q90_restart_none",
                    "rgb8_422_512x512_batch1_q90_restart_none",
                    "rgb8_422_64x64_batch1_q90_restart_none",
                    "rgb8_422_512x512_batch8_q90_restart16",
                    "rgb8_422_512x512_batch8_q90_restart32",
                    "input_sha256=",
                    "output_sha256=",
                    "exact_codestreams=true",
                    "deterministic=true",
                    "repository_decode=true",
                    "independent_decode=true",
                    "kernel_dispatches=",
                    "host_to_device_transfers=",
                    "device_to_host_transfers=",
                    "device_allocations=",
                    "host_synchronizations=",
                    "coefficient_scratch_bytes=",
                    "route=staged",
                    "encode_jpeg_baseline_batch_from_cuda_buffers",
                    "J2K_REQUIRE_CUDA_BENCH",
                ])
                .forbidden(&[
                    "J2K_CUDA_JPEG_STAGED_ENCODE",
                    "J2K_CUDA_DISABLE_STAGED_JPEG_ENCODE",
                    "J2K_CUDA_JPEG_DISABLE_STAGED_BASELINE_ENCODE",
                    "serial_baseline",
                    "set_var(",
                    "remove_var(",
                ]),
            FilePatternCheck::new("crates/j2k-cuda-jpeg-engine/src/jpeg/encode_staging.rs")
                .required(&[
                    "checked_staged_encode_plan",
                    "coefficient_bytes",
                    "total_mcus",
                ])
                .forbidden(&[
                    "J2K_CUDA_JPEG_DISABLE_STAGED_BASELINE_ENCODE",
                    "OnceLock",
                    "set_var(",
                    "remove_var(",
                ]),
            FilePatternCheck::new("crates/j2k-cuda-jpeg-engine/src/kernels.rs")
                .required(&[
                    "JpegEncodeBaselinePrecomputeBatch",
                    "JpegEncodeBaselineEntropyFromCoeffsBatch",
                ])
                .forbidden(&[
                    "JpegEncodeBaselineEntropy,",
                    "JpegEncodeBaselineEntropyBatch",
                    "j2k_jpeg_encode_baseline_entropy\\0",
                    "j2k_jpeg_encode_baseline_entropy_batch\\0",
                ]),
            FilePatternCheck::new(
                "crates/j2k-cuda-jpeg-engine/src/cuda_oxide_jpeg_encode/simt/src/main.rs",
            )
            .required(&[
                "j2k_jpeg_encode_baseline_precompute_batch",
                "j2k_jpeg_encode_baseline_entropy_from_coeffs_batch",
            ])
            .forbidden(&[
                "pub unsafe fn j2k_jpeg_encode_baseline_entropy(",
                "pub unsafe fn j2k_jpeg_encode_baseline_entropy_batch(",
            ]),
            FilePatternCheck::new("crates/j2k-jpeg-cuda/tests/encode/p18_baseline.rs")
                .required(&[
                    "P18_EXACT_MATRIX_OUTPUT_SHA256",
                    "repository decoder accepts",
                    "independent jpeg-decoder accepts",
                ])
                .forbidden(&[
                    "Command::new",
                    "P18_EXACT_CHILD_ROUTE",
                    "serial_baseline",
                    "J2K_CUDA_JPEG_DISABLE_STAGED_BASELINE_ENCODE",
                ]),
            FilePatternCheck::new("docs/env-vars.md").required(&[
                "J2K_CUDA_JPEG_DISABLE_STAGED_BASELINE_ENCODE",
                "Historical P18",
                "No effect",
            ]),
        ],
    );
}

fn assert_rejected_p19_candidate_files_absent(root: &Path) {
    for candidate in [
        "crates/j2k-cuda-jpeg-engine/src/jpeg/decode_launch/coefficient_idct_split.rs",
        "crates/j2k-cuda-jpeg-engine/src/jpeg/decode_launch/decode_process.rs",
        "crates/j2k-cuda-jpeg-engine/src/jpeg/decode_launch/kernel_launch.rs",
        "crates/j2k-cuda-jpeg-engine/src/cuda_oxide_jpeg_decode/simt/src/coefficient_idct_split.rs",
    ] {
        assert!(
            !root.join(candidate).exists(),
            "rejected P19 coefficient/IDCT split candidate must be removed: {candidate}"
        );
    }
}

fn p19_benchmark_harness_check() -> FilePatternCheck<'static> {
    FilePatternCheck::new("crates/j2k-jpeg-cuda/benches/decode_defusion.rs")
        .required(&[
            "j2k_cuda_p19_decode_adaptive_checkpoints",
            "ybr420_512x512_batch16_restart_none",
            "ybr420_512x512_batch1_restart_none",
            "ybr420_512x512_batch16_restart16",
            "ybr420_512x512_batch1_restart16",
            "ybr422_512x512_batch16_restart_none",
            "ybr422_512x512_batch1_restart_none",
            "ybr444_512x512_batch16_restart_none",
            "ybr444_512x512_batch1_restart_none",
            "ybr420_64x64_batch1_restart_none",
            "ybr420_1024x1024_batch1_restart_none",
            "odd_420_513x517",
            "odd_422_513x257",
            "caller_owned_padded_output",
            "strict_region_rejected=true",
            "strict_scaled_rejected=true",
            "auto_fallback=cpu",
            "input_sha256=",
            "output_sha256=",
            "exact_production_output=true",
            "deterministic=true",
            "cpu_conformance=true",
            "checkpoint_count=",
            "checkpoint_mcu_range=",
            "checkpoint_entropy_range=",
            "total_mcus=",
            "blocks_per_mcu=",
            "route=serial_below_threshold",
            "route=packed_checkpoints",
            "decode_grid=",
            "decode_block=",
            "kernel_dispatches=",
            "coefficient_scratch_bytes=0",
            "resource_upload_us=",
            "fused_decode_kernel_us=",
            "conversion_us=",
            "status_readback_us=",
            "product_wall_us=",
            "probe_repeat=2",
            "warm_cached_packet_product=true",
            "restart32_420=true",
            "J2K_REQUIRE_CUDA_BENCH",
        ])
        .forbidden(&[
            "J2K_CUDA_JPEG_DECODE_DEFUSION",
            "J2K_CUDA_DISABLE_FUSED_JPEG_DECODE",
            "J2K_CUDA_JPEG_DISABLE_PACKED_CHECKPOINTS",
            "J2K_CUDA_JPEG_DISABLE_COEFFICIENT_IDCT_SPLIT",
            "serial_block1_baseline",
            "adaptive_serial_below_threshold_treatment",
            "adaptive_packed_checkpoint_treatment",
            "exact_baseline_output=true",
            "decode_process_arm=",
            "logical_coefficient_scratch_traffic_bytes=",
            "coefficient_scratch_clear_us=",
            "entropy_coefficients_us=",
            "idct_deposit_us=",
            "stale_scratch_dc_only_after_ac=true",
            "CheckpointLaunchArm",
            "set_var(",
            "remove_var(",
        ])
}

#[test]
fn rejected_cuda_jpeg_coefficient_idct_split_retains_adaptive_fused_baseline() {
    let root = repo_root();
    assert_rejected_p19_candidate_files_absent(root);
    assert_file_pattern_checks(
        root,
        &[
            FilePatternCheck::new("crates/j2k-jpeg-cuda/Cargo.toml").required(&[
                "name = \"decode_defusion\"",
                "required-features = [\"cuda-runtime\"]",
            ]),
            FilePatternCheck::new("xtask/src/benchmark_registry.rs")
                .required(&["Some(\"decode_defusion\")"]),
            p19_benchmark_harness_check(),
            FilePatternCheck::new("crates/j2k-cuda-jpeg-engine/src/jpeg/decode_launch.rs")
                .required(&[
                    "CudaJpegDecodeStageProfiler",
                    "stage_profiler.begin_fused(self)",
                    "stage_profiler.finish_fused(self)",
                    "stage_profiler.finish_conversion(self)",
                    "stage_profiler.synchronize_device_stages()",
                    "stage_profiler.begin_status_readback()",
                ])
                .forbidden(&[
                    "J2K_CUDA_JPEG_DECODE_DEFUSION",
                    "J2K_CUDA_DISABLE_FUSED_JPEG_DECODE",
                    "coefficient_idct_split",
                    "JpegDecode420",
                ]),
            FilePatternCheck::new(
                "crates/j2k-cuda-jpeg-engine/src/jpeg/decode_launch/profiling.rs",
            )
            .required(&[
                "resource_upload_us",
                "fused_decode_kernel_us",
                "conversion_us",
                "status_readback_us",
                "CudaJpegDecodeStageTimings",
            ])
            .forbidden(&[
                "J2K_CUDA_JPEG_DECODE_DEFUSION",
                "J2K_CUDA_DISABLE_FUSED_JPEG_DECODE",
                "coefficient_scratch_clear_us",
                "entropy_coefficients_us",
                "idct_deposit_us",
            ]),
            FilePatternCheck::new("crates/j2k-cuda-jpeg-engine/src/jpeg/validation/decode_plan.rs")
                .required(&[
                    "JPEG_CHECKPOINT_THREADS_PER_BLOCK: u32 = 128",
                    "JPEG_PACKED_CHECKPOINT_MIN_COUNT: u32 = 128",
                    "checkpoint_count < JPEG_PACKED_CHECKPOINT_MIN_COUNT",
                    "checkpoint_count.div_ceil(JPEG_CHECKPOINT_THREADS_PER_BLOCK)",
                    "CudaLaunchGeometry::new((grid_x, 1, 1), (block_x, 1, 1))",
                ])
                .forbidden(&[
                    "J2K_CUDA_JPEG_DISABLE_PACKED_CHECKPOINTS",
                    "OnceLock",
                    "serial_fallback",
                    "std::env",
                ]),
            FilePatternCheck::new("crates/j2k-cuda-jpeg-engine/src/kernels.rs").forbidden(&[
                "j2k_jpeg_decode_420_entropy_coefficients",
                "j2k_jpeg_decode_420_idct_deposit",
                "JpegDecode420EntropyCoefficients",
                "JpegDecode420IdctDeposit",
            ]),
            FilePatternCheck::new(
                "crates/j2k-cuda-jpeg-engine/src/cuda_oxide_jpeg_decode/simt/src/main.rs",
            )
            .forbidden(&["mod coefficient_idct_split;"]),
            FilePatternCheck::new(
                "crates/j2k-cuda-jpeg-engine/src/cuda_oxide_jpeg_decode/simt/src/component_planes.rs",
            )
            .forbidden(&["store_420_block"]),
            FilePatternCheck::new("docs/env-vars.md").required(&[
                "J2K_CUDA_JPEG_DISABLE_PACKED_CHECKPOINTS",
                "J2K_CUDA_JPEG_DISABLE_COEFFICIENT_IDCT_SPLIT",
                "Historical P19",
                "No effect",
            ]),
        ],
    );
}

#[test]
fn performance_experiment_records_are_validator_owned() {
    assert_file_pattern_checks(
        repo_root(),
        &[
            FilePatternCheck::new("xtask/src/main.rs").required(&["\"gpu-experiment\" =>"]),
            FilePatternCheck::new("xtask/src/gpu_experiment.rs").required(&[
                "schema_version",
                "input_corpus_sha256",
                "sample_count",
                "warm_up_seconds",
                "measurement_seconds",
                "wall_time_ns",
                "private_bytes_per_thread",
                "occupancy_percent",
                "output_sha256",
                "exact_parity",
                "confidence_interval_supports_improvement",
                "baseline and treatment",
            ]),
            FilePatternCheck::new("docs/performance-experiments/README.md").required(&[
                "cargo xtask gpu-experiment validate",
                "reversible 5/3",
                "irreversible 9/7",
                "2592×1944",
                "4:4:4",
                "4:2:2",
                "4:2:0",
                "split-command",
            ]),
        ],
    );
}
