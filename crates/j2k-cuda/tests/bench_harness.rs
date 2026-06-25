// SPDX-License-Identifier: MIT OR Apache-2.0

#[test]
fn cuda_htj2k_decode_bench_exposes_gray_rgb_rgba_rows() {
    let bench = include_str!("../benches/htj2k_decode.rs");

    for expected in [
        "cpu_gray8",
        "cuda_gray8",
        "cpu_rgb8",
        "cuda_rgb8",
        "cpu_rgba8",
        "cuda_rgba8",
        "j2k_cuda_htj2k_full_tile_decode",
        "j2k_cuda_htj2k_roi_decode",
        "j2k_cuda_htj2k_scaled_decode",
        "j2k_cuda_htj2k_roi_scaled_decode",
        "j2k_cuda_htj2k_tile_batch_decode",
        "j2k_cuda_htj2k_external_mixed_tile_batch_decode",
        "cpu_external_mixed_",
        "cuda_external_mixed_",
        "BATCH_SIZES",
        "[8, 16, 32, 64]",
        "J2K_CUDA_DECODE_BATCH_SIZES",
        "J2K_CUDA_DECODE_CASE_BATCH_SIZES",
        "J2K_CUDA_DECODE_SAMPLE_SIZE",
        "J2K_CUDA_DECODE_FORMATS",
        "J2K_CUDA_DECODE_INPUT_DIRS",
        "J2K_CUDA_DECODE_MANIFEST",
        "J2K_CUDA_DECODE_INCLUDE_GENERATED",
        "J2K_REQUIRE_CUDA_BENCH",
        "j2k_cuda_decode_batch_sizes",
        "j2k_cuda_decode_case_batch_sizes",
        "j2k_cuda_decode_mixed_batch_sizes",
        "j2k_cuda_decode_sample_size",
        "j2k_cuda_decode_batch_policy",
        "j2k_cuda_decode_io_policy",
        "j2k_cuda_decode_external_case_count",
        "j2k_cuda_decode_external_fixture_count",
        "j2k_cuda_decode_external_skipped_non_htj2k_count",
        "j2k_cuda_decode_external_skipped_unsupported_shape_count",
        "j2k_cuda_decode_external_skipped_format_disabled_count",
        "validate_manifest_entry",
        "external CUDA decode fixture",
        "input_fnv1a64",
        "codec",
        "container",
    ] {
        assert!(
            bench.contains(expected),
            "CUDA HTJ2K decode benchmark is missing `{expected}`"
        );
    }
}

#[test]
fn cuda_htj2k_decode_bench_reuses_session_in_timed_cuda_rows() {
    let bench = include_str!("../benches/htj2k_decode.rs");

    assert!(
        bench
            .matches("let mut session = CudaSession::default();")
            .count()
            >= 6,
        "CUDA decode benchmarks must create reusable sessions outside timed iterations"
    );

    for forbidden in [
        ".decode_to_device(case.fmt, BackendRequest::Cuda)",
        ".decode_region_to_device(case.fmt, roi, BackendRequest::Cuda)",
        ".decode_scaled_to_device(case.fmt, scale, BackendRequest::Cuda)",
        ".decode_region_scaled_to_device(case.fmt, roi, scale, BackendRequest::Cuda)",
    ] {
        assert!(
            !bench.contains(forbidden),
            "CUDA decode benchmark timed row must not call context-creating helper `{forbidden}`"
        );
    }

    for expected in [
        ".submit_to_device(",
        ".submit_region_to_device(",
        ".submit_scaled_to_device(",
        ".submit_region_scaled_to_device(",
    ] {
        assert!(
            bench.contains(expected),
            "CUDA decode benchmark is missing reusable-session path `{expected}`"
        );
    }
}

#[test]
fn cuda_htj2k_tile_batch_bench_uses_cuda_batch_entrypoint() {
    let bench = include_str!("../benches/htj2k_decode.rs");
    let batch_body = extract_function_body(bench, "fn bench_tile_batch");
    let cuda_branch_start = batch_body
        .find("if case.cuda_available && cuda_batch_decode_supported(fmt)")
        .expect("CUDA batch branch exists");
    let cuda_batch_body = &batch_body[cuda_branch_start..];

    assert!(
        cuda_batch_body.contains("J2kDecoder::decode_batch_to_device_with_session("),
        "CUDA HTJ2K tile batch row must use a real batch decode entrypoint"
    );
    assert!(
        !cuda_batch_body.contains("Codec::submit_tile_to_device("),
        "CUDA HTJ2K tile batch row must not submit one tile at a time"
    );
}

#[test]
fn cuda_htj2k_encode_bench_accepts_external_staged_pnm_sources() {
    let bench = include_str!("../benches/htj2k_encode.rs");

    for expected in [
        "j2k_cuda_htj2k_external_host_input_encode",
        "cpu_external_",
        "cuda_external_",
        "J2K_CUDA_ENCODE_INPUT_DIRS",
        "J2K_CUDA_ENCODE_MANIFEST",
        "J2K_CUDA_ENCODE_INCLUDE_GENERATED",
        "j2k_cuda_encode_io_policy",
        "j2k_cuda_encode_external_case_count",
        "j2k_cuda_encode_external_input_format",
        "staged-pnm-p5-p6",
        "read_pnm_image",
        "validate_cuda_encode_manifest_entry",
        "input_fnv1a64",
    ] {
        assert!(
            bench.contains(expected),
            "CUDA HTJ2K encode benchmark is missing `{expected}`"
        );
    }
}

#[test]
fn cuda_htj2k_decode_profile_example_uses_batch_entrypoint() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/htj2k_decode_profile.rs"
    );
    let example = std::fs::read_to_string(path).expect("read CUDA HTJ2K profile example");

    for expected in [
        "J2K_CUDA_PROFILE_BATCH_SIZE",
        "J2K_CUDA_PROFILE_ITERATIONS",
        "let mut session = CudaSession::default();",
        "decode_batch_to_device_with_session(",
        "mode=batch_no_download",
    ] {
        assert!(
            example.contains(expected),
            "CUDA HTJ2K profile example is missing `{expected}`"
        );
    }

    assert!(
        !example.contains(".decode_to_device_with_session(PixelFormat::Rgb8, &mut session)")
            && !example.contains("for fixture in &fixtures"),
        "CUDA HTJ2K profile example must not report batch_size while looping through single-tile decodes"
    );
}

#[test]
fn cuda_htj2k_decode_steady_state_uses_untimed_runtime_path() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/src/decoder.rs");
    let decoder = std::fs::read_to_string(path).expect("read CUDA HTJ2K decoder");

    for expected in [
        "decode_to_cuda_resident_surface_with_profile_control(decoder, session, fmt, false)",
        "decode_to_cuda_resident_surface_with_profile_impl(self, session, fmt)",
        "decode_to_cuda_resident_surface_with_profile_control(decoder, session, fmt, true)",
        "collect_stage_timings",
        "decode_htj2k_codeblocks_cleanup_multi_enqueue_with_resources_and_pool",
        "j2k_inverse_dwt_single_device_untimed_with_pool",
        "time_default_stream_named_us_if",
    ] {
        assert!(
            decoder.contains(expected),
            "steady-state CUDA HTJ2K decode path is missing `{expected}`"
        );
    }
}

#[test]
fn cuda_runtime_exposes_untimed_htj2k_decode_helpers() {
    let runtime = read_cuda_runtime_sources();

    for expected in [
        "pub fn synchronize(&self) -> Result<(), CudaError>",
        "pub fn time_default_stream_named_us_if",
        "pub fn decode_htj2k_codeblocks_with_resources_untimed",
        "pub fn j2k_inverse_dwt_single_device_untimed",
        "pinned_upload_staging",
        "take_pinned_upload_staging",
        "recycle_pinned_upload_staging",
        "enum CudaLaunchMode",
        "CudaLaunchMode::Async",
        "fn launch_htj2k_decode_codeblocks(",
        "fn launch_j2k_dequantize_htj2k_codeblocks(",
        "fn launch_j2k_idwt_interleave(",
    ] {
        assert!(
            runtime.contains(expected),
            "CUDA runtime is missing steady-state decode helper `{expected}`"
        );
    }
}

fn read_cuda_runtime_sources() -> String {
    let src_dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../j2k-cuda-runtime/src");
    let mut runtime = String::new();

    for module in [
        "lib.rs",
        "context.rs",
        "execution.rs",
        "memory.rs",
        "htj2k_decode.rs",
        "j2k_decode.rs",
    ] {
        let path = format!("{src_dir}/{module}");
        runtime.push_str(&std::fs::read_to_string(&path).expect("read CUDA runtime module"));
        runtime.push('\n');
    }

    runtime
}

fn extract_function_body<'a>(source: &'a str, signature: &str) -> &'a str {
    let start = source.find(signature).expect("function signature exists");
    let function = &source[start..];
    let mut depth = 0usize;
    let mut saw_open = false;
    for (index, ch) in function.char_indices() {
        match ch {
            '{' => {
                saw_open = true;
                depth = depth.saturating_add(1);
            }
            '}' if saw_open => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    return &function[..=index];
                }
            }
            _ => {}
        }
    }
    panic!("function body for `{signature}` is incomplete");
}
