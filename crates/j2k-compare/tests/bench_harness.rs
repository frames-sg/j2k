// SPDX-License-Identifier: MIT OR Apache-2.0

use std::process::{Command, Output};

use j2k::{
    encode_j2k_lossless, EncodeBackendPreference, J2kBlockCodingMode, J2kEncodeValidation,
    J2kLosslessEncodeOptions, J2kLosslessSamples,
};
use j2k_test_support::{patterned_gray8, wrap_jp2_codestream};

#[test]
fn roi_batch_compare_binary_exposes_grok_wsi_surfaces() {
    let source = include_str!("../src/bin/jp2k_roi_batch_compare.rs");

    for expected in [
        "htj2k_raw_rgb8_512_roi256_q4_repeated_batch16",
        "htj2k_jph_rgb8_512_roi256_q4_repeated_batch16",
        "htj2k_jph_rgb8_256_roi128_q4_repeated_batch16",
        "j2k",
        "grok",
    ] {
        assert!(
            source.contains(expected),
            "ROI batch compare binary is missing `{expected}`"
        );
    }
}

#[test]
fn fixture_compare_binary_exposes_fair_fixture_matrix() {
    let source = include_str!("../src/bin/jp2k_fixture_compare.rs");

    for expected in [
        "J2K_FIXTURE_COMPARE_REPEATS",
        "J2K_FIXTURE_COMPARE_MODE",
        "classic_jp2_rgb8_128_roi64_q4",
        "htj2k_jph_rgb8_512_roi256_q4",
        "J2K_FIXTURE_COMPARE_BATCH_SIZES",
        "J2K_FIXTURE_COMPARE_CASE_BATCH_SIZES",
        "J2K_FIXTURE_COMPARE_MIXED_BATCH_SIZES",
        "J2K_FIXTURE_COMPARE_INPUT_DIRS",
        "J2K_FIXTURE_COMPARE_INPUT_DIR",
        "J2K_FIXTURE_COMPARE_INCLUDE_GENERATED",
        "J2K_FIXTURE_COMPARE_MANIFEST",
        "J2K_INCLUDE_OPENJPH",
        "J2K_REQUIRE_OPENJPH",
        "J2K_OPENJPH_EXPAND_BIN",
        "J2K_INCLUDE_KAKADU",
        "J2K_REQUIRE_KAKADU",
        "J2K_KDU_EXPAND_BIN",
        "correctness_preflight",
        "benchmark_complete",
        "validate_cases",
        "corpus_category",
        "input_fnv1a64",
        "source_fnv1a64",
        "openjpeg_version",
        "grok_version",
        "openjph_included",
        "openjph_available",
        "openjph_expand_command",
        "openjph_version",
        "kakadu_included",
        "kakadu_available",
        "kakadu_expand_command",
        "kakadu_version",
        "generated_case_count",
        "external_case_count",
        "case_batch_sizes",
        "mixed_batch_sizes",
        "external_native_case_count",
        "external_materialized_case_count",
        "external_unique_input_count",
        "external_native_unique_input_count",
        "cargo-xtask-adoption-materialize",
        "mixed_external_batch_group_count",
        "mixed_external_min_distinct_inputs",
        "mixed_external_max_distinct_inputs",
        "mixed_external_group_distinct_inputs",
        "mixed-input-count-below",
        "required_comparators",
        "matched_comparators",
        "skipped_comparators",
        "publication_gate_skipped_comparators",
        "publication_eligible",
        "benchmark_mode",
        "decode_method",
        "portable-native",
        "portable-emulated",
        "capability",
        "mode_excluded_case_count",
        "mode_excluded_cases",
        "sample_order_policy",
        "interleaved-rotating-decoder-order",
        "batch_input_policy",
        "rotating-owned-copies-built-outside-timed-loop",
        "mixed_external_batch_policy",
        "group-external-cases-by-format-operation-cycle-distinct-inputs",
        "batch_input_copy_counts_by_batch",
        "publication_blockers",
        "default-mixed-batch-sizes-missing",
        "generated-fixtures-included",
        "decoded_mib_per_second_median",
        "mixed_decode_method_label",
        "mixed-methods:",
        "fixture_manifest",
        "manifest_status",
        "covered-unpinned",
        "external_manifest_covered_case_count",
        "external_manifest_missing_case_count",
        "j2k_inner_parallelism_by_batch",
        "external_decoder_internal_threads",
        "execution_policy",
        "collect_j2k_paths",
        "codec_from_bytes",
        "build_profile",
        "debug_assertions",
        "git_revision",
        "git_dirty",
        "jph",
        "jhc",
        "external-license-status-not-publishable",
        "native-mixed-external-batch",
        "openjph-cli-process-output-pnm",
        "kakadu-cli-process-output-pnm",
        "openjph-htj2k-only",
        "openjpeg-external-gray-roi-scaled-noncomparable",
        "openjph-jph-compatible-stream-required",
        "openjph-roi-unsupported",
        "openjph-htj2k-full-scaled-only",
        "kakadu-roi-unsupported",
        "kakadu-full-scaled-only",
    ] {
        assert!(
            source.contains(expected),
            "fixture compare binary is missing `{expected}`"
        );
    }
}

#[test]
fn encode_compare_binary_exposes_fair_encoder_matrix() {
    let source = include_str!("../src/bin/jp2k_encode_compare.rs");

    for expected in [
        "J2K_ENCODE_COMPARE_REPEATS",
        "J2K_ENCODE_COMPARE_BATCH_SIZES",
        "J2K_ENCODE_COMPARE_CASE_BATCH_SIZES",
        "J2K_ENCODE_COMPARE_MIXED_BATCH_SIZES",
        "J2K_ENCODE_COMPARE_INPUT_DIRS",
        "J2K_ENCODE_COMPARE_INCLUDE_GENERATED",
        "J2K_ENCODE_COMPARE_MANIFEST",
        "J2K_ENCODE_COMPARE_ENCODERS",
        "J2K_REQUIRE_OPENJPEG",
        "J2K_REQUIRE_GROK",
        "J2K_INCLUDE_KAKADU",
        "J2K_REQUIRE_KAKADU",
        "J2K_KDU_COMPRESS_BIN",
        "classic-lossless-cli",
        "pnm-input-cli-process-output-jp2",
        "encode_profile",
        "classic-lossless-jp2-single-tile-lrcp-rct53-3resolutions-64x64-codeblocks-no-precinct-overrides-no-sop-eph",
        "image-crate-decode-to-pnm",
        "collect_source_image_paths",
        "selected_encoders",
        "interleaved-rotating-encoder-order",
        "host_hardware",
        "build_profile",
        "debug_assertions",
        "git_revision",
        "git_dirty",
        "openjpeg_version",
        "openjpeg_linked_library_version",
        "grok_version",
        "grok_linked_library_version",
        "kakadu_included",
        "kakadu_compress_available",
        "kakadu_compress_command",
        "kakadu_version",
        "encode_manifest",
        "external_manifest_covered_case_count",
        "external_manifest_missing_case_count",
        "external_component_group_count",
        "external_dimension_count",
        "external_source_format_count",
        "mixed_external_min_distinct_inputs",
        "mixed_external_group_distinct_inputs",
        "publication_eligible",
        "publication_blockers",
        "default-mixed-batch-sizes-missing",
        "mixed-input-count-below",
        "generated-fixtures-included",
        "input_mib_per_second_median",
        "validate_encoded_profile",
        "jp2_codestream_payload",
        "cod_profile",
        "openjpeg-compress-version-unavailable",
        "grok-compress-version-unavailable",
        "external-gray8-source-missing",
        "external-rgb8-source-missing",
        "external-dimension-diversity-below",
        "external-source-format-diversity-below",
        "encoder-filter-present",
        "external-manifest-coverage-missing",
        "covered-unpinned",
        "external-license-status-not-publishable",
        "external-workload-corpus-missing",
        "OPJ_NUM_THREADS=1 opj_compress -i INPUT.pnm -o OUTPUT.jp2 -n 3 -b 64,64 -p LRCP -threads 1",
        "grk_compress -i INPUT.pnm -o OUTPUT.jp2 -n 3 -b 64,64 -p LRCP -H 1",
        "kdu_compress -i INPUT.pnm -o OUTPUT.jp2 Creversible=yes Clevels=2 Cblk={64,64} Corder=LRCP -rate -",
        "--encode-one",
        "benchmark_complete",
    ] {
        assert!(
            source.contains(expected),
            "encode compare binary is missing `{expected}`"
        );
    }
}

#[test]
fn encode_compare_j2k_smoke_emits_completion_and_publishability_metadata() {
    let output = encode_compare_command()
        .arg("generated_gray8_128")
        .output()
        .expect("run encode compare");

    assert_success(&output);
    let stdout = String::from_utf8(output.stdout).expect("stdout is utf8");
    assert_encode_table_rows_match_header(&stdout);
    assert!(stdout.contains("benchmark_mode\tclassic-lossless-cli"));
    assert!(stdout.contains("encode_method\tpnm-input-cli-process-output-jp2"));
    assert!(stdout.contains("selected_encoders\tj2k"));
    assert!(stdout.contains("build_profile\tdebug"));
    assert!(stdout.contains("debug_assertions\ttrue"));
    assert!(stdout.contains("git_revision\t"));
    assert!(stdout.contains("git_dirty\t"));
    assert!(stdout.contains("encode_manifest\tnot set"));
    assert!(stdout.contains("generated_case_count\t1"));
    assert!(stdout.contains("external_case_count\t0"));
    assert!(stdout.contains("external_manifest_covered_case_count\t0"));
    assert!(stdout.contains("external_manifest_missing_case_count\t0"));
    assert!(stdout.contains("external_component_group_count\t0"));
    assert!(stdout.contains("external_dimension_count\t0"));
    assert!(stdout.contains("external_source_format_count\t0"));
    assert!(stdout.contains("kakadu_included\tfalse"));
    assert!(stdout.contains("kakadu_compress_available\t"));
    assert!(stdout.contains("kakadu_compress_command\t"));
    assert!(stdout.contains("kakadu_version\t"));
    assert!(stdout.contains("publication_eligible\tfalse"));
    assert!(stdout.contains("debug-build"));
    assert!(stdout.contains("encoder-filter-present"));
    assert!(stdout.contains("openjpeg-not-selected"));
    assert!(stdout.contains("grok-not-selected"));
    assert!(stdout.contains("external-unique-input-count-below-24"));
    assert!(stdout.contains("external-gray8-source-missing"));
    assert!(stdout.contains("external-rgb8-source-missing"));
    assert!(stdout.contains("external-dimension-diversity-below-3"));
    assert!(stdout.contains("external-source-format-diversity-below-2"));
    assert!(stdout.contains("encoder\tcase\tbenchmark_mode\tencode_method\tinput_source"));
    assert!(stdout.contains(
        "j2k\tgenerated_gray8_128\tclassic-lossless-cli\tpnm-input-cli-process-output-jp2\tj2k-generated-image\tgenerated-dev\tj2k-generated-encode-matrix\trepo-generated\tj2k-test-support-pattern\tgenerated\tj2k\tjp2\tgray8"
    ));
    assert!(stdout.ends_with("benchmark_complete\ttrue\n"));
}

#[test]
fn encode_compare_manifest_overrides_external_source_metadata() {
    let fixture_root = std::env::current_dir()
        .expect("current dir")
        .join("target")
        .join("j2k-fixture-bench")
        .join(format!("encode-manifest-dir-test-{}", std::process::id()));
    std::fs::create_dir_all(&fixture_root).expect("create encode fixture dir");
    let image_path = fixture_root.join("kodak_like.png");
    write_gray_png(&image_path, 64, 64);
    let image_hash = fnv1a64_hex(&patterned_gray8(64, 64));
    let manifest_path = fixture_root.join("encode-fixtures.tsv");
    std::fs::write(
        &manifest_path,
        format!(
            "path\tcorpus_category\tcorpus_name\tlicense_status\tsource_command\tinput_fnv1a64\n{}\tnatural-image\tmanifest-kodak-subset\tcc0\tconverted-to-pgm\t{image_hash}\n",
            image_path.display(),
        ),
    )
    .expect("write encode manifest");

    let output = encode_compare_command()
        .env("J2K_ENCODE_COMPARE_INPUT_DIRS", &fixture_root)
        .env("J2K_ENCODE_COMPARE_INCLUDE_GENERATED", "0")
        .env("J2K_ENCODE_COMPARE_MANIFEST", &manifest_path)
        .output()
        .expect("run encode compare");

    assert_success(&output);
    let stdout = String::from_utf8(output.stdout).expect("stdout is utf8");
    assert_encode_table_rows_match_header(&stdout);
    assert!(stdout.contains("generated_case_count\t0"));
    assert!(stdout.contains("external_case_count\t1"));
    assert!(stdout.contains("external_manifest_covered_case_count\t1"));
    assert!(stdout.contains("external_manifest_missing_case_count\t0"));
    assert!(stdout.contains("external_unique_input_count\t1"));
    assert!(stdout.contains("external_component_group_count\t1"));
    assert!(stdout.contains("external_dimension_count\t1"));
    assert!(stdout.contains("external_source_format_count\t1"));
    assert!(stdout.contains(
        "\tnatural-image\tmanifest-kodak-subset\tcc0\tconverted-to-pgm\tcovered\tj2k\tjp2\tgray8\t64x64"
    ));
    assert!(stdout.ends_with("benchmark_complete\ttrue\n"));
}

#[test]
fn encode_compare_emits_split_mixed_external_batches() {
    let fixture_root = std::env::current_dir()
        .expect("current dir")
        .join("target")
        .join("j2k-fixture-bench")
        .join(format!("encode-mixed-dir-test-{}", std::process::id()));
    std::fs::create_dir_all(&fixture_root).expect("create encode fixture dir");
    write_gray_png(&fixture_root.join("mixed_a.png"), 128, 128);
    write_gray_png(&fixture_root.join("mixed_b.png"), 160, 128);

    let output = encode_compare_command()
        .env("J2K_ENCODE_COMPARE_INPUT_DIRS", &fixture_root)
        .env("J2K_ENCODE_COMPARE_INCLUDE_GENERATED", "0")
        .env("J2K_ENCODE_COMPARE_CASE_BATCH_SIZES", "1")
        .env("J2K_ENCODE_COMPARE_MIXED_BATCH_SIZES", "2")
        .output()
        .expect("run encode compare");

    assert_success(&output);
    let stdout = String::from_utf8(output.stdout).expect("stdout is utf8");
    assert_encode_table_rows_match_header(&stdout);
    assert!(stdout.contains("generated_case_count\t0"));
    assert!(stdout.contains("case_batch_sizes\t1"));
    assert!(stdout.contains("mixed_batch_sizes\t2"));
    assert!(stdout.contains("external_case_count\t2"));
    assert!(stdout.contains("external_unique_input_count\t2"));
    assert!(stdout.contains("mixed_external_batch_group_count\t1"));
    assert!(stdout.contains("mixed_external_min_distinct_inputs\t2"));
    assert!(stdout.contains("mixed_external_max_distinct_inputs\t2"));
    assert!(stdout.contains("mixed_external_group_distinct_inputs\texternal_mixed_gray8_encode:2"));
    assert!(stdout.contains("external_mixed_gray8_encode"));
    assert!(stdout
        .contains("\tclassic-lossless-cli\tpnm-input-cli-process-output-jp2\texternal:mixed\t"));
    assert!(stdout.ends_with("benchmark_complete\ttrue\n"));
}

#[test]
fn fixture_compare_success_emits_completion_and_publishability_metadata() {
    let output = fixture_compare_command()
        .arg("htj2k_raw_gray8_128_full")
        .output()
        .expect("run fixture compare");

    assert_success(&output);
    let stdout = String::from_utf8(output.stdout).expect("stdout is utf8");
    assert!(stdout
        .contains("correctness_preflight\tnon-skipped-comparators-match-j2k-baseline-all-batches"));
    assert_table_rows_match_header(&stdout);
    assert!(stdout.contains("corpus_category"));
    assert!(stdout.contains("benchmark_mode\tportable-native"));
    assert!(stdout.contains("comparable_scope\tnative-operations-only"));
    assert!(stdout.contains("mode_excluded_case_count\t0"));
    assert!(stdout.contains("mode_excluded_cases\tnone"));
    assert!(stdout.contains("sample_order_policy\tinterleaved-rotating-decoder-order"));
    assert!(stdout.contains("batch_input_policy\trotating-owned-copies-built-outside-timed-loop"));
    assert!(stdout.contains("batch_input_copy_counts_by_batch\t1:1"));
    assert!(stdout.contains("decoder\tcase\tbenchmark_mode\tdecode_method\tinput_source"));
    assert!(stdout.contains("generated_case_count\t1"));
    assert!(stdout.contains("external_case_count\t0"));
    assert!(stdout.contains("external_unique_input_count\t0"));
    assert!(stdout.contains("mixed_external_batch_group_count\t0"));
    assert!(stdout.contains("openjph_included\tfalse"));
    assert!(stdout.contains("openjph_available\t"));
    assert!(stdout.contains("openjph_expand_command\t"));
    assert!(stdout.contains("openjph_version\t"));
    assert!(stdout.contains("kakadu_included\tfalse"));
    assert!(stdout.contains("kakadu_available\t"));
    assert!(stdout.contains("kakadu_expand_command\t"));
    assert!(stdout.contains("kakadu_version\t"));
    assert!(stdout.contains("publication_eligible\tfalse"));
    assert!(stdout.contains("publication_blockers\t"));
    assert!(stdout.contains("debug-build"));
    assert!(stdout.contains("case-filters-present"));
    assert!(stdout.contains("external-case-count-below-24"));
    assert!(stdout.contains("external-unique-input-count-below-24"));
    assert!(stdout.contains("mixed-external-batches-missing"));
    assert!(stdout.contains("j2k_inner_parallelism_by_batch\t1:serial"));
    assert!(stdout.contains("external_decoder_internal_threads\t1"));
    assert!(stdout.ends_with("benchmark_complete\ttrue\n"));
}

#[test]
fn fixture_compare_kakadu_opt_in_reports_cli_context_rows_or_skip() {
    let output = fixture_compare_command()
        .env("J2K_INCLUDE_KAKADU", "1")
        .arg("classic_raw_gray8_128_full")
        .output()
        .expect("run fixture compare");

    assert_success(&output);
    let stdout = String::from_utf8(output.stdout).expect("stdout is utf8");
    assert_table_rows_match_header(&stdout);
    assert!(stdout.contains("kakadu_included\ttrue"));
    assert!(stdout.contains("kakadu_available\t"));
    assert!(stdout.contains("kakadu_expand_command\t"));
    assert!(stdout.contains("kakadu_version\t"));
    assert!(!metadata_value(&stdout, "publication_gate_skipped_comparators").contains("kakadu"));
    if metadata_value(&stdout, "publication_gate_skipped_comparators") == "none" {
        assert!(!metadata_value(&stdout, "publication_blockers")
            .contains("skipped-comparators-present"));
    }
    if stdout.contains("kakadu_available\tfalse") {
        assert!(stdout.contains("kakadu:kakadu-unavailable"));
        assert!(stdout.contains(
            "kakadu\tclassic_raw_gray8_128_full\tportable-native\tskipped\tj2k-generated"
        ));
    } else {
        assert!(stdout.contains(
            "kakadu\tclassic_raw_gray8_128_full\tportable-native\tkakadu-cli-process-output-pnm\tj2k-generated"
        ));
    }
    assert!(stdout.ends_with("benchmark_complete\ttrue\n"));
}

#[test]
fn encode_compare_kakadu_opt_in_skip_does_not_block_default_publication_gate() {
    let output = encode_compare_command()
        .env_remove("J2K_ENCODE_COMPARE_ENCODERS")
        .env("J2K_INCLUDE_KAKADU", "1")
        .arg("generated_gray8_128")
        .output()
        .expect("run encode compare");

    assert_success(&output);
    let stdout = String::from_utf8(output.stdout).expect("stdout is utf8");
    assert_encode_table_rows_match_header(&stdout);
    assert!(stdout.contains("kakadu_included\ttrue"));
    if stdout.contains("kakadu_compress_available\tfalse") {
        assert!(stdout.contains("kakadu\tgenerated_gray8_128\tclassic-lossless-cli\tskipped"));
        assert!(stdout.contains("encoder-tool-unavailable"));
    }
    assert!(
        !metadata_value(&stdout, "publication_blockers").contains("kakadu-compress-unavailable")
    );
    assert!(stdout.ends_with("benchmark_complete\ttrue\n"));
}

#[test]
fn fixture_compare_default_portable_native_excludes_openjpeg_noncomparable_rows() {
    let output = fixture_compare_command()
        .arg("htj2k_jph_rgb8")
        .output()
        .expect("run fixture compare");

    assert_success(&output);
    let stdout = String::from_utf8(output.stdout).expect("stdout is utf8");
    assert_table_rows_match_header(&stdout);
    assert!(stdout.contains("benchmark_mode\tportable-native"));
    assert!(stdout.contains("selected_cases\t1"));
    assert!(stdout.contains("mode_excluded_case_count\t2"));
    assert!(stdout
        .contains("mode_excluded_cases\thtj2k_jph_rgb8_128_roi64_q4,htj2k_jph_rgb8_512_roi256_q4"));
    assert!(stdout.contains("skipped_comparators\tnone"));
    assert!(!stdout.contains("htj2k_jph_rgb8_128_roi64_q4\t"));
    assert!(!stdout.contains("htj2k_jph_rgb8_512_roi256_q4\t"));
    assert!(stdout.ends_with("benchmark_complete\ttrue\n"));
}

#[test]
fn fixture_compare_capability_marks_openjpeg_htj2k_roi_scaled_noncomparable() {
    let output = fixture_compare_command()
        .env("J2K_FIXTURE_COMPARE_MODE", "capability")
        .arg("htj2k_jph_rgb8_128_roi64_q4")
        .output()
        .expect("run fixture compare");

    assert_success(&output);
    let stdout = String::from_utf8(output.stdout).expect("stdout is utf8");
    assert_table_rows_match_header(&stdout);
    assert!(
        stdout.contains("skipped_comparators\topenjpeg:openjpeg-htj2k-roi-scaled-noncomparable")
    );
    assert!(stdout.contains("publication_eligible\tfalse"));
    assert!(stdout.contains(
        "openjpeg\thtj2k_jph_rgb8_128_roi64_q4\tcapability\tskipped\tj2k-generated-jph-wrapper\tgenerated-dev\tj2k-generated-fixture-matrix\trepo-generated\tj2k-lossless-cpu-roundtrip\tgenerated\thtj2k\tjph\troi-scaled"
    ));
    assert!(stdout.contains("openjpeg-htj2k-roi-scaled-noncomparable"));
    assert!(stdout.contains(
        "grok\thtj2k_jph_rgb8_128_roi64_q4\tcapability\tnative\tj2k-generated-jph-wrapper\tgenerated-dev\tj2k-generated-fixture-matrix\trepo-generated\tj2k-lossless-cpu-roundtrip\tgenerated\thtj2k\tjph\troi-scaled"
    ));
    assert!(stdout.ends_with("benchmark_complete\ttrue\n"));
}

#[test]
fn fixture_compare_portable_emulated_labels_openjpeg_task_equivalent_decode() {
    let output = fixture_compare_command()
        .env("J2K_FIXTURE_COMPARE_MODE", "portable-emulated")
        .arg("htj2k_jph_rgb8_128_roi64_q4")
        .output()
        .expect("run fixture compare");

    assert_success(&output);
    let stdout = String::from_utf8(output.stdout).expect("stdout is utf8");
    assert_table_rows_match_header(&stdout);
    assert!(stdout.contains("benchmark_mode\tportable-emulated"));
    assert!(stdout.contains("comparable_scope\ttask-equivalent-with-method-labels"));
    assert!(stdout.contains("mode_excluded_case_count\t0"));
    assert!(stdout.contains("skipped_comparators\tnone"));
    assert!(stdout.contains(
        "openjpeg\thtj2k_jph_rgb8_128_roi64_q4\tportable-emulated\temulated-full-scaled-crop\tj2k-generated-jph-wrapper\tgenerated-dev\tj2k-generated-fixture-matrix\trepo-generated\tj2k-lossless-cpu-roundtrip\tgenerated\thtj2k\tjph\troi-scaled"
    ));
    assert!(stdout.contains(
        "grok\thtj2k_jph_rgb8_128_roi64_q4\tportable-emulated\tnative\tj2k-generated-jph-wrapper\tgenerated-dev\tj2k-generated-fixture-matrix\trepo-generated\tj2k-lossless-cpu-roundtrip\tgenerated\thtj2k\tjph\troi-scaled"
    ));
    assert!(stdout.ends_with("benchmark_complete\ttrue\n"));
}

#[test]
fn fixture_compare_empty_external_dir_fails_before_stdout() {
    let empty_dir = std::env::current_dir()
        .expect("current dir")
        .join("target")
        .join("j2k-fixture-bench")
        .join("empty-fixture-dir-test");
    std::fs::create_dir_all(&empty_dir).expect("create empty fixture dir");

    let output = fixture_compare_command()
        .env("J2K_FIXTURE_COMPARE_INPUT_DIRS", &empty_dir)
        .arg("htj2k_raw_gray8_128_full")
        .output()
        .expect("run fixture compare");

    assert!(!output.status.success(), "empty external dir should fail");
    assert!(
        output.stdout.is_empty(),
        "failed run should not emit benchmark stdout"
    );
    let stderr = String::from_utf8(output.stderr).expect("stderr is utf8");
    assert!(
        stderr.contains("contains no .j2k/.j2c/.jp2/.jph/.jhc fixtures"),
        "unexpected stderr: {stderr}"
    );
}

#[test]
fn fixture_compare_recurses_external_dir_and_labels_htj2k_codec() {
    let fixture_root = std::env::current_dir()
        .expect("current dir")
        .join("target")
        .join("j2k-fixture-bench")
        .join("external-fixture-dir-test");
    let nested_dir = fixture_root.join("nested");
    std::fs::create_dir_all(&nested_dir).expect("create nested fixture dir");
    std::fs::write(
        nested_dir.join("ht_fixture.j2k"),
        htj2k_gray_fixture(64, 64),
    )
    .expect("write external fixture");

    let output = fixture_compare_command()
        .env("J2K_FIXTURE_COMPARE_INPUT_DIRS", &fixture_root)
        .arg("external_ht_fixture_full")
        .output()
        .expect("run fixture compare");

    assert_success(&output);
    let stdout = String::from_utf8(output.stdout).expect("stdout is utf8");
    assert_table_rows_match_header(&stdout);
    assert!(stdout.contains("generated_case_count\t0"));
    assert!(stdout.contains("external_case_count\t1"));
    assert!(stdout.contains("external_unique_input_count\t1"));
    assert!(stdout.contains("external_ht_fixture_full"));
    assert!(
        stdout.contains("\thtj2k\traw-codestream\t"),
        "external fixture was not labeled as HTJ2K: {stdout}"
    );
    assert!(stdout.ends_with("benchmark_complete\ttrue\n"));
}

#[test]
fn fixture_compare_manifest_overrides_external_corpus_metadata() {
    let fixture_root = std::env::current_dir()
        .expect("current dir")
        .join("target")
        .join("j2k-fixture-bench")
        .join("manifest-fixture-dir-test");
    std::fs::create_dir_all(&fixture_root).expect("create fixture dir");
    let fixture_path = fixture_root.join("manifest_fixture.j2k");
    let fixture = htj2k_gray_fixture(64, 64);
    let fixture_hash = fnv1a64_hex(&fixture);
    std::fs::write(&fixture_path, fixture).expect("write external fixture");
    let manifest_path = fixture_root.join("fixtures.tsv");
    std::fs::write(
        &manifest_path,
        format!(
            "path\tcorpus_category\tcorpus_name\tlicense_status\tencode_command\tinput_fnv1a64\tcodec\tcontainer\n{}\tinterop\tharness-manifest\tpermissive-test-fixture\tsource-native\t{fixture_hash}\thtj2k\traw-codestream\n",
            fixture_path.display(),
        ),
    )
    .expect("write fixture manifest");

    let output = fixture_compare_command()
        .env("J2K_FIXTURE_COMPARE_INPUT_DIRS", &fixture_root)
        .env("J2K_FIXTURE_COMPARE_MANIFEST", &manifest_path)
        .arg("external_manifest_fixture_full")
        .output()
        .expect("run fixture compare");

    assert_success(&output);
    let stdout = String::from_utf8(output.stdout).expect("stdout is utf8");
    assert_table_rows_match_header(&stdout);
    assert!(stdout.contains("fixture_manifest\t"));
    assert!(stdout.contains("external_manifest_covered_case_count\t1"));
    assert!(stdout.contains("external_manifest_missing_case_count\t0"));
    assert!(stdout.contains(
        "\tinterop\tharness-manifest\tpermissive-test-fixture\tsource-native\tcovered\thtj2k\traw-codestream\t"
    ));
    assert!(stdout.ends_with("benchmark_complete\ttrue\n"));
}

#[test]
fn fixture_compare_source_hash_prevents_container_variants_inflating_unique_inputs() {
    let fixture_root = std::env::current_dir()
        .expect("current dir")
        .join("target")
        .join("j2k-fixture-bench")
        .join(format!("source-hash-container-test-{}", std::process::id()));
    std::fs::create_dir_all(&fixture_root).expect("create fixture dir");
    let raw_path = fixture_root.join("source_raw.j2k");
    let jp2_path = fixture_root.join("source_boxed.jp2");
    let raw_fixture = classic_gray_fixture(128, 128);
    let jp2_fixture = wrap_jp2_codestream(&raw_fixture, 128, 128, 1, 8, 17);
    let raw_hash = fnv1a64_hex(&raw_fixture);
    let jp2_hash = fnv1a64_hex(&jp2_fixture);
    let source_hash = fnv1a64_hex(&patterned_gray8(128, 128));
    std::fs::write(&raw_path, raw_fixture).expect("write raw fixture");
    std::fs::write(&jp2_path, jp2_fixture).expect("write jp2 fixture");
    let manifest_path = fixture_root.join("fixtures.tsv");
    std::fs::write(
        &manifest_path,
        format!(
            "path\tcorpus_category\tcorpus_name\tlicense_status\tencode_command\tinput_fnv1a64\tsource_fnv1a64\tcodec\tcontainer\n{}\tnatural-image\tcontainer-pair\tcc0\tsource-image\t{raw_hash}\t{source_hash}\tj2k\traw-codestream\n{}\tnatural-image\tcontainer-pair\tcc0\tsource-image\t{jp2_hash}\t{source_hash}\tj2k\tjp2\n",
            raw_path.display(),
            jp2_path.display(),
        ),
    )
    .expect("write fixture manifest");

    let output = fixture_compare_command()
        .env("J2K_FIXTURE_COMPARE_INPUT_DIRS", &fixture_root)
        .env("J2K_FIXTURE_COMPARE_MANIFEST", &manifest_path)
        .env("J2K_FIXTURE_COMPARE_INCLUDE_GENERATED", "0")
        .env("J2K_FIXTURE_COMPARE_BATCH_SIZES", "1")
        .arg("external_source_")
        .output()
        .expect("run fixture compare");

    assert_success(&output);
    let stdout = String::from_utf8(output.stdout).expect("stdout is utf8");
    assert_table_rows_match_header(&stdout);
    assert!(stdout.contains("external_case_count\t2"));
    assert!(stdout.contains("external_unique_input_count\t1"));
    assert!(stdout.contains("\traw-codestream\t"));
    assert!(stdout.contains("\tjp2\t"));
    assert!(stdout.contains(&format!("\t{source_hash}\t")));
}

#[test]
fn fixture_compare_emits_mixed_external_batches_for_distinct_inputs() {
    let fixture_root = std::env::current_dir()
        .expect("current dir")
        .join("target")
        .join("j2k-fixture-bench")
        .join(format!("mixed-fixture-dir-test-{}", std::process::id()));
    std::fs::create_dir_all(&fixture_root).expect("create fixture dir");
    std::fs::write(
        fixture_root.join("mixed_a.j2k"),
        classic_gray_fixture(128, 128),
    )
    .expect("write first external fixture");
    std::fs::write(
        fixture_root.join("mixed_b.j2k"),
        classic_gray_fixture(160, 128),
    )
    .expect("write second external fixture");

    let output = fixture_compare_command()
        .env("J2K_FIXTURE_COMPARE_INPUT_DIRS", &fixture_root)
        .env("J2K_FIXTURE_COMPARE_INCLUDE_GENERATED", "0")
        .env("J2K_FIXTURE_COMPARE_CASE_BATCH_SIZES", "1")
        .env("J2K_FIXTURE_COMPARE_MIXED_BATCH_SIZES", "2")
        .output()
        .expect("run fixture compare");

    assert_success(&output);
    let stdout = String::from_utf8(output.stdout).expect("stdout is utf8");
    assert_table_rows_match_header(&stdout);
    assert!(stdout.contains("generated_case_count\t0"));
    assert!(stdout.contains("case_batch_sizes\t1"));
    assert!(stdout.contains("mixed_batch_sizes\t2"));
    assert!(stdout.contains("external_case_count\t2"));
    assert!(stdout.contains("external_unique_input_count\t2"));
    assert!(stdout.contains("mixed_external_batch_group_count\t1"));
    assert!(stdout.contains("mixed_external_min_distinct_inputs\t2"));
    assert!(stdout.contains("mixed_external_max_distinct_inputs\t2"));
    assert!(stdout.contains("mixed_external_group_distinct_inputs\texternal_mixed_gray8_full:2"));
    assert!(stdout.contains("external_mixed_gray8_full"));
    assert!(stdout.contains("\tnative-mixed-external-batch\texternal:mixed\t"));
    assert!(stdout.contains("\t2\t1\t"));
    assert!(stdout.ends_with("benchmark_complete\ttrue\n"));
}

#[test]
fn fixture_compare_labels_jhc_container_from_extension() {
    let fixture_root = std::env::current_dir()
        .expect("current dir")
        .join("target")
        .join("j2k-fixture-bench")
        .join(format!("jhc-fixture-dir-test-{}", std::process::id()));
    std::fs::create_dir_all(&fixture_root).expect("create fixture dir");
    std::fs::write(
        fixture_root.join("ht_fixture.jhc"),
        htj2k_gray_fixture(64, 64),
    )
    .expect("write external fixture");

    let output = fixture_compare_command()
        .env("J2K_FIXTURE_COMPARE_INPUT_DIRS", &fixture_root)
        .arg("external_ht_fixture_full")
        .output()
        .expect("run fixture compare");

    assert_success(&output);
    let stdout = String::from_utf8(output.stdout).expect("stdout is utf8");
    assert_table_rows_match_header(&stdout);
    assert!(
        stdout.contains("\thtj2k\tjhc\t"),
        "external fixture was not labeled as JHC: {stdout}"
    );
    assert!(stdout.ends_with("benchmark_complete\ttrue\n"));
}

fn fixture_compare_command() -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_jp2k_fixture_compare"));
    command
        .env_remove("J2K_REQUIRE_OPENJPEG")
        .env_remove("J2K_REQUIRE_GROK")
        .env_remove("J2K_REQUIRE_OPENJPH")
        .env_remove("J2K_INCLUDE_OPENJPH")
        .env_remove("J2K_OPENJPH_EXPAND_BIN")
        .env_remove("J2K_REQUIRE_KAKADU")
        .env_remove("J2K_INCLUDE_KAKADU")
        .env_remove("J2K_KDU_EXPAND_BIN")
        .env_remove("J2K_FIXTURE_COMPARE_MODE")
        .env_remove("J2K_FIXTURE_COMPARE_THREADS")
        .env_remove("J2K_FIXTURE_COMPARE_BATCH_SIZE")
        .env_remove("J2K_FIXTURE_COMPARE_BATCH_SIZES")
        .env_remove("J2K_FIXTURE_COMPARE_INPUT_DIRS")
        .env_remove("J2K_FIXTURE_COMPARE_INPUT_DIR")
        .env_remove("J2K_FIXTURE_COMPARE_MANIFEST")
        .env_remove("J2K_FIXTURE_COMPARE_INCLUDE_GENERATED")
        .env("J2K_FIXTURE_COMPARE_REPEATS", "1")
        .env("J2K_FIXTURE_COMPARE_CASE_BATCH_SIZES", "1")
        .env("J2K_FIXTURE_COMPARE_MIXED_BATCH_SIZES", "1");
    command
}

fn encode_compare_command() -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_jp2k_encode_compare"));
    command
        .env_remove("J2K_REQUIRE_OPENJPEG")
        .env_remove("J2K_REQUIRE_GROK")
        .env_remove("J2K_REQUIRE_KAKADU")
        .env_remove("J2K_INCLUDE_KAKADU")
        .env_remove("J2K_KDU_COMPRESS_BIN")
        .env_remove("J2K_ENCODE_COMPARE_BATCH_SIZES")
        .env_remove("J2K_ENCODE_COMPARE_INPUT_DIRS")
        .env_remove("J2K_ENCODE_COMPARE_MANIFEST")
        .env_remove("J2K_ENCODE_COMPARE_INCLUDE_GENERATED")
        .env("J2K_ENCODE_COMPARE_ENCODERS", "j2k")
        .env("J2K_ENCODE_COMPARE_REPEATS", "1")
        .env("J2K_ENCODE_COMPARE_CASE_BATCH_SIZES", "1")
        .env("J2K_ENCODE_COMPARE_MIXED_BATCH_SIZES", "1");
    command
}

fn assert_success(output: &Output) {
    assert!(
        output.status.success(),
        "fixture compare failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

fn assert_table_rows_match_header(stdout: &str) {
    let mut lines = stdout.lines();
    let header = lines
        .by_ref()
        .find(|line| line.starts_with("decoder\tcase\t"))
        .expect("fixture compare header");
    let columns = header.split('\t').count();
    for line in lines.take_while(|line| *line != "benchmark_complete\ttrue") {
        assert_eq!(
            line.split('\t').count(),
            columns,
            "row column count differs from header: {line}"
        );
    }
}

fn assert_encode_table_rows_match_header(stdout: &str) {
    let mut lines = stdout.lines();
    let header = lines
        .by_ref()
        .find(|line| line.starts_with("encoder\tcase\t"))
        .expect("encode compare header");
    let columns = header.split('\t').count();
    for line in lines.take_while(|line| *line != "benchmark_complete\ttrue") {
        assert_eq!(
            line.split('\t').count(),
            columns,
            "row column count differs from header: {line}"
        );
    }
}

fn metadata_value<'a>(stdout: &'a str, key: &str) -> &'a str {
    stdout
        .lines()
        .find_map(|line| line.strip_prefix(key)?.strip_prefix('\t'))
        .unwrap_or("")
}

fn write_gray_png(path: &std::path::Path, width: u32, height: u32) {
    let pixels = patterned_gray8(width, height);
    let image = image::GrayImage::from_raw(width, height, pixels).expect("gray image dimensions");
    image.save(path).expect("write PNG");
}

fn htj2k_gray_fixture(width: u32, height: u32) -> Vec<u8> {
    gray_fixture(width, height, J2kBlockCodingMode::HighThroughput)
}

fn classic_gray_fixture(width: u32, height: u32) -> Vec<u8> {
    gray_fixture(width, height, J2kBlockCodingMode::Classic)
}

fn fnv1a64_hex(bytes: &[u8]) -> String {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{hash:016x}")
}

fn gray_fixture(width: u32, height: u32, block_coding_mode: J2kBlockCodingMode) -> Vec<u8> {
    let pixels = patterned_gray8(width, height);
    let samples = J2kLosslessSamples::new(&pixels, width, height, 1, 8, false).expect("samples");
    let options = J2kLosslessEncodeOptions::default()
        .with_backend(EncodeBackendPreference::CpuOnly)
        .with_block_coding_mode(block_coding_mode)
        .with_max_decomposition_levels(Some(2))
        .with_validation(J2kEncodeValidation::CpuRoundTrip);
    encode_j2k_lossless(samples, &options)
        .expect("encode J2K fixture")
        .codestream
}
