// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;

use super::super::{
    run_cpu_encode_compare, run_cpu_fixture_compare, run_cpu_public_api_decode,
    run_cpu_public_api_encode, run_cuda_htj2k_decode, run_cuda_htj2k_encode,
    run_metal_decode_benchmark, run_metal_encode_auto_routing, run_metal_transcode_benchmark,
    use_test_cargo_program,
};
use super::support::{read, recording_program, temp_dir};
use crate::adoption_benchmark::options::AdoptionBenchmarkOptions;

fn options(root: &std::path::Path) -> AdoptionBenchmarkOptions {
    let decode = root.join("decode fixtures");
    let encode = root.join("encode fixtures");
    let decode_manifest = root.join("decode.tsv");
    let encode_manifest = root.join("encode.tsv");
    fs::create_dir_all(&decode).expect("create decode fixtures");
    fs::create_dir_all(&encode).expect("create encode fixtures");
    fs::write(&decode_manifest, "fixture\n").expect("write decode manifest");
    fs::write(&encode_manifest, "fixture\n").expect("write encode manifest");
    let out_dir = root.join("out");
    fs::create_dir_all(&out_dir).expect("create output directory");
    AdoptionBenchmarkOptions {
        out_dir,
        input_dirs: Some(decode.to_string_lossy().into_owned()),
        manifest: Some(decode_manifest),
        encode_input_dirs: Some(encode.to_string_lossy().into_owned()),
        encode_manifest: Some(encode_manifest),
        cuda_decode_batch_sizes: Some("3,5".to_string()),
        include_generated: false,
        quick: true,
        cuda: true,
        metal: true,
        openjph: true,
        kakadu: true,
        require_cuda: true,
        require_metal: true,
        require_openjph: true,
        require_kakadu: true,
        finalize_existing: false,
    }
}

fn output(options: &AdoptionBenchmarkOptions, name: &str) -> String {
    read(&options.out_dir.join(format!("{name}.out")))
}

#[test]
fn runner_option_branches_build_expected_hermetic_commands() {
    let root = temp_dir("runner-options");
    let program = recording_program(&root);
    let _cargo = use_test_cargo_program(program.into_os_string());
    let options = options(&root);

    run_cpu_fixture_compare(&options).expect("CPU fixture runner");
    run_cpu_encode_compare(&options).expect("CPU encode runner");
    run_cpu_public_api_encode(&options).expect("public encode runner");
    run_cpu_public_api_decode(&options).expect("public decode runner");
    run_cuda_htj2k_decode(&options).expect("CUDA decode runner");
    run_cuda_htj2k_encode(&options).expect("CUDA encode runner");
    run_metal_decode_benchmark(&options).expect("Metal decode runner");
    run_metal_encode_auto_routing(&options).expect("Metal encode runner");
    run_metal_transcode_benchmark(&options).expect("Metal transcode runner");

    let fixture = output(&options, "cpu-fixture-compare");
    assert!(fixture.contains("arg=jp2k_fixture_compare"));
    assert!(fixture.contains("J2K_FIXTURE_COMPARE_REPEATS=1"));
    assert!(fixture.contains("J2K_FIXTURE_COMPARE_INPUT_DIRS="));
    assert!(fixture.contains("decode fixtures"));
    assert!(fixture.contains("J2K_FIXTURE_COMPARE_MANIFEST="));
    assert!(fixture.contains("J2K_FIXTURE_COMPARE_INCLUDE_GENERATED=0"));
    assert!(fixture.contains("J2K_INCLUDE_OPENJPH=1"));
    assert!(fixture.contains("J2K_REQUIRE_OPENJPH=1"));
    assert!(fixture.contains("J2K_INCLUDE_KAKADU=1"));
    assert!(fixture.contains("J2K_REQUIRE_KAKADU=1"));

    let encode = output(&options, "cpu-encode-compare");
    assert!(encode.contains("arg=jp2k_encode_compare"));
    assert!(encode.contains("J2K_ENCODE_COMPARE_REPEATS=1"));
    assert!(encode.contains("J2K_ENCODE_COMPARE_INPUT_DIRS="));
    assert!(encode.contains("encode fixtures"));
    assert!(encode.contains("J2K_ENCODE_COMPARE_MANIFEST="));
    assert!(encode.contains("J2K_ENCODE_COMPARE_INCLUDE_GENERATED=0"));
    assert!(encode.contains("J2K_INCLUDE_KAKADU=1"));
    assert!(encode.contains("J2K_REQUIRE_KAKADU=1"));

    for name in ["cpu-public-api-encode", "cpu-public-api-decode"] {
        let public = output(&options, name);
        assert!(public.contains("arg=--quick"));
        assert!(public.contains("CARGO_TARGET_DIR="));
    }

    let cuda_decode = output(&options, "cuda-htj2k-decode");
    assert!(cuda_decode.contains("arg=cuda-runtime"));
    assert!(cuda_decode.contains("arg=--quick"));
    assert!(cuda_decode.contains("J2K_CUDA_DECODE_BATCH_SIZES=3,5"));
    assert!(cuda_decode.contains("J2K_CUDA_DECODE_INCLUDE_GENERATED=0"));
    assert!(cuda_decode.contains("J2K_REQUIRE_CUDA_BENCH=1"));

    let cuda_encode = output(&options, "cuda-htj2k-encode");
    assert!(cuda_encode.contains("J2K_CUDA_ENCODE_INPUT_DIRS="));
    assert!(cuda_encode.contains("J2K_CUDA_ENCODE_INCLUDE_GENERATED=0"));
    assert!(cuda_encode.contains("J2K_REQUIRE_CUDA_BENCH=1"));
    assert!(cuda_encode.contains("J2K_REQUIRE_CUDA_OXIDE_BUILD=1"));

    for name in [
        "metal-decode-benchmark",
        "metal-encode-auto-routing",
        "metal-transcode-benchmark",
    ] {
        assert!(output(&options, name).contains("J2K_REQUIRE_METAL_BENCH=1"));
    }
    assert!(
        output(&options, "metal-decode-benchmark").contains("J2K_METAL_DECODE_INCLUDE_GENERATED=0")
    );
    assert!(output(&options, "metal-encode-auto-routing")
        .contains("J2K_METAL_ENCODE_INCLUDE_GENERATED=0"));
    let transcode = output(&options, "metal-transcode-benchmark");
    assert!(transcode.contains("J2K_TRANSCODE_METAL_PROFILE_STAGES=1"));
    assert!(transcode.contains("arg=jpeg_to_htj2k_wsi_integer_53_tile_batch/"));
}

#[test]
fn non_macos_optional_metal_runners_emit_skipped_steps() {
    if std::env::consts::OS == "macos" {
        return;
    }
    let root = temp_dir("metal-skips");
    let mut options = options(&root);
    options.require_metal = false;
    let steps = [
        run_metal_decode_benchmark(&options).expect("decode skip"),
        run_metal_encode_auto_routing(&options).expect("encode skip"),
        run_metal_transcode_benchmark(&options).expect("transcode skip"),
    ];
    for step in steps {
        assert!(matches!(
            step.status,
            crate::adoption_benchmark::summary::StepStatus::Skipped { .. }
        ));
    }
}

#[test]
fn runner_paths_are_owned_without_external_process_state() {
    let root = temp_dir("runner-paths");
    let options = options(&root);
    assert!(options.out_dir.is_absolute());
    assert!(options
        .manifest
        .as_deref()
        .is_some_and(std::path::Path::is_absolute));
}
