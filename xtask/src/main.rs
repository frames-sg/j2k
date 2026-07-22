use std::env;
use std::process::ExitCode;

#[cfg(feature = "adoption")]
mod adoption_benchmark;
#[cfg(feature = "adoption")]
mod adoption_corpus;
#[cfg(feature = "adoption")]
mod adoption_curate;
#[cfg(not(feature = "adoption"))]
mod adoption_disabled;
#[cfg(feature = "adoption")]
mod adoption_manifest;
#[cfg(feature = "adoption")]
mod adoption_materialize;
#[cfg(feature = "adoption")]
mod adoption_report;
mod benchmark_commands;
mod benchmark_registry;
mod clone_audit;
mod codegen_commands;
mod command_support;
mod coverage;
mod cuda;
mod gpu_validation;
#[cfg(feature = "adoption")]
mod markdown;
mod metal;
mod panic_surface;
mod perf_guard;
mod process;
mod public_support;
#[cfg(feature = "adoption")]
mod publication_gate;
mod quality_commands;
mod release_commands;
mod release_status;
mod semver;
mod source_audit;
mod stable_api;
#[cfg(all(test, unix))]
mod test_command;

use benchmark_commands::{
    bench_build, bench_report, j2k_bench_signoff, j2k_ml_batch_bench_cuda, j2k_ml_batch_bench_metal,
};
use clone_audit::clone_audit;
use codegen_commands::{codec_math_codegen, stable_api};
#[cfg(test)]
use command_support::passed_test_count;
use panic_surface::panic_surface;
use quality_commands::{
    ci, clippy, clippy_strict, deny, doc, downstream_smoke, fmt, fuzz_build, fuzz_run, machete,
    miri, nextest, no_std, repo_lint, test, typos, verify_unsafe_audit,
};
use release_commands::{package, release_cpu, release_integrity, STABLE_SEMVER_PACKAGES};
use stable_api::CARGO_PUBLIC_API_VERSION;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("xtask failed: {err}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let task = env::args().nth(1).unwrap_or_else(|| "help".to_string());
    match task.as_str() {
        "fmt" => fmt(),
        "clippy" => clippy(),
        "clippy-strict" => clippy_strict(),
        "test" => test(),
        "nextest" => nextest(),
        "doc" | "docs" => doc(),
        "typos" => typos(),
        "bench-build" => bench_build(env::args().skip(2)),
        "bench-report" => bench_report(env::args().skip(2)),
        "j2k-ml-bench-metal" => j2k_ml_batch_bench_metal(),
        "j2k-ml-bench-cuda" => j2k_ml_batch_bench_cuda(),
        #[cfg(feature = "adoption")]
        "adoption-benchmark" => adoption_benchmark::adoption_benchmark(env::args().skip(2)),
        #[cfg(not(feature = "adoption"))]
        "adoption-benchmark" => adoption_disabled::adoption_benchmark(env::args().skip(2)),
        #[cfg(feature = "adoption")]
        "adoption-curate" => adoption_curate::adoption_curate(env::args().skip(2)),
        #[cfg(not(feature = "adoption"))]
        "adoption-curate" => adoption_disabled::adoption_curate(env::args().skip(2)),
        #[cfg(feature = "adoption")]
        "adoption-manifest" => adoption_manifest::adoption_manifest(env::args().skip(2)),
        #[cfg(not(feature = "adoption"))]
        "adoption-manifest" => adoption_disabled::adoption_manifest(env::args().skip(2)),
        #[cfg(feature = "adoption")]
        "adoption-materialize" => adoption_materialize::adoption_materialize(env::args().skip(2)),
        #[cfg(not(feature = "adoption"))]
        "adoption-materialize" => adoption_disabled::adoption_materialize(env::args().skip(2)),
        #[cfg(feature = "adoption")]
        "adoption-report" => adoption_report::adoption_report(env::args().skip(2)),
        #[cfg(not(feature = "adoption"))]
        "adoption-report" => adoption_disabled::adoption_report(env::args().skip(2)),
        "public-support" => public_support::public_support(env::args().skip(2)),
        "j2k-bench-signoff" => j2k_bench_signoff(),
        "j2k-perf-guard" => perf_guard::j2k_perf_guard(env::args().skip(2)),
        "codec-math-codegen" => codec_math_codegen(env::args().skip(2)),
        "fuzz-build" => fuzz_build(),
        "fuzz-run" => fuzz_run(),
        "stable-api" => stable_api(env::args().skip(2)),
        "semver" => semver::semver(
            env::args().skip(2),
            STABLE_SEMVER_PACKAGES,
            CARGO_PUBLIC_API_VERSION,
        ),
        "deny" => deny(),
        "miri" => miri(),
        "machete" => machete(),
        "clone-audit" => clone_audit(env::args().skip(2)),
        "panic-surface" => panic_surface(),
        "no-std" => no_std(),
        "unsafe-audit" => verify_unsafe_audit(),
        "downstream-smoke" => downstream_smoke(),
        "repo-lint" => repo_lint(env::args().skip(2)),
        "release-integrity" => release_integrity(env::args().skip(2)),
        "release-status" => release_status::release_status(env::args().skip(2)),
        "release-cpu" => release_cpu(),
        "release-cuda" => cuda::release_cuda(env::args().skip(2)),
        "metal-compile" => metal::metal_compile(),
        "release-metal" => metal::release_metal(env::args().skip(2)),
        "coverage" => coverage::coverage(env::args().skip(2)),
        "package" => package(),
        "ci" => ci(),
        "help" | "-h" | "--help" => {
            print_help();
            Ok(())
        }
        other => Err(format!("unknown task `{other}`")),
    }
}

fn print_help() {
    println!(
        "usage: cargo xtask <task>\n\n\
         tasks:\n\
          ci            fmt, clippy, panic-surface, test, docs, and unsafe-audit\n\
           fmt           check rustfmt\n\
           clippy        run clippy with warnings denied\n\
           clippy-strict run stricter J2K clippy gates\n\
           test          run workspace tests\n\
           nextest       run workspace tests with cargo-nextest\n\
           doc           build workspace docs with warnings denied\n\
           typos         run typos\n\
           bench-build   compile benchmark targets [--lane host|cuda|metal|all]\n\
           bench-report  print or write a benchmark publication report\n\
           j2k-ml-bench-metal benchmark Metal codec-resident and Burn-direct batch decode\n\
           j2k-ml-bench-cuda benchmark CUDA codec-resident and Burn-direct batch decode\n\
           adoption-benchmark run CPU comparator and optional CUDA/Metal adoption benchmark bundle [--features adoption]\n\
           adoption-curate stage inspectable external J2K fixtures and a pinned manifest [--features adoption]\n\
           adoption-manifest generate decode and encode fixture manifests for adoption benchmarks [--features adoption]\n\
           adoption-materialize stage source images into fixed J2K/HTJ2K fixtures and manifests [--features adoption]\n\
           adoption-report render a marketing-safe report from an adoption benchmark bundle [--features adoption]\n\
          public-support verify the public J2K/HTJ2K support matrix and publication gates [--final]\n\
          j2k-bench-signoff run required OpenJPEG/Grok parity and J2K compare bench compile gates\n\
          j2k-perf-guard compare one strict host/CUDA/Metal Criterion lane against a baseline git ref\n\
          codec-math-codegen check generated codec-math Rust and Metal fragments\n\
           fuzz-build    compile fuzz harnesses\n\
           fuzz-run      run scheduled fuzz targets with J2K_FUZZ_RUNS\n\
           stable-api    check the generated 1.0 public API inventory snapshot\n\
           semver        verify computed release types and reviewed API diff [--write-report]\n\
           deny          run cargo-deny\n\
           miri          run selected CPU/no_std crates under Miri\n\
           machete       run cargo-machete unused-dependency scan\n\
           clone-audit   stage source-aware production Rust and run pinned jscpd\n\
           panic-surface run production-library unwrap/expect and explicit panic-macro ratchets\n\
           no-std        check no_std-compatible codec crates\n\
           unsafe-audit  verify docs/unsafe-audit.md lists unsafe Rust sources\n\
           downstream-smoke run facade and transcode examples used by integration docs\n\
           repo-lint     run repository policy checks owned by xtask [--strict]\n\
           release-integrity validate offline release metadata; --publish requires final dated/signoff state\n\
           release-status verify one frozen SHA's CI aggregate and both GPU jobs [--sha SHA] [--repository owner/name]\n\
           release-cpu   run release-mode CPU codec tests\n\
           release-cuda  run fail-closed CUDA validation on Linux x86_64 [--mode quick|full]\n\
           metal-compile compile all Metal targets and run default/pure tests on hosted macOS\n\
           release-metal run fail-closed Metal hardware validation on macOS [--mode quick|full]\n\
           coverage      enforce >=80% host-wide or accelerator critical-path coverage [host|metal|cuda] [--base REV]\n\
           package       construct all staged packages from a clean worktree and publish-dry-run registry-independent crates"
    );
}

#[cfg(test)]
mod tests {
    use super::passed_test_count;
    use super::perf_guard::{
        compare_estimates, discover_estimates, sync_benchmark_sources, BenchEstimate,
        RegressionOutcome,
    };
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn passed_test_count_sums_rust_test_summaries() {
        let output = "\
running 8 tests
test result: ok. 8 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
running 1 test
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
";

        assert_eq!(passed_test_count(output), 9);
    }

    #[test]
    fn passed_test_count_ignores_failed_or_unrelated_lines() {
        let output = "\
test result: FAILED. 1 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out
some other output: 12 passed
";

        assert_eq!(passed_test_count(output), 0);
    }

    #[test]
    fn compare_estimates_flags_median_regression_above_threshold() {
        let baseline = BenchEstimate {
            id: "j2k_public_decode/htj2k_gray8_full_512x512".to_string(),
            median_ns: 1_000.0,
            median_lower_ns: 990.0,
            median_upper_ns: 1_000.0,
        };
        let current = BenchEstimate {
            id: baseline.id.clone(),
            median_ns: 1_120.0,
            median_lower_ns: 1_120.0,
            median_upper_ns: 1_130.0,
        };

        let outcomes = compare_estimates(&[baseline], &[current], 10.0).unwrap();

        assert_eq!(
            outcomes,
            vec![RegressionOutcome {
                id: "j2k_public_decode/htj2k_gray8_full_512x512".to_string(),
                baseline_ns: 1_000.0,
                current_ns: 1_120.0,
                delta_percent: 12.0,
                enforced: true,
                threshold_exceeded: true,
                regressed: true,
            }]
        );
    }

    #[test]
    fn compare_estimates_allows_median_delta_at_threshold() {
        let baseline = BenchEstimate {
            id: "tier1_bitplane_decode/decode_64x64/default".to_string(),
            median_ns: 2_000.0,
            median_lower_ns: 1_990.0,
            median_upper_ns: 2_000.0,
        };
        let current = BenchEstimate {
            id: baseline.id.clone(),
            median_ns: 2_200.0,
            median_lower_ns: 2_200.0,
            median_upper_ns: 2_210.0,
        };

        let outcomes = compare_estimates(&[baseline], &[current], 10.0).unwrap();

        assert!(!outcomes[0].regressed);
        assert!((outcomes[0].delta_percent - 10.0).abs() < f64::EPSILON);
    }

    #[test]
    fn compare_estimates_reports_missing_current_result() {
        let baseline = BenchEstimate {
            id: "j2k_public_decode/htj2k_gray8_full_512x512".to_string(),
            median_ns: 500.0,
            median_lower_ns: 490.0,
            median_upper_ns: 510.0,
        };

        let err = compare_estimates(&[baseline], &[], 10.0).unwrap_err();

        assert!(
            err.contains("missing current benchmark result"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn compare_estimates_requires_confident_material_regression() {
        let tiny_baseline = BenchEstimate {
            id: "htj2k_refinement_block_decode/ds0_ht_09_b11_sigprop".to_string(),
            median_ns: 100.0,
            median_lower_ns: 99.0,
            median_upper_ns: 101.0,
        };
        let tiny_current = BenchEstimate {
            id: tiny_baseline.id.clone(),
            median_ns: 130.0,
            median_lower_ns: 129.0,
            median_upper_ns: 131.0,
        };
        let noisy_baseline = BenchEstimate {
            id: "j2k_public_decode_gray/gray8_full_128x128".to_string(),
            median_ns: 610_000.0,
            median_lower_ns: 606_000.0,
            median_upper_ns: 615_000.0,
        };
        let noisy_current = BenchEstimate {
            id: noisy_baseline.id.clone(),
            median_ns: 673_000.0,
            median_lower_ns: 664_000.0,
            median_upper_ns: 681_000.0,
        };

        let outcomes = compare_estimates(
            &[tiny_baseline, noisy_baseline],
            &[tiny_current, noisy_current],
            10.0,
        )
        .unwrap();

        assert!(outcomes.iter().all(|outcome| !outcome.regressed));
    }

    #[test]
    fn discover_estimates_reads_criterion_median_point_estimates() {
        let root = temp_dir("j2k-perf-guard-test");
        let estimate_path = root
            .join("target")
            .join("criterion")
            .join("j2k_public_decode")
            .join("rgb8_full_128x128")
            .join("new");
        fs::create_dir_all(&estimate_path).unwrap();
        fs::write(
            estimate_path.join("estimates.json"),
            r#"{"median":{"point_estimate":321.5,"confidence_interval":{"lower_bound":321.5,"upper_bound":321.5}}}"#,
        )
        .unwrap();

        let estimates = discover_estimates(&root.join("target").join("criterion")).unwrap();

        assert_eq!(
            estimates,
            vec![BenchEstimate {
                id: "j2k_public_decode/rgb8_full_128x128".to_string(),
                median_ns: 321.5,
                median_lower_ns: 321.5,
                median_upper_ns: 321.5,
            }]
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn sync_benchmark_sources_overlays_current_bench_files() {
        let root = temp_dir("j2k-perf-guard-sync-test");
        let source = root.join("source");
        let target = root.join("target");
        let public_bench = "crates/j2k/benches/public_api.rs";
        let jpeg_encode_bench = "crates/j2k-jpeg/benches/encode_cpu.rs";
        let cuda_decode_bench = "crates/j2k-cuda/benches/htj2k_decode.rs";
        let cuda_encode_bench = "crates/j2k-cuda/benches/htj2k_encode.rs";
        let metal_bench = "crates/j2k-jpeg-metal/benches/compare.rs";
        let native_bench = "crates/j2k-native/benches/tier1_bitplane.rs";
        let native_sigprop_bench = "crates/j2k-native/benches/htj2k_sigprop_phase.rs";
        let native_fixture = "crates/j2k-native/fixtures/htj2k/openhtj2k_ds0_ht_09_b11.j2k";
        for (path, current) in [
            (public_bench, "current public bench"),
            (jpeg_encode_bench, "current jpeg encode bench"),
            (cuda_decode_bench, "current cuda decode bench"),
            (cuda_encode_bench, "current cuda encode bench"),
            (metal_bench, "current metal bench"),
            (native_bench, "current native bench"),
            (native_sigprop_bench, "current sigprop bench"),
            (native_fixture, "current fixture"),
        ] {
            write_fixture(&source, path, current);
            write_fixture(&target, path, "old fixture content");
        }
        for (path, package) in [
            ("crates/j2k-jpeg/Cargo.toml", "j2k-jpeg"),
            ("crates/j2k-cuda/Cargo.toml", "j2k-cuda"),
            ("crates/j2k-jpeg-metal/Cargo.toml", "j2k-jpeg-metal"),
        ] {
            write_fixture(&target, path, &format!("[package]\nname = \"{package}\"\n"));
        }

        sync_benchmark_sources(&source, &target).unwrap();

        assert_eq!(
            fs::read_to_string(target.join(public_bench)).unwrap(),
            "current public bench"
        );
        assert_eq!(
            fs::read_to_string(target.join(jpeg_encode_bench)).unwrap(),
            "current jpeg encode bench"
        );
        let target_jpeg_manifest =
            fs::read_to_string(target.join("crates/j2k-jpeg/Cargo.toml")).unwrap();
        assert!(
            target_jpeg_manifest.contains("name = \"encode_cpu\"")
                && target_jpeg_manifest.contains("harness = false"),
            "baseline overlay must register encode_cpu as a Criterion bench"
        );
        assert_eq!(
            fs::read_to_string(target.join(cuda_decode_bench)).unwrap(),
            "current cuda decode bench"
        );
        assert_eq!(
            fs::read_to_string(target.join(cuda_encode_bench)).unwrap(),
            "current cuda encode bench"
        );
        let target_cuda_manifest =
            fs::read_to_string(target.join("crates/j2k-cuda/Cargo.toml")).unwrap();
        assert!(
            target_cuda_manifest.contains("name = \"htj2k_decode\"")
                && target_cuda_manifest.contains("name = \"htj2k_encode\"")
                && target_cuda_manifest.contains("required-features = [\"cuda-runtime\"]"),
            "baseline overlay must register CUDA HTJ2K Criterion benches"
        );
        assert_eq!(
            fs::read_to_string(target.join(metal_bench)).unwrap(),
            "current metal bench"
        );
        assert!(
            fs::read_to_string(target.join("crates/j2k-jpeg-metal/Cargo.toml"))
                .unwrap()
                .contains("name = \"compare\""),
            "baseline overlay must register the Metal Criterion bench"
        );
        assert_eq!(
            fs::read_to_string(target.join(native_bench)).unwrap(),
            "current native bench"
        );
        assert_eq!(
            fs::read_to_string(target.join(native_sigprop_bench)).unwrap(),
            "current sigprop bench"
        );
        assert_eq!(
            fs::read_to_string(target.join(native_fixture)).unwrap(),
            "current fixture"
        );
        let _ = fs::remove_dir_all(root);
    }

    fn write_fixture(root: &std::path::Path, relative: &str, contents: &str) {
        let path = root.join(relative);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, contents).unwrap();
    }

    fn temp_dir(name: &str) -> std::path::PathBuf {
        let mut dir = std::env::temp_dir();
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        dir.push(format!("{name}-{}-{nanos}", std::process::id()));
        dir
    }
}
