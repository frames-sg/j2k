//! Quick/full CUDA command-mode regressions.

use super::super::{quick_runtime_args, validate_required_named_tests};
use crate::gpu_validation::ValidationMode;

#[test]
fn quick_cuda_plan_uses_one_production_feature_runtime_graph_and_fast_profile() {
    let tests = quick_runtime_args();

    assert!(tests
        .windows(2)
        .any(|pair| pair == ["--profile", "gpu-quick"]));
    assert!(!tests.contains(&"--release"));
    for package in [
        "j2k-cuda-runtime",
        "j2k-jpeg-cuda",
        "j2k-cuda",
        "j2k-transcode-cuda",
        "j2k-ml",
    ] {
        assert!(tests.windows(2).any(|pair| pair == ["-p", package]));
    }
    assert!(tests.contains(&"j2k-cuda-runtime/cuda-oxide,j2k-jpeg-cuda/cuda-runtime,j2k-cuda/cuda-runtime,j2k-transcode-cuda/cuda-runtime,j2k-ml/cuda"));
    assert!(tests.contains(&"--show-output"));
}

#[test]
fn required_named_validation_accepts_extra_combined_tests_and_rejects_missing_tests() {
    let combined = "\
test unrelated ... ok
test alpha ... ok
test beta ... ok
test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
";
    validate_required_named_tests(combined, "CUDA", &["alpha", "beta"]).unwrap();

    let error = validate_required_named_tests(combined, "CUDA", &["alpha", "missing"])
        .expect_err("a missing required test must fail quick CUDA validation");
    assert!(error.contains("missing"));
}

#[test]
fn validation_mode_defaults_full_and_rejects_unknown_arguments() {
    assert_eq!(
        ValidationMode::parse(std::iter::empty()).unwrap(),
        ValidationMode::Full
    );
    assert_eq!(
        ValidationMode::parse(["--mode".to_string(), "quick".to_string()].into_iter()).unwrap(),
        ValidationMode::Quick
    );
    assert!(ValidationMode::parse(["--mode".to_string()].into_iter()).is_err());
    assert!(
        ValidationMode::parse(["--mode".to_string(), "other".to_string()].into_iter()).is_err()
    );
    assert!(ValidationMode::parse(["--unknown".to_string()].into_iter()).is_err());
}
