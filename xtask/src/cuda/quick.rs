//! Consolidated production-feature command plan for quick CUDA validation.

use super::CUDA_RUNTIME_TEST_TARGETS;

const QUICK_CUDA_FEATURES: &str = "j2k-cuda-runtime/cuda-oxide,j2k-jpeg-cuda/cuda-runtime,j2k-cuda/cuda-runtime,j2k-transcode-cuda/cuda-runtime,j2k-ml/cuda";
const QUICK_CUDA_PACKAGES: &[&str] = &[
    "j2k-cuda-runtime",
    "j2k-jpeg-cuda",
    "j2k-cuda",
    "j2k-transcode-cuda",
    "j2k-ml",
];

fn append_packages(args: &mut Vec<&'static str>) {
    for package in QUICK_CUDA_PACKAGES {
        args.extend_from_slice(&["-p", package]);
    }
}

pub(super) fn quick_runtime_args() -> Vec<&'static str> {
    let mut args = vec!["test", "--profile", "gpu-quick"];
    append_packages(&mut args);
    args.extend_from_slice(&["--features", QUICK_CUDA_FEATURES]);
    args.extend_from_slice(CUDA_RUNTIME_TEST_TARGETS);
    args.extend_from_slice(&["--", "--show-output"]);
    args
}
