// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{ffi::OsStr, fs};

use super::super::{assert_rust_source_scan_checks, repo_root, rust_sources, RustSourceScanCheck};

#[test]
fn production_j2k_cuda_code_does_not_reference_nvjpeg() {
    assert_rust_source_scan_checks(
        repo_root(),
        &[RustSourceScanCheck::new(
            "production J2K CUDA nvJPEG references",
            &[
                "crates/j2k-cuda-runtime/src",
                "crates/j2k-jpeg-cuda/src",
                "crates/j2k-jpeg-cuda/benches",
            ],
        )
        .forbidden(&["nvjpeg", "nvJPEG", "Nvjpeg", "NVJPEG"])],
    );
}

#[test]
fn cuda_runtime_rejects_product_cuda_c_and_checked_in_ptx() {
    let root = repo_root();
    let cuda_runtime_src = root.join("crates/j2k-cuda-runtime/src");
    let mut forbidden = Vec::new();
    let mut pending = vec![cuda_runtime_src.clone()];
    while let Some(directory) = pending.pop() {
        for entry in fs::read_dir(&directory)
            .unwrap_or_else(|error| panic!("read {}: {error}", directory.display()))
        {
            let path = entry.expect("read directory entry").path();
            if path.is_dir() {
                pending.push(path);
                continue;
            }
            if !matches!(path.extension().and_then(OsStr::to_str), Some("cu" | "ptx")) {
                continue;
            }
            let relative = path
                .strip_prefix(&cuda_runtime_src)
                .unwrap_or(&path)
                .to_string_lossy()
                .replace('\\', "/");
            if !relative.starts_with("fixtures/") && !relative.starts_with("test-fixtures/") {
                forbidden.push(relative);
            }
        }
    }
    forbidden.sort();
    assert!(
        forbidden.is_empty(),
        "product CUDA C sources and checked-in product PTX are retired; use CUDA Oxide projects instead:\n{}",
        forbidden.join("\n")
    );
}

#[test]
fn cuda_runtime_dispatch_does_not_read_deprecated_oxide_route_selectors() {
    let root = repo_root();
    let deprecated_selector = ["J2K", "CUDA", "USE", "OXIDE"].join("_");
    let mut violations = Vec::new();
    for path in rust_sources(&root.join("crates/j2k-cuda-runtime/src")) {
        let source = fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
        if source.contains(&deprecated_selector) {
            violations.push(
                path.strip_prefix(root)
                    .unwrap_or(&path)
                    .display()
                    .to_string(),
            );
        }
    }
    assert!(
        violations.is_empty(),
        "CUDA Oxide route selection must be feature-driven, not runtime-env driven:\n{}",
        violations.join("\n")
    );
}
