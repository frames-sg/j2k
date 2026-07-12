// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{ffi::OsStr, fs};

use super::{
    assert_file_pattern_checks, assert_rust_source_scan_checks, repo_root, rust_sources,
    FilePatternCheck, RustSourceScanCheck,
};

#[test]
fn adapter_crates_do_not_import_codec_private_modules() {
    assert_rust_source_scan_checks(
        repo_root(),
        &[RustSourceScanCheck::new(
            "adapter codec-private module imports",
            &[
                "crates/j2k-jpeg-metal",
                "crates/j2k-jpeg-cuda",
                "crates/j2k-metal",
                "crates/j2k-cuda",
            ],
        )
        .forbidden(&["::__private", " __private::"])],
    );
}

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

    while let Some(dir) = pending.pop() {
        for entry in
            fs::read_dir(&dir).unwrap_or_else(|err| panic!("read {}: {err}", dir.display()))
        {
            let entry = entry.expect("read directory entry");
            let path = entry.path();
            if path.is_dir() {
                pending.push(path);
                continue;
            }
            if !matches!(path.extension().and_then(OsStr::to_str), Some("cu" | "ptx")) {
                continue;
            }
            let rel = path
                .strip_prefix(&cuda_runtime_src)
                .unwrap_or(&path)
                .to_string_lossy()
                .replace('\\', "/");
            if rel.starts_with("fixtures/") || rel.starts_with("test-fixtures/") {
                continue;
            }
            forbidden.push(rel);
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
    let runtime_src = root.join("crates/j2k-cuda-runtime/src");
    let deprecated_selector = ["J2K", "CUDA", "USE", "OXIDE"].join("_");
    let mut violations = Vec::new();

    for path in rust_sources(&runtime_src) {
        let source = fs::read_to_string(&path)
            .unwrap_or_else(|err| panic!("read {}: {err}", path.display()));
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

#[test]
fn cuda_trace_export_is_non_clobbering_and_documented() {
    assert_file_pattern_checks(
        repo_root(),
        &[
            FilePatternCheck::new("crates/j2k-cuda/src/profile/trace.rs")
                .named("CUDA profile trace writer")
                .required(&[
                    "OpenOptions::new().write(true).create_new(true).open(path)",
                    "fn write_trace_file",
                    "emit_trace_write_error(\"cuda_htj2k_trace_write\"",
                    "emit_trace_write_error(\"cuda_htj2k_encode_trace_write\"",
                    "std::io::ErrorKind::AlreadyExists",
                    "CUDA trace path already exists",
                ])
                .forbidden(&["std::fs::write(&trace_path"]),
            FilePatternCheck::new("crates/j2k-cuda/src/profile/tests.rs")
                .named("CUDA trace non-clobber regression")
                .required(&[
                    "write_trace_file(&path, trace)",
                    "write_trace_file(&path, \"replace\")",
                    "ErrorKind::AlreadyExists",
                ]),
            FilePatternCheck::new("docs/env-vars.md")
                .named("environment variable docs")
                .required(&[
                    "operator-supplied path",
                    "Existing files are not overwritten",
                    "parent directories are not created",
                ]),
        ],
    );
}

#[test]
fn cuda_adapter_crates_keep_public_libs_as_module_shells() {
    let root = repo_root();
    let expected_modules = [
        (
            "crates/j2k-jpeg-cuda",
            [
                "codec.rs",
                "decoder.rs",
                "error.rs",
                "runtime.rs",
                "session.rs",
                "surface.rs",
            ]
            .as_slice(),
        ),
        (
            "crates/j2k-cuda",
            [
                "codec.rs",
                "decoder.rs",
                "encode.rs",
                "error.rs",
                "runtime.rs",
                "session.rs",
                "surface.rs",
            ]
            .as_slice(),
        ),
    ];

    for (crate_dir, modules) in expected_modules {
        let src_dir = root.join(crate_dir).join("src");
        let lib_path = src_dir.join("lib.rs");
        let lib = fs::read_to_string(&lib_path)
            .unwrap_or_else(|err| panic!("read {}: {err}", lib_path.display()));
        let line_count = lib.lines().count();
        assert!(
            line_count <= 220,
            "{} should stay a thin public module shell; found {line_count} lines",
            lib_path.strip_prefix(root).unwrap_or(&lib_path).display()
        );

        for module in modules {
            let module_path = src_dir.join(module);
            assert!(
                module_path.exists(),
                "{} must exist to keep CUDA adapter responsibilities focused",
                module_path
                    .strip_prefix(root)
                    .unwrap_or(&module_path)
                    .display()
            );
        }
    }
}

#[test]
fn reusable_benchmark_generators_live_in_test_support() {
    assert_file_pattern_checks(
        repo_root(),
        &[FilePatternCheck::new("crates/j2k-test-support/src/lib.rs")
            .named("j2k-test-support")
            .required(&[
                "pub fn gradient_u8",
                "pub fn patterned_rgb8_tiles",
                "pub fn gpu_bench_rgb8",
            ])],
    );
}

#[test]
fn gpu_runtime_tests_do_not_silently_return_on_missing_hardware_gates() {
    assert_rust_source_scan_checks(
        repo_root(),
        &[RustSourceScanCheck::new(
            "GPU runtime tests must use j2k-test-support gates so optional local skips are visible and CI require gates fail closed",
            &[
                "crates/j2k-cuda-runtime/src",
                "crates/j2k-cuda/src",
                "crates/j2k-cuda/tests",
                "crates/j2k-jpeg-cuda/tests",
                "crates/j2k-transcode-cuda/tests",
                "crates/j2k-metal/src",
                "crates/j2k-metal/tests",
                "crates/j2k-jpeg-metal/tests",
                "crates/j2k-transcode-metal/tests",
            ],
        )
        .forbidden(&[
            "if !cuda_runtime_required() {\n        return;",
            "if !cuda_strict_oxide_required() {\n        return;",
            "if !cuda_jpeg_hardware_decode_required() {\n        return;",
            "if std::env::var_os(\"J2K_REQUIRE_CUDA_RUNTIME\").is_none() {\n        return;",
            "if Device::system_default().is_none() {\n            eprintln!(\"skipping",
            "no Metal device is available",
        ])],
    );
}
