// SPDX-License-Identifier: MIT OR Apache-2.0

//! Structural ownership and size ratchets for the xtask command dispatcher.

use std::fs;

use super::{assert_pattern_checks, repo_root, rust_sources, PatternCheck};

fn read(relative_path: &str) -> String {
    fs::read_to_string(repo_root().join(relative_path))
        .unwrap_or_else(|error| panic!("read {relative_path}: {error}"))
}

#[test]
fn xtask_dispatcher_stays_split_by_command_family() {
    let main = read("xtask/src/main.rs");
    let benchmark = read("xtask/src/benchmark_commands.rs");
    let codegen = read("xtask/src/codegen_commands.rs");
    let support = read("xtask/src/command_support.rs");
    let panic_surface = read("xtask/src/panic_surface.rs");
    let quality = read("xtask/src/quality_commands.rs");
    let release = read("xtask/src/release_commands.rs");

    for (relative_path, source, max_lines) in [
        ("xtask/src/main.rs", main.as_str(), 700),
        ("xtask/src/benchmark_commands.rs", benchmark.as_str(), 800),
        ("xtask/src/codegen_commands.rs", codegen.as_str(), 800),
        ("xtask/src/command_support.rs", support.as_str(), 300),
        ("xtask/src/panic_surface.rs", panic_surface.as_str(), 650),
        ("xtask/src/quality_commands.rs", quality.as_str(), 800),
        ("xtask/src/release_commands.rs", release.as_str(), 800),
    ] {
        let line_count = source.lines().count();
        assert!(
            line_count < max_lines,
            "{relative_path} has {line_count} lines; expected fewer than {max_lines}"
        );
        assert!(
            !source.contains("::*"),
            "{relative_path} must keep explicit imports"
        );
        assert!(
            !source.contains("include!("),
            "{relative_path} must remain a real Rust module"
        );
        assert!(
            !source.contains("#[allow("),
            "{relative_path} must not add lint suppressions"
        );
    }

    assert_pattern_checks(&[
        PatternCheck::new("xtask root dispatcher", &main)
            .required(&[
                "mod benchmark_commands;",
                "mod codegen_commands;",
                "mod command_support;",
                "mod panic_surface;",
                "mod quality_commands;",
                "mod release_commands;",
                "fn run() -> Result<(), String>",
                "fn print_help()",
            ])
            .forbidden(&[
                "fn bench_build()",
                "fn codec_math_codegen(",
                "fn ensure_clean_worktree()",
                "fn panic_surface()",
                "fn release_integrity()",
            ]),
        PatternCheck::new("xtask benchmark command ownership", &benchmark).required(&[
            "pub(super) fn bench_build()",
            "pub(super) fn j2k_bench_signoff()",
            "pub(super) fn bench_report(",
            "fn render_benchmark_report(",
        ]),
        PatternCheck::new("xtask codegen command ownership", &codegen).required(&[
            "pub(super) fn stable_api(",
            "pub(super) fn codec_math_codegen(",
            "fn render_codec_math_dwt97_metal_fragment()",
            "fn render_stable_api_snapshot()",
        ]),
        PatternCheck::new("xtask process and path helper ownership", &support).required(&[
            "pub(super) fn ensure_clean_worktree()",
            "pub(super) fn run_cargo(",
            "pub(super) fn run_cargo_test_with_pass_floor(",
            "pub(super) fn command_output_os_detailed(",
            "pub(super) fn rust_sources(",
        ]),
        PatternCheck::new("xtask panic-surface command ownership", &panic_surface).required(&[
            "pub(super) fn panic_surface()",
            "fn parse_panic_surface_selection(",
            "fn parse_panic_surface_output(",
        ]),
        PatternCheck::new("xtask quality command ownership", &quality).required(&[
            "pub(super) fn ci()",
            "pub(super) fn clippy_strict()",
            "pub(super) fn fuzz_build()",
            "pub(super) fn verify_unsafe_audit()",
        ]),
        PatternCheck::new("xtask release command ownership", &release).required(&[
            "const PUBLISHABLE_PACKAGES:",
            "pub(super) const STABLE_SEMVER_PACKAGES:",
            "pub(super) fn release_integrity()",
            "pub(super) fn release_cpu()",
            "pub(super) fn package()",
        ]),
    ]);
}

#[test]
fn xtask_manifest_keeps_pedantic_clippy_enabled() {
    let manifest = read("xtask/Cargo.toml");
    assert_pattern_checks(&[PatternCheck::new("xtask manifest", &manifest)
        .normalized_required(&["pedantic = { level = \"warn\", priority = -1 }"])
        .normalized_forbidden(&["pedantic = \"allow\"", "pedantic = { level = \"allow\""])]);
    for relative_dir in ["xtask/src", "xtask/tests"] {
        for path in rust_sources(&repo_root().join(relative_dir)) {
            let source = fs::read_to_string(&path)
                .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
            assert!(
                source
                    .lines()
                    .all(|line| !line.trim_start().starts_with(concat!("#!", "[allow"))),
                "{} must not use crate- or module-wide lint allows",
                path.strip_prefix(repo_root()).unwrap_or(&path).display()
            );
        }
    }
}
