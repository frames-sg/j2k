// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;

use super::{
    assert_contains_all, assert_file_pattern_checks, assert_pattern_checks,
    cargo_metadata_workspace_edges, const_array_block, repo_root, FilePatternCheck, PatternCheck,
};

#[test]
fn crates_io_publish_policy_is_explicit() {
    let root = repo_root();
    let workspace = fs::read_to_string(root.join("Cargo.toml")).expect("read workspace manifest");
    let changelog = fs::read_to_string(root.join("CHANGELOG.md")).expect("read changelog");
    let xtask = fs::read_to_string(root.join("xtask/src/main.rs")).expect("read xtask");
    let publishable = const_array_block(&xtask, "PUBLISHABLE_PACKAGES");
    let publish_workflow = fs::read_to_string(root.join(".github/workflows/publish.yml"))
        .expect("read publish workflow");
    let version = workspace_package_version(&workspace);

    assert!(
        changelog.contains(&format!("## [{version}]")),
        "CHANGELOG.md must contain a section for the current staged release version {version}"
    );

    assert_pattern_checks(&[
        PatternCheck::new("xtask publishable package gate", publishable)
            .required(&[
                "\"j2k-core\"",
                "\"j2k-cuda-runtime\"",
                "\"j2k-profile\"",
                "\"j2k-native\"",
                "\"j2k-tilecodec\"",
                "\"j2k-jpeg\"",
                "\"j2k\"",
                "\"j2k-jpeg-metal\"",
                "\"j2k-jpeg-cuda\"",
                "\"j2k-metal\"",
                "\"j2k-cuda\"",
                "\"j2k-cli\"",
            ])
            .forbidden(&["\"j2k-compare\""]),
        PatternCheck::new("publish workflow package jobs", &publish_workflow)
            .required(&[
                "publish-j2k-core:",
                "publish-j2k-cuda-runtime:",
                "publish-j2k-profile:",
                "publish-j2k-native:",
                "publish-j2k-tilecodec:",
                "publish-j2k-jpeg:",
                "publish-j2k:",
                "publish-j2k-jpeg-metal:",
                "publish-j2k-jpeg-cuda:",
                "publish-j2k-metal:",
                "publish-j2k-cuda:",
                "publish-j2k-cli:",
            ])
            .forbidden(&["publish-j2k-compare:"]),
    ]);
}

#[test]
fn release_docs_use_manifest_versions_for_publish_order() {
    let xtask = fs::read_to_string(repo_root().join("xtask/src/main.rs")).expect("read xtask");

    assert_file_pattern_checks(
        repo_root(),
        &[FilePatternCheck::new("docs/release.md")
            .named("release docs")
            .required(&[
                "manifest versions",
                "cargo metadata --locked --no-deps",
                "block v0.1.6",
                "metal v0.33.0",
                "future-incompatibility",
                "current crates.io release",
            ])
            .forbidden(&["`j2k` `1.1.0`", "`j2k-native` `0.3.0`", "`j2k` `1.0.0`"])],
    );
    let cargo_metadata_fn = xtask
        .split("fn cargo_metadata()")
        .nth(1)
        .and_then(|rest| rest.split("fn package_name").next())
        .expect("xtask cargo_metadata function");
    assert_pattern_checks(&[PatternCheck::new(
        "release integrity cargo metadata call",
        cargo_metadata_fn,
    )
    .required(&["\"metadata\"", "\"--locked\"", "\"--no-deps\""])]);
}

fn workspace_package_version(workspace_manifest: &str) -> &str {
    workspace_manifest
        .lines()
        .find_map(|line| {
            let line = line.trim();
            line.strip_prefix("version")
                .and_then(|rest| rest.split('"').nth(1))
        })
        .expect("workspace package version")
}

#[test]
fn j2k_compare_stays_unpublished_and_out_of_j2k_package_deps() {
    assert_file_pattern_checks(
        repo_root(),
        &[
            FilePatternCheck::new("crates/j2k-compare/Cargo.toml")
                .named("j2k-compare manifest")
                .required(&["publish = false"]),
            FilePatternCheck::new("crates/j2k/Cargo.toml")
                .named("j2k manifest")
                .forbidden(&["j2k-compare"]),
        ],
    );
}

#[test]
fn package_preflight_is_staged_dependency_aware() {
    let root = repo_root();
    let xtask = fs::read_to_string(root.join("xtask/src/main.rs")).expect("read xtask");

    assert_file_pattern_checks(
        root,
        &[
            FilePatternCheck::new("xtask/src/main.rs")
                .named("xtask package preflight")
                .required(&[
                    "PUBLISHABLE_PACKAGES",
                    "REGISTRY_INDEPENDENT_PACKAGES",
                    "STAGED_DEPENDENCY_PACKAGES",
                    "\"--list\"",
                    "[\"package\", \"-p\", package, \"--no-verify\"]",
                    "[\"publish\", \"-p\", package, \"--dry-run\"]",
                ]),
            FilePatternCheck::new("scripts/publish-crate.sh")
                .named("publish script")
                .required(&[
                    "registry_independent_crates=(",
                    "if [[ \"$#\" -ne 1 ]]",
                    "--preflight-all",
                    "scripts/crates_io_version.py verify-set",
                    "scripts/crates_io_version.py state",
                    "require_positive_decimal \"CRATES_IO_PUBLISH_ATTEMPTS\"",
                    "require_nonnegative_decimal \"CRATES_IO_RATE_LIMIT_RETRY_SECONDS\"",
                    "require_nonnegative_decimal \"CRATES_IO_INDEX_SETTLE_SECONDS\"",
                    "j2k-cli",
                    "cargo package -p \"$crate\" --no-verify",
                    "cargo publish -p \"$crate\" --dry-run",
                ])
                .forbidden(&["dry-run package list only", "cargo info"]),
            FilePatternCheck::new("scripts/crates_io_version.py")
                .named("fail-closed crates.io version helper")
                .required(&[
                    "VersionState.AVAILABLE",
                    "VersionState.PUBLISHED",
                    "error.code == 404",
                    "dependency-order prefix",
                    "allow_published_rerun",
                ]),
            FilePatternCheck::new("docs/release.md")
                .named("release docs")
                .required(&[
                    "cargo xtask package",
                    "cargo package --list",
                    "cargo package --no-verify",
                    "cargo publish --dry-run",
                    "already-published prefix",
                    "Only an exact HTTP 404",
                ]),
        ],
    );
    let publishable_packages =
        const_array_entries(const_array_block(&xtask, "PUBLISHABLE_PACKAGES"));
    let strict_packages = const_array_block(&xtask, "REGISTRY_INDEPENDENT_PACKAGES");
    let staged_packages = const_array_block(&xtask, "STAGED_DEPENDENCY_PACKAGES");
    for package in publishable_packages {
        let strict = strict_packages.contains(&format!("\"{package}\""));
        let staged = staged_packages.contains(&format!("\"{package}\""));
        assert_ne!(
            strict, staged,
            "publishable package `{package}` must appear in exactly one package-gate partition"
        );
    }
    for (package, dependency) in cargo_metadata_workspace_edges(root) {
        assert!(
            !strict_packages.contains(&format!("\"{package}\"")),
            "strict package preflight must not include `{package}` while it depends on staged workspace crate `{dependency}`"
        );
    }
    assert!(
        staged_packages.contains("\"j2k-cuda-runtime\""),
        "j2k-cuda-runtime depends on staged j2k-core and must not run strict package verification before publication"
    );
}

fn const_array_entries(block: &str) -> Vec<&str> {
    block
        .lines()
        .map(|line| line.trim().trim_matches([',', '"']))
        .filter(|entry| {
            !entry.is_empty()
                && !entry.starts_with("const ")
                && !entry.starts_with(']')
                && !entry.starts_with('&')
        })
        .collect()
}

#[test]
fn publish_script_covers_all_publishable_crates() {
    let root = repo_root();
    let xtask = fs::read_to_string(root.join("xtask/src/main.rs")).expect("read xtask");
    let publish_script =
        fs::read_to_string(root.join("scripts/publish-crate.sh")).expect("read publish script");
    let publishable = const_array_block(&xtask, "PUBLISHABLE_PACKAGES");
    let publishable_packages: Vec<&str> = publishable
        .lines()
        .map(|line| line.trim().trim_matches([',', '"']))
        .filter(|package| {
            !package.is_empty()
                && !package.starts_with("const ")
                && !package.starts_with(']')
                && !package.starts_with('&')
        })
        .collect();

    assert_contains_all(
        "publish script publishable package coverage",
        &publish_script,
        &publishable_packages,
    );
}
