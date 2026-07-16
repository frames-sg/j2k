// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{collections::BTreeSet, fs};

use crate::repo_lint_support::{
    assert_pattern_checks, const_array_block, repo_root, workflow_job, xtask_sources, PatternCheck,
};

#[test]
fn ci_stable_api_jobs_pin_inputs_and_never_write() {
    let root = repo_root();
    let workflow = fs::read_to_string(root.join(".github/workflows/full-validation.yml"))
        .expect("read full validation workflow");
    let stable_api_job = workflow_job(&workflow, "stable-api");
    let semver_job = workflow_job(&workflow, "semver");
    let release_candidate_job = workflow_job(&workflow, "release-candidate");

    assert_pattern_checks(&[
        PatternCheck::new("CI stable API job", stable_api_job)
            .required(&[
                "runs-on: macos-latest",
                "toolchain: nightly-2026-06-28",
                "targets: aarch64-apple-darwin",
                "cargo-public-api@0.52.0",
                "- run: cargo xtask stable-api",
            ])
            .forbidden(&["cargo xtask stable-api --write"]),
        PatternCheck::new("CI semver job", semver_job)
            .required(&[
                "runs-on: macos-latest",
                "toolchain: \"1.96\"",
                "toolchain: nightly-2026-06-28",
                "targets: aarch64-apple-darwin",
                "cargo install cargo-semver-checks --version 0.48.0 --locked",
                "cargo-public-api@0.52.0",
                "cargo xtask semver",
            ])
            .forbidden(&["release-type: minor", "cargo xtask semver --write-report"]),
        PatternCheck::new(
            "release candidate stable API dependency",
            release_candidate_job,
        )
        .required(&["- stable-api", "- semver"]),
    ]);
}

#[test]
fn stable_api_and_semver_share_one_fail_closed_inventory_contract() {
    let root = repo_root();
    let xtask = xtask_sources(root);
    let semver = fs::read_to_string(root.join("xtask/src/semver.rs")).expect("read semver xtask");
    let semver_review = fs::read_to_string(root.join("xtask/src/semver/review.rs"))
        .expect("read semver review policy");
    let stable_api =
        fs::read_to_string(root.join("xtask/src/stable_api.rs")).expect("read API collector");
    let codegen = fs::read_to_string(root.join("xtask/src/codegen_commands.rs"))
        .expect("read stable API generator");
    let codegen_transaction =
        fs::read_to_string(root.join("xtask/src/codegen_commands/transaction.rs"))
            .expect("read generated-file transaction owner");
    let command_support = fs::read_to_string(root.join("xtask/src/command_support.rs"))
        .expect("read xtask command support");
    let policy =
        fs::read_to_string(root.join("docs/stable-api-1.0.md")).expect("read stable API policy");

    assert_inventory_contracts(
        &stable_api,
        &codegen,
        &codegen_transaction,
        &semver,
        &semver_review,
        &command_support,
        &policy,
    );
    assert_stable_package_partition(&xtask, &semver, &policy);
}

fn assert_inventory_contracts(
    stable_api: &str,
    codegen: &str,
    codegen_transaction: &str,
    semver: &str,
    semver_review: &str,
    command_support: &str,
    policy: &str,
) {
    assert_pattern_checks(&[
        PatternCheck::new("shared stable API collector", stable_api).required(&[
            "PUBLIC_API_SNAPSHOT: &str = \"docs/stable-api-1.0.public-api.txt\"",
            "docs/stable-api-1.0.implementation-public-api.txt",
            "CARGO_PUBLIC_API_VERSION: &str = \"0.52.0\"",
            "PUBLIC_API_TOOLCHAIN: &str = \"nightly-2026-06-28\"",
            "PUBLIC_API_TARGET: &str = \"aarch64-apple-darwin\"",
            "ORDINARY_RUSTDOCFLAGS: &str = \"-D warnings\"",
            "-D warnings --document-hidden-items",
            "collect_package_apis(",
            ".union(hidden_enabled)",
            ".difference(&ordinary)",
            "[\"run\", PUBLIC_API_TOOLCHAIN, \"cargo\"]",
            "--target",
            "validate_public_api_environment()?",
            "CARGO_ENCODED_RUSTDOCFLAGS",
            "RUSTC_BOOTSTRAP",
            "CARGO_TARGET_",
            "_RUSTFLAGS",
        ]),
        PatternCheck::new("transactional stable API writer", codegen).required(&[
            "write_generated_pair_transactionally(&snapshots)",
            "PUBLIC_API_SNAPSHOT",
            "HIDDEN_API_SNAPSHOT",
        ]),
        PatternCheck::new("generated-file transaction owner", codegen_transaction).required(&[
            "fn stage_generated_file(",
            "rollback_generated_pair_install(",
            "fn restore_originals(",
            "fn sync_generated_directories(",
        ]),
        PatternCheck::new("live semver inventory ratchet", semver)
            .required(&[
                "SEMVER_TOOLCHAIN: &str = \"1.96\"",
                "SEMVER_BASELINE_VERSION: &str = \"0.7.3\"",
                "SEMVER_BASELINE_TAG: &str = \"v0.7.3\"",
                "SOURCE_INCOMPATIBLE_PATCH_EXCEPTION_VERSION: &str = \"0.7.4\"",
                "SEMVER_BASELINE_TAG}:docs/stable-api-1.0.public-api.txt",
                "collect_package_apis(stable_packages)?",
                "SnapshotKind::Ordinary",
                "SnapshotKind::Hidden",
                "stale_ordinary_packages",
                "stale_hidden_packages",
                "hidden_fingerprint",
                "J2K_SEMVER_TOOLCHAIN overrides are not accepted",
                "[\"run\", SEMVER_TOOLCHAIN, \"cargo\"]",
                "validate_snapshot_scope(",
            ])
            .forbidden(&["unwrap_or_else(|_| \"1.96\".to_string())"]),
        PatternCheck::new("semver review schema", semver_review).required(&[
            "API review config version must be 2",
            "hidden_count",
            "hidden_fingerprint",
            "hidden_rationale",
            "nonempty hidden inventory",
            "removed_fingerprint",
            "added_fingerprint",
        ]),
        PatternCheck::new("stable API environment command support", command_support).required(&[
            "fn command_output_os_detailed_with_env(",
            "CommandContext::new().envs(envs)",
            "String::from_utf8(output.stdout)",
        ]),
        PatternCheck::new("stable API policy", policy).required(&[
            "published 0.7.3 artifact recorded both ordinary and hidden-enabled passes",
            "explicit, maintainer-approved source-compatibility exception",
            "exception applies only to `0.7.4`",
            "complete hidden-inventory count and fingerprint",
            "Every semver invocation collects both live passes",
            "Nonempty hidden inventories also require a package-specific hidden rationale",
            "rollback-capable transaction",
            "nightly-2026-06-28",
            "aarch64-apple-darwin",
            "does not accept the former `J2K_SEMVER_TOOLCHAIN` override",
        ]),
    ]);
}

fn assert_stable_package_partition(xtask: &str, semver: &str, policy: &str) {
    let stable = const_string_array_values(xtask, "STABLE_SEMVER_PACKAGES");
    let baseline = const_string_array_values(semver, "SEMVER_BASELINE_PACKAGES");
    let new = const_string_array_values(semver, "SEMVER_NEW_PACKAGES");
    assert!(!stable.is_empty());
    assert!(!baseline.is_empty());
    assert!(baseline.is_disjoint(&new));
    assert_eq!(
        baseline.union(&new).cloned().collect::<BTreeSet<_>>(),
        stable
    );
    for package in stable {
        assert!(
            policy.contains(&format!("`{package}`")),
            "stable API policy must list `{package}`"
        );
    }
}

fn const_string_array_values(source: &str, name: &str) -> BTreeSet<String> {
    let values = const_array_block(source, name)
        .split('"')
        .skip(1)
        .step_by(2)
        .map(str::to_string)
        .collect::<BTreeSet<_>>();
    values
}
