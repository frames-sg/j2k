// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{collections::BTreeSet, fs};

use crate::repo_lint_support::{
    assert_pattern_checks, const_array_block, release_manifest_entries, repo_root, workflow_job,
    PatternCheck,
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
    assert_published_library_partition(root, &semver, &policy);
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
                "SEMVER_BASELINE_VERSION: &str = \"0.7.5\"",
                "SEMVER_BASELINE_TAG: &str = \"v0.7.5\"",
                "INTENTIONAL_BREAK_TRANSITION",
                "candidate_version: \"0.8.0\"",
                "required_next_baseline_tag: \"v0.8.0\"",
                "validate_baseline_transition(",
                "SEMVER_BASELINE_TAG}:docs/stable-api-1.0.public-api.txt",
                "collect_package_apis(published_library_packages)?",
                "SnapshotKind::Ordinary",
                "SnapshotKind::Hidden",
                "stale_ordinary_packages",
                "stale_hidden_packages",
                "hidden_fingerprint",
                "J2K_SEMVER_TOOLCHAIN overrides are not accepted",
                "[\"run\", SEMVER_TOOLCHAIN, \"cargo\"]",
                "validate_snapshot_scope(",
            ])
            .forbidden(&[
                "unwrap_or_else(|_| \"1.96\".to_string())",
                "SOURCE_INCOMPATIBLE_PATCH_EXCEPTION_VERSION",
            ]),
        PatternCheck::new("semver review schema", semver_review).required(&[
            "API review config version must be 3",
            "break_ledger",
            "BreakKind::Source",
            "BreakKind::Behavior",
            "removed_items",
            "summary",
            "migration",
            "source-break ledger does not exactly cover",
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
            "published 0.7.5 artifact recorded both ordinary and hidden-enabled passes",
            "`0.8.0` is the only candidate permitted to compare against `v0.7.5`",
            "rotate the semver baseline to the published `v0.8.0`",
            "Source-break entries must enumerate every exact removed API item",
            "Behavior-break entries carry the same summary and migration requirements",
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

fn assert_published_library_partition(root: &std::path::Path, semver: &str, policy: &str) {
    let entries = release_manifest_entries(root);
    let published = entries
        .iter()
        .filter(|entry| entry.api_contract != "binary")
        .map(|entry| entry.name.clone())
        .collect::<BTreeSet<_>>();
    let stable = entries
        .iter()
        .filter(|entry| entry.api_contract == "stable")
        .map(|entry| entry.name.clone())
        .collect::<BTreeSet<_>>();
    let baseline = const_string_array_values(semver, "SEMVER_BASELINE_PACKAGES");
    let new = const_string_array_values(semver, "SEMVER_NEW_PACKAGES");
    assert_eq!(published.len(), 18);
    assert_eq!(
        stable,
        ["j2k", "j2k-core", "j2k-jpeg", "j2k-tilecodec"]
            .into_iter()
            .map(str::to_string)
            .collect()
    );
    assert!(!baseline.is_empty());
    assert!(new.is_empty());
    assert!(baseline.contains("j2k-ml"));
    assert!(baseline.is_disjoint(&new));
    assert_eq!(
        baseline.union(&new).cloned().collect::<BTreeSet<_>>(),
        published
    );
    for package in published {
        assert!(
            policy.contains(&format!("`{package}`")),
            "published API policy must list `{package}`"
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
