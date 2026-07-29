// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{collections::BTreeSet, fs};

use super::{
    assert_contains_all, assert_file_pattern_checks, assert_pattern_checks,
    cargo_metadata_workspace_edges, release_manifest_entries, repo_root, xtask_sources,
    FilePatternCheck, PatternCheck,
};

#[path = "../../src/release_commands/release_integrity_policy/markdown.rs"]
mod release_markdown;

use release_markdown::content_lines as markdown_content_lines;

#[test]
fn j2k_ml_is_a_publishable_release_crate() {
    let root = repo_root();
    let manifest =
        fs::read_to_string(root.join("crates/j2k-ml/Cargo.toml")).expect("read j2k-ml manifest");
    let ordered_crates = release_manifest_entries(root)
        .into_iter()
        .map(|entry| entry.name)
        .collect::<Vec<_>>();
    let j2k_ml_index = ordered_crates
        .iter()
        .position(|name| name == "j2k-ml")
        .expect("j2k-ml must be in the release manifest");

    assert!(
        !manifest
            .lines()
            .any(|line| line.trim() == "publish = false"),
        "j2k-ml must be publishable"
    );
    assert_contains_all(
        "j2k-ml package metadata",
        &manifest,
        &[
            "homepage.workspace = true",
            "keywords.workspace = true",
            "categories.workspace = true",
            "[package.metadata.docs.rs]",
        ],
    );
    for dependency in ["j2k", "j2k-cuda-runtime", "j2k-cuda", "j2k-metal"] {
        let dependency_index = ordered_crates
            .iter()
            .position(|name| name == dependency)
            .unwrap_or_else(|| panic!("{dependency} must be in the release manifest"));
        assert!(
            dependency_index < j2k_ml_index,
            "{dependency} must be published before j2k-ml"
        );
    }
}

#[test]
fn crates_io_publish_policy_is_explicit() {
    let root = repo_root();
    let workspace = fs::read_to_string(root.join("Cargo.toml")).expect("read workspace manifest");
    let changelog = fs::read_to_string(root.join("CHANGELOG.md")).expect("read changelog");
    let publish_workflow = fs::read_to_string(root.join(".github/workflows/publish.yml"))
        .expect("read publish workflow");
    let release_manifest = fs::read_to_string(root.join("release-crates.json"))
        .expect("read ordered release manifest");
    let publisher = fs::read_to_string(root.join("scripts/publish_release.py"))
        .expect("read release publisher");
    let version = workspace_package_version(&workspace);

    assert!(
        changelog_has_release_state(&changelog, version),
        "CHANGELOG.md must have either a real Unreleased heading plus the exact staged-version declaration for {version}, or a real dated release heading; inline heading examples do not count"
    );
    assert_contains_all(
        "0.7 changelog compatibility truth",
        &changelog,
        &[
            "intentionally contracts the published pre-1.0",
            "does not claim source compatibility",
            "Surface::as_bytes",
            "MetalEncodedJ2k::codestream_bytes",
            "DecodeErrorClass",
        ],
    );

    assert_pattern_checks(&[
        PatternCheck::new(
            "release manifest package and API contracts",
            &release_manifest,
        )
        .required(&[
            "\"schema\": 2",
            "\"crates\"",
            "\"api_contract\"",
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
            "\"j2k-ml\"",
            "\"j2k-cli\"",
        ])
        .forbidden(&["\"j2k-compare\""]),
        PatternCheck::new("single-runner publish workflow", &publish_workflow)
            .required(&[
                "preflight:",
                "publish:",
                "environment: crates-io-publish",
                "python3 scripts/publish_release.py preflight",
                "python3 scripts/publish_release.py publish",
                "CARGO_PUBLISH_TIMEOUT: \"600\"",
            ])
            .forbidden(&["publish-j2k-core:", "cargo publish --workspace", "sleep "]),
        PatternCheck::new("ordered release manifest", &release_manifest)
            .required(&["\"crates\"", "\"j2k-core\"", "\"j2k-cli\""])
            .forbidden(&["j2k-compare"]),
        PatternCheck::new("checksum-aware release publisher", &publisher).required(&[
            "hashlib.sha256",
            "validate_release_graph",
            "validate_registry_state",
            "RETRY_DELAYS_SECONDS = (5, 15, 30)",
            "[\"cargo\", \"publish\", \"--locked\", \"-p\", crate]",
        ]),
    ]);
}

#[test]
fn release_docs_use_schema_two_manifest_and_workspace_versions() {
    let xtask = xtask_sources(repo_root());

    assert_file_pattern_checks(
        repo_root(),
        &[FilePatternCheck::new("docs/release.md")
            .named("release docs")
            .required(&[
                "Schema 2 records only ordered crate names",
                "workspace.package.version",
                "cargo metadata --locked --no-deps",
                "block v0.1.6",
                "metal v0.33.0",
                "future-incompatibility",
                "workspace resolves",
                "PATCH_PROVENANCE.md",
                "not release signoff",
                "provisional until",
                "actual intended tag date",
                "After the candidate is frozen and committed",
                "cargo xtask clippy-strict",
                "candidate verifier reads that repository setting",
                "non-production source dispositions",
                "codec-math DWT fragment",
                "vendored patched `block` FFI binding",
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

#[test]
fn typo_gate_ignores_exact_backticked_git_object_ids_without_word_allowlists() {
    let config = fs::read_to_string(repo_root().join("typos.toml")).expect("read typo config");

    assert!(
        config.contains(r"`[0-9a-f]{8}(?:[0-9a-f]{32})?`"),
        "typo config must ignore exact backticked 8- or 40-hex Git object ids"
    );
    assert!(
        !config
            .lines()
            .any(|line| line.trim_start().starts_with(concat!("b", "a ="))),
        "typo config must not hide the false positive with a word-level allowance"
    );
}

#[test]
fn no_tag_candidate_freeze_runs_offline_publish_integrity_and_package_gates() {
    let release = fs::read_to_string(repo_root().join("docs/release.md"))
        .expect("read release documentation");
    let candidate_freeze = release
        .split("## Candidate freeze and exact-SHA evidence")
        .nth(1)
        .and_then(|rest| rest.split("## Versions and publish order").next())
        .expect("candidate freeze section");

    assert_pattern_checks(&[
        PatternCheck::new("no-tag candidate freeze", candidate_freeze)
            .required(&[
                "RC_SHA=$(git rev-parse HEAD)",
                "cargo xtask release-integrity --publish",
                "cargo xtask package",
                "invalidates `RC_SHA`",
            ])
            .forbidden(&["publish-crate.sh --preflight-all", "git tag"]),
    ]);
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

fn changelog_has_release_state(changelog: &str, version: &str) -> bool {
    let markdown_lines = markdown_content_lines(changelog);
    let version_marker = format!("## [{version}");
    let version_headings = markdown_lines
        .iter()
        .copied()
        .filter(|line| line.starts_with(&version_marker))
        .collect::<Vec<_>>();
    let staged_version = format!("Staged workspace version: `{version}`.");
    let unreleased_count = markdown_lines
        .iter()
        .filter(|line| **line == "## [Unreleased]")
        .count();
    let staged_count = markdown_lines
        .iter()
        .filter(|line| **line == staged_version)
        .count();

    if version_headings.is_empty() {
        return unreleased_count == 1 && staged_count == 1;
    }
    let prefix = format!("## [{version}] - ");
    version_headings.len() == 1
        && unreleased_count == 0
        && staged_count == 0
        && version_headings[0]
            .strip_prefix(&prefix)
            .is_some_and(is_calendar_date)
}

fn is_calendar_date(date: &str) -> bool {
    if date.len() != 10
        || !date.bytes().enumerate().all(|(index, byte)| match index {
            4 | 7 => byte == b'-',
            _ => byte.is_ascii_digit(),
        })
    {
        return false;
    }
    let (Ok(year), Ok(month), Ok(day)) = (
        date[0..4].parse::<u32>(),
        date[5..7].parse::<u32>(),
        date[8..10].parse::<u32>(),
    ) else {
        return false;
    };
    let leap_year =
        year.is_multiple_of(4) && (!year.is_multiple_of(100) || year.is_multiple_of(400));
    let days_in_month = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if leap_year => 29,
        2 => 28,
        _ => return false,
    };
    year != 0 && (1..=days_in_month).contains(&day)
}

#[test]
fn changelog_release_state_requires_real_headings_and_structured_version() {
    let inline_example = "## [Unreleased]\nAt release use `## [0.7.0] - 2026-07-10`.\n";
    assert!(!changelog_has_release_state(inline_example, "0.7.0"));

    let staged = "## [Unreleased]\n\nStaged workspace version: `0.7.0`.\n";
    assert!(changelog_has_release_state(staged, "0.7.0"));

    let fenced_staged = "```markdown\n## [Unreleased]\nStaged workspace version: `0.7.0`.\n```\n";
    assert!(!changelog_has_release_state(fenced_staged, "0.7.0"));

    let commented_staged = "<!--\n## [Unreleased]\nStaged workspace version: `0.7.0`.\n-->\n";
    assert!(!changelog_has_release_state(commented_staged, "0.7.0"));

    let malformed = "## [Unreleased]\nStaged workspace version: `0.7.0`.\n## [0.7.0 - 2026-07-10\n";
    assert!(!changelog_has_release_state(malformed, "0.7.0"));

    let released = "## [0.7.0] - 2028-02-29\n";
    assert!(changelog_has_release_state(released, "0.7.0"));
    let released_with_fenced_example =
        "## [0.7.0] - 2028-02-29\n~~~markdown\n## [0.7.0] - 2026-07-10\n~~~\n";
    assert!(changelog_has_release_state(
        released_with_fenced_example,
        "0.7.0"
    ));
    for invalid in [
        "## [0.7.0] - July 10, 2026\n",
        "## [0.7.0] - 2025-02-29\n",
        "## [0.7.0] - 2026-04-31\n",
        "## [0.7.0] - 2026-13-01\n",
        "```markdown\n## [0.7.0] - 2026-07-10\n```\n",
        "    ## [0.7.0] - 2026-07-10\n",
        "<!--\n## [0.7.0] - 2026-07-10\n-->\n",
        "<pre>\n## [0.7.0] - 2026-07-10\n</pre>\n",
        "<div>\n## [0.7.0] - 2026-07-10\n</div>\n",
        "## [0.7.0] - 2026-07-10\n## [0.7.0] - 2026-07-11\n",
        "## [Unreleased]\nStaged workspace version: `0.7.0`.\n## [0.7.0] - 2026-07-10\n",
    ] {
        assert!(!changelog_has_release_state(invalid, "0.7.0"));
    }
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
    assert_package_gate_file_contracts(root);
    let package_gate = fs::read_to_string(root.join("xtask/src/release_commands/package_gate.rs"))
        .expect("read package-gate module");
    assert!(
        package_gate.lines().count() <= 175,
        "package-gate module must stay within its focused 175-line ownership ceiling"
    );
    let publishable_packages = release_manifest_entries(root)
        .into_iter()
        .map(|entry| entry.name)
        .collect::<Vec<_>>();
    let workspace_edges = cargo_metadata_workspace_edges(root);
    let staged_packages = workspace_edges
        .iter()
        .map(|(package, _)| package.clone())
        .collect::<BTreeSet<_>>();
    let strict_packages = publishable_packages
        .iter()
        .filter(|package| !staged_packages.contains(*package))
        .cloned()
        .collect::<BTreeSet<_>>();
    for package in &publishable_packages {
        let strict = strict_packages.contains(package);
        let staged = staged_packages.contains(package);
        assert_ne!(
            strict, staged,
            "publishable package `{package}` must appear in exactly one package-gate partition"
        );
    }
    assert_package_gate_dependency_order(&publishable_packages, &strict_packages, &workspace_edges);
    assert!(
        staged_packages.contains("j2k-cuda-runtime"),
        "j2k-cuda-runtime depends on staged j2k-core and must not run strict package verification before publication"
    );
    let codec_math = publishable_packages
        .iter()
        .position(|package| package == "j2k-codec-math")
        .expect("j2k-codec-math publish position");
    let cuda_runtime = publishable_packages
        .iter()
        .position(|package| package == "j2k-cuda-runtime")
        .expect("j2k-cuda-runtime publish position");
    assert!(
        codec_math < cuda_runtime,
        "j2k-codec-math must be staged before dependent j2k-cuda-runtime"
    );
}

fn assert_package_gate_file_contracts(root: &std::path::Path) {
    assert_file_pattern_checks(
        root,
        &[
            FilePatternCheck::new("xtask/src/release_commands.rs")
                .named("xtask package preflight")
                .required(&[
                    "mod package_gate;",
                    "release_manifest_contract()?",
                    "package_gate::run(&metadata, &release_manifest)",
                    "\"--list\"",
                ]),
            FilePatternCheck::new("xtask/src/release_commands/package_gate.rs")
                .named("dependency-aware package construction")
                .required(&[
                    "package_gate_plan(metadata, manifest)?",
                    "validate_release_manifest_contract(manifest, &packages, &mut validation_errors)?",
                    "release_dependencies_by_package(manifest, &packages)?",
                    "registry_independent_packages(manifest, &packages)?",
                    "dependency_closure",
                    "processed.contains(dependency_name.as_str())",
                    "--config",
                    "patch.crates-io.",
                    "\"--no-verify\"",
                    "\"--dry-run\"",
                ]),
            FilePatternCheck::new("scripts/publish-crate.sh")
                .named("publish script")
                .required(&[
                    "publish_release.py\" manifest",
                    "--field ordered-crates",
                    "--field registry-independent",
                    "if [[ \"$#\" -ne 1 ]]",
                    "--preflight-all",
                    "scripts/crates_io_version.py verify-set",
                    "scripts/crates_io_version.py state",
                    "cargo xtask release-integrity --publish",
                    "workspace_repository",
                    "normalize_repository_identity",
                    "git config --get-all remote.origin.url",
                    "git remote get-url --all origin",
                    "git ls-remote --tags origin",
                    "git show-ref --verify --quiet \"refs/tags/${expected_tag}\"",
                    "git cat-file -t \"refs/tags/${expected_tag}\"",
                    "refs/tags/${expected_tag}^{}",
                    "refs/tags/${expected_tag}^{commit}",
                    "HEAD^{commit}",
                    "git status --porcelain=v1 --untracked-files=all",
                    "require_positive_decimal \"CRATES_IO_PUBLISH_ATTEMPTS\"",
                    "require_nonnegative_decimal \"CRATES_IO_RATE_LIMIT_RETRY_SECONDS\"",
                    "require_nonnegative_decimal \"CRATES_IO_INDEX_SETTLE_SECONDS\"",
                    "cargo package -p \"$crate\" --no-verify",
                    "cargo publish -p \"$crate\" --dry-run",
                ])
                .forbidden(&["dry-run package list only", "cargo info"]),
            FilePatternCheck::new("scripts/publish_release.py")
                .named("ordered checksum-aware publisher")
                .required(&[
                    "release-crates.json",
                    "hashlib.sha256",
                    "validate_release_graph",
                    "validate_registry_state_with_retry",
                    "RETRY_DELAYS_SECONDS = (5, 15, 30)",
                    "cargo\", \"publish\", \"--locked",
                ])
                .forbidden(&["cargo publish --workspace"]),
            FilePatternCheck::new("scripts/release_manifest.py")
                .named("strict release manifest and graph validator")
                .required(&[
                    "object_pairs_hook=_reject_duplicate_fields",
                    "if type(schema) is not int or schema != 2",
                    "\"workspace_members\"",
                    "return \"crates-io\" in registries",
                    "must be workspace/path sourced",
                    "expected_requirement = f\"={versions_by_package[dependency_name]}\"",
                    "if kind == \"dev\"",
                    "if positions[dependency_name] >= positions[crate]",
                    "registry_independent",
                ]),
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
}

fn assert_package_gate_dependency_order(
    publishable_packages: &[String],
    strict_packages: &BTreeSet<String>,
    workspace_edges: &BTreeSet<(String, String)>,
) {
    for (package, dependency) in workspace_edges {
        assert!(
            !strict_packages.contains(package),
            "strict package preflight must not include `{package}` while it depends on staged workspace crate `{dependency}`"
        );
        let package_position = publishable_packages
            .iter()
            .position(|candidate| candidate == package);
        let dependency_position = publishable_packages
            .iter()
            .position(|candidate| candidate == dependency);
        if let (Some(package_position), Some(dependency_position)) =
            (package_position, dependency_position)
        {
            assert!(
                dependency_position < package_position,
                "publish package `{package}` must be processed after unpublished workspace dependency `{dependency}`"
            );
        }
    }
    assert!(
        workspace_edges.contains(&("j2k-cuda-runtime".to_string(), "j2k-codec-math".to_string())),
        "package policy regression must exercise the j2k-codec-math -> j2k-cuda-runtime edge"
    );
}

#[test]
fn release_manifest_is_schema_two_and_drives_the_legacy_preflight() {
    let root = repo_root();
    let release_manifest =
        fs::read_to_string(root.join("release-crates.json")).expect("read release manifest");
    let publish_script =
        fs::read_to_string(root.join("scripts/publish-crate.sh")).expect("read publish script");
    let entries = release_manifest_entries(root);
    let publishable_packages = entries
        .iter()
        .map(|entry| entry.name.as_str())
        .collect::<Vec<_>>();

    assert_eq!(entries.len(), 19);
    assert_eq!(
        entries
            .iter()
            .map(|entry| entry.name.as_str())
            .collect::<BTreeSet<_>>()
            .len(),
        entries.len(),
        "release manifest crate names must be unique"
    );
    assert_contains_all(
        "release manifest publishable package coverage",
        &release_manifest,
        &publishable_packages,
    );
    assert_contains_all(
        "release manifest schema and tiers",
        &release_manifest,
        &[
            "\"schema\": 2",
            "\"crates\"",
            "\"api_contract\": \"stable\"",
            "\"api_contract\": \"experimental\"",
            "\"api_contract\": \"implementation\"",
            "\"api_contract\": \"binary\"",
        ],
    );
    assert_contains_all(
        "legacy preflight consumes release manifest",
        &publish_script,
        &["publish_release.py\" manifest", "--field ordered-crates"],
    );
}

#[test]
fn publish_script_tag_proof_has_behavior_regressions() {
    let tests = fs::read_to_string(repo_root().join("scripts/tests/test_publish_script.py"))
        .expect("read publish script tests");
    assert_pattern_checks(&[
        PatternCheck::new("publish tag proof tests", &tests).required(&[
            "test_workflow_ref_cannot_replace_a_missing_git_tag",
            "test_lightweight_or_stale_annotated_tags_are_rejected",
            "test_verified_tag_must_also_match_workflow_ref",
            "test_annotated_tag_at_head_passes_tag_proof",
            "test_supported_origin_url_forms_accept_valid_remote_tag",
            "test_wrong_origin_is_rejected_before_cargo",
            "test_origin_url_rewrite_cannot_redirect_remote_proof",
            "test_missing_remote_tag_is_rejected_before_cargo",
            "test_lightweight_remote_tag_is_rejected_before_cargo",
            "test_stale_remote_tag_is_rejected_before_cargo",
            "test_origin_errors_do_not_expose_embedded_credentials",
            "test_dirty_tracked_worktree_fails_before_cargo",
            "test_untracked_worktree_fails_before_cargo",
        ]),
    ]);
}
