// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use super::support::TestRepository;
use crate::coverage::source_analysis::{SourceIndex, SourceRole};

#[test]
fn nonterminal_external_test_modules_do_not_truncate_production_files() {
    struct Case {
        root: &'static str,
        analyzed: &'static str,
        tests: &'static str,
        production_marker: &'static str,
        later_function: Option<&'static str>,
    }

    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let cases = [
        Case {
            root: "crates/j2k-cuda-runtime/src/lib.rs",
            analyzed: "crates/j2k-cuda-runtime/src/lib.rs",
            tests: "crates/j2k-cuda-runtime/src/tests.rs",
            production_marker: "pub use build_flags::transcode_kernels_built;",
            later_function: None,
        },
        Case {
            root: "crates/j2k-jpeg/src/backend/mod.rs",
            analyzed: "crates/j2k-jpeg/src/backend/mod.rs",
            tests: "crates/j2k-jpeg/src/backend/tests.rs",
            production_marker: "#[derive(Debug, Clone, Copy, PartialEq, Eq)]",
            later_function: Some("new"),
        },
        Case {
            root: "crates/j2k-native/src/j2c/encode.rs",
            analyzed: "crates/j2k-native/src/j2c/encode/single_tile.rs",
            tests: "crates/j2k-native/src/j2c/encode/single_tile/tests.rs",
            production_marker: "enum PreparedSingleTile {",
            later_function: Some("encode_impl"),
        },
    ];

    for case in cases {
        let source = fs::read_to_string(root.join(case.analyzed)).unwrap();
        let production_line = source
            .lines()
            .position(|line| line == case.production_marker)
            .map_or_else(
                || {
                    panic!(
                        "{} must contain production marker {:?}",
                        case.analyzed, case.production_marker
                    )
                },
                |index| index + 1,
            );
        let changed =
            BTreeMap::from([(case.analyzed.to_string(), BTreeSet::from([production_line]))]);
        let index =
            SourceIndex::repository_subset(&root, &changed, &[(case.root, SourceRole::Production)])
                .unwrap();
        let parent = index.file(case.analyzed).unwrap();

        assert_eq!(parent.role, SourceRole::Production, "{}", case.analyzed);
        assert!(
            !parent.test_only_lines.contains(&production_line),
            "{}:{} must remain production",
            case.analyzed,
            production_line
        );
        if let Some(function) = case.later_function {
            assert!(
                parent
                    .functions
                    .iter()
                    .any(|candidate| candidate.name == function),
                "{} must retain function {function}",
                case.analyzed
            );
        }
        assert_eq!(
            index.file(case.tests).unwrap().role,
            SourceRole::TestOnly,
            "{}",
            case.tests
        );
    }
}

#[test]
fn cfg_test_helper_trees_are_not_production_source() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let metal_helpers = [
        "crates/j2k-metal/src/compute/test_counters.rs",
        "crates/j2k-metal/src/compute/tier1_encode/test_support.rs",
        "crates/j2k-metal/src/compute/tier1_encode/test_support/gpu_pack.rs",
        "crates/j2k-metal/src/compute/tier1_encode/test_support/ordered_pack.rs",
        "crates/j2k-metal/src/compute/tier1_encode/test_support/split_cpu_pack.rs",
    ];
    let changed = metal_helpers
        .iter()
        .map(|path| ((*path).to_string(), BTreeSet::from([1])))
        .collect::<BTreeMap<_, _>>();
    let metal = SourceIndex::repository_subset(
        &root,
        &changed,
        &[("crates/j2k-metal/src/lib.rs", SourceRole::Production)],
    )
    .unwrap();
    for path in metal_helpers {
        assert_eq!(
            metal.file(path).unwrap().role,
            SourceRole::TestOnly,
            "{path}"
        );
    }

    let cuda_helper = "crates/j2k-cuda-runtime/src/context/test_kernels.rs";
    let cuda = SourceIndex::repository_subset(
        &root,
        &BTreeMap::from([(cuda_helper.to_string(), BTreeSet::from([1]))]),
        &[("crates/j2k-cuda-runtime/src/lib.rs", SourceRole::Production)],
    )
    .unwrap();
    assert_eq!(cuda.file(cuda_helper).unwrap().role, SourceRole::TestOnly);
}

#[test]
fn build_script_root_keeps_its_distinct_source_role() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let path = "crates/j2k-jpeg/build.rs";
    let changed = BTreeMap::from([(path.to_string(), BTreeSet::from([1]))]);
    let index = SourceIndex::repository_subset(&root, &changed, &[(path, SourceRole::BuildScript)])
        .unwrap();

    assert_eq!(index.file(path).unwrap().role, SourceRole::BuildScript);
}

#[test]
fn repository_owned_xtask_rust_fixtures_are_explicitly_test_only() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let fixture_paths = [
        "xtask/tests/fixtures/clone_audit/inline_test_a.rs",
        "xtask/tests/fixtures/clone_audit/inline_test_b.rs",
        "xtask/tests/fixtures/clone_audit/production_clone_a.rs",
        "xtask/tests/fixtures/clone_audit/production_clone_b.rs",
    ];
    let changed = BTreeMap::from(fixture_paths.map(|path| (path.to_string(), BTreeSet::from([1]))));
    let roots = [("xtask/src/main.rs", SourceRole::Production)];
    let index = SourceIndex::repository_subset(&root, &changed, &roots).unwrap();
    for path in fixture_paths {
        assert_eq!(index.file(path).unwrap().role, SourceRole::TestOnly);
    }
}

#[test]
fn unreachable_role_named_directories_fail_closed() {
    let repository = TestRepository::new();
    repository.write("crate/src/lib.rs", "pub fn production() {}\n");
    let orphan_paths = [
        "crate/src/tests/orphan.rs",
        "crate/src/examples/orphan.rs",
        "crate/src/benches/orphan.rs",
        "crate/src/fuzz/orphan.rs",
        "xtask/tests/fixtures/other/orphan.rs",
        "xtask/tests/fixtures/clone_audit/nested/orphan.rs",
    ];
    for path in orphan_paths {
        repository.write(path, "fn orphan() {}\n");
        let changed = BTreeMap::from([(path.to_string(), BTreeSet::from([1]))]);

        let error = SourceIndex::repository_subset(
            repository.root(),
            &changed,
            &[("crate/src/lib.rs", SourceRole::Production)],
        )
        .unwrap_err();

        assert!(
            error.contains("unreachable from Cargo metadata roots"),
            "{path}: {error}"
        );
    }
}

#[test]
fn cargo_target_roots_retain_metadata_roles() {
    let repository = TestRepository::new();
    let roots = [
        ("crate/tests/integration.rs", SourceRole::TestTarget),
        ("crate/examples/demo.rs", SourceRole::ExampleBenchFuzz),
        ("crate/benches/perf.rs", SourceRole::ExampleBenchFuzz),
    ];
    for (path, _) in roots {
        repository.write(path, "fn target_entry() {}\n");
    }
    let changed = roots
        .iter()
        .map(|(path, _)| ((*path).to_string(), BTreeSet::from([1])))
        .collect::<BTreeMap<_, _>>();

    let index = SourceIndex::repository_subset(repository.root(), &changed, &roots).unwrap();

    for (path, role) in roots {
        assert_eq!(index.file(path).unwrap().role, role, "{path}");
    }
}

#[test]
fn cargo_fuzz_manifest_only_grants_reachable_targets_the_fuzz_role() {
    let repository = TestRepository::new();
    repository.write(
        "crate/Cargo.toml",
        "[package]\nname = \"manifest-fuzz\"\nversion = \"0.0.0\"\n\
[package.metadata]\ncargo-fuzz = true\n\
[[bin]]\nname = \"target\"\npath = \"fuzz_targets/target.rs\"\n",
    );
    repository.write(
        "crate/fuzz_targets/target.rs",
        "#[path = \"helper.rs\"]\nmod helper;\nfn main() {}\n",
    );
    repository.write("crate/fuzz_targets/helper.rs", "pub fn helper() {}\n");
    repository.write("crate/src/fuzz/orphan.rs", "fn orphan() {}\n");
    let target_changes = BTreeMap::from([(
        "crate/fuzz_targets/helper.rs".to_string(),
        BTreeSet::from([1]),
    )]);

    let index =
        SourceIndex::repository_manifest_fuzz_subset(repository.root(), &target_changes).unwrap();

    assert_eq!(
        index.file("crate/fuzz_targets/target.rs").unwrap().role,
        SourceRole::ExampleBenchFuzz
    );
    assert_eq!(
        index.file("crate/fuzz_targets/helper.rs").unwrap().role,
        SourceRole::ExampleBenchFuzz
    );

    let orphan_changes =
        BTreeMap::from([("crate/src/fuzz/orphan.rs".to_string(), BTreeSet::from([1]))]);
    let error = SourceIndex::repository_manifest_fuzz_subset(repository.root(), &orphan_changes)
        .unwrap_err();
    assert!(error.contains("unreachable from Cargo metadata roots"));
}
