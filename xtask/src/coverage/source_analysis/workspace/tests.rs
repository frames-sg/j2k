// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use serde_json::{json, Value};

use crate::coverage::build_outputs::{BuildOutputEvidence, CurrentBuildTarget};
use crate::coverage::source_analysis::cfg_eval::attributes_state;

use super::*;

static NEXT_REPOSITORY_ID: AtomicU64 = AtomicU64::new(0);

struct TestRepository {
    root: PathBuf,
}

impl TestRepository {
    fn new() -> Self {
        let id = NEXT_REPOSITORY_ID.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "j2k-coverage-workspace-{}-{id}",
            std::process::id()
        ));
        fs::create_dir_all(&root).expect("create workspace test repository");
        Self { root }
    }

    fn root(&self) -> &Path {
        &self.root
    }

    fn write(&self, relative: &str, source: &str) -> PathBuf {
        let path = self.root.join(relative);
        fs::create_dir_all(path.parent().expect("fixture parent")).expect("create fixture parent");
        fs::write(&path, source).expect("write workspace fixture");
        path
    }

    fn cargo_target(&self, relative: &str, kinds: &[&str]) -> Value {
        json!({
            "kind": kinds,
            "src_path": self.root.join(relative).display().to_string(),
        })
    }

    fn package(&self, id: &str, name: &str, targets: &[Value]) -> Value {
        json!({
            "id": id,
            "name": name,
            "manifest_path": self.root.join(format!("crates/{name}/Cargo.toml")).display().to_string(),
            "features": {"fast": []},
            "targets": targets,
        })
    }

    fn empty_build_evidence(&self) -> BuildOutputEvidence {
        let target = CurrentBuildTarget::create(&self.root).expect("create current build target");
        BuildOutputEvidence::capture(target).expect("capture empty build evidence")
    }
}

impl Drop for TestRepository {
    fn drop(&mut self) {
        if let Err(error) = fs::remove_dir_all(&self.root) {
            eprintln!("failed to remove {}: {error}", self.root.display());
        }
    }
}

#[test]
fn metadata_discovers_every_cargo_root_for_changed_workspace_members() {
    let repository = TestRepository::new();
    repository.write(
        "crates/demo/Cargo.toml",
        "[package]\nname='demo'\nversion='0.0.0'\n",
    );
    repository.write("crates/demo/src/lib.rs", "pub fn library() {}\n");
    repository.write("crates/demo/tests/api.rs", "#[test] fn api() {}\n");
    repository.write("crates/demo/examples/sample.rs", "fn main() {}\n");
    let package = repository.package(
        "path+demo#0.0.0",
        "demo",
        &[
            repository.cargo_target("crates/demo/src/lib.rs", &["lib"]),
            repository.cargo_target("crates/demo/tests/api.rs", &["test"]),
            repository.cargo_target("crates/demo/examples/sample.rs", &["example"]),
            repository.cargo_target("crates/demo/src/lib.rs", &["unsupported"]),
        ],
    );
    let metadata = json!({
        "workspace_members": ["path+demo#0.0.0"],
        "packages": [package, {"id": "external"}],
    });
    let changed = BTreeMap::from([("crates/demo/src/lib.rs".to_string(), BTreeSet::from([1]))]);

    let (roots, contexts) = metadata_roots(
        repository.root(),
        CoverageLane::Host,
        &changed,
        &metadata,
        &repository.empty_build_evidence(),
    )
    .unwrap();

    let discovered = roots
        .iter()
        .map(|root| (root.path.as_str(), root.kind))
        .collect::<BTreeSet<_>>();
    assert_eq!(
        discovered,
        BTreeSet::from([
            (
                "crates/demo/examples/sample.rs",
                ReachKind::ExampleBenchFuzz
            ),
            ("crates/demo/src/lib.rs", ReachKind::Production),
            ("crates/demo/tests/api.rs", ReachKind::TestTarget),
        ])
    );
    let feature: syn::Attribute = syn::parse_quote!(#[cfg(feature = "fast")]);
    assert!(
        attributes_state(&[feature], contexts.get("demo").unwrap())
            .unwrap()
            .active
    );
    assert!(contexts.get("external").is_err());
}

#[test]
fn missing_current_build_script_evidence_fails_closed() {
    let repository = TestRepository::new();
    repository.write(
        "crates/demo/Cargo.toml",
        "[package]\nname='demo'\nversion='0.0.0'\n",
    );
    repository.write("crates/demo/src/lib.rs", "pub fn library() {}\n");
    repository.write("crates/demo/build.rs", "fn main() {}\n");
    let package = repository.package(
        "demo-id",
        "demo",
        &[
            repository.cargo_target("crates/demo/src/lib.rs", &["lib"]),
            repository.cargo_target("crates/demo/build.rs", &["custom-build"]),
        ],
    );
    let metadata = json!({"workspace_members": ["demo-id"], "packages": [package]});
    let changed = BTreeMap::from([("crates/demo/src/lib.rs".to_string(), BTreeSet::from([1]))]);

    let error = metadata_roots(
        repository.root(),
        CoverageLane::Host,
        &changed,
        &metadata,
        &repository.empty_build_evidence(),
    )
    .unwrap_err();

    assert!(error.contains("no build-script output"), "{error}");
}

#[test]
fn cargo_metadata_shape_and_target_kinds_are_validated() {
    for (metadata, expected) in [
        (
            json!({"workspace_members": [], "packages": null}),
            "packages",
        ),
        (
            json!({"workspace_members": [7], "packages": []}),
            "workspace_members[0]",
        ),
    ] {
        let error = workspace_member_ids(&metadata)
            .and_then(|_| required_array(&metadata, "packages", "cargo metadata").map(|_| ()))
            .unwrap_err();
        assert!(error.contains(expected), "{error}");
    }

    for (kinds, expected) in [
        (json!(["custom-build", "lib"]), Some(ReachKind::BuildScript)),
        (json!(["test"]), Some(ReachKind::TestTarget)),
        (json!(["bench"]), Some(ReachKind::ExampleBenchFuzz)),
        (json!(["proc-macro"]), Some(ReachKind::Production)),
        (json!(["unknown"]), None),
    ] {
        let target = json!({"kind": kinds});
        assert_eq!(
            cargo_target_reach_kind(&target, "target").unwrap(),
            expected
        );
    }
    assert!(cargo_target_reach_kind(&json!({"kind": [1]}), "target").is_err());
}

#[test]
fn package_selection_requires_directory_boundaries() {
    assert!(repository_path_is_within(
        "crates/demo/src/lib.rs",
        "crates/demo"
    ));
    assert!(repository_path_is_within("crates/demo", "crates/demo"));
    assert!(repository_path_is_within("any/path.rs", "."));
    assert!(!repository_path_is_within(
        "crates/demo-extra/src/lib.rs",
        "crates/demo"
    ));
}

#[test]
fn unreachable_source_dispositions_are_narrow_and_syntax_checked() {
    let repository = TestRepository::new();
    for path in [
        "crates/j2k-codec-math/generated/dwt97_constants.rs",
        "third_party/block-0.1.6-patched/src/lib.rs",
        "third_party/block-0.1.6-patched/src/test_utils.rs",
        "xtask/tests/fixtures/clone_audit/fixture.rs",
        "crates/j2k-cuda-runtime/src/cuda_oxide_simt_prelude.rs",
        "crates/j2k-cuda-runtime/src/cuda_oxide_demo/simt/src/lib.rs",
        "crates/j2k-cuda-runtime/src/cuda_oxide_demo/src/main.rs",
        "src/unreviewed.rs",
    ] {
        repository.write(path, "pub fn fixture() {}\n");
    }

    assert!(matches!(
        classify_unreached_source(
            repository.root(),
            "crates/j2k-codec-math/generated/dwt97_constants.rs"
        ),
        Ok(SourceRole::Generated(GENERATED_DWT_DISPOSITION))
    ));
    assert!(matches!(
        classify_unreached_source(
            repository.root(),
            "third_party/block-0.1.6-patched/src/lib.rs"
        ),
        Ok(SourceRole::VendoredReviewed(VENDORED_BLOCK_DISPOSITION))
    ));
    for path in [
        "third_party/block-0.1.6-patched/src/test_utils.rs",
        "xtask/tests/fixtures/clone_audit/fixture.rs",
    ] {
        assert_eq!(
            classify_unreached_source(repository.root(), path).unwrap(),
            SourceRole::TestOnly
        );
    }
    for path in [
        "crates/j2k-cuda-runtime/src/cuda_oxide_simt_prelude.rs",
        "crates/j2k-cuda-runtime/src/cuda_oxide_demo/simt/src/lib.rs",
        "crates/j2k-cuda-runtime/src/cuda_oxide_demo/src/main.rs",
    ] {
        assert!(matches!(
            classify_unreached_source(repository.root(), path),
            Ok(SourceRole::Generated(_))
        ));
    }
    assert!(classify_unreached_source(repository.root(), "src/unreviewed.rs").is_err());
    assert!(read_source(repository.root(), "src/missing.rs").is_err());
}
