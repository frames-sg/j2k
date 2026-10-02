// SPDX-License-Identifier: MIT OR Apache-2.0

//! Cross-check the documented production graph against Cargo's graph.

use std::{collections::BTreeSet, fs, process::Command};

use super::repo_root;

type Edge = (String, String);

#[derive(Debug)]
struct WorkspaceGraph {
    edges: BTreeSet<Edge>,
    publishable_packages: BTreeSet<String>,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum ForbiddenEdgeRule {
    SupportToPublicAdapter,
    RuntimeToPublicAdapter,
    TranscodeAdapterToFullCodecAdapter,
    ProductionToTestSupport,
}

impl ForbiddenEdgeRule {
    const fn description(self) -> &'static str {
        match self {
            Self::SupportToPublicAdapter => "support crate depends on a public adapter",
            Self::RuntimeToPublicAdapter => "runtime crate depends on a public adapter",
            Self::TranscodeAdapterToFullCodecAdapter => {
                "transcode adapter depends on a full public codec adapter"
            }
            Self::ProductionToTestSupport => "publishable production crate depends on test support",
        }
    }
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct ForbiddenArchitectureEdge {
    rule: ForbiddenEdgeRule,
    edge: Edge,
}

#[test]
fn architecture_dependency_graph_matches_cargo_metadata() {
    let metadata_edges = cargo_metadata_workspace_graph().edges;
    let docs = fs::read_to_string(repo_root().join("docs/architecture.md"))
        .expect("read architecture docs");
    let docs_edges = architecture_doc_dependency_edges(&docs);

    let missing = metadata_edges
        .difference(&docs_edges)
        .map(format_edge)
        .collect::<Vec<_>>();
    let extra = docs_edges
        .difference(&metadata_edges)
        .map(format_edge)
        .collect::<Vec<_>>();

    assert!(
        missing.is_empty() && extra.is_empty(),
        "docs/architecture.md crate dependency graph drifted from cargo metadata\n\
         missing from docs: {missing:#?}\n\
         not in cargo metadata: {extra:#?}"
    );
}

#[test]
fn forbidden_architecture_edges_match_reviewed_migration_inventory() {
    let violations = forbidden_architecture_edges(&cargo_metadata_workspace_graph());
    let expected = BTreeSet::new();

    assert_eq!(
        violations, expected,
        "forbidden workspace dependency edges changed; new violations must be removed, and resolved \
         inventory entries must be deleted rather than retained\n{}",
        format_forbidden_edges(&violations)
    );
}

fn cargo_metadata_workspace_graph() -> WorkspaceGraph {
    let output = Command::new("cargo")
        .args(["metadata", "--locked", "--no-deps", "--format-version=1"])
        .current_dir(repo_root())
        .output()
        .expect("run cargo metadata");
    assert!(
        output.status.success(),
        "cargo metadata failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let metadata: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("parse cargo metadata");
    let packages = metadata["packages"]
        .as_array()
        .expect("metadata packages array");
    let workspace_members = metadata["workspace_members"]
        .as_array()
        .expect("metadata workspace_members array")
        .iter()
        .map(|id| id.as_str().expect("workspace member id"))
        .collect::<BTreeSet<_>>();
    let workspace_packages = packages
        .iter()
        .filter(|package| {
            package["id"]
                .as_str()
                .is_some_and(|id| workspace_members.contains(id))
        })
        .filter_map(|package| package["name"].as_str())
        .collect::<BTreeSet<_>>();
    let publishable_packages = packages
        .iter()
        .filter(|package| {
            package["id"]
                .as_str()
                .is_some_and(|id| workspace_members.contains(id))
        })
        .filter(|package| {
            package["publish"]
                .as_array()
                .is_none_or(|registries| !registries.is_empty())
        })
        .filter_map(|package| package["name"].as_str().map(str::to_owned))
        .collect();

    let mut edges = BTreeSet::new();
    for package in packages.iter().filter(|package| {
        package["id"]
            .as_str()
            .is_some_and(|id| workspace_members.contains(id))
    }) {
        let source = package["name"].as_str().expect("package name");
        for dependency in package["dependencies"]
            .as_array()
            .expect("package dependencies array")
            .iter()
            .filter(|dependency| dependency["kind"].is_null())
            .filter(|dependency| dependency["source"].is_null())
            .filter_map(|dependency| dependency["name"].as_str())
            .filter(|dependency| workspace_packages.contains(dependency))
        {
            edges.insert((source.to_owned(), dependency.to_owned()));
        }
    }
    WorkspaceGraph {
        edges,
        publishable_packages,
    }
}

fn forbidden_architecture_edges(graph: &WorkspaceGraph) -> BTreeSet<ForbiddenArchitectureEdge> {
    graph
        .edges
        .iter()
        .filter_map(|edge| {
            forbidden_edge_rule(edge, &graph.publishable_packages).map(|rule| {
                ForbiddenArchitectureEdge {
                    rule,
                    edge: edge.clone(),
                }
            })
        })
        .collect()
}

fn forbidden_edge_rule(
    (source, dependency): &Edge,
    publishable_packages: &BTreeSet<String>,
) -> Option<ForbiddenEdgeRule> {
    if is_runtime_crate(source) && is_public_codec_adapter(dependency) {
        return Some(ForbiddenEdgeRule::RuntimeToPublicAdapter);
    }
    if is_support_crate(source) && is_public_codec_adapter(dependency) {
        return Some(ForbiddenEdgeRule::SupportToPublicAdapter);
    }
    if is_transcode_adapter(source) && is_full_codec_adapter(dependency) {
        return Some(ForbiddenEdgeRule::TranscodeAdapterToFullCodecAdapter);
    }
    if publishable_packages.contains(source) && is_test_support_crate(dependency) {
        return Some(ForbiddenEdgeRule::ProductionToTestSupport);
    }
    None
}

fn is_runtime_crate(package: &str) -> bool {
    package.ends_with("-runtime")
}

fn is_support_crate(package: &str) -> bool {
    package.ends_with("-support") && !is_test_support_crate(package)
}

fn is_test_support_crate(package: &str) -> bool {
    package.ends_with("-test-support")
}

fn is_transcode_adapter(package: &str) -> bool {
    package.starts_with("j2k-transcode-") && !is_test_support_crate(package)
}

fn is_public_codec_adapter(package: &str) -> bool {
    package == "j2k" || package.ends_with("-cuda") || package.ends_with("-metal")
}

fn is_full_codec_adapter(package: &str) -> bool {
    package
        .strip_prefix("j2k-")
        .is_some_and(|backend| matches!(backend, "cuda" | "metal"))
}

fn format_forbidden_edges(violations: &BTreeSet<ForbiddenArchitectureEdge>) -> String {
    violations
        .iter()
        .map(|violation| {
            format!(
                "{}: {}",
                violation.rule.description(),
                format_edge(&violation.edge)
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn architecture_doc_dependency_edges(docs: &str) -> BTreeSet<Edge> {
    let graph = docs
        .split("## Crate dependency graph")
        .nth(1)
        .and_then(|section| section.split("```").nth(1))
        .expect("architecture dependency graph code block");
    let mut edges = BTreeSet::new();
    for line in graph.lines().filter(|line| line.contains("->")) {
        let (source, dependencies) = line.split_once("->").expect("graph edge line");
        for dependency in dependencies.split(',') {
            let dependency = dependency
                .split_whitespace()
                .next()
                .expect("graph dependency token");
            edges.insert((source.trim().to_owned(), dependency.to_owned()));
        }
    }
    edges
}

fn format_edge((source, dependency): &Edge) -> String {
    format!("{source} -> {dependency}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn semantic_rules_reject_forbidden_dependency_directions() {
        let graph = WorkspaceGraph {
            edges: BTreeSet::from([
                ("j2k-cuda-runtime".into(), "j2k-cuda".into()),
                ("j2k-metal-support".into(), "j2k-metal".into()),
                ("j2k-transcode-metal".into(), "j2k-metal".into()),
                ("j2k".into(), "j2k-test-support".into()),
            ]),
            publishable_packages: BTreeSet::from([
                "j2k".into(),
                "j2k-cuda".into(),
                "j2k-cuda-runtime".into(),
                "j2k-metal".into(),
                "j2k-metal-support".into(),
                "j2k-transcode-metal".into(),
            ]),
        };

        let actual = forbidden_architecture_edges(&graph)
            .into_iter()
            .map(|violation| violation.edge)
            .collect::<BTreeSet<_>>();

        assert_eq!(actual, graph.edges);
    }

    #[test]
    fn semantic_rules_allow_dependencies_toward_contracts_and_engines() {
        let graph = WorkspaceGraph {
            edges: BTreeSet::from([
                ("j2k-metal".into(), "j2k-core".into()),
                ("j2k-metal-support".into(), "j2k-core".into()),
                ("j2k-transcode-metal".into(), "j2k-transcode".into()),
                ("j2k-test-support".into(), "j2k".into()),
            ]),
            publishable_packages: BTreeSet::from([
                "j2k".into(),
                "j2k-core".into(),
                "j2k-metal".into(),
                "j2k-metal-support".into(),
                "j2k-transcode".into(),
                "j2k-transcode-metal".into(),
            ]),
        };

        assert!(forbidden_architecture_edges(&graph).is_empty());
    }
}
