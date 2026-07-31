// SPDX-License-Identifier: MIT OR Apache-2.0

//! Cross-check the documented production graph against Cargo's graph.

use std::{collections::BTreeSet, fs, process::Command};

use super::repo_root;

type Edge = (String, String);

#[test]
fn architecture_dependency_graph_matches_cargo_metadata() {
    let metadata_edges = cargo_metadata_workspace_edges();
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

fn cargo_metadata_workspace_edges() -> BTreeSet<Edge> {
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
    edges
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
