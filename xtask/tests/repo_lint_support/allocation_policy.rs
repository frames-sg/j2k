// SPDX-License-Identifier: MIT OR Apache-2.0

//! Enforce one codec-neutral host phase-budget implementation.

use std::{collections::BTreeMap, collections::BTreeSet, fs};

use super::{repo_root, rust_sources};

const PHASE_BUDGET_DECLARATIONS: &[(&str, &str)] = &[
    ("struct HostPhaseBudget", "struct"),
    ("enum HostPhaseBudget", "enum"),
    ("type HostPhaseBudget", "type"),
];

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct PhaseBudgetDefinition {
    relative_path: String,
    declaration: &'static str,
    count: usize,
}

fn host_phase_budget_definitions() -> BTreeSet<PhaseBudgetDefinition> {
    let root = repo_root();
    let mut definitions = BTreeSet::new();
    for path in rust_sources(&root.join("crates")) {
        let relative_path = path
            .strip_prefix(root)
            .expect("crate source must be inside the repository")
            .to_string_lossy();
        let source = fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
        definitions.extend(phase_budget_definitions_in(&relative_path, &source));
    }
    definitions
}

fn phase_budget_definitions_in(
    relative_path: &str,
    source: &str,
) -> BTreeSet<PhaseBudgetDefinition> {
    let mut counts = BTreeMap::<&'static str, usize>::new();
    for line in source
        .lines()
        .map(str::trim_start)
        .filter(|line| !line.starts_with("//"))
    {
        for &(pattern, declaration) in PHASE_BUDGET_DECLARATIONS {
            if line.contains(pattern) {
                *counts.entry(declaration).or_default() += 1;
            }
        }
    }
    counts
        .into_iter()
        .map(|(declaration, count)| PhaseBudgetDefinition {
            relative_path: relative_path.to_owned(),
            declaration,
            count,
        })
        .collect()
}

#[test]
fn host_phase_budget_has_one_shared_owner() {
    let actual = host_phase_budget_definitions();
    let expected = BTreeSet::from([definition(
        "crates/j2k-core/src/host_allocation.rs",
        "struct",
    )]);

    assert_eq!(
        actual, expected,
        "HostPhaseBudget must have exactly one codec-neutral implementation"
    );
}

#[test]
fn phase_budget_definition_parser_recognizes_supported_declaration_forms() {
    let actual = phase_budget_definitions_in(
        "fixture.rs",
        "pub struct HostPhaseBudget;\nenum HostPhaseBudget {}\ntype HostPhaseBudget = ();\n",
    );
    assert_eq!(
        actual,
        BTreeSet::from([
            definition("fixture.rs", "enum"),
            definition("fixture.rs", "struct"),
            definition("fixture.rs", "type"),
        ])
    );
}

fn definition(relative_path: &str, declaration: &'static str) -> PhaseBudgetDefinition {
    PhaseBudgetDefinition {
        relative_path: relative_path.to_owned(),
        declaration,
        count: 1,
    }
}
