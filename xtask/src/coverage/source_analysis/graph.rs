// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::path::Path;

use super::ast::analyze_source;
use super::workspace::{read_source, CoverageCfgContexts, SourceRoot};
use super::SourceRole;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(super) enum ReachKind {
    Production,
    BuildScript,
    TestOnly,
    TestTarget,
    ExampleBenchFuzz,
}

#[derive(Clone, Debug, Default)]
pub(super) struct ReachState {
    package: Option<String>,
    kinds: BTreeSet<ReachKind>,
    required_kinds: BTreeSet<ReachKind>,
}

impl ReachState {
    fn record(&mut self, package: &str, kind: ReachKind, required: bool) -> Result<(), String> {
        if let Some(existing) = &self.package {
            if existing != package {
                return Err(format!(
                    "Rust source is owned by both workspace packages `{existing}` and `{package}`"
                ));
            }
        } else {
            self.package = Some(package.to_string());
        }
        self.kinds.insert(kind);
        if required {
            self.required_kinds.insert(kind);
        }
        Ok(())
    }

    pub(super) fn package(&self) -> Result<&str, String> {
        self.package
            .as_deref()
            .ok_or_else(|| "module reachability state has no package ownership".to_string())
    }

    pub(super) fn role(&self) -> SourceRole {
        if self.kinds.contains(&ReachKind::Production) {
            SourceRole::Production
        } else if self.kinds.contains(&ReachKind::BuildScript) {
            SourceRole::BuildScript
        } else if self.kinds.contains(&ReachKind::TestTarget) {
            SourceRole::TestTarget
        } else if self.kinds.contains(&ReachKind::ExampleBenchFuzz) {
            SourceRole::ExampleBenchFuzz
        } else {
            debug_assert!(self.kinds.contains(&ReachKind::TestOnly));
            SourceRole::TestOnly
        }
    }

    pub(super) fn required_on_host(&self) -> bool {
        self.required_kinds.contains(&ReachKind::Production)
            || self.required_kinds.contains(&ReachKind::BuildScript)
    }
}

pub(super) fn module_reachability(
    root: &Path,
    roots: &[SourceRoot],
    contexts: &CoverageCfgContexts,
) -> Result<BTreeMap<String, ReachState>, String> {
    let mut states = BTreeMap::<String, ReachState>::new();
    let mut pending = roots
        .iter()
        .map(|root| (root.path.clone(), root.package.clone(), root.kind, true))
        .collect::<VecDeque<_>>();
    let mut seen = BTreeSet::new();
    while let Some((path, package, kind, required)) = pending.pop_front() {
        if !seen.insert((path.clone(), package.clone(), kind, required)) {
            continue;
        }
        states
            .entry(path.clone())
            .or_default()
            .record(&package, kind, required)?;
        let source = read_source(root, &path)?;
        let analysis = analyze_source(
            root,
            &path,
            &source,
            kind,
            required,
            contexts.get(&package)?,
        )?;
        for edge in analysis.edges {
            pending.push_back((edge.path, package.clone(), edge.kind, edge.required_on_host));
        }
    }
    Ok(states)
}
