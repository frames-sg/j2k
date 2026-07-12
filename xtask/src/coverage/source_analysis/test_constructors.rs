// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use super::ast::analyze_source;
use super::cfg_eval::CoverageCfgContext;
use super::graph::ReachKind;
use super::workspace::{discover_manifest_fuzz_source_roots, CoverageCfgContexts, SourceRoot};
use super::{SourceFileAnalysis, SourceIndex, SourceRole};
use crate::coverage::model::CoverageLane;

impl SourceIndex {
    pub(in crate::coverage) fn single(path: &str, source: &str) -> Result<Self, String> {
        Self::single_with_custom_cfg(path, source, [])
    }

    pub(in crate::coverage) fn single_with_custom_cfg(
        path: &str,
        source: &str,
        custom_flags: impl IntoIterator<Item = (&'static str, bool)>,
    ) -> Result<Self, String> {
        let context = CoverageCfgContext::synthetic(custom_flags);
        let parsed = analyze_source(
            Path::new("."),
            path,
            source,
            ReachKind::Production,
            true,
            true,
            &context,
        )?;
        Ok(Self {
            files: BTreeMap::from([(
                path.to_string(),
                SourceFileAnalysis {
                    role: SourceRole::Production,
                    test_only_lines: parsed.test_only_lines,
                    test_only_spans: parsed.test_only_spans,
                    executable_lines: parsed.executable_lines,
                    functions: parsed.functions,
                    executable_bodies: parsed.executable_bodies,
                    opaque_macros: parsed.opaque_macros,
                },
            )]),
        })
    }

    pub(in crate::coverage) fn repository_subset(
        root: &Path,
        changed: &BTreeMap<String, BTreeSet<usize>>,
        roots: &[(&str, SourceRole)],
    ) -> Result<Self, String> {
        let package = "coverage-source-analysis-test";
        let roots = roots
            .iter()
            .map(|(path, role)| {
                let kind = match role {
                    SourceRole::Production => ReachKind::Production,
                    SourceRole::BuildScript => ReachKind::BuildScript,
                    SourceRole::TestTarget => ReachKind::TestTarget,
                    SourceRole::ExampleBenchFuzz => ReachKind::ExampleBenchFuzz,
                    other => {
                        return Err(format!(
                            "repository_subset root `{path}` has invalid role {other:?}"
                        ));
                    }
                };
                Ok(SourceRoot {
                    package: package.to_string(),
                    path: (*path).to_string(),
                    kind,
                    crate_root: *role != SourceRole::Production,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let contexts = CoverageCfgContexts::synthetic(package, CoverageCfgContext::synthetic([]));
        Self::build_from_roots(root, CoverageLane::Host, changed, &roots, &contexts)
    }

    pub(in crate::coverage) fn repository_manifest_fuzz_subset(
        root: &Path,
        changed: &BTreeMap<String, BTreeSet<usize>>,
    ) -> Result<Self, String> {
        let roots = discover_manifest_fuzz_source_roots(root, changed)?;
        let packages = roots
            .iter()
            .map(|root| root.package.clone())
            .collect::<BTreeSet<_>>();
        let contexts = CoverageCfgContexts::synthetic_packages(&packages);
        Self::build_from_roots(root, CoverageLane::Host, changed, &roots, &contexts)
    }
}
