// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use super::build_outputs::BuildOutputEvidence;
use super::compiler_regions::SourceSpan;
use super::model::CoverageLane;

mod ast;
mod audit;
mod cfg_eval;
mod graph;
mod module_resolver;
mod node_attrs;
#[cfg(test)]
mod test_constructors;
mod test_lines;
mod workspace;

use ast::analyze_source;
pub(crate) use audit::{analyze_test_only_syntax, SourceAuditSyntax, SourceAuditTestSpan};
use cfg_eval::CoverageCfgContext;
use graph::{module_reachability, ReachKind};
pub(in crate::coverage) use module_resolver::{
    existing_repository_source, has_module_path_attribute, resolve_external_module,
    source_module_dir, source_parent_dir,
};
use workspace::{
    classify_unreached_source, discover_source_roots, read_source, CoverageCfgContexts, SourceRoot,
};

pub(super) const GENERATED_DWT_DISPOSITION: &str = "generated-codec-math-fragment";
pub(super) const VENDORED_BLOCK_DISPOSITION: &str = "vendored-block-ffi-binding";
pub(super) const VENDORED_GPU_INTEROP_DISPOSITION: &str = "vendored-gpu-interop-patch";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum SourceRole {
    Production,
    BuildScript,
    TestOnly,
    TestTarget,
    ExampleBenchFuzz,
    Generated(&'static str),
    VendoredReviewed(&'static str),
}

impl SourceRole {
    pub(super) const fn disposition(self) -> &'static str {
        match self {
            Self::Production => "production",
            Self::BuildScript => "build-script",
            Self::TestOnly => "syntax-test-only",
            Self::TestTarget => "test-target",
            Self::ExampleBenchFuzz => "example-bench-fuzz",
            Self::Generated(id) | Self::VendoredReviewed(id) => id,
        }
    }

    pub(super) const fn is_measurable(self) -> bool {
        matches!(self, Self::Production | Self::BuildScript)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct FunctionSpan {
    pub(super) name: String,
    pub(super) start: usize,
    pub(super) end: usize,
    pub(super) body_start: usize,
    pub(super) body_end: usize,
    pub(super) required_on_host: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct ExecutableBodySpan {
    pub(super) label: String,
    pub(super) start: usize,
    pub(super) end: usize,
    pub(super) evidence: DeferredBodyEvidence,
    pub(super) required_on_host: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum DeferredBodyEvidence {
    DistinctLines { start: usize, end: usize },
    CompilerRegion(SourceSpan),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct TestOnlySpan {
    pub(super) start_line: usize,
    pub(super) start_column: usize,
    pub(super) end_line: usize,
    pub(super) end_column: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum TestOnlyLineDisposition {
    Production,
    TestOnly,
    Mixed,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct OpaqueMacroSpan {
    pub(super) label: String,
    pub(super) start: usize,
    pub(super) end: usize,
    pub(super) kind: OpaqueMacroKind,
    pub(super) required_on_host: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum OpaqueMacroKind {
    Definition,
    Invocation,
}

#[derive(Clone, Debug)]
pub(super) struct SourceFileAnalysis {
    pub(super) role: SourceRole,
    pub(super) test_only_lines: BTreeSet<usize>,
    pub(super) test_only_spans: Vec<TestOnlySpan>,
    pub(super) executable_lines: BTreeSet<usize>,
    pub(super) functions: Vec<FunctionSpan>,
    pub(super) executable_bodies: Vec<ExecutableBodySpan>,
    pub(super) opaque_macros: Vec<OpaqueMacroSpan>,
}

#[derive(Debug, Default)]
pub(super) struct SourceIndex {
    files: BTreeMap<String, SourceFileAnalysis>,
}

impl SourceIndex {
    pub(super) fn build(
        root: &Path,
        lane: CoverageLane,
        changed: &BTreeMap<String, BTreeSet<usize>>,
        build_output_evidence: &BuildOutputEvidence,
    ) -> Result<Self, String> {
        let (roots, contexts) = discover_source_roots(root, lane, changed, build_output_evidence)?;
        Self::build_from_roots(root, lane, changed, &roots, &contexts)
    }

    fn build_from_roots(
        root: &Path,
        lane: CoverageLane,
        changed: &BTreeMap<String, BTreeSet<usize>>,
        roots: &[SourceRoot],
        contexts: &CoverageCfgContexts,
    ) -> Result<Self, String> {
        let states = module_reachability(root, roots, contexts)?;
        let root_paths = roots
            .iter()
            .filter(|root| root.crate_root)
            .map(|root| root.path.as_str())
            .collect::<BTreeSet<_>>();
        let mut files = BTreeMap::new();
        for (path, state) in &states {
            let role = state.role();
            let analysis = if role.is_measurable() {
                let source = read_source(root, path)?;
                let kind = if role == SourceRole::BuildScript {
                    ReachKind::BuildScript
                } else {
                    ReachKind::Production
                };
                let parsed = analyze_source(
                    root,
                    path,
                    &source,
                    kind,
                    state.required_on_host(),
                    root_paths.contains(path.as_str()),
                    contexts.get(state.package()?)?,
                )?;
                SourceFileAnalysis {
                    role,
                    test_only_lines: parsed.test_only_lines,
                    test_only_spans: parsed.test_only_spans,
                    executable_lines: parsed.executable_lines,
                    functions: parsed.functions,
                    executable_bodies: parsed.executable_bodies,
                    opaque_macros: parsed.opaque_macros,
                }
            } else {
                SourceFileAnalysis {
                    role,
                    test_only_lines: BTreeSet::new(),
                    test_only_spans: Vec::new(),
                    executable_lines: BTreeSet::new(),
                    functions: Vec::new(),
                    executable_bodies: Vec::new(),
                    opaque_macros: Vec::new(),
                }
            };
            files.insert(path.clone(), analysis);
        }

        for path in changed.keys().filter(|path| lane.owns_path(path)) {
            if files.contains_key(path) {
                continue;
            }
            let role = classify_unreached_source(root, path)?;
            files.insert(
                path.clone(),
                SourceFileAnalysis {
                    role,
                    test_only_lines: BTreeSet::new(),
                    test_only_spans: Vec::new(),
                    executable_lines: BTreeSet::new(),
                    functions: Vec::new(),
                    executable_bodies: Vec::new(),
                    opaque_macros: Vec::new(),
                },
            );
        }

        Ok(Self { files })
    }

    pub(super) fn file(&self, path: &str) -> Result<&SourceFileAnalysis, String> {
        self.files
            .get(path)
            .ok_or_else(|| format!("changed Rust source `{path}` has no fail-closed source role"))
    }
}
