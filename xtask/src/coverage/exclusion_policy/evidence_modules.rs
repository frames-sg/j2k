// SPDX-License-Identifier: MIT OR Apache-2.0

//! AST-backed resolution for evidence tests moved into external Rust modules.

use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
};

use syn::Item;

use super::{collect_evidence_symbols, enclosing_cfg_is_conditional, EvidenceSymbolMatches};
use crate::coverage::source_analysis::{
    existing_repository_source, has_module_path_attribute, resolve_external_module,
    source_module_dir, source_parent_dir,
};

pub(super) fn collect_evidence_symbols_from_file(
    root: &Path,
    relative: &Path,
    expected_name: &str,
    inherited_conditional: bool,
    matches: &mut EvidenceSymbolMatches,
) -> Result<(), String> {
    EvidenceTraversal {
        root,
        expected_name,
        matches,
        visited: BTreeSet::new(),
    }
    .collect_file(relative, inherited_conditional)
}

struct EvidenceTraversal<'a> {
    root: &'a Path,
    expected_name: &'a str,
    matches: &'a mut EvidenceSymbolMatches,
    visited: BTreeSet<PathBuf>,
}

impl EvidenceTraversal<'_> {
    fn collect_file(&mut self, relative: &Path, inherited_conditional: bool) -> Result<(), String> {
        let relative = existing_repository_source(self.root, relative)?;
        let relative = Path::new(&relative);
        if !self.visited.insert(relative.to_path_buf()) {
            return Ok(());
        }

        let source = fs::read_to_string(self.root.join(relative)).map_err(|error| {
            format!(
                "evidence source {} is unavailable: {error}",
                relative.display()
            )
        })?;
        let file = syn::parse_file(&source).map_err(|error| {
            format!(
                "failed to parse evidence source `{}`: {error}",
                relative.display()
            )
        })?;
        let file_conditional = inherited_conditional || enclosing_cfg_is_conditional(&file.attrs);
        collect_evidence_symbols(
            &file.items,
            self.expected_name,
            file_conditional,
            self.matches,
        );
        let current = relative.to_str().ok_or_else(|| {
            format!(
                "evidence source path is not valid UTF-8: {}",
                relative.display()
            )
        })?;
        let module_dir = source_module_dir(current, false)?;
        let path_attr_dir = source_parent_dir(current);
        self.collect_external_modules(
            current,
            &module_dir,
            &path_attr_dir,
            &file.items,
            file_conditional,
        )
    }

    fn collect_external_modules(
        &mut self,
        current: &str,
        module_dir: &Path,
        path_attr_dir: &Path,
        items: &[Item],
        inherited_conditional: bool,
    ) -> Result<(), String> {
        for item in items {
            let Item::Mod(module) = item else {
                continue;
            };
            let module_conditional =
                inherited_conditional || enclosing_cfg_is_conditional(&module.attrs);
            if let Some((_, nested)) = &module.content {
                if has_module_path_attribute(&module.attrs)? {
                    return Err(format!(
                        "inline module `{}` in `{current}` must not have a path attribute",
                        module.ident
                    ));
                }
                let nested_module_dir = module_dir.join(module.ident.to_string());
                self.collect_external_modules(
                    current,
                    &nested_module_dir,
                    &nested_module_dir,
                    nested,
                    module_conditional,
                )?;
            } else if let Some(module_path) = resolve_external_module(
                self.root,
                current,
                module_dir,
                path_attr_dir,
                module,
                false,
            )? {
                self.collect_file(Path::new(&module_path), module_conditional)?;
            }
        }
        Ok(())
    }
}
