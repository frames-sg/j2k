// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;
use std::path::{Path, PathBuf};

use syn::ext::IdentExt;
use syn::{Attribute, Expr, Lit, Meta};

pub(in crate::coverage) fn source_module_dir(
    current: &str,
    crate_root: bool,
) -> Result<PathBuf, String> {
    let current = Path::new(current);
    let parent = current.parent().unwrap_or_else(|| Path::new("."));
    if crate_root {
        return Ok(parent.to_path_buf());
    }
    let stem = current
        .file_stem()
        .and_then(|stem| stem.to_str())
        .ok_or_else(|| format!("Rust module path is not valid UTF-8: {}", current.display()))?;
    Ok(if matches!(stem, "lib" | "main" | "mod") {
        parent.to_path_buf()
    } else {
        parent.join(stem)
    })
}

pub(in crate::coverage) fn source_parent_dir(current: &str) -> PathBuf {
    Path::new(current)
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .to_path_buf()
}

pub(in crate::coverage) fn resolve_external_module(
    root: &Path,
    current: &str,
    module_dir: &Path,
    path_attr_dir: &Path,
    module: &syn::ItemMod,
    active: bool,
) -> Result<Option<String>, String> {
    if let Some(path) = module_path_attribute(&module.attrs)? {
        let candidate = path_attr_dir.join(path);
        if root.join(&candidate).is_file() {
            return existing_repository_source(root, &candidate).map(Some);
        }
        if active {
            return existing_repository_source(root, &candidate).map(Some);
        }
        return Ok(None);
    }
    let module_name = module.ident.unraw().to_string();
    let direct = module_dir.join(format!("{module_name}.rs"));
    let nested = module_dir.join(module_name).join("mod.rs");
    match (root.join(&direct).is_file(), root.join(&nested).is_file()) {
        (true, false) => existing_repository_source(root, &direct).map(Some),
        (false, true) => existing_repository_source(root, &nested).map(Some),
        (true, true) => Err(format!(
            "module `{}` in {current} resolves to both {} and {}",
            module.ident,
            direct.display(),
            nested.display()
        )),
        (false, false) if active => Err(format!(
            "module `{}` in {current} has no source file at {} or {}",
            module.ident,
            direct.display(),
            nested.display()
        )),
        (false, false) => Ok(None),
    }
}

pub(in crate::coverage) fn has_module_path_attribute(attrs: &[Attribute]) -> Result<bool, String> {
    module_path_attribute(attrs).map(|path| path.is_some())
}

fn module_path_attribute(attrs: &[Attribute]) -> Result<Option<PathBuf>, String> {
    let mut path = None;
    for attribute in attrs
        .iter()
        .filter(|attribute| attribute.path().is_ident("path"))
    {
        if path.is_some() {
            return Err("module has more than one path attribute".to_string());
        }
        let Meta::NameValue(value) = &attribute.meta else {
            return Err("module path attribute must be a string name-value".to_string());
        };
        let Expr::Lit(expression) = &value.value else {
            return Err("module path attribute must contain a string literal".to_string());
        };
        let Lit::Str(literal) = &expression.lit else {
            return Err("module path attribute must contain a string literal".to_string());
        };
        path = Some(PathBuf::from(literal.value()));
    }
    Ok(path)
}

pub(in crate::coverage) fn existing_repository_source(
    root: &Path,
    relative: &Path,
) -> Result<String, String> {
    let canonical_root = fs::canonicalize(root)
        .map_err(|error| format!("failed to canonicalize repository root: {error}"))?;
    let candidate = fs::canonicalize(root.join(relative)).map_err(|error| {
        format!(
            "failed to resolve Rust module source {}: {error}",
            root.join(relative).display()
        )
    })?;
    let relative = candidate.strip_prefix(&canonical_root).map_err(|_| {
        format!(
            "Rust module source {} resolves outside repository root {}",
            candidate.display(),
            canonical_root.display()
        )
    })?;
    relative
        .to_str()
        .map(|path| path.replace(std::path::MAIN_SEPARATOR, "/"))
        .ok_or_else(|| {
            format!(
                "Rust module path is not valid UTF-8: {}",
                relative.display()
            )
        })
}
