// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use crate::source_audit::{
    inventory_panic_macro_sites, mask_test_only_syntax, production_rust_sources,
    PanicMacroInventory, PanicMacroSite,
};

const PANIC_MACRO_BASELINE: PanicMacroInventory = PanicMacroInventory {
    panic: 0,
    unreachable: 50,
    assert: 8,
    assert_eq: 3,
    assert_ne: 0,
    debug_assert: 91,
    debug_assert_eq: 66,
    debug_assert_ne: 0,
};

pub(super) fn enforce_panic_macro_inventory(
    metadata: &str,
    selected_packages: &[String],
    repository_root: &Path,
) -> Result<PanicMacroInventory, String> {
    let (inventory, sites) =
        collect_panic_macro_inventory(metadata, selected_packages, repository_root)?;
    let violations = macro_ratchet_violations(inventory, PANIC_MACRO_BASELINE);
    if violations.is_empty() {
        Ok(inventory)
    } else {
        Err(format!(
            "production panic-macro ratchet exceeded: {}; current inventory: {inventory}; sites in exceeded categories:\n{}",
            violations.join(", "),
            format_exceeded_sites(&violations, &sites)
        ))
    }
}

fn collect_panic_macro_inventory(
    metadata: &str,
    selected_packages: &[String],
    repository_root: &Path,
) -> Result<(PanicMacroInventory, Vec<PanicMacroSite>), String> {
    let source_roots = parse_library_source_roots(metadata, selected_packages)?;
    let sources = production_rust_sources(repository_root, &source_roots)?;
    let bin_roots = source_roots
        .iter()
        .map(|root| root.join("bin"))
        .collect::<Vec<_>>();
    let mut inventory = PanicMacroInventory::default();
    let mut sites = Vec::new();
    for source_path in sources {
        if bin_roots
            .iter()
            .any(|bin_root| source_path.absolute.starts_with(bin_root))
        {
            continue;
        }
        let source = fs::read_to_string(&source_path.absolute).map_err(|error| {
            format!(
                "read panic-surface source {}: {error}",
                source_path.relative.display()
            )
        })?;
        let masked = mask_test_only_syntax(repository_root, &source_path.relative, &source)?;
        let relative = source_path.relative.to_str().ok_or_else(|| {
            format!(
                "panic-surface source path is not UTF-8: {}",
                source_path.relative.display()
            )
        })?;
        let (source_inventory, source_sites) = inventory_panic_macro_sites(relative, &masked.text)?;
        inventory = inventory.checked_add(source_inventory)?;
        sites.extend(source_sites);
    }

    Ok((inventory, sites))
}

fn format_exceeded_sites(violations: &[String], sites: &[PanicMacroSite]) -> String {
    sites
        .iter()
        .filter(|site| {
            violations
                .iter()
                .any(|violation| violation.starts_with(site.name))
        })
        .map(|site| format!("{}:{}:{}: {}", site.path, site.line, site.column, site.name))
        .collect::<Vec<_>>()
        .join("\n")
}

fn parse_library_source_roots(
    metadata: &str,
    selected_packages: &[String],
) -> Result<Vec<PathBuf>, String> {
    let metadata = serde_json::from_str::<serde_json::Value>(metadata)
        .map_err(|error| format!("parse panic-surface source metadata: {error}"))?;
    let packages = metadata
        .get("packages")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| "panic-surface source metadata has no packages array".to_string())?;
    let mut roots = BTreeSet::new();
    for selected in selected_packages {
        let package = packages
            .iter()
            .find(|package| {
                package.get("name").and_then(serde_json::Value::as_str) == Some(selected.as_str())
            })
            .ok_or_else(|| {
                format!("panic-surface source metadata omits selected package {selected}")
            })?;
        let targets = package
            .get("targets")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| format!("panic-surface package {selected} has no targets array"))?;
        let mut package_roots = BTreeSet::new();
        for target in targets {
            let crate_types = target
                .get("crate_types")
                .and_then(serde_json::Value::as_array)
                .ok_or_else(|| {
                    format!("panic-surface package {selected} target has no crate_types")
                })?;
            if !crate_types.iter().any(|crate_type| {
                crate_type
                    .as_str()
                    .is_some_and(|crate_type| super::LIBRARY_CRATE_TYPES.contains(&crate_type))
            }) {
                continue;
            }
            let source_path = target
                .get("src_path")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| {
                    format!("panic-surface library target {selected} has no src_path")
                })?;
            let source_root = Path::new(source_path).parent().ok_or_else(|| {
                format!("panic-surface library source {source_path} has no parent")
            })?;
            package_roots.insert(source_root.to_path_buf());
        }
        if package_roots.len() != 1 {
            return Err(format!(
                "panic-surface package {selected} must have exactly one library source root, found {}",
                package_roots.len()
            ));
        }
        roots.extend(package_roots);
    }
    if roots.len() != selected_packages.len() {
        return Err(
            "panic-surface selected packages do not map one-to-one to source roots".to_string(),
        );
    }
    Ok(roots.into_iter().collect())
}

fn macro_ratchet_violations(
    actual: PanicMacroInventory,
    baseline: PanicMacroInventory,
) -> Vec<String> {
    actual
        .entries()
        .into_iter()
        .zip(baseline.entries())
        .filter_map(|((name, actual), (baseline_name, baseline))| {
            debug_assert_eq!(name, baseline_name);
            (actual > baseline).then(|| format!("{name} {actual}/{baseline}"))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{format_exceeded_sites, macro_ratchet_violations, parse_library_source_roots};
    use crate::source_audit::{PanicMacroInventory, PanicMacroSite};

    #[test]
    fn source_roots_are_derived_from_selected_library_targets() {
        let metadata = serde_json::json!({
            "packages": [{
                "name": "codec",
                "targets": [
                    {"crate_types": ["lib"], "src_path": "/repo/crates/codec/src/lib.rs"},
                    {"crate_types": ["bin"], "src_path": "/repo/crates/codec/src/bin/tool.rs"}
                ]
            }]
        })
        .to_string();

        assert_eq!(
            parse_library_source_roots(&metadata, &["codec".to_string()]).unwrap(),
            [std::path::PathBuf::from("/repo/crates/codec/src")]
        );
    }

    #[test]
    fn macro_ratchet_reports_every_category_above_baseline() {
        let baseline = PanicMacroInventory {
            panic: 1,
            assert_eq: 2,
            ..PanicMacroInventory::default()
        };
        let actual = PanicMacroInventory {
            panic: 2,
            assert_eq: 4,
            debug_assert: 1,
            ..PanicMacroInventory::default()
        };

        assert_eq!(
            macro_ratchet_violations(actual, baseline),
            ["panic! 2/1", "assert_eq! 4/2", "debug_assert! 1/0"]
        );
    }

    #[test]
    fn exceeded_site_report_includes_only_actionable_categories() {
        let sites = [
            PanicMacroSite {
                name: "panic!",
                path: "crates/codec/src/decode.rs".to_string(),
                line: 17,
                column: 9,
            },
            PanicMacroSite {
                name: "debug_assert!",
                path: "crates/codec/src/plan.rs".to_string(),
                line: 23,
                column: 5,
            },
        ];

        assert_eq!(
            format_exceeded_sites(&["panic! 1/0".to_string()], &sites),
            "crates/codec/src/decode.rs:17:9: panic!"
        );
    }
}
