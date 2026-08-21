// SPDX-License-Identifier: MIT OR Apache-2.0

//! Ratchet duplicated backend error vocabulary until shared taxonomy work owns it.

use std::{collections::BTreeMap, fs};

use syn::visit::{self, Visit};

use super::{repo_root, rust_sources};

const DUPLICATED_ERROR_VARIANTS: &[&str] = &[
    "HostAllocationFailed",
    "HostAllocationTooLarge",
    "UnsupportedCudaRequest",
    "UnsupportedMetalRequest",
];

fn duplicated_error_variant_inventory() -> BTreeMap<String, usize> {
    let root = repo_root();
    let mut inventory = BTreeMap::new();
    for path in rust_sources(&root.join("crates")) {
        let relative = path
            .strip_prefix(root)
            .expect("error source must be inside the repository")
            .to_string_lossy()
            .replace('\\', "/");
        if !relative.contains("/src/")
            || relative.contains("/tests/")
            || relative.contains("/test_support/")
            || path
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name == "tests.rs" || name.ends_with("_tests.rs"))
        {
            continue;
        }
        let source =
            fs::read_to_string(&path).unwrap_or_else(|error| panic!("read {relative}: {error}"));
        merge_counts(
            &mut inventory,
            duplicated_error_variants_in(&relative, &source),
        );
    }
    inventory
}

fn direct_static_rejection_inventory() -> BTreeMap<String, usize> {
    let root = repo_root();
    let mut inventory = BTreeMap::new();
    let adapter_roots = [
        "crates/j2k-cuda/src",
        "crates/j2k-metal/src",
        "crates/j2k-jpeg-cuda/src",
        "crates/j2k-jpeg-metal/src",
    ];
    for path in adapter_roots
        .into_iter()
        .flat_map(|source_root| rust_sources(&root.join(source_root)))
    {
        let relative = path
            .strip_prefix(root)
            .expect("error source must be inside the repository")
            .to_string_lossy()
            .replace('\\', "/");
        if !relative.contains("/src/")
            || relative.contains("/tests/")
            || relative.contains("/test_support/")
            || path
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| {
                    name == "error.rs" || name == "tests.rs" || name.ends_with("_tests.rs")
                })
        {
            continue;
        }
        let source =
            fs::read_to_string(&path).unwrap_or_else(|error| panic!("read {relative}: {error}"));
        let count = direct_rejection_constructions_in(&relative, &source);
        if count != 0 {
            inventory.insert(relative, count);
        }
    }
    inventory
}

fn direct_rejection_constructions_in(context: &str, source: &str) -> usize {
    let syntax =
        syn::parse_file(source).unwrap_or_else(|error| panic!("parse {context} as Rust: {error}"));
    let mut visitor = DirectRejectionVisitor::default();
    visitor.visit_file(&syntax);
    visitor.count
}

fn duplicated_error_variants_in(context: &str, source: &str) -> BTreeMap<String, usize> {
    let syntax =
        syn::parse_file(source).unwrap_or_else(|error| panic!("parse {context} as Rust: {error}"));
    let mut visitor = ErrorVariantVisitor::default();
    visitor.visit_file(&syntax);
    visitor.counts
}

fn merge_counts(target: &mut BTreeMap<String, usize>, source: BTreeMap<String, usize>) {
    for (name, count) in source {
        *target.entry(name).or_default() += count;
    }
}

#[derive(Default)]
struct ErrorVariantVisitor {
    counts: BTreeMap<String, usize>,
}

#[derive(Default)]
struct DirectRejectionVisitor {
    count: usize,
}

impl<'ast> Visit<'ast> for DirectRejectionVisitor {
    fn visit_expr_struct(&mut self, expression: &'ast syn::ExprStruct) {
        if expression.path.segments.last().is_some_and(|segment| {
            matches!(
                segment.ident.to_string().as_str(),
                "UnsupportedCudaRequest" | "UnsupportedMetalRequest"
            )
        }) {
            self.count += 1;
        }
        visit::visit_expr_struct(self, expression);
    }
}

impl<'ast> Visit<'ast> for ErrorVariantVisitor {
    fn visit_item_enum(&mut self, item: &'ast syn::ItemEnum) {
        for variant in &item.variants {
            let name = variant.ident.to_string();
            if DUPLICATED_ERROR_VARIANTS.contains(&name.as_str()) {
                *self.counts.entry(name).or_default() += 1;
            }
        }
        visit::visit_item_enum(self, item);
    }
}

#[test]
fn duplicated_error_variants_match_reviewed_migration_inventory() {
    assert_eq!(
        duplicated_error_variant_inventory(),
        BTreeMap::from([
            ("HostAllocationFailed".to_string(), 22),
            ("HostAllocationTooLarge".to_string(), 5),
            ("UnsupportedCudaRequest".to_string(), 2),
            ("UnsupportedMetalRequest".to_string(), 2),
        ]),
        "duplicated error taxonomy changed; new variants are forbidden and shared migrations must \
         lower this inventory"
    );
}

#[test]
fn production_rejections_use_typed_internal_reasons() {
    assert_eq!(
        direct_static_rejection_inventory(),
        BTreeMap::new(),
        "production adapters must construct static rejections through CapabilityRejection"
    );
}

#[test]
fn error_variant_parser_counts_declarations_not_uses() {
    let source = r"
        enum First { HostAllocationFailed { bytes: usize } }
        enum Second { HostAllocationFailed, Other }
        fn use_variant() { let _ = First::HostAllocationFailed { bytes: 1 }; }
    ";

    assert_eq!(
        duplicated_error_variants_in("fixture.rs", source),
        BTreeMap::from([("HostAllocationFailed".to_string(), 2)])
    );
}

#[test]
fn direct_rejection_parser_distinguishes_construction_from_matching() {
    let source = r"
        fn map(error: Error) -> Error {
            match error {
                Error::UnsupportedCudaRequest { reason } =>
                    Error::UnsupportedCudaRequest { reason },
                other => other,
            }
        }
    ";
    assert_eq!(direct_rejection_constructions_in("fixture.rs", source), 1);
}
