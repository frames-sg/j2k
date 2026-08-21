// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{collections::BTreeSet, fs};

use syn::{
    meta::ParseNestedMeta,
    spanned::Spanned,
    visit::{self, Visit},
    Attribute, Expr, File, ItemMod, Path, Token,
};

use super::relative_path;
use crate::repo_lint_support::{repo_root, rust_sources};

const REVIEWED_ALLOWS: &[(&str, &str)] = &[
    (
        "crates/j2k-cuda/src/batch/types.rs",
        "clippy::disallowed_methods",
    ),
    (
        "crates/j2k-jpeg-metal/src/viewport/model.rs",
        "clippy::disallowed_methods",
    ),
    (
        "crates/j2k-jpeg/src/bench_support.rs",
        "clippy::disallowed_macros",
    ),
    (
        "crates/j2k-metal/src/batch_decoder/contracts.rs",
        "clippy::disallowed_methods",
    ),
    (
        "crates/j2k-native/src/j2c/quantize.rs",
        "clippy::disallowed_macros",
    ),
    (
        "crates/j2k-transcode/src/pipeline_map.rs",
        "clippy::disallowed_macros",
    ),
    (
        "crates/j2k-ml/src/metal.rs",
        "clippy::trivially_copy_pass_by_ref",
    ),
    (
        "crates/j2k-cuda-j2k-engine/src/cuda_oxide_htj2k_encode/simt/src/main.rs",
        "clippy::manual_div_ceil",
    ),
    (
        "crates/j2k-cuda-j2k-engine/src/cuda_oxide_htj2k_encode/simt/src/main.rs",
        "clippy::too_many_arguments",
    ),
    (
        "crates/j2k-cuda-j2k-engine/src/cuda_oxide_htj2k_encode/simt/src/main.rs",
        "clippy::too_many_lines",
    ),
    (
        "crates/j2k-cuda-j2k-engine/src/cuda_oxide_j2k_encode/simt/src/main.rs",
        "clippy::manual_div_ceil",
    ),
    (
        "crates/j2k-cuda-j2k-engine/src/cuda_oxide_j2k_encode/simt/src/main.rs",
        "clippy::manual_is_multiple_of",
    ),
    (
        "crates/j2k-cuda-j2k-engine/src/cuda_oxide_j2k_encode/simt/src/main.rs",
        "clippy::too_many_arguments",
    ),
    (
        "crates/j2k-cuda-j2k-engine/src/cuda_oxide_j2k_encode/simt/src/main.rs",
        "static_mut_refs",
    ),
    (
        "crates/j2k-cuda-j2k-engine/src/cuda_oxide_j2k_idwt/simt/src/main.rs",
        "static_mut_refs",
    ),
    (
        "crates/j2k-cuda-j2k-engine/src/cuda_oxide_j2k_classic_decode/simt/src/main.rs",
        "static_mut_refs",
    ),
    (
        "crates/j2k-cuda-j2k-engine/src/cuda_oxide_j2k_ml/simt/src/main.rs",
        "clippy::too_many_arguments",
    ),
    (
        "crates/j2k-cuda-jpeg-engine/src/cuda_oxide_jpeg_encode/simt/src/main.rs",
        "clippy::cast_possible_truncation",
    ),
    (
        "crates/j2k-cuda-jpeg-engine/src/cuda_oxide_jpeg_encode/simt/src/main.rs",
        "clippy::cast_sign_loss",
    ),
    (
        "crates/j2k-cuda-jpeg-engine/src/cuda_oxide_jpeg_encode/simt/src/main.rs",
        "clippy::many_single_char_names",
    ),
    (
        "crates/j2k-cuda-jpeg-engine/src/cuda_oxide_jpeg_encode/simt/src/main.rs",
        "clippy::too_many_arguments",
    ),
    (
        "crates/j2k-cuda-build-support/src/cuda_oxide_simt_prelude.rs",
        "dead_code",
    ),
];

const NEVER_EXPECT_LINTS: &[&str] = &[
    "clippy::undocumented_unsafe_blocks",
    "clippy::uninit_assumed_init",
    "clippy::uninit_vec",
    "invalid_value",
    "unsafe_op_in_unsafe_fn",
];

#[test]
fn suppressions_stay_in_reviewed_device_generation_scopes() {
    let root = repo_root();
    let reviewed = REVIEWED_ALLOWS.iter().copied().collect::<BTreeSet<_>>();
    let mut sources = rust_sources(&root.join("crates"));
    sources.extend(rust_sources(&root.join("xtask")));
    sources.sort();

    let mut unreviewed = Vec::new();
    let mut file_expectations = Vec::new();
    let mut dangerous_expectations = Vec::new();
    let mut module_dead_code_expectations = Vec::new();
    let mut unexplained_expectations = Vec::new();
    for path in sources {
        let relative = relative_path(root, &path);
        let source =
            fs::read_to_string(&path).unwrap_or_else(|error| panic!("read {relative}: {error}"));
        for suppression in scan_suppressions(&source, &relative) {
            match suppression.action {
                SuppressionAction::Allow => {
                    let line = suppression.line;
                    assert!(
                        suppression.has_reason,
                        "reviewed source allowance {relative}:{line} must state its device-specific reason"
                    );
                    assert!(
                        !suppression.lints.is_empty(),
                        "source allowance {relative}:{line} must name at least one lint"
                    );
                    for lint in suppression.lints {
                        if !reviewed.contains(&(relative.as_str(), lint.as_str())) {
                            unreviewed.push(format!("{relative}:{line} `{lint}`"));
                        }
                    }
                }
                SuppressionAction::Expect => {
                    let line = suppression.line;
                    if !suppression.has_reason {
                        unexplained_expectations.push(format!("{relative}:{line}"));
                    }
                    for lint in NEVER_EXPECT_LINTS {
                        if suppression.lints.iter().any(|found| found == lint) {
                            dangerous_expectations.push(format!("{relative}:{line} `{lint}`"));
                        }
                    }
                    if suppression.owner == AttributeOwner::Module
                        && suppression.lints.iter().any(|lint| lint == "dead_code")
                    {
                        module_dead_code_expectations.push(format!("{relative}:{line}"));
                    }
                    if suppression.owner == AttributeOwner::File {
                        file_expectations.push(format!("{relative}:{line}"));
                    }
                }
            }
        }
    }

    assert!(
        unreviewed.is_empty(),
        "host or unreviewed source lint allowances are forbidden: {unreviewed:?}"
    );
    assert!(
        file_expectations.is_empty(),
        "file-level lint expectations hide future findings; localize them to items: {file_expectations:?}"
    );
    assert!(
        unexplained_expectations.is_empty(),
        "lint expectations must explain the preserved contract or boundary: {unexplained_expectations:?}"
    );
    assert!(
        dangerous_expectations.is_empty(),
        "memory-safety lint expectations are forbidden; fix the unsafe boundary: {dangerous_expectations:?}"
    );
    assert!(
        module_dead_code_expectations.is_empty(),
        "module-level `dead_code` expectations hide an unbounded subtree; localize expectations to the unused items: {module_dead_code_expectations:?}"
    );
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SuppressionAction {
    Allow,
    Expect,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AttributeOwner {
    File,
    Module,
    Other,
}

#[derive(Debug, Eq, PartialEq)]
struct Suppression {
    action: SuppressionAction,
    lints: Vec<String>,
    has_reason: bool,
    line: usize,
    owner: AttributeOwner,
}

#[derive(Default)]
struct SuppressionVisitor {
    suppressions: Vec<Suppression>,
}

impl SuppressionVisitor {
    fn record(&mut self, attribute: &Attribute, owner: AttributeOwner) {
        collect_suppressions(attribute, owner, &mut self.suppressions).unwrap_or_else(|error| {
            panic!(
                "parse suppression attribute at line {}: {error}",
                attribute.span().start().line
            )
        });
    }
}

impl<'ast> Visit<'ast> for SuppressionVisitor {
    fn visit_file(&mut self, file: &'ast File) {
        for attribute in &file.attrs {
            self.record(attribute, AttributeOwner::File);
        }
        for item in &file.items {
            self.visit_item(item);
        }
    }

    fn visit_item_mod(&mut self, module: &'ast ItemMod) {
        for attribute in &module.attrs {
            self.record(attribute, AttributeOwner::Module);
        }
        if let Some((_, items)) = &module.content {
            for item in items {
                self.visit_item(item);
            }
        }
    }

    fn visit_attribute(&mut self, attribute: &'ast Attribute) {
        self.record(attribute, AttributeOwner::Other);
        visit::visit_attribute(self, attribute);
    }
}

fn scan_suppressions(source: &str, context: &str) -> Vec<Suppression> {
    let file = syn::parse_file(source).unwrap_or_else(|error| panic!("parse {context}: {error}"));
    let mut visitor = SuppressionVisitor::default();
    visitor.visit_file(&file);
    visitor.suppressions
}

fn collect_suppressions(
    attribute: &Attribute,
    owner: AttributeOwner,
    suppressions: &mut Vec<Suppression>,
) -> syn::Result<()> {
    if let Some(action) = suppression_action(attribute.path()) {
        collect_lint_arguments(
            action,
            attribute.span().start().line,
            owner,
            |visitor| attribute.parse_nested_meta(visitor),
            suppressions,
        )
    } else if attribute.path().is_ident("cfg_attr") {
        let mut condition = true;
        attribute.parse_nested_meta(|meta| {
            if std::mem::replace(&mut condition, false) {
                consume_nested_meta(&meta)
            } else {
                collect_nested_attribute(&meta, owner, suppressions)
            }
        })
    } else {
        Ok(())
    }
}

fn collect_nested_attribute(
    meta: &ParseNestedMeta<'_>,
    owner: AttributeOwner,
    suppressions: &mut Vec<Suppression>,
) -> syn::Result<()> {
    if let Some(action) = suppression_action(&meta.path) {
        let line = meta.path.span().start().line;
        collect_lint_arguments(
            action,
            line,
            owner,
            |visitor| meta.parse_nested_meta(visitor),
            suppressions,
        )
    } else if meta.path.is_ident("cfg_attr") {
        let mut condition = true;
        meta.parse_nested_meta(|nested| {
            if std::mem::replace(&mut condition, false) {
                consume_nested_meta(&nested)
            } else {
                collect_nested_attribute(&nested, owner, suppressions)
            }
        })
    } else {
        consume_nested_meta(meta)
    }
}

fn collect_lint_arguments(
    action: SuppressionAction,
    line: usize,
    owner: AttributeOwner,
    parse: impl FnOnce(&mut dyn FnMut(ParseNestedMeta<'_>) -> syn::Result<()>) -> syn::Result<()>,
    suppressions: &mut Vec<Suppression>,
) -> syn::Result<()> {
    let mut lints = Vec::new();
    let mut has_reason = false;
    let mut visitor = |meta: ParseNestedMeta<'_>| {
        if meta.path.is_ident("reason") && meta.input.peek(Token![=]) {
            let _: Expr = meta.value()?.parse()?;
            has_reason = true;
        } else if !meta.input.peek(Token![=]) && !meta.input.peek(syn::token::Paren) {
            lints.push(path_text(&meta.path));
        } else {
            consume_nested_meta(&meta)?;
        }
        Ok(())
    };
    parse(&mut visitor)?;
    suppressions.push(Suppression {
        action,
        lints,
        has_reason,
        line,
        owner,
    });
    Ok(())
}

fn consume_nested_meta(meta: &ParseNestedMeta<'_>) -> syn::Result<()> {
    if meta.input.peek(Token![=]) {
        let _: Expr = meta.value()?.parse()?;
    } else if meta.input.peek(syn::token::Paren) {
        meta.parse_nested_meta(|nested| consume_nested_meta(&nested))?;
    }
    Ok(())
}

fn suppression_action(path: &Path) -> Option<SuppressionAction> {
    if path.is_ident("allow") {
        Some(SuppressionAction::Allow)
    } else if path.is_ident("expect") {
        Some(SuppressionAction::Expect)
    } else {
        None
    }
}

fn path_text(path: &Path) -> String {
    path.segments
        .iter()
        .map(|segment| segment.ident.to_string())
        .collect::<Vec<_>>()
        .join("::")
}

#[test]
fn attribute_block_captures_multiline_expect_reasons() {
    let source = r#"
#[expect(
    dead_code,
    reason = "shared target-specific fixture helpers"
)]
mod fixture;
"#;
    let suppressions = scan_suppressions(source, "multiline expectation fixture");

    assert_eq!(suppressions.len(), 1);
    assert_eq!(suppressions[0].action, SuppressionAction::Expect);
    assert_eq!(suppressions[0].lints, ["dead_code"]);
    assert!(suppressions[0].has_reason);
    assert_eq!(suppressions[0].owner, AttributeOwner::Module);
}

#[test]
fn file_expectations_cannot_use_whitespace_to_bypass_detection() {
    let direct = r#"#![ expect (
    dead_code,
    reason = "file-wide suppression"
)]"#;
    let conditional = r#"#![ cfg_attr(
    test,
    expect (dead_code, reason = "conditional file-wide suppression")
)]"#;
    let item = r#"#[ expect (dead_code, reason = "localized helper")]
fn helper() {}"#;

    for source in [direct, conditional] {
        let suppressions = scan_suppressions(source, "file expectation fixture");
        assert_eq!(suppressions.len(), 1);
        assert_eq!(suppressions[0].owner, AttributeOwner::File);
    }
    assert_eq!(
        scan_suppressions(item, "item expectation fixture")[0].owner,
        AttributeOwner::Other
    );
}

#[test]
fn module_dead_code_expectations_cannot_hide_a_subtree() {
    let direct = r#"
#[expect(
    reason = "temporary fixture module, with legacy helpers",
    dead_code,
)]
#[path = "support/fixture.rs"]
pub(crate) mod fixture;
"#;
    let conditional = r#"
#[cfg_attr(
    feature = "fixture",
    expect(dead_code, reason = "conditional helper module")
)]
mod fixture;
"#;
    let localized = r#"
#[expect(dead_code, reason = "single target-specific helper")]
fn fixture_helper() {}
"#;

    for source in [direct, conditional] {
        let suppression = scan_suppressions(source, "module expectation fixture")
            .into_iter()
            .next()
            .expect("module expectation");
        assert_eq!(suppression.lints, ["dead_code"]);
        assert_eq!(suppression.owner, AttributeOwner::Module);
    }
    assert_eq!(
        scan_suppressions(localized, "localized expectation fixture")[0].owner,
        AttributeOwner::Other
    );
}

#[test]
fn allow_lints_extract_the_exact_registered_ceiling() {
    let source = r#"#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "bounded device ABI narrowing"
)]"#;
    let suppression = scan_suppressions(source, "allow fixture")
        .into_iter()
        .next()
        .expect("allow suppression");

    assert_eq!(
        suppression.lints,
        ["clippy::cast_possible_truncation", "clippy::cast_sign_loss"]
    );
    assert!(suppression.has_reason);

    let unexplained = scan_suppressions("#[allow(dead_code)] fn helper() {}", "allow fixture");
    assert_eq!(unexplained[0].lints, ["dead_code"]);
    assert!(!unexplained[0].has_reason);
}
