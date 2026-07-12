// SPDX-License-Identifier: MIT OR Apache-2.0

//! Typed CUDA-to-transcode-stage error classification policy.

use std::collections::BTreeMap;

use syn::visit::{self, Visit};
use syn::{Expr, ImplItem, Item, Pat, Path, Stmt, Type};

use super::CudaTranscodeSources;

fn last_path_identifier(path: &Path) -> Option<String> {
    path.segments
        .last()
        .map(|segment| segment.ident.to_string())
}

fn pattern_variant(pattern: &Pat) -> Option<String> {
    match pattern {
        Pat::Path(pattern) => last_path_identifier(&pattern.path),
        Pat::Struct(pattern) => last_path_identifier(&pattern.path),
        Pat::TupleStruct(pattern) => last_path_identifier(&pattern.path),
        _ => None,
    }
}

#[derive(Default)]
struct SelfVariants(Vec<String>);

impl<'ast> Visit<'ast> for SelfVariants {
    fn visit_path(&mut self, path: &'ast Path) {
        let mut segments = path.segments.iter();
        if segments
            .next()
            .is_some_and(|segment| segment.ident == "Self")
        {
            if let Some(target) = segments.next() {
                let target = target.ident.to_string();
                self.0.push(if target == "backend" {
                    "Backend".to_string()
                } else {
                    target
                });
            }
        }
        visit::visit_path(self, path);
    }
}

fn stage_mapping(sources: &CudaTranscodeSources) -> BTreeMap<String, Vec<String>> {
    let mut mappings = Vec::new();
    for source in &sources.files {
        let file = syn::parse_file(&source.production)
            .unwrap_or_else(|error| panic!("parse {}: {error}", source.relative));
        for item in file.items {
            let Item::Impl(item) = item else {
                continue;
            };
            let Some((_, trait_path, _)) = &item.trait_ else {
                continue;
            };
            let Type::Path(self_type) = item.self_ty.as_ref() else {
                continue;
            };
            if last_path_identifier(trait_path).as_deref() != Some("From")
                || last_path_identifier(&self_type.path).as_deref() != Some("TranscodeStageError")
            {
                continue;
            }
            let method = item.items.iter().find_map(|item| match item {
                ImplItem::Fn(method) if method.sig.ident == "from" => Some(method),
                _ => None,
            });
            let method = method.expect("TranscodeStageError From impl must define from");
            let expression = method
                .block
                .stmts
                .iter()
                .find_map(|statement| match statement {
                    Stmt::Expr(Expr::Match(expression), _) => Some(expression),
                    _ => None,
                });
            let expression =
                expression.expect("TranscodeStageError::from must directly match the CUDA error");
            for arm in &expression.arms {
                let source_variant = pattern_variant(&arm.pat)
                    .expect("every CUDA stage mapping arm must match a named variant");
                let mut targets = SelfVariants::default();
                targets.visit_expr(&arm.body);
                targets.0.sort();
                targets.0.dedup();
                mappings.push((source_variant, targets.0));
            }
        }
    }
    assert_eq!(
        mappings.len(),
        6,
        "exactly one exhaustive CUDA TranscodeStageError mapping"
    );
    mappings.into_iter().collect()
}

pub(super) fn assert_policy(sources: &CudaTranscodeSources) {
    let actual = stage_mapping(sources);
    let expected = BTreeMap::from([
        (
            "CudaUnavailable".to_string(),
            vec!["DeviceUnavailable".to_string()],
        ),
        (
            "UnsupportedJob".to_string(),
            vec!["Unsupported".to_string()],
        ),
        ("Kernel".to_string(), vec!["Backend".to_string()]),
        (
            "HostAllocationTooLarge".to_string(),
            vec!["MemoryCapExceeded".to_string()],
        ),
        (
            "HostAllocationFailed".to_string(),
            vec!["HostAllocationFailed".to_string()],
        ),
        ("Runtime".to_string(), vec!["Backend".to_string()]),
    ]);
    assert_eq!(
        actual, expected,
        "CUDA errors must retain typed TranscodeStageError classification"
    );
    assert!(
        sources
            .full_combined()
            .contains("allocation_failures_preserve_typed_stage_classification"),
        "typed CUDA allocation-stage mappings require a behavior regression"
    );
    assert!(
        include_str!("stage_error_mapping.rs").lines().count() < 150,
        "CUDA stage-error mapping policy must remain focused"
    );
}
