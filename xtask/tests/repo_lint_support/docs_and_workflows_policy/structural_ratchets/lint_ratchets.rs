// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;

use crate::repo_lint_support::{repo_root, rust_sources};
use syn::{
    parse::Parser,
    punctuated::Punctuated,
    visit::{self, Visit},
    Attribute, Meta, Token,
};

fn source_before_cfg_test_module<'a>(source: &'a str, relative: &str) -> &'a str {
    source.split_once("#[cfg(test)]\nmod tests").map_or_else(
        || {
            assert!(
                !relative.ends_with("/tests.rs"),
                "{relative} is test-only and must not enter the production panic scan"
            );
            source
        },
        |(production, _)| production,
    )
}

#[test]
fn panic_hotspot_production_paths_do_not_use_unwrap_or_expect() {
    let root = repo_root();
    for relative in [
        "crates/j2k-cuda/src/encode.rs",
        "crates/j2k-jpeg/src/entropy/block.rs",
        "crates/j2k-jpeg/src/entropy/huffman.rs",
        "crates/j2k-jpeg/src/entropy/progressive.rs",
        "crates/j2k-jpeg/src/entropy/progressive/model.rs",
        "crates/j2k-jpeg/src/entropy/progressive/allocation.rs",
        "crates/j2k-jpeg/src/entropy/progressive/scan.rs",
        "crates/j2k-jpeg/src/entropy/progressive/terminal.rs",
        "crates/j2k-jpeg/src/entropy/progressive/render.rs",
        "crates/j2k-jpeg/src/entropy/sequential.rs",
    ] {
        let source = fs::read_to_string(root.join(relative))
            .unwrap_or_else(|err| panic!("read {relative}: {err}"));
        let production = source_before_cfg_test_module(&source, relative);
        for forbidden in [".unwrap(", ".expect("] {
            assert!(
                !production.contains(forbidden),
                "{relative} production path must not use panic-on-error `{forbidden}`"
            );
        }
    }
}

#[test]
fn too_many_arguments_suppressions_stay_below_current_ratchet() {
    const MAX_SUPPRESSIONS: usize = 106;
    let root = repo_root();
    let mut sources = rust_sources(&root.join("crates"));
    sources.extend(rust_sources(&root.join("xtask")));
    assert!(
        !sources.is_empty(),
        "too_many_arguments ratchet must scan Rust sources"
    );

    let mut count = 0usize;
    for path in sources {
        let source = fs::read_to_string(&path).unwrap_or_else(|err| panic!("read {path:?}: {err}"));
        count += count_too_many_arguments_suppressions(&source);
    }

    assert!(
        count <= MAX_SUPPRESSIONS,
        "too_many_arguments suppression count must not exceed the current ratchet: found {count}, expected <= {MAX_SUPPRESSIONS}"
    );
}

fn count_too_many_arguments_suppressions(source: &str) -> usize {
    #[derive(Default)]
    struct SuppressionCounter(usize);

    impl<'ast> Visit<'ast> for SuppressionCounter {
        fn visit_attribute(&mut self, attribute: &'ast Attribute) {
            if meta_suppresses_too_many_arguments(&attribute.meta) {
                self.0 += 1;
            }
            visit::visit_attribute(self, attribute);
        }
    }

    fn meta_suppresses_too_many_arguments(meta: &Meta) -> bool {
        let Meta::List(list) = meta else {
            return false;
        };
        let nested = Punctuated::<Meta, Token![,]>::parse_terminated
            .parse2(list.tokens.clone())
            .unwrap_or_default();
        if list.path.is_ident("allow") || list.path.is_ident("expect") {
            return nested.iter().any(|meta| {
                matches!(meta, Meta::Path(path) if path.segments.len() == 2
                    && path.segments[0].ident == "clippy"
                    && path.segments[1].ident == "too_many_arguments")
            });
        }
        list.path.is_ident("cfg_attr") && nested.iter().any(meta_suppresses_too_many_arguments)
    }

    let syntax = syn::parse_file(source).expect("parse Rust source for lint-suppression ratchet");
    let mut counter = SuppressionCounter::default();
    counter.visit_file(&syntax);
    counter.0
}

#[test]
fn too_many_arguments_ratchet_counts_allow_and_expect_attributes() {
    let lint = ["clippy::too_many_", "arguments"].concat();
    let source = [
        "#[",
        "allow(",
        lint.as_str(),
        ", reason = \"device ABI\")]\nfn allowed() {}\n#[",
        "expect(",
        lint.as_str(),
        ", reason = \"stable codec boundary\")]\nfn expected() {}\n#[cfg_attr(test, expect(",
        lint.as_str(),
        ", reason = \"test fixture\"))]\nfn test_expected() {}\n#[cfg_attr(feature = \"ffi\", allow(",
        lint.as_str(),
        ", reason = \"conditional ABI\"))]\nfn conditionally_allowed() {}\n",
    ]
    .concat();
    assert_eq!(count_too_many_arguments_suppressions(&source), 4);
}
