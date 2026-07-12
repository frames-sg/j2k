// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;

use crate::repo_lint_support::{repo_root, rust_sources};

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
        count <= 4,
        "too_many_arguments suppression count must not exceed the current ratchet: found {count}, expected <= 4"
    );
}

fn count_too_many_arguments_suppressions(source: &str) -> usize {
    let bytes = source.as_bytes();
    let mut count = 0usize;
    let mut index = 0usize;

    while index < bytes.len() {
        if bytes[index] != b'#' {
            index += 1;
            continue;
        }

        let bracket = match bytes.get(index + 1) {
            Some(b'[') => index + 1,
            Some(b'!') if bytes.get(index + 2) == Some(&b'[') => index + 2,
            _ => {
                index += 1;
                continue;
            }
        };

        let mut depth = 0usize;
        let mut end = bracket;
        while end < bytes.len() {
            match bytes[end] {
                b'[' => depth += 1,
                b']' => {
                    depth -= 1;
                    if depth == 0 {
                        end += 1;
                        break;
                    }
                }
                _ => {}
            }
            end += 1;
        }

        let attribute = &source[index..end.min(source.len())];
        if attribute.contains("allow") && attribute.contains("clippy::too_many_arguments") {
            count += 1;
        }
        index = end.max(index + 1);
    }

    count
}
