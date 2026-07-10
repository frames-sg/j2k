// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{collections::BTreeSet, fs, path::Path};

use super::{repo_root, rust_sources};

mod manifest;
mod source;

const REVIEWED_WILDCARD_EXPORT_FILES: &[&str] = &[
    "crates/j2k-test-support/src/jpeg_fixtures.rs",
    "crates/j2k-test-support/src/lib.rs",
];

const REVIEWED_DEVICE_INCLUDE_FILES: &[&str] = &[
    "crates/j2k-cuda-runtime/src/cuda_oxide_copy_u8/simt/src/main.rs",
    "crates/j2k-cuda-runtime/src/cuda_oxide_htj2k_decode/simt/src/main.rs",
    "crates/j2k-cuda-runtime/src/cuda_oxide_htj2k_encode/simt/src/main.rs",
    "crates/j2k-cuda-runtime/src/cuda_oxide_j2k_decode_store/simt/src/main.rs",
    "crates/j2k-cuda-runtime/src/cuda_oxide_j2k_dequantize/simt/src/main.rs",
    "crates/j2k-cuda-runtime/src/cuda_oxide_j2k_encode/simt/src/main.rs",
    "crates/j2k-cuda-runtime/src/cuda_oxide_j2k_idwt/simt/src/main.rs",
    "crates/j2k-cuda-runtime/src/cuda_oxide_jpeg_decode/simt/src/main.rs",
    "crates/j2k-cuda-runtime/src/cuda_oxide_jpeg_encode/simt/src/main.rs",
    "crates/j2k-cuda-runtime/src/cuda_oxide_transcode/simt/src/main.rs",
];

#[test]
fn production_includes_and_wildcard_exports_stay_in_reviewed_scopes() {
    let root = repo_root();
    let reviewed_includes = REVIEWED_DEVICE_INCLUDE_FILES
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let reviewed_wildcards = REVIEWED_WILDCARD_EXPORT_FILES
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let mut sources = rust_sources(&root.join("crates"));
    sources.extend(rust_sources(&root.join("xtask")));
    sources.sort();

    let mut unreviewed_includes = Vec::new();
    let mut unreviewed_wildcards = Vec::new();
    for path in sources {
        let relative = relative_path(root, &path);
        let source =
            fs::read_to_string(&path).unwrap_or_else(|error| panic!("read {relative}: {error}"));
        let lines = source.lines().collect::<Vec<_>>();
        for (line_index, line) in lines.iter().enumerate() {
            let trimmed = line.trim();
            if trimmed.starts_with("include!(")
                && (trimmed != "include!(\"../../../cuda_oxide_simt_prelude.rs\");"
                    || !reviewed_includes.contains(relative.as_str()))
            {
                unreviewed_includes.push(format!("{relative}:{}", line_index + 1));
            }

            let public_use =
                trimmed.starts_with("pub use ") || trimmed.starts_with("pub(crate) use ");
            if public_use {
                let statement = statement_block(&lines, line_index);
                if statement.contains('*') && !reviewed_wildcards.contains(relative.as_str()) {
                    unreviewed_wildcards.push(format!("{relative}:{}", line_index + 1));
                }
            }
        }
    }

    assert!(
        unreviewed_includes.is_empty(),
        "host-production include seams are forbidden: {unreviewed_includes:?}"
    );
    assert!(
        unreviewed_wildcards.is_empty(),
        "unreviewed production wildcard re-exports are forbidden: {unreviewed_wildcards:?}"
    );
}

fn statement_block(lines: &[&str], start: usize) -> String {
    let mut block = String::new();
    for line in lines.iter().skip(start).take(32) {
        block.push_str(line);
        block.push('\n');
        if line.contains(';') {
            break;
        }
    }
    block
}

#[test]
fn statement_block_captures_multiline_public_globs() {
    let lines = [
        "pub use fixtures::{",
        "    Builder,",
        "    *,",
        "};",
        "fn unrelated() {}",
    ];
    let statement = statement_block(&lines, 0);
    assert!(statement.contains('*'));
    assert!(!statement.contains("unrelated"));
}

fn relative_path(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}
