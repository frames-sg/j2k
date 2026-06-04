// SPDX-License-Identifier: Apache-2.0

use std::path::{Path, PathBuf};

fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn cargo_toml() -> String {
    let path = manifest_dir().join("Cargo.toml");
    std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("must read {}: {error}", path.display()))
}

fn bench_sources_under(path: &Path) -> Vec<PathBuf> {
    if !path.exists() {
        return Vec::new();
    }

    let mut sources = Vec::new();
    let mut pending = vec![path.to_path_buf()];
    while let Some(dir) = pending.pop() {
        let entries = std::fs::read_dir(&dir)
            .unwrap_or_else(|error| panic!("must read {}: {error}", dir.display()));
        for entry in entries {
            let entry = entry.unwrap_or_else(|error| {
                panic!("must enumerate {}: {error}", dir.display());
            });
            let path = entry.path();
            if path.is_dir() {
                pending.push(path);
            } else if path.extension().is_some_and(|extension| extension == "rs") {
                sources.push(path);
            }
        }
    }
    sources.sort();
    sources
}

#[test]
fn j2k_metal_has_no_legacy_criterion_bench_targets() {
    let cargo = cargo_toml();

    assert!(
        !cargo.contains("[[bench]]"),
        "signinum-j2k-metal bench targets were reset for a clean profiling redesign"
    );

    for target in ["device_upload", "compare", "encode_stages", "decode_stages"] {
        assert!(
            !cargo.contains(&format!("name = \"{target}\"")),
            "legacy signinum-j2k-metal bench target must stay removed: {target}"
        );
    }
}

#[test]
fn j2k_metal_has_no_legacy_bench_only_dev_dependencies() {
    let cargo = cargo_toml();

    for dependency in ["criterion", "signinum-j2k-compare"] {
        assert!(
            !cargo.contains(&format!("{dependency} =")),
            "legacy bench-only dev dependency must stay removed: {dependency}"
        );
    }
}

#[test]
fn j2k_metal_benches_directory_is_clean_for_redesign() {
    let sources = bench_sources_under(&manifest_dir().join("benches"));

    assert!(
        sources.is_empty(),
        "remove stale signinum-j2k-metal bench sources before adding new profiling benches: {sources:?}"
    );
}
