// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;

use super::super::{repo_root, rust_sources};

pub(super) struct JpegTranscodeSources {
    production: Vec<String>,
    full: Vec<String>,
}

impl JpegTranscodeSources {
    pub(super) fn read() -> Self {
        let root = repo_root();
        let source_root = root.join("crates/j2k-transcode/src/jpeg_to_htj2k");
        let mut paths = rust_sources(&source_root);
        paths.push(root.join("crates/j2k-transcode/src/jpeg_to_htj2k.rs"));
        paths.sort();
        paths.dedup();
        let mut production = Vec::new();
        let mut full = Vec::new();
        for path in paths {
            let relative = path.strip_prefix(root).unwrap_or(&path).to_string_lossy();
            let source = fs::read_to_string(&path)
                .unwrap_or_else(|error| panic!("read {relative}: {error}"));
            if !relative.ends_with("/tests.rs") {
                production.push(production_before_tests(&source).to_string());
            }
            full.push(source);
        }
        assert!(
            !production.is_empty(),
            "JPEG transcode production inventory"
        );
        Self { production, full }
    }

    pub(super) fn production(&self) -> Vec<&str> {
        self.production.iter().map(String::as_str).collect()
    }

    pub(super) fn combined(&self) -> String {
        self.production.join("\n")
    }

    pub(super) fn full_combined(&self) -> String {
        self.full.join("\n")
    }
}

fn production_before_tests(source: &str) -> &str {
    source
        .split_once("#[cfg(test)]\nmod tests")
        .map_or(source, |(production, _)| production)
}
