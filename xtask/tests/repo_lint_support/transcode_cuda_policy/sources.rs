// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;

use super::super::{repo_root, rust_sources};

pub(super) struct CudaTranscodeSources {
    pub(super) files: Vec<CudaTranscodeSource>,
}

pub(super) struct CudaTranscodeSource {
    pub(super) relative: String,
    pub(super) production: String,
    pub(super) full: String,
}

impl CudaTranscodeSources {
    pub(super) fn read() -> Self {
        let root = repo_root();
        let source_root = root.join("crates/j2k-transcode-cuda/src");
        let files = rust_sources(&source_root)
            .into_iter()
            .map(|path| {
                let relative = path
                    .strip_prefix(root)
                    .unwrap_or(&path)
                    .to_string_lossy()
                    .into_owned();
                let full = fs::read_to_string(&path)
                    .unwrap_or_else(|error| panic!("read {relative}: {error}"));
                CudaTranscodeSource {
                    relative,
                    production: production_before_tests(&full).to_string(),
                    full,
                }
            })
            .collect::<Vec<_>>();
        assert!(!files.is_empty(), "CUDA transcode source inventory");
        Self { files }
    }

    pub(super) fn sources(&self) -> Vec<&str> {
        self.files
            .iter()
            .map(|source| source.production.as_str())
            .collect()
    }

    pub(super) fn combined(&self) -> String {
        self.files
            .iter()
            .map(|source| source.production.as_str())
            .collect::<Vec<_>>()
            .join("\n")
    }

    pub(super) fn full_combined(&self) -> String {
        self.files
            .iter()
            .map(|source| source.full.as_str())
            .collect::<Vec<_>>()
            .join("\n")
    }
}

fn production_before_tests(source: &str) -> &str {
    source
        .split_once("#[cfg(test)]\nmod tests")
        .map_or(source, |(production, _)| production)
}
