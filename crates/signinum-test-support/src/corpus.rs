// SPDX-License-Identifier: Apache-2.0

//! Filesystem helpers for benchmark corpora.

use std::path::{Path, PathBuf};

/// Expands an environment variable containing platform-separated paths.
pub fn paths_from_env(env_var: &str) -> Vec<PathBuf> {
    std::env::var_os(env_var)
        .map(|paths| std::env::split_paths(&paths).collect())
        .unwrap_or_default()
}

/// Returns true when `path` has a JPEG file extension.
pub fn is_jpeg_path(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| matches!(ext.to_ascii_lowercase().as_str(), "jpg" | "jpeg"))
}

/// Collects JPEG file paths from a file or directory tree.
pub fn collect_jpeg_paths(path: &Path) -> Vec<PathBuf> {
    if path.is_file() {
        return is_jpeg_path(path)
            .then(|| path.to_path_buf())
            .into_iter()
            .collect();
    }
    if !path.is_dir() {
        return Vec::new();
    }

    let mut out = Vec::new();
    let mut stack = vec![path.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let child = entry.path();
            if child.is_dir() {
                stack.push(child);
            } else if is_jpeg_path(&child) {
                out.push(child);
            }
        }
    }
    out.sort();
    out
}
