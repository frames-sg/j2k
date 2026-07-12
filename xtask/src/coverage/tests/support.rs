// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_REPOSITORY_ID: AtomicU64 = AtomicU64::new(0);

pub(super) struct TestRepository {
    root: PathBuf,
}

impl TestRepository {
    pub(super) fn new() -> Self {
        let id = NEXT_REPOSITORY_ID.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "j2k-coverage-source-analysis-{}-{id}",
            std::process::id()
        ));
        fs::create_dir_all(&root).unwrap_or_else(|error| {
            panic!(
                "create temporary coverage repository {}: {error}",
                root.display()
            )
        });
        Self { root }
    }

    pub(super) fn root(&self) -> &Path {
        &self.root
    }

    pub(super) fn write(&self, relative: &str, source: &str) {
        let path = self.root.join(relative);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap_or_else(|error| {
                panic!(
                    "create temporary source directory {}: {error}",
                    parent.display()
                )
            });
        }
        fs::write(&path, source)
            .unwrap_or_else(|error| panic!("write temporary source {}: {error}", path.display()));
    }
}

impl Drop for TestRepository {
    fn drop(&mut self) {
        if let Err(error) = fs::remove_dir_all(&self.root) {
            eprintln!(
                "failed to remove temporary coverage repository {}: {error}",
                self.root.display()
            );
        }
    }
}
