// SPDX-License-Identifier: MIT OR Apache-2.0

use std::env;
use std::fs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_TARGET_ID: AtomicU64 = AtomicU64::new(0);

#[derive(Debug)]
pub(in crate::coverage) struct CurrentBuildTarget {
    path: Option<PathBuf>,
}

impl CurrentBuildTarget {
    pub(in crate::coverage) fn create(root: &Path) -> Result<Self, String> {
        let configured = env::var_os("CARGO_LLVM_COV_TARGET_DIR")
            .map(PathBuf::from)
            .or_else(|| {
                env::var_os("CARGO_TARGET_DIR")
                    .map(PathBuf::from)
                    .map(|target| target.join("llvm-cov-target"))
            })
            .unwrap_or_else(|| PathBuf::from("target/llvm-cov-target"));
        let base = if configured.is_absolute() {
            configured
        } else {
            root.join(configured)
        };
        Self::create_in_base(&base)
    }

    pub(in crate::coverage) fn path(&self) -> Result<&Path, String> {
        self.path
            .as_deref()
            .ok_or_else(|| "current coverage build target was already consumed".to_string())
    }

    pub(super) fn cleanup(&mut self) -> Result<(), String> {
        let path = self
            .path
            .take()
            .ok_or_else(|| "current coverage build target was already consumed".to_string())?;
        if let Err(error) = fs::remove_dir_all(&path) {
            let message = format!(
                "failed to remove current coverage build target {}: {error}",
                path.display()
            );
            self.path = Some(path);
            return Err(message);
        }
        Ok(())
    }

    pub(super) fn create_in_base(base: &Path) -> Result<Self, String> {
        fs::create_dir_all(base).map_err(|error| {
            format!(
                "failed to create coverage build target base {}: {error}",
                base.display()
            )
        })?;
        for _ in 0..1_024 {
            let id = NEXT_TARGET_ID.fetch_add(1, Ordering::Relaxed);
            let path = base.join(format!(".j2k-current-coverage-{}-{id}", std::process::id()));
            match fs::create_dir(&path) {
                Ok(()) => {
                    return Ok(Self { path: Some(path) });
                }
                Err(error) if error.kind() == ErrorKind::AlreadyExists => {}
                Err(error) => {
                    return Err(format!(
                        "failed to create current coverage build target {}: {error}",
                        path.display()
                    ));
                }
            }
        }
        Err(format!(
            "failed to allocate a unique current coverage build target under {}",
            base.display()
        ))
    }
}

impl Drop for CurrentBuildTarget {
    fn drop(&mut self) {
        let Some(path) = self.path.take() else {
            return;
        };
        if let Err(error) = fs::remove_dir_all(&path) {
            eprintln!(
                "failed to remove current coverage build target {}: {error}",
                path.display()
            );
        }
    }
}
