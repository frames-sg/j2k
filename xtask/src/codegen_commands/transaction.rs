// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs::{self, File, OpenOptions};
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

pub(super) static PAIR_TRANSACTION_NONCE: AtomicU64 = AtomicU64::new(0);

#[derive(Debug)]
pub(super) struct GeneratedPairEntry {
    pub(super) target: PathBuf,
    pub(super) staged: PathBuf,
    pub(super) backup: PathBuf,
    pub(super) had_original: bool,
}

pub(super) fn write_generated_pair_transactionally(
    generated: &[(&str, String)],
) -> Result<(), String> {
    let mut entries = stage_generated_entries(generated)?;

    for index in 0..entries.len() {
        match fs::symlink_metadata(&entries[index].target) {
            Ok(metadata) if metadata.file_type().is_file() => {
                if let Err(error) = fs::rename(&entries[index].target, &entries[index].backup) {
                    let rollback_errors = restore_originals(&entries[..index]);
                    let cleanup_errors = cleanup_staged_files(&entries);
                    return Err(with_cleanup_errors(
                        format!(
                            "failed to stage existing generated file {} for replacement: {error}",
                            entries[index].target.display()
                        ),
                        &[rollback_errors, cleanup_errors].concat(),
                    ));
                }
                entries[index].had_original = true;
            }
            Ok(_) => {
                let rollback_errors = restore_originals(&entries[..index]);
                let cleanup_errors = cleanup_staged_files(&entries);
                return Err(with_cleanup_errors(
                    format!(
                        "generated-file target {} exists but is not a regular file",
                        entries[index].target.display()
                    ),
                    &[rollback_errors, cleanup_errors].concat(),
                ));
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                let rollback_errors = restore_originals(&entries[..index]);
                let cleanup_errors = cleanup_staged_files(&entries);
                return Err(with_cleanup_errors(
                    format!(
                        "failed to inspect generated-file target {}: {error}",
                        entries[index].target.display()
                    ),
                    &[rollback_errors, cleanup_errors].concat(),
                ));
            }
        }
    }

    for index in 0..entries.len() {
        if let Err(error) = fs::rename(&entries[index].staged, &entries[index].target) {
            let rollback_errors = rollback_generated_pair_install(&entries, index);
            return Err(with_cleanup_errors(
                format!(
                    "failed to install staged generated file {}: {error}",
                    entries[index].target.display()
                ),
                &rollback_errors,
            ));
        }
    }

    if let Err(error) = sync_generated_directories(&entries) {
        let rollback_errors = rollback_generated_pair_install(&entries, entries.len());
        return Err(with_cleanup_errors(error, &rollback_errors));
    }

    let backup_cleanup_errors = entries
        .iter()
        .filter(|entry| entry.had_original)
        .filter_map(|entry| {
            fs::remove_file(&entry.backup).err().map(|error| {
                format!(
                    "failed to remove committed generated-file backup {}: {error}",
                    entry.backup.display()
                )
            })
        })
        .collect::<Vec<_>>();
    if !backup_cleanup_errors.is_empty() {
        return Err(format!(
            "both generated files were committed, but backup cleanup failed: {}",
            backup_cleanup_errors.join("; ")
        ));
    }
    sync_generated_directories(&entries).map_err(|error| {
        format!("both generated files were committed, but final directory sync failed: {error}")
    })
}

fn stage_generated_entries(
    generated: &[(&str, String)],
) -> Result<Vec<GeneratedPairEntry>, String> {
    if generated.len() != 2 {
        return Err(format!(
            "generated-file transaction requires exactly two files, found {}",
            generated.len()
        ));
    }
    if generated[0].0 == generated[1].0 {
        return Err("generated-file transaction paths must be distinct".to_string());
    }

    let nonce = PAIR_TRANSACTION_NONCE.fetch_add(1, Ordering::Relaxed);
    let mut entries = Vec::with_capacity(generated.len());
    for (index, (target, rendered)) in generated.iter().enumerate() {
        let target = PathBuf::from(target);
        let staged = sidecar_path(&target, nonce, index, "staged")?;
        let backup = sidecar_path(&target, nonce, index, "backup")?;
        if let Err(error) = ensure_path_absent(&backup) {
            let cleanup_errors = cleanup_staged_files(&entries);
            return Err(with_cleanup_errors(error, &cleanup_errors));
        }
        if let Err(error) = stage_generated_file(&staged, rendered) {
            let cleanup_errors = cleanup_staged_files(&entries);
            return Err(with_cleanup_errors(error, &cleanup_errors));
        }
        entries.push(GeneratedPairEntry {
            target,
            staged,
            backup,
            had_original: false,
        });
    }
    Ok(entries)
}

pub(super) fn ensure_path_absent(path: &Path) -> Result<(), String> {
    match fs::symlink_metadata(path) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(format!(
            "failed to inspect sidecar {}: {error}",
            path.display()
        )),
        Ok(_) => Err(format!(
            "refuse to overwrite existing generated-file sidecar {}",
            path.display()
        )),
    }
}

pub(super) fn sidecar_path(
    target: &Path,
    nonce: u64,
    index: usize,
    role: &str,
) -> Result<PathBuf, String> {
    let parent = target
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let file_name = target
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| {
            format!(
                "generated-file path {} has no UTF-8 file name",
                target.display()
            )
        })?;
    Ok(parent.join(format!(
        ".{file_name}.xtask-{}-{nonce}-{index}.{role}",
        std::process::id()
    )))
}

fn stage_generated_file(path: &Path, rendered: &str) -> Result<(), String> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| format!("create staged generated file {}: {error}", path.display()))?;
    let result = (|| {
        file.write_all(rendered.as_bytes())
            .map_err(|error| format!("write staged generated file {}: {error}", path.display()))?;
        file.sync_all()
            .map_err(|error| format!("sync staged generated file {}: {error}", path.display()))
    })();
    match result {
        Ok(()) => Ok(()),
        Err(primary) => match remove_file_if_present(path) {
            Ok(()) => Err(primary),
            Err(cleanup) => Err(with_cleanup_errors(primary, &[cleanup])),
        },
    }
}

fn cleanup_staged_files(entries: &[GeneratedPairEntry]) -> Vec<String> {
    entries
        .iter()
        .filter_map(|entry| remove_file_if_present(&entry.staged).err())
        .collect()
}

fn restore_originals(entries: &[GeneratedPairEntry]) -> Vec<String> {
    entries
        .iter()
        .rev()
        .filter(|entry| entry.had_original)
        .filter_map(|entry| {
            fs::rename(&entry.backup, &entry.target).err().map(|error| {
                format!(
                    "failed to restore {} from {}: {error}",
                    entry.target.display(),
                    entry.backup.display()
                )
            })
        })
        .collect()
}

pub(super) fn rollback_generated_pair_install(
    entries: &[GeneratedPairEntry],
    installed_count: usize,
) -> Vec<String> {
    let mut errors = Vec::new();
    for entry in entries[..installed_count].iter().rev() {
        if let Err(error) = remove_file_if_present(&entry.target) {
            errors.push(error);
        }
    }
    errors.extend(restore_originals(entries));
    errors.extend(cleanup_staged_files(entries));
    errors
}

pub(super) fn remove_file_if_present(path: &Path) -> Result<(), String> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(format!("remove {}: {error}", path.display())),
    }
}

fn sync_generated_directories(entries: &[GeneratedPairEntry]) -> Result<(), String> {
    let mut directories = entries
        .iter()
        .map(|entry| {
            entry
                .target
                .parent()
                .filter(|parent| !parent.as_os_str().is_empty())
                .unwrap_or_else(|| Path::new("."))
        })
        .collect::<Vec<_>>();
    directories.sort_unstable();
    directories.dedup();
    for directory in directories {
        File::open(directory)
            .and_then(|file| file.sync_all())
            .map_err(|error| {
                format!(
                    "sync generated-file directory {}: {error}",
                    directory.display()
                )
            })?;
    }
    Ok(())
}

pub(super) fn with_cleanup_errors(primary: String, cleanup_errors: &[String]) -> String {
    if cleanup_errors.is_empty() {
        primary
    } else {
        format!(
            "{primary}; rollback/cleanup failures: {}",
            cleanup_errors.join("; ")
        )
    }
}
