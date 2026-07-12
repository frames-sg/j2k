use std::fmt::Write as _;
use std::fs::{self, File, OpenOptions};
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::release_commands::STABLE_DOC_LIBRARY_PACKAGES;
use crate::stable_api::{
    collect_package_apis, verify_cargo_public_api_version, CARGO_PUBLIC_API_VERSION,
    HIDDEN_API_SNAPSHOT, ORDINARY_RUSTDOCFLAGS, PUBLIC_API_SNAPSHOT, PUBLIC_API_TARGET,
    PUBLIC_API_TOOLCHAIN,
};

const CODEC_MATH_DWT97_METAL_FRAGMENT: &str =
    "crates/j2k-codec-math/generated/dwt97_constants.metal";
const CODEC_MATH_DWT97_RUST_FRAGMENT: &str = "crates/j2k-codec-math/generated/dwt97_constants.rs";
static SNAPSHOT_TRANSACTION_NONCE: AtomicU64 = AtomicU64::new(0);

pub(super) fn stable_api(args: impl Iterator<Item = String>) -> Result<(), String> {
    let mut write_snapshot = false;
    for arg in args {
        match arg.as_str() {
            "--write" => write_snapshot = true,
            "--help" | "-h" => {
                print_stable_api_help();
                return Ok(());
            }
            other => return Err(format!("unknown stable-api argument `{other}`")),
        }
    }

    let (public_api, implementation_api) = render_stable_api_snapshots()?;
    let snapshots = [
        (PUBLIC_API_SNAPSHOT, public_api),
        (HIDDEN_API_SNAPSHOT, implementation_api),
    ];
    if write_snapshot {
        return write_snapshot_pair_transactionally(&snapshots);
    }

    let mut stale = Vec::new();
    for (path, rendered) in &snapshots {
        let committed =
            fs::read_to_string(path).map_err(|err| format!("failed to read {path}: {err}"))?;
        if committed != *rendered {
            stale.push(*path);
        }
    }
    if stale.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "stable API snapshots are stale: {}; run `cargo xtask stable-api --write` and review both API inventories",
            stale.join(", ")
        ))
    }
}

pub(super) fn codec_math_codegen(args: impl Iterator<Item = String>) -> Result<(), String> {
    let mut write_fragments = false;
    for arg in args {
        match arg.as_str() {
            "--write" => write_fragments = true,
            "--help" | "-h" => {
                print_codec_math_codegen_help();
                return Ok(());
            }
            other => return Err(format!("unknown codec-math-codegen argument `{other}`")),
        }
    }

    let fragments = [
        (
            CODEC_MATH_DWT97_METAL_FRAGMENT,
            render_codec_math_dwt97_metal_fragment(),
        ),
        (
            CODEC_MATH_DWT97_RUST_FRAGMENT,
            render_codec_math_dwt97_rust_fragment(),
        ),
    ];

    if write_fragments {
        for (path, rendered) in fragments {
            let path = Path::new(path);
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent)
                    .map_err(|err| format!("failed to create {}: {err}", parent.display()))?;
            }
            fs::write(path, rendered)
                .map_err(|err| format!("failed to write {}: {err}", path.display()))?;
        }
        return Ok(());
    }

    let mut stale = Vec::new();
    for (path, rendered) in fragments {
        let committed =
            fs::read_to_string(path).map_err(|err| format!("failed to read {path}: {err}"))?;
        if committed != rendered {
            stale.push(path);
        }
    }
    if stale.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "codec math generated fragments are stale: {}; run `cargo xtask codec-math-codegen --write` and review the diff",
            stale.join(", ")
        ))
    }
}

fn render_codec_math_dwt97_metal_fragment() -> String {
    use j2k_codec_math::dwt;

    [
        "// Generated from crates/j2k-codec-math/src/dwt.rs.".to_string(),
        format!(
            "constant float CODEC_MATH_DWT97_ALPHA = {}f;",
            compact_f32(dwt::DWT97_ALPHA_F32)
        ),
        format!(
            "constant float CODEC_MATH_DWT97_BETA = {}f;",
            compact_f32(dwt::DWT97_BETA_F32)
        ),
        format!(
            "constant float CODEC_MATH_DWT97_GAMMA = {}f;",
            compact_f32(dwt::DWT97_GAMMA_F32)
        ),
        format!(
            "constant float CODEC_MATH_DWT97_DELTA = {}f;",
            compact_f32(dwt::DWT97_DELTA_F32)
        ),
        format!(
            "constant float CODEC_MATH_DWT97_KAPPA = {}f;",
            compact_f32(dwt::DWT97_KAPPA_F32)
        ),
        "constant float CODEC_MATH_DWT97_INV_KAPPA = 1.0f / CODEC_MATH_DWT97_KAPPA;".to_string(),
        format!(
            "constant float CODEC_MATH_IDWT97_NEG_ALPHA = {}f;",
            compact_f32(dwt::IDWT97_NEG_ALPHA_F32)
        ),
        format!(
            "constant float CODEC_MATH_IDWT97_NEG_BETA = {}f;",
            compact_f32(dwt::IDWT97_NEG_BETA_F32)
        ),
        format!(
            "constant float CODEC_MATH_IDWT97_NEG_GAMMA = {}f;",
            compact_f32(dwt::IDWT97_NEG_GAMMA_F32)
        ),
        format!(
            "constant float CODEC_MATH_IDWT97_NEG_DELTA = {}f;",
            compact_f32(dwt::IDWT97_NEG_DELTA_F32)
        ),
    ]
    .join("\n")
        + "\n"
}

fn render_codec_math_dwt97_rust_fragment() -> String {
    use j2k_codec_math::dwt;

    [
        "// Generated from crates/j2k-codec-math/src/dwt.rs.".to_string(),
        format!(
            "pub const CODEC_MATH_DWT97_ALPHA: f32 = {};",
            compact_f32(dwt::DWT97_ALPHA_F32)
        ),
        format!(
            "pub const CODEC_MATH_DWT97_BETA: f32 = {};",
            compact_f32(dwt::DWT97_BETA_F32)
        ),
        format!(
            "pub const CODEC_MATH_DWT97_GAMMA: f32 = {};",
            compact_f32(dwt::DWT97_GAMMA_F32)
        ),
        format!(
            "pub const CODEC_MATH_DWT97_DELTA: f32 = {};",
            compact_f32(dwt::DWT97_DELTA_F32)
        ),
        format!(
            "pub const CODEC_MATH_DWT97_KAPPA: f32 = {};",
            compact_f32(dwt::DWT97_KAPPA_F32)
        ),
        "pub const CODEC_MATH_DWT97_INV_KAPPA: f32 = 1.0 / CODEC_MATH_DWT97_KAPPA;".to_string(),
        format!(
            "pub const CODEC_MATH_IDWT97_NEG_ALPHA: f32 = {};",
            compact_f32(dwt::IDWT97_NEG_ALPHA_F32)
        ),
        format!(
            "pub const CODEC_MATH_IDWT97_NEG_BETA: f32 = {};",
            compact_f32(dwt::IDWT97_NEG_BETA_F32)
        ),
        format!(
            "pub const CODEC_MATH_IDWT97_NEG_GAMMA: f32 = {};",
            compact_f32(dwt::IDWT97_NEG_GAMMA_F32)
        ),
        format!(
            "pub const CODEC_MATH_IDWT97_NEG_DELTA: f32 = {};",
            compact_f32(dwt::IDWT97_NEG_DELTA_F32)
        ),
    ]
    .join("\n")
        + "\n"
}

fn compact_f32(value: f32) -> String {
    format!("{value:?}")
}

fn render_stable_api_snapshots() -> Result<(String, String), String> {
    if !cfg!(target_os = "macos") {
        return Err(
            "stable-api snapshot must be generated on macOS so target-gated Metal APIs are included"
                .to_string(),
        );
    }
    verify_cargo_public_api_version()?;
    let inventories = collect_package_apis(STABLE_DOC_LIBRARY_PACKAGES)?;
    let tool_version = format!("cargo-public-api {CARGO_PUBLIC_API_VERSION}");

    let mut public_out = String::new();
    writeln!(
        &mut public_out,
        "# J2K 1.0 Public API Snapshot\n\n\
         This file is generated by `cargo xtask stable-api --write` from \
         `RUSTDOCFLAGS='{ORDINARY_RUSTDOCFLAGS}' rustup run \
         {PUBLIC_API_TOOLCHAIN} cargo \
         public-api -p <package> --all-features -sss --color never \
         --target {PUBLIC_API_TARGET}`.\n\
         It is generated on macOS for the pinned target so target-gated Metal \
         APIs are included.\n\n\
         Generator: `{tool_version}`.\n\n\
         Rustdoc toolchain: `{PUBLIC_API_TOOLCHAIN}`.\n\
         Target: `{PUBLIC_API_TARGET}`.\n\n\
         It is the item-level companion to `docs/stable-api-1.0.md`: every \
         public module, type, trait, function, method, constant, variant, and \
         field reported here is semver-visible unless moved private before 1.0. \
         Rustdoc-hidden items are tracked separately in `{HIDDEN_API_SNAPSHOT}`.\n"
    )
    .unwrap();

    let mut implementation_out = String::new();
    writeln!(
        &mut implementation_out,
        "# J2K 1.0 Rustdoc-Hidden Public API Snapshot\n\n\
         This file is generated by `cargo xtask stable-api --write`. For each \
         package it records the conservative additional reachable inventory \
         formed from the union of the ordinary and hidden-enabled passes. \
         Rustdoc may rewrite equivalent re-export paths when hidden modules \
         become visible, so rewritten variants remain reviewable here. The \
         hidden-enabled pass uses \
         `RUSTDOCFLAGS='-D warnings --document-hidden-items' rustup run \
         {PUBLIC_API_TOOLCHAIN} cargo public-api -p <package> --all-features -sss \
         --color never --target {PUBLIC_API_TARGET}`. The ordinary public \
         inventory remains in `{PUBLIC_API_SNAPSHOT}` so its \
         comparison with the 0.6.2 baseline keeps the same generator scope.\n\n\
         The published 0.6.2 artifact did not record a hidden-enabled pass; \
         this companion is staged-current inventory, not a reconstructed \
         historical hidden-API baseline.\n\n\
         Rustdoc-hidden implementation adapters are still public Rust API. \
         They must be reviewed explicitly and must not become a compatibility \
         escape hatch.\n\n\
         Generator: `{tool_version}`.\n\n\
         Rustdoc toolchain: `{PUBLIC_API_TOOLCHAIN}`.\n\
         Target: `{PUBLIC_API_TARGET}`.\n"
    )
    .unwrap();

    for package in STABLE_DOC_LIBRARY_PACKAGES {
        let inventory = inventories.get(*package).ok_or_else(|| {
            format!("collected public API inventory is missing package `{package}`")
        })?;

        writeln!(&mut public_out, "## `{package}`\n\n```text").unwrap();
        for item in &inventory.ordinary {
            writeln!(&mut public_out, "{item}").unwrap();
        }
        writeln!(&mut public_out, "```\n").unwrap();
        writeln!(&mut implementation_out, "## `{package}`\n\n```text").unwrap();
        for item in &inventory.hidden {
            writeln!(&mut implementation_out, "{item}").unwrap();
        }
        writeln!(&mut implementation_out, "```\n").unwrap();
    }

    writeln!(
        &mut public_out,
        "## `j2k-cli`\n\n\
         `j2k-cli` is a binary package. Its stable command, stdout/stderr, \
         and exit-code contract is documented in `docs/stable-api-1.0.md`.\n"
    )
    .unwrap();

    Ok((
        finalize_text_snapshot(&public_out),
        finalize_text_snapshot(&implementation_out),
    ))
}

fn finalize_text_snapshot(snapshot: &str) -> String {
    let content = snapshot.trim_end_matches('\n');
    if content.is_empty() {
        String::new()
    } else {
        format!("{content}\n")
    }
}

#[derive(Debug)]
struct SnapshotTransactionEntry {
    target: PathBuf,
    staged: PathBuf,
    backup: PathBuf,
    had_original: bool,
}

fn write_snapshot_pair_transactionally(snapshots: &[(&str, String)]) -> Result<(), String> {
    let mut entries = stage_snapshot_entries(snapshots)?;

    for index in 0..entries.len() {
        match fs::symlink_metadata(&entries[index].target) {
            Ok(metadata) if metadata.file_type().is_file() => {
                if let Err(error) = fs::rename(&entries[index].target, &entries[index].backup) {
                    let rollback_errors = restore_originals(&entries[..index]);
                    let cleanup_errors = cleanup_staged_files(&entries);
                    return Err(with_cleanup_errors(
                        format!(
                            "failed to stage existing snapshot {} for replacement: {error}",
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
                        "snapshot target {} exists but is not a regular file",
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
                        "failed to inspect snapshot target {}: {error}",
                        entries[index].target.display()
                    ),
                    &[rollback_errors, cleanup_errors].concat(),
                ));
            }
        }
    }

    for index in 0..entries.len() {
        if let Err(error) = fs::rename(&entries[index].staged, &entries[index].target) {
            let rollback_errors = rollback_snapshot_install(&entries, index);
            return Err(with_cleanup_errors(
                format!(
                    "failed to install staged snapshot {}: {error}",
                    entries[index].target.display()
                ),
                &rollback_errors,
            ));
        }
    }

    if let Err(error) = sync_snapshot_directories(&entries) {
        let rollback_errors = rollback_snapshot_install(&entries, entries.len());
        return Err(with_cleanup_errors(error, &rollback_errors));
    }

    let backup_cleanup_errors = entries
        .iter()
        .filter(|entry| entry.had_original)
        .filter_map(|entry| {
            fs::remove_file(&entry.backup).err().map(|error| {
                format!(
                    "failed to remove committed snapshot backup {}: {error}",
                    entry.backup.display()
                )
            })
        })
        .collect::<Vec<_>>();
    if !backup_cleanup_errors.is_empty() {
        return Err(format!(
            "both stable API snapshots were committed, but backup cleanup failed: {}",
            backup_cleanup_errors.join("; ")
        ));
    }
    sync_snapshot_directories(&entries).map_err(|error| {
        format!(
            "both stable API snapshots were committed, but final directory sync failed: {error}"
        )
    })
}

fn stage_snapshot_entries(
    snapshots: &[(&str, String)],
) -> Result<Vec<SnapshotTransactionEntry>, String> {
    if snapshots.len() != 2 {
        return Err(format!(
            "stable API snapshot transaction requires exactly two files, found {}",
            snapshots.len()
        ));
    }
    if snapshots[0].0 == snapshots[1].0 {
        return Err("stable API snapshot transaction paths must be distinct".to_string());
    }

    let nonce = SNAPSHOT_TRANSACTION_NONCE.fetch_add(1, Ordering::Relaxed);
    let mut entries = Vec::with_capacity(snapshots.len());
    for (index, (target, rendered)) in snapshots.iter().enumerate() {
        let target = PathBuf::from(target);
        let staged = snapshot_sidecar_path(&target, nonce, index, "staged")?;
        let backup = snapshot_sidecar_path(&target, nonce, index, "backup")?;
        if let Err(error) = ensure_path_absent(&backup) {
            let cleanup_errors = cleanup_staged_files(&entries);
            return Err(with_cleanup_errors(error, &cleanup_errors));
        }
        if let Err(error) = stage_snapshot(&staged, rendered) {
            let cleanup_errors = cleanup_staged_files(&entries);
            return Err(with_cleanup_errors(error, &cleanup_errors));
        }
        entries.push(SnapshotTransactionEntry {
            target,
            staged,
            backup,
            had_original: false,
        });
    }
    Ok(entries)
}

fn ensure_path_absent(path: &Path) -> Result<(), String> {
    match fs::symlink_metadata(path) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(format!(
            "failed to inspect sidecar {}: {error}",
            path.display()
        )),
        Ok(_) => Err(format!(
            "refuse to overwrite existing snapshot sidecar {}",
            path.display()
        )),
    }
}

fn snapshot_sidecar_path(
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
        .ok_or_else(|| format!("snapshot path {} has no UTF-8 file name", target.display()))?;
    Ok(parent.join(format!(
        ".{file_name}.xtask-{}-{nonce}-{index}.{role}",
        std::process::id()
    )))
}

fn stage_snapshot(path: &Path, rendered: &str) -> Result<(), String> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| format!("create staged snapshot {}: {error}", path.display()))?;
    let result = (|| {
        file.write_all(rendered.as_bytes())
            .map_err(|error| format!("write staged snapshot {}: {error}", path.display()))?;
        file.sync_all()
            .map_err(|error| format!("sync staged snapshot {}: {error}", path.display()))
    })();
    match result {
        Ok(()) => Ok(()),
        Err(primary) => match remove_file_if_present(path) {
            Ok(()) => Err(primary),
            Err(cleanup) => Err(with_cleanup_errors(primary, &[cleanup])),
        },
    }
}

fn cleanup_staged_files(entries: &[SnapshotTransactionEntry]) -> Vec<String> {
    entries
        .iter()
        .filter_map(|entry| remove_file_if_present(&entry.staged).err())
        .collect()
}

fn restore_originals(entries: &[SnapshotTransactionEntry]) -> Vec<String> {
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

fn rollback_snapshot_install(
    entries: &[SnapshotTransactionEntry],
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

fn remove_file_if_present(path: &Path) -> Result<(), String> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(format!("remove {}: {error}", path.display())),
    }
}

fn sync_snapshot_directories(entries: &[SnapshotTransactionEntry]) -> Result<(), String> {
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
            .map_err(|error| format!("sync snapshot directory {}: {error}", directory.display()))?;
    }
    Ok(())
}

fn with_cleanup_errors(primary: String, cleanup_errors: &[String]) -> String {
    if cleanup_errors.is_empty() {
        primary
    } else {
        format!(
            "{primary}; rollback/cleanup failures: {}",
            cleanup_errors.join("; ")
        )
    }
}

fn print_stable_api_help() {
    println!(
        "usage: cargo xtask stable-api [--write]\n\n\
         Without --write, checks the ordinary and rustdoc-hidden API snapshots \
         against cargo-public-api output for all 1.0-stable library crates. \
         With --write, refreshes both snapshots. This task must run on macOS \
         so target-gated Metal APIs are included."
    );
}

fn print_codec_math_codegen_help() {
    println!(
        "usage: cargo xtask codec-math-codegen [--write]\n\n\
         Without --write, checks generated Rust and Metal codec-math fragments \
         against the Rust source of truth. With --write, refreshes the fragments."
    );
}

#[cfg(test)]
mod tests {
    use std::{fs, path::PathBuf};

    use super::{
        finalize_text_snapshot, write_snapshot_pair_transactionally, SNAPSHOT_TRANSACTION_NONCE,
    };

    fn transaction_test_directory(label: &str) -> PathBuf {
        let nonce = SNAPSHOT_TRANSACTION_NONCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "j2k-stable-api-{label}-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir(&path).expect("create transaction test directory");
        path
    }

    #[test]
    fn text_snapshots_end_with_exactly_one_newline() {
        assert_eq!(finalize_text_snapshot("content\n\n"), "content\n");
        assert_eq!(finalize_text_snapshot("content"), "content\n");
        assert_eq!(finalize_text_snapshot(""), "");
    }

    #[test]
    fn snapshot_pair_transaction_replaces_both_files() {
        let directory = transaction_test_directory("commit");
        let ordinary = directory.join("ordinary.txt");
        let hidden = directory.join("hidden.txt");
        fs::write(&ordinary, "old ordinary").expect("seed ordinary snapshot");
        fs::write(&hidden, "old hidden").expect("seed hidden snapshot");

        write_snapshot_pair_transactionally(&[
            (
                ordinary.to_str().expect("ordinary UTF-8 path"),
                "new ordinary".to_string(),
            ),
            (
                hidden.to_str().expect("hidden UTF-8 path"),
                "new hidden".to_string(),
            ),
        ])
        .expect("commit snapshot pair");

        assert_eq!(fs::read_to_string(&ordinary).unwrap(), "new ordinary");
        assert_eq!(fs::read_to_string(&hidden).unwrap(), "new hidden");
        assert_eq!(fs::read_dir(&directory).unwrap().count(), 2);
        fs::remove_dir_all(directory).expect("clean transaction test directory");
    }

    #[test]
    fn staging_failure_leaves_existing_snapshot_unchanged() {
        let directory = transaction_test_directory("staging-failure");
        let ordinary = directory.join("ordinary.txt");
        let hidden = directory.join("missing-parent/hidden.txt");
        fs::write(&ordinary, "old ordinary").expect("seed ordinary snapshot");

        let error = write_snapshot_pair_transactionally(&[
            (
                ordinary.to_str().expect("ordinary UTF-8 path"),
                "new ordinary".to_string(),
            ),
            (
                hidden.to_str().expect("hidden UTF-8 path"),
                "new hidden".to_string(),
            ),
        ])
        .unwrap_err();

        assert!(error.contains("create staged snapshot"));
        assert_eq!(fs::read_to_string(&ordinary).unwrap(), "old ordinary");
        assert_eq!(fs::read_dir(&directory).unwrap().count(), 1);
        fs::remove_dir_all(directory).expect("clean transaction test directory");
    }

    #[test]
    fn snapshot_transaction_requires_two_distinct_paths() {
        assert!(write_snapshot_pair_transactionally(&[]).is_err());
        assert!(write_snapshot_pair_transactionally(&[
            ("same", "ordinary".to_string()),
            ("same", "hidden".to_string()),
        ])
        .is_err());
    }
}
