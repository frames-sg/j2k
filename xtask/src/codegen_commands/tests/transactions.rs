// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{
    fs,
    path::PathBuf,
    sync::{Mutex, MutexGuard},
};

use super::super::transaction::{
    ensure_path_absent, remove_file_if_present, rollback_generated_pair_install, sidecar_path,
    with_cleanup_errors, write_generated_pair_transactionally, GeneratedPairEntry,
    PAIR_TRANSACTION_NONCE,
};

static TRANSACTION_TEST_SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    TRANSACTION_TEST_SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn transaction_test_directory(label: &str) -> PathBuf {
    let nonce = PAIR_TRANSACTION_NONCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!(
        "j2k-stable-api-{label}-{}-{nonce}",
        std::process::id()
    ));
    fs::create_dir(&path).expect("create transaction test directory");
    path
}

#[test]
fn snapshot_pair_transaction_replaces_both_files() {
    let _serial = serial();
    let directory = transaction_test_directory("commit");
    let ordinary = directory.join("ordinary.txt");
    let hidden = directory.join("hidden.txt");
    fs::write(&ordinary, "old ordinary").expect("seed ordinary snapshot");
    fs::write(&hidden, "old hidden").expect("seed hidden snapshot");

    write_generated_pair_transactionally(&[
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
fn snapshot_pair_transaction_installs_two_new_files_without_sidecars() {
    let _serial = serial();
    let directory = transaction_test_directory("new-files");
    let ordinary = directory.join("ordinary.txt");
    let hidden = directory.join("hidden.txt");

    write_generated_pair_transactionally(&[
        (
            ordinary.to_str().expect("ordinary UTF-8 path"),
            "ordinary".to_string(),
        ),
        (
            hidden.to_str().expect("hidden UTF-8 path"),
            "hidden".to_string(),
        ),
    ])
    .expect("install new snapshot pair");

    assert_eq!(fs::read_to_string(&ordinary).unwrap(), "ordinary");
    assert_eq!(fs::read_to_string(&hidden).unwrap(), "hidden");
    assert_eq!(fs::read_dir(&directory).unwrap().count(), 2);
    fs::remove_dir_all(directory).expect("clean transaction test directory");
}

#[test]
fn staging_failure_leaves_existing_snapshot_unchanged() {
    let _serial = serial();
    let directory = transaction_test_directory("staging-failure");
    let ordinary = directory.join("ordinary.txt");
    let hidden = directory.join("missing-parent/hidden.txt");
    fs::write(&ordinary, "old ordinary").expect("seed ordinary snapshot");

    let error = write_generated_pair_transactionally(&[
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

    assert!(error.contains("create staged generated file"));
    assert_eq!(fs::read_to_string(&ordinary).unwrap(), "old ordinary");
    assert_eq!(fs::read_dir(&directory).unwrap().count(), 1);
    fs::remove_dir_all(directory).expect("clean transaction test directory");
}

#[test]
fn snapshot_transaction_preflight_rejects_existing_sidecars_without_mutation() {
    let _serial = serial();
    let directory = transaction_test_directory("sidecar-preflight");
    let ordinary = directory.join("ordinary.txt");
    let hidden = directory.join("hidden.txt");
    fs::write(&ordinary, "old ordinary").expect("seed ordinary snapshot");
    fs::write(&hidden, "old hidden").expect("seed hidden snapshot");
    let nonce = PAIR_TRANSACTION_NONCE.load(std::sync::atomic::Ordering::Relaxed);
    let sidecar = sidecar_path(&ordinary, nonce, 0, "backup").expect("sidecar path");
    fs::write(&sidecar, "collision").expect("seed colliding sidecar");

    let error = write_generated_pair_transactionally(&[
        (
            ordinary.to_str().expect("ordinary UTF-8 path"),
            "new ordinary".to_string(),
        ),
        (
            hidden.to_str().expect("hidden UTF-8 path"),
            "new hidden".to_string(),
        ),
    ])
    .expect_err("existing transaction sidecar");

    assert!(
        error.contains("refuse to overwrite existing generated-file sidecar"),
        "unexpected error: {error}"
    );
    assert_eq!(fs::read_to_string(&ordinary).unwrap(), "old ordinary");
    assert_eq!(fs::read_to_string(&hidden).unwrap(), "old hidden");
    assert_eq!(fs::read_to_string(&sidecar).unwrap(), "collision");
    fs::remove_dir_all(directory).expect("clean transaction test directory");
}

#[test]
fn snapshot_transaction_requires_two_distinct_paths() {
    let _serial = serial();
    assert!(write_generated_pair_transactionally(&[]).is_err());
    assert!(write_generated_pair_transactionally(&[
        ("same", "ordinary".to_string()),
        ("same", "hidden".to_string()),
    ])
    .is_err());
}

#[test]
fn transaction_file_helpers_distinguish_absent_files_and_cleanup_context() {
    let _serial = serial();
    let directory = transaction_test_directory("file-helpers");
    let path = directory.join("entry");
    assert_eq!(ensure_path_absent(&path), Ok(()));
    assert_eq!(remove_file_if_present(&path), Ok(()));
    fs::write(&path, "value").expect("seed entry");
    assert!(ensure_path_absent(&path).is_err());
    assert_eq!(remove_file_if_present(&path), Ok(()));
    assert!(!path.exists());

    assert_eq!(with_cleanup_errors("primary".to_string(), &[]), "primary");
    assert_eq!(
        with_cleanup_errors(
            "primary".to_string(),
            &["first".to_string(), "second".to_string()]
        ),
        "primary; rollback/cleanup failures: first; second"
    );
    fs::remove_dir_all(directory).expect("clean transaction test directory");
}

#[test]
fn rollback_removes_partial_installs_restores_both_originals_and_cleans_staging() {
    let _serial = serial();
    let directory = transaction_test_directory("rollback");
    let first_target = directory.join("ordinary.txt");
    let second_target = directory.join("hidden.txt");
    let first_staged = directory.join("ordinary.staged");
    let second_staged = directory.join("hidden.staged");
    let first_backup = directory.join("ordinary.backup");
    let second_backup = directory.join("hidden.backup");
    fs::write(&first_target, "partially installed ordinary").expect("seed partial install");
    fs::write(&second_staged, "staged hidden").expect("seed remaining staged file");
    fs::write(&first_backup, "old ordinary").expect("seed ordinary backup");
    fs::write(&second_backup, "old hidden").expect("seed hidden backup");

    let entries = [
        GeneratedPairEntry {
            target: first_target.clone(),
            staged: first_staged.clone(),
            backup: first_backup,
            had_original: true,
        },
        GeneratedPairEntry {
            target: second_target.clone(),
            staged: second_staged.clone(),
            backup: second_backup,
            had_original: true,
        },
    ];

    let errors = rollback_generated_pair_install(&entries, 1);

    assert!(errors.is_empty(), "rollback errors: {errors:#?}");
    assert_eq!(fs::read_to_string(&first_target).unwrap(), "old ordinary");
    assert_eq!(fs::read_to_string(&second_target).unwrap(), "old hidden");
    assert!(!first_staged.exists());
    assert!(!second_staged.exists());
    fs::remove_dir_all(directory).expect("clean transaction test directory");
}
