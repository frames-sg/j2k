// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{
    fs,
    path::PathBuf,
    sync::{Mutex, MutexGuard},
};

use super::super::{
    ensure_path_absent, remove_file_if_present, snapshot_sidecar_path, with_cleanup_errors,
    write_snapshot_pair_transactionally, SNAPSHOT_TRANSACTION_NONCE,
};

static TRANSACTION_TEST_SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    TRANSACTION_TEST_SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

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
fn snapshot_pair_transaction_replaces_both_files() {
    let _serial = serial();
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
fn snapshot_pair_transaction_installs_two_new_files_without_sidecars() {
    let _serial = serial();
    let directory = transaction_test_directory("new-files");
    let ordinary = directory.join("ordinary.txt");
    let hidden = directory.join("hidden.txt");

    write_snapshot_pair_transactionally(&[
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
fn snapshot_transaction_preflight_rejects_existing_sidecars_without_mutation() {
    let _serial = serial();
    let directory = transaction_test_directory("sidecar-preflight");
    let ordinary = directory.join("ordinary.txt");
    let hidden = directory.join("hidden.txt");
    fs::write(&ordinary, "old ordinary").expect("seed ordinary snapshot");
    fs::write(&hidden, "old hidden").expect("seed hidden snapshot");
    let nonce = SNAPSHOT_TRANSACTION_NONCE.load(std::sync::atomic::Ordering::Relaxed);
    let sidecar = snapshot_sidecar_path(&ordinary, nonce, 0, "backup").expect("sidecar path");
    fs::write(&sidecar, "collision").expect("seed colliding sidecar");

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
    .expect_err("existing transaction sidecar");

    assert!(
        error.contains("refuse to overwrite existing snapshot sidecar"),
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
    assert!(write_snapshot_pair_transactionally(&[]).is_err());
    assert!(write_snapshot_pair_transactionally(&[
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
