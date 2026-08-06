// SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg(feature = "runner")]

use std::{
    fs,
    io::{Cursor, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicUsize, Ordering},
};

use j2k_t803::{
    runner::{extract_selected_archive, verify_corpus, ArchiveLimits},
    CorpusFile,
};
use zip::{write::SimpleFileOptions, CompressionMethod, ZipWriter};

const EXPECTED_SHA256: &str = "cea23dd4b87e8b00d19fb9ccaaef93e97353c7353e2070f3baf05aeb3995dff4";

fn corpus_file() -> CorpusFile {
    CorpusFile {
        path: "files/required.pgx".to_string(),
        sha256: EXPECTED_SHA256.to_string(),
    }
}

fn archive(entries: &[(&str, &[u8])]) -> Cursor<Vec<u8>> {
    let mut writer = ZipWriter::new(Cursor::new(Vec::new()));
    let options = SimpleFileOptions::default().compression_method(CompressionMethod::Stored);
    for (name, bytes) in entries {
        writer.start_file(*name, options).expect("start ZIP entry");
        writer.write_all(bytes).expect("write ZIP entry");
    }
    writer.finish().expect("finish ZIP")
}

fn symlink_archive() -> Cursor<Vec<u8>> {
    let mut writer = ZipWriter::new(Cursor::new(Vec::new()));
    writer
        .add_symlink(
            "files/required.pgx",
            "elsewhere",
            SimpleFileOptions::default(),
        )
        .expect("add ZIP symlink");
    writer.finish().expect("finish ZIP")
}

fn temporary_directory(label: &str) -> PathBuf {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let path = std::env::temp_dir().join(format!(
        "j2k-t803-{label}-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir(&path).expect("create temporary directory");
    path
}

fn cleanup(path: &Path) {
    fs::remove_dir_all(path).expect("remove temporary directory");
}

#[test]
fn extraction_writes_only_the_pinned_inventory() {
    let output = temporary_directory("selected");
    let zip = archive(&[
        ("files/required.pgx", b"expected"),
        ("files/unselected.txt", b"not extracted"),
    ]);

    extract_selected_archive(zip, &output, &[corpus_file()], ArchiveLimits::default())
        .expect("safe extraction");

    assert_eq!(
        fs::read(output.join("files/required.pgx")).expect("required file"),
        b"expected"
    );
    assert!(!output.join("files/unselected.txt").exists());
    verify_corpus(&output, &[corpus_file()]).expect("exact extracted inventory");
    cleanup(&output);
}

#[test]
fn extraction_rejects_traversal_duplicates_missing_and_oversized_entries() {
    let cases = [
        (
            "traversal",
            archive(&[("../escape", b"x"), ("files/required.pgx", b"expected")]),
            ArchiveLimits::default(),
            "unsafe path",
        ),
        (
            "duplicate",
            archive(&[
                ("files/required.pgx", b"expected"),
                ("files//required.pgx", b"expected"),
            ]),
            ArchiveLimits::default(),
            "duplicate",
        ),
        (
            "missing",
            archive(&[("files/other.pgx", b"expected")]),
            ArchiveLimits::default(),
            "missing",
        ),
        (
            "oversized",
            archive(&[("files/required.pgx", b"expected")]),
            ArchiveLimits {
                max_entries: 8,
                max_entry_bytes: 7,
                max_total_bytes: 64,
            },
            "too large",
        ),
    ];

    for (label, zip, limits, expected) in cases {
        let output = temporary_directory(label);
        let error = extract_selected_archive(zip, &output, &[corpus_file()], limits)
            .expect_err("unsafe archive must fail");
        assert!(
            error.to_string().contains(expected),
            "{error:?} did not mention {expected:?}"
        );
        assert!(!output.join("files/required.pgx").exists());
        cleanup(&output);
    }
}

#[test]
fn corpus_verification_rejects_changed_or_extra_files() {
    let output = temporary_directory("verify");
    fs::create_dir(output.join("files")).expect("create corpus subdirectory");
    fs::write(output.join("files/required.pgx"), b"changed").expect("write changed file");

    let changed = verify_corpus(&output, &[corpus_file()]).expect_err("changed hash must fail");
    assert!(changed.to_string().contains("SHA-256"));

    fs::write(output.join("files/required.pgx"), b"expected").expect("restore required file");
    fs::write(output.join("files/extra.pgx"), b"extra").expect("write extra file");
    let extra = verify_corpus(&output, &[corpus_file()]).expect_err("extra file must fail");
    assert!(extra.to_string().contains("extra"));
    cleanup(&output);
}

#[test]
fn extraction_rejects_symlinks() {
    let output = temporary_directory("symlink");
    let error = extract_selected_archive(
        symlink_archive(),
        &output,
        &[corpus_file()],
        ArchiveLimits::default(),
    )
    .expect_err("symlink must fail");

    assert!(error.to_string().contains("symlink"));
    assert!(!output.join("files/required.pgx").exists());
    cleanup(&output);
}
