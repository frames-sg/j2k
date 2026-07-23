// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{ffi::OsStr, fs};

use super::super::{
    assert_file_pattern_checks, assert_pattern_checks, normalize_path, publishable_crate_dirs,
    repo_root, rust_include_paths, rust_sources, FilePatternCheck, PatternCheck,
};

#[test]
fn public_docs_route_users_to_current_crates() {
    let root = repo_root();
    let readme = fs::read_to_string(root.join("README.md")).expect("read README");
    assert_file_pattern_checks(
        root,
        &[FilePatternCheck::new("README.md")
            .named("README current crate routing")
            .required(&[
                "Which crate should I use?",
                "Fast Path For LLM-Assisted Use",
                "cargo add j2k",
                "statumen",
                "wsi-dicom",
                "j2k-jpeg",
                "j2k",
                "j2k-cli",
            ])],
    );

    let legacy_terms = [
        format!("{}{}", "ash", "lar"),
        format!("{}{}", "zig", "gurat"),
    ];
    for legacy in &legacy_terms {
        assert!(
            !readme.to_ascii_lowercase().contains(legacy),
            "README.md must use current package names only"
        );
    }
}

#[test]
fn published_crates_have_crates_io_landing_readmes() {
    let root = repo_root();

    for crate_dir in publishable_crate_dirs(root) {
        let manifest_path = crate_dir.join("Cargo.toml");
        let manifest = fs::read_to_string(&manifest_path)
            .unwrap_or_else(|err| panic!("read {}: {err}", manifest_path.display()));
        let readme_path = crate_dir.join("README.md");
        let package = crate_dir
            .file_name()
            .and_then(OsStr::to_str)
            .expect("publishable crate dir has UTF-8 name");

        assert!(
            manifest.contains("readme"),
            "{} must declare a readme for crates.io landing pages",
            manifest_path
                .strip_prefix(root)
                .unwrap_or(&manifest_path)
                .display()
        );
        assert!(
            readme_path.exists(),
            "{} must exist for crates.io landing pages",
            readme_path
                .strip_prefix(root)
                .unwrap_or(&readme_path)
                .display()
        );

        let readme = fs::read_to_string(&readme_path)
            .unwrap_or_else(|err| panic!("read {}: {err}", readme_path.display()));
        let readme_source_name = readme_path
            .strip_prefix(root)
            .unwrap_or(&readme_path)
            .display()
            .to_string();
        let docs_url = format!("https://docs.rs/{package}");
        assert_pattern_checks(
            &[PatternCheck::new(&readme_source_name, &readme).required(&[
                docs_url.as_str(),
                "https://github.com/frames-sg/j2k",
                "docs/public-support.md",
            ])],
        );
    }
}

#[test]
fn publishable_crates_configure_docs_rs_metadata() {
    let root = repo_root();

    for crate_dir in publishable_crate_dirs(root) {
        let manifest_path = crate_dir.join("Cargo.toml");
        let manifest = fs::read_to_string(&manifest_path)
            .unwrap_or_else(|err| panic!("read {}: {err}", manifest_path.display()));

        let manifest_source_name = manifest_path
            .strip_prefix(root)
            .unwrap_or(&manifest_path)
            .display()
            .to_string();
        assert_pattern_checks(&[
            PatternCheck::new(&manifest_source_name, &manifest).required(&[
                "[package.metadata.docs.rs]",
                "all-features = true",
                "targets = []",
            ]),
        ]);
    }
}

#[test]
fn support_matrix_is_linked_and_covers_adoption_surfaces() {
    assert_file_pattern_checks(
        repo_root(),
        &[FilePatternCheck::new("README.md")
            .named("README support matrix")
            .required(&[
                "Stable APIs",
                "Experimental APIs",
                "BackendRequest::Auto",
                "Security",
                "Benchmark and parity policy",
                "MSRV",
                "OpenJPEG",
                "Grok",
            ])],
    );
}

#[test]
fn public_codec_and_transcode_examples_are_publicly_linked() {
    let root = repo_root();
    let readme = fs::read_to_string(root.join("README.md")).expect("read README");

    let examples = [
        "crates/j2k/examples/decode_generated.rs",
        "crates/j2k-jpeg/examples/inspect.rs",
        "crates/j2k-metal/examples/decode_route_report.rs",
        "crates/j2k-metal/examples/htj2k_encode_auto_report.rs",
        "crates/j2k-metal/examples/resident_encode_buffer.rs",
        "crates/j2k-ml/examples/training_batcher.rs",
        "crates/j2k-ml/examples/cuda_upload.rs",
        "crates/j2k-ml/examples/metal_upload.rs",
        "crates/j2k-tilecodec/examples/decompress.rs",
        "crates/j2k-transcode/examples/jpeg_to_htj2k.rs",
        "crates/j2k-transcode-metal/examples/jpeg_to_htj2k_route_report.rs",
    ];
    for example in examples {
        assert!(
            root.join(example).exists(),
            "expected runnable example `{example}`"
        );
    }
    assert_pattern_checks(&[
        PatternCheck::new("README public example links", &readme).required(&examples)
    ]);
}

#[test]
fn packaged_rust_sources_do_not_include_files_outside_their_crate() {
    let root = repo_root();
    let workspace_crates = root.join("crates");
    let mut escaping = Vec::new();

    for source_path in rust_sources(&workspace_crates) {
        let Ok(relative_to_crates) = source_path.strip_prefix(&workspace_crates) else {
            continue;
        };
        let Some(crate_name) = relative_to_crates.components().next() else {
            continue;
        };
        let member_root = workspace_crates.join(crate_name.as_os_str());
        let source = fs::read_to_string(&source_path)
            .unwrap_or_else(|err| panic!("read {}: {err}", source_path.display()));

        for include_path in rust_include_paths(&source) {
            let resolved = normalize_path(
                &source_path
                    .parent()
                    .expect("source file has parent")
                    .join(&include_path),
            );
            if !resolved.starts_with(&member_root) {
                escaping.push(format!(
                    "{} includes {} outside package root",
                    source_path
                        .strip_prefix(root)
                        .unwrap_or(&source_path)
                        .display(),
                    include_path
                ));
            }
        }
    }

    assert!(
    escaping.is_empty(),
    "package source include paths must stay inside their crate so packaged tests/benches/examples are not dead: {escaping:?}"
);
}

#[test]
fn public_search_metadata_routes_generic_queries_to_one_landing_page() {
    let root = repo_root();
    let read = |relative: &str| {
        let path = root.join(relative);
        fs::read_to_string(&path).unwrap_or_else(|err| panic!("read {}: {err}", path.display()))
    };

    let root_readme = read("README.md");
    let workspace_manifest = read("Cargo.toml");
    let crate_manifest = read("crates/j2k/Cargo.toml");
    let crate_readme = read("crates/j2k/README.md");
    let crate_lib = read("crates/j2k/src/lib.rs");
    let home = read("docs/index.html");
    let landing = read("docs/rust-jpeg2000-codec/index.html");
    let sitemap = read("docs/sitemap.xml");

    assert!(root_readme.starts_with("# J2K — Pure-Rust JPEG 2000 and HTJ2K Codec\n"));
    assert!(root_readme.contains("[Pure-Rust JPEG 2000 codec documentation](https://frames-sg.github.io/j2k/rust-jpeg2000-codec/)"));
    assert!(workspace_manifest
        .contains("homepage     = \"https://frames-sg.github.io/j2k/rust-jpeg2000-codec/\""));
    assert!(crate_manifest.contains("description = \"Pure-Rust JPEG 2000"));
    assert!(crate_manifest.contains("homepage.workspace = true"));
    assert!(crate_readme.contains("[Pure-Rust JPEG 2000 codec documentation](https://frames-sg.github.io/j2k/rust-jpeg2000-codec/)"));
    assert!(crate_lib.contains("//! Pure-Rust JPEG 2000"));
    assert!(landing.contains("<title>Pure-Rust JPEG 2000 Codec — J2K</title>"));
    assert!(landing.contains("<h1>Pure-Rust JPEG 2000 Codec</h1>"));
    assert!(landing.contains(
        "<link rel=\"canonical\" href=\"https://frames-sg.github.io/j2k/rust-jpeg2000-codec/\">"
    ));
    assert!(!home.contains("<title>J2K: Rust JPEG 2000 / HTJ2K Codec</title>"));
    assert!(sitemap.contains("<loc>https://frames-sg.github.io/j2k/rust-jpeg2000-codec/</loc>\n    <priority>1.0</priority>"));
}
