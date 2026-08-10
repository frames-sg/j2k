// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{fs, path::Path};

use j2k_t803::{T803Manifest, T803Suite};

fn repo_root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root")
}

#[test]
fn committed_manifest_covers_part1_and_the_formal_part15_scope() {
    let text = fs::read_to_string(repo_root().join("corpus/j2k-conformance/t803-v3.toml"))
        .expect("read committed T.803 manifest");

    let manifest = T803Manifest::parse(&text).expect("valid committed manifest");

    assert_eq!(manifest.decoder_cases.len(), 81);
    assert_eq!(manifest.jp2_cases.len(), 9);
    assert_eq!(manifest.ht_bsets.len(), 26);
    assert_eq!(manifest.jph_bsets.len(), 10);
    assert_eq!(manifest.files.len(), 189);
    assert!(manifest
        .files
        .windows(2)
        .all(|pair| pair[0].path < pair[1].path));
    assert_eq!(manifest.table_case_count("C.1"), 18);
    assert_eq!(manifest.table_case_count("C.4"), 8);
    assert_eq!(manifest.table_case_count("C.6"), 35);
    assert_eq!(manifest.table_case_count("C.7"), 17);
    assert_eq!(manifest.table_case_count("C.8"), 3);
    assert!(manifest
        .decoder_cases
        .iter()
        .all(|case| !case.codestream.contains("htj2k")));

    let selected = manifest
        .decoder_cases_for_suite(T803Suite::Part15)
        .expect("select official Part 15 cases");
    assert_eq!(selected.len(), 60);

    let selected_case = |bset: &str| {
        selected
            .iter()
            .find(|case| {
                case.part15
                    .as_ref()
                    .is_some_and(|metadata| metadata.bset == bset)
            })
            .expect("selected BSET comparison")
    };
    assert_eq!(
        selected_case("p0-06-ds0-ht").codestream,
        "files/htj2k_bsets_profile0/p0_06_bset/ds0_ht_06_b15.j2k"
    );
    assert_eq!(
        selected_case("p0-06-ds0-hm").codestream,
        "files/htj2k_bsets_profile0/p0_06_bset/ds0_hm_06_b11.j2k"
    );
    let high_fidelity = selected_case("hifi-ht1-02")
        .part15
        .as_ref()
        .expect("Part 15 metadata");
    assert_eq!(high_fidelity.mmagb, 20);
    assert_eq!(high_fidelity.bmagb, 14);
    let selected_jph = |id: &str| {
        manifest
            .jph_bsets
            .iter()
            .find(|bset| bset.id == id)
            .and_then(|bset| bset.selected_candidate())
            .expect("selected JPH BSET candidate")
    };
    assert_eq!(
        selected_jph("jph-7").path,
        "files/testfiles_jph/file7_b15.jph"
    );
    assert_eq!(
        selected_jph("jph-10").path,
        "files/testfiles_jph/file10_b12.jph"
    );
}

#[test]
fn manifest_rejects_unknown_fields_and_untrusted_paths() {
    let unknown = minimal_manifest("path = \"files/input.j2k\"\nextra = true");
    assert!(T803Manifest::parse(&unknown)
        .expect_err("unknown field must fail")
        .to_string()
        .contains("unknown field"));

    for path in ["../input.j2k", "/tmp/input.j2k", "files/./input.j2k"] {
        let text = minimal_manifest(&format!("path = {path:?}"));
        let error = T803Manifest::parse(&text).expect_err("unsafe path must fail");
        assert!(error.to_string().contains("relative normalized path"));
    }
}

#[test]
fn manifest_rejects_bad_hashes_duplicates_and_missing_inventory() {
    let bad_hash = minimal_manifest("path = \"files/input.j2k\"\nsha256 = \"ABC\"");
    assert!(T803Manifest::parse(&bad_hash)
        .expect_err("bad hash must fail")
        .to_string()
        .contains("SHA-256"));

    let duplicate = format!(
        "{}\n[[files]]\npath = \"files/input.j2k\"\nsha256 = \"{}\"\n",
        minimal_manifest("path = \"files/input.j2k\""),
        "0".repeat(64)
    );
    assert!(T803Manifest::parse(&duplicate)
        .expect_err("duplicate path must fail")
        .to_string()
        .contains("duplicate file"));

    let missing = minimal_manifest("path = \"files/unrelated.j2k\"");
    assert!(T803Manifest::parse(&missing)
        .expect_err("case input absent from inventory must fail")
        .to_string()
        .contains("not present in the file inventory"));
}

fn minimal_manifest(file_fields: &str) -> String {
    let file_hash = if file_fields.contains("sha256") {
        String::new()
    } else {
        format!("sha256 = \"{}\"", "0".repeat(64))
    };
    format!(
        r#"schema_version = 2
standard = "ISO/IEC 15444-4:2024 / ITU-T T.803 v3"

[source]
url = "https://www.itu.int/wftp3/public/t/testsignal/SpeImage/T803/v2024_02/T.803v3_15444-4ed4-ElecAtt-codestreams.zip"
archive_sha256 = "{hash}"
archive_bytes = 1

[[files]]
{file_fields}
{file_hash}

[[decoder_cases]]
id = "c1-p0-01-0"
table = "C.1"
codestream = "files/input.j2k"
reference = "files/reference.pgx"
component = 0
reduction_levels = 0
signed = false
bit_depth = 8
width = 1
height = 1
peak = 0
mse = 0.0

[[jp2_cases]]
id = "jp2-1"
input = "files/file1.jp2"
reference = "files/jp2_1.tif"
components = 3
bit_depth = 8
width = 1
height = 1
peak = 4
"#,
        hash = "0".repeat(64)
    )
}
