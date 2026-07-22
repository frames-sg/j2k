// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;

use super::super::{assert_rust_source_scan_checks, repo_root, RustSourceScanCheck};

#[test]
fn adapter_crates_do_not_import_codec_private_modules() {
    assert_rust_source_scan_checks(
        repo_root(),
        &[RustSourceScanCheck::new(
            "adapter codec-private module imports",
            &[
                "crates/j2k-jpeg-metal",
                "crates/j2k-jpeg-cuda",
                "crates/j2k-metal",
                "crates/j2k-cuda",
            ],
        )
        .forbidden(&["::__private", " __private::"])],
    );
}

#[test]
fn cuda_adapter_crates_keep_public_libs_as_module_shells() {
    let root = repo_root();
    for (crate_dir, modules) in [
        (
            "crates/j2k-jpeg-cuda",
            &[
                "codec.rs",
                "decoder.rs",
                "error.rs",
                "runtime.rs",
                "session.rs",
                "surface.rs",
            ][..],
        ),
        (
            "crates/j2k-cuda",
            &[
                "codec.rs",
                "decoder.rs",
                "encode.rs",
                "error.rs",
                "runtime.rs",
                "session.rs",
                "surface.rs",
            ][..],
        ),
    ] {
        let source_directory = root.join(crate_dir).join("src");
        let lib_path = source_directory.join("lib.rs");
        let lib = fs::read_to_string(&lib_path)
            .unwrap_or_else(|error| panic!("read {}: {error}", lib_path.display()));
        assert!(
            lib.lines().count() <= 220,
            "{} should stay a thin public module shell",
            lib_path.display()
        );
        for module in modules {
            assert!(
                source_directory.join(module).exists(),
                "{crate_dir}/src/{module} must exist"
            );
        }
    }
}
