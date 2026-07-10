// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{fs, path::Path};

const MODULES: &[(&str, usize)] = &[
    ("j2k_encode/dwt.rs", 500),
    ("j2k_encode/launch.rs", 300),
    ("j2k_encode/preprocess.rs", 400),
    ("j2k_encode/quantization.rs", 150),
    ("j2k_encode/readback.rs", 325),
    ("j2k_encode/types.rs", 250),
    ("j2k_encode/validation.rs", 75),
];

fn source(path: &str) -> String {
    fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("src").join(path))
        .unwrap_or_else(|error| panic!("read {path}: {error}"))
}

#[test]
fn cuda_j2k_encode_host_uses_focused_real_modules() {
    let allow_attribute = ["#[", "allow"].concat();
    let include_macro = ["include", "!("].concat();
    let wildcard_import = ["use super::", "*"].concat();
    let wildcard_reexport = ["pub use self::types::", "*"].concat();
    let root = source("j2k_encode.rs");
    assert!(
        root.lines().count() < 50,
        "j2k_encode.rs must remain a focused module shell"
    );
    assert!(root.contains("mod abi_tests;"));
    assert!(root.contains("mod validation_tests;"));
    for module in [
        "mod dwt;",
        "mod launch;",
        "mod preprocess;",
        "mod quantization;",
        "mod readback;",
        "mod types;",
        "mod validation;",
    ] {
        assert!(root.contains(module), "j2k_encode.rs must contain {module}");
    }
    assert!(root.contains("pub use self::types::{"));
    for public_type in [
        "CudaDwt53LevelShape",
        "CudaDwt53Output",
        "CudaDwt97BatchStageTimings",
        "CudaDwt97Output",
        "CudaJ2kDeinterleavedComponents",
        "CudaJ2kQuantizeJob",
        "CudaJ2kQuantizeSubbandRegionJob",
        "CudaJ2kQuantizedSubband",
        "CudaJ2kResidentComponents",
        "CudaJ2kResidentQuantizedSubband",
        "CudaResidentDwt53Output",
        "CudaResidentDwt97Output",
    ] {
        assert!(
            root.contains(public_type),
            "j2k_encode.rs must reexport {public_type}"
        );
    }
    assert!(!root.contains(&include_macro));
    assert!(!root.contains(&allow_attribute));

    for (path, max_lines) in MODULES {
        let module = source(path);
        assert!(
            module.lines().count() < *max_lines,
            "{path} must stay below its focused-module line-count ratchet"
        );
        assert!(
            !module.contains(&include_macro),
            "{path} must be a real module"
        );
        assert!(
            !module.contains(&wildcard_import),
            "{path} must use explicit imports"
        );
        assert!(
            !module.contains(&wildcard_reexport),
            "{path} must use explicit reexports"
        );
        assert!(
            !module.contains(&allow_attribute),
            "{path} must avoid broad allows"
        );
    }
}
