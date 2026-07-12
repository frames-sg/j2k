// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{fs, path::Path};

const MODULES: &[(&str, usize)] = &[
    ("j2k_encode/dwt.rs", 500),
    ("j2k_encode/dwt/tests.rs", 300),
    ("j2k_encode/dwt/tests/launch_geometry.rs", 100),
    ("j2k_encode/dwt/validation.rs", 75),
    ("j2k_encode/dwt/validation/tests.rs", 100),
    ("j2k_encode/dwt/validation/tests/launch_geometry.rs", 100),
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

#[test]
fn forward_dwt_geometry_validation_remains_shared_and_adversarially_tested() {
    let dwt = source("j2k_encode/dwt.rs");
    let validation = source("j2k_encode/dwt/validation.rs");
    let validation_tests = source("j2k_encode/dwt/validation/tests.rs");
    let cuda_tests = source("j2k_encode/dwt/tests.rs");

    assert!(dwt.contains("mod validation;"));
    assert_eq!(
        dwt.matches("validate_forward_dwt_request(width, height, num_levels)?;")
            .count(),
        4,
        "host and resident 5/3 and 9/7 APIs must share pre-launch geometry validation"
    );
    for required in [
        "use j2k_codec_math::dwt::max_decomposition_levels;",
        "let max_levels = max_decomposition_levels(width, height);",
        "if num_levels > max_levels",
        "FORWARD_DWT_LEVELS_EXCEED_GEOMETRY",
        "FORWARD_DWT_GEOMETRY_EXCEEDS_LAUNCH_LIMITS",
        "j2k_dwt53_launch_geometry(width, height).is_none()",
    ] {
        assert!(
            validation.contains(required),
            "forward DWT validation must contain {required}"
        );
    }
    for required in [
        "maximum_levels_match_native_minimum_axis_contract",
        "zero_and_boundary_level_requests_are_valid",
        "one_axis_only_and_excess_level_requests_are_rejected",
    ] {
        assert!(
            validation_tests.contains(required),
            "forward DWT pure tests must contain {required}"
        );
    }
    for required in [
        "safe_host_and_resident_dwt_apis_reject_degenerate_later_levels",
        "resident_level_validation_precedes_component_copy",
        "boundary_valid_53_and_97_levels_produce_complete_host_and_resident_outputs",
        "(2, 8, 2)",
        "(8, 2, 2)",
        "(1, 8, 1)",
        "(1, 7, 1)",
    ] {
        assert!(
            cuda_tests.contains(required),
            "forward DWT CUDA tests must contain {required}"
        );
    }
}
