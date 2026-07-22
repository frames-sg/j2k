// SPDX-License-Identifier: MIT OR Apache-2.0

const ROOT: &str = include_str!("../jpeg.rs");
const MODULES: &[(&str, &str, usize)] = &[
    ("jpeg/abi_tests.rs", include_str!("abi_tests.rs"), 100),
    ("jpeg/decode.rs", include_str!("decode.rs"), 300),
    (
        "jpeg/decode_launch.rs",
        include_str!("decode_launch.rs"),
        250,
    ),
    (
        "jpeg/decode_launch/decode_workspace.rs",
        include_str!("decode_launch/decode_workspace.rs"),
        150,
    ),
    ("jpeg/diagnostics.rs", include_str!("diagnostics.rs"), 250),
    (
        "jpeg/diagnostics_allocation.rs",
        include_str!("diagnostics_allocation.rs"),
        125,
    ),
    (
        "jpeg/diagnostics_execution.rs",
        include_str!("diagnostics_execution.rs"),
        250,
    ),
    ("jpeg/encode.rs", include_str!("encode.rs"), 325),
    ("jpeg/encode_batch.rs", include_str!("encode_batch.rs"), 250),
    (
        "jpeg/encode_allocation.rs",
        include_str!("encode_allocation.rs"),
        150,
    ),
    (
        "jpeg/encode_allocation/tests.rs",
        include_str!("encode_allocation/tests.rs"),
        125,
    ),
    (
        "jpeg/encode_launch.rs",
        include_str!("encode_launch.rs"),
        200,
    ),
    (
        "jpeg/encode_validation.rs",
        include_str!("encode_validation.rs"),
        250,
    ),
    (
        "jpeg/encode_validation/layout.rs",
        include_str!("encode_validation/layout.rs"),
        225,
    ),
    (
        "jpeg/encode_validation/tables.rs",
        include_str!("encode_validation/tables.rs"),
        150,
    ),
    (
        "jpeg/encode_validation/tests.rs",
        include_str!("encode_validation/tests.rs"),
        350,
    ),
    (
        "jpeg/encode_validation/tests/boundaries.rs",
        include_str!("encode_validation/tests/boundaries.rs"),
        125,
    ),
    (
        "jpeg/encode_validation/tests/huffman.rs",
        include_str!("encode_validation/tests/huffman.rs"),
        150,
    ),
    (
        "jpeg/encode_validation/tests/launch_geometry.rs",
        include_str!("encode_validation/tests/launch_geometry.rs"),
        75,
    ),
    (
        "jpeg/structure_tests.rs",
        include_str!("structure_tests.rs"),
        150,
    ),
    ("jpeg/types.rs", include_str!("types.rs"), 575),
    ("jpeg/validation.rs", include_str!("validation.rs"), 125),
    (
        "jpeg/validation/decode_plan.rs",
        include_str!("validation/decode_plan.rs"),
        250,
    ),
    (
        "jpeg/validation/huffman.rs",
        include_str!("validation/huffman.rs"),
        150,
    ),
    (
        "jpeg/validation/tests.rs",
        include_str!("validation/tests.rs"),
        200,
    ),
    (
        "jpeg/validation/tests/decode_security.rs",
        include_str!("validation/tests/decode_security.rs"),
        350,
    ),
];

#[test]
fn cuda_jpeg_runtime_uses_focused_real_modules() {
    let include_macro = ["include", "!("].concat();
    let wildcard_import = ["use super::", "*"].concat();
    assert!(
        ROOT.lines().count() < 60,
        "jpeg.rs must remain a focused module shell"
    );
    for module in [
        "mod decode;",
        "mod decode_launch;",
        "mod diagnostics;",
        "mod diagnostics_allocation;",
        "mod diagnostics_execution;",
        "mod encode;",
        "mod encode_batch;",
        "mod encode_allocation;",
        "mod encode_launch;",
        "mod encode_validation;",
        "mod structure_tests;",
        "mod types;",
        "mod validation;",
    ] {
        assert!(ROOT.contains(module), "jpeg.rs must contain {module}");
    }
    assert!(!ROOT.contains(&include_macro));
    for (path, source, max_lines) in MODULES {
        assert!(
            source.lines().count() < *max_lines,
            "{path} must stay below its focused-module line-count ratchet"
        );
        assert!(
            !source.contains(&include_macro),
            "{path} must be a real module"
        );
        assert!(
            !source.contains(&wildcard_import),
            "{path} must use explicit imports"
        );
    }
}
