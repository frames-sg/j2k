// SPDX-License-Identifier: MIT OR Apache-2.0

const ROOT: &str = include_str!("../j2k_decode.rs");
const IDWT: &str = include_str!("idwt.rs");
const IDWT_PREFLIGHT: &str = include_str!("idwt/preflight.rs");
const STORE: &str = include_str!("store.rs");
const STORE_TESTS: &str = include_str!("store/tests.rs");
const MODULES: &[(&str, &str, usize)] = &[
    ("j2k_decode/idwt.rs", IDWT, 650),
    (
        "j2k_decode/idwt/context_validation.rs",
        include_str!("idwt/context_validation.rs"),
        125,
    ),
    (
        "j2k_decode/idwt/job_validation.rs",
        include_str!("idwt/job_validation.rs"),
        175,
    ),
    (
        "j2k_decode/idwt/job_validation/tests.rs",
        include_str!("idwt/job_validation/tests.rs"),
        225,
    ),
    (
        "j2k_decode/idwt/launch_validation.rs",
        include_str!("idwt/launch_validation.rs"),
        150,
    ),
    (
        "j2k_decode/idwt/launch_validation/tests.rs",
        include_str!("idwt/launch_validation/tests.rs"),
        100,
    ),
    ("j2k_decode/idwt/preflight.rs", IDWT_PREFLIGHT, 100),
    (
        "j2k_decode/idwt/sequence.rs",
        include_str!("idwt/sequence.rs"),
        225,
    ),
    (
        "j2k_decode/idwt/tests.rs",
        include_str!("idwt/tests.rs"),
        250,
    ),
    (
        "j2k_decode/idwt_launch.rs",
        include_str!("idwt_launch.rs"),
        350,
    ),
    ("j2k_decode/store.rs", STORE, 600),
    (
        "j2k_decode/store/batch.rs",
        include_str!("store/batch.rs"),
        350,
    ),
    (
        "j2k_decode/store/batch/external.rs",
        include_str!("store/batch/external.rs"),
        175,
    ),
    (
        "j2k_decode/store/batch/tests.rs",
        include_str!("store/batch/tests.rs"),
        50,
    ),
    (
        "j2k_decode/store/destination.rs",
        include_str!("store/destination.rs"),
        125,
    ),
    (
        "j2k_decode/store/color_native_batch.rs",
        include_str!("store/color_native_batch.rs"),
        450,
    ),
    (
        "j2k_decode/store/color_native_batch/plan.rs",
        include_str!("store/color_native_batch/plan.rs"),
        250,
    ),
    ("j2k_decode/store/tests.rs", STORE_TESTS, 500),
    (
        "j2k_decode/store/tests/color_native.rs",
        include_str!("store/tests/color_native.rs"),
        125,
    ),
    (
        "j2k_decode/store/tests/zero_init.rs",
        include_str!("store/tests/zero_init.rs"),
        300,
    ),
    (
        "j2k_decode/store/validation.rs",
        include_str!("store/validation.rs"),
        150,
    ),
    (
        "j2k_decode/store_launch.rs",
        include_str!("store_launch.rs"),
        225,
    ),
    (
        "j2k_decode/store_launch/color_native.rs",
        include_str!("store_launch/color_native.rs"),
        75,
    ),
    (
        "j2k_decode/store_launch/color_native_rgba.rs",
        include_str!("store_launch/color_native_rgba.rs"),
        75,
    ),
    ("j2k_decode/trace.rs", include_str!("trace.rs"), 175),
    ("j2k_decode/types.rs", include_str!("types.rs"), 385),
    (
        "j2k_decode/types/color_native.rs",
        include_str!("types/color_native.rs"),
        125,
    ),
    (
        "j2k_decode/validation.rs",
        include_str!("validation.rs"),
        100,
    ),
];

#[test]
fn cuda_j2k_decode_uses_focused_real_modules() {
    let include_macro = ["include", "!("].concat();
    let wildcard_import = ["use super::", "*"].concat();
    assert!(
        ROOT.lines().count() < 75,
        "j2k_decode.rs must remain a focused module shell"
    );
    for module in [
        "mod idwt;",
        "mod idwt_launch;",
        "mod store;",
        "mod store_launch;",
        "mod trace;",
        "mod types;",
        "mod validation;",
    ] {
        assert!(ROOT.contains(module), "j2k_decode.rs must contain {module}");
    }
    assert!(
        IDWT.contains("mod context_validation;"),
        "j2k_decode/idwt.rs must delegate ownership and alias validation"
    );
    assert!(
        IDWT.contains("pub(super) mod job_validation;"),
        "j2k_decode/idwt.rs must delegate full job validation"
    );
    assert!(
        IDWT.contains("mod preflight;")
            && IDWT.contains("mod launch_validation;")
            && IDWT.contains("mod sequence;"),
        "j2k_decode/idwt.rs must delegate preflight, launch validation, and sequence ownership"
    );
    assert!(
        IDWT_PREFLIGHT.contains("pub fn j2k_inverse_dwt_single_output_bytes("),
        "IDWT output allocation must have a public runtime preflight"
    );
    assert!(
        STORE.contains("mod batch;")
            && STORE.contains("mod color_native_batch;")
            && STORE.contains("mod destination;"),
        "j2k_decode/store.rs must delegate batch planning and destination validation"
    );
    assert!(
        STORE.contains("mod validation;"),
        "j2k_decode/store.rs must delegate store and MCT validation"
    );
    assert!(
        STORE_TESTS.contains("mod color_native;") && STORE_TESTS.contains("mod zero_init;"),
        "store tests must delegate exact-color and zero-initialization coverage"
    );
    assert!(
        include_str!("store/color_native_batch.rs").contains("mod plan;"),
        "exact-native RGB store must delegate target planning and validation"
    );
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
