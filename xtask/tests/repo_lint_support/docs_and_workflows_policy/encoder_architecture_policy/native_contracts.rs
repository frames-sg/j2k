// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;

use crate::repo_lint_support::{assert_pattern_checks, repo_root, PatternCheck};

#[test]
fn native_public_contracts_live_in_focused_modules() {
    let root = repo_root();
    let lib =
        fs::read_to_string(root.join("crates/j2k-native/src/lib.rs")).expect("read j2k-native lib");
    let backend = fs::read_to_string(root.join("crates/j2k-native/src/backend.rs"))
        .expect("read j2k-native backend module");
    let color = fs::read_to_string(root.join("crates/j2k-native/src/color.rs"))
        .expect("read j2k-native color module");
    let ht_adapter = fs::read_to_string(root.join("crates/j2k-native/src/ht_adapter.rs"))
        .expect("read j2k-native HT adapter module");
    let roi = fs::read_to_string(root.join("crates/j2k-native/src/roi.rs"))
        .expect("read j2k-native ROI module");
    let types = fs::read_to_string(root.join("crates/j2k-types/src/lib.rs"))
        .expect("read j2k-types module");

    assert_pattern_checks(&[
        PatternCheck::new("j2k-native focused public module wiring", &lib).required(&[
            "mod backend;",
            "mod color;",
            "mod ht_adapter;",
            "mod roi;",
            "pub use backend::{",
            "pub use color::{",
            "pub use ht_adapter::{",
            "pub use roi::idwt_band_index;",
        ]),
    ]);
    assert!(
        lib.lines().count() < 2_260,
        "j2k-native lib.rs must keep shrinking after the test-module split"
    );

    let backend_items = [
        "pub trait HtCodeBlockDecoder",
        "pub struct HtCodeBlockDecodeJob",
        "pub struct J2kCodeBlockDecodeJob",
        "pub struct J2kRect",
    ];
    let color_items = [
        "pub enum ColorSpace",
        "pub struct Bitmap",
        "pub struct RawBitmap",
        "pub struct DecodedNativeComponents",
        "pub(crate) fn resolve_alpha_and_color_space",
        "pub(crate) fn convert_color_space",
        "pub(crate) fn cielab_to_rgb",
    ];
    let roi_items = [
        "pub fn idwt_band_index",
        "pub(crate) fn add_roi_shift_to_bitplanes",
        "pub(crate) fn apply_roi_maxshift_inverse_i64",
        "pub(crate) fn apply_roi_maxshift_inverse_i32",
        "pub(crate) fn validate_roi",
    ];
    let ht_adapter_items = [
        "pub struct HtSigPropBenchmarkState",
        "pub fn prepare_ht_sigprop_benchmark_state",
        "pub fn decode_ht_sigprop_benchmark_state",
        "pub fn ht_vlc_table0",
        "pub fn ht_uvlc_encode_table_bytes",
    ];
    assert_pattern_checks(&[
        PatternCheck::new("j2k-types encode-stage accelerator ownership", &types)
            .required(&["pub trait J2kEncodeStageAccelerator"]),
        PatternCheck::new("j2k-native backend accelerator exclusion", &backend)
            .forbidden(&["pub trait J2kEncodeStageAccelerator"]),
        PatternCheck::new("j2k-native backend contract exclusion", &lib).forbidden(&backend_items),
        PatternCheck::new("j2k-native backend contract ownership", &backend)
            .required(&backend_items),
        PatternCheck::new("j2k-native color/output exclusion", &lib).forbidden(&color_items),
        PatternCheck::new("j2k-native color/output ownership", &color).required(&color_items),
        PatternCheck::new("j2k-native ROI helper exclusion", &lib).forbidden(&roi_items),
        PatternCheck::new("j2k-native ROI helper ownership", &roi).required(&roi_items),
        PatternCheck::new("j2k-native HT adapter helper exclusion", &lib)
            .forbidden(&ht_adapter_items),
        PatternCheck::new("j2k-native HT adapter helper ownership", &ht_adapter)
            .required(&ht_adapter_items),
    ]);
}

#[test]
fn native_adapter_exports_are_doc_hidden() {
    let root = repo_root();
    let lib =
        fs::read_to_string(root.join("crates/j2k-native/src/lib.rs")).expect("read j2k-native lib");
    let scalar_encode = fs::read_to_string(root.join("crates/j2k-native/src/scalar/encode.rs"))
        .expect("read j2k-native scalar encode module");
    let scalar_decode =
        fs::read_to_string(root.join("crates/j2k-native/src/scalar/classic_decode.rs"))
            .expect("read j2k-native scalar classic decode module");
    let image = fs::read_to_string(root.join("crates/j2k-native/src/image.rs"))
        .expect("read j2k-native image module");

    assert_pattern_checks(&[
        PatternCheck::new("j2k-native hidden adapter exports", &lib).required(&[
            "#[doc(hidden)]\npub use backend::",
            "#[doc(hidden)]\npub use direct_plan::",
            "#[doc(hidden)]\npub use ht_adapter::",
            "#[doc(hidden)]\npub use j2k_types::",
            "#[doc(hidden)]\npub use scalar::",
        ]),
        PatternCheck::new("j2k-native hidden scalar encode adapters", &scalar_encode)
            .required(&["#[doc(hidden)]\npub fn forward_dwt53_reference"]),
        PatternCheck::new("j2k-native hidden scalar decode adapters", &scalar_decode)
            .required(&["#[doc(hidden)]\npub fn decode_j2k_code_block_scalar"]),
        PatternCheck::new("j2k-native image contract ownership", &image)
            .required(&["pub struct DecodeSettings", "pub struct Image"]),
    ]);
}

#[test]
fn native_ht_adapter_benchmark_state_has_focused_ownership_regressions() {
    let root = repo_root();
    let production = fs::read_to_string(root.join("crates/j2k-native/src/ht_adapter.rs"))
        .expect("read j2k-native HT adapter module");
    let tests = fs::read_to_string(root.join("crates/j2k-native/src/ht_adapter/tests.rs"))
        .expect("read j2k-native HT adapter tests");

    assert_pattern_checks(&[
        PatternCheck::new("j2k-native HT adapter test ownership", &production)
            .required(&["#[cfg(test)]\nmod tests;"]),
        PatternCheck::new("j2k-native HT adapter ownership regressions", &tests).required(&[
            "fn prepared_state_owns_inputs_and_decodes_exact_sigprop_output()",
            "fn short_output_fails_transactionally_and_state_remains_usable()",
            "fn overflowing_segment_metadata_is_rejected_before_state_construction()",
        ]),
    ]);
    assert!(
        production.lines().count() < 125,
        "j2k-native HT adapter must stay below its focused production line-count ratchet"
    );
    assert!(
        tests.lines().count() < 100,
        "j2k-native HT adapter tests must stay below their focused line-count ratchet"
    );
}
