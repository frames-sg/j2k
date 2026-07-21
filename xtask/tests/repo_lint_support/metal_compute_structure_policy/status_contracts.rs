// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;

use crate::repo_lint_support::{assert_pattern_checks, repo_root, PatternCheck};

#[test]
fn metal_inverse_idwt_has_no_unwritten_status_contract() {
    let root = repo_root();
    let irreversible = fs::read_to_string(
        root.join("crates/j2k-metal/src/compute/decode_dispatch/idwt/irreversible.rs"),
    )
    .expect("read Metal irreversible IDWT dispatch");
    let direct_status =
        fs::read_to_string(root.join("crates/j2k-metal/src/compute/direct_status.rs"))
            .expect("read Metal direct status lifecycle");
    let abi = fs::read_to_string(root.join("crates/j2k-metal/src/compute/abi.rs"))
        .expect("read Metal compute ABI");
    let shader = fs::read_to_string(root.join("crates/j2k-metal/src/idwt.metal"))
        .expect("read Metal IDWT shader");

    assert_pattern_checks(&[
        PatternCheck::new("irreversible IDWT dispatch", &irreversible).forbidden(&[
            "J2kIdwtStatus",
            "DirectStatusCheck::Idwt",
            "decode_idwt_status_error",
            "_status_buffer",
        ]),
        PatternCheck::new("direct status lifecycle", &direct_status)
            .forbidden(&["Idwt(Buffer)", "DirectStatusCheck::Idwt"]),
        PatternCheck::new("Metal compute ABI", &abi)
            .forbidden(&["struct J2kIdwtStatus", "J2K_IDWT_STATUS_"]),
        PatternCheck::new("Metal IDWT shader", &shader)
            .forbidden(&["struct J2kIdwtStatus", "J2K_IDWT_STATUS_"]),
    ]);
}

#[test]
fn metal_inverse_mct_has_no_unreachable_status_contract() {
    let root = repo_root();
    let dispatch =
        fs::read_to_string(root.join("crates/j2k-metal/src/compute/decode_dispatch/mct.rs"))
            .expect("read Metal inverse MCT dispatch");
    let direct_status =
        fs::read_to_string(root.join("crates/j2k-metal/src/compute/direct_status.rs"))
            .expect("read Metal direct status lifecycle");
    let shader = fs::read_to_string(root.join("crates/j2k-metal/src/mct.metal"))
        .expect("read Metal MCT shader");
    let inverse_shader = shader
        .split_once("kernel void j2k_inverse_mct(")
        .and_then(|(_, tail)| tail.split_once("kernel void j2k_forward_rct("))
        .map(|(inverse, _)| inverse)
        .expect("isolate inverse MCT shader");

    assert_pattern_checks(&[
        PatternCheck::new("inverse MCT dispatch", &dispatch).forbidden(&[
            "DirectStatusCheck",
            "J2kMctStatus",
            "decode_mct_status_error",
            "status_buffer",
        ]),
        PatternCheck::new("direct status lifecycle", &direct_status)
            .forbidden(&["Mct(Buffer)", "DirectStatusCheck::Mct"]),
        PatternCheck::new("inverse MCT shader", inverse_shader)
            .forbidden(&["J2kMctStatus", "status->"]),
    ]);
}

#[test]
fn metal_status_attribution_has_no_impossible_optional_success() {
    let source =
        fs::read_to_string(repo_root().join("crates/j2k-metal/src/compute/direct_status.rs"))
            .expect("read Metal direct status lifecycle");

    assert!(source.contains(") -> Result<&mut Vec<usize>, Error> {"));
    assert!(!source.contains("Result<Option<&mut Vec<usize>>, Error>"));
    assert!(!source.contains("let Some(indices) = self.attributed_sources_mut"));
}
