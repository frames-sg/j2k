// SPDX-License-Identifier: MIT OR Apache-2.0

//! Responsibility ratchets for native direct-plan construction.

use std::fs;

use super::{assert_pattern_checks, repo_root, PatternCheck};

#[test]
fn native_direct_plan_storage_is_split_by_component_and_sub_band_ownership() {
    let root = repo_root();
    let component =
        fs::read_to_string(root.join("crates/j2k-native/src/j2c/decode/direct_plan/storage.rs"))
            .expect("read native direct-plan component storage");
    let sub_band = fs::read_to_string(
        root.join("crates/j2k-native/src/j2c/decode/direct_plan/storage/sub_band.rs"),
    )
    .expect("read native direct-plan sub-band storage");

    assert_pattern_checks(&[
        PatternCheck::new("native direct-plan component assembly", &component)
            .required(&[
                "mod sub_band;",
                "fn build_component_plan_from_storage(",
                "fn append_idwt_steps(",
                "fn direct_store_geometry(",
                "J2kDirectStoreStep",
            ])
            .forbidden(&[
                "fn build_grayscale_sub_band_step(",
                "HtOwnedCodeBlockBatchJob",
                "J2kOwnedCodeBlockBatchJob",
                "fn encoded_input_range(",
                "fn strip_grayscale_payload_owners(",
                "fn strip_classic_payload_owners(",
                "clippy::too_many_lines",
            ]),
        PatternCheck::new(
            "native direct-plan sub-band payload construction",
            &sub_band,
        )
        .required(&[
            "fn build_grayscale_sub_band_step(",
            "fn build_ht_sub_band_step(",
            "fn build_classic_sub_band_step(",
            "HtOwnedCodeBlockBatchJob",
            "J2kOwnedCodeBlockBatchJob",
            "fn encoded_input_range(",
            "fn strip_grayscale_payload_owners(",
            "fn strip_classic_payload_owners(",
        ])
        .forbidden(&[
            "fn build_component_plan_from_storage(",
            "fn append_idwt_steps(",
            "fn direct_store_geometry(",
            "J2kDirectIdwtStep",
            "J2kDirectStoreStep",
            "clippy::too_many_lines",
        ]),
    ]);

    assert!(
        component.lines().count() < 350,
        "native direct-plan component assembly must stay below its focused 350-line ratchet"
    );
    assert!(
        sub_band.lines().count() < 425,
        "native direct-plan sub-band payload construction must stay below its focused 425-line ratchet"
    );
}

#[test]
fn referenced_direct_plan_tests_keep_focused_owners() {
    let root = repo_root();
    let direct_plan = fs::read_to_string(root.join("crates/j2k-native/src/direct_plan.rs"))
        .expect("read j2k-native direct plan");
    let referenced =
        fs::read_to_string(root.join("crates/j2k-native/src/direct_plan/referenced_ht.rs"))
            .expect("read referenced HT direct plan");
    assert!(direct_plan.contains("mod referenced_ht;"));
    assert!(direct_plan.contains("pub use referenced_ht::"));
    assert!(!direct_plan.contains("pub enum J2kReferencedHtj2kPlan"));
    assert!(referenced.contains("pub enum J2kReferencedHtj2kPlan"));
    assert!(!referenced
        .lines()
        .any(|line| line.trim() == "use super::*;"));
    assert!(direct_plan.lines().count() < 250);

    let tests = fs::read_to_string(root.join("crates/j2k-native/src/tests.rs"))
        .expect("read native tests shell");
    let workspace = fs::read_to_string(root.join("crates/j2k-native/src/tests/workspace_reuse.rs"))
        .expect("read native workspace-reuse tests");
    let symbol = "fn decoder_workspace_reuses_component_owners_across_distinct_input_lifetimes";
    assert!(tests.contains("mod workspace_reuse;"));
    assert!(!tests.contains(symbol));
    assert!(workspace.contains(symbol));
    assert!(workspace.lines().count() < 300);
    assert!(!workspace.lines().any(|line| line.trim() == "use super::*;"));
}
