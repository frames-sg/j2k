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
