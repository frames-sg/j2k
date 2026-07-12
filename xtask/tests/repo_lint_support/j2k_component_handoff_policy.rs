// SPDX-License-Identifier: MIT OR Apache-2.0

//! Move-based, fallible J2K facade component handoff policy.

use std::fs;

use super::{assert_pattern_checks, repo_root, PatternCheck};

fn read(relative: &str) -> String {
    fs::read_to_string(repo_root().join(relative))
        .unwrap_or_else(|error| panic!("read {relative}: {error}"))
}

#[test]
fn native_component_handoff_moves_payload_owners_and_preserves_capacity_evidence() {
    let native = read("crates/j2k-native/src/color.rs");
    let facade = read("crates/j2k/src/decode/component_handoff.rs");
    assert_pattern_checks(&[
        PatternCheck::new("native component consuming seam", &native)
            .required(&[
                "pub fn allocated_bytes(&self) -> Option<usize>",
                "pub fn live_bytes(&self) -> usize",
                "pub fn into_parts(self)",
                "self.data.capacity()",
                "size_of::<NativeComponentPlane>()",
            ])
            .forbidden(&["#[derive(Debug, Clone)]\npub enum ColorSpace"]),
        PatternCheck::new("facade component move conversion", &facade)
            .required(&[
                ".allocated_bytes()",
                ".live_bytes()",
                "decoded.into_parts()",
                "plane.into_parts()",
                "try_destination_metadata",
                "values.capacity()",
                "BufferError::HostAllocationFailed",
            ])
            .forbidden(&[
                ".to_vec()",
                ".clone()",
                ".collect::<Vec",
                "#[derive(Debug, Clone, PartialEq, Eq)]\n#[non_exhaustive]\npub enum J2kDecodedColorSpace",
                "#[derive(Debug, Clone)]\npub struct J2kDecodedComponents",
                "#[derive(Debug, Clone, PartialEq, Eq)]\npub struct J2kNativeComponentPlane",
                "#[derive(Debug, Clone, PartialEq, Eq)]\npub struct J2kDecodedNativeComponents",
            ]),
    ]);
}

#[test]
fn facade_component_apis_propagate_handoff_errors_and_keep_decode_owner_focused() {
    let decode = read("crates/j2k/src/decode.rs");
    let handoff = read("crates/j2k/src/decode/component_handoff.rs");
    let view = read("crates/j2k/src/view.rs");
    assert_pattern_checks(&[
        PatternCheck::new("focused component handoff module", &decode)
            .required(&["mod component_handoff;", "pub use component_handoff::{"])
            .forbidden(&["pub struct J2kDecodedNativeComponents"]),
        PatternCheck::new("fallible view conversion", &view).required(&[
            "component_handoff_image_bytes(image)?",
            ".retained_allocation_bytes()",
            "J2kDecodedComponents::try_from_native(decoded, retained_image_bytes)",
            "J2kDecodedNativeComponents::try_from_native(decoded, retained_image_bytes)",
        ]),
        PatternCheck::new("component handoff regression evidence", &handoff).required(&[
            "component_handoff_has_an_exact_shared_cap_boundary",
            "native_plane_parts_move_payload_without_copying",
            "native_color_handoff_moves_icc_profile_without_copying",
        ]),
    ]);
    assert_eq!(
        view.matches("let retained_image_bytes = component_handoff_image_bytes(image)?;")
            .count(),
        4,
        "all four full/region borrowed/owned component entry points must retain the cached Image baseline through facade handoff"
    );
    assert_eq!(
        view.matches("::try_from_native(decoded, retained_image_bytes)")
            .count(),
        4,
        "all four component entry points must pass the cached Image baseline into facade handoff"
    );
}

#[test]
fn metal_plane_stage_reconstructs_only_heap_free_color_variants() {
    let metal = read("crates/j2k-metal/src/compute/direct_plane_pack.rs");
    assert_pattern_checks(&[
        PatternCheck::new("move-only Metal component color handoff", &metal)
            .required(&[
                "fn supported_plane_color_space(",
                "NativeColorSpace::Gray => Ok(NativeColorSpace::Gray)",
                "NativeColorSpace::RGB => Ok(NativeColorSpace::RGB)",
                "plane_stage_color_space_ownership_accepts_only_heap_free_variants",
                "profile.as_ptr()",
                "profile.capacity()",
            ])
            .forbidden(&["decoded.color_space().clone()"]),
    ]);
}
