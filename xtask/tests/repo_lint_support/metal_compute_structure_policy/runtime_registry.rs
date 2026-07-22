// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;

use crate::repo_lint_support::{assert_pattern_checks, repo_root, PatternCheck};

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "the Metal compute registry split is enforced by one cohesive ownership matrix"
)]
fn metal_compute_runtime_registry_is_split_from_compute_god_file() {
    let root = repo_root();
    let compute = fs::read_to_string(root.join("crates/j2k-metal/src/compute.rs"))
        .expect("read Metal compute module");
    let runtime = fs::read_to_string(root.join("crates/j2k-metal/src/compute/runtime.rs"))
        .expect("read Metal compute runtime module");
    let forward_transform =
        fs::read_to_string(root.join("crates/j2k-metal/src/compute/forward_transform.rs"))
            .expect("read Metal compute forward-transform module");
    let resident_tier1 =
        fs::read_to_string(root.join("crates/j2k-metal/src/compute/resident_tier1.rs"))
            .expect("read Metal compute resident tier1 module");
    let resident_tier1_types =
        fs::read_to_string(root.join("crates/j2k-metal/src/compute/resident_tier1/types.rs"))
            .expect("read Metal compute resident tier1 types module");
    let lossless_prepare =
        fs::read_to_string(root.join("crates/j2k-metal/src/compute/lossless_prepare.rs"))
            .expect("read Metal compute lossless prepare module");
    let lossless_prepare_single =
        fs::read_to_string(root.join("crates/j2k-metal/src/compute/lossless_prepare/single.rs"))
            .expect("read Metal compute single-item lossless prepare module");
    let decode_dispatch =
        fs::read_to_string(root.join("crates/j2k-metal/src/compute/decode_dispatch.rs"))
            .expect("read Metal compute decode dispatch module");
    let decode_dispatch_mct =
        fs::read_to_string(root.join("crates/j2k-metal/src/compute/decode_dispatch/mct.rs"))
            .expect("read Metal compute decode MCT module");
    let tier1_encode =
        fs::read_to_string(root.join("crates/j2k-metal/src/compute/tier1_encode.rs"))
            .expect("read Metal compute tier1 encode module");
    let resident_codestream =
        fs::read_to_string(root.join("crates/j2k-metal/src/compute/resident_codestream.rs"))
            .expect("read Metal compute resident codestream module");
    let resident_codestream_tier2 = fs::read_to_string(
        root.join("crates/j2k-metal/src/compute/resident_codestream/tier2_packetization.rs"),
    )
    .expect("read Metal compute resident codestream tier-2 packetization module");
    let resident_codestream_tier2_tests = fs::read_to_string(
        root.join("crates/j2k-metal/src/compute/resident_codestream/tier2_packetization/tests.rs"),
    )
    .expect("read Metal compute resident codestream tier-2 packetization tests");
    let resident_codestream_ht_cleanup = fs::read_to_string(
        root.join("crates/j2k-metal/src/compute/resident_codestream/ht_cleanup.rs"),
    )
    .expect("read Metal compute resident codestream HT cleanup module");
    let resident_codestream_classic_labels = fs::read_to_string(
        root.join("crates/j2k-metal/src/compute/resident_codestream/classic_labels.rs"),
    )
    .expect("read Metal compute resident codestream classic labels module");
    let decode_cleanup =
        fs::read_to_string(root.join("crates/j2k-metal/src/compute/decode_cleanup.rs"))
            .expect("read Metal compute decode cleanup module");
    assert!(
        compute.lines().count() < 450,
        "compute.rs must stay below the honest post-include-removal line-count ratchet"
    );
    assert!(
        resident_codestream.lines().count() < 150,
        "resident_codestream.rs must remain a focused module shell"
    );
    assert!(
        resident_codestream_tier2.lines().count() < 500,
        "tier2_packetization.rs must stay below its post-extraction line-count ratchet"
    );
    assert!(
        resident_codestream_tier2_tests.lines().count() < 200,
        "tier2_packetization/tests.rs must stay below its focused-test line-count ratchet"
    );
    assert!(
        resident_codestream_tier2.contains("#[cfg(test)]\nmod tests;")
            && !resident_codestream_tier2
                .contains("fn tier2_metadata_plan_honors_exact_aggregate_cap"),
        "Tier-2 packetization regression tests must remain in their focused test module"
    );
    assert!(
        resident_codestream.contains("mod ht_cleanup;")
            && resident_codestream_ht_cleanup
                .contains("pub(in crate::compute) fn dispatch_ht_cleanup")
            && !resident_codestream.contains("fn dispatch_ht_cleanup("),
        "resident_codestream HT cleanup dispatch helpers must live in resident_codestream/ht_cleanup.rs"
    );
    assert!(
        resident_codestream.contains("mod classic_labels;")
            && resident_codestream_classic_labels.contains("CLASSIC_TIER1_DENSITY_LABEL")
            && resident_codestream_classic_labels.contains("next_enabled_classic_stage_label")
            && !resident_codestream.contains("const CLASSIC_TIER1_DENSITY_LABEL")
            && !resident_codestream.contains("fn next_enabled_classic_stage_label("),
        "resident_codestream classic profiling labels must live in resident_codestream/classic_labels.rs"
    );

    assert_pattern_checks(&[
        PatternCheck::new("Metal compute runtime module shell", &compute)
            .required(&[
                "mod runtime;",
                "pub(crate) use self::runtime",
                "MetalRuntime",
                "runtime_initialization_error",
            ])
            .forbidden(&[
                "pub(crate) struct MetalRuntime",
                "MetalPipelineLoader::new(device",
            ]),
        PatternCheck::new("Metal compute runtime implementation", &runtime).required(&[
            "pub(crate) struct MetalRuntime",
            "MetalPipelineLoader::new(device",
            "DEFAULT_METAL_SESSION",
            "METAL_RUNTIME_OVERRIDE",
            "with_runtime_for_session",
        ]),
        PatternCheck::new("Metal resident codestream module wiring", &compute)
            .required(&["mod resident_codestream;"])
            .forbidden(&["pub(crate) fn encode_tier2_packetization"]),
        PatternCheck::new(
            "Metal resident codestream tier-2 module shell",
            &resident_codestream,
        )
        .required(&[
            "mod tier2_packetization;",
            "pub(crate) use self::tier2_packetization::encode_tier2_packetization;",
        ])
        .forbidden(&[
            "pub(crate) use self::tier2_packetization::*;",
            "pub(crate) fn encode_tier2_packetization",
        ]),
        PatternCheck::new(
            "Metal resident codestream tier-2 packetization implementation",
            &resident_codestream_tier2,
        )
        .required(&["pub(crate) fn encode_tier2_packetization"]),
        PatternCheck::new("Metal resident Tier-1 module shell", &resident_tier1)
            .required(&[
                "mod counter_validation;",
                "mod profile_dispatch;",
                "mod readback;",
                "mod result_harvest;",
                "mod types;",
            ])
            .forbidden(&["pub(crate) struct J2kLosslessDeviceCodeBlock"]),
        PatternCheck::new("Metal lossless prepare module shell", &lossless_prepare)
            .required(&[
                "mod batch;",
                "mod batch_item;",
                "mod commands;",
                "mod forward_encode;",
                "mod single;",
                "mod sizes;",
            ])
            .forbidden(&["pub(crate) fn prepare_lossless_device_code_blocks("]),
        PatternCheck::new("Metal decode dispatch module shell", &decode_dispatch)
            .required(&[
                "mod classic_cleanup;",
                "mod classic_subband;",
                "mod ht_distinct;",
                "mod ht_subband;",
                "mod idwt;",
                "mod mct;",
                "mod store;",
            ])
            .forbidden(&["pub(crate) fn decode_inverse_mct"]),
    ]);
    for (relative, max_lines) in [
        ("compute/resident_tier1.rs", 100),
        ("compute/resident_tier1/types.rs", 450),
        ("compute/resident_tier1/readback.rs", 250),
        ("compute/resident_tier1/result_harvest.rs", 500),
        ("compute/resident_tier1/profile_dispatch/analysis.rs", 500),
        ("compute/resident_tier1/profile_dispatch/tokens.rs", 700),
        ("compute/resident_tier1/counter_validation/record.rs", 500),
        ("compute/resident_tier1/counter_validation/validate.rs", 500),
        ("compute/lossless_prepare.rs", 100),
        ("compute/lossless_prepare/batch.rs", 150),
        ("compute/lossless_prepare/batch_item.rs", 400),
        ("compute/lossless_prepare/commands.rs", 700),
        ("compute/lossless_prepare/forward_encode.rs", 300),
        ("compute/lossless_prepare/single.rs", 200),
        ("compute/decode_dispatch.rs", 100),
        ("compute/decode_dispatch/classic_cleanup.rs", 525),
        (
            "compute/decode_dispatch/classic_cleanup/status_sources.rs",
            100,
        ),
        (
            "compute/decode_dispatch/classic_cleanup/distinct_allocation.rs",
            100,
        ),
        (
            "compute/decode_dispatch/classic_cleanup/distinct_batch.rs",
            350,
        ),
        ("compute/decode_dispatch/classic_subband.rs", 450),
        ("compute/decode_dispatch/ht_distinct.rs", 250),
        ("compute/decode_dispatch/ht_subband.rs", 300),
        ("compute/decode_dispatch/idwt.rs", 550),
        ("compute/decode_dispatch/idwt/irreversible.rs", 325),
        ("compute/decode_dispatch/mct.rs", 250),
        ("compute/decode_dispatch/store.rs", 400),
    ] {
        let path = root.join("crates/j2k-metal/src").join(relative);
        let source = fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
        assert!(
            source.lines().count() < max_lines,
            "{} must stay below its focused-module line-count ratchet",
            path.display()
        );
    }
    for (module_wire, module_source, owned_item) in [
        (
            "mod forward_transform;",
            &forward_transform,
            "pub(crate) fn encode_forward_dwt53",
        ),
        (
            "mod resident_tier1;",
            &resident_tier1_types,
            "pub(crate) struct J2kLosslessDeviceCodeBlock",
        ),
        (
            "mod lossless_prepare;",
            &lossless_prepare_single,
            "pub(crate) fn prepare_lossless_device_code_blocks",
        ),
        (
            "mod decode_dispatch;",
            &decode_dispatch_mct,
            "pub(crate) fn decode_inverse_mct",
        ),
        (
            "mod tier1_encode;",
            &tier1_encode,
            "pub(crate) fn encode_classic_tier1_code_blocks",
        ),
        (
            "mod decode_cleanup;",
            &decode_cleanup,
            "pub(crate) fn decode_classic_cleanup_code_block",
        ),
    ] {
        assert_pattern_checks(&[
            PatternCheck::new("Metal compute module wiring", &compute).required(&[module_wire]),
            PatternCheck::new("split Metal compute module owned item", module_source)
                .required(&[owned_item]),
            PatternCheck::new("Metal compute module shell owned-item exclusion", &compute)
                .forbidden(&[owned_item]),
        ]);
    }
}
