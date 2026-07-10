// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;

use super::*;

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
            "pub(crate) use self::tier2_packetization::*;",
        ])
        .forbidden(&["pub(crate) fn encode_tier2_packetization"]),
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
        ("compute/decode_dispatch/classic_cleanup.rs", 750),
        ("compute/decode_dispatch/classic_subband.rs", 450),
        ("compute/decode_dispatch/ht_distinct.rs", 250),
        ("compute/decode_dispatch/ht_subband.rs", 300),
        ("compute/decode_dispatch/idwt.rs", 550),
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

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "direct-plan type ownership and source ratchets form one structural contract"
)]
fn metal_direct_plan_types_live_in_focused_module() {
    let root = repo_root();
    let compute = fs::read_to_string(root.join("crates/j2k-metal/src/compute.rs"))
        .expect("read Metal compute module");
    let direct_execute =
        fs::read_to_string(root.join("crates/j2k-metal/src/compute/direct_execute.rs"))
            .expect("read Metal direct execute module");
    let plan_types =
        fs::read_to_string(root.join("crates/j2k-metal/src/compute/direct_plan_types.rs"))
            .expect("read Metal direct plan types module");
    let plane_pack =
        fs::read_to_string(root.join("crates/j2k-metal/src/compute/direct_plane_pack.rs"))
            .expect("read Metal direct plane-pack module");
    let prepare = fs::read_to_string(root.join("crates/j2k-metal/src/compute/direct_prepare.rs"))
        .expect("read Metal direct prepare module");
    let roi = fs::read_to_string(root.join("crates/j2k-metal/src/compute/direct_roi.rs"))
        .expect("read Metal direct ROI module");
    let grayscale_execute =
        fs::read_to_string(root.join("crates/j2k-metal/src/compute/direct_grayscale_execute.rs"))
            .expect("read Metal direct grayscale executor module");
    let grayscale_component = fs::read_to_string(
        root.join("crates/j2k-metal/src/compute/direct_grayscale_execute/component_plane.rs"),
    )
    .expect("read Metal direct grayscale component-plane executor");
    let grayscale_single = fs::read_to_string(
        root.join("crates/j2k-metal/src/compute/direct_grayscale_execute/single.rs"),
    )
    .expect("read Metal direct single-grayscale executor");
    let stacked_batch =
        fs::read_to_string(root.join("crates/j2k-metal/src/compute/direct_stacked_batch.rs"))
            .expect("read Metal direct stacked batch module");
    let repeated_grayscale = fs::read_to_string(
        root.join("crates/j2k-metal/src/compute/direct_stacked_batch/repeated_grayscale.rs"),
    )
    .expect("read Metal repeated-grayscale shell");
    let repeated_grayscale_execution =
        fs::read_to_string(root.join(
            "crates/j2k-metal/src/compute/direct_stacked_batch/repeated_grayscale/execution.rs",
        ))
        .expect("read Metal repeated-grayscale execution module");
    let surface_pack =
        fs::read_to_string(root.join("crates/j2k-metal/src/compute/direct_surface_pack.rs"))
            .expect("read Metal direct surface-pack module");

    assert!(
        compute.lines().count() < 450,
        "compute.rs must stay below the honest post-include-removal line-count ratchet"
    );
    assert!(
        direct_execute.lines().count() < 150,
        "direct_execute.rs must remain a focused planning and execution module"
    );

    assert_pattern_checks(&[
        PatternCheck::new("Metal compute direct module ownership", &compute)
            .required(&[
                "mod direct_plan_types;",
                "pub(crate) use self::direct_plan_types",
                "mod direct_execute;",
                "pub(crate) use self::direct_execute",
            ])
            .forbidden(&[
                "include!(\"compute/direct_execute_impl.rs\")",
                "fn prepare_direct_color_plan_with_tier1_mode(",
            ]),
        PatternCheck::new("Metal direct execute module", &direct_execute)
            .required(&[
                "direct_plan_types::{",
                "direct_prepare::{",
                "direct_roi::crop_prepared_direct_grayscale_plan_to_output_region",
                "direct_tier1::DirectTier1Mode",
                "pub(crate) fn prepare_direct_color_plan(",
                "pub(crate) fn prepare_direct_color_plan_for_cpu_upload(",
                "pub(crate) fn crop_prepared_direct_color_plan_to_output_region(",
            ])
            .forbidden(&[
                "include!(",
                "use super::*",
                "mod direct_plan_types;",
                "mod direct_prepare;",
                "mod direct_roi;",
            ]),
    ]);
    assert_pattern_checks(&[
        PatternCheck::new("Metal direct grayscale execution shell", &grayscale_execute)
            .required(&[
                "mod component_plane;",
                "mod single;",
                "pub(in crate::compute) use self::component_plane::*;",
                "pub(in crate::compute) use self::single::*;",
            ])
            .forbidden(&["fn encode_prepared_direct_component_plane_in_command_buffer("]),
        PatternCheck::new(
            "Metal repeated grayscale execution shell",
            &repeated_grayscale,
        )
        .required(&[
            "mod execution;",
            "encode_repeated_direct_grayscale_plan_in_command_buffer",
        ])
        .forbidden(&["struct RepeatedGrayscaleExecution"]),
        PatternCheck::new(
            "Metal repeated grayscale execution implementation",
            &repeated_grayscale_execution,
        )
        .required(&[
            "struct RepeatedGrayscaleExecution",
            "fn encode_step(&mut self",
            "fn encode_stacked_idwt(&mut self",
            "fn encode_per_instance_idwt(&mut self",
        ]),
    ]);
    for (relative, source, max_lines) in [
        (
            "compute/direct_grayscale_execute.rs",
            &grayscale_execute,
            500,
        ),
        (
            "compute/direct_grayscale_execute/single.rs",
            &grayscale_single,
            350,
        ),
        (
            "compute/direct_grayscale_execute/component_plane.rs",
            &grayscale_component,
            450,
        ),
        (
            "compute/direct_stacked_batch/repeated_grayscale.rs",
            &repeated_grayscale,
            100,
        ),
        (
            "compute/direct_stacked_batch/repeated_grayscale/execution.rs",
            &repeated_grayscale_execution,
            600,
        ),
    ] {
        assert!(
            source.lines().count() < max_lines,
            "{relative} must stay below its direct-executor line-count ratchet"
        );
    }
    for item in [
        "pub(crate) struct PreparedDirectGrayscalePlan",
        "pub(crate) struct PreparedDirectColorPlan",
        "pub(super) enum PreparedDirectGrayscaleStep",
        "pub(super) struct PreparedDirectIdwt",
        "pub(super) struct PreparedClassicSubBand",
        "pub(super) struct PreparedClassicSubBandGroup",
        "pub(super) struct PreparedHtSubBand",
        "pub(super) struct PreparedHtSubBandGroup",
    ] {
        assert_pattern_checks(&[
            PatternCheck::new(
                "direct_execute.rs direct plan type exclusion",
                &direct_execute,
            )
            .forbidden(&[item]),
            PatternCheck::new(
                "compute/direct_plan_types.rs direct plan type ownership",
                &plan_types,
            )
            .required(&[item]),
        ]);
    }
    for required in [
        (
            "mod direct_plane_pack;",
            &plane_pack,
            "pub(super) struct PlaneStage",
        ),
        (
            "mod direct_prepare;",
            &prepare,
            "pub(crate) fn prepare_direct_grayscale_plan",
        ),
        (
            "mod direct_roi;",
            &roi,
            "pub(crate) fn crop_prepared_direct_grayscale_plan_to_output_region",
        ),
        (
            "mod direct_grayscale_execute;",
            &grayscale_component,
            "pub(in crate::compute) fn encode_prepared_direct_component_plane_in_command_buffer",
        ),
        (
            "mod direct_stacked_batch;",
            &stacked_batch,
            "pub(super) fn encode_stacked_direct_component_plane_batch",
        ),
        (
            "mod direct_surface_pack;",
            &surface_pack,
            "pub(super) fn output_shape_for",
        ),
    ] {
        let (module_wire, module_source, owned_item) = required;
        assert_pattern_checks(&[
            PatternCheck::new("compute.rs direct child module wiring", &compute)
                .required(&[module_wire]),
            PatternCheck::new("direct split module owned item", module_source)
                .required(&[owned_item]),
            PatternCheck::new("direct_execute.rs owned item exclusion", &direct_execute)
                .forbidden(&[owned_item]),
        ]);
    }
}
