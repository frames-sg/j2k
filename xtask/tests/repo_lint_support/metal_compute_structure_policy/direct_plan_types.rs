// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;

use super::direct_plan_ratchets;
use crate::repo_lint_support::{assert_pattern_checks, repo_root, PatternCheck};

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
    let direct_commands =
        fs::read_to_string(root.join("crates/j2k-metal/src/compute/direct_commands.rs"))
            .expect("read Metal direct command lifecycle module");
    let plan_types =
        fs::read_to_string(root.join("crates/j2k-metal/src/compute/direct_plan_types.rs"))
            .expect("read Metal direct plan types module");
    let plane_pack =
        fs::read_to_string(root.join("crates/j2k-metal/src/compute/direct_plane_pack.rs"))
            .expect("read Metal direct plane-pack module");
    let prepare = fs::read_to_string(root.join("crates/j2k-metal/src/compute/direct_prepare.rs"))
        .expect("read Metal direct prepare module");
    let prepare_grayscale =
        fs::read_to_string(root.join("crates/j2k-metal/src/compute/direct_prepare/grayscale.rs"))
            .expect("read Metal grayscale direct preparation module");
    let roi = fs::read_to_string(root.join("crates/j2k-metal/src/compute/direct_roi.rs"))
        .expect("read Metal direct ROI module");
    let grayscale_execute =
        fs::read_to_string(root.join("crates/j2k-metal/src/compute/direct_grayscale_execute.rs"))
            .expect("read Metal direct grayscale executor module");
    let grayscale_allocation = fs::read_to_string(
        root.join("crates/j2k-metal/src/compute/direct_grayscale_execute/allocation.rs"),
    )
    .expect("read Metal direct grayscale allocation module");
    let grayscale_component = fs::read_to_string(
        root.join("crates/j2k-metal/src/compute/direct_grayscale_execute/component_plane.rs"),
    )
    .expect("read Metal direct grayscale component-plane facade");
    let grayscale_component_execution = fs::read_to_string(root.join(
        "crates/j2k-metal/src/compute/direct_grayscale_execute/component_plane/execution.rs",
    ))
    .expect("read Metal direct grayscale component-plane execution stage");
    let grayscale_component_final_plane = fs::read_to_string(root.join(
        "crates/j2k-metal/src/compute/direct_grayscale_execute/component_plane/execution/final_plane.rs",
    ))
    .expect("read Metal direct grayscale final component-plane owner");
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
    let repeated_grayscale_reconstruction = fs::read_to_string(root.join(
        "crates/j2k-metal/src/compute/direct_stacked_batch/repeated_grayscale/execution/reconstruction.rs",
    ))
    .expect("read Metal repeated-grayscale reconstruction stage");
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
                "mod direct_commands;",
                "mod direct_plan_types;",
                "pub(crate) use self::direct_plan_types",
                "mod direct_execute;",
                "pub(crate) use self::direct_execute",
            ])
            .forbidden(&[
                "include!(\"compute/direct_execute_impl.rs\")",
                "fn prepare_direct_color_plan_with_tier1_mode(",
                "fn new_command_buffer(",
            ]),
        PatternCheck::new("Metal direct command lifecycle", &direct_commands).required(&[
            "fn new_command_buffer(",
            "fn new_compute_command_encoder(",
            "fn new_blit_command_encoder(",
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
        PatternCheck::new("Metal direct preparation facade", &prepare)
            .required(&[
                "mod classic;",
                "mod color;",
                "mod grayscale;",
                "mod ht;",
                "mod referenced;",
            ])
            .forbidden(&["fn prepare_direct_grayscale_plan("]),
    ]);
    assert_pattern_checks(&[
        PatternCheck::new("Metal direct grayscale execution shell", &grayscale_execute)
            .required(&[
                "mod allocation;",
                "mod component_plane;",
                "mod single;",
                "pub(in crate::compute) use self::component_plane::{",
                "checked_coefficient_len",
                "encode_prepared_direct_component_plane_in_command_buffer",
                "upload_cpu_decoded_coefficients",
                "DirectComponentPlaneRequest",
                "allocate_direct_execution_metadata, direct_ht_job_count, DirectExecutionMetadata,",
                "pub(in crate::compute) use self::single::encode_prepared_direct_grayscale_plan_in_command_buffer;",
            ])
            .forbidden(&[
                "pub(in crate::compute) use self::component_plane::*;",
                "pub(in crate::compute) use self::single::*;",
                "fn allocate_direct_execution_metadata(",
                "fn encode_prepared_direct_component_plane_in_command_buffer(",
            ]),
        PatternCheck::new("Metal component-plane facade", &grayscale_component)
            .required(&[
                "mod execution;",
                "pub(in crate::compute) use self::execution::encode_prepared_direct_component_plane_in_encoder;",
                "pub(in crate::compute) struct DirectComponentPlaneRequest",
                "fn encode_prepared_direct_component_plane_in_command_buffer(",
            ])
            .forbidden(&[
                "struct ComponentPlaneExecution",
                "fn encode_store(&mut self",
                "use super::*",
            ]),
        PatternCheck::new(
            "Metal component-plane execution stage",
            &grayscale_component_execution,
        )
        .required(&[
            "mod final_plane;",
            "struct ComponentPlaneExecution",
            "fn encode_store(&mut self",
            "self.bands.clear();",
            "fn encode_prepared_direct_component_plane_in_encoder(",
        ])
        .forbidden(&["struct RetainedComponentPlane", "use super::*"]),
        PatternCheck::new(
            "Metal final component-plane lifecycle",
            &grayscale_component_final_plane,
        )
        .required(&[
            "struct FinalComponentPlane",
            "struct RetainedComponentPlane",
            "fn buffer_for_store(",
            "fn validate_later_component_store(",
            "later_component_store_requires_exact_final_plane_geometry",
        ])
        .forbidden(&["use super::*"]),
        PatternCheck::new("Metal direct execution allocation owner", &grayscale_allocation)
            .required(&[
                "pub(super) struct DirectExecutionMetadata",
                "pub(super) fn allocate_direct_execution_metadata(",
                "direct_execution_resources_honor_exact_cap_and_one_byte_over",
            ]),
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
        .required(&["struct RepeatedGrayscaleExecution", "fn encode_step(&mut self"])
        .forbidden(&[
            "fn encode_stacked_idwt(&mut self",
            "fn encode_per_instance_idwt(&mut self",
        ]),
        PatternCheck::new(
            "Metal repeated grayscale reconstruction stage",
            &repeated_grayscale_reconstruction,
        )
        .required(&[
            "fn encode_stacked_idwt(",
            "fn encode_per_instance_idwt(",
        ]),
    ]);
    direct_plan_ratchets::assert_direct_executor_line_ratchets(root);
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
            &prepare_grayscale,
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
