// SPDX-License-Identifier: MIT OR Apache-2.0

//! Stage telemetry for region-scaled hybrid plan construction.

use crate::profile_env::{decode_profile_label, MetalDirectProfileRow, MetalProfileFormat};

pub(super) fn emit_region_scaled_color_plan_build_timings(
    native_image_us: u128,
    direct_plan_us: u128,
    prepare_us: u128,
    crop_us: u128,
    total_us: u128,
) {
    if !crate::compute::metal_profile_stages_enabled() {
        return;
    }

    let label = match decode_profile_label() {
        Ok(label) => label,
        Err(error) => {
            j2k_profile::emit_profile_error("metal_hybrid_plan_label", &error);
            return;
        }
    };
    for (stage, elapsed_us) in [
        ("native_image", native_image_us),
        ("direct_color_plan", direct_plan_us),
        ("prepare_cpu_upload", prepare_us),
        ("crop_prepared_plan", crop_us),
        ("plan_total", total_us),
    ] {
        let processor = plan_stage_processor(stage);
        let metric = plan_stage_metric(stage);
        let metric_kind = plan_stage_metric_kind(stage);
        let aggregation = plan_stage_aggregation(stage);
        crate::profile_env::emit_metal_profile_row(
            "j2k",
            "decode",
            "metal_cpu_hybrid_plan",
            &MetalDirectProfileRow {
                pipeline: "decode_hybrid",
                label: &label,
                stage,
                processor,
                metric,
                metric_kind,
                aggregation,
                fmt: MetalProfileFormat::Family("Rgb"),
                batch_count: 1,
                elapsed_us,
            },
        );
    }
}

fn plan_stage_processor(stage: &str) -> &'static str {
    match stage {
        "native_image" | "direct_color_plan" | "prepare_cpu_upload" | "crop_prepared_plan" => "cpu",
        _ => "hybrid",
    }
}

fn plan_stage_metric(stage: &str) -> &'static str {
    match stage {
        "native_image" => "native_image_us",
        "direct_color_plan" => "direct_color_plan_us",
        "prepare_cpu_upload" => "prepare_cpu_upload_us",
        "crop_prepared_plan" => "crop_prepared_plan_us",
        "plan_total" => "plan_total_us",
        _ => "wall_us",
    }
}

fn plan_stage_metric_kind(_stage: &str) -> &'static str {
    "wall_elapsed"
}

fn plan_stage_aggregation(stage: &str) -> &'static str {
    match stage {
        "plan_total" => "inclusive",
        _ => "exclusive",
    }
}
