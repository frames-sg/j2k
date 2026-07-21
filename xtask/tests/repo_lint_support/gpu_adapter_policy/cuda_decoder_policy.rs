// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;

use super::super::{assert_pattern_checks, repo_root, PatternCheck};

mod allocations;
mod architecture;
mod color_runtime;
mod direct_plan;
mod resident_leaf_structure;

struct CudaDecoderSources {
    decoder: String,
    api: String,
    plan: String,
    plan_grayscale: String,
    plan_color: String,
    plan_color_decoder: String,
    plan_color_owners: String,
    resident: String,
    resident_buffer_access: String,
    resident_cleanup_dequant: String,
    resident_component: String,
    resident_error: String,
    resident_idwt: String,
    resident_idwt_conversions: String,
    resident_routing: String,
    resident_surface: String,
    color_batch: String,
    color_batch_execution: String,
    color_batch_single: String,
    color_batch_finish: String,
    color_batch_finish_component: String,
    color_batch_finish_surface: String,
    color_batch_native: String,
    color_batch_native_completion: String,
    color_batch_native_execution: String,
    color_batch_native_prepare: String,
    color_batch_host_owners: String,
    color_store: String,
    color_store_batch: String,
    color_store_validation: String,
    profile: String,
}

impl CudaDecoderSources {
    fn read() -> Self {
        let root = repo_root();
        let read = |relative: &str| {
            fs::read_to_string(root.join(relative))
                .unwrap_or_else(|error| panic!("read {relative}: {error}"))
        };
        Self {
            decoder: read("crates/j2k-cuda/src/decoder.rs"),
            api: read("crates/j2k-cuda/src/decoder/api.rs"),
            plan: read("crates/j2k-cuda/src/decoder/plan.rs"),
            plan_grayscale: read("crates/j2k-cuda/src/decoder/plan/grayscale.rs"),
            plan_color: read("crates/j2k-cuda/src/decoder/plan/color.rs"),
            plan_color_decoder: read("crates/j2k-cuda/src/decoder/plan/color_decoder.rs"),
            plan_color_owners: read("crates/j2k-cuda/src/decoder/plan/color_owners.rs"),
            resident: read("crates/j2k-cuda/src/decoder/resident.rs"),
            resident_buffer_access: read("crates/j2k-cuda/src/decoder/resident/buffer_access.rs"),
            resident_cleanup_dequant: read(
                "crates/j2k-cuda/src/decoder/resident/cleanup_dequant.rs",
            ),
            resident_component: read("crates/j2k-cuda/src/decoder/resident/component.rs"),
            resident_error: read("crates/j2k-cuda/src/decoder/resident/error.rs"),
            resident_idwt: read("crates/j2k-cuda/src/decoder/resident/idwt.rs"),
            resident_idwt_conversions: read(
                "crates/j2k-cuda/src/decoder/resident/idwt/conversions.rs",
            ),
            resident_routing: read("crates/j2k-cuda/src/decoder/resident/routing.rs"),
            resident_surface: read("crates/j2k-cuda/src/decoder/resident/surface.rs"),
            color_batch: read("crates/j2k-cuda/src/decoder/color_batch.rs"),
            color_batch_execution: read(
                "crates/j2k-cuda/src/decoder/color_batch/batch_execution.rs",
            ),
            color_batch_single: read("crates/j2k-cuda/src/decoder/color_batch/single.rs"),
            color_batch_finish: read("crates/j2k-cuda/src/decoder/color_batch/finish.rs"),
            color_batch_finish_component: read(
                "crates/j2k-cuda/src/decoder/color_batch/finish/component.rs",
            ),
            color_batch_finish_surface: read(
                "crates/j2k-cuda/src/decoder/color_batch/finish/surface.rs",
            ),
            color_batch_native: read("crates/j2k-cuda/src/decoder/color_batch/native_batch.rs"),
            color_batch_native_completion: read(
                "crates/j2k-cuda/src/decoder/color_batch/native_batch/completion.rs",
            ),
            color_batch_native_execution: read(
                "crates/j2k-cuda/src/decoder/color_batch/native_batch/execution.rs",
            ),
            color_batch_native_prepare: read(
                "crates/j2k-cuda/src/decoder/color_batch/native_batch/prepare.rs",
            ),
            color_batch_host_owners: read("crates/j2k-cuda/src/decoder/color_batch/host_owners.rs"),
            color_store: read("crates/j2k-cuda/src/decoder/color_batch/store.rs"),
            color_store_batch: read("crates/j2k-cuda/src/decoder/color_batch/store/batch.rs"),
            color_store_validation: read(
                "crates/j2k-cuda/src/decoder/color_batch/store/validation.rs",
            ),
            profile: read("crates/j2k-cuda/src/decoder/profile.rs"),
        }
    }
}

#[test]
fn focused_modules_stay_below_line_ratchets() {
    let sources = CudaDecoderSources::read();
    assert_facade_and_plan_line_ratchets(&sources);
    assert_resident_line_ratchets(&sources);
    assert_color_line_ratchets(&sources);
}

fn assert_facade_and_plan_line_ratchets(sources: &CudaDecoderSources) {
    assert!(
        sources.decoder.lines().count() < 1_500,
        "j2k-cuda/src/decoder.rs must stay below the post-runtime-split line-count ratchet"
    );
    for (module_name, source, maximum_lines) in [
        ("api.rs", &sources.api, 1_800),
        ("plan.rs", &sources.plan, 75),
        ("plan/grayscale.rs", &sources.plan_grayscale, 475),
        ("plan/color.rs", &sources.plan_color, 200),
        ("plan/color_decoder.rs", &sources.plan_color_decoder, 275),
        ("profile.rs", &sources.profile, 1_800),
    ] {
        assert!(
            source.lines().count() < maximum_lines,
            "j2k-cuda/src/decoder/{module_name} must stay below the focused-module line-count ratchet"
        );
    }
    assert!(
        sources.plan_color_owners.lines().count() < 100,
        "j2k-cuda/src/decoder/plan/color_owners.rs must remain a focused owner-accounting leaf"
    );
}

fn assert_resident_line_ratchets(sources: &CudaDecoderSources) {
    for (module_name, source, maximum_lines) in [
        ("resident.rs", &sources.resident, 50),
        (
            "resident/cleanup_dequant.rs",
            &sources.resident_cleanup_dequant,
            325,
        ),
        ("resident/component.rs", &sources.resident_component, 225),
        (
            "resident/buffer_access.rs",
            &sources.resident_buffer_access,
            50,
        ),
        ("resident/error.rs", &sources.resident_error, 50),
        ("resident/idwt.rs", &sources.resident_idwt, 350),
        (
            "resident/idwt/conversions.rs",
            &sources.resident_idwt_conversions,
            75,
        ),
        ("resident/routing.rs", &sources.resident_routing, 425),
        ("resident/surface.rs", &sources.resident_surface, 175),
    ] {
        assert!(
            source.lines().count() < maximum_lines,
            "j2k-cuda/src/decoder/{module_name} must stay below its semantic-module line-count ratchet"
        );
    }
}

fn assert_color_line_ratchets(sources: &CudaDecoderSources) {
    assert!(
        sources.color_batch.lines().count() < 100,
        "j2k-cuda decoder/color_batch.rs must remain a facade"
    );
    for (module_name, source, maximum_lines) in [
        (
            "color_batch/batch_execution.rs",
            &sources.color_batch_execution,
            400,
        ),
        ("color_batch/single.rs", &sources.color_batch_single, 200),
        ("color_batch/finish.rs", &sources.color_batch_finish, 125),
        (
            "color_batch/finish/component.rs",
            &sources.color_batch_finish_component,
            100,
        ),
        (
            "color_batch/finish/surface.rs",
            &sources.color_batch_finish_surface,
            75,
        ),
        (
            "color_batch/native_batch.rs",
            &sources.color_batch_native,
            325,
        ),
        (
            "color_batch/native_batch/completion.rs",
            &sources.color_batch_native_completion,
            125,
        ),
        (
            "color_batch/native_batch/execution.rs",
            &sources.color_batch_native_execution,
            275,
        ),
        (
            "color_batch/native_batch/prepare.rs",
            &sources.color_batch_native_prepare,
            150,
        ),
    ] {
        assert!(
            source.lines().count() < maximum_lines,
            "j2k-cuda/src/decoder/{module_name} must stay below its focused-module line-count ratchet"
        );
    }
    assert!(
        sources.color_batch_host_owners.lines().count() < 125,
        "j2k-cuda decoder/color_batch/host_owners.rs must remain a focused owner-accounting leaf"
    );
    assert!(
        sources.color_store.lines().count() < 500,
        "j2k-cuda decoder/color_batch/store.rs must stay below its focused-module line-count ratchet"
    );
    assert!(
        sources.color_store_batch.lines().count() < 150,
        "j2k-cuda decoder/color_batch/store/batch.rs must remain a focused preparation leaf"
    );
    assert!(
        sources.color_store_validation.lines().count() < 100,
        "j2k-cuda decoder/color_batch/store/validation.rs must remain a focused validation leaf"
    );
}
