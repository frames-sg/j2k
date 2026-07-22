// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;

use super::super::{assert_pattern_checks, repo_root, PatternCheck};

mod allocations;
mod architecture;
mod color_runtime;
mod direct_plan;
mod line_ratchets;
mod native_color_batch;
mod resident_color_structure;
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
    color_batch_execution_completion: String,
    color_batch_execution_completion_batch_store: String,
    color_batch_execution_completion_fallback: String,
    color_batch_execution_execution: String,
    color_batch_execution_preparation: String,
    color_batch_single: String,
    color_batch_finish: String,
    color_batch_finish_component: String,
    color_batch_finish_surface: String,
    color_batch_native: String,
    color_batch_native_completion: String,
    color_batch_native_execution: String,
    color_batch_native_lifecycle: String,
    color_batch_native_prepare: String,
    color_batch_host_owners: String,
    color_store: String,
    color_store_batch: String,
    color_store_validation: String,
    grayscale_batch: String,
    grayscale_batch_completion: String,
    grayscale_batch_execution: String,
    grayscale_batch_preparation: String,
    grayscale_batch_store: String,
    grayscale_batch_tests: String,
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
            color_batch_execution_completion: read(
                "crates/j2k-cuda/src/decoder/color_batch/batch_execution/completion.rs",
            ),
            color_batch_execution_completion_batch_store: read(
                "crates/j2k-cuda/src/decoder/color_batch/batch_execution/completion/batch_store.rs",
            ),
            color_batch_execution_completion_fallback: read(
                "crates/j2k-cuda/src/decoder/color_batch/batch_execution/completion/fallback.rs",
            ),
            color_batch_execution_execution: read(
                "crates/j2k-cuda/src/decoder/color_batch/batch_execution/execution.rs",
            ),
            color_batch_execution_preparation: read(
                "crates/j2k-cuda/src/decoder/color_batch/batch_execution/preparation.rs",
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
            color_batch_native_lifecycle: read(
                "crates/j2k-cuda/src/decoder/color_batch/native_batch/execution/lifecycle.rs",
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
            grayscale_batch: read("crates/j2k-cuda/src/decoder/grayscale_batch.rs"),
            grayscale_batch_completion: read(
                "crates/j2k-cuda/src/decoder/grayscale_batch/completion.rs",
            ),
            grayscale_batch_execution: read(
                "crates/j2k-cuda/src/decoder/grayscale_batch/execution.rs",
            ),
            grayscale_batch_preparation: read(
                "crates/j2k-cuda/src/decoder/grayscale_batch/preparation.rs",
            ),
            grayscale_batch_store: read("crates/j2k-cuda/src/decoder/grayscale_batch/store.rs"),
            grayscale_batch_tests: read("crates/j2k-cuda/src/decoder/grayscale_batch/tests.rs"),
            profile: read("crates/j2k-cuda/src/decoder/profile.rs"),
        }
    }
}
