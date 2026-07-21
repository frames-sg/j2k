// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{assert_pattern_checks, CudaDecoderSources, PatternCheck};

mod async_cleanup;
mod owner_completion;

#[test]
fn color_runtime_owns_store_and_queued_idwt_completion() {
    let sources = CudaDecoderSources::read();
    let finish = &sources.color_batch_finish;
    assert_color_decode_ownership(&sources, finish);
    assert_color_completion_ownership(&sources);
    assert!(
        finish.lines().count() < 125,
        "CUDA color completion orchestrator must stay focused"
    );
    owner_completion::assert_contract(&sources, finish);
}

fn assert_color_decode_ownership(sources: &CudaDecoderSources, finish: &str) {
    assert_pattern_checks(&[
        PatternCheck::new(
            "CUDA color surface orchestration ownership",
            &sources.color_batch,
        )
        .required(&[
            "mod store;",
            "mod finish;",
            "mod batch_execution;",
            "mod single;",
        ])
        .forbidden(&[
            "pub(super) mod store;",
            "fn run_pending_color_idwt(",
            "fn finish_color_components(",
            "fn finalize_color_surface(",
            "struct ColorStorePlan",
            "struct CudaPreparedRgb8MctBatchStore",
            "fn prepare_rgb8_mct_batch_store(",
            "fn rgb8_mct_batch_store_target(",
            "CudaJ2kStoreRgb8Job {",
            "fn run_color_mct(",
            "fn dispatch_color_store(",
        ]),
        PatternCheck::new("CUDA color completion orchestration", finish).required(&[
            "mod component;",
            "mod surface;",
            "fn finish_color_cuda_resident_surface_with_component_work(",
            "CudaQueuedIdwtBatch::resolve_optional_after_completed_work(",
        ]),
        PatternCheck::new(
            "CUDA color component-completion ownership",
            &sources.color_batch_finish_component,
        )
        .required(&[
            "fn run_pending_color_idwt(",
            "fn finish_color_components(",
            "struct PreparedColorComponents",
        ]),
        PatternCheck::new(
            "CUDA color surface-finalization ownership",
            &sources.color_batch_finish_surface,
        )
        .required(&[
            "fn finalize_color_surface(",
            "struct FinalizeColorSurfaceRequest",
            "SurfaceResidency::CudaResidentDecode",
        ]),
        PatternCheck::new(
            "CUDA color transform and store ownership",
            &sources.color_store,
        )
        .required(&[
            "mod batch;",
            "mod validation;",
            "struct ColorStorePlan",
            "fn run_color_mct(",
            "fn dispatch_color_store(",
            "fn dispatch_color_store_u8(",
            "fn dispatch_color_store_u16(",
            "store_route: ColorStoreRoute",
            "color_store_plan_builds_rgb_and_rgba_jobs_for_both_sample_widths",
            "color_store_plan_distinguishes_fused_transform_and_separate_routes",
        ])
        .forbidden(&[
            "include!(",
            "#[path",
            "pub(in crate::decoder)",
            "pub(super) can_fuse_store: bool",
            "pub(super) irreversible97: u32",
        ]),
        PatternCheck::new(
            "CUDA color batch store preparation ownership",
            &sources.color_store_batch,
        )
        .required(&[
            "struct CudaPreparedRgb8MctBatchStore",
            "fn prepare_rgb8_mct_batch_store(",
            "fn rgb8_mct_batch_store_target(",
            "ColorStorePlan::new(",
            "store_plan.rgb8_job(",
        ])
        .forbidden(&["include!(", "#[path", "CudaJ2kStoreRgb8Job {"]),
    ]);
}

fn assert_color_completion_ownership(sources: &CudaDecoderSources) {
    assert_pattern_checks(&[
        PatternCheck::new("CUDA queued IDWT ownership guard", &sources.decoder).required(&[
            "context: CudaContext",
            "fn resources_pending(&self) -> bool",
            "fn resolve_optional_after_completed_work<T>(",
            "pending.synchronize_and_release()?;",
            "Err(error) => match pending.synchronize_and_release()",
            "combine_cuda_cleanup_errors(error, cleanup_error)",
            "release_pool_reuse_after_completion()",
        ]),
        PatternCheck::new("CUDA queued IDWT completion paths", &sources.resident_idwt).required(&[
            "queued_batch.finish()?;",
            "j2k_inverse_dwt_batch_sequence_enqueue_with_pool_and_live_host_bytes(",
            "HostPhaseBudget::with_cuda_live_bytes(",
            "context.submit_default_stream_named(",
            "if let Err(completion) = context.synchronize()",
            "CudaError::CompletionFailed {",
        ]),
        PatternCheck::new(
            "CUDA cleanup/dequant aggregate host ownership",
            &sources.resident_cleanup_dequant,
        )
        .required(&[
            "HostPhaseBudget::with_live_bytes(",
            "decode_htj2k_codeblocks_cleanup_dequantize_multi_with_resources_and_pool_timed_and_live_host_bytes(",
            "decode_htj2k_codeblocks_cleanup_multi_with_resources_and_pool_timed_and_live_host_bytes(",
            "decode_htj2k_codeblocks_cleanup_multi_enqueue_with_resources_and_pool_and_live_host_bytes(",
            "j2k_dequantize_htj2k_codeblocks_multi_device_with_pool_and_live_host_bytes(",
            "if !collect_stage_timings {",
            "normal CUDA HTJ2K refinement requires retained queued cleanup metadata",
        ]),
    ]);
}
