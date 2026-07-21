// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::super::{assert_pattern_checks, PatternCheck};
use super::CudaDecoderSources;

#[test]
fn runtime_paths_live_in_focused_modules() {
    let sources = CudaDecoderSources::read();
    assert_decoder_facade_ownership(&sources);
    assert_resident_pipeline_ownership(&sources);
    assert_decoder_support_ownership(&sources);
}

fn assert_decoder_facade_ownership(sources: &CudaDecoderSources) {
    assert_pattern_checks(&[
        PatternCheck::new("CUDA decoder shell", &sources.decoder)
            .required(&[
                "mod api;",
                "mod color_batch;",
                "decoder/profile.rs",
                "mod plan;",
                "mod resident;",
                "pub struct J2kDecoder",
            ])
            .forbidden(&[
                "impl<'a> CpuBackedImageDecode<'a>",
                "fn build_cuda_htj2k_grayscale_plan_with_profile",
                "fn decode_cuda_component_plan",
                "fn decode_color_cuda_resident_surface_with_profile",
                "fn aggregate_decode_reports",
            ]),
        PatternCheck::new("CUDA decoder API module", &sources.api).required(&[
            "impl<'a> J2kDecoder<'a>",
            "pub fn new(input: &'a [u8])",
            "pub fn decode_to_device_with_session",
            "impl<'a> CpuBackedImageDecode<'a>",
        ]),
        PatternCheck::new("CUDA decoder plan facade", &sources.plan)
            .required(&[
                "mod color;",
                "mod color_decoder;",
                "mod color_referenced;",
                "mod grayscale;",
                "pub(super) use self::color::{",
                "pub(super) use self::grayscale::{",
            ])
            .forbidden(&["fn build_cuda_htj2k_grayscale_plan_with_profile("]),
        PatternCheck::new("CUDA grayscale plan ownership", &sources.plan_grayscale)
            .required(&["fn build_cuda_htj2k_grayscale_plan_with_profile("]),
        PatternCheck::new("CUDA color byte-plan ownership", &sources.plan_color)
            .required(&["fn build_cuda_htj2k_color_plans_from_bytes_with_profile"]),
        PatternCheck::new("CUDA color plans", &sources.plan_color_decoder)
            .required(&["fn build_cuda_htj2k_color_plans_with_profile"]),
        PatternCheck::new("CUDA decoder resident facade", &sources.resident)
            .required(&[
                "mod buffer_access;",
                "mod cleanup_dequant;",
                "mod component;",
                "mod error;",
                "mod idwt;",
                "mod routing;",
                "mod surface;",
                "pub(super) use self::buffer_access::pooled_cuda_buffer;",
                "pub(super) use self::cleanup_dequant::{",
                "pub(super) use self::component::{",
                "pub(super) use self::idwt::{",
                "pub(super) use self::routing::{",
            ])
            .forbidden(&["fn ", "include!(", "use super::*"]),
    ]);
}

fn assert_resident_pipeline_ownership(sources: &CudaDecoderSources) {
    assert_pattern_checks(&[
        PatternCheck::new("CUDA resident routing ownership", &sources.resident_routing)
            .required(&[
                "fn decode_to_cuda_resident_surface_impl",
                "fn decode_region_to_cuda_resident_surface_impl",
                "fn decode_scaled_to_cuda_resident_surface_impl",
                "fn decode_region_scaled_to_cuda_resident_surface_impl",
            ])
            .forbidden(&[
                "fn decode_cuda_component_plan",
                "fn run_component_cleanup_dequant_batches",
                "fn run_color_component_idwt_batches",
            ]),
        PatternCheck::new("CUDA resident components", &sources.resident_component)
            .required(&[
                "fn decode_cuda_component_plan",
                "fn decode_cuda_component_subbands_with_resources",
                "fn finish_cuda_component_decode",
                "mod ht;",
            ])
            .forbidden(&[
                "fn run_component_cleanup_dequant_batches",
                "fn run_cuda_component_idwt_steps",
                "fn decode_grayscale_cuda_resident_surface_with_plan_profile",
            ]),
        PatternCheck::new(
            "CUDA resident cleanup/dequant ownership",
            &sources.resident_cleanup_dequant,
        )
        .required(&[
            "fn run_component_cleanup_dequant_batches",
            "fn htj2k_batched_cleanup_dequant_dispatches",
            "queued.finish()",
            "combine_cuda_cleanup_errors(",
        ])
        .forbidden(&[
            "fn run_cuda_component_idwt_steps",
            "fn enqueue_color_component_idwt_batches",
        ]),
        PatternCheck::new("CUDA resident IDWT ownership", &sources.resident_idwt)
            .required(&[
                "mod conversions;",
                "fn run_cuda_component_idwt_steps",
                "fn run_color_component_idwt_batches",
                "fn enqueue_color_component_idwt_batches",
                "queued_batch.finish()?;",
                "CudaError::CompletionFailed {",
            ])
            .forbidden(&[
                "fn run_component_cleanup_dequant_batches",
                "fn decode_cuda_component_subbands_with_resources",
            ]),
        PatternCheck::new(
            "CUDA resident IDWT conversion ownership",
            &sources.resident_idwt_conversions,
        )
        .required(&[
            "fn find_cuda_band",
            "fn cuda_runtime_rect",
            "fn cuda_idwt_job_from_step",
        ]),
        PatternCheck::new(
            "CUDA resident surface assembly ownership",
            &sources.resident_surface,
        )
        .required(&[
            "fn decode_grayscale_cuda_resident_surface_with_plan_profile",
            "Surface {",
            "CudaSurfaceStats {",
            "profile::finalize_decode_total_us(report);",
        ])
        .forbidden(&[
            "fn decode_region_to_cuda_resident_surface_impl",
            "fn run_component_cleanup_dequant_batches",
        ]),
    ]);
}

fn assert_decoder_support_ownership(sources: &CudaDecoderSources) {
    assert_pattern_checks(&[
        PatternCheck::new("CUDA decoder color batch module", &sources.color_batch).required(&[
            "decode_color_cuda_resident_surface_with_profile",
            "decode_color_cuda_resident_batch_surfaces_with_profile",
            "prepare_rgb8_mct_batch_store",
        ]),
        PatternCheck::new(
            "CUDA decoder color-store validation module",
            &sources.color_store_validation,
        )
        .required(&["fn validate_color_stores", "fn bit_depth_addend"]),
        PatternCheck::new("CUDA resident error module", &sources.resident_error)
            .required(&["fn cuda_invalid_decode_plan"]),
        PatternCheck::new("CUDA decoder profile module", &sources.profile).required(&[
            "struct CudaDecodeStageTimings",
            "fn aggregate_decode_reports",
            "struct CudaIdwtBatchHostTraceRow",
        ]),
    ]);
}
