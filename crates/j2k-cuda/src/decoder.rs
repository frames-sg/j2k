// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(feature = "cuda-runtime")]
use std::sync::Arc;

use j2k::{
    DeviceDecodePlan, DeviceDecodeRequest, J2kDecodeWarning, J2kDecoder as CpuDecoder,
    J2kScratchPool as CpuJ2kScratchPool, J2kView,
};
#[cfg(feature = "cuda-runtime")]
use j2k_core::BackendKind;
use j2k_core::{
    checked_surface_len, submit_ready_device, BackendRequest, CpuBackedImageDecode, DecodeOutcome,
    Downscale, ImageCodec, ImageDecodeDevice, ImageDecodeSubmit, PixelFormat, ReadySubmission,
    Rect, DEFAULT_MAX_HOST_ALLOCATION_BYTES,
};
#[cfg(feature = "cuda-runtime")]
use j2k_cuda_runtime::{
    CudaBufferPool, CudaBufferPoolTakeTrace, CudaClassicCodeBlockJob, CudaClassicDecodeTarget,
    CudaClassicSegment, CudaContext, CudaDeviceBuffer, CudaError, CudaExecutionStats,
    CudaHtj2kCleanupTarget, CudaHtj2kCodeBlockJob, CudaHtj2kDecodeResources,
    CudaHtj2kDecodeTableResources, CudaHtj2kDequantizeTarget, CudaJ2kIdwtJob, CudaJ2kIdwtTarget,
    CudaJ2kRect, CudaJ2kStoreGray16Job, CudaJ2kStoreGray8Job, CudaPooledDeviceBuffer,
    CudaQueuedExecution, CudaQueuedHtj2kCleanup,
};
#[cfg(feature = "cuda-runtime")]
use j2k_native::{DecodeSettings, DecoderContext as NativeDecoderContext, Image as NativeImage};

#[cfg(feature = "cuda-runtime")]
use crate::error::{combine_cuda_cleanup_errors, native_decode_error};
#[cfg(feature = "cuda-runtime")]
use crate::runtime::cuda_error;
use crate::runtime::{validate_surface_request, wrap_cpu_staged_cuda_surface, wrap_surface};
#[cfg(feature = "cuda-runtime")]
use crate::surface::{cuda_range_storage, Storage};
#[cfg(feature = "cuda-runtime")]
use crate::{
    profile, CudaHtj2kBandId, CudaHtj2kDecodePlan, CudaHtj2kDecodeProfileDetail, CudaHtj2kIdwtStep,
    CudaHtj2kStoreStep, CudaHtj2kTransform, CudaSurfaceStats, SurfaceResidency,
};
use crate::{CudaHtj2kProfileReport, CudaSession, Error, Surface};

#[cfg(feature = "cuda-runtime")]
const CUDA_HTJ2K_KERNELS_NOT_READY: &str =
    "strict CUDA HTJ2K resident codestream decode kernels are not available in this build";
#[cfg(feature = "cuda-runtime")]
const CUDA_HTJ2K_OUTPUT_FORMAT_UNSUPPORTED: &str =
    "strict CUDA HTJ2K resident decode currently accepts Gray8, Gray16, Rgb8, Rgba8, Rgb16, and Rgba16 output";
#[cfg(feature = "cuda-runtime")]
const CUDA_HTJ2K_PLAN_INVARIANT_FAILED: &str =
    "strict CUDA HTJ2K resident decode plan has invalid internal ranges";
#[cfg(feature = "cuda-runtime")]
const CUDA_HTJ2K_STORE_UNSUPPORTED: &str =
    "strict CUDA HTJ2K resident decode requires a single grayscale store step";
#[cfg(feature = "cuda-runtime")]
const CUDA_HTJ2K_BATCH_PAYLOAD_TOO_LARGE: &str =
    "strict CUDA HTJ2K resident batch decode payload is too large";
#[cfg(feature = "cuda-runtime")]
const CUDA_IDWT_TRACE_ENV_VAR: &str = "J2K_CUDA_IDWT_TRACE";

mod api;
#[cfg(feature = "cuda-runtime")]
mod color_batch;
#[path = "decoder/profile.rs"]
mod decode_profile;
#[cfg(feature = "cuda-runtime")]
mod plan;
#[cfg(feature = "cuda-runtime")]
mod resident;

#[cfg(all(test, feature = "cuda-runtime"))]
pub(crate) use self::color_batch::{
    testing_cuda_htj2k_batch_decode_calls, testing_reset_cuda_htj2k_batch_decode_calls,
};
#[cfg(feature = "cuda-runtime")]
use self::decode_profile::CudaDecodeStageTimings;
#[cfg(all(test, feature = "cuda-runtime"))]
use self::decode_profile::{format_cuda_idwt_batch_host_trace_row, CudaIdwtBatchHostTraceRow};
#[cfg(all(test, feature = "cuda-runtime"))]
use self::plan::build_cuda_htj2k_color_plans_from_bytes_with_profile;
#[cfg(all(test, feature = "cuda-runtime"))]
use self::resident::{
    can_batch_color_idwt, cuda_code_block_job_from_plan_block,
    htj2k_batched_cleanup_dequant_dispatches, htj2k_batched_cleanup_dispatches,
};
#[cfg(all(test, feature = "cuda-runtime"))]
use self::resident::{htj2k_batched_dequant_dispatches, split_htj2k_subband_decode_dispatches};

/// CUDA-facing JPEG 2000 decoder wrapper.
pub struct J2kDecoder<'a> {
    #[cfg_attr(
        not(feature = "cuda-runtime"),
        expect(
            dead_code,
            reason = "raw codestream bytes are consumed only by CUDA decode routes"
        )
    )]
    bytes: &'a [u8],
    inner: CpuDecoder<'a>,
    pool: CpuJ2kScratchPool,
}

#[cfg(feature = "cuda-runtime")]
struct CudaCoefficientBand {
    band_id: CudaHtj2kBandId,
    buffer: CudaPooledDeviceBuffer,
}

#[cfg(feature = "cuda-runtime")]
struct CudaPendingDequantBand {
    band_index: usize,
    jobs: Vec<CudaHtj2kCodeBlockJob>,
    output_words: usize,
}

#[cfg(feature = "cuda-runtime")]
struct CudaPendingClassicBand {
    band_index: usize,
    jobs: Vec<CudaClassicCodeBlockJob>,
    segments: Vec<CudaClassicSegment>,
    output_words: usize,
}

#[cfg(feature = "cuda-runtime")]
struct CudaComponentDecodeWork {
    bands: Vec<CudaCoefficientBand>,
    pending_classic_bands: Vec<CudaPendingClassicBand>,
    pending_dequant_bands: Vec<CudaPendingDequantBand>,
    store: CudaHtj2kStoreStep,
    dispatches: usize,
    decode_dispatches: usize,
    timings: CudaDecodeStageTimings,
}

#[cfg(feature = "cuda-runtime")]
struct CudaQueuedIdwtBatch {
    context: CudaContext,
    queued: Vec<CudaQueuedExecution>,
    kernel_dispatches: usize,
    decode_dispatches: usize,
}

#[cfg(feature = "cuda-runtime")]
impl CudaQueuedIdwtBatch {
    fn resources_pending(&self) -> bool {
        self.kernel_dispatches != 0 && !self.queued.is_empty()
    }

    fn release_after_completion(&mut self) -> Result<(), Error> {
        // The decode pools are private to an exclusively borrowed session.
        // MCT/store helpers use completion-synchronizing runtime launches, so
        // reaching this path proves every IDWT reference has completed.
        for queued in &mut self.queued {
            // SAFETY: `CudaSession` owns this pool privately behind an
            // exclusive mutable borrow. Callers reach this method only after
            // context synchronization or a completion-synchronizing MCT/store
            // dispatch on the same context.
            unsafe { queued.release_pool_reuse_after_completion() }.map_err(cuda_error)?;
        }
        self.queued.clear();
        Ok(())
    }

    fn synchronize_and_release(&mut self) -> Result<(), Error> {
        if self.resources_pending() {
            self.context.synchronize().map_err(cuda_error)?;
        }
        self.release_after_completion()
    }

    fn finish(mut self) -> Result<(), Error> {
        self.synchronize_and_release()
    }

    fn resolve_optional_after_completed_work<T>(
        pending: Option<Self>,
        result: Result<(T, bool), Error>,
    ) -> Result<T, Error> {
        let Some(mut pending) = pending else {
            return result.map(|(output, _completion_established)| output);
        };
        match result {
            Ok((output, completion_established)) => {
                if pending.resources_pending() && !completion_established {
                    pending.synchronize_and_release()?;
                } else {
                    pending.release_after_completion()?;
                }
                Ok(output)
            }
            Err(error) => match pending.synchronize_and_release() {
                Ok(()) => Err(error),
                Err(cleanup_error) => Err(combine_cuda_cleanup_errors(error, cleanup_error)),
            },
        }
    }
}

#[cfg(feature = "cuda-runtime")]
struct CudaDecodedComponent {
    buffer: CudaPooledDeviceBuffer,
    store: CudaHtj2kStoreStep,
    dispatches: usize,
    decode_dispatches: usize,
    timings: CudaDecodeStageTimings,
}

#[cfg(feature = "cuda-runtime")]
struct CudaHtj2kColorDecodePlans {
    dimensions: (u32, u32),
    mct_dimensions: (u32, u32),
    bit_depths: [u8; 3],
    mct: bool,
    transform: CudaHtj2kTransform,
    payload: Vec<u8>,
    components: Vec<CudaHtj2kDecodePlan>,
    report: CudaHtj2kProfileReport,
}

#[cfg(all(test, feature = "cuda-runtime"))]
mod tests {
    use super::{
        build_cuda_htj2k_color_plans_from_bytes_with_profile, can_batch_color_idwt,
        cuda_code_block_job_from_plan_block, htj2k_batched_cleanup_dequant_dispatches,
        htj2k_batched_cleanup_dispatches, htj2k_batched_dequant_dispatches, CudaDecodeStageTimings,
    };
    use j2k_core::PixelFormat;
    use j2k_native::{encode_htj2k, DecoderContext as NativeDecoderContext, EncodeOptions};

    use crate::CudaHtj2kCodeBlock;

    #[test]
    fn cuda_runtime_code_block_job_preserves_plan_output_stride() {
        let block = CudaHtj2kCodeBlock {
            subband_index: 0,
            payload_offset: 13,
            payload_len: 5,
            cleanup_length: 5,
            refinement_length: 0,
            output_x: 3,
            output_y: 2,
            width: 4,
            height: 5,
            output_stride: 99,
            missing_bit_planes: 1,
            number_of_coding_passes: 1,
            num_bitplanes: 8,
            stripe_causal: 0,
            dequantization_step: 1.0,
        };

        let job = cuda_code_block_job_from_plan_block(&block, 64)
            .expect("valid CUDA code-block runtime job");

        assert_eq!(job.output_offset, 131);
        assert_eq!(job.output_stride, 99);
    }

    #[test]
    fn batched_cleanup_and_dequant_dispatch_helpers_count_one_shared_dispatch() {
        assert_eq!(htj2k_batched_cleanup_dispatches(0), 0);
        assert_eq!(htj2k_batched_cleanup_dispatches(1), 1);
        assert_eq!(htj2k_batched_cleanup_dispatches(3), 1);
        assert_eq!(htj2k_batched_dequant_dispatches(0), 0);
        assert_eq!(htj2k_batched_dequant_dispatches(1), 1);
        assert_eq!(htj2k_batched_dequant_dispatches(3), 1);
        assert_eq!(htj2k_batched_cleanup_dequant_dispatches(0, true), (0, 0));
        assert_eq!(htj2k_batched_cleanup_dequant_dispatches(1, true), (1, 0));
        assert_eq!(htj2k_batched_cleanup_dequant_dispatches(3, true), (1, 0));
        assert_eq!(htj2k_batched_cleanup_dequant_dispatches(1, false), (1, 1));
        assert_eq!(htj2k_batched_cleanup_dequant_dispatches(3, false), (1, 1));
    }

    #[test]
    fn profiled_cuda_batch_decode_api_accepts_empty_batch() {
        let mut session = crate::CudaSession::default();
        let inputs: [&[u8]; 0] = [];

        let (surfaces, report) =
            crate::J2kDecoder::decode_batch_to_device_with_session_and_profile(
                &inputs,
                PixelFormat::Rgb8,
                &mut session,
            )
            .expect("empty CUDA batch decode");

        assert!(surfaces.is_empty());
        assert_eq!(report.block_count, 0);
        assert_eq!(report.payload_bytes, 0);
    }

    #[test]
    fn cuda_batch_decode_two_color_images_matches_single_when_runtime_required() {
        let pixels_a: Vec<u8> = (0u16..16 * 16 * 3)
            .map(|idx| u8::try_from((idx * 7 + idx / 5) & 0xff).expect("masked byte"))
            .collect();
        let pixels_b: Vec<u8> = (0u16..16 * 16 * 3)
            .map(|idx| u8::try_from((idx * 11 + 23) & 0xff).expect("masked byte"))
            .collect();
        let options = EncodeOptions {
            reversible: true,
            num_decomposition_levels: 1,
            ..EncodeOptions::default()
        };
        let codestream_a =
            encode_htj2k(&pixels_a, 16, 16, 3, 8, false, &options).expect("encode fixture A");
        let codestream_b =
            encode_htj2k(&pixels_b, 16, 16, 3, 8, false, &options).expect("encode fixture B");
        let inputs = [codestream_a.as_slice(), codestream_b.as_slice()];
        let mut batch_session = crate::CudaSession::default();

        let batch = crate::J2kDecoder::decode_batch_to_device_with_session_and_profile(
            &inputs,
            PixelFormat::Rgb8,
            &mut batch_session,
        );
        let (surfaces, report) = match batch {
            Ok(result) => result,
            Err(crate::Error::CudaUnavailable | crate::Error::CudaRuntime { .. })
                if !cuda_runtime_gate() =>
            {
                return;
            }
            Err(error) => panic!("batch CUDA decode failed: {error}"),
        };

        assert_eq!(surfaces.len(), 2);
        assert_eq!(report.detail.ht_dispatch_count, 1);
        assert_eq!(report.detail.dequant_dispatch_count, 0);
        assert_eq!(report.detail.store_dispatch_count, 1);
        let batch_pixels_tight =
            crate::Surface::download_batch_tight(&surfaces).expect("download tight CUDA batch");
        assert_eq!(batch_pixels_tight.len(), surfaces.len() * 16 * 16 * 3);
        for (index, codestream) in inputs.iter().enumerate() {
            let mut single_session = crate::CudaSession::default();
            let mut decoder = crate::J2kDecoder::new(codestream).expect("single decoder");
            let single = decoder
                .decode_to_device_with_session(PixelFormat::Rgb8, &mut single_session)
                .expect("single CUDA decode");
            let mut single_pixels = vec![0u8; 16 * 16 * 3];
            let mut batch_pixels = vec![0u8; 16 * 16 * 3];
            single
                .download_into(&mut single_pixels, 16 * 3)
                .expect("download single decode");
            surfaces[index]
                .download_into(&mut batch_pixels, 16 * 3)
                .expect("download batch decode");
            assert_eq!(batch_pixels, single_pixels);
            assert_eq!(
                &batch_pixels_tight[index * 16 * 16 * 3..(index + 1) * 16 * 16 * 3],
                single_pixels.as_slice()
            );
        }
    }

    #[test]
    fn cuda_batch_decode_mixed_idwt_shapes_avoids_fused_batch_store_without_idwt_batch() {
        let codestream_a = rgb8_htj2k_fixture(32, 32, 1, 7);
        let codestream_b = rgb8_htj2k_fixture(32, 32, 2, 19);
        let inputs = [codestream_a.as_slice(), codestream_b.as_slice()];
        let mut batch_session = crate::CudaSession::default();

        let result = crate::J2kDecoder::decode_batch_to_device_with_session(
            &inputs,
            PixelFormat::Rgb8,
            &mut batch_session,
        );
        let surfaces = match result {
            Ok(surfaces) => surfaces,
            Err(crate::Error::CudaUnavailable | crate::Error::CudaRuntime { .. })
                if !cuda_runtime_gate() =>
            {
                return;
            }
            Err(crate::Error::UnsupportedCudaRequest { .. }) => return,
            Err(error) => panic!("mixed-shape batch CUDA decode failed: {error}"),
        };

        assert_eq!(surfaces.len(), inputs.len());
        for (index, codestream) in inputs.iter().enumerate() {
            let mut single_session = crate::CudaSession::default();
            let mut decoder = crate::J2kDecoder::new(codestream).expect("single decoder");
            let single = decoder
                .decode_to_device_with_session(PixelFormat::Rgb8, &mut single_session)
                .expect("single CUDA decode");
            let mut single_pixels = vec![0u8; 32 * 32 * 3];
            let mut batch_pixels = vec![0u8; 32 * 32 * 3];
            single
                .download_into(&mut single_pixels, 32 * 3)
                .expect("download single decode");
            surfaces[index]
                .download_into(&mut batch_pixels, 32 * 3)
                .expect("download mixed-shape batch decode");
            assert_eq!(batch_pixels, single_pixels);
        }
    }

    #[test]
    fn decode_stage_timings_report_status_download_detail() {
        let mut report = crate::CudaHtj2kProfileReport::default();
        let timings = CudaDecodeStageTimings {
            h2d: 17,
            table_upload: 7,
            job_upload: 10,
            status_d2h: 5,
            classic_tier1: 11,
            ..CudaDecodeStageTimings::default()
        };

        timings.add_to_report(&mut report);

        assert_eq!(report.h2d_us, 17);
        assert_eq!(report.detail.table_upload_us, 7);
        assert_eq!(report.detail.job_upload_us, 10);
        assert_eq!(report.detail.status_d2h_us, 5);
        assert_eq!(report.classic_tier1_us, 11);
    }

    fn cuda_runtime_gate() -> bool {
        j2k_test_support::cuda_runtime_gate(module_path!())
    }

    fn rgb8_htj2k_fixture(width: u32, height: u32, levels: u8, seed: u16) -> Vec<u8> {
        let mut pixels = Vec::with_capacity(width as usize * height as usize * 3);
        for idx in 0..width * height {
            let seed = u32::from(seed);
            pixels.push(u8::try_from((idx * seed + idx / 3) & 0xff).expect("red"));
            pixels.push(u8::try_from((idx * (seed + 11) + 7) & 0xff).expect("green"));
            pixels.push(u8::try_from((idx * (seed + 23) + 19) & 0xff).expect("blue"));
        }
        let options = EncodeOptions {
            reversible: true,
            num_decomposition_levels: levels,
            ..EncodeOptions::default()
        };
        encode_htj2k(&pixels, width, height, 3, 8, false, &options)
            .expect("encode RGB HTJ2K fixture")
    }

    #[test]
    fn color_plan_flattens_one_shared_payload_for_component_decode() {
        let pixels: Vec<u8> = (0u16..4 * 4 * 3)
            .map(|idx| u8::try_from((idx * 13 + idx / 3) & 0xff).expect("masked byte"))
            .collect();
        let options = EncodeOptions {
            reversible: true,
            num_decomposition_levels: 1,
            ..EncodeOptions::default()
        };
        let codestream =
            encode_htj2k(&pixels, 4, 4, 3, 8, false, &options).expect("encode HTJ2K RGB fixture");
        let mut decoder = crate::J2kDecoder::new(&codestream).expect("decoder");

        let color = decoder
            .build_cuda_htj2k_color_plans_with_profile(PixelFormat::Rgb8)
            .expect("CUDA color plans");

        assert_eq!(color.components.len(), 3);
        assert!(!color.payload.is_empty());
        assert_eq!(color.report.payload_bytes, color.payload.len());
        for component in &color.components {
            assert!(component.payload().is_empty());
            for block in component.code_blocks() {
                let start = usize::try_from(block.payload_offset).expect("payload offset");
                let end = start + block.payload_len as usize;
                assert!(end <= color.payload.len());
            }
        }
    }

    #[test]
    fn byte_color_plan_builder_matches_decoder_color_plan() {
        let pixels: Vec<u8> = (0u16..8 * 8 * 3)
            .map(|idx| u8::try_from((idx * 19 + idx / 5) & 0xff).expect("masked byte"))
            .collect();
        let options = EncodeOptions {
            reversible: true,
            num_decomposition_levels: 1,
            ..EncodeOptions::default()
        };
        let codestream =
            encode_htj2k(&pixels, 8, 8, 3, 8, false, &options).expect("encode HTJ2K RGB fixture");
        let mut decoder = crate::J2kDecoder::new(&codestream).expect("decoder");
        let decoder_plan = decoder
            .build_cuda_htj2k_color_plans_with_profile(PixelFormat::Rgb8)
            .expect("decoder CUDA color plans");
        let mut native_context = NativeDecoderContext::default();
        let byte_plan = build_cuda_htj2k_color_plans_from_bytes_with_profile(
            &codestream,
            PixelFormat::Rgb8,
            &mut native_context,
        )
        .expect("byte CUDA color plans");

        assert_eq!(byte_plan.dimensions, decoder_plan.dimensions);
        assert_eq!(byte_plan.mct_dimensions, decoder_plan.mct_dimensions);
        assert_eq!(byte_plan.bit_depths, decoder_plan.bit_depths);
        assert_eq!(byte_plan.mct, decoder_plan.mct);
        assert_eq!(byte_plan.components.len(), decoder_plan.components.len());
        assert_eq!(byte_plan.payload.len(), decoder_plan.payload.len());
        assert_eq!(
            byte_plan
                .components
                .iter()
                .map(|component| component.code_blocks().len())
                .collect::<Vec<_>>(),
            decoder_plan
                .components
                .iter()
                .map(|component| component.code_blocks().len())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn multi_image_color_components_can_share_one_idwt_batch() {
        let pixels: Vec<u8> = (0u16..16 * 16 * 3)
            .map(|idx| u8::try_from((idx * 17 + idx / 7) & 0xff).expect("masked byte"))
            .collect();
        let options = EncodeOptions {
            reversible: true,
            num_decomposition_levels: 1,
            ..EncodeOptions::default()
        };
        let codestream =
            encode_htj2k(&pixels, 16, 16, 3, 8, false, &options).expect("encode HTJ2K RGB fixture");
        let mut first = crate::J2kDecoder::new(&codestream).expect("first decoder");
        let mut second = crate::J2kDecoder::new(&codestream).expect("second decoder");
        let first = first
            .build_cuda_htj2k_color_plans_with_profile(PixelFormat::Rgb8)
            .expect("first CUDA color plans");
        let second = second
            .build_cuda_htj2k_color_plans_with_profile(PixelFormat::Rgb8)
            .expect("second CUDA color plans");
        let components = first
            .components
            .iter()
            .chain(second.components.iter())
            .collect::<Vec<_>>();

        assert_eq!(components.len(), 6);
        assert!(can_batch_color_idwt(&components));
    }

    #[test]
    fn batched_color_idwt_defers_completion_to_store_sync() {
        let source = include_str!("decoder.rs");

        assert!(
            !source.contains(
                "if !collect_stage_timings {\n        context.synchronize().map_err(cuda_error)?;\n    }"
            ),
            "batched color IDWT should keep queued resources live and let the following store synchronize"
        );
    }

    #[test]
    fn batched_color_idwt_preflights_each_output_before_pool_take() {
        let source = include_str!("decoder/resident/idwt.rs");
        let function = source
            .split("fn enqueue_color_component_idwt_batches")
            .nth(1)
            .expect("batched IDWT enqueue function");
        let preflight = function
            .find("j2k_inverse_dwt_single_output_bytes")
            .expect("runtime IDWT output preflight");
        for pool_take in [
            "pool.take_with_trace(output_bytes)",
            "pool.take(output_bytes)",
        ] {
            let pool_take = function.find(pool_take).expect("IDWT output pool take");
            assert!(
                preflight < pool_take,
                "IDWT job semantics and launch geometry must be validated before output allocation"
            );
        }
    }
}

#[cfg(all(test, feature = "cuda-runtime"))]
#[path = "htj2k_plan_tests.rs"]
mod htj2k_plan_tests;

#[cfg(all(test, feature = "cuda-runtime"))]
mod dispatch_tests {
    use super::{
        format_cuda_idwt_batch_host_trace_row, htj2k_batched_dequant_dispatches,
        split_htj2k_subband_decode_dispatches, CudaIdwtBatchHostTraceRow,
    };

    #[test]
    fn htj2k_decode_dispatch_split_separates_ht_and_dequant_counts() {
        assert_eq!(split_htj2k_subband_decode_dispatches(0), (0, 0));
        assert_eq!(split_htj2k_subband_decode_dispatches(1), (1, 0));
        assert_eq!(split_htj2k_subband_decode_dispatches(2), (1, 1));
        assert_eq!(split_htj2k_subband_decode_dispatches(3), (2, 1));
    }

    #[test]
    fn htj2k_batched_dequant_dispatch_count_is_one_for_any_non_empty_batch() {
        assert_eq!(htj2k_batched_dequant_dispatches(0), 0);
        assert_eq!(htj2k_batched_dequant_dispatches(1), 1);
        assert_eq!(htj2k_batched_dequant_dispatches(48), 1);
    }

    #[test]
    fn cuda_idwt_batch_host_trace_row_reports_host_split() {
        let row = CudaIdwtBatchHostTraceRow {
            component_count: 327,
            step_count: 5,
            output_alloc_us: 11,
            target_build_us: 22,
            enqueue_us: 33,
            output_take_count: 1635,
            output_pool_reuse_count: 1600,
            output_pool_alloc_count: 35,
            output_pool_scanned_count: 2400,
            output_pool_max_free_count: 1700,
            output_requested_bytes: 28,
        };

        assert_eq!(
            format_cuda_idwt_batch_host_trace_row(row).expect("bounded host trace row"),
            "j2k_profile codec=j2k op=cuda_idwt_batch_host path=decode component_count=327 step_count=5 output_alloc_us=11 target_build_us=22 enqueue_us=33 output_take_count=1635 output_pool_reuse_count=1600 output_pool_alloc_count=35 output_pool_scanned_count=2400 output_pool_max_free_count=1700 output_requested_bytes=28"
        );
    }
}
