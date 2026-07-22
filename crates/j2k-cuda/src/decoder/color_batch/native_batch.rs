// SPDX-License-Identifier: MIT OR Apache-2.0

mod completion;
mod execution;
mod prepare;
mod store;

use j2k::{BatchLayout, DeviceDecodePlan};
use j2k_core::{DeviceSubmitSession, PixelFormat};
use j2k_cuda_runtime::{
    CudaDeviceBuffer, CudaDeviceBufferRange, CudaExternalDeviceBufferViewMut,
    CudaQueuedJ2kStoreBatch,
};

use super::{
    append_color_payload_to_shared, can_batch_color_idwt, cuda_error,
    finalize_color_batch_decode_report, host_owners, profile, run_color_component_idwt_batches,
    CudaHtj2kColorDecodePlans, CudaHtj2kProfileReport, CudaQueuedIdwtBatch, CudaSession, Error,
    NativeDecoderContext,
};
use crate::allocation::HostPhaseBudget;
use crate::decoder::pending_completion::PendingDecodeCompletion;
use crate::decoder::plan::{
    build_cuda_classic_color_plans_from_referenced_with_profile,
    build_cuda_color_plan_from_bytes_for_device_plan_with_profile,
    build_cuda_htj2k_color_plans_from_referenced_with_profile,
};
use crate::decoder::resident::{
    decode_cuda_component_subbands_with_resources, enqueue_chunked_htj2k_cleanup_dequant,
    enqueue_component_classic_batches, ChunkedHtj2kCleanup,
};
use crate::decoder::CUDA_HTJ2K_OUTPUT_FORMAT_UNSUPPORTED;

use self::completion::{
    NativeColorBatchOutput, NativeColorPendingCompletion, StoredNativeColorBatch,
};
pub(crate) use self::completion::{
    SubmittedNativeColorExternalBatch, SubmittedNativeColorResidentBatch,
};
use self::execution::decode_native_color_batch;
use self::prepare::{prepare_native_color_batch, PreparedNativeColorBatch};
use self::store::finish_and_store_native_color;

#[derive(Clone, Copy)]
pub(crate) struct NativeColorBatchInput<'a> {
    pub(crate) source_index: usize,
    pub(crate) bytes: &'a [u8],
    pub(crate) device_plan: DeviceDecodePlan,
    pub(crate) referenced_plan: Option<&'a j2k_native::J2kReferencedHtj2kPlan>,
    pub(crate) referenced_classic_plan: Option<&'a j2k_native::J2kReferencedClassicPlan>,
    pub(crate) settings: j2k_native::DecodeSettings,
}

pub(crate) struct NativeColorOwnedBatch {
    pub(crate) buffer: CudaDeviceBuffer,
    pub(crate) ranges: Vec<CudaDeviceBufferRange>,
    pub(crate) execution: j2k_cuda_runtime::CudaExecutionStats,
}

pub(crate) fn submit_native_color_resident_prepared_batch_into(
    inputs: &[NativeColorBatchInput<'_>],
    session: &mut CudaSession,
    fmt: PixelFormat,
    layout: BatchLayout,
    destination: &mut CudaExternalDeviceBufferViewMut<'_>,
) -> Result<SubmittedNativeColorExternalBatch, Error> {
    let (output, report, completion) =
        decode_native_color_batch(inputs, session, fmt, layout, Some(destination), true)?;
    let NativeColorBatchOutput::External(ranges) = output else {
        return Err(Error::UnsupportedCudaRequest {
            reason: CUDA_HTJ2K_OUTPUT_FORMAT_UNSUPPORTED,
        });
    };
    let completion = completion.ok_or(Error::UnsupportedCudaRequest {
        reason: "CUDA exact RGB submission did not retain a completion owner",
    })?;
    Ok(SubmittedNativeColorExternalBatch {
        ranges,
        report,
        completion: Some(completion),
    })
}

pub(crate) fn submit_native_color_resident_prepared_batch(
    inputs: &[NativeColorBatchInput<'_>],
    session: &mut CudaSession,
    fmt: PixelFormat,
    layout: BatchLayout,
) -> Result<SubmittedNativeColorResidentBatch, Error> {
    let (output, report, completion) =
        decode_native_color_batch(inputs, session, fmt, layout, None, true)?;
    let NativeColorBatchOutput::Owned(output) = output else {
        return Err(Error::UnsupportedCudaRequest {
            reason: CUDA_HTJ2K_OUTPUT_FORMAT_UNSUPPORTED,
        });
    };
    let completion = completion.ok_or(Error::UnsupportedCudaRequest {
        reason: "CUDA resident RGB submission did not retain a completion owner",
    })?;
    Ok(SubmittedNativeColorResidentBatch {
        output: Some(output),
        report,
        completion: Some(completion),
    })
}

#[cfg(test)]
mod tests {
    use std::{num::NonZeroUsize, sync::Arc};

    use j2k::{
        prepare_batch, BatchDecodeOptions, BatchLayout, CpuBatchDecoder, CpuBatchSamples,
        DeviceDecodePlan, DeviceDecodeRequest, EncodedImage,
    };
    use j2k_core::{Downscale, HtGpuJobChunkLimits, PixelFormat, Rect};
    use j2k_cuda_runtime::{htj2k_cleanup_multi_descriptor_bytes, CudaContext};
    use j2k_native::{encode_htj2k, DecodeSettings, EncodeOptions, Image};

    use super::{
        prepare_native_color_batch, submit_native_color_resident_prepared_batch,
        NativeColorBatchInput,
    };
    use crate::CudaSession;

    #[test]
    fn retained_rgb_plan_prepares_full_roi_and_reduced_without_reparse() {
        let pixels = (0_u16..16 * 16 * 3)
            .map(|value| u8::try_from(value & 0x7f).unwrap())
            .collect::<Vec<_>>();
        let encoded = encode_htj2k(
            &pixels,
            16,
            16,
            3,
            7,
            false,
            &EncodeOptions {
                reversible: true,
                num_decomposition_levels: 2,
                ..EncodeOptions::default()
            },
        )
        .expect("RGB HTJ2K fixture");
        for request in [
            DeviceDecodeRequest::Full,
            DeviceDecodeRequest::Region {
                roi: Rect {
                    x: 2,
                    y: 3,
                    w: 8,
                    h: 7,
                },
            },
            DeviceDecodeRequest::Scaled {
                scale: Downscale::Half,
            },
            DeviceDecodeRequest::RegionScaled {
                roi: Rect {
                    x: 1,
                    y: 2,
                    w: 6,
                    h: 5,
                },
                scale: Downscale::Half,
            },
        ] {
            let plan = DeviceDecodePlan::for_image((16, 16), request).unwrap();
            let mut settings = DecodeSettings::strict();
            settings.target_resolution = (plan.scale() != Downscale::None).then_some((
                plan.source_dims().0.div_ceil(plan.scale().denominator()),
                plan.source_dims().1.div_ceil(plan.scale().denominator()),
            ));
            let image = Image::new(&encoded, &settings).expect("parse RGB fixture");
            let output = plan.output_rect();
            let mut context = j2k_native::DecoderContext::default();
            let referenced = image
                .build_referenced_htj2k_plan_region_with_context(
                    &mut context,
                    (output.x, output.y, output.w, output.h),
                )
                .expect("referenced RGB plan");
            let prepared = prepare_native_color_batch(
                &[NativeColorBatchInput {
                    source_index: 0,
                    bytes: &encoded,
                    device_plan: plan,
                    referenced_plan: Some(&referenced),
                    referenced_classic_plan: None,
                    settings: DecodeSettings::strict(),
                }],
                PixelFormat::Rgb8,
            )
            .expect("prepare retained CUDA RGB plan");
            assert_eq!(prepared.colors[0].dimensions, plan.output_dims());
            assert_eq!(prepared.colors[0].report.parse_us, 0);
            assert!(!prepared.shared_payload.is_empty());
        }
        let _ = BatchLayout::Nhwc;
    }

    #[test]
    fn tiny_caps_force_multiple_exact_rgb_chunks_without_changing_output_when_runtime_required() {
        if !j2k_test_support::cuda_runtime_gate(module_path!()) {
            return;
        }
        let fixture = j2k_test_support::openjph_batch_fixtures()
            .iter()
            .find(|fixture| fixture.name == "openjph-rgb-s12-53-single-raw")
            .expect("independent signed RGB12 single-tile fixture");
        let encoded = Arc::<[u8]>::from(fixture.encoded);
        let options = BatchDecodeOptions {
            layout: BatchLayout::Nhwc,
            ..BatchDecodeOptions::default()
        };
        let prepared = prepare_batch(vec![EncodedImage::full(Arc::clone(&encoded))], options)
            .expect("prepare independent signed RGB12 fixture");
        let image = &prepared.groups()[0].images()[0];
        let referenced_plan = image
            .htj2k_plan()
            .expect("retained HTJ2K plan")
            .adapter_view()
            .downcast_ref::<j2k_native::J2kReferencedHtj2kPlan>()
            .expect("native referenced HTJ2K plan adapter");

        let mut cpu = CpuBatchDecoder::new(options);
        let oracle = cpu
            .decode_prepared(&prepared)
            .expect("CPU signed RGB12 oracle");
        let CpuBatchSamples::I16(expected) = oracle.groups()[0].samples() else {
            panic!("signed RGB12 oracle must use I16 storage")
        };
        let expected = expected
            .iter()
            .flat_map(|sample| sample.to_ne_bytes())
            .collect::<Vec<_>>();

        let context = CudaContext::system_default().expect("CUDA context");
        let before = context
            .diagnostics()
            .expect("diagnostics before chunked decode");
        let mut session = CudaSession::with_context(context.clone());
        session.set_htj2k_decode_chunk_limits_for_test(HtGpuJobChunkLimits::new(
            NonZeroUsize::MIN,
            fixture.encoded.len(),
            htj2k_cleanup_multi_descriptor_bytes(),
        ));
        let pending = submit_native_color_resident_prepared_batch(
            &[NativeColorBatchInput {
                source_index: 0,
                bytes: &encoded,
                device_plan: image.plan(),
                referenced_plan: Some(referenced_plan),
                referenced_classic_plan: None,
                settings: DecodeSettings::strict(),
            }],
            &mut session,
            PixelFormat::RgbI16,
            BatchLayout::Nhwc,
        )
        .expect("submit tiny-cap signed RGB12 chunks");
        let (output, _report) = pending
            .finish()
            .expect("finish tiny-cap signed RGB12 chunks");
        assert!(session.last_htj2k_decode_chunk_count_for_test() > 1);
        let after = context
            .diagnostics()
            .expect("diagnostics after chunked decode");
        assert_eq!(
            after.device_to_host_operations - before.device_to_host_operations,
            1,
            "one homogeneous output group must validate all chunk statuses with one transfer"
        );
        let mut actual = vec![0_u8; expected.len()];
        output
            .buffer
            .copy_range_to_host(output.ranges[0].offset, &mut actual)
            .expect("download tiny-cap signed RGB12 output");
        assert_eq!(actual, expected);
    }
}
