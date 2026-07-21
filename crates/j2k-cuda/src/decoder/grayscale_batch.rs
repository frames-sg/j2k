// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use j2k_core::DeviceSubmitSession;
use j2k_cuda_runtime::{
    CudaDeviceBufferRange, CudaExternalDeviceBufferViewMut, CudaJ2kStoreGray16Job,
    CudaJ2kStoreGray16Target, CudaJ2kStoreGray8Job, CudaJ2kStoreGray8Target,
    CudaJ2kStoreGrayI16Target, CudaQueuedJ2kStoreBatch,
};

use super::color_batch::finalize_color_batch_decode_report;
use super::pending_completion::{PendingCleanup, PendingDecodeCompletion};
use super::plan::{
    build_cuda_classic_grayscale_plans_from_referenced_with_profile,
    build_cuda_htj2k_grayscale_plan_from_bytes_for_device_plan_with_profile,
    build_cuda_htj2k_grayscale_plan_from_bytes_with_profile,
    build_cuda_htj2k_grayscale_plans_from_referenced_with_profile,
};
use super::resident::{
    can_batch_color_idwt, decode_cuda_component_subbands_with_resources,
    enqueue_component_classic_batches, enqueue_component_cleanup_dequant_batches,
    finish_cuda_component_decode, pooled_cuda_buffer, run_color_component_idwt_batches,
    run_component_cleanup_dequant_batches, run_cuda_component_idwt_steps,
};
use super::{
    cuda_error, cuda_range_storage, profile, BackendKind, CudaDecodedComponent,
    CudaHtj2kDecodePlan, CudaHtj2kProfileReport, CudaQueuedIdwtBatch, CudaSession,
    CudaSurfaceStats, DecodeSettings, DeviceDecodePlan, Error, NativeDecoderContext, PixelFormat,
    Surface, SurfaceResidency, CUDA_HTJ2K_OUTPUT_FORMAT_UNSUPPORTED,
};
use crate::allocation::HostPhaseBudget;

mod completion;
mod execution;
mod preparation;
mod store;

use self::completion::{
    grayscale_htj2k_job_identities, GrayscaleBatchOutput, GrayscaleHtj2kCleanup,
    GrayscalePendingCompletion, StoredGrayscaleBatch,
};
pub(crate) use self::completion::{
    GrayscaleOwnedBatch, SubmittedGrayscaleExternalBatch, SubmittedGrayscaleResidentBatch,
};
use self::execution::decode_grayscale_cuda_batch_with_profile;
#[cfg(test)]
use self::preparation::prepare_grayscale_batch;
pub(crate) use self::preparation::GrayscaleBatchInput;

pub(super) fn decode_grayscale_cuda_resident_batch_surfaces_with_profile(
    inputs: &[&[u8]],
    session: &mut CudaSession,
    fmt: PixelFormat,
    collect_stage_timings: bool,
) -> Result<(Vec<Surface>, CudaHtj2kProfileReport), Error> {
    let inputs = inputs
        .iter()
        .map(|input| GrayscaleBatchInput::full(input))
        .collect::<Vec<_>>();
    decode_grayscale_cuda_resident_prepared_batch_surfaces_with_profile(
        &inputs,
        DecodeSettings::default(),
        session,
        fmt,
        collect_stage_timings,
    )
}

pub(crate) fn decode_grayscale_cuda_resident_prepared_batch_surfaces_with_profile(
    inputs: &[GrayscaleBatchInput<'_>],
    settings: DecodeSettings,
    session: &mut CudaSession,
    fmt: PixelFormat,
    collect_stage_timings: bool,
) -> Result<(Vec<Surface>, CudaHtj2kProfileReport), Error> {
    let (output, report, pending) = decode_grayscale_cuda_batch_with_profile(
        inputs,
        settings,
        session,
        fmt,
        collect_stage_timings,
        None,
        false,
    )?;
    if pending.is_some() {
        return Err(Error::UnsupportedCudaRequest {
            reason: "synchronous CUDA grayscale decode unexpectedly retained pending work",
        });
    }
    let GrayscaleBatchOutput::Owned(output) = output else {
        return Err(Error::UnsupportedCudaRequest {
            reason: CUDA_HTJ2K_OUTPUT_FORMAT_UNSUPPORTED,
        });
    };
    Ok((output.surfaces, report))
}

pub(super) fn decode_grayscale_cuda_resident_batch_into_with_profile(
    inputs: &[&[u8]],
    session: &mut CudaSession,
    fmt: PixelFormat,
    destination: &mut CudaExternalDeviceBufferViewMut<'_>,
    collect_stage_timings: bool,
) -> Result<(Vec<CudaDeviceBufferRange>, CudaHtj2kProfileReport), Error> {
    let inputs = inputs
        .iter()
        .map(|input| GrayscaleBatchInput::full(input))
        .collect::<Vec<_>>();
    decode_grayscale_cuda_resident_prepared_batch_into_with_profile(
        &inputs,
        DecodeSettings::default(),
        session,
        fmt,
        destination,
        collect_stage_timings,
    )
}

pub(crate) fn decode_grayscale_cuda_resident_prepared_batch_into_with_profile(
    inputs: &[GrayscaleBatchInput<'_>],
    settings: DecodeSettings,
    session: &mut CudaSession,
    fmt: PixelFormat,
    destination: &mut CudaExternalDeviceBufferViewMut<'_>,
    collect_stage_timings: bool,
) -> Result<(Vec<CudaDeviceBufferRange>, CudaHtj2kProfileReport), Error> {
    let (output, report, pending) = decode_grayscale_cuda_batch_with_profile(
        inputs,
        settings,
        session,
        fmt,
        collect_stage_timings,
        Some(destination),
        false,
    )?;
    if pending.is_some() {
        return Err(Error::UnsupportedCudaRequest {
            reason: "synchronous external CUDA grayscale decode unexpectedly retained pending work",
        });
    }
    let GrayscaleBatchOutput::External(ranges) = output else {
        return Err(Error::UnsupportedCudaRequest {
            reason: CUDA_HTJ2K_OUTPUT_FORMAT_UNSUPPORTED,
        });
    };
    Ok((ranges, report))
}

pub(crate) fn submit_grayscale_cuda_resident_prepared_batch_into(
    inputs: &[GrayscaleBatchInput<'_>],
    settings: DecodeSettings,
    session: &mut CudaSession,
    fmt: PixelFormat,
    destination: &mut CudaExternalDeviceBufferViewMut<'_>,
) -> Result<SubmittedGrayscaleExternalBatch, Error> {
    let (output, report, completion) = decode_grayscale_cuda_batch_with_profile(
        inputs,
        settings,
        session,
        fmt,
        false,
        Some(destination),
        true,
    )?;
    let GrayscaleBatchOutput::External(ranges) = output else {
        return Err(Error::UnsupportedCudaRequest {
            reason: CUDA_HTJ2K_OUTPUT_FORMAT_UNSUPPORTED,
        });
    };
    let completion = completion.ok_or(Error::UnsupportedCudaRequest {
        reason: "CUDA external batch submission did not retain a completion owner",
    })?;
    Ok(SubmittedGrayscaleExternalBatch {
        ranges,
        report,
        completion: Some(completion),
    })
}

pub(crate) fn submit_grayscale_cuda_resident_prepared_batch(
    inputs: &[GrayscaleBatchInput<'_>],
    settings: DecodeSettings,
    session: &mut CudaSession,
    fmt: PixelFormat,
) -> Result<SubmittedGrayscaleResidentBatch, Error> {
    let (output, report, completion) = decode_grayscale_cuda_batch_with_profile(
        inputs, settings, session, fmt, false, None, true,
    )?;
    let GrayscaleBatchOutput::Owned(output) = output else {
        return Err(Error::UnsupportedCudaRequest {
            reason: CUDA_HTJ2K_OUTPUT_FORMAT_UNSUPPORTED,
        });
    };
    let completion = completion.ok_or(Error::UnsupportedCudaRequest {
        reason: "CUDA resident grayscale submission did not retain a completion owner",
    })?;
    Ok(SubmittedGrayscaleResidentBatch {
        output: Some(output),
        report,
        completion: Some(completion),
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use j2k::{
        prepare_batch, BatchDecodeOptions, DeviceDecodePlan, DeviceDecodeRequest, EncodedImage,
        PreparationDepth,
    };
    use j2k_core::{Downscale, PixelFormat, Rect};
    use j2k_native::{encode_htj2k, DecodeSettings, EncodeOptions};

    use super::completion::{map_grayscale_status_error, GrayscaleJobIdentity};
    use super::{prepare_grayscale_batch, GrayscaleBatchInput};

    #[test]
    fn grayscale_kernel_failure_maps_to_responsible_source_index() {
        let identities = [
            GrayscaleJobIdentity {
                source_index: 3,
                original_job_index: 0,
            },
            GrayscaleJobIdentity {
                source_index: 9,
                original_job_index: 1,
            },
        ];
        let mapped = map_grayscale_status_error(
            j2k_cuda_runtime::CudaError::KernelJobStatus {
                kernel: "injected",
                job_index: 1,
                code: 7,
                detail: 11,
            },
            &identities,
        );
        assert!(matches!(
            mapped,
            crate::Error::CudaTier1JobFailed {
                source_index: 9,
                original_job_index: 1,
                ..
            }
        ));
    }

    #[test]
    fn grayscale_batch_rebases_two_plans_into_one_shared_payload() {
        let pixels = (0_u16..64)
            .map(|value| u8::try_from(value).expect("fixture byte"))
            .collect::<Vec<_>>();
        let encoded = encode_htj2k(
            &pixels,
            8,
            8,
            1,
            8,
            false,
            &EncodeOptions {
                reversible: true,
                num_decomposition_levels: 1,
                ..EncodeOptions::default()
            },
        )
        .expect("HTJ2K grayscale fixture");
        let prepared = prepare_grayscale_batch(
            &[
                GrayscaleBatchInput::full(encoded.as_slice()),
                GrayscaleBatchInput::full(encoded.as_slice()),
            ],
            PixelFormat::Gray8,
            DecodeSettings::strict(),
        )
        .expect("shared grayscale batch plan");

        assert_eq!(prepared.plans.len(), 2);
        assert!(!prepared.shared_payload.is_empty());
        assert!(prepared.plans.iter().all(|plan| plan.payload().is_empty()));
        let first_max = prepared.plans[0]
            .code_blocks()
            .iter()
            .map(|block| block.payload_offset + u64::from(block.payload_len))
            .max()
            .expect("first block payload");
        let second_min = prepared.plans[1]
            .code_blocks()
            .iter()
            .map(|block| block.payload_offset)
            .min()
            .expect("second block payload");
        assert!(second_min >= first_max);
    }

    #[test]
    fn prepared_htj2k_batch_uses_retained_offsets_without_reparsing() {
        let pixels = (0_u8..64).collect::<Vec<_>>();
        let encoded = encode_htj2k(
            &pixels,
            8,
            8,
            1,
            8,
            false,
            &EncodeOptions {
                reversible: true,
                num_decomposition_levels: 1,
                ..EncodeOptions::default()
            },
        )
        .expect("HTJ2K retained-plan fixture");
        let prepared = prepare_batch(
            vec![EncodedImage::full(Arc::from(encoded))],
            BatchDecodeOptions::default(),
        )
        .expect("shared retained-plan preparation");
        let [group] = prepared.groups() else {
            panic!("expected one prepared group")
        };
        let [image] = group.images() else {
            panic!("expected one prepared image")
        };
        assert_eq!(image.preparation_depth(), PreparationDepth::Htj2kOffsetPlan);
        let referenced_plan = image
            .htj2k_plan()
            .expect("retained HTJ2K plan")
            .adapter_view()
            .downcast_ref::<j2k_native::J2kReferencedHtj2kPlan>()
            .expect("native referenced HTJ2K plan adapter");
        let settings = DecodeSettings {
            resolve_palette_indices: true,
            strict: group.options().settings.is_strict(),
            target_resolution: None,
        };

        let cuda = prepare_grayscale_batch(
            &[GrayscaleBatchInput {
                source_index: 0,
                bytes: image.bytes(),
                device_plan: Some(image.plan()),
                referenced_plan: Some(referenced_plan),
                referenced_classic_plan: None,
            }],
            PixelFormat::Gray8,
            settings,
        )
        .expect("CUDA retained-plan preparation");

        assert_eq!(cuda.reports[0].parse_us, 0);
        assert_eq!(cuda.reports[0].plan_us, 0);
        assert!(cuda.plans[0].payload().is_empty());
        assert!(!cuda.shared_payload.is_empty());
    }

    #[test]
    fn grayscale_batch_prepares_roi_and_reduced_requests_in_one_payload_arena() {
        let pixels = (0_u16..16 * 16)
            .map(|value| u8::try_from(value).expect("fixture byte"))
            .collect::<Vec<_>>();
        let encoded = encode_htj2k(
            &pixels,
            16,
            16,
            1,
            8,
            false,
            &EncodeOptions {
                reversible: true,
                num_decomposition_levels: 2,
                ..EncodeOptions::default()
            },
        )
        .expect("HTJ2K grayscale geometry fixture");
        let roi_plan = DeviceDecodePlan::for_image(
            (16, 16),
            DeviceDecodeRequest::Region {
                roi: Rect {
                    x: 3,
                    y: 5,
                    w: 7,
                    h: 6,
                },
            },
        )
        .expect("ROI plan");
        let reduced_plan = DeviceDecodePlan::for_image(
            (16, 16),
            DeviceDecodeRequest::Scaled {
                scale: Downscale::Half,
            },
        )
        .expect("reduced plan");

        let prepared = prepare_grayscale_batch(
            &[
                GrayscaleBatchInput {
                    source_index: 0,
                    bytes: &encoded,
                    device_plan: Some(roi_plan),
                    referenced_plan: None,
                    referenced_classic_plan: None,
                },
                GrayscaleBatchInput {
                    source_index: 1,
                    bytes: &encoded,
                    device_plan: Some(reduced_plan),
                    referenced_plan: None,
                    referenced_classic_plan: None,
                },
            ],
            PixelFormat::Gray8,
            DecodeSettings::strict(),
        )
        .expect("prepare geometry batch");

        assert_eq!(prepared.plans[0].dimensions(), roi_plan.output_dims());
        assert_eq!(prepared.plans[1].dimensions(), reduced_plan.output_dims());
        assert!(prepared.plans.iter().all(|plan| plan.payload().is_empty()));
        assert!(!prepared.shared_payload.is_empty());
    }
}
