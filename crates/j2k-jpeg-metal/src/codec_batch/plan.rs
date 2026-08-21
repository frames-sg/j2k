// SPDX-License-Identifier: MIT OR Apache-2.0

//! Normalized RGB8 batch planning for raw and prepared sources.

use super::owner_accounting::{distinct_decoder_retained_bytes, Rgb8BatchBuildContext};
use super::request::{Rgb8MetalBatchOp, Rgb8MetalBatchSource};
use super::source::{
    decoder_resident_restart_interval_mcus, decoder_resident_sampling_family,
    ResolvedRgb8BatchSource,
};
use crate::{
    batch, plan_owner_ledger::preflight_collective_metadata, scaled_dims, session, Codec, Error,
};
use j2k_core::{BackendRequest, PixelFormat, Rect};

pub(super) struct Rgb8MetalBatchPlan {
    pub(super) requests: Vec<batch::QueuedRequest>,
    pub(super) output_dimensions: Option<(u32, u32)>,
}

pub(super) fn build_rgb8_batch_plan<S>(
    sources: &[S],
    op: Rgb8MetalBatchOp,
    context: &mut Rgb8BatchBuildContext,
    mut resolve: impl FnMut(&S, &Rgb8BatchBuildContext) -> Result<ResolvedRgb8BatchSource, Error>,
) -> Result<Rgb8MetalBatchPlan, Error> {
    let mut output_dimensions = None;
    let mut sampling_family = None;
    for source in sources {
        let resolved = resolve(source, context)?;
        Codec::observe_rgb8_batch_output_dimensions(
            &mut output_dimensions,
            resolved.output_dimensions,
        )?;
        if let Some(first) = sampling_family {
            if first != resolved.sampling_family {
                return Err(Error::capability_rejected(j2k_core::CapabilityRejection::unsupported_sampling("JPEG Metal reusable resident batch output requires one batch to use the same fast-packet sampling family")));
            }
        } else {
            sampling_family = Some(resolved.sampling_family);
        }
        if op == Rgb8MetalBatchOp::Full
            && matches!(
                resolved.sampling_family,
                batch::SamplingFamily::Fast422 | batch::SamplingFamily::Fast444
            )
            && resolved.restart_coded
        {
            return Err(Error::capability_rejected(j2k_core::CapabilityRejection::unsupported_sampling("JPEG Metal reusable resident batch output does not support restart-coded full-tile 4:2:2 or 4:4:4 batches")));
        }

        let admission = context.plan_owners.preflight(
            &context.requests,
            &resolved.request,
            resolved.cache_retained_bytes,
        )?;
        let execution_external_live_bytes = context
            .budget
            .live_bytes()
            .checked_add(context.external_live_bytes)
            .ok_or(j2k_jpeg::adapter::JpegPlanCacheError::Invariant(
                "JPEG Metal batch execution external baseline overflow",
            ))?;
        preflight_collective_metadata(
            context.collective_what,
            admission.retained_bytes(),
            resolved.cache_retained_bytes,
            execution_external_live_bytes,
        )?;
        context.requests.push(resolved.request);
        context.plan_owners.commit(admission);
    }
    let execution_external_live_bytes = context
        .budget
        .live_bytes()
        .checked_add(context.external_live_bytes)
        .ok_or(j2k_jpeg::adapter::JpegPlanCacheError::Invariant(
            "JPEG Metal batch execution baseline overflow",
        ))?;
    batch::stamp_execution_owner_baseline(&mut context.requests, 0, execution_external_live_bytes);
    Ok(Rgb8MetalBatchPlan {
        requests: core::mem::take(&mut context.requests),
        output_dimensions,
    })
}

pub(super) fn rgb8_metal_output_dimensions_for_op(
    full_dimensions: (u32, u32),
    op: j2k_jpeg::JpegDecodeOp,
) -> Option<(u32, u32)> {
    match op {
        j2k_jpeg::JpegDecodeOp::Full => Some(full_dimensions),
        j2k_jpeg::JpegDecodeOp::Scaled(scale) => Some(scaled_dims(full_dimensions, scale)),
        j2k_jpeg::JpegDecodeOp::RegionScaled { roi, scale } => {
            let scaled = Rect {
                x: roi.x,
                y: roi.y,
                w: roi.w,
                h: roi.h,
            }
            .scaled_covering(scale);
            Some((scaled.w, scaled.h))
        }
        j2k_jpeg::JpegDecodeOp::Region(_) => None,
    }
}

impl Codec {
    pub(super) fn observe_rgb8_batch_output_dimensions(
        first_output_dimensions: &mut Option<(u32, u32)>,
        output_dimensions: (u32, u32),
    ) -> Result<(), Error> {
        if let Some(first) = *first_output_dimensions {
            if first != output_dimensions {
                return Err(Error::capability_rejected(
                    j2k_core::CapabilityRejection::unsupported_format(
                        "JPEG Metal reusable RGB8 batch output requires matching output dimensions",
                    ),
                ));
            }
        } else {
            *first_output_dimensions = Some(output_dimensions);
        }
        Ok(())
    }

    pub(super) fn rgb8_batch_op_and_dimensions(
        op: Rgb8MetalBatchOp,
        dimensions: (u32, u32),
    ) -> (batch::BatchOp, (u32, u32)) {
        match op {
            Rgb8MetalBatchOp::Full => (batch::BatchOp::Full, dimensions),
            Rgb8MetalBatchOp::Scaled(scale) => {
                let (w, h) = dimensions;
                (
                    batch::BatchOp::RegionScaled {
                        roi: Rect { x: 0, y: 0, w, h },
                        scale,
                    },
                    scaled_dims((w, h), scale),
                )
            }
            Rgb8MetalBatchOp::RegionScaled { roi, scale } => {
                let scaled = roi.scaled_covering(scale);
                (
                    batch::BatchOp::RegionScaled { roi, scale },
                    (scaled.w, scaled.h),
                )
            }
        }
    }

    pub(super) fn rgb8_batch_jpeg_decode_op(op: Rgb8MetalBatchOp) -> j2k_jpeg::JpegDecodeOp {
        match op {
            Rgb8MetalBatchOp::Full => j2k_jpeg::JpegDecodeOp::Full,
            Rgb8MetalBatchOp::Scaled(scale) => j2k_jpeg::JpegDecodeOp::Scaled(scale),
            Rgb8MetalBatchOp::RegionScaled { roi, scale } => j2k_jpeg::JpegDecodeOp::RegionScaled {
                roi: roi.into(),
                scale,
            },
        }
    }

    pub(super) fn plan_rgb8_metal_batch(
        source: Rgb8MetalBatchSource<'_, '_>,
        op: Rgb8MetalBatchOp,
        track_output_dimensions: bool,
    ) -> Result<(Rgb8MetalBatchPlan, usize), Error> {
        match source {
            Rgb8MetalBatchSource::Bytes(inputs) => {
                let mut state = session::SessionState::default();
                let mut context = Rgb8BatchBuildContext::new(
                    inputs.len(),
                    "JPEG Metal RGB8 batch request plan",
                    "JPEG Metal RGB8 batch requests",
                    0,
                    "JPEG Metal RGB8 raw request owners and metadata",
                )?;
                let mut plan =
                    build_rgb8_batch_plan(inputs, op, &mut context, |input, context| {
                        let external_live_bytes = context.resolver_external_live_bytes()?;
                        let (resolved, decoder) = state
                            .resolve_jpeg_plan_with_decoder_and_external_live(
                                input,
                                external_live_bytes,
                            )?;
                        let (batch_op, output_dimensions) =
                            Self::rgb8_batch_op_and_dimensions(op, decoder.info().dimensions);
                        let sampling_family = resolved.shape.sampling_family;
                        let restart_coded = resolved.shape.restart_interval.is_some();
                        drop(decoder);
                        let request = batch::QueuedRequest::new_shared(
                            resolved.input,
                            PixelFormat::Rgb8,
                            BackendRequest::Metal,
                            batch_op,
                            resolved.fast_packet,
                            resolved.shape,
                        );
                        Ok(ResolvedRgb8BatchSource {
                            request,
                            output_dimensions,
                            sampling_family,
                            restart_coded,
                            cache_retained_bytes: state
                                .jpeg_plan_cache_diagnostics()
                                .retained_bytes,
                        })
                    })?;
                if !track_output_dimensions {
                    plan.output_dimensions = None;
                }
                Ok((plan, inputs.len()))
            }
            Rgb8MetalBatchSource::Decoders(decoders) => {
                let decoder_owner_bytes = distinct_decoder_retained_bytes(decoders)?;
                let mut context = Rgb8BatchBuildContext::new(
                    decoders.len(),
                    "JPEG Metal RGB8 decoder batch request plan",
                    "JPEG Metal RGB8 decoder batch requests",
                    decoder_owner_bytes,
                    "JPEG Metal RGB8 decoder request owners and metadata",
                )?;
                let mut plan =
                    build_rgb8_batch_plan(decoders, op, &mut context, |decoder, _context| {
                        let (batch_op, output_dimensions) = Self::rgb8_batch_op_and_dimensions(
                            op,
                            decoder.inner().info().dimensions,
                        );
                        Ok(ResolvedRgb8BatchSource {
                            request: decoder.rgb8_metal_request(batch_op),
                            output_dimensions,
                            sampling_family: decoder_resident_sampling_family(decoder),
                            restart_coded: decoder_resident_restart_interval_mcus(decoder) != 0,
                            cache_retained_bytes: 0,
                        })
                    })?;
                if !track_output_dimensions {
                    plan.output_dimensions = None;
                }
                Ok((plan, decoders.len()))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Decoder;
    use j2k_core::{Downscale, DEFAULT_MAX_HOST_ALLOCATION_BYTES};
    use j2k_jpeg::{
        encode_jpeg_baseline, JpegBackend, JpegEncodeOptions, JpegSamples, JpegSubsampling,
    };

    const BASELINE_420: &[u8] = include_bytes!("../../fixtures/jpeg/baseline_420_16x16.jpg");
    const BASELINE_444: &[u8] = include_bytes!("../../fixtures/jpeg/baseline_444_8x8.jpg");

    fn assert_raw_and_prepared_plans_match(inputs: &[&[u8]], op: Rgb8MetalBatchOp) {
        let decoders = inputs
            .iter()
            .map(|input| Decoder::new(input).expect("prepared decoder"))
            .collect::<Vec<_>>();
        let decoder_refs = decoders.iter().collect::<Vec<_>>();
        let raw = Codec::plan_rgb8_metal_batch(Rgb8MetalBatchSource::Bytes(inputs), op, true);
        let prepared =
            Codec::plan_rgb8_metal_batch(Rgb8MetalBatchSource::Decoders(&decoder_refs), op, true);
        match (raw, prepared) {
            (Ok((raw, raw_count)), Ok((prepared, prepared_count))) => {
                assert_eq!(raw_count, prepared_count);
                assert_eq!(raw.output_dimensions, prepared.output_dimensions);
                assert_eq!(raw.requests.len(), prepared.requests.len());
                for (raw, prepared) in raw.requests.iter().zip(&prepared.requests) {
                    assert_eq!(raw.key(), prepared.key());
                }
            }
            (Err(raw), Err(prepared)) => assert_eq!(raw.to_string(), prepared.to_string()),
            (raw, prepared) => panic!(
                "raw and prepared planning diverged: raw={:?}, prepared={:?}",
                raw.err(),
                prepared.err()
            ),
        }
    }

    #[test]
    fn raw_and_prepared_sources_produce_equivalent_normalized_requests() {
        let inputs = [BASELINE_420, BASELINE_420];
        for op in [
            Rgb8MetalBatchOp::Full,
            Rgb8MetalBatchOp::Scaled(Downscale::Quarter),
            Rgb8MetalBatchOp::RegionScaled {
                roi: Rect {
                    x: 1,
                    y: 2,
                    w: 10,
                    h: 9,
                },
                scale: Downscale::Half,
            },
        ] {
            assert_raw_and_prepared_plans_match(&inputs, op);
        }
    }

    #[test]
    fn raw_and_prepared_sources_reject_mismatched_dimensions_identically() {
        assert_raw_and_prepared_plans_match(&[BASELINE_420, BASELINE_444], Rgb8MetalBatchOp::Full);
    }

    #[test]
    fn raw_and_prepared_sources_reject_mixed_sampling_identically() {
        let rgb = vec![127; 16 * 16 * 3];
        let fast444 = encode_jpeg_baseline(
            JpegSamples::Rgb8 {
                data: &rgb,
                width: 16,
                height: 16,
            },
            JpegEncodeOptions {
                quality: 90,
                subsampling: JpegSubsampling::Ybr444,
                restart_interval: None,
                backend: JpegBackend::Cpu,
            },
        )
        .expect("encode 4:4:4 fixture");
        assert_raw_and_prepared_plans_match(&[BASELINE_420, &fast444.data], Rgb8MetalBatchOp::Full);
    }

    #[test]
    fn raw_and_prepared_sources_apply_restart_restrictions_identically() {
        let rgb = vec![63; 64 * 32 * 3];
        let restart422 = encode_jpeg_baseline(
            JpegSamples::Rgb8 {
                data: &rgb,
                width: 64,
                height: 32,
            },
            JpegEncodeOptions {
                quality: 90,
                subsampling: JpegSubsampling::Ybr422,
                restart_interval: Some(4),
                backend: JpegBackend::Cpu,
            },
        )
        .expect("encode restart-coded 4:2:2 fixture");
        let inputs = [&restart422.data[..], &restart422.data[..]];
        assert_raw_and_prepared_plans_match(&inputs, Rgb8MetalBatchOp::Full);
        assert_raw_and_prepared_plans_match(&inputs, Rgb8MetalBatchOp::Scaled(Downscale::Half));
    }

    #[test]
    fn shared_builder_accounts_for_plan_cache_bytes() {
        let decoder = Decoder::new(BASELINE_420).expect("prepared decoder");
        let mut context = Rgb8BatchBuildContext::new(
            1,
            "cache accounting test",
            "cache accounting requests",
            0,
            "cache accounting owners",
        )
        .expect("batch context");
        let result = build_rgb8_batch_plan(&[()], Rgb8MetalBatchOp::Full, &mut context, |(), _| {
            Ok(ResolvedRgb8BatchSource {
                request: decoder.rgb8_metal_request(batch::BatchOp::Full),
                output_dimensions: (16, 16),
                sampling_family: batch::SamplingFamily::Fast420,
                restart_coded: false,
                cache_retained_bytes: DEFAULT_MAX_HOST_ALLOCATION_BYTES,
            })
        });
        let Err(error) = result else {
            panic!("cache plus request owners must exceed the host cap");
        };
        assert!(error.to_string().contains("allocation"));
    }

    #[test]
    fn source_has_one_rgb8_batch_builder() {
        let source = include_str!("plan.rs");
        let builder = concat!("fn build_rgb8_", "batch_plan");
        assert_eq!(source.matches(builder).count(), 1);
    }
}
