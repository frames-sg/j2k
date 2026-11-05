// SPDX-License-Identifier: MIT OR Apache-2.0

//! Per-image metadata validation and retained native plan construction.

use super::super::{
    extract_j2k_codestream_payload, parse_image_info_with_settings, Arc, BatchAlpha,
    BatchCodecRoute, BatchColor, BatchDecodeOptions, BatchErrorStage, BatchExecutionShape,
    BatchGroupInfo, BatchItemError, BatchWaveletTransform, BatchWorker, Colorspace,
    CompressedTransferSyntax, CpuExecution, DeviceDecodePlan, EncodedImage, J2kCodestreamRange,
    J2kError, J2kSupportInfo, NativeSampleType, NonRepresentableReason, PreparationDepth,
    PrepareImageResult, PreparedClassicPlan, PreparedCodecPlan, PreparedHtj2kPlan, PreparedImage,
    PreparedImageInner,
};

pub(super) fn prepare_image(
    input: EncodedImage,
    source_index: usize,
    options: BatchDecodeOptions,
    worker: &mut BatchWorker,
) -> PrepareImageResult {
    // Parse with the caller's settings, as the native image below parses:
    // a lenient batch then plans from the metadata its decode recovers.
    let parsed =
        parse_image_info_with_settings(&input.bytes, options.settings).map_err(|source| {
            BatchItemError::Codec {
                stage: BatchErrorStage::Prepare,
                source: Arc::new(source),
            }
        })?;
    let support = parsed.into_support_info();
    let plan = DeviceDecodePlan::for_image(support.info.dimensions, input.request.device_request())
        .map_err(|source| BatchItemError::Codec {
            stage: BatchErrorStage::Prepare,
            source: Arc::new(source),
        })?;
    let mut info = batch_group_info(&support, plan, options)?;
    let codestream =
        extract_j2k_codestream_payload(&input.bytes).map_err(|source| BatchItemError::Codec {
            stage: BatchErrorStage::Prepare,
            source: Arc::new(source),
        })?;
    let codestream_range = J2kCodestreamRange {
        offset: codestream.codestream_offset(),
        length: codestream.codestream().len(),
    };
    // Check the main-header ladder here; plan construction below also checks
    // tile overrides. Neither failure can be repaired by a fallback decode.
    let native_image = crate::backend::image_at_scale(&input.bytes, options.settings, plan.scale())
        .map_err(|source| BatchItemError::Codec {
            stage: BatchErrorStage::Prepare,
            source: Arc::new(source),
        })?;
    // Size reduced outputs on the reference grid, now that the image's origin
    // is known: an origin that is not a multiple of the scale narrows them.
    let plan = DeviceDecodePlan::for_reference_grid(
        crate::J2kReferenceGrid::of_native(&native_image, support.info.dimensions),
        input.request.device_request(),
    )
    .map_err(|source| BatchItemError::Codec {
        stage: BatchErrorStage::Prepare,
        source: Arc::new(source),
    })?;
    info.dimensions = plan.output_dims();
    let used_lenient_metadata_recovery = native_image.used_lenient_metadata_recovery();
    // The planned alpha must be the alpha the native decode writes.
    if (info.alpha != BatchAlpha::None) != native_image.has_alpha() {
        return Err(nonrepresentable(NonRepresentableReason::UnsupportedColor));
    }
    let codec_plan = match info.route {
        BatchCodecRoute::Classic => {
            prepare_classic_offset_plan(&native_image, &info, plan, worker)?
                .map_or(PreparedCodecPlan::MetadataOnly, PreparedCodecPlan::Classic)
        }
        BatchCodecRoute::Htj2k => prepare_htj2k_offset_plan(&native_image, &info, plan, worker)?
            .map_or(PreparedCodecPlan::MetadataOnly, PreparedCodecPlan::Htj2k),
    };
    reconcile_codec_plan_metadata(&mut info, &codec_plan)?;
    let cpu_execution = choose_cpu_execution(
        &codec_plan,
        &info,
        native_image.region_decode_streams_rows(),
        options.planned_decode_bound(planned_decode_bound()),
    );
    let preparation_depth = codec_plan.preparation_depth();
    let execution_shape = batch_execution_shape(&support, plan, preparation_depth);
    let image = PreparedImage {
        inner: Arc::new(PreparedImageInner {
            bytes: input.bytes,
            request: input.request,
            source_index,
            decode_settings: options.settings,
            used_lenient_metadata_recovery,
            support: Arc::new(support),
            plan,
            codestream_range,
            codec_plan,
            cpu_execution,
        }),
    };
    Ok((image, info, execution_shape))
}

/// Choose the CPU execution from the planned working memory, before any
/// decode work, with the same accounting the offset plan's execution
/// enforces.
///
/// A Gray8 or RGB8 image whose planned execution exceeds
/// [`STREAM_ABOVE_PLANNED_BYTES`] streams its rows when the region decode
/// can stream them (`region_streams_rows`): the offset plan holds
/// whole-image f32 planes and coefficients (about 14 times the 8-bit
/// output), which every image in flight kept until its decode ended, while
/// the streamed route holds a band of rows. While its plan fits the
/// per-decode bound, the image keeps it for decoding alone in a Gray8 group
/// ([`CpuExecution::StreamedRowsOrPlan`]). Other outputs keep their offset
/// plan unless its serial execution exceeds the per-decode bound: a 16-bit
/// Gray or RGB image whose rows stream then streams them, and any other
/// image is rejected, as are images whose region decode would hand over
/// whole component planes instead of rows. The claim of an offset plan is
/// its parallel peak.
pub(super) fn choose_cpu_execution(
    codec_plan: &PreparedCodecPlan,
    info: &BatchGroupInfo,
    region_streams_rows: bool,
    decode_bound: usize,
) -> CpuExecution {
    let planned = |parallel| match codec_plan {
        PreparedCodecPlan::Classic(plan) => {
            j2k_native::referenced_classic_plan_execution_bytes(plan.cpu_plan(), parallel)
        }
        PreparedCodecPlan::Htj2k(plan) => {
            j2k_native::referenced_htj2k_plan_execution_bytes(plan.cpu_plan(), parallel)
        }
        PreparedCodecPlan::MetadataOnly => Ok(0),
    };
    if matches!(codec_plan, PreparedCodecPlan::MetadataOnly) {
        return CpuExecution::ComponentPlanes;
    }
    let streams_rows = region_streams_rows && has_streamed_output(info);
    // Only 8-bit outputs stream by preference, where the streamed route is
    // measured to be the better one; 16-bit outputs stream only in place of
    // a plan the bound refuses.
    let prefers_streaming = streams_rows && info.sample_type == NativeSampleType::U8;
    let exceeds_bound = if streams_rows {
        CpuExecution::StreamedRows
    } else {
        CpuExecution::ExceedsDecodeBound
    };
    match (planned(false), planned(true)) {
        (
            Err(j2k_native::DecodeError::Validation(j2k_native::ValidationError::ImageTooLarge)),
            _,
        ) => exceeds_bound,
        (Ok(serial), _) if serial > decode_bound => exceeds_bound,
        (Ok(_), Ok(bytes)) if !prefers_streaming || bytes <= stream_above_planned_bytes() => {
            CpuExecution::OffsetPlan { bytes }
        }
        (Ok(_), Ok(bytes)) => CpuExecution::StreamedRowsOrPlan { bytes },
        // An 8-bit plan that fits only serially streams.
        (Ok(_), _) if prefers_streaming => CpuExecution::StreamedRows,
        // Any other plan that fits only serially, or a malformed plan, keeps
        // the worst-case claim and reports any error from execution.
        _ => CpuExecution::OffsetPlan {
            bytes: crate::batch::allocation::GENERIC_WORKER_CLAIM_BYTES,
        },
    }
}

/// Gray and RGB outputs of every sample type have a streamed route: the
/// region decode hands over finished rows of components where the image lets
/// it (see `j2k_native::Image::region_decode_streams_rows`), each sample
/// converted as the planned route converts it. RGBA outputs keep the bound:
/// their alpha needs a channel-definition box, which keeps the region decode
/// on whole component planes.
fn has_streamed_output(info: &BatchGroupInfo) -> bool {
    matches!(
        info.sample_type,
        NativeSampleType::U8 | NativeSampleType::U16 | NativeSampleType::I16
    ) && matches!(info.color, BatchColor::Gray | BatchColor::Rgb)
}

/// Planned offset-plan execution above which Gray8/RGB8 images stream
/// their rows (see [`choose_cpu_execution`]). Smaller images keep their
/// parse-free plan, whose working memory is negligible at that size.
pub(super) const STREAM_ABOVE_PLANNED_BYTES: usize = 4 << 20;

#[cfg(test)]
std::thread_local! {
    /// Test-only lower bound for [`choose_cpu_execution`] on this thread.
    pub(super) static PLANNED_DECODE_BOUND: core::cell::Cell<Option<usize>> =
        const { core::cell::Cell::new(None) };
    /// Test-only streaming threshold for [`choose_cpu_execution`] on this
    /// thread.
    pub(super) static STREAM_ABOVE_OVERRIDE: core::cell::Cell<Option<usize>> =
        const { core::cell::Cell::new(None) };
}

fn stream_above_planned_bytes() -> usize {
    #[cfg(test)]
    if let Some(bytes) = STREAM_ABOVE_OVERRIDE.with(core::cell::Cell::get) {
        return bytes;
    }
    STREAM_ABOVE_PLANNED_BYTES
}

/// The per-decode bound planned executions are held to. Native planning
/// already fails above the default, so this only lowers it in tests.
fn planned_decode_bound() -> usize {
    #[cfg(test)]
    if let Some(bound) = PLANNED_DECODE_BOUND.with(core::cell::Cell::get) {
        return bound;
    }
    j2k_native::DEFAULT_MAX_DECODE_BYTES
}

pub(super) fn batch_execution_shape(
    support: &J2kSupportInfo,
    plan: DeviceDecodePlan,
    preparation_depth: PreparationDepth,
) -> BatchExecutionShape {
    let source_rect = plan.source_rect();
    BatchExecutionShape {
        source_dimensions: plan.source_dims(),
        source_rect_dimensions: (source_rect.w, source_rect.h),
        scale: plan.scale(),
        tile_layout: support.info.tile_layout,
        resolution_levels: support.info.resolution_levels,
        preparation_depth,
    }
}

fn exceeds_tile_resolution_ladder(error: &j2k_native::DecodeError) -> bool {
    // This describes an invalid request, not a limitation of the optimized plan.
    matches!(
        error,
        j2k_native::DecodeError::Decoding(j2k_native::DecodingError::UnsupportedFeature(
            "tile coding style has fewer levels than the requested reduction"
        ))
    )
}

fn prepare_classic_offset_plan(
    image: &j2k_native::Image<'_>,
    info: &BatchGroupInfo,
    plan: DeviceDecodePlan,
    worker: &mut BatchWorker,
) -> Result<Option<PreparedClassicPlan>, BatchItemError> {
    if info.route != BatchCodecRoute::Classic
        || !matches!(
            info.color,
            BatchColor::Gray | BatchColor::Rgb | BatchColor::Rgba
        )
    {
        return Ok(None);
    }
    let output = plan.output_rect();
    match worker.build_prepared_classic_plan(image, (output.x, output.y, output.w, output.h)) {
        Ok(plan) => PreparedClassicPlan::from_native(plan)
            .map(Some)
            .map_err(|source| BatchItemError::Codec {
                stage: BatchErrorStage::Prepare,
                source: Arc::new(J2kError::from_native_decode_error(source)),
            }),
        Err(
            source @ j2k_native::DecodeError::Decoding(
                j2k_native::DecodingError::DirectPlanUnsupported(_)
                | j2k_native::DecodingError::UnsupportedFeature(_),
            ),
        ) if !exceeds_tile_resolution_ladder(&source) => Ok(None),
        Err(source) => Err(BatchItemError::Codec {
            stage: BatchErrorStage::Prepare,
            source: Arc::new(J2kError::from_native_decode_error(source)),
        }),
    }
}

fn prepare_htj2k_offset_plan(
    image: &j2k_native::Image<'_>,
    info: &BatchGroupInfo,
    plan: DeviceDecodePlan,
    worker: &mut BatchWorker,
) -> Result<Option<PreparedHtj2kPlan>, BatchItemError> {
    if info.route != BatchCodecRoute::Htj2k
        || !matches!(
            info.color,
            BatchColor::Gray | BatchColor::Rgb | BatchColor::Rgba
        )
    {
        return Ok(None);
    }
    let output = plan.output_rect();
    match worker.build_prepared_htj2k_plan(image, (output.x, output.y, output.w, output.h)) {
        Ok(plan) => PreparedHtj2kPlan::from_native(plan)
            .map(Some)
            .map_err(|source| BatchItemError::Codec {
                stage: BatchErrorStage::Prepare,
                source: Arc::new(J2kError::from_native_decode_error(source)),
            }),
        Err(source)
            if matches!(
                source.classify(),
                j2k_native::DecodeErrorClass::Unsupported { .. }
            ) && !exceeds_tile_resolution_ladder(&source) =>
        {
            Ok(None)
        }
        Err(source) => Err(BatchItemError::Codec {
            stage: BatchErrorStage::Prepare,
            source: Arc::new(J2kError::from_native_decode_error(source)),
        }),
    }
}

pub(super) fn batch_group_info(
    support: &J2kSupportInfo,
    plan: DeviceDecodePlan,
    options: BatchDecodeOptions,
) -> Result<BatchGroupInfo, BatchItemError> {
    let Some(first) = support.components.first().copied() else {
        return Err(nonrepresentable(
            NonRepresentableReason::UnsupportedComponentCount,
        ));
    };
    if support
        .components
        .iter()
        .any(|component| component.bit_depth != first.bit_depth)
    {
        return Err(nonrepresentable(NonRepresentableReason::MixedPrecision));
    }
    if support
        .components
        .iter()
        .any(|component| component.signed != first.signed)
    {
        return Err(nonrepresentable(NonRepresentableReason::MixedSignedness));
    }
    // When components use different sampling, decode replicates the
    // subsampled ones onto the reference grid, as single-image decode does.
    // Offset plans reject them, so they retain metadata only and never share
    // an execution group with unit-sampled images. When every component
    // shares one non-unit sampling, native decode instead produces the smaller
    // component grid, which the reference-grid geometry here cannot describe.
    if has_uniform_component_subsampling(support) {
        return Err(nonrepresentable(
            NonRepresentableReason::ComponentSubsampling,
        ));
    }
    if first.bit_depth > 16 {
        return Err(nonrepresentable(
            NonRepresentableReason::PrecisionAboveSixteen,
        ));
    }

    let (color, alpha) = representable_batch_color(support, options.settings)?;
    let sample_type = if first.signed {
        NativeSampleType::I16
    } else if first.bit_depth <= 8 {
        NativeSampleType::U8
    } else {
        NativeSampleType::U16
    };
    let (route, transform) = match support.transfer_syntax {
        CompressedTransferSyntax::Jpeg2000Lossless => (
            BatchCodecRoute::Classic,
            BatchWaveletTransform::Reversible53,
        ),
        CompressedTransferSyntax::Jpeg2000Lossy => (
            BatchCodecRoute::Classic,
            BatchWaveletTransform::Irreversible97,
        ),
        CompressedTransferSyntax::HtJpeg2000Lossless => {
            (BatchCodecRoute::Htj2k, BatchWaveletTransform::Reversible53)
        }
        CompressedTransferSyntax::HtJpeg2000Lossy => (
            BatchCodecRoute::Htj2k,
            BatchWaveletTransform::Irreversible97,
        ),
        _ => return Err(nonrepresentable(NonRepresentableReason::UnsupportedColor)),
    };

    Ok(BatchGroupInfo {
        dimensions: plan.output_dims(),
        color,
        alpha,
        precision: first.bit_depth,
        signed: first.signed,
        sample_type,
        layout: options.layout,
        colorspace: support.info.colorspace,
        route,
        transform,
        transfer_syntax: support.transfer_syntax,
        payload_kind: support.payload_kind,
    })
}

pub(super) fn reconcile_codec_plan_metadata(
    info: &mut BatchGroupInfo,
    codec_plan: &PreparedCodecPlan,
) -> Result<(), BatchItemError> {
    let (route, transform) = match codec_plan {
        PreparedCodecPlan::MetadataOnly => return Ok(()),
        PreparedCodecPlan::Classic(plan) => {
            (BatchCodecRoute::Classic, plan.uniform_wavelet_transform())
        }
        PreparedCodecPlan::Htj2k(plan) => {
            (BatchCodecRoute::Htj2k, plan.uniform_wavelet_transform())
        }
    };
    let transform =
        transform.ok_or_else(|| nonrepresentable(NonRepresentableReason::MixedWaveletTransform))?;
    let transform = match transform {
        j2k_native::J2kWaveletTransform::Reversible53 => BatchWaveletTransform::Reversible53,
        j2k_native::J2kWaveletTransform::Irreversible97 => BatchWaveletTransform::Irreversible97,
    };
    info.route = route;
    info.transform = transform;
    info.transfer_syntax = match (route, transform) {
        (BatchCodecRoute::Classic, BatchWaveletTransform::Reversible53) => {
            CompressedTransferSyntax::Jpeg2000Lossless
        }
        (BatchCodecRoute::Classic, BatchWaveletTransform::Irreversible97) => {
            CompressedTransferSyntax::Jpeg2000Lossy
        }
        (BatchCodecRoute::Htj2k, BatchWaveletTransform::Reversible53) => {
            CompressedTransferSyntax::HtJpeg2000Lossless
        }
        (BatchCodecRoute::Htj2k, BatchWaveletTransform::Irreversible97) => {
            CompressedTransferSyntax::HtJpeg2000Lossy
        }
    };
    Ok(())
}

fn representable_batch_color(
    support: &J2kSupportInfo,
    settings: crate::DecodeSettings,
) -> Result<(BatchColor, BatchAlpha), BatchItemError> {
    let file_metadata = support.file_metadata.as_ref();
    if file_metadata.is_some_and(|metadata| {
        metadata.has_palette
            || metadata.has_component_mapping
            || metadata.palette.is_some()
            || !metadata.component_mappings.is_empty()
            || metadata.has_icc_profile()
    }) || support.info.colorspace == Colorspace::IccTagged
    {
        return Err(nonrepresentable(NonRepresentableReason::UnsupportedColor));
    }

    match (support.component_count(), support.info.colorspace) {
        (1, Colorspace::Grayscale | Colorspace::SGray) => Ok((BatchColor::Gray, BatchAlpha::None)),
        (3, Colorspace::Rgb | Colorspace::SRgb | Colorspace::Rct | Colorspace::Ict)
            if file_metadata.is_none_or(|metadata| {
                metadata.channel_definitions.is_empty()
                    || has_identity_rgb_channel_definitions(&metadata.channel_definitions)
            }) =>
        {
            Ok((BatchColor::Rgb, BatchAlpha::None))
        }
        // A lenient decode takes a fourth component without channel
        // definitions as straight alpha, as `j2k_native` infers it.
        (4, Colorspace::Rgb | Colorspace::SRgb | Colorspace::Rct | Colorspace::Ict)
            if settings.lenient_tolerance_enabled()
                && file_metadata
                    .is_some_and(|metadata| metadata.channel_definitions.is_empty()) =>
        {
            Ok((BatchColor::Rgba, BatchAlpha::Straight))
        }
        (4, Colorspace::Rgb | Colorspace::SRgb | Colorspace::Rct | Colorspace::Ict) => {
            file_metadata
                .and_then(|metadata| identity_rgba_alpha(&metadata.channel_definitions))
                .map(|alpha| (BatchColor::Rgba, alpha))
                .ok_or_else(|| nonrepresentable(NonRepresentableReason::UnsupportedColor))
        }
        (1 | 3 | 4, _) => Err(nonrepresentable(NonRepresentableReason::UnsupportedColor)),
        _ => Err(nonrepresentable(
            NonRepresentableReason::UnsupportedComponentCount,
        )),
    }
}

fn has_identity_rgb_channel_definitions(definitions: &[crate::J2kChannelDefinition]) -> bool {
    definitions.len() == 3
        && definitions.iter().enumerate().all(|(index, definition)| {
            definition.channel_index == u16::try_from(index).unwrap_or(u16::MAX)
                && definition.channel_type == crate::J2kChannelType::Color
                && definition.association
                    == crate::J2kChannelAssociation::Color {
                        index: u16::try_from(index + 1).unwrap_or(u16::MAX),
                    }
        })
}

fn identity_rgba_alpha(definitions: &[crate::J2kChannelDefinition]) -> Option<BatchAlpha> {
    let (alpha, rgb) = definitions.split_last()?;
    let alpha_interpretation = match alpha.channel_type {
        crate::J2kChannelType::Opacity => BatchAlpha::Straight,
        crate::J2kChannelType::PremultipliedOpacity => BatchAlpha::Premultiplied,
        _ => return None,
    };
    (has_identity_rgb_channel_definitions(rgb)
        && alpha.channel_index == 3
        && alpha.association == crate::J2kChannelAssociation::WholeImage)
        .then_some(alpha_interpretation)
}

fn has_uniform_component_subsampling(support: &J2kSupportInfo) -> bool {
    let Some(first) = support.components.first() else {
        return false;
    };
    let sampling = (first.x_rsiz, first.y_rsiz);
    sampling != (1, 1)
        && support
            .components
            .iter()
            .all(|component| (component.x_rsiz, component.y_rsiz) == sampling)
}

fn nonrepresentable(reason: NonRepresentableReason) -> BatchItemError {
    BatchItemError::NonRepresentableBatchOutput { reason }
}
