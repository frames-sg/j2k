// SPDX-License-Identifier: MIT OR Apache-2.0

//! Per-image metadata validation and retained native plan construction.

use super::super::{
    extract_j2k_codestream_payload, parse_image_info, Arc, BatchAlpha, BatchCodecRoute, BatchColor,
    BatchDecodeOptions, BatchErrorStage, BatchExecutionShape, BatchGroupInfo, BatchItemError,
    BatchLayout, BatchWaveletTransform, BatchWorker, Colorspace, CompressedTransferSyntax,
    DecodeSettings, DeviceDecodePlan, Downscale, EncodedImage, J2kCodestreamRange, J2kError,
    J2kSupportInfo, NativeSampleType, NonRepresentableReason, PreparationDepth, PrepareImageResult,
    PreparedClassicPlan, PreparedCodecPlan, PreparedHtj2kPlan, PreparedImage, PreparedImageInner,
};

pub(super) fn prepare_image(
    input: EncodedImage,
    source_index: usize,
    options: BatchDecodeOptions,
    worker: &mut BatchWorker,
) -> PrepareImageResult {
    let parsed = parse_image_info(&input.bytes).map_err(|source| BatchItemError::Codec {
        stage: BatchErrorStage::Prepare,
        source: Arc::new(source),
    })?;
    let support = parsed.into_support_info();
    let plan = DeviceDecodePlan::for_image(support.info.dimensions, input.request.device_request())
        .map_err(|source| BatchItemError::Codec {
            stage: BatchErrorStage::Prepare,
            source: Arc::new(source),
        })?;
    let mut info = batch_group_info(&support, plan, options.layout)?;
    let codestream =
        extract_j2k_codestream_payload(&input.bytes).map_err(|source| BatchItemError::Codec {
            stage: BatchErrorStage::Prepare,
            source: Arc::new(source),
        })?;
    let codestream_range = J2kCodestreamRange {
        offset: codestream.codestream_offset(),
        length: codestream.codestream().len(),
    };
    let codec_plan = match info.route {
        BatchCodecRoute::Classic => {
            prepare_classic_offset_plan(&input, &info, plan, options.settings, worker)?
                .map_or(PreparedCodecPlan::MetadataOnly, PreparedCodecPlan::Classic)
        }
        BatchCodecRoute::Htj2k => {
            prepare_htj2k_offset_plan(&input, &info, plan, options.settings, worker)?
                .map_or(PreparedCodecPlan::MetadataOnly, PreparedCodecPlan::Htj2k)
        }
    };
    reconcile_codec_plan_metadata(&mut info, &codec_plan)?;
    let preparation_depth = codec_plan.preparation_depth();
    let execution_shape = batch_execution_shape(&support, plan, preparation_depth);
    let image = PreparedImage {
        inner: Arc::new(PreparedImageInner {
            bytes: input.bytes,
            request: input.request,
            source_index,
            decode_settings: options.settings,
            support: Arc::new(support),
            plan,
            codestream_range,
            codec_plan,
        }),
    };
    Ok((image, info, execution_shape))
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

fn prepare_classic_offset_plan(
    input: &EncodedImage,
    info: &BatchGroupInfo,
    plan: DeviceDecodePlan,
    decode_settings: DecodeSettings,
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
    let target_resolution = (plan.scale() != Downscale::None).then_some((
        plan.source_dims().0.div_ceil(plan.scale().denominator()),
        plan.source_dims().1.div_ceil(plan.scale().denominator()),
    ));
    let native_settings = decode_settings.to_native(target_resolution);
    let image = j2k_native::Image::new(&input.bytes, &native_settings).map_err(|source| {
        BatchItemError::Codec {
            stage: BatchErrorStage::Prepare,
            source: Arc::new(J2kError::from_native_decode_error(source)),
        }
    })?;
    let output = plan.output_rect();
    match worker.build_prepared_classic_plan(&image, (output.x, output.y, output.w, output.h)) {
        Ok(plan) => Ok(Some(PreparedClassicPlan::from_native(plan))),
        Err(j2k_native::DecodeError::Decoding(
            j2k_native::DecodingError::DirectPlanUnsupported(_)
            | j2k_native::DecodingError::UnsupportedFeature(_),
        )) => Ok(None),
        Err(source) => Err(BatchItemError::Codec {
            stage: BatchErrorStage::Prepare,
            source: Arc::new(J2kError::from_native_decode_error(source)),
        }),
    }
}

fn prepare_htj2k_offset_plan(
    input: &EncodedImage,
    info: &BatchGroupInfo,
    plan: DeviceDecodePlan,
    decode_settings: DecodeSettings,
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
    let target_resolution = (plan.scale() != Downscale::None).then_some((
        plan.source_dims().0.div_ceil(plan.scale().denominator()),
        plan.source_dims().1.div_ceil(plan.scale().denominator()),
    ));
    let native_settings = decode_settings.to_native(target_resolution);
    let image = j2k_native::Image::new(&input.bytes, &native_settings).map_err(|source| {
        BatchItemError::Codec {
            stage: BatchErrorStage::Prepare,
            source: Arc::new(J2kError::from_native_decode_error(source)),
        }
    })?;
    let output = plan.output_rect();
    match worker.build_prepared_htj2k_plan(&image, (output.x, output.y, output.w, output.h)) {
        Ok(plan) => Ok(Some(PreparedHtj2kPlan::from_native(plan))),
        Err(j2k_native::DecodeError::Decoding(
            j2k_native::DecodingError::DirectPlanUnsupported(_),
        )) => Ok(None),
        Err(source) => Err(BatchItemError::Codec {
            stage: BatchErrorStage::Prepare,
            source: Arc::new(J2kError::from_native_decode_error(source)),
        }),
    }
}

pub(super) fn batch_group_info(
    support: &J2kSupportInfo,
    plan: DeviceDecodePlan,
    layout: BatchLayout,
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
    if support.has_component_subsampling() {
        return Err(nonrepresentable(
            NonRepresentableReason::ComponentSubsampling,
        ));
    }
    if first.bit_depth > 16 {
        return Err(nonrepresentable(
            NonRepresentableReason::PrecisionAboveSixteen,
        ));
    }

    let (color, alpha) = representable_batch_color(support)?;
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
        layout,
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

fn nonrepresentable(reason: NonRepresentableReason) -> BatchItemError {
    BatchItemError::NonRepresentableBatchOutput { reason }
}
