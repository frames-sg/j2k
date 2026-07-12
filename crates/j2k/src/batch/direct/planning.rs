// SPDX-License-Identifier: MIT OR Apache-2.0

//! Direct HTJ2K region-plan construction and capability filtering.

use j2k_core::{CompressedTransferSyntax, Downscale, Rect};
use j2k_native::{
    inspect_j2k_codestream_header, DecodeError as NativeDecodeError,
    DecodingError as NativeDecodingError, J2kDirectColorPlan, J2kRect,
};

use crate::backend::{self, DecodeSettings};
use crate::decode::validate_region;
use crate::parse::parse_image_info;
use crate::J2kError;

pub(super) fn build_direct_color_region_plan(
    input: &[u8],
    roi: Rect,
    scale: Downscale,
) -> Result<Option<(J2kDirectColorPlan, J2kRect)>, J2kError> {
    if !input_declares_htj2k(input) {
        return Ok(None);
    }
    let Ok(parsed) = parse_image_info(input) else {
        return Ok(None);
    };
    if !matches!(
        parsed.transfer_syntax,
        CompressedTransferSyntax::HtJpeg2000Lossless | CompressedTransferSyntax::HtJpeg2000Lossy
    ) {
        return Ok(None);
    }

    validate_region(roi, parsed.info.dimensions)?;
    let target_dims = (
        parsed.info.dimensions.0.div_ceil(scale.denominator()),
        parsed.info.dimensions.1.div_ceil(scale.denominator()),
    );
    let output_region = roi.scaled_covering(scale);
    let image = backend::image(
        input,
        DecodeSettings {
            target_resolution: Some(target_dims),
            ..DecodeSettings::default()
        },
    )?;
    validate_region(output_region, (image.width(), image.height()))?;

    let mut native_context = j2k_native::DecoderContext::default();
    match image.build_direct_color_plan_region_with_context(
        &mut native_context,
        (
            output_region.x,
            output_region.y,
            output_region.w,
            output_region.h,
        ),
    ) {
        Ok(plan) if direct_color_plan_uses_only_htj2k(&plan) => Ok(Some((
            plan,
            J2kRect {
                x0: output_region.x,
                y0: output_region.y,
                x1: output_region.x + output_region.w,
                y1: output_region.y + output_region.h,
            },
        ))),
        Ok(_) => Ok(None),
        Err(error) if is_unsupported_direct_color_plan_error(error) => Ok(None),
        Err(error) => Err(J2kError::from_native_decode_error(error)),
    }
}

pub(super) fn input_declares_htj2k(input: &[u8]) -> bool {
    crate::extract_j2k_codestream_payload(input)
        .ok()
        .and_then(|payload| inspect_j2k_codestream_header(payload.codestream()).ok())
        .is_some_and(|metadata| metadata.high_throughput)
}

fn direct_color_plan_uses_only_htj2k(plan: &J2kDirectColorPlan) -> bool {
    plan.component_plans.iter().all(|component| {
        component.steps.iter().any(|step| {
            matches!(
                step,
                j2k_native::J2kDirectGrayscaleStep::HtSubBand(sub_band)
                    if !sub_band.jobs.is_empty()
            )
        }) && component
            .steps
            .iter()
            .all(|step| !matches!(step, j2k_native::J2kDirectGrayscaleStep::ClassicSubBand(_)))
    })
}

fn is_unsupported_direct_color_plan_error(error: NativeDecodeError) -> bool {
    matches!(
        error,
        NativeDecodeError::Decoding(NativeDecodingError::UnsupportedFeature(_))
    )
}
