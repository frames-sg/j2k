// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::metal_types::Device;
#[cfg(target_os = "macos")]
use j2k_core::{PixelFormat, Rect};
use j2k_native::{
    ColorSpace as NativeColorSpace, DecodeSettings as NativeDecodeSettings,
    DecodedComponents as NativeDecodedComponents, DecoderContext as NativeDecoderContext,
    Image as NativeImage,
};

use super::{
    with_runtime, with_runtime_for_device, MetalCodeBlockDecoder, MetalRuntime, PlaneStage,
};
use crate::error::native_decode_error;
use crate::{Error, Surface};

#[cfg(target_os = "macos")]
pub(crate) fn decode_image_to_surface<'a>(
    image: &NativeImage<'a>,
    context: &mut NativeDecoderContext<'a>,
    fmt: PixelFormat,
) -> Result<Surface, Error> {
    with_runtime(|runtime| {
        let mut code_block_decoder = MetalCodeBlockDecoder::default();
        // The planes are packed to integer surfaces, so irreversible samples
        // must be rounded before the level shift, as the CPU decode does.
        let decoded = image
            .decode_components_for_integer_output_with_ht_decoder(context, &mut code_block_decoder)
            .map_err(native_decode_error)?;
        let stage = select_plane_stage(runtime, image, &decoded, &mut code_block_decoder)?;
        stage.finish_with_runtime(runtime, fmt)
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn decode_image_to_surface_with_device<'a>(
    image: &NativeImage<'a>,
    context: &mut NativeDecoderContext<'a>,
    fmt: PixelFormat,
    device: &Device,
) -> Result<Surface, Error> {
    with_runtime_for_device(device, |_| decode_image_to_surface(image, context, fmt))
}

#[cfg(target_os = "macos")]
pub(crate) fn decode_image_region_to_surface<'a>(
    image: &NativeImage<'a>,
    context: &mut NativeDecoderContext<'a>,
    fmt: PixelFormat,
    roi: Rect,
) -> Result<Surface, Error> {
    with_runtime(|runtime| {
        let mut code_block_decoder = MetalCodeBlockDecoder::default();
        let decoded = image
            .decode_region_components_for_integer_output_with_ht_decoder(
                context,
                (roi.x, roi.y, roi.w, roi.h),
                &mut code_block_decoder,
            )
            .map_err(native_decode_error)?;
        let stage = select_plane_stage(runtime, image, &decoded, &mut code_block_decoder)?;
        stage.finish_with_runtime(runtime, fmt)
    })
}

#[cfg(target_os = "macos")]
fn select_plane_stage(
    runtime: &MetalRuntime,
    image: &NativeImage<'_>,
    decoded: &NativeDecodedComponents<'_>,
    code_block_decoder: &mut MetalCodeBlockDecoder,
) -> Result<PlaneStage, Error> {
    if image.supports_direct_device_plane_reuse() {
        if matches!(decoded.color_space(), NativeColorSpace::RGB)
            && !decoded.has_alpha()
            && decoded.planes().len() == 3
        {
            if let Some(stage) = PlaneStage::from_captured_planes(
                decoded,
                code_block_decoder.mct.take_captured_planes(),
            ) {
                return Ok(stage);
            }
        }
        if matches!(decoded.color_space(), NativeColorSpace::Gray)
            && !decoded.has_alpha()
            && decoded.planes().len() == 1
        {
            if let Some(stage) = PlaneStage::from_captured_planes(
                decoded,
                code_block_decoder.store.take_captured_planes(),
            ) {
                return Ok(stage);
            }
        }
    }

    PlaneStage::from_planes(&runtime.device, decoded, None)
}

/// The native image of `bytes` reduced by exactly `scale`'s levels of its
/// resolution ladder, and the reduced region covering `roi` on the reference
/// grid, as the CPU decoder decodes and reports it.
///
/// A reduction spans `[ceil(origin / s), ceil((origin + size) / s))` on each
/// axis, so an image whose origin is not a multiple of `s` reduces to fewer
/// samples than `ceil(size / s)` and its regions shift against
/// `roi.scaled_covering(scale)`.
pub(crate) fn reduced_image_and_region(
    bytes: &[u8],
    roi: j2k_core::Rect,
    scale: j2k_core::Downscale,
) -> Result<(NativeImage<'_>, j2k_core::Rect), Error> {
    let grid = j2k::J2kDecoder::inspect_reference_grid(bytes)?;
    let request = if scale == j2k_core::Downscale::None {
        j2k::DeviceDecodeRequest::Region { roi }
    } else {
        j2k::DeviceDecodeRequest::RegionScaled { roi, scale }
    };
    let region = j2k::DeviceDecodePlan::for_reference_grid(grid, request)?.output_rect();
    // A level count past the ladder fails the reduced image below.
    let levels = u8::try_from(scale.denominator().trailing_zeros()).unwrap_or(u8::MAX);
    let image = NativeImage::new_with_reduction(bytes, &NativeDecodeSettings::default(), levels)
        .map_err(native_decode_error)?;
    Ok((image, region))
}

#[cfg(target_os = "macos")]
pub(crate) fn decode_region_scaled_to_surface(
    bytes: &[u8],
    fmt: PixelFormat,
    roi: j2k_core::Rect,
    scale: j2k_core::Downscale,
) -> Result<Surface, Error> {
    let (image, region) = reduced_image_and_region(bytes, roi, scale)?;
    let mut context = NativeDecoderContext::default();
    decode_image_region_to_surface(&image, &mut context, fmt, region)
}

#[cfg(target_os = "macos")]
pub(crate) fn decode_region_scaled_to_surface_with_device(
    bytes: &[u8],
    fmt: PixelFormat,
    roi: j2k_core::Rect,
    scale: j2k_core::Downscale,
    device: &Device,
) -> Result<Surface, Error> {
    with_runtime_for_device(device, |_| {
        decode_region_scaled_to_surface(bytes, fmt, roi, scale)
    })
}
