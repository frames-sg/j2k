// SPDX-License-Identifier: Apache-2.0

use j2k_core::{PixelFormat, Rect};
use j2k_native::{
    ColorSpace as NativeColorSpace, DecodeSettings as NativeDecodeSettings,
    DecodedComponents as NativeDecodedComponents, DecoderContext as NativeDecoderContext,
    Image as NativeImage,
};
use metal::Device;

use super::{
    with_runtime, with_runtime_for_device, MetalCodeBlockDecoder, MetalRuntime, PlaneStage,
};
use crate::{Error, Surface};

#[cfg(target_os = "macos")]
pub(crate) fn decode_image_to_surface<'a>(
    image: &NativeImage<'a>,
    context: &mut NativeDecoderContext<'a>,
    fmt: PixelFormat,
) -> Result<Surface, Error> {
    with_runtime(|runtime| {
        let mut code_block_decoder = MetalCodeBlockDecoder::default();
        let decoded = image
            .decode_components_with_ht_decoder(context, &mut code_block_decoder)
            .map_err(|error| Error::Decode(j2k::J2kError::Backend(error.to_string())))?;
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
            .decode_region_components_with_ht_decoder(
                context,
                (roi.x, roi.y, roi.w, roi.h),
                &mut code_block_decoder,
            )
            .map_err(|error| Error::Decode(j2k::J2kError::Backend(error.to_string())))?;
        let stage = select_plane_stage(runtime, image, &decoded, &mut code_block_decoder)?;
        stage.finish_with_runtime(runtime, fmt)
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn decode_image_region_to_surface_with_device<'a>(
    image: &NativeImage<'a>,
    context: &mut NativeDecoderContext<'a>,
    fmt: PixelFormat,
    roi: Rect,
    device: &Device,
) -> Result<Surface, Error> {
    with_runtime_for_device(device, |_| {
        decode_image_region_to_surface(image, context, fmt, roi)
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

#[cfg(target_os = "macos")]
pub(crate) fn decode_scaled_to_surface(
    bytes: &[u8],
    dims: (u32, u32),
    fmt: PixelFormat,
    scale: j2k_core::Downscale,
) -> Result<Surface, Error> {
    let target_dims = (
        dims.0.div_ceil(scale.denominator()),
        dims.1.div_ceil(scale.denominator()),
    );
    let settings = NativeDecodeSettings {
        target_resolution: Some(target_dims),
        ..NativeDecodeSettings::default()
    };
    let image = NativeImage::new(bytes, &settings)
        .map_err(|error| Error::Decode(j2k::J2kError::Backend(error.to_string())))?;
    let mut context = NativeDecoderContext::default();
    decode_image_to_surface(&image, &mut context, fmt)
}

#[cfg(target_os = "macos")]
pub(crate) fn decode_region_scaled_to_surface(
    bytes: &[u8],
    dims: (u32, u32),
    fmt: PixelFormat,
    roi: j2k_core::Rect,
    scale: j2k_core::Downscale,
) -> Result<Surface, Error> {
    let target_dims = (
        dims.0.div_ceil(scale.denominator()),
        dims.1.div_ceil(scale.denominator()),
    );
    let settings = NativeDecodeSettings {
        target_resolution: Some(target_dims),
        ..NativeDecodeSettings::default()
    };
    let image = NativeImage::new(bytes, &settings)
        .map_err(|error| Error::Decode(j2k::J2kError::Backend(error.to_string())))?;
    let mut context = NativeDecoderContext::default();
    decode_image_region_to_surface(&image, &mut context, fmt, roi.scaled_covering(scale))
}

#[cfg(target_os = "macos")]
pub(crate) fn decode_scaled_to_surface_with_device(
    bytes: &[u8],
    dims: (u32, u32),
    fmt: PixelFormat,
    scale: j2k_core::Downscale,
    device: &Device,
) -> Result<Surface, Error> {
    with_runtime_for_device(device, |_| {
        decode_scaled_to_surface(bytes, dims, fmt, scale)
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn decode_region_scaled_to_surface_with_device(
    bytes: &[u8],
    dims: (u32, u32),
    fmt: PixelFormat,
    roi: j2k_core::Rect,
    scale: j2k_core::Downscale,
    device: &Device,
) -> Result<Surface, Error> {
    with_runtime_for_device(device, |_| {
        decode_region_scaled_to_surface(bytes, dims, fmt, roi, scale)
    })
}
