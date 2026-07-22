// SPDX-License-Identifier: MIT OR Apache-2.0

//! One-shot classic and HTJ2K color-plan construction.

use super::{
    flatten_cuda_color_components, native_decode_error, profile, rgba_bit_depths_from_rgb,
    CudaHtj2kColorDecodePlans, CudaHtj2kDecodePlan, CudaHtj2kDecodeProfileDetail,
    CudaHtj2kProfileReport, CudaHtj2kTransform, DecodeSettings, DeviceDecodePlan, Downscale, Error,
    NativeDecoderContext, NativeImage, PixelFormat, Rect,
};

#[cfg(feature = "cuda-runtime")]
pub(in crate::decoder) fn build_cuda_color_plan_from_bytes_for_device_plan_with_profile<'a>(
    input: &'a [u8],
    fmt: PixelFormat,
    device_plan: DeviceDecodePlan,
    settings: DecodeSettings,
    native_context: &mut NativeDecoderContext<'a>,
) -> Result<CudaHtj2kColorDecodePlans, Error> {
    let total_start = profile::profile_now(true);
    let parse_start = profile::profile_now(true);
    let target_resolution = (device_plan.scale() != Downscale::None).then_some((
        device_plan
            .source_dims()
            .0
            .div_ceil(device_plan.scale().denominator()),
        device_plan
            .source_dims()
            .1
            .div_ceil(device_plan.scale().denominator()),
    ));
    let image = NativeImage::new(
        input,
        &DecodeSettings {
            target_resolution,
            ..settings
        },
    )
    .map_err(native_decode_error)?;
    let parse_us = profile::elapsed_us(parse_start);

    let plan_start = profile::profile_now(true);
    let native_plan = image
        .build_direct_color_plan_with_context(native_context)
        .map_err(native_decode_error)?;
    let plan_us = profile::elapsed_us(plan_start);
    let full = Rect::full(device_plan.source_dims());
    let selected_rect = if device_plan.source_rect() == full {
        None
    } else if device_plan.scale() == Downscale::None {
        Some(device_plan.source_rect())
    } else {
        Some(device_plan.output_rect())
    };

    let flatten_start = profile::profile_now(true);
    let (payload, components) = flatten_cuda_color_components(
        &native_plan,
        fmt,
        selected_rect.map(|rect| ((rect.x, rect.y), (rect.w, rect.h))),
        "j2k CUDA exact classic color plan owners",
    )?;
    let flatten_us = profile::elapsed_us(flatten_start);
    let block_count = components
        .iter()
        .map(CudaHtj2kDecodePlan::block_count)
        .sum::<usize>();
    let classic_block_count = components
        .iter()
        .map(|plan| plan.classic_code_blocks().len())
        .sum::<usize>();
    let ht_block_count = components
        .iter()
        .map(|plan| plan.code_blocks().len())
        .sum::<usize>();
    let payload_bytes = payload.len();
    let report = CudaHtj2kProfileReport {
        parse_us,
        plan_us,
        flatten_us,
        total_us: profile::elapsed_us(total_start),
        block_count,
        classic_block_count,
        ht_block_count,
        payload_bytes,
        dispatch_count: 0,
        residency: crate::SurfaceResidency::CudaResidentDecode,
        detail: CudaHtj2kDecodeProfileDetail::default(),
        ..CudaHtj2kProfileReport::default()
    };
    report.emit("classic_prepared_plan");
    Ok(CudaHtj2kColorDecodePlans {
        output_index: 0,
        dimensions: device_plan.output_dims(),
        mct_dimensions: native_plan.dimensions,
        bit_depths: rgba_bit_depths_from_rgb(native_plan.bit_depths),
        mct: native_plan.mct,
        transform: CudaHtj2kTransform::from_native(native_plan.transform),
        payload,
        components,
        report,
    })
}

#[cfg(feature = "cuda-runtime")]
pub(in crate::decoder) fn build_cuda_htj2k_color_plans_from_bytes_with_profile<'a>(
    input: &'a [u8],
    fmt: PixelFormat,
    native_context: &mut NativeDecoderContext<'a>,
) -> Result<CudaHtj2kColorDecodePlans, Error> {
    let total_start = profile::profile_now(true);

    let parse_start = profile::profile_now(true);
    let image = NativeImage::new(input, &DecodeSettings::default()).map_err(native_decode_error)?;
    let parse_us = profile::elapsed_us(parse_start);

    let plan_start = profile::profile_now(true);
    let native_plan = image
        .build_direct_color_plan_with_context(native_context)
        .map_err(native_decode_error)?;
    let plan_us = profile::elapsed_us(plan_start);

    let flatten_start = profile::profile_now(true);
    let (payload, components) =
        flatten_cuda_color_components(&native_plan, fmt, None, "j2k CUDA color decode plans")?;
    let flatten_us = profile::elapsed_us(flatten_start);
    let block_count = components
        .iter()
        .map(CudaHtj2kDecodePlan::block_count)
        .sum::<usize>();
    let classic_block_count = components
        .iter()
        .map(|plan| plan.classic_code_blocks().len())
        .sum::<usize>();
    let ht_block_count = components
        .iter()
        .map(|plan| plan.code_blocks().len())
        .sum::<usize>();
    let payload_bytes = payload.len();
    let report = CudaHtj2kProfileReport {
        parse_us,
        plan_us,
        flatten_us,
        total_us: profile::elapsed_us(total_start),
        block_count,
        classic_block_count,
        ht_block_count,
        payload_bytes,
        dispatch_count: 0,
        residency: crate::SurfaceResidency::CudaResidentDecode,
        detail: CudaHtj2kDecodeProfileDetail::default(),
        ..CudaHtj2kProfileReport::default()
    };
    report.emit("plan");

    Ok(CudaHtj2kColorDecodePlans {
        output_index: 0,
        dimensions: native_plan.dimensions,
        mct_dimensions: native_plan.dimensions,
        bit_depths: rgba_bit_depths_from_rgb(native_plan.bit_depths),
        mct: native_plan.mct,
        transform: CudaHtj2kTransform::from_native(native_plan.transform),
        payload,
        components,
        report,
    })
}
