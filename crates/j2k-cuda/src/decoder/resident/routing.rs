// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(all(test, feature = "cuda-runtime"))]
use super::super::color_batch::CUDA_HTJ2K_BATCH_DECODE_CALLS;
#[cfg(feature = "cuda-runtime")]
use super::super::color_batch::{
    decode_color_cuda_resident_batch_surfaces_with_profile,
    decode_color_cuda_resident_region_scaled_surface, decode_color_cuda_resident_region_surface,
    decode_color_cuda_resident_scaled_surface, decode_color_cuda_resident_surface_with_profile,
};
#[cfg(feature = "cuda-runtime")]
use super::super::grayscale_batch::decode_grayscale_cuda_resident_batch_surfaces_with_profile;
use super::super::{
    profile, CudaHtj2kProfileReport, CudaSession, DeviceDecodePlan, DeviceDecodeRequest, Downscale,
    Error, J2kDecoder, PixelFormat, Rect, Surface, SurfaceResidency,
    CUDA_HTJ2K_OUTPUT_FORMAT_UNSUPPORTED,
};
#[cfg(feature = "cuda-runtime")]
use super::surface::decode_grayscale_cuda_resident_surface_with_plan_profile;

#[cfg(feature = "cuda-runtime")]
pub(in crate::decoder) fn decode_to_cuda_resident_surface_impl(
    decoder: &mut J2kDecoder<'_>,
    session: &mut CudaSession,
    fmt: PixelFormat,
) -> Result<Surface, Error> {
    decode_to_cuda_resident_surface_with_profile_control(decoder, session, fmt, false)
        .map(|(surface, _report)| surface)
}

#[cfg(feature = "cuda-runtime")]
pub(in crate::decoder) fn decode_to_cuda_resident_surface_with_profile_impl(
    decoder: &mut J2kDecoder<'_>,
    session: &mut CudaSession,
    fmt: PixelFormat,
) -> Result<(Surface, CudaHtj2kProfileReport), Error> {
    decode_to_cuda_resident_surface_with_profile_control(decoder, session, fmt, true)
}

#[cfg(feature = "cuda-runtime")]
fn decode_to_cuda_resident_surface_with_profile_control(
    decoder: &mut J2kDecoder<'_>,
    session: &mut CudaSession,
    fmt: PixelFormat,
    collect_stage_timings: bool,
) -> Result<(Surface, CudaHtj2kProfileReport), Error> {
    let collect_stage_timings = collect_stage_timings || profile::profile_stages_enabled();
    let wall_started = profile::profile_now(collect_stage_timings);
    match fmt {
        PixelFormat::Gray8 | PixelFormat::Gray16 => {
            decode_grayscale_cuda_resident_surface_with_profile(
                decoder,
                session,
                fmt,
                wall_started,
                collect_stage_timings,
            )
        }
        PixelFormat::Rgb8 | PixelFormat::Rgba8 | PixelFormat::Rgb16 | PixelFormat::Rgba16 => {
            decode_color_cuda_resident_surface_with_profile(
                decoder,
                session,
                fmt,
                wall_started,
                collect_stage_timings,
            )
        }
        _ => Err(Error::UnsupportedCudaRequest {
            reason: CUDA_HTJ2K_OUTPUT_FORMAT_UNSUPPORTED,
        }),
    }
}

#[cfg(feature = "cuda-runtime")]
pub(in crate::decoder) fn decode_batch_to_cuda_resident_surface_with_profile_control(
    inputs: &[&[u8]],
    session: &mut CudaSession,
    fmt: PixelFormat,
    collect_stage_timings: bool,
) -> Result<(Vec<Surface>, CudaHtj2kProfileReport), Error> {
    #[cfg(all(test, feature = "cuda-runtime"))]
    CUDA_HTJ2K_BATCH_DECODE_CALLS.with(|calls| calls.set(calls.get().saturating_add(1)));

    let collect_stage_timings = collect_stage_timings || profile::profile_stages_enabled();
    if inputs.is_empty() {
        return Ok((
            Vec::new(),
            CudaHtj2kProfileReport {
                residency: SurfaceResidency::CudaResidentDecode,
                ..CudaHtj2kProfileReport::default()
            },
        ));
    }
    match fmt {
        PixelFormat::Gray8 | PixelFormat::Gray16 | PixelFormat::GrayI16 => {
            decode_grayscale_cuda_resident_batch_surfaces_with_profile(
                inputs,
                session,
                fmt,
                collect_stage_timings,
            )
        }
        PixelFormat::Rgb8 | PixelFormat::Rgba8 | PixelFormat::Rgb16 | PixelFormat::Rgba16 => {
            decode_color_cuda_resident_batch_surfaces_with_profile(
                inputs,
                session,
                fmt,
                collect_stage_timings,
            )
        }
        _ => Err(Error::UnsupportedCudaRequest {
            reason: CUDA_HTJ2K_OUTPUT_FORMAT_UNSUPPORTED,
        }),
    }
}

#[cfg(feature = "cuda-runtime")]
pub(in crate::decoder) fn decode_region_to_cuda_resident_surface_impl(
    decoder: &mut J2kDecoder<'_>,
    session: &mut CudaSession,
    fmt: PixelFormat,
    roi: Rect,
) -> Result<Surface, Error> {
    let plan = DeviceDecodePlan::for_image(
        decoder.inner.info().dimensions,
        DeviceDecodeRequest::Region { roi },
    )?;
    if plan.is_full_frame() {
        return decode_to_cuda_resident_surface_impl(decoder, session, fmt);
    }

    match fmt {
        PixelFormat::Gray8 | PixelFormat::Gray16 => {
            decode_grayscale_cuda_resident_region_surface(decoder, session, fmt, plan.source_rect())
        }
        PixelFormat::Rgb8 | PixelFormat::Rgba8 | PixelFormat::Rgb16 | PixelFormat::Rgba16 => {
            decode_color_cuda_resident_region_surface(decoder, session, fmt, plan.source_rect())
        }
        _ => Err(Error::UnsupportedCudaRequest {
            reason: CUDA_HTJ2K_OUTPUT_FORMAT_UNSUPPORTED,
        }),
    }
}

#[cfg(feature = "cuda-runtime")]
pub(in crate::decoder) fn decode_scaled_to_cuda_resident_surface_impl(
    decoder: &mut J2kDecoder<'_>,
    session: &mut CudaSession,
    fmt: PixelFormat,
    scale: Downscale,
) -> Result<Surface, Error> {
    if scale == Downscale::None {
        return decode_to_cuda_resident_surface_impl(decoder, session, fmt);
    }
    let output_dimensions = DeviceDecodePlan::for_image(
        decoder.inner.info().dimensions,
        DeviceDecodeRequest::Scaled { scale },
    )?
    .output_dims();

    match fmt {
        PixelFormat::Gray8 | PixelFormat::Gray16 => {
            decode_grayscale_cuda_resident_scaled_surface(decoder, session, fmt, output_dimensions)
        }
        PixelFormat::Rgb8 | PixelFormat::Rgba8 | PixelFormat::Rgb16 | PixelFormat::Rgba16 => {
            decode_color_cuda_resident_scaled_surface(decoder, session, fmt, output_dimensions)
        }
        _ => Err(Error::UnsupportedCudaRequest {
            reason: CUDA_HTJ2K_OUTPUT_FORMAT_UNSUPPORTED,
        }),
    }
}

#[cfg(feature = "cuda-runtime")]
pub(in crate::decoder) fn decode_region_scaled_to_cuda_resident_surface_impl(
    decoder: &mut J2kDecoder<'_>,
    session: &mut CudaSession,
    fmt: PixelFormat,
    roi: Rect,
    scale: Downscale,
) -> Result<Surface, Error> {
    if scale == Downscale::None {
        return decode_region_to_cuda_resident_surface_impl(decoder, session, fmt, roi);
    }
    let source_dimensions = decoder.inner.info().dimensions;
    let scaled_dimensions =
        DeviceDecodePlan::for_image(source_dimensions, DeviceDecodeRequest::Scaled { scale })?
            .output_dims();
    let plan = DeviceDecodePlan::for_image(
        source_dimensions,
        DeviceDecodeRequest::RegionScaled { roi, scale },
    )?;
    let scaled_roi = plan.output_rect();

    match fmt {
        PixelFormat::Gray8 | PixelFormat::Gray16 => {
            decode_grayscale_cuda_resident_region_scaled_surface(
                decoder,
                session,
                fmt,
                scaled_roi,
                scaled_dimensions,
            )
        }
        PixelFormat::Rgb8 | PixelFormat::Rgba8 | PixelFormat::Rgb16 | PixelFormat::Rgba16 => {
            decode_color_cuda_resident_region_scaled_surface(
                decoder,
                session,
                fmt,
                scaled_roi,
                scaled_dimensions,
            )
        }
        _ => Err(Error::UnsupportedCudaRequest {
            reason: CUDA_HTJ2K_OUTPUT_FORMAT_UNSUPPORTED,
        }),
    }
}

#[cfg(feature = "cuda-runtime")]
fn decode_grayscale_cuda_resident_surface_with_profile(
    decoder: &mut J2kDecoder<'_>,
    session: &mut CudaSession,
    fmt: PixelFormat,
    wall_started: Option<profile::ProfileInstant>,
    collect_stage_timings: bool,
) -> Result<(Surface, CudaHtj2kProfileReport), Error> {
    let (plan, mut report) = decoder.build_cuda_htj2k_grayscale_plan_with_profile(fmt)?;
    decode_grayscale_cuda_resident_surface_with_plan_profile(
        session,
        fmt,
        &plan,
        &mut report,
        wall_started,
        collect_stage_timings,
    )
}

#[cfg(feature = "cuda-runtime")]
fn decode_grayscale_cuda_resident_region_surface(
    decoder: &mut J2kDecoder<'_>,
    session: &mut CudaSession,
    fmt: PixelFormat,
    roi: Rect,
) -> Result<Surface, Error> {
    let collect_stage_timings = profile::profile_stages_enabled();
    let wall_started = profile::profile_now(collect_stage_timings);
    let (plan, mut report) =
        decoder.build_cuda_htj2k_grayscale_region_plan_with_profile(fmt, roi)?;
    decode_grayscale_cuda_resident_surface_with_plan_profile(
        session,
        fmt,
        &plan,
        &mut report,
        wall_started,
        collect_stage_timings,
    )
    .map(|(surface, _report)| surface)
}

#[cfg(feature = "cuda-runtime")]
fn decode_grayscale_cuda_resident_scaled_surface(
    decoder: &mut J2kDecoder<'_>,
    session: &mut CudaSession,
    fmt: PixelFormat,
    output_dimensions: (u32, u32),
) -> Result<Surface, Error> {
    let collect_stage_timings = profile::profile_stages_enabled();
    let wall_started = profile::profile_now(collect_stage_timings);
    let (plan, mut report) =
        decoder.build_cuda_htj2k_grayscale_scaled_plan_with_profile(fmt, output_dimensions)?;
    decode_grayscale_cuda_resident_surface_with_plan_profile(
        session,
        fmt,
        &plan,
        &mut report,
        wall_started,
        collect_stage_timings,
    )
    .map(|(surface, _report)| surface)
}

#[cfg(feature = "cuda-runtime")]
fn decode_grayscale_cuda_resident_region_scaled_surface(
    decoder: &mut J2kDecoder<'_>,
    session: &mut CudaSession,
    fmt: PixelFormat,
    scaled_roi: Rect,
    scaled_dimensions: (u32, u32),
) -> Result<Surface, Error> {
    let collect_stage_timings = profile::profile_stages_enabled();
    let wall_started = profile::profile_now(collect_stage_timings);
    let (plan, mut report) = decoder.build_cuda_htj2k_grayscale_region_scaled_plan_with_profile(
        fmt,
        scaled_roi,
        scaled_dimensions,
    )?;
    decode_grayscale_cuda_resident_surface_with_plan_profile(
        session,
        fmt,
        &plan,
        &mut report,
        wall_started,
        collect_stage_timings,
    )
    .map(|(surface, _report)| surface)
}

#[cfg(not(feature = "cuda-runtime"))]
pub(in crate::decoder) fn decode_to_cuda_resident_surface_impl(
    _decoder: &mut J2kDecoder<'_>,
    _session: &mut CudaSession,
    _fmt: PixelFormat,
) -> Result<Surface, Error> {
    Err(Error::CudaUnavailable)
}

#[cfg(not(feature = "cuda-runtime"))]
pub(in crate::decoder) fn decode_to_cuda_resident_surface_with_profile_impl(
    _decoder: &mut J2kDecoder<'_>,
    _session: &mut CudaSession,
    _fmt: PixelFormat,
) -> Result<(Surface, CudaHtj2kProfileReport), Error> {
    Err(Error::CudaUnavailable)
}

#[cfg(not(feature = "cuda-runtime"))]
pub(in crate::decoder) fn decode_region_to_cuda_resident_surface_impl(
    _decoder: &mut J2kDecoder<'_>,
    _session: &mut CudaSession,
    _fmt: PixelFormat,
    _roi: Rect,
) -> Result<Surface, Error> {
    Err(Error::CudaUnavailable)
}

#[cfg(not(feature = "cuda-runtime"))]
pub(in crate::decoder) fn decode_scaled_to_cuda_resident_surface_impl(
    _decoder: &mut J2kDecoder<'_>,
    _session: &mut CudaSession,
    _fmt: PixelFormat,
    _scale: Downscale,
) -> Result<Surface, Error> {
    Err(Error::CudaUnavailable)
}

#[cfg(not(feature = "cuda-runtime"))]
pub(in crate::decoder) fn decode_region_scaled_to_cuda_resident_surface_impl(
    _decoder: &mut J2kDecoder<'_>,
    _session: &mut CudaSession,
    _fmt: PixelFormat,
    _roi: Rect,
    _scale: Downscale,
) -> Result<Surface, Error> {
    Err(Error::CudaUnavailable)
}

#[cfg(not(feature = "cuda-runtime"))]
pub(in crate::decoder) fn decode_batch_to_cuda_resident_surface_with_profile_control(
    _inputs: &[&[u8]],
    _session: &mut CudaSession,
    _fmt: PixelFormat,
    _collect_stage_timings: bool,
) -> Result<(Vec<Surface>, CudaHtj2kProfileReport), Error> {
    Err(Error::CudaUnavailable)
}
