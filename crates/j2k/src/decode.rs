// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::backend::Image;
use crate::J2kError;
use alloc::vec::Vec;
use core::fmt;
use j2k_core::{validate_strided_output_buffer, DecodeOutcome, PixelFormat, Rect, Unsupported};

mod component_handoff;
pub use component_handoff::{
    J2kComponentPlane, J2kDecodedColorSpace, J2kDecodedComponents, J2kDecodedNativeComponents,
    J2kNativeComponentPlane,
};
mod output;
pub(crate) use output::{
    decode_native_components_with_native_context, native_components_route,
    native_region_components_route, resolve_u8_color_space, NativeRoute,
};
mod route;
mod settings;
mod srgb8;
mod stripes;
use output::{
    can_decode_u8_directly, expand_rgb8_to_rgba8, region_rows_preferred_u16, rows_match_native_u16,
    write_components_u16_output, write_components_u8_output, write_u16_output,
    write_u16_output_rows, write_u8_output, U16RowSink, U8RowSink,
};
pub(crate) use route::{
    full_image_route, full_image_route_beside, output_streams_rows, FullImageRoute, PlanesBudget,
    RouteRequest,
};
pub use settings::DecodeSettings;
pub(crate) use srgb8::{
    check_icc_working_control, decode_image_region_srgb8, decode_image_srgb8,
    decode_image_srgb8_with_control, srgb8_output_layout, RoutedSrgb8,
};
pub use srgb8::{J2kSrgb8Image, J2kSrgb8Layout};
pub(crate) use stripes::{
    decode_image_rows_in_stripes, rows_stream_whole, DeliverRow, StreamedRowsError,
};
#[cfg(test)]
pub(crate) use stripes::{InjectedFailure, INJECTED_FAILURE, STREAMED_ROW_DECODES};

/// Non-fatal JPEG 2000 decode warning surfaced through decode outcomes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum J2kDecodeWarning {
    /// Parsing used one of [`DecodeSettings::lenient`]'s documented JP2/JPH
    /// metadata recoveries.
    LenientMetadataRecovery,
}

// A successful batch can produce at most one warning. Keep that owner
// allocation-free until warning construction is made explicitly fallible and
// the batch metadata planner is updated for non-zero warning storage.
const _: [(); 0] = [(); core::mem::size_of::<J2kDecodeWarning>()];

impl fmt::Display for J2kDecodeWarning {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LenientMetadataRecovery => f.write_str("lenient JP2/JPH metadata recovery used"),
        }
    }
}

pub(crate) type J2kDecodeOutcome = DecodeOutcome<J2kDecodeWarning>;

pub(crate) fn decode_warnings_for_recovery(
    used_lenient_metadata_recovery: bool,
) -> Vec<J2kDecodeWarning> {
    let mut warnings = Vec::new();
    if used_lenient_metadata_recovery {
        // `J2kDecodeWarning` is statically constrained to a ZST above, so this
        // push cannot enter the allocator.
        warnings.push(J2kDecodeWarning::LenientMetadataRecovery);
    }
    warnings
}

pub(crate) fn decode_warnings_for_image(image: &Image<'_>) -> Vec<J2kDecodeWarning> {
    decode_warnings_for_recovery(image.used_lenient_metadata_recovery())
}

/// The whole of an image decoded once, from which its rows are written a
/// stripe at a time as the planes route of `decode_into` writes them: the
/// row decode of images deeper than 24 bits, whose exact samples no region
/// decode streams. Decoding once keeps a stripe height from multiplying the
/// decode work.
pub(crate) enum WholeImageRows {
    U8(crate::backend::Bitmap),
    U16(crate::backend::RawBitmap),
}

impl WholeImageRows {
    /// Decode the whole of `image` for rows in `fmt`.
    pub(crate) fn decode<'a>(
        image: &Image<'a>,
        native_context: &mut j2k_native::DecoderContext<'a>,
        fmt: PixelFormat,
    ) -> Result<Self, J2kError> {
        match fmt {
            PixelFormat::Gray8 | PixelFormat::Rgb8 | PixelFormat::Rgba8 => image
                .decode_with_context(native_context)
                .map(Self::U8)
                .map_err(J2kError::from_native_decode_error),
            PixelFormat::Gray16 | PixelFormat::Rgb16 | PixelFormat::Rgba16 => image
                .decode_native_with_context(native_context)
                .map(Self::U16)
                .map_err(J2kError::from_native_decode_error),
            _ => Err(Unsupported {
                what: "pixel format is not yet supported by j2k",
            }
            .into()),
        }
    }

    /// Write rows `rows` of `image`, decoded as `self`, to `out` from its
    /// first row.
    pub(crate) fn write_rows(
        &self,
        image: &Image<'_>,
        rows: core::ops::Range<usize>,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
    ) -> Result<(), J2kError> {
        match self {
            Self::U8(bitmap) => {
                let channels =
                    usize::from(bitmap.color_space.num_channels()) + usize::from(bitmap.has_alpha);
                let row_bytes = (bitmap.width as usize).checked_mul(channels).ok_or(
                    J2kError::DimensionOverflow {
                        width: bitmap.width,
                        height: bitmap.height,
                    },
                )?;
                if rows.start > rows.end || rows.end > bitmap.height as usize {
                    return Err(J2kError::internal_backend("bitmap rows outside the bitmap"));
                }
                let height = u32::try_from(rows.len())
                    .map_err(|_| J2kError::internal_backend("bitmap rows exceed u32"))?;
                let data = bitmap
                    .data
                    .get(rows.start * row_bytes..rows.end * row_bytes)
                    .ok_or_else(|| J2kError::internal_backend("bitmap shorter than its rows"))?;
                write_u8_output(
                    image.color_space(),
                    image.has_alpha(),
                    (bitmap.width, height),
                    data,
                    out,
                    stride,
                    fmt,
                )
            }
            Self::U16(raw) => write_u16_output_rows(
                image.color_space(),
                image.has_alpha(),
                raw,
                rows,
                out,
                stride,
                fmt,
            ),
        }
    }
}

/// Decode the full `image` into `out` on `route`, which
/// [`full_image_route`] chose for it.
pub(crate) fn decode_full_image_into_with_native_context<'a>(
    image: &Image<'a>,
    native_context: &mut j2k_native::DecoderContext<'a>,
    out: &mut [u8],
    stride: usize,
    fmt: PixelFormat,
    route: &FullImageRoute,
) -> Result<(), J2kError> {
    match route {
        FullImageRoute::Planes => {
            decode_image_into_with_native_context(image, native_context, out, stride, fmt)
        }
        FullImageRoute::StreamedRegion => decode_image_region_rows_into_with_native_context(
            image,
            native_context,
            out,
            stride,
            fmt,
            Rect::full((image.width(), image.height())),
        ),
        FullImageRoute::StreamedTiles { tiles, workers } => {
            route::tiles::decode_tiles_into_with_control(
                image,
                tiles,
                *workers,
                (out, stride, fmt),
                native_context.operation_control(),
            )
        }
    }
}

/// Decode the full `image` into `out` on `route`, which [`full_image_route`]
/// chose for `request`. A stream the budget refuses runs on the planes
/// instead, from a fresh context, when they fit the budget (see
/// `route::planes_fit`); the planes write every row, including rows the
/// stream wrote before it was refused.
pub(crate) fn decode_full_image_on_route<'a>(
    image: &Image<'a>,
    native_context: &mut j2k_native::DecoderContext<'a>,
    out: &mut [u8],
    stride: usize,
    request: RouteRequest,
    route: &FullImageRoute,
) -> Result<(), J2kError> {
    let fmt = request.fmt;
    match decode_full_image_into_with_native_context(image, native_context, out, stride, fmt, route)
    {
        Err(J2kError::ResourceLimit { .. })
            if route.streams() && route::planes_fit(image, request, native_context) =>
        {
            *native_context = native_context.fresh_for_operation();
            let planes = FullImageRoute::Planes;
            decode_full_image_into_with_native_context(
                image,
                native_context,
                out,
                stride,
                fmt,
                &planes,
            )
        }
        decoded => decoded,
    }
}

pub(crate) fn decode_image_into_with_native_context<'a>(
    image: &Image<'a>,
    native_context: &mut j2k_native::DecoderContext<'a>,
    out: &mut [u8],
    stride: usize,
    fmt: PixelFormat,
) -> Result<(), J2kError> {
    let dims = (image.width(), image.height());
    match fmt {
        PixelFormat::Rgb8 | PixelFormat::Rgba8 | PixelFormat::Gray8 => {
            if can_decode_u8_directly(image.color_space(), image.has_alpha(), dims, stride, fmt) {
                image
                    .decode_into(out, native_context)
                    .map_err(J2kError::from_native_decode_error)?;
                return Ok(());
            }
            if fmt == PixelFormat::Rgba8
                && stride == dims.0 as usize * 4
                && can_decode_u8_directly(
                    image.color_space(),
                    image.has_alpha(),
                    dims,
                    dims.0 as usize * 3,
                    PixelFormat::Rgb8,
                )
            {
                // The caller's validated tight RGBA buffer also fits RGB.
                // Keep the fused RGB route, then expand within that allocation.
                let pixels = dims.0 as usize * dims.1 as usize;
                image
                    .decode_into(&mut out[..pixels * 3], native_context)
                    .map_err(J2kError::from_native_decode_error)?;
                expand_rgb8_to_rgba8(&mut out[..pixels * 4]);
                return Ok(());
            }
            let decoded = image
                .decode_with_context(native_context)
                .map_err(J2kError::from_native_decode_error)?;
            write_u8_output(
                image.color_space(),
                image.has_alpha(),
                dims,
                &decoded.data,
                out,
                stride,
                fmt,
            )
        }
        PixelFormat::Rgb16 | PixelFormat::Rgba16 | PixelFormat::Gray16 => {
            let raw = image
                .decode_native_with_context(native_context)
                .map_err(J2kError::from_native_decode_error)?;
            write_u16_output(
                image.color_space(),
                image.has_alpha(),
                &raw,
                out,
                stride,
                fmt,
            )
        }
        _ => Err(Unsupported {
            what: "pixel format is not yet supported by j2k",
        }
        .into()),
    }
}

pub(crate) fn decode_image_region_into_with_native_context<'a>(
    image: &Image<'a>,
    native_context: &mut j2k_native::DecoderContext<'a>,
    out: &mut [u8],
    stride: usize,
    fmt: PixelFormat,
    roi: Rect,
) -> Result<(), J2kError> {
    if region_decode_uses_rows(image, fmt, roi)? {
        decode_image_region_rows_into_with_native_context(
            image,
            native_context,
            out,
            stride,
            fmt,
            roi,
        )
    } else {
        let raw = image
            .decode_native_region_with_context((roi.x, roi.y, roi.w, roi.h), native_context)
            .map_err(J2kError::from_native_decode_error)?;
        write_u16_output(
            image.color_space(),
            image.has_alpha(),
            &raw,
            out,
            stride,
            fmt,
        )
    }
}

/// Region execution policy shared with the public planning query.
pub(crate) fn region_decode_uses_rows(
    image: &Image<'_>,
    fmt: PixelFormat,
    roi: Rect,
) -> Result<bool, J2kError> {
    let rows = match fmt {
        PixelFormat::Rgb8 | PixelFormat::Rgba8 | PixelFormat::Gray8 => true,
        PixelFormat::Rgb16 | PixelFormat::Rgba16 | PixelFormat::Gray16 => {
            region_rows_preferred_u16(image, fmt, roi)
        }
        _ => {
            return Err(Unsupported {
                what: "pixel format is not yet supported by j2k",
            }
            .into())
        }
    };
    image
        .validate_region_sample_layout(rows)
        .map_err(J2kError::from_native_decode_error)?;
    Ok(rows)
}

/// Decode `roi` of `image` into `out` through a row sink, which writes
/// rows as they are finished instead of holding the region's component
/// planes: always at 8 bits, and at 16 bits when the rows carry the native
/// samples (see `rows_match_native_u16`), whose conversion the sink then
/// matches. A decode that fails may have written some rows.
pub(crate) fn decode_image_region_rows_into_with_native_context<'a>(
    image: &Image<'a>,
    native_context: &mut j2k_native::DecoderContext<'a>,
    out: &mut [u8],
    stride: usize,
    fmt: PixelFormat,
    roi: Rect,
) -> Result<(), J2kError> {
    decode_image_region_rows_into_beside(image, native_context, (out, stride, fmt), roi, 0)
}

/// [`decode_image_region_rows_into_with_native_context`] whose decode
/// charges `baseline_bytes` the caller holds beside it to the decode budget.
pub(crate) fn decode_image_region_rows_into_beside<'a>(
    image: &Image<'a>,
    native_context: &mut j2k_native::DecoderContext<'a>,
    (out, stride, fmt): (&mut [u8], usize, PixelFormat),
    roi: Rect,
    baseline_bytes: usize,
) -> Result<(), J2kError> {
    let roi = (roi.x, roi.y, roi.w, roi.h);
    match fmt {
        PixelFormat::Rgb8 | PixelFormat::Rgba8 | PixelFormat::Gray8 => {
            let mut sink = U8RowSink::new(image.color_space(), image.has_alpha(), out, stride, fmt);
            let decoded = image.decode_region_rows_for_integer_output_with_retained_baseline(
                roi,
                native_context,
                &mut sink,
                baseline_bytes,
            );
            sink.finish_decode(decoded)
        }
        PixelFormat::Rgb16 | PixelFormat::Rgba16 | PixelFormat::Gray16
            if rows_match_native_u16(image, fmt) =>
        {
            let mut sink =
                U16RowSink::new(image.color_space(), image.has_alpha(), out, stride, fmt);
            let decoded = image.decode_region_rows_for_integer_output_with_retained_baseline(
                roi,
                native_context,
                &mut sink,
                baseline_bytes,
            );
            sink.finish_decode(decoded)
        }
        _ => Err(Unsupported {
            what: "pixel format cannot stream its rows for this image",
        }
        .into()),
    }
}

pub(crate) fn decode_prepared_image_region_into(
    decoder: &mut j2k_native::PreparedRegionDecoder<'_, '_, '_>,
    out: &mut [u8],
    stride: usize,
    fmt: PixelFormat,
    roi: Rect,
) -> Result<(), J2kError> {
    let components = decoder
        .decode_region_components_for_integer_output((roi.x, roi.y, roi.w, roi.h))
        .map_err(J2kError::from_native_decode_error)?;
    match fmt {
        PixelFormat::Rgb8 | PixelFormat::Rgba8 | PixelFormat::Gray8 => {
            write_components_u8_output(&components, out, stride, fmt)
        }
        PixelFormat::Rgb16 | PixelFormat::Rgba16 | PixelFormat::Gray16 => {
            write_components_u16_output(&components, out, stride, fmt)
        }
        _ => Err(Unsupported {
            what: "pixel format is not yet supported by j2k",
        }
        .into()),
    }
}

pub(crate) fn validate_buffer(
    dims: (u32, u32),
    out_len: usize,
    stride: usize,
    fmt: PixelFormat,
) -> Result<(), J2kError> {
    validate_strided_output_buffer(dims, out_len, stride, fmt).map_err(Into::into)
}

pub(crate) fn validate_region(roi: Rect, dims: (u32, u32)) -> Result<(), J2kError> {
    if roi.w != 0 && roi.h != 0 && roi.is_within(dims) {
        return Ok(());
    }
    Err(J2kError::InvalidRegion {
        x: roi.x,
        y: roi.y,
        w: roi.w,
        h: roi.h,
        image_w: dims.0,
        image_h: dims.1,
    })
}

#[cfg(test)]
mod tests {
    use super::{decode_warnings_for_recovery, J2kDecodeWarning};

    #[test]
    fn decode_warnings_only_report_an_actual_lenient_recovery() {
        let warnings = decode_warnings_for_recovery(false);
        assert!(warnings.is_empty(), "{warnings:?}");
        assert_eq!(
            decode_warnings_for_recovery(true),
            vec![J2kDecodeWarning::LenientMetadataRecovery]
        );
    }

    #[test]
    fn decode_warning_owner_is_statically_allocation_free() {
        assert_eq!(core::mem::size_of::<J2kDecodeWarning>(), 0);
        let warnings = decode_warnings_for_recovery(true);
        assert_eq!(warnings.len(), 1);
        assert_eq!(
            warnings
                .capacity()
                .checked_mul(core::mem::size_of::<J2kDecodeWarning>()),
            Some(0)
        );
    }
}
