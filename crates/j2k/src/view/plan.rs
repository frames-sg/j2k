// SPDX-License-Identifier: MIT OR Apache-2.0

//! Header-planned decode routes, queryable before decoding.

use super::{validate_region, J2kDecoder};
use crate::{
    backend::{ColorSpace, Image},
    decode::{
        check_icc_working_control, full_image_route_beside, output_streams_rows,
        region_decode_uses_rows, resolve_u8_color_space, rows_stream_whole, srgb8_output_layout,
        FullImageRoute, RouteRequest,
    },
    BatchAlpha, J2kError, J2kReferenceGrid,
};
use j2k_core::{Downscale, PixelFormat, Rect, Unsupported};
use j2k_native::RegionRowsBlocker;

/// How a decode is planned to run, from the image's headers and the decoder's
/// settings, before anything is decoded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum J2kDecodeRoute {
    /// Components are reconstructed into planes before packing. Full-image
    /// work grows with image area. Region output normally holds region-sized
    /// planes and the native reconstruction support window. Unit-sampled
    /// exact-integer regions use bounded windows; subsampled ones reconstruct
    /// the tiles they touch and keep only the region's samples.
    Planes,
    /// One region decode of the whole image hands rows to the output as
    /// they are finished, holding a band of rows rather than planes.
    StreamedRows,
    /// Several tiles decode at once, each handing its rows over as they are
    /// finished; `workers` tiles are in flight.
    StreamedTiles {
        /// Tiles decoding at once.
        workers: usize,
    },
    /// Each stripe of rows decodes as a region of its own: output scratch
    /// is bounded by the stripe, and code-blocks a stripe overlaps are
    /// decoded again for the next.
    StripeRegions,
    /// The whole image decodes once at native precision, and its rows are
    /// handed over a stripe at a time: working memory includes the whole
    /// image's samples.
    WholeImageThenRows,
}

/// What keeps a decode's rows from streaming, as the image's headers say.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum J2kStreamingBlocker {
    /// A JP2 palette maps the samples.
    Palette,
    /// A JP2 channel-definition box reorders or interprets the channels.
    ChannelDefinitions,
    /// An sYCC or `CIELab` colour space is converted after decoding.
    ColorConversion,
    /// A component is deeper than 24 bits.
    PrecisionAbove24Bits,
    /// A component is subsampled.
    SubsampledComponents,
    /// The image or its tile grid is offset from the reference grid's origin.
    GridOffset,
    /// The image has more than four components.
    ComponentCount,
    /// The image has no wavelet decomposition levels to stream through.
    NoDecompositionLevels,
    /// The image's tiles lie side by side, so a row spans several tiles.
    TilesSideBySide,
    /// A tile's own coding, such as an inverse colour transform over fewer
    /// than three components, keeps its decode on component planes.
    TileCoding,
    /// The requested output cannot be written from streamed rows.
    OutputFormat,
}

/// A decode's planned route and, when its rows cannot stream, the first
/// reason the headers give.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct J2kDecodePlan {
    /// How the decode is planned to run.
    pub route: J2kDecodeRoute,
    /// Why the rows cannot stream, when they cannot. `None` when they can,
    /// including a [`J2kDecodeRoute::Planes`] decode chosen because the
    /// planes fit the decode budget and run faster for this image.
    pub blocker: Option<J2kStreamingBlocker>,
}

/// How packed output samples represent the decoded component values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum J2kSampleInterpretation {
    /// Unsigned eight-bit samples normalized from the component precision.
    /// This uses ordinary codec colour handling, without ICC normalization.
    NormalizedU8,
    /// Eight-bit sRGB colour, including restricted ICC conversion, with
    /// straight alpha when the output has an alpha channel.
    Srgb8,
    /// Native integer samples clamped to sixteen bits, in native-endian bytes.
    /// Component signedness is preserved (signed values use two's complement).
    Native16,
}

/// Geometry, packed layout and expected reconstruction route for one region.
///
/// Geometry and byte lengths are exact. Route information is derived from main
/// and tile headers; deeper validation can still reject data or require the
/// component-plane fallback, as with [`J2kDecodePlan`]. Planning parses bounded
/// metadata and allocates no decoded image or output buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct J2kRegionDecodePlan {
    /// Full-resolution reference grid, including the image origin.
    pub reference_grid: J2kReferenceGrid,
    /// Requested rectangle relative to the full-resolution image's origin.
    pub source_region: Rect,
    /// Covered rectangle relative to the reduced image's origin.
    pub output_region: Rect,
    /// Complete image dimensions at the requested reduction.
    pub reduced_image_dimensions: (u32, u32),
    /// Exact number of discarded JPEG 2000 resolution levels.
    pub reduction_levels: u8,
    /// Interleaved output channel layout and storage width.
    pub format: PixelFormat,
    /// Interpretation of the packed sample values.
    pub sample_interpretation: J2kSampleInterpretation,
    /// Interpretation of the packed alpha channel. Layouts without alpha
    /// report [`BatchAlpha::None`]; opaque RGBA fill and explicit sRGB RGBA
    /// report [`BatchAlpha::Straight`]. Ordinary RGBA preserves the source's
    /// straight or premultiplied convention. Dropping alpha leaves colour
    /// values unchanged; it does not unassociate premultiplied colour.
    pub alpha: BatchAlpha,
    /// Bytes written per row, without padding.
    pub row_bytes: usize,
    /// Bytes required for tightly packed output.
    pub buffer_bytes: usize,
    /// Expected reconstruction route and any header-derived streaming blocker.
    pub decode: J2kDecodePlan,
}

impl J2kRegionDecodePlan {
    /// Checked buffer span for a caller-selected stride, excluding final padding.
    ///
    /// # Errors
    /// Returns a buffer error for a short stride, overflow or an unaddressable span.
    pub fn buffer_len(&self, stride: usize) -> Result<usize, J2kError> {
        let dimensions = (self.output_region.w, self.output_region.h);
        j2k_core::validate_strided_output_buffer(dimensions, usize::MAX, stride, self.format)?;
        Ok(j2k_core::strided_output_len_capped(
            dimensions,
            stride,
            self.format,
            isize::MAX as usize,
            "planned region output",
        )?)
    }
}

impl J2kDecoder<'_> {
    /// Largest representable native reduction common to every component and tile.
    ///
    /// This includes tile COD/COC overrides, not just the main-header ladder.
    /// A region can still cover no samples at an available reduction;
    /// [`Self::region_decode_plan`] checks that geometry separately.
    ///
    /// # Errors
    /// Returns a metadata, tile-header or resource-limit error before output allocation.
    pub fn max_reduction_levels(&mut self) -> Result<u8, J2kError> {
        let result = self.source_max_reduction_levels();
        self.trim_preparation_cache();
        result
    }

    /// Plan packed region output at an exact native reduction before allocating it.
    ///
    /// `roi` uses full-resolution image coordinates. Geometry is shared with
    /// [`Self::decode_region_scaled_pow2_into`], including nonzero origins and
    /// reductions deeper than [`Downscale::Eighth`]. ICC normalization is a
    /// separate operation.
    ///
    /// # Errors
    /// Rejects invalid regions, unavailable reductions, regions with no reduced
    /// samples, unsupported colour/layout combinations, overflow and resource limits.
    pub fn region_decode_plan(
        &mut self,
        fmt: PixelFormat,
        roi: Rect,
        reduction_levels: u8,
    ) -> Result<J2kRegionDecodePlan, J2kError> {
        let result = self.region_decode_plan_with_interpretation(Some(fmt), roi, reduction_levels);
        self.trim_preparation_cache();
        result
    }

    /// Plan explicit sRGB output for a source-coordinate region at an exact
    /// native reduction. Colour conversion uses bounded native region planes;
    /// ICC and gray-alpha output may need a second region-sized packed buffer.
    ///
    /// # Errors
    /// Returns geometry, reduction, colour/alpha/precision, malformed ICC or
    /// resource-limit errors before decoded samples or output are allocated.
    pub fn srgb8_region_decode_plan(
        &mut self,
        roi: Rect,
        reduction_levels: u8,
    ) -> Result<J2kRegionDecodePlan, J2kError> {
        let result = self.region_decode_plan_with_interpretation(None, roi, reduction_levels);
        self.trim_preparation_cache();
        result
    }

    pub(super) fn region_decode_plan_with_interpretation(
        &mut self,
        fmt: Option<PixelFormat>,
        roi: Rect,
        reduction_levels: u8,
    ) -> Result<J2kRegionDecodePlan, J2kError> {
        self.region_decode_plan_with_interpretation_and_control(
            fmt,
            roi,
            reduction_levels,
            &crate::J2kDecodeControl::default(),
        )
    }

    pub(super) fn region_decode_plan_with_interpretation_and_control(
        &mut self,
        fmt: Option<PixelFormat>,
        roi: Rect,
        reduction_levels: u8,
        control: &crate::J2kDecodeControl,
    ) -> Result<J2kRegionDecodePlan, J2kError> {
        control.check_cancelled()?;
        validate_region(roi, self.info.dimensions)?;
        let grid = self.reference_grid();
        let output = grid
            .reduced_region(roi, reduction_levels)
            .ok_or(Unsupported {
                what: "requested reduction exceeds supported image geometry",
            })?;
        if output.w == 0 || output.h == 0 {
            return Err(Unsupported {
                what: "region has no samples at the requested reduction",
            }
            .into());
        }
        if reduction_levels > self.source_max_reduction_levels_with_control(control)? {
            return Err(Unsupported {
                what: "requested reduction exceeds a component or tile resolution ladder",
            }
            .into());
        }
        self.ensure_image_at_reduction_with_control(reduction_levels, control)?;
        let image = self.cached_image()?;
        check_icc_working_control(image, control)?;
        super::deep_scale::validate_reduced_image_dimensions(
            image,
            self.info.dimensions,
            super::deep_scale::reduction_denominator(reduction_levels)?,
        )?;
        plan_packed_region(
            image,
            grid,
            roi,
            output,
            fmt,
            reduction_levels,
            (&self.native_context, &control.to_native()),
        )
        .map_err(|error| control.normalize_error(error))
    }

    /// The route [`Self::decode_into`] plans for a full-image decode in
    /// `fmt`, without decoding.
    ///
    /// The plan comes from the main and tile headers. Coding the headers do
    /// not show, such as a tile-level coding style without decomposition
    /// levels, can still keep a planned streamed decode on component planes.
    ///
    /// # Errors
    /// Returns [`J2kError`] when the image cannot be parsed.
    pub fn full_decode_plan(&mut self, fmt: PixelFormat) -> Result<J2kDecodePlan, J2kError> {
        let result = (|| {
            self.ensure_image()?;
            let image = self.cached_image()?;
            let request = RouteRequest {
                fmt,
                budget: self.full_decode_planes_budget(),
                parallelism: self.native_context.cpu_decode_parallelism(),
                components: self.info.components,
                coded_bytes: self.bytes.len(),
                scale: Downscale::None,
            };
            let route = match full_image_route_beside(
                image,
                request,
                self.native_context
                    .retained_allocation_bytes()
                    .map_err(J2kError::from_native_decode_error)?,
                self.native_context.operation_control(),
            ) {
                FullImageRoute::Planes => J2kDecodeRoute::Planes,
                FullImageRoute::StreamedRegion => J2kDecodeRoute::StreamedRows,
                FullImageRoute::StreamedTiles { workers, .. } => {
                    J2kDecodeRoute::StreamedTiles { workers }
                }
            };
            let blocker = match route {
                J2kDecodeRoute::Planes => full_decode_blocker(image, fmt, &self.native_context),
                _ => None,
            };
            Ok(J2kDecodePlan { route, blocker })
        })();
        self.trim_preparation_cache();
        result
    }

    /// The route [`Self::decode_rows_u8_bounded`] and
    /// [`Self::decode_rows_u16_bounded`] plan, without decoding.
    ///
    /// The plan comes from the main and tile headers; see
    /// [`Self::full_decode_plan`].
    ///
    /// # Errors
    /// Returns [`J2kError`] when the image cannot be parsed.
    pub fn rows_decode_plan(&mut self) -> Result<J2kDecodePlan, J2kError> {
        let result = (|| {
            self.ensure_image()?;
            if self.info.bit_depth > 24 {
                return Ok(J2kDecodePlan {
                    route: J2kDecodeRoute::WholeImageThenRows,
                    blocker: Some(J2kStreamingBlocker::PrecisionAbove24Bits),
                });
            }
            let levels = self.info.resolution_levels;
            let image = self.cached_image()?;
            if rows_stream_whole(image, levels) {
                return Ok(J2kDecodePlan {
                    route: J2kDecodeRoute::StreamedRows,
                    blocker: None,
                });
            }
            let blocker = if levels <= 1 {
                J2kStreamingBlocker::NoDecompositionLevels
            } else if let Some(blocker) = image.region_rows_blocker() {
                native_blocker(blocker)
            } else if image.output_tile_rects().is_ok_and(|rects| {
                rects
                    .iter()
                    .any(|&(x, _, width, _)| x != 0 || width != image.width())
            }) {
                J2kStreamingBlocker::TilesSideBySide
            } else {
                J2kStreamingBlocker::TileCoding
            };
            Ok(J2kDecodePlan {
                route: J2kDecodeRoute::StripeRegions,
                blocker: Some(blocker),
            })
        })();
        self.trim_preparation_cache();
        result
    }
}

/// Why a full-image decode of `image` in `fmt` that the route planner kept
/// on the planes could not stream, or `None` when it could and the planes
/// were chosen on their merits.
fn full_decode_blocker(
    image: &Image<'_>,
    fmt: PixelFormat,
    context: &j2k_native::DecoderContext<'_>,
) -> Option<J2kStreamingBlocker> {
    if let Some(blocker) = image.region_rows_blocker() {
        return Some(native_blocker(blocker));
    }
    if !output_streams_rows(image, fmt) {
        return Some(J2kStreamingBlocker::OutputFormat);
    }
    if !matches!(
        image.full_decode_planes_floor_with_operation_control(
            0,
            context.retained_allocation_bytes().unwrap_or(usize::MAX),
            context.operation_control(),
        ),
        Ok(Some(_))
    ) {
        return Some(J2kStreamingBlocker::TileCoding);
    }
    None
}

/// The packed layout determines whether source association survives or an
/// opaque/normalized straight alpha channel is produced.
fn packed_output_alpha(
    image: &Image<'_>,
    format: PixelFormat,
    normalized: bool,
) -> Result<BatchAlpha, J2kError> {
    if !matches!(format, PixelFormat::Rgba8 | PixelFormat::Rgba16) {
        return Ok(BatchAlpha::None);
    }
    if !normalized
        && image.has_alpha()
        && image
            .alpha_is_premultiplied()
            .map_err(J2kError::from_native_decode_error)?
    {
        Ok(BatchAlpha::Premultiplied)
    } else {
        Ok(BatchAlpha::Straight)
    }
}

fn plan_packed_region(
    image: &Image<'_>,
    grid: J2kReferenceGrid,
    source: Rect,
    output: Rect,
    requested_format: Option<PixelFormat>,
    reduction_levels: u8,
    (context, control): (
        &j2k_native::DecoderContext<'_>,
        &j2k_native::DecodeOperationControl,
    ),
) -> Result<J2kRegionDecodePlan, J2kError> {
    let normalized = requested_format.is_none();
    let fmt = requested_format.map_or_else(
        || srgb8_output_layout(image).map(crate::J2kSrgb8Layout::format),
        Ok,
    )?;
    let uses_rows = if normalized {
        image
            .validate_region_bitmap_layout_with_operation_control(
                context
                    .retained_allocation_bytes()
                    .map_err(J2kError::from_native_decode_error)?,
                control,
            )
            .map_err(J2kError::from_native_decode_error)?;
        false
    } else {
        region_decode_uses_rows(image, fmt, output)?
    };
    let supported = normalized
        || matches!(
            (
                resolve_u8_color_space(image.color_space(), fmt),
                image.has_alpha(),
                fmt
            ),
            (
                ColorSpace::Gray,
                false,
                PixelFormat::Gray8 | PixelFormat::Gray16
            ) | (
                ColorSpace::RGB,
                _,
                PixelFormat::Rgb8 | PixelFormat::Rgba8 | PixelFormat::Rgb16 | PixelFormat::Rgba16
            )
        );
    if !supported {
        return Err(Unsupported {
            what: "backend color space cannot be mapped to requested pixel format",
        }
        .into());
    }
    let alpha = packed_output_alpha(image, fmt, normalized)?;
    let (row_bytes, buffer_bytes) = j2k_core::checked_surface_len(
        (output.w, output.h),
        fmt.bytes_per_pixel(),
        isize::MAX as usize,
        "planned region output",
    )?;
    let mut blocker = image.region_rows_blocker().map(native_blocker).or_else(|| {
        (normalized
            && (image.primary_icc_profile().is_some()
                || matches!(image.color_space(), ColorSpace::Icc { .. })
                || (matches!(image.color_space(), ColorSpace::Gray) && image.has_alpha())))
        .then_some(J2kStreamingBlocker::ColorConversion)
    });
    let streams = uses_rows
        && blocker.is_none()
        && image
            .region_rows_plan_with_operation_control(
                (output.x, output.y, output.w, output.h),
                context.cpu_decode_parallelism(),
                context
                    .retained_allocation_bytes()
                    .map_err(J2kError::from_native_decode_error)?,
                control,
            )
            .map_err(J2kError::from_native_decode_error)?;
    if uses_rows && !streams && blocker.is_none() {
        blocker = Some(J2kStreamingBlocker::TileCoding);
    }
    Ok(J2kRegionDecodePlan {
        reference_grid: grid,
        source_region: source,
        output_region: output,
        reduced_image_dimensions: (image.width(), image.height()),
        reduction_levels,
        format: fmt,
        sample_interpretation: if normalized {
            J2kSampleInterpretation::Srgb8
        } else if matches!(
            fmt,
            PixelFormat::Gray8 | PixelFormat::Rgb8 | PixelFormat::Rgba8
        ) {
            J2kSampleInterpretation::NormalizedU8
        } else {
            J2kSampleInterpretation::Native16
        },
        alpha,
        row_bytes,
        buffer_bytes,
        decode: J2kDecodePlan {
            route: if streams {
                J2kDecodeRoute::StreamedRows
            } else {
                J2kDecodeRoute::Planes
            },
            blocker,
        },
    })
}

fn native_blocker(blocker: RegionRowsBlocker) -> J2kStreamingBlocker {
    match blocker {
        RegionRowsBlocker::Palette => J2kStreamingBlocker::Palette,
        RegionRowsBlocker::ChannelDefinitions => J2kStreamingBlocker::ChannelDefinitions,
        RegionRowsBlocker::ColorConversion => J2kStreamingBlocker::ColorConversion,
        RegionRowsBlocker::PrecisionAbove24Bits => J2kStreamingBlocker::PrecisionAbove24Bits,
        RegionRowsBlocker::SubsampledComponents => J2kStreamingBlocker::SubsampledComponents,
        RegionRowsBlocker::GridOffset => J2kStreamingBlocker::GridOffset,
        // `ComponentCount`, and any later native reason, which this facade
        // names by the condition it extends.
        _ => J2kStreamingBlocker::ComponentCount,
    }
}
