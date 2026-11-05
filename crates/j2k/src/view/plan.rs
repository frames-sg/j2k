// SPDX-License-Identifier: MIT OR Apache-2.0

//! Header-planned decode routes, queryable before decoding.

use super::J2kDecoder;
use crate::{
    backend::Image,
    decode::{
        full_image_route, output_streams_rows, rows_stream_whole, FullImageRoute, RouteRequest,
    },
    J2kError,
};
use j2k_core::{Downscale, PixelFormat};
use j2k_native::RegionRowsBlocker;

/// How a decode is planned to run, from the image's headers and the decoder's
/// settings, before anything is decoded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum J2kDecodeRoute {
    /// Every component is decoded into an image-sized plane, which the
    /// output is packed from: working memory grows with the image's area.
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

impl J2kDecoder<'_> {
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
        let route = match full_image_route(image, request) {
            FullImageRoute::Planes => J2kDecodeRoute::Planes,
            FullImageRoute::StreamedRegion => J2kDecodeRoute::StreamedRows,
            FullImageRoute::StreamedTiles { workers, .. } => {
                J2kDecodeRoute::StreamedTiles { workers }
            }
        };
        let blocker = match route {
            J2kDecodeRoute::Planes => full_decode_blocker(image, fmt),
            _ => None,
        };
        Ok(J2kDecodePlan { route, blocker })
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
    }
}

/// Why a full-image decode of `image` in `fmt` that the route planner kept
/// on the planes could not stream, or `None` when it could and the planes
/// were chosen on their merits.
fn full_decode_blocker(image: &Image<'_>, fmt: PixelFormat) -> Option<J2kStreamingBlocker> {
    if let Some(blocker) = image.region_rows_blocker() {
        return Some(native_blocker(blocker));
    }
    if !output_streams_rows(image, fmt) {
        return Some(J2kStreamingBlocker::OutputFormat);
    }
    if !matches!(image.full_decode_planes_floor(0), Ok(Some(_))) {
        return Some(J2kStreamingBlocker::TileCoding);
    }
    None
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
