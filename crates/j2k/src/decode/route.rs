// SPDX-License-Identifier: MIT OR Apache-2.0

//! Plan-time choice of how a full-image 8-bit decode runs.
//!
//! The planes route decodes every component into an image-sized plane of
//! `f32` samples and packs the output from them. Its working memory grows
//! with the image, and the native decoder refuses a decode whose planes and
//! largest tile's workspace pass its budget. Such an image instead runs as a
//! region decode of the whole image, which hands finished rows to the output
//! and keeps no planes (see
//! `Image::decode_region_rows_for_integer_output_with_context`). The native
//! decoder plans the planes route's charge from the codestream before
//! anything is decoded (see `Image::full_decode_planes_floor`), so a decode
//! never reruns after a failure, and an image the planes route can decode
//! keeps it, with its output and speed.

use crate::backend::Image;
use j2k_core::PixelFormat;

/// How a full-image decode runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FullImageRoute {
    /// Component planes for the whole image.
    Planes,
    /// A region decode of the whole image that streams its rows.
    StreamedRegion,
}

/// How a full-image decode of `image` to `fmt` runs when its planes may
/// charge `planes_budget` bytes: as a streamed region when the output has
/// 8 bits per sample and the planes would pass the budget.
///
/// A planning error keeps the planes route, which meets the same error and
/// reports it.
pub(crate) fn full_image_route(
    image: &Image<'_>,
    fmt: PixelFormat,
    planes_budget: usize,
) -> FullImageRoute {
    if !matches!(
        fmt,
        PixelFormat::Rgb8 | PixelFormat::Rgba8 | PixelFormat::Gray8
    ) {
        return FullImageRoute::Planes;
    }
    match image.full_decode_planes_floor(planes_budget) {
        Ok(Some(floor)) if floor > planes_budget => FullImageRoute::StreamedRegion,
        _ => FullImageRoute::Planes,
    }
}

#[cfg(test)]
mod tests;
