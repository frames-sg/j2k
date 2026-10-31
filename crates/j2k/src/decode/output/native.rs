// SPDX-License-Identifier: MIT OR Apache-2.0

//! Plan-time choice of how a full-image decode into owned native-bit-depth
//! component planes runs, and the planes route's charge for packing them.

use crate::backend::Image;
use crate::decode::FullImageRoute;
use crate::J2kError;
use j2k_core::Rect;

/// The bytes the planes route charges at least when it packs `image`'s
/// output from its component planes at `bytes_per_sample` native bytes per
/// sample: every `f32` plane and the packed samples. It holds both at once,
/// after the decode's other buffers are released, and refuses a pack past
/// the decode budget. Counts the components of `image`'s color space and
/// alpha only, so it is at most the real charge.
pub(in crate::decode) fn planes_pack_floor(image: &Image<'_>, bytes_per_sample: usize) -> usize {
    let components = usize::from(image.color_space().num_channels())
        .saturating_add(usize::from(image.has_alpha()));
    (image.width() as usize)
        .saturating_mul(image.height() as usize)
        .saturating_mul(components)
        .saturating_mul(bytes_per_sample.saturating_add(size_of::<f32>()))
}

/// How a full-image decode of `image` into owned native component planes
/// runs when its planes may charge `planes_budget` bytes: as a region decode
/// of the whole image that packs its rows into the planes as they are
/// finished, when its rows carry the native samples (see
/// `Image::streamed_rows_native_precision`) and the planes route would be
/// refused, by its decode (see `Image::full_decode_planes_floor`) or by its
/// pack (see [`planes_pack_floor`]).
///
/// A planning error keeps the planes route, which meets the same error and
/// reports it.
pub(crate) fn native_components_route(image: &Image<'_>, planes_budget: usize) -> FullImageRoute {
    let Some(precision) = image.streamed_rows_native_precision() else {
        return FullImageRoute::Planes;
    };
    let pack_floor = planes_pack_floor(image, usize::from(precision).div_ceil(8));
    match image.full_decode_planes_floor(planes_budget) {
        Ok(Some(floor)) if floor.max(pack_floor) > planes_budget => FullImageRoute::StreamedRegion,
        _ => FullImageRoute::Planes,
    }
}

/// How a region decode of `image` into owned native component planes runs:
/// as one that packs its rows into the planes as they are finished whenever
/// its rows carry the native samples, as 8- and 16-bit region decodes
/// stream theirs, so that it holds no region-sized `f32` planes.
pub(crate) fn native_region_components_route(image: &Image<'_>) -> FullImageRoute {
    if image.streamed_rows_native_precision().is_some() {
        FullImageRoute::StreamedRegion
    } else {
        FullImageRoute::Planes
    }
}

/// Decode `roi` of `image`, or the whole image, into owned native component
/// planes on `route`, which [`native_components_route`] or
/// [`native_region_components_route`] chose for it.
pub(crate) fn decode_native_components_with_native_context<'a>(
    image: &Image<'a>,
    native_context: &mut j2k_native::DecoderContext<'a>,
    roi: Option<Rect>,
    route: FullImageRoute,
) -> Result<j2k_native::DecodedNativeComponents, J2kError> {
    match (roi, route) {
        (None, FullImageRoute::Planes) => {
            image.decode_native_components_with_context(native_context)
        }
        (Some(roi), FullImageRoute::Planes) => image.decode_native_region_components_with_context(
            (roi.x, roi.y, roi.w, roi.h),
            native_context,
        ),
        (roi, FullImageRoute::StreamedRegion) => {
            let roi = roi.unwrap_or_else(|| Rect::full((image.width(), image.height())));
            image.decode_native_region_components_rows_with_context(
                (roi.x, roi.y, roi.w, roi.h),
                native_context,
            )
        }
    }
    .map_err(J2kError::from_native_decode_error)
}
