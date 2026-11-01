// SPDX-License-Identifier: MIT OR Apache-2.0

//! Bounded row decodes of a whole image in one streamed region decode.
//!
//! A region decode of the whole image hands its rows to a sink as they are
//! finished, top to bottom when every tile spans the image's width. The
//! sink here packs them into a stripe buffer of at most the caller's
//! stripe height and hands each full stripe to the caller, so every
//! code-block and wavelet row is decoded once however short the stripes
//! are, while the packed output held stays one stripe.

use core::ops::Range;

use crate::backend::{ColorSpace, Image};
use crate::J2kError;
use j2k_core::{DecodeRowsError, PixelFormat};

use super::output::{write_plane_rows_u16, write_plane_rows_u8};

#[cfg(test)]
std::thread_local! {
    /// Test-only count of the streamed bounded row decodes this thread
    /// started.
    pub(crate) static STREAMED_ROW_DECODES: core::cell::Cell<usize> =
        const { core::cell::Cell::new(0) };
    /// Test-only: the next streamed bounded row decode on this thread ends
    /// with a decode error, in place of its region decode or after it.
    pub(crate) static INJECTED_FAILURE: core::cell::Cell<Option<InjectedFailure>> =
        const { core::cell::Cell::new(None) };
}

/// Where a test makes a streamed bounded row decode fail.
#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum InjectedFailure {
    /// The region decode fails before it hands over any row.
    BeforeRows,
    /// The region decode fails after it handed over every row.
    AfterRows,
}

/// Whether a bounded row decode of `image`, whose main header codes
/// `resolution_levels` resolutions, can run as one streamed region decode
/// of the whole image: the main header codes at least one decomposition
/// level, every tile spans the image's width, so its rows arrive whole and
/// in order, and the region decode streams its rows as far as the main and
/// tile headers say (see `Image::full_decode_planes_floor`, which parses
/// the tiles and so comes last).
///
/// A planning error keeps the stripe-by-stripe decode, which meets the same
/// error and reports it.
pub(crate) fn rows_stream_whole(image: &Image<'_>, resolution_levels: u8) -> bool {
    resolution_levels > 1
        && image.region_decode_streams_rows()
        && (image.tile_count() == 1
            || image.output_tile_rects().is_ok_and(|rects| {
                rects
                    .iter()
                    .all(|&(x, _, width, _)| x == 0 && width == image.width())
            }))
        && matches!(image.full_decode_planes_floor(0), Ok(Some(_)))
}

/// Why a streamed bounded row decode did not hand over every row.
pub(crate) enum StreamedRowsError<E: core::error::Error + 'static> {
    /// The decode failed before any row reached the caller, who can decode
    /// the image a stripe at a time instead and meet that decode's rows and
    /// errors, as for an image that cannot stream.
    BeforeFirstRow,
    /// The caller's sink failed, or the decode failed after rows reached it.
    Rows(DecodeRowsError<J2kError, E>),
}

/// Hands one packed row, at its image row, to the caller.
pub(crate) type DeliverRow<'d, E> = dyn FnMut(u32, &[u8]) -> Result<(), E> + 'd;

/// Writes rows `rows` of a rectangle's component planes to packed output.
type PackRows = for<'p> fn(
    &ColorSpace,
    bool,
    &[j2k_native::ComponentPlane<'p>],
    (u32, u32),
    Range<usize>,
    &mut [u8],
    usize,
    PixelFormat,
) -> Result<(), J2kError>;

/// Decode the whole `image` in one streamed region decode, packing its rows
/// in `fmt` into `stripe`, which holds whole rows of `row_bytes` bytes, and
/// handing each full stripe, then the last, to `deliver` a row at a time.
///
/// `image` must stream its rows in order (see [`rows_stream_whole`]); a
/// decode that falls back to component planes hands them over at the end,
/// and they are packed a stripe at a time too. The first error stops the
/// rows reaching `deliver`: a caller error is returned as such even though
/// the decode runs to its end, and a decode error after rows reached the
/// caller leaves the rows of every stripe before it delivered and the
/// stripe it failed in held back.
pub(crate) fn decode_image_rows_in_stripes<'a, E: core::error::Error + 'static>(
    image: &Image<'a>,
    native_context: &mut j2k_native::DecoderContext<'a>,
    fmt: PixelFormat,
    (stripe, row_bytes): (&mut [u8], usize),
    deliver: &mut DeliverRow<'_, E>,
) -> Result<(), StreamedRowsError<E>> {
    let internal =
        |what| StreamedRowsError::Rows(DecodeRowsError::Decode(J2kError::internal_backend(what)));
    let pack: PackRows = match fmt {
        PixelFormat::Rgb8 | PixelFormat::Rgba8 | PixelFormat::Gray8 => write_plane_rows_u8,
        PixelFormat::Rgb16 | PixelFormat::Rgba16 | PixelFormat::Gray16 => write_plane_rows_u16,
        _ => return Err(internal("bounded row decode format without a row writer")),
    };
    let stripe_rows = stripe.len().checked_div(row_bytes).unwrap_or(0);
    if stripe_rows == 0 {
        return Err(internal("bounded row decode stripe holds no row"));
    }
    #[cfg(test)]
    STREAMED_ROW_DECODES.with(|count| count.set(count.get() + 1));
    #[cfg(test)]
    let injected = INJECTED_FAILURE.with(core::cell::Cell::take);
    #[cfg(not(test))]
    let skip_decode = false;
    #[cfg(test)]
    let skip_decode = injected == Some(InjectedFailure::BeforeRows);
    let mut sink = StripeSink {
        color_space: image.color_space(),
        has_alpha: image.has_alpha(),
        fmt,
        pack,
        width: image.width(),
        stripe,
        row_bytes,
        stripe_rows,
        first: 0,
        held: 0,
        deliver,
        delivered: false,
        error: None,
    };
    let decoded = if skip_decode {
        Ok(())
    } else {
        image.decode_region_rows_for_integer_output_with_context(
            (0, 0, image.width(), image.height()),
            native_context,
            &mut sink,
        )
    };
    #[cfg(test)]
    let decoded = match injected {
        Some(_) => decoded.and(Err(j2k_native::DecodingError::CodeBlockDecodeFailure.into())),
        None => decoded,
    };
    if let Some(error) = sink.error.take() {
        return Err(StreamedRowsError::Rows(error));
    }
    if let Err(error) = decoded {
        return Err(if sink.delivered {
            StreamedRowsError::Rows(DecodeRowsError::Decode(J2kError::from_native_decode_error(
                error,
            )))
        } else {
            StreamedRowsError::BeforeFirstRow
        });
    }
    sink.flush().map_err(StreamedRowsError::Rows)?;
    if sink.first != image.height() {
        return Err(internal("streamed region decode left rows undelivered"));
    }
    Ok(())
}

/// Packs a streamed region decode's rows into a stripe buffer and hands
/// each full stripe over. The first error stops every later row.
struct StripeSink<'s, E: core::error::Error + 'static> {
    color_space: &'s ColorSpace,
    has_alpha: bool,
    fmt: PixelFormat,
    pack: PackRows,
    width: u32,
    stripe: &'s mut [u8],
    row_bytes: usize,
    stripe_rows: usize,
    /// The image row the stripe's first row holds.
    first: u32,
    /// The rows the stripe holds.
    held: usize,
    deliver: &'s mut DeliverRow<'s, E>,
    /// Whether any row was handed to `deliver`.
    delivered: bool,
    error: Option<DecodeRowsError<J2kError, E>>,
}

impl<E: core::error::Error + 'static> StripeSink<'_, E> {
    /// Pack a rectangle of whole rows from row `y`, which must be the next
    /// row, delivering every stripe it fills.
    fn take_rows(
        &mut self,
        x: u32,
        y: u32,
        planes: &[j2k_native::ComponentPlane<'_>],
    ) -> Result<(), DecodeRowsError<J2kError, E>> {
        let dims = planes
            .first()
            .map_or((0, 0), j2k_native::ComponentPlane::dimensions);
        let next = u32::try_from(self.held)
            .ok()
            .and_then(|held| self.first.checked_add(held));
        if x != 0 || dims.0 != self.width || Some(y) != next {
            return Err(DecodeRowsError::Decode(J2kError::internal_backend(
                "streamed rows arrived out of order",
            )));
        }
        let rows = dims.1 as usize;
        let mut row = 0;
        while row < rows {
            let take = (self.stripe_rows - self.held).min(rows - row);
            let out = &mut self.stripe[self.held * self.row_bytes..];
            (self.pack)(
                self.color_space,
                self.has_alpha,
                planes,
                dims,
                row..row + take,
                out,
                self.row_bytes,
                self.fmt,
            )
            .map_err(DecodeRowsError::Decode)?;
            self.held += take;
            row += take;
            if self.held == self.stripe_rows {
                self.flush()?;
            }
        }
        Ok(())
    }

    /// Hand the rows the stripe holds over, a row at a time.
    fn flush(&mut self) -> Result<(), DecodeRowsError<J2kError, E>> {
        let mut y = self.first;
        for row in self.stripe.chunks_exact(self.row_bytes).take(self.held) {
            self.delivered = true;
            (self.deliver)(y, row).map_err(DecodeRowsError::Sink)?;
            y += 1;
        }
        self.first = y;
        self.held = 0;
        Ok(())
    }
}

impl<E: core::error::Error + 'static> j2k_native::ComponentRowSink for StripeSink<'_, E> {
    fn write_rows(&mut self, x: u32, y: u32, planes: &[j2k_native::ComponentPlane<'_>]) {
        if self.error.is_some() {
            return;
        }
        if let Err(error) = self.take_rows(x, y, planes) {
            self.error = Some(error);
        }
    }
}
