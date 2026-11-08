// SPDX-License-Identifier: MIT OR Apache-2.0

//! Bounded row-decode planning, scratch ownership, and sink emission.

use super::{J2kDecoder, J2kRowDecodeOptions};
use crate::{
    decode::{
        decode_image_rows_in_stripes, decode_prepared_image_region_into, decode_warnings_for_image,
        rows_stream_whole, DeliverRow, J2kDecodeOutcome, StreamedRowsError, WholeImageRows,
    },
    scratch::J2kScratchPool,
    J2kError,
};
use alloc::vec::Vec;
use j2k_core::{
    BufferError, DecodeRowsError, PixelFormat, Rect, RowSink, DEFAULT_MAX_HOST_ALLOCATION_BYTES,
};

impl J2kRowDecodeOptions {
    /// Create row decode options with the requested maximum stripe height.
    ///
    /// A zero value is normalized to one row.
    pub const fn new(max_rows_per_stripe: u32) -> Self {
        Self {
            max_rows_per_stripe,
            max_stripe_bytes: DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        }
    }

    /// Create row decode options with explicit stripe height and byte caps.
    pub const fn new_with_max_stripe_bytes(
        max_rows_per_stripe: u32,
        max_stripe_bytes: usize,
    ) -> Self {
        Self {
            max_rows_per_stripe,
            max_stripe_bytes,
        }
    }

    /// Return a copy of these options with an explicit stripe byte cap.
    #[must_use]
    pub const fn with_max_stripe_bytes(mut self, max_stripe_bytes: usize) -> Self {
        self.max_stripe_bytes = max_stripe_bytes;
        self
    }

    /// Maximum number of decoded rows held per bounded row-decode stripe.
    pub const fn max_rows_per_stripe(self) -> u32 {
        if self.max_rows_per_stripe == 0 {
            1
        } else {
            self.max_rows_per_stripe
        }
    }

    /// Maximum number of packed bytes held per bounded row-decode stripe.
    pub const fn max_stripe_bytes(self) -> usize {
        self.max_stripe_bytes
    }
}

impl Default for J2kRowDecodeOptions {
    fn default() -> Self {
        Self::new(64)
    }
}

impl J2kDecoder<'_> {
    /// Decode rows into a `u8` row sink while bounding host output scratch to
    /// at most `options.max_rows_per_stripe()` rows.
    ///
    /// Whether the image decodes once is planned from its main and tile
    /// headers before anything is decoded: an image whose tiles span its
    /// width and whose region decode streams its rows as far as those headers
    /// say decodes as one region decode of the whole image, which hands
    /// finished rows over a stripe at a time; other images decode each stripe
    /// as a region of its own. Coding the plan does not see, such as a
    /// tile-level coding style without decomposition levels or HTJ2K
    /// code-blocks mixed with classic ones, can keep that region decode from
    /// streaming: it then decodes into component planes of the whole image
    /// and hands their rows over a stripe at a time once they are complete.
    /// [`Self::rows_decode_plan`] reports the plan, and why rows cannot
    /// stream, before anything is decoded.
    ///
    /// # Memory
    /// `max_rows_per_stripe` and `max_stripe_bytes` bound the packed output
    /// stripe. These limits do not bound decoder working memory or total
    /// process memory. Decoder memory depends on the image and parallelism
    /// and may include the working set of a whole-image decode, including
    /// component planes when native streaming is unavailable. Smaller stripes
    /// do not necessarily reduce decoder memory.
    ///
    /// # Errors
    /// Returns a decode error for unsupported formats or malformed input, and
    /// forwards sink errors without converting them to successful decodes.
    /// An error may leave a prefix of rows delivered. A sink error stops
    /// further sink calls and is returned as [`DecodeRowsError::Sink`], even
    /// if decoding subsequently fails; a decode that streams its rows stops
    /// at its next hand-over instead of decoding the rest of the image. A decode failure is retried stripe by
    /// stripe only if no row has been handed to the sink. After delivery
    /// starts, decode failures return [`DecodeRowsError::Decode`] without
    /// replaying rows.
    pub fn decode_rows_u8_bounded<R: RowSink<u8>>(
        &mut self,
        sink: &mut R,
        options: J2kRowDecodeOptions,
    ) -> Result<J2kDecodeOutcome, DecodeRowsError<J2kError, R::Error>> {
        let result = (|| {
            self.ensure_image().map_err(DecodeRowsError::Decode)?;
            let fmt = self
                .cached_image()
                .and_then(row_format_u8)
                .map_err(DecodeRowsError::Decode)?;
            let row_bytes = row_bytes_for(self.info(), fmt).map_err(DecodeRowsError::Decode)?;
            let width = self.info.dimensions.0;
            let height = self.info.dimensions.1;
            let options = options.with_max_stripe_bytes(
                options
                    .max_stripe_bytes()
                    .min(DEFAULT_MAX_HOST_ALLOCATION_BYTES),
            );
            let (stripe_rows, max_stripe_len) =
                bounded_row_stripe_layout(row_bytes, height, options)
                    .map_err(DecodeRowsError::Decode)?;
            if self.info.bit_depth > 24 {
                let stripes = WholeStripes {
                    fmt,
                    row_bytes,
                    stripe_rows,
                    max_stripe_len,
                };
                self.decode_rows_from_whole(stripes, &mut |y, row| sink.write_row(y, row))?;
                return Ok(self.full_rows_outcome());
            }
            let mut pool = J2kScratchPool::new();
            if self.rows_stream_whole() {
                let stripe = pool
                    .packed_bytes(max_stripe_len)
                    .map_err(|error| DecodeRowsError::Decode(J2kError::Buffer(error)))?;
                match self.decode_rows_streamed(fmt, (stripe, row_bytes), &mut |y, row| {
                    sink.write_row(y, row)
                }) {
                    Ok(()) => return Ok(self.full_rows_outcome()),
                    Err(StreamedRowsError::Rows(error)) => return Err(error),
                    Err(StreamedRowsError::BeforeFirstRow) => {}
                }
            }
            let (Some(image), native_context) = (self.image.as_ref(), &mut self.native_context)
            else {
                return Err(DecodeRowsError::Decode(J2kError::internal_backend(
                    "internal image cache missing",
                )));
            };
            let mut decoder = image
                .prepare_region_decoder_with_context(native_context)
                .map_err(|error| {
                    DecodeRowsError::Decode(J2kError::from_native_decode_error(error))
                })?;
            let mut y = 0_u32;
            while y < height {
                let rows = stripe_rows.min(height - y);
                let stripe_len = row_bytes.checked_mul(rows as usize).ok_or_else(|| {
                    DecodeRowsError::Decode(J2kError::Buffer(BufferError::SizeOverflow {
                        what: "J2K bounded row decode stripe buffer",
                    }))
                })?;
                let stripe = pool
                    .packed_bytes(max_stripe_len)
                    .map_err(|error| DecodeRowsError::Decode(J2kError::Buffer(error)))?;
                decode_prepared_image_region_into(
                    &mut decoder,
                    &mut stripe[..stripe_len],
                    row_bytes,
                    fmt,
                    Rect {
                        x: 0,
                        y,
                        w: width,
                        h: rows,
                    },
                )
                .map_err(DecodeRowsError::Decode)?;
                for row_index in 0..rows {
                    let start = row_index as usize * row_bytes;
                    sink.write_row(y + row_index, &stripe[start..start + row_bytes])
                        .map_err(DecodeRowsError::Sink)?;
                }
                y += rows;
            }
            Ok(self.full_rows_outcome())
        })();
        self.trim_preparation_cache();
        result
    }

    /// Decode rows into a `u16` row sink while bounding host output scratch to
    /// at most `options.max_rows_per_stripe()` rows.
    ///
    /// Images decode once or a stripe at a time as planned for
    /// [`Self::decode_rows_u8_bounded`]; images deeper than 24 bits decode
    /// whole once and hand their rows over a stripe at a time.
    ///
    /// # Memory
    /// `max_rows_per_stripe` and `max_stripe_bytes` bound the packed output
    /// stripe, beside which this method uses one row of `u16` conversion
    /// scratch. These limits do not bound decoder working memory or total
    /// process memory. Decoder memory depends on the image and parallelism
    /// and may include the working set of a whole-image decode, including
    /// component planes when native streaming is unavailable. Smaller stripes
    /// do not necessarily reduce decoder memory.
    ///
    /// # Errors
    /// Returns a decode error for unsupported formats or malformed input, and
    /// forwards sink errors without converting them to successful decodes.
    /// An error may leave a prefix of rows delivered. A sink error stops
    /// further sink calls and is returned as [`DecodeRowsError::Sink`], even
    /// if decoding subsequently fails; a decode that streams its rows stops
    /// at its next hand-over instead of decoding the rest of the image. A decode failure is retried stripe by
    /// stripe only if no row has been handed to the sink. After delivery
    /// starts, decode failures return [`DecodeRowsError::Decode`] without
    /// replaying rows.
    pub fn decode_rows_u16_bounded<R: RowSink<u16>>(
        &mut self,
        sink: &mut R,
        options: J2kRowDecodeOptions,
    ) -> Result<J2kDecodeOutcome, DecodeRowsError<J2kError, R::Error>> {
        let result = (|| {
            self.ensure_image().map_err(DecodeRowsError::Decode)?;
            let fmt = self
                .cached_image()
                .and_then(row_format_u16)
                .map_err(DecodeRowsError::Decode)?;
            let row_bytes = row_bytes_for(self.info(), fmt).map_err(DecodeRowsError::Decode)?;
            let samples_per_row =
                row_samples_for(self.info(), fmt).map_err(DecodeRowsError::Decode)?;
            let width = self.info.dimensions.0;
            let height = self.info.dimensions.1;
            let row_scratch_bytes = samples_per_row
                .checked_mul(core::mem::size_of::<u16>())
                .ok_or(DecodeRowsError::Decode(J2kError::Buffer(
                    BufferError::SizeOverflow {
                        what: "J2K bounded row u16 scratch",
                    },
                )))?;
            let packed_cap = DEFAULT_MAX_HOST_ALLOCATION_BYTES
                .checked_sub(row_scratch_bytes)
                .ok_or(DecodeRowsError::Decode(J2kError::Buffer(
                    BufferError::AllocationTooLarge {
                        requested: row_scratch_bytes,
                        cap: DEFAULT_MAX_HOST_ALLOCATION_BYTES,
                        what: "J2K bounded row u16 scratch",
                    },
                )))?;
            let options = options.with_max_stripe_bytes(options.max_stripe_bytes().min(packed_cap));
            let (stripe_rows, max_stripe_len) =
                bounded_row_stripe_layout(row_bytes, height, options)
                    .map_err(DecodeRowsError::Decode)?;
            let layout = U16StripeLayout {
                width,
                height,
                row_bytes,
                samples_per_row,
                stripe_rows,
                max_stripe_len,
            };
            if self.info.bit_depth > 24 {
                let stripes = WholeStripes {
                    fmt,
                    row_bytes,
                    stripe_rows,
                    max_stripe_len,
                };
                let mut row = Vec::new();
                row.try_reserve_exact(samples_per_row).map_err(|_| {
                    DecodeRowsError::Decode(J2kError::Buffer(BufferError::HostAllocationFailed {
                        bytes: row_scratch_bytes,
                        what: "J2K bounded row u16 scratch",
                    }))
                })?;
                row.resize(samples_per_row, 0_u16);
                self.decode_rows_from_whole(stripes, &mut |y, packed_row| {
                    widen_row_u16(packed_row, &mut row);
                    sink.write_row(y, &row)
                })?;
                return Ok(self.full_rows_outcome());
            }
            if self.rows_stream_whole() {
                let mut pool = J2kScratchPool::new();
                let (packed, row) = pool
                    .packed_bytes_and_row_u16(layout.max_stripe_len, layout.samples_per_row)
                    .map_err(|error| DecodeRowsError::Decode(J2kError::Buffer(error)))?;
                match self.decode_rows_streamed(fmt, (packed, row_bytes), &mut |y, packed_row| {
                    widen_row_u16(packed_row, row);
                    sink.write_row(y, row)
                }) {
                    Ok(()) => return Ok(self.full_rows_outcome()),
                    Err(StreamedRowsError::Rows(error)) => return Err(error),
                    Err(StreamedRowsError::BeforeFirstRow) => {}
                }
            }
            let (Some(image), native_context) = (self.image.as_ref(), &mut self.native_context)
            else {
                return Err(DecodeRowsError::Decode(J2kError::internal_backend(
                    "internal image cache missing",
                )));
            };
            let mut decoder = image
                .prepare_region_decoder_with_context(native_context)
                .map_err(|error| {
                    DecodeRowsError::Decode(J2kError::from_native_decode_error(error))
                })?;
            pump_u16_rows(sink, layout, |packed, region| {
                decode_prepared_image_region_into(&mut decoder, packed, row_bytes, fmt, region)
            })?;
            Ok(self.full_rows_outcome())
        })();
        self.trim_preparation_cache();
        result
    }

    /// Whether this decoder's image decodes its rows in one streamed region
    /// decode (see [`rows_stream_whole`]).
    fn rows_stream_whole(&self) -> bool {
        self.cached_image()
            .is_ok_and(|image| rows_stream_whole(image, self.info.resolution_levels))
    }

    /// Decode the whole image in one streamed region decode, handing its
    /// rows to `deliver` a stripe of `stripe` at a time (see
    /// [`decode_image_rows_in_stripes`]).
    fn decode_rows_streamed<E: core::error::Error + 'static>(
        &mut self,
        fmt: PixelFormat,
        stripe: (&mut [u8], usize),
        deliver: &mut DeliverRow<'_, E>,
    ) -> Result<(), StreamedRowsError<E>> {
        let (Some(image), native_context) = (self.image.as_ref(), &mut self.native_context) else {
            return Err(StreamedRowsError::Rows(DecodeRowsError::Decode(
                J2kError::internal_backend("internal image cache missing"),
            )));
        };
        let decoded = decode_image_rows_in_stripes(image, native_context, fmt, stripe, deliver);
        // As after a region decode, release the decode buffers rather than
        // hold them while the decoder is idle.
        self.native_context = self.fresh_native_context();
        decoded
    }

    /// Decode the whole image once and hand its rows to `deliver` a stripe
    /// of `stripes` at a time, as the planes route of `decode_into` writes
    /// them: rows of images deeper than 24 bits, whose exact samples no
    /// region decode streams.
    fn decode_rows_from_whole<E: core::error::Error + 'static>(
        &mut self,
        stripes: WholeStripes,
        deliver: &mut DeliverRow<'_, E>,
    ) -> Result<(), DecodeRowsError<J2kError, E>> {
        let WholeStripes {
            fmt,
            row_bytes,
            stripe_rows,
            max_stripe_len,
        } = stripes;
        let height = self.info.dimensions.1;
        let (Some(image), native_context) = (self.image.as_ref(), &mut self.native_context) else {
            return Err(DecodeRowsError::Decode(J2kError::internal_backend(
                "internal image cache missing",
            )));
        };
        let whole =
            WholeImageRows::decode(image, native_context, fmt).map_err(DecodeRowsError::Decode)?;
        let mut pool = J2kScratchPool::new();
        let stripe = pool
            .packed_bytes(max_stripe_len)
            .map_err(|error| DecodeRowsError::Decode(J2kError::Buffer(error)))?;
        let mut y = 0_u32;
        while y < height {
            let rows = stripe_rows.min(height - y);
            let stripe_len = row_bytes.checked_mul(rows as usize).ok_or_else(|| {
                DecodeRowsError::Decode(J2kError::Buffer(BufferError::SizeOverflow {
                    what: "J2K bounded row decode stripe buffer",
                }))
            })?;
            let first = y as usize;
            whole
                .write_rows(
                    image,
                    first..first + rows as usize,
                    &mut stripe[..stripe_len],
                    row_bytes,
                    fmt,
                )
                .map_err(DecodeRowsError::Decode)?;
            for (index, row) in stripe[..stripe_len].chunks_exact(row_bytes).enumerate() {
                let row_y = y + u32::try_from(index).unwrap_or(u32::MAX);
                deliver(row_y, row).map_err(DecodeRowsError::Sink)?;
            }
            y += rows;
        }
        // As after a region decode, release the decode buffers rather than
        // hold them while the decoder is idle.
        self.native_context = self.fresh_native_context();
        Ok(())
    }

    /// The outcome of a rows decode that handed over every row: the whole
    /// image, with the warnings of the image it decoded.
    fn full_rows_outcome(&self) -> J2kDecodeOutcome {
        let warnings = self
            .image
            .as_ref()
            .map_or_else(Vec::new, decode_warnings_for_image);
        j2k_core::DecodeOutcome::new(Rect::full(self.info.dimensions), warnings)
    }
}

/// The stripes [`J2kDecoder::decode_rows_from_whole`] hands rows over in.
#[derive(Clone, Copy)]
struct WholeStripes {
    fmt: PixelFormat,
    row_bytes: usize,
    stripe_rows: u32,
    max_stripe_len: usize,
}

#[derive(Clone, Copy)]
struct U16StripeLayout {
    width: u32,
    height: u32,
    row_bytes: usize,
    samples_per_row: usize,
    stripe_rows: u32,
    max_stripe_len: usize,
}

fn pump_u16_rows<R: RowSink<u16>>(
    sink: &mut R,
    layout: U16StripeLayout,
    mut decode_stripe: impl FnMut(&mut [u8], Rect) -> Result<(), J2kError>,
) -> Result<(), DecodeRowsError<J2kError, R::Error>> {
    let mut pool = J2kScratchPool::new();
    let mut y = 0_u32;
    while y < layout.height {
        let rows = layout.stripe_rows.min(layout.height - y);
        let stripe_len = layout.row_bytes.checked_mul(rows as usize).ok_or_else(|| {
            DecodeRowsError::Decode(J2kError::Buffer(BufferError::SizeOverflow {
                what: "J2K bounded row decode stripe buffer",
            }))
        })?;
        let (packed, row) = pool
            .packed_bytes_and_row_u16(layout.max_stripe_len, layout.samples_per_row)
            .map_err(|error| DecodeRowsError::Decode(J2kError::Buffer(error)))?;
        decode_stripe(
            &mut packed[..stripe_len],
            Rect {
                x: 0,
                y,
                w: layout.width,
                h: rows,
            },
        )
        .map_err(DecodeRowsError::Decode)?;
        for row_index in 0..rows {
            let start = row_index as usize * layout.row_bytes;
            widen_row_u16(&packed[start..start + layout.row_bytes], row);
            sink.write_row(y + row_index, row)
                .map_err(DecodeRowsError::Sink)?;
        }
        y += rows;
    }
    Ok(())
}

/// The 16-bit samples of a packed little-endian row.
fn widen_row_u16(packed: &[u8], row: &mut [u16]) {
    for (dst, src) in row.iter_mut().zip(packed.as_chunks::<2>().0) {
        *dst = u16::from_le_bytes(*src);
    }
}

fn row_format_u8(image: &j2k_native::Image<'_>) -> Result<PixelFormat, J2kError> {
    match (image.color_space().num_channels(), image.has_alpha()) {
        (1, false) => Ok(PixelFormat::Gray8),
        (3, false) => Ok(PixelFormat::Rgb8),
        (3, true) => Ok(PixelFormat::Rgba8),
        _ => Err(j2k_core::Unsupported {
            what: "row decode only supports Gray/RGB/RGBA images",
        }
        .into()),
    }
}

fn row_format_u16(image: &j2k_native::Image<'_>) -> Result<PixelFormat, J2kError> {
    match (image.color_space().num_channels(), image.has_alpha()) {
        (1, false) => Ok(PixelFormat::Gray16),
        (3, false) => Ok(PixelFormat::Rgb16),
        (3, true) => Ok(PixelFormat::Rgba16),
        _ => Err(j2k_core::Unsupported {
            what: "row decode only supports Gray/RGB/RGBA images",
        }
        .into()),
    }
}

fn bounded_row_stripe_layout(
    row_bytes: usize,
    height: u32,
    options: J2kRowDecodeOptions,
) -> Result<(u32, usize), J2kError> {
    let cap = options.max_stripe_bytes();
    if row_bytes > cap {
        return Err(J2kError::Buffer(BufferError::AllocationTooLarge {
            requested: row_bytes,
            cap,
            what: "J2K bounded row decode stripe buffer",
        }));
    }

    let max_rows = options.max_rows_per_stripe();
    let image_rows = height.max(1);
    let rows_by_cap = cap.checked_div(row_bytes).map_or(max_rows, |capped_rows| {
        u32::try_from(capped_rows).unwrap_or(u32::MAX).max(1)
    });
    let stripe_rows = max_rows.min(image_rows).min(rows_by_cap);
    let max_stripe_len = row_bytes
        .checked_mul(stripe_rows as usize)
        .ok_or(J2kError::Buffer(BufferError::SizeOverflow {
            what: "J2K bounded row decode stripe buffer",
        }))?;

    Ok((stripe_rows, max_stripe_len))
}

fn row_bytes_for(info: &j2k_core::Info, fmt: PixelFormat) -> Result<usize, J2kError> {
    (info.dimensions.0 as usize)
        .checked_mul(fmt.bytes_per_pixel())
        .ok_or(J2kError::DimensionOverflow {
            width: info.dimensions.0,
            height: info.dimensions.1,
        })
}

fn row_samples_for(info: &j2k_core::Info, fmt: PixelFormat) -> Result<usize, J2kError> {
    (info.dimensions.0 as usize)
        .checked_mul(fmt.channels())
        .ok_or(J2kError::DimensionOverflow {
            width: info.dimensions.0,
            height: info.dimensions.1,
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::{InjectedFailure, INJECTED_FAILURE, STREAMED_ROW_DECODES};
    use j2k_native::EncodeOptions;

    /// A sink error carrying the row the sink refused.
    #[derive(Debug, PartialEq, Eq)]
    struct Refused(u32);

    impl core::fmt::Display for Refused {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            write!(f, "row {} refused", self.0)
        }
    }

    impl core::error::Error for Refused {}

    /// Rows as little-endian bytes and every row index handed over, in
    /// order, refusing row `refuse` when it is set.
    #[derive(Default)]
    struct Collect {
        bytes: Vec<u8>,
        order: Vec<u32>,
        refuse: Option<u32>,
    }

    impl Collect {
        fn take(&mut self, y: u32) -> Result<(), Refused> {
            self.order.push(y);
            if self.refuse == Some(y) {
                return Err(Refused(y));
            }
            Ok(())
        }
    }

    impl RowSink<u8> for Collect {
        type Error = Refused;

        fn write_row(&mut self, y: u32, row: &[u8]) -> Result<(), Refused> {
            self.take(y)?;
            self.bytes.extend_from_slice(row);
            Ok(())
        }
    }

    impl RowSink<u16> for Collect {
        type Error = Refused;

        fn write_row(&mut self, y: u32, row: &[u16]) -> Result<(), Refused> {
            self.take(y)?;
            self.bytes
                .extend(row.iter().flat_map(|sample| sample.to_le_bytes()));
            Ok(())
        }
    }

    fn rgb_codestream(tile_size: Option<(u32, u32)>, num_decomposition_levels: u8) -> Vec<u8> {
        let pixels: Vec<u8> = (0..64 * 48 * 3_u32)
            .map(|index| u8::try_from(index * 7 % 251).expect("a byte"))
            .collect();
        let options = EncodeOptions {
            tile_size,
            num_decomposition_levels,
            ..EncodeOptions::default()
        };
        j2k_native::encode(&pixels, 64, 48, 3, 8, false, &options).expect("encode")
    }

    /// The `decode_into` output in `fmt`.
    fn full_decode(bytes: &[u8], fmt: PixelFormat) -> Vec<u8> {
        let mut decoder = J2kDecoder::new(bytes).expect("decoder");
        let stride = 64 * fmt.bytes_per_pixel();
        let mut full = vec![0; stride * 48];
        decoder
            .decode_into(&mut full, stride, fmt)
            .expect("full decode");
        full
    }

    /// A bounded decode in 7-row stripes into `sink`, with `injected` set
    /// for it, and the streamed decodes it started.
    fn decode_rows(
        bytes: &[u8],
        fmt: PixelFormat,
        injected: Option<InjectedFailure>,
        sink: &mut Collect,
    ) -> (Result<(), DecodeRowsError<J2kError, Refused>>, usize) {
        let streamed = || STREAMED_ROW_DECODES.with(core::cell::Cell::get);
        let mut decoder = J2kDecoder::new(bytes).expect("decoder");
        let before = streamed();
        INJECTED_FAILURE.with(|failure| failure.set(injected));
        let options = J2kRowDecodeOptions::new(7);
        let result = if fmt == PixelFormat::Rgb16 {
            decoder.decode_rows_u16_bounded(sink, options)
        } else {
            decoder.decode_rows_u8_bounded(sink, options)
        };
        assert_eq!(INJECTED_FAILURE.with(core::cell::Cell::get), None);
        (result.map(drop), streamed() - before)
    }

    #[test]
    fn rows_decode_in_one_streamed_decode_only_when_they_arrive_whole_and_in_order() {
        for (label, bytes, streams) in [
            ("untiled", rgb_codestream(None, 3), true),
            (
                "tiles spanning the width",
                rgb_codestream(Some((64, 16)), 3),
                true,
            ),
            (
                "tiles side by side",
                rgb_codestream(Some((32, 16)), 3),
                false,
            ),
            ("no decomposition levels", rgb_codestream(None, 0), false),
        ] {
            for fmt in [PixelFormat::Rgb8, PixelFormat::Rgb16] {
                let mut sink = Collect::default();
                let (result, streamed) = decode_rows(&bytes, fmt, None, &mut sink);
                result.expect("rows");
                assert_eq!(streamed, usize::from(streams), "{label} {fmt:?}");
                assert_eq!(sink.order, (0..48).collect::<Vec<_>>(), "{label} {fmt:?}");
                assert!(
                    sink.bytes == full_decode(&bytes, fmt),
                    "{label} {fmt:?}: rows differ from decode_into"
                );
            }
        }
    }

    #[test]
    fn a_streamed_decode_failing_before_its_first_row_decodes_a_stripe_at_a_time() {
        let bytes = rgb_codestream(None, 3);
        for fmt in [PixelFormat::Rgb8, PixelFormat::Rgb16] {
            let full = full_decode(&bytes, fmt);
            let mut sink = Collect::default();
            let (result, streamed) =
                decode_rows(&bytes, fmt, Some(InjectedFailure::BeforeRows), &mut sink);
            result.expect("rows a stripe at a time");
            assert_eq!(streamed, 1, "{fmt:?}");
            assert_eq!(sink.order, (0..48).collect::<Vec<_>>(), "{fmt:?}");
            assert!(sink.bytes == full, "{fmt:?}: rows differ from decode_into");
            // The stripe-at-a-time decode meets the caller's sink as it is.
            let mut sink = Collect {
                refuse: Some(10),
                ..Collect::default()
            };
            let (result, _) =
                decode_rows(&bytes, fmt, Some(InjectedFailure::BeforeRows), &mut sink);
            assert!(
                matches!(result, Err(DecodeRowsError::Sink(Refused(10)))),
                "{fmt:?}"
            );
            assert_eq!(sink.order, (0..=10).collect::<Vec<_>>(), "{fmt:?}");
        }
    }

    #[test]
    fn a_streamed_decode_failing_after_rows_reached_the_sink_reports_it_without_a_rerun() {
        let bytes = rgb_codestream(None, 3);
        for fmt in [PixelFormat::Rgb8, PixelFormat::Rgb16] {
            let full = full_decode(&bytes, fmt);
            let row_bytes = 64 * fmt.bytes_per_pixel();
            let mut sink = Collect::default();
            let (result, streamed) =
                decode_rows(&bytes, fmt, Some(InjectedFailure::AfterRows), &mut sink);
            assert!(matches!(result, Err(DecodeRowsError::Decode(_))), "{fmt:?}");
            assert_eq!(streamed, 1, "{fmt:?}");
            // Six whole 7-row stripes; the last one, which the failure
            // ended, is held back, and nothing is handed over twice.
            assert_eq!(sink.order, (0..42).collect::<Vec<_>>(), "{fmt:?}");
            assert!(sink.bytes == full[..42 * row_bytes], "{fmt:?}");
            // A sink error comes first.
            let mut sink = Collect {
                refuse: Some(10),
                ..Collect::default()
            };
            let (result, _) = decode_rows(&bytes, fmt, Some(InjectedFailure::AfterRows), &mut sink);
            assert!(
                matches!(result, Err(DecodeRowsError::Sink(Refused(10)))),
                "{fmt:?}"
            );
            assert_eq!(sink.order, (0..=10).collect::<Vec<_>>(), "{fmt:?}");
        }
    }

    #[test]
    fn bounded_row_stripe_layout_clamps_rows_to_byte_cap() {
        let options = J2kRowDecodeOptions::new_with_max_stripe_bytes(100, 1_024);

        let (rows, bytes) =
            bounded_row_stripe_layout(100, 50, options).expect("stripe layout should fit");

        assert_eq!(rows, 10);
        assert_eq!(bytes, 1_000);
    }

    #[test]
    fn bounded_row_stripe_layout_rejects_single_row_over_cap() {
        let options = J2kRowDecodeOptions::new_with_max_stripe_bytes(100, 99);

        let err =
            bounded_row_stripe_layout(100, 50, options).expect_err("single row should exceed cap");

        assert!(matches!(
            err,
            J2kError::Buffer(BufferError::AllocationTooLarge {
                requested: 100,
                cap: 99,
                what: "J2K bounded row decode stripe buffer",
            })
        ));
    }
}
