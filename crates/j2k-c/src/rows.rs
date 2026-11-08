// SPDX-License-Identifier: MIT OR Apache-2.0

//! Row-by-row decode of the full image in the native layout of
//! `j2k_decoder_output_info`, without a whole-image output buffer.

#![forbid(unsafe_code)]

use crate::{
    abi::{
        J2kCDecodeOptions, J2kCOutputInfo, J2K_ALLOCATION_FAILED, J2K_I16_LE, J2K_STOPPED,
        J2K_U16_LE, J2K_U8, J2K_UNSUPPORTED,
    },
    decoder::{in_pool, warning_bits, Decoder},
    error::{Error, Result},
};
use j2k::{DecodeRequest, DecodeRowsError, J2kDecoder, J2kRowDecodeOptions, Rect, RowSink};

/// Rows a stripe holds when the caller passes zero.
const DEFAULT_STRIPE_ROWS: u32 = 64;

/// The caller's row consumer: `false` stops the decode.
pub(crate) type RowConsumer<'a> = &'a mut (dyn FnMut(u32, &[u8]) -> bool + Send);

impl Decoder {
    /// The layout `decode_rows` delivers: the full image at full resolution
    /// with the channels, sample type, precision and alpha a decode of it
    /// would write. No whole-image output limit applies, since rows are
    /// never held together.
    pub(crate) fn row_info(&mut self, options: J2kCDecodeOptions) -> Result<J2kCOutputInfo> {
        if options.reserved != 0 {
            return Err(Error::argument("reserved decode options must be zero"));
        }
        if [
            options.x,
            options.y,
            options.width,
            options.height,
            options.reduction,
        ] != [0; 5]
        {
            return Err(Error::new(
                J2K_UNSUPPORTED,
                "row decode covers the full image at full resolution",
            ));
        }
        // The codec classifies support and sample layout the same for any
        // region, so a one-pixel request yields the layout without planning
        // or limiting a whole-image output.
        let probe = self.prepare_layout(DecodeRequest::Region {
            roi: Rect {
                x: 0,
                y: 0,
                w: 1,
                h: 1,
            },
        })?;
        let sample_bytes = match (probe.sample_type, probe.precision) {
            (J2K_U8, 8) => 1,
            (J2K_U16_LE | J2K_I16_LE, 9..=16) => 2,
            _ => {
                return Err(Error::new(
                    J2K_UNSUPPORTED,
                    "row decode keeps native values only for 8-bit unsigned and 9- to 16-bit samples; use decode_into",
                ))
            }
        };
        let (width, height) = self.image_size();
        let row_bytes = u64::from(width) * u64::from(probe.channels) * sample_bytes;
        Ok(J2kCOutputInfo {
            x: 0,
            y: 0,
            width,
            height,
            row_bytes,
            buffer_bytes: row_bytes.saturating_mul(u64::from(height)),
            ..probe
        })
    }

    /// Decode the full image into `consumer` one row at a time, in order and
    /// once each, holding at most `stripe_rows` decoded rows at once.
    pub(crate) fn decode_rows(
        &mut self,
        options: J2kCDecodeOptions,
        stripe_rows: u32,
        consumer: RowConsumer<'_>,
    ) -> Result<J2kCOutputInfo> {
        let info = self.row_info(options)?;
        let row_bytes = usize::try_from(info.row_bytes)
            .map_err(|_| Error::limit("a row does not fit the host address space"))?;
        let rows = J2kRowDecodeOptions::new(if stripe_rows == 0 {
            DEFAULT_STRIPE_ROWS
        } else {
            stripe_rows
        });
        let (bytes, settings) = (&self.bytes, self.settings);
        let outcome = in_pool(self.pool.as_ref(), || {
            let mut decoder = J2kDecoder::new_with_settings(bytes, settings)?;
            let result = if info.sample_type == J2K_U8 {
                let mut sink = Rows8 {
                    row_bytes,
                    consumer,
                };
                decoder.decode_rows_u8_bounded(&mut sink, rows)
            } else {
                let mut sink = Rows16 {
                    row: le_row(row_bytes)?,
                    consumer,
                };
                decoder.decode_rows_u16_bounded(&mut sink, rows)
            };
            result.map_err(row_error)
        })?;
        if !outcome.warnings.iter().all(|warning| self.allows(*warning)) {
            return Err(Error::internal(
                "row decode reported a warning its settings do not allow",
            ));
        }
        Ok(J2kCOutputInfo {
            warnings: warning_bits(&outcome.warnings),
            ..info
        })
    }
}

/// A zeroed row of `row_bytes` for little-endian 16-bit samples.
fn le_row(row_bytes: usize) -> Result<Vec<u8>> {
    let mut row = Vec::new();
    row.try_reserve_exact(row_bytes)
        .map_err(|_| Error::new(J2K_ALLOCATION_FAILED, "could not allocate a row buffer"))?;
    row.resize(row_bytes, 0);
    Ok(row)
}

fn row_error(error: DecodeRowsError<j2k::J2kError, RowError>) -> Error {
    match error {
        DecodeRowsError::Decode(error) => Error::codec(&error),
        DecodeRowsError::Sink(RowError::Stopped(y)) => Error::new(
            J2K_STOPPED,
            format!("the row callback stopped the decode at row {y}"),
        ),
        DecodeRowsError::Sink(RowError::Layout) => {
            Error::internal("a decoded row disagrees with the row layout")
        }
    }
}

/// Why a row sink refused a row.
#[derive(Debug)]
enum RowError {
    /// The consumer asked to stop at this row.
    Stopped(u32),
    /// The codec handed over a row of another length than the layout's.
    Layout,
}

impl core::fmt::Display for RowError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Stopped(y) => write!(formatter, "the row callback stopped at row {y}"),
            Self::Layout => formatter.write_str("a decoded row disagrees with the row layout"),
        }
    }
}

impl core::error::Error for RowError {}

/// Hands 8-bit rows over as they are.
struct Rows8<'a> {
    row_bytes: usize,
    consumer: RowConsumer<'a>,
}

impl RowSink<u8> for Rows8<'_> {
    type Error = RowError;

    fn write_row(&mut self, y: u32, row: &[u8]) -> core::result::Result<(), RowError> {
        if row.len() != self.row_bytes {
            return Err(RowError::Layout);
        }
        if (self.consumer)(y, row) {
            Ok(())
        } else {
            Err(RowError::Stopped(y))
        }
    }
}

/// Hands 16-bit rows over as little-endian bytes. A signed image's samples
/// arrive as their two's-complement bit patterns, which are its I16 bytes.
struct Rows16<'a> {
    row: Vec<u8>,
    consumer: RowConsumer<'a>,
}

impl RowSink<u16> for Rows16<'_> {
    type Error = RowError;

    fn write_row(&mut self, y: u32, row: &[u16]) -> core::result::Result<(), RowError> {
        if row.len() * 2 != self.row.len() {
            return Err(RowError::Layout);
        }
        for (sample, bytes) in row.iter().zip(self.row.as_chunks_mut::<2>().0) {
            *bytes = sample.to_le_bytes();
        }
        if (self.consumer)(y, &self.row) {
            Ok(())
        } else {
            Err(RowError::Stopped(y))
        }
    }
}
