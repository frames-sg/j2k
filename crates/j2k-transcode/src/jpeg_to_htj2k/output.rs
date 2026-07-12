// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{BatchTranscodeReport, JpegToHtj2kError, TranscodeReport};

/// Encoded transcode output and validation/report metadata.
#[derive(Debug)]
pub struct EncodedTranscode {
    /// HTJ2K codestream bytes.
    pub codestream: Vec<u8>,
    /// Summary of the experimental path used.
    pub report: TranscodeReport,
}

/// One JPEG tile input for batch transcode.
#[derive(Debug, Clone, Copy)]
pub struct JpegTileBatchInput<'a> {
    /// JPEG codestream bytes for one tile.
    pub bytes: &'a [u8],
}

/// Batch transcode output. Tile-level parse/encode failures are preserved so a
/// WSI ingest queue can continue past isolated bad tiles.
#[derive(Debug)]
pub struct EncodedTranscodeBatch {
    /// Per-input tile result in input order.
    pub tiles: Vec<Result<EncodedTranscode, JpegToHtj2kError>>,
    /// Aggregate batch report.
    pub report: BatchTranscodeReport,
}
