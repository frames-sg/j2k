// SPDX-License-Identifier: Apache-2.0

//! Public [`Decoder`] entry points. M1a exposes [`Decoder::inspect`] only;
//! [`Decoder::new`] and the decode methods land in M1b.

use crate::error::JpegError;
use crate::error::Warning;
use crate::info::Info;
use crate::info::Rect;
use crate::parse::header::parse_header;
use alloc::vec::Vec;

/// Non-fatal outcome of a successful decode. See spec Section 2.
///
/// `DecodeOutcome` lives on `decoder.rs` rather than `info.rs` because it
/// carries `Warning` values from `error.rs`, and moving it into `info` would
/// create a `info → error` cycle (see `info.rs` header note).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodeOutcome {
    /// The rectangle actually written to the output buffer. For `decode_into`
    /// this is always `Rect::full(info.dimensions)`; later milestones add
    /// `decode_region_into` which can return a narrower rect.
    pub decoded: Rect,
    /// Warnings emitted during parse or decode. Empty when the stream is
    /// syntactically clean and every capability was exercised without fallback.
    pub warnings: Vec<Warning>,
}

/// A borrowed view of a JPEG stream. In M1a this type only supports
/// header-only inspection; later milestones add full decode methods.
pub struct Decoder<'a> {
    _bytes: &'a [u8],
}

impl<'a> Decoder<'a> {
    /// Parse the headers without decoding any pixels. Cheap — O(header size).
    ///
    /// # Errors
    /// Returns any structural, unsupported-SOF, or sanity-check error
    /// encountered before the Start-of-Scan marker. See [`JpegError`].
    pub fn inspect(input: &'a [u8]) -> Result<Info, JpegError> {
        parse_header(input).map(|h| h.info())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Warning;
    use crate::info::Rect;
    use alloc::vec;

    #[test]
    fn decode_outcome_carries_decoded_rect_and_warnings() {
        let outcome = DecodeOutcome {
            decoded: Rect { x: 0, y: 0, w: 32, h: 16 },
            warnings: vec![Warning::MissingEoi],
        };
        assert_eq!(outcome.decoded.w, 32);
        assert_eq!(outcome.decoded.h, 16);
        assert_eq!(outcome.warnings.len(), 1);
    }

    #[test]
    fn decode_outcome_defaults_to_empty_warnings() {
        let outcome = DecodeOutcome {
            decoded: Rect::full((8, 8)),
            warnings: Vec::new(),
        };
        assert!(outcome.warnings.is_empty());
    }
}
