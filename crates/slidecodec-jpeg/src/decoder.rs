// SPDX-License-Identifier: Apache-2.0

//! Public [`Decoder`] entry points. M1a exposes [`Decoder::inspect`] only;
//! [`Decoder::new`] and the decode methods land in M1b.

use crate::error::JpegError;
use crate::info::Info;
use crate::parse::header::parse_header;

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
