// SPDX-License-Identifier: MIT OR Apache-2.0

//! Reporting which reductions of a partial codestream can be decoded.

use alloc::vec::Vec;

use super::J2kDecoder;
use crate::J2kError;

/// How much of a partial codestream is present, by resolution level.
///
/// Returned by [`J2kDecoder::codestream_completeness`]. A decode at a
/// reduction of [`Self::min_exact_reduction`] levels or more reads only
/// packets that are present, so it returns the same samples as the complete
/// codestream decoded at that reduction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct J2kCodestreamCompleteness {
    complete: bool,
    min_exact_reduction: Option<u8>,
    tile_min_exact_reduction: Vec<Option<u8>>,
}

impl J2kCodestreamCompleteness {
    /// Whether every packet of every tile is present. A codestream can be
    /// complete without its final EOC marker.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.complete
    }

    /// The fewest resolution levels to drop for a decode of the whole image
    /// that reads only packets that are present. `Some(0)` when every packet
    /// is present, `None` while some tile lacks part of its lowest resolution
    /// or would need a deeper reduction than every tile supports.
    ///
    /// Pass it as the reduction of a scaled decode, for example
    /// [`J2kDecoder::decode_region_scaled_pow2_into`] or
    /// [`J2kDecoder::region_decode_plan`].
    #[must_use]
    pub fn min_exact_reduction(&self) -> Option<u8> {
        self.min_exact_reduction
    }

    /// [`Self::min_exact_reduction`] for each tile, in raster order. A region
    /// that covers only tiles whose value is `Some(r)` or less decodes at
    /// reduction `r` even while other tiles are incomplete.
    #[must_use]
    pub fn tile_min_exact_reduction(&self) -> &[Option<u8>] {
        &self.tile_min_exact_reduction
    }
}

impl J2kDecoder<'_> {
    /// Report which reductions of this decoder's input decode exactly.
    ///
    /// Meant for decoders made with
    /// [`crate::DecodeSettings::with_partial_codestream`], whose input may be
    /// only the start of a codestream. This reads the packet headers of every
    /// tile, without decoding any samples. Call it again after more of the
    /// codestream arrives, on a decoder made from the longer input.
    ///
    /// # Errors
    /// Returns [`J2kError`] when a tile part that is present in full is
    /// malformed, a resource limit is reached, or, for a decoder without
    /// [`crate::DecodeSettings::with_partial_codestream`], when the input is
    /// cut off.
    pub fn codestream_completeness(&mut self) -> Result<J2kCodestreamCompleteness, J2kError> {
        // Packet geometry is that of the full-resolution image, whatever
        // reduction the cached image was parsed at.
        let image = crate::backend::image_for_operation(self.bytes, self.settings, 0, 0, None)?;
        let native = image
            .codestream_completeness()
            .map_err(J2kError::from_native_decode_error)?;
        Ok(J2kCodestreamCompleteness {
            complete: native.is_complete(),
            min_exact_reduction: native.min_exact_reduction(),
            tile_min_exact_reduction: native.into_tile_min_exact_reduction(),
        })
    }
}
