// SPDX-License-Identifier: MIT OR Apache-2.0

//! Public batch source, operation, request, and target contracts.

use crate::{Decoder, MetalBatchOutputBuffer, MetalBatchTextureOutput};
use j2k_core::{Downscale, Rect};

/// Inputs for a batched RGB8 Metal decode.
#[derive(Clone, Copy)]
pub enum Rgb8MetalBatchSource<'a, 'b> {
    /// Raw JPEG byte streams, parsed per call.
    Bytes(&'a [&'a [u8]]),
    /// Already parsed decoders with cached Metal fast-packet state.
    Decoders(&'a [&'a Decoder<'b>]),
}

impl Rgb8MetalBatchSource<'_, '_> {
    pub(super) fn is_empty(&self) -> bool {
        match self {
            Self::Bytes(inputs) => inputs.is_empty(),
            Self::Decoders(decoders) => decoders.is_empty(),
        }
    }
}

/// Geometry operation applied to every tile of a batched RGB8 Metal decode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rgb8MetalBatchOp {
    /// Full-tile decode at native dimensions.
    Full,
    /// Whole-tile downscale.
    Scaled(Downscale),
    /// Scaled decode of one region shared by every tile.
    RegionScaled {
        /// Region of interest to decode from every source tile.
        roi: Rect,
        /// Downscale factor applied to the selected region.
        scale: Downscale,
    },
}

/// A batched RGB8 Metal decode request.
#[derive(Clone, Copy)]
pub struct Rgb8MetalBatchRequest<'a, 'b> {
    /// Source JPEG bytes or prepared decoders for the batch.
    pub source: Rgb8MetalBatchSource<'a, 'b>,
    /// Geometry operation applied to each source tile.
    pub op: Rgb8MetalBatchOp,
}

/// Caller-owned Metal buffer target for a batched RGB8 decode.
pub enum MetalBufferBatchTarget<'a> {
    /// Reuse the buffer as-is; its shape must already fit the batch.
    Reusable(&'a MetalBatchOutputBuffer),
    /// Grow the buffer to fit the batch before decoding.
    Resizable(&'a mut MetalBatchOutputBuffer),
}

/// Caller-owned Metal RGBA8 texture target for a batched RGB8 decode.
pub enum MetalTextureBatchTarget<'a> {
    /// Reuse the texture set as-is; its shape must already fit the batch.
    Reusable(&'a MetalBatchTextureOutput),
    /// Grow the texture set to fit the batch before decoding.
    Resizable(&'a mut MetalBatchTextureOutput),
}
