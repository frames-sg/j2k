// SPDX-License-Identifier: MIT OR Apache-2.0

//! Metal codec decode followed by an explicit staged Burn upload.

mod batch;

pub use batch::{MetalBurnDecoder, SubmittedMetalBurnBatch};

/// Explicit name for [`MetalBurnDecoder`]'s staged host-memory upload behavior.
///
/// This alias has the same explicitly staged behavior; it does not restore the
/// former direct-destination implementation.
pub type MetalUploadBurnDecoder = MetalBurnDecoder;

/// Explicit name for the staged upload represented by [`SubmittedMetalBurnBatch`].
pub type SubmittedMetalUploadBurnBatch = SubmittedMetalBurnBatch;
