// SPDX-License-Identifier: MIT OR Apache-2.0

//! Metal codec decode followed by an explicit staged Burn upload.

mod batch;

pub use batch::{MetalUploadBurnDecoder, SubmittedMetalUploadBurnBatch};

/// Compatibility alias for [`MetalUploadBurnDecoder`].
///
/// This alias has the same explicitly staged behavior; it does not restore the
/// former direct-destination implementation.
#[deprecated(
    since = "0.7.6",
    note = "use MetalUploadBurnDecoder to make staging explicit"
)]
pub type MetalBurnDecoder = MetalUploadBurnDecoder;

/// Compatibility alias for [`SubmittedMetalUploadBurnBatch`].
#[deprecated(
    since = "0.7.6",
    note = "use SubmittedMetalUploadBurnBatch to make staging explicit"
)]
pub type SubmittedMetalBurnBatch = SubmittedMetalUploadBurnBatch;
