// SPDX-License-Identifier: MIT OR Apache-2.0

//! Declarative Annex D/F encoder test scope and inventory validation.

mod ics;
mod matrix;
mod model;

pub use ics::EncoderIcs;
#[cfg(feature = "runner")]
pub(crate) use ics::{ics_path, matrix_path, reference_decoder_identity};
pub use matrix::{EncoderMatrix, EncoderMatrixError};
#[cfg(feature = "runner")]
pub(crate) use model::EncoderPattern;
use model::TABLE_F1_MARKERS;
pub use model::{
    EncoderBlockCoding, EncoderCase, EncoderIut, EncoderMarker, EncoderMode, EncoderOperation,
    EncoderPayload, EncoderProgression, EncoderReferenceDecoder,
};
pub(crate) use model::{EncoderInputKind, EncoderPairwiseScope, EncoderRateTarget};
