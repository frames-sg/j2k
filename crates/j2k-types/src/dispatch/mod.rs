// SPDX-License-Identifier: MIT OR Apache-2.0

//! Low-level accelerator dispatch SPI and its reporting/error contracts.
//!
//! This is the expert integration surface for codec-stage accelerators. Stable
//! transform, Tier-1, and packetization value types live in their respective
//! ownership modules and are only referenced by this SPI.

mod accelerator;
mod error;
mod report;

pub use accelerator::{
    CpuOnlyJ2kEncodeStageAccelerator, J2kDeinterleaveMctToF32Job, J2kDeinterleaveToF32Job,
    J2kEncodeContext, J2kEncodeStageAccelerator, J2kHtj2kTileEncodeJob,
};
pub use error::{J2kEncodeStageError, J2kEncodeStageErrorKind, J2kEncodeStageResult};
pub use report::J2kEncodeDispatchReport;
