// SPDX-License-Identifier: MIT OR Apache-2.0

//! Public adapter-facing JPEG 2000 adapter contracts.

#[cfg(test)]
mod adaptive_route;

/// Device decode request normalization.
pub(crate) mod device_plan;

/// Encode-stage adapter contracts.
pub(crate) mod encode_stage;
