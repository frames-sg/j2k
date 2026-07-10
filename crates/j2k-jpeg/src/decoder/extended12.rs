// SPDX-License-Identifier: MIT OR Apache-2.0

//! Extended-precision JPEG decode routing.
//!
//! Focused submodules own sampling validation, entropy state, plane
//! construction, upsampling, output writers, and sequential/progressive paths.

mod planes;
mod progressive;
mod rgba;
mod sampling;
mod sequential;
mod state;
mod upsample;
mod writers;

pub(super) use self::sampling::lossless_color_sampling;
pub(super) use self::upsample::{upsample_h2v1_sample_at, upsample_h2v2_rows_at};

#[cfg(test)]
mod tests;
