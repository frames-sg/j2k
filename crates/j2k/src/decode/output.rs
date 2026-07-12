// SPDX-License-Identifier: MIT OR Apache-2.0

//! Pixel-layout conversion for decoded native output.

mod u16;
mod u8;

pub(super) use u16::write_u16_output;
pub(super) use u8::{can_decode_u8_directly, write_components_u8_output, write_u8_output};
