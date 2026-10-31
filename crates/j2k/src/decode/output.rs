// SPDX-License-Identifier: MIT OR Apache-2.0

//! Pixel-layout conversion for decoded native output.

mod native;
mod u16;
mod u8;

pub(crate) use native::{
    decode_native_components_with_native_context, native_components_route,
    native_region_components_route,
};
use native::{planes_pack_floor, region_rows_preferred};

pub(super) use u16::{
    region_rows_preferred_u16, rows_match_native_u16, u16_rows_pack_floor,
    write_components_u16_output, write_u16_output, U16RowSink,
};
pub(super) use u8::{
    can_decode_u8_directly, expand_rgb8_to_rgba8, write_components_u8_output, write_u8_output,
    U8RowSink,
};
