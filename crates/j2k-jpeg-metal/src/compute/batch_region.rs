// SPDX-License-Identifier: MIT OR Apache-2.0

mod common;
mod repeated;
mod rgb;
mod texture;

pub(super) use repeated::try_decode_repeated_region_scaled_batch_to_surfaces;
pub(super) use rgb::{
    try_decode_fast420_region_scaled_rgb_batch_to_surfaces,
    try_decode_fast420_region_scaled_rgb_batch_to_surfaces_into_output,
    try_decode_fast422_region_scaled_rgb_batch_to_surfaces,
    try_decode_fast422_region_scaled_rgb_batch_to_surfaces_into_output,
    try_decode_fast444_region_scaled_rgb_batch_to_surfaces,
    try_decode_fast444_region_scaled_rgb_batch_to_surfaces_into_output,
    try_decode_fast_subsampled_region_scaled_rgb_batch_to_surfaces_with_output,
};
pub(super) use texture::{
    try_decode_fast420_region_scaled_rgba_batch_to_textures,
    try_decode_fast422_region_scaled_rgba_batch_to_textures,
    try_decode_fast444_region_scaled_rgba_batch_to_textures,
};
