// SPDX-License-Identifier: MIT OR Apache-2.0

mod fast444;
mod rgb;
mod texture;
mod texture_grouped;

pub(super) use fast444::{
    try_decode_fast444_full_rgb_batch_to_surfaces,
    try_decode_fast444_full_rgb_batch_to_surfaces_into_output,
    try_decode_fast444_full_rgba_batch_to_textures,
};
#[cfg(test)]
pub(super) use rgb::try_decode_fast_subsampled_full_rgb_batch_to_surfaces_with_mode_and_output;
pub(super) use rgb::{
    try_decode_fast_subsampled_full_rgb_batch_to_surfaces,
    try_decode_fast_subsampled_full_rgb_batch_to_surfaces_into_output,
};
pub(super) use texture::try_decode_fast_subsampled_full_rgba_batch_to_textures;
