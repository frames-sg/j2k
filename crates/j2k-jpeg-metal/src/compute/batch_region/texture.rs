// SPDX-License-Identifier: MIT OR Apache-2.0

mod fast444;
mod subsampled;

pub(in crate::compute) use fast444::try_decode_fast444_region_scaled_rgba_batch_to_textures;
pub(in crate::compute) use subsampled::{
    try_decode_fast420_region_scaled_rgba_batch_to_textures,
    try_decode_fast422_region_scaled_rgba_batch_to_textures,
};
