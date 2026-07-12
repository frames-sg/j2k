// SPDX-License-Identifier: MIT OR Apache-2.0

mod conversion;
mod fast444;
mod grouped_output;
mod requests;
#[cfg(all(test, target_os = "macos"))]
mod split_coeff_idct;
mod subsampled;
mod surface;
mod texture;
mod texture_dispatch;

pub(super) use conversion::checked_u32;
pub(super) use fast444::{
    encode_fast444_batch_item, encode_fast444_region_batch_item, encode_fast444_scaled_batch_item,
    encode_fast444_scaled_region_batch_item,
};
pub(super) use grouped_output::{batch_output_buffer_or_new, copy_grouped_surfaces_to_output};
pub(super) use requests::{Fast444ScaledRegionBatchItemRequest, FastSubsampledOpBatchItemRequest};
#[cfg(test)]
pub(super) use split_coeff_idct::{encode_split_coeff_idct_passes, SplitCoeffIdctPasses};
pub(super) use subsampled::{
    encode_fast_subsampled_op_batch_item, encode_fast_subsampled_region_batch_item,
    encode_fast_subsampled_scaled_batch_item,
};
pub(super) use texture::{
    copy_rgb8_surfaces_to_rgba_textures, texture_batch_success_results,
    validate_rgba_texture_batch_output,
};
pub(super) use texture_dispatch::{
    dispatch_rgba_texture_pack, dispatch_windowed_rgba_texture_pack,
};
