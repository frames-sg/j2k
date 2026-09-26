// SPDX-License-Identifier: MIT OR Apache-2.0

mod fast444;
mod rgb;
mod rgb_grouped;
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

/// Unwraps per-tile results merged from grouped sub-batches, in request order.
/// A slot no group filled is an internal error naming the tile.
#[cfg(target_os = "macos")]
fn ordered_grouped_results<T>(
    budget: &mut crate::batch_allocation::BatchMetadataBudget,
    merged: Vec<Option<Result<T, super::Error>>>,
    phase: &'static str,
    missing: impl Fn(usize) -> String,
) -> Result<Vec<Result<T, super::Error>>, super::Error> {
    let mut results = budget.try_vec(merged.len(), phase)?;
    for (index, result) in merged.into_iter().enumerate() {
        results.push(result.ok_or_else(|| super::Error::MetalKernel {
            message: missing(index),
        })?);
    }
    Ok(results)
}
