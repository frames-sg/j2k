// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::{
    batch, validate_rgba_texture_batch_output, Error, FastBatchDecodeMode, FastSubsampledMetal,
    MetalRuntime, PixelFormat, PlaneMode,
};
use super::texture::try_decode_fast_subsampled_full_rgba_batch_to_textures;

#[cfg(target_os = "macos")]
pub(super) fn try_decode_grouped_fast_subsampled_full_rgba_batch_to_textures<
    P: FastSubsampledMetal,
>(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    family_packets: &[&P],
    family_modes: &[PlaneMode],
    output: &crate::MetalBatchTextureOutput,
    decode_mode: FastBatchDecodeMode,
    groups: Vec<Vec<usize>>,
) -> Result<Option<Vec<Result<crate::MetalTextureTile, Error>>>, Error> {
    for packet in family_packets {
        let out_stride = packet.dimensions().0 as usize * PixelFormat::Rgba8.bytes_per_pixel();
        let out_tile_len = out_stride * packet.dimensions().1 as usize;
        validate_rgba_texture_batch_output(
            output,
            packet.dimensions(),
            requests.len(),
            out_tile_len,
        )?;
    }

    let mut merged_results: Vec<Option<Result<crate::MetalTextureTile, Error>>> =
        (0..requests.len()).map(|_| None).collect();
    for group_indices in groups {
        let group_output = output.clone_slots(&group_indices)?;
        let group_requests = group_indices
            .iter()
            .map(|&index| requests[index].clone())
            .collect::<Vec<_>>();
        let group_packets = group_indices
            .iter()
            .map(|&index| family_packets[index].to_batched_with_texture_mode(family_modes[index]))
            .collect::<Vec<_>>();

        let Some(group_results) = try_decode_fast_subsampled_full_rgba_batch_to_textures::<P>(
            runtime,
            &group_requests,
            &group_packets,
            &group_output,
            decode_mode,
        )?
        else {
            return Ok(None);
        };
        if group_results.len() != group_indices.len() {
            return Err(Error::MetalKernel {
                message: format!(
                    "JPEG Metal grouped {} texture result count mismatch",
                    P::FAMILY_NAME
                ),
            });
        }
        for (original_index, result) in group_indices.into_iter().zip(group_results) {
            merged_results[original_index] = Some(result);
        }
    }

    let mut results = Vec::with_capacity(requests.len());
    for (index, result) in merged_results.into_iter().enumerate() {
        results.push(result.ok_or_else(|| Error::MetalKernel {
            message: format!(
                "JPEG Metal grouped {} texture result for tile {index} was missing",
                P::FAMILY_NAME
            ),
        })?);
    }
    Ok(Some(results))
}
