// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::{
    batch, batch_output_buffer_or_new, copy_grouped_surfaces_to_output, Error, FastBatchDecodeMode,
    FastSubsampledMetal, MetalRuntime, PixelFormat, Surface,
};
use super::rgb::try_decode_fast_subsampled_full_rgb_batch_to_surfaces_with_mode_and_output;

#[cfg(target_os = "macos")]
pub(super) fn try_decode_grouped_fast_subsampled_full_rgb_batch_to_surfaces_with_output<
    P: FastSubsampledMetal,
>(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    family_packets: &[&P],
    decode_mode: FastBatchDecodeMode,
    output: Option<&crate::MetalBatchOutputBuffer>,
    groups: Vec<Vec<usize>>,
) -> Result<Option<Vec<Result<Surface, Error>>>, Error> {
    if let Some(output) = output {
        for packet in family_packets {
            let out_stride = packet.dimensions().0 as usize * PixelFormat::Rgb8.bytes_per_pixel();
            let out_tile_len = out_stride * packet.dimensions().1 as usize;
            batch_output_buffer_or_new(
                runtime,
                Some(output),
                packet.dimensions(),
                requests.len(),
                out_stride,
                out_tile_len,
            )?;
        }
    }

    let mut result_budget = crate::plan_owner_ledger::batch_execution_budget(
        "JPEG Metal grouped full RGB results",
        requests,
    )?;
    let mut merged_results = result_budget.try_filled(
        requests.len(),
        None,
        "JPEG Metal grouped full RGB result slots",
    )?;
    for group_indices in groups {
        let mut group_budget = crate::plan_owner_ledger::batch_execution_budget(
            "JPEG Metal grouped full RGB sub-batch",
            requests,
        )?;
        let mut group_requests =
            group_budget.try_vec(group_indices.len(), "JPEG Metal grouped full RGB requests")?;
        group_requests.extend(group_indices.iter().map(|&index| requests[index].clone()));
        let mut group_packets =
            group_budget.try_vec(group_indices.len(), "JPEG Metal grouped full RGB packets")?;
        group_packets.extend(
            group_indices
                .iter()
                .map(|&index| family_packets[index].to_batched()),
        );
        batch::stamp_execution_owner_baseline(&mut group_requests, 0, group_budget.live_bytes());

        let Some(group_results) =
            try_decode_fast_subsampled_full_rgb_batch_to_surfaces_with_mode_and_output::<P>(
                runtime,
                &group_requests,
                &group_packets,
                decode_mode,
                None,
            )?
        else {
            return Ok(None);
        };

        merge_group_results::<P>(
            runtime,
            output,
            family_packets,
            group_indices,
            group_results,
            &mut merged_results,
            result_budget.live_bytes(),
        )?;
    }

    let mut results = result_budget.try_vec(
        requests.len(),
        "JPEG Metal ordered grouped full RGB results",
    )?;
    for (index, result) in merged_results.into_iter().enumerate() {
        results.push(result.ok_or_else(|| Error::MetalKernel {
            message: format!(
                "JPEG Metal grouped {} buffer result for tile {index} was missing",
                P::FAMILY_NAME
            ),
        })?);
    }
    Ok(Some(results))
}

#[cfg(target_os = "macos")]
fn merge_group_results<P: FastSubsampledMetal>(
    runtime: &MetalRuntime,
    output: Option<&crate::MetalBatchOutputBuffer>,
    family_packets: &[&P],
    group_indices: Vec<usize>,
    group_results: Vec<Result<Surface, Error>>,
    merged_results: &mut [Option<Result<Surface, Error>>],
    external_live_bytes: usize,
) -> Result<(), Error> {
    if let Some(output) = output {
        let Some(&first_group_index) = group_indices.first() else {
            return Ok(());
        };
        let packet = family_packets[first_group_index];
        let out_stride = packet.dimensions().0 as usize * PixelFormat::Rgb8.bytes_per_pixel();
        let out_tile_len = out_stride * packet.dimensions().1 as usize;
        for (original_index, result) in copy_grouped_surfaces_to_output(
            runtime,
            output,
            packet.dimensions(),
            out_tile_len,
            &group_indices,
            group_results,
            external_live_bytes,
        )? {
            merged_results[original_index] = Some(result);
        }
        return Ok(());
    }

    if group_results.len() != group_indices.len() {
        return Err(Error::MetalKernel {
            message: format!(
                "JPEG Metal grouped {} buffer result count mismatch",
                P::FAMILY_NAME
            ),
        });
    }
    for (original_index, result) in group_indices.into_iter().zip(group_results) {
        merged_results[original_index] = Some(result);
    }
    Ok(())
}
