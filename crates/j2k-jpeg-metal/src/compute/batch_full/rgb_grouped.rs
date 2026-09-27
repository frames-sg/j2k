// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::scratch_pool::BatchScratchLease;
use super::super::{
    batch, batch_output_buffer_or_new, copy_grouped_surfaces_to_output, Error, FastBatchDecodeMode,
    FastSubsampledMetal, MetalRuntime, PixelFormat, Surface,
};
use super::ordered_grouped_results;
use super::rgb::{try_submit_compatible_full_rgb_batch, PendingFullRgbBatch};

struct PendingFullRgbGroup<'runtime, 'packet, P> {
    indices: Vec<usize>,
    requests: Vec<batch::QueuedRequest>,
    pending: PendingFullRgbBatch<'runtime, 'packet, P>,
}

impl<P: FastSubsampledMetal> PendingFullRgbGroup<'_, '_, P> {
    fn finish(
        mut self,
        runtime: &MetalRuntime,
        output: Option<&crate::MetalBatchOutputBuffer>,
        family_packets: &[&P],
        merged_results: &mut [Option<Result<Surface, Error>>],
        external_live_bytes: usize,
    ) -> Result<(), Error> {
        let mut completion_budget =
            crate::batch_allocation::BatchMetadataBudget::with_external_live(
                "JPEG Metal pending full RGB completion",
                external_live_bytes,
            );
        completion_budget.account_capacity::<usize>(self.indices.capacity())?;
        completion_budget.account_capacity::<batch::QueuedRequest>(self.requests.capacity())?;
        batch::stamp_execution_owner_baseline(
            &mut self.requests,
            0,
            completion_budget.live_bytes(),
        );
        let group_results = self.pending.finish(&self.requests, None)?;
        merge_group_results::<P>(
            runtime,
            output,
            family_packets,
            self.indices,
            group_results,
            merged_results,
            completion_budget.live_bytes(),
        )
    }
}

fn pending_full_rgb_group_live_bytes<P>(
    base_live_bytes: usize,
    pending_groups: &[PendingFullRgbGroup<'_, '_, P>],
) -> Result<usize, Error> {
    let mut budget = crate::batch_allocation::BatchMetadataBudget::with_external_live(
        "JPEG Metal live pending full RGB groups",
        base_live_bytes,
    );
    for group in pending_groups {
        budget.account_capacity::<usize>(group.indices.capacity())?;
        budget.account_capacity::<batch::QueuedRequest>(group.requests.capacity())?;
    }
    Ok(budget.live_bytes())
}

/// Waits for the oldest in-flight group and merges its results; the groups
/// still in flight stay charged to the completion budget.
#[cfg(target_os = "macos")]
fn finish_oldest_full_rgb_group<P: FastSubsampledMetal>(
    runtime: &MetalRuntime,
    output: Option<&crate::MetalBatchOutputBuffer>,
    family_packets: &[&P],
    result_live_bytes: usize,
    pending_groups: &mut Vec<PendingFullRgbGroup<'_, '_, P>>,
    merged_results: &mut [Option<Result<Surface, Error>>],
) -> Result<(), Error> {
    let group = pending_groups.remove(0);
    let external_live_bytes = pending_full_rgb_group_live_bytes(result_live_bytes, pending_groups)?;
    group.finish(
        runtime,
        output,
        family_packets,
        merged_results,
        external_live_bytes,
    )
}

/// Keeps at most two groups in flight and leases scratch for the next one.
/// Never waits for another lease while retaining one: concurrent batches may
/// each own one of the pool's two slots, so the oldest group finishes first.
#[cfg(target_os = "macos")]
fn lease_scratch_for_next_full_rgb_group<'runtime, P: FastSubsampledMetal>(
    runtime: &'runtime MetalRuntime,
    output: Option<&crate::MetalBatchOutputBuffer>,
    family_packets: &[&P],
    result_live_bytes: usize,
    pending_groups: &mut Vec<PendingFullRgbGroup<'runtime, '_, P>>,
    merged_results: &mut [Option<Result<Surface, Error>>],
) -> Result<BatchScratchLease<'runtime>, Error> {
    if pending_groups.len() == 2 {
        finish_oldest_full_rgb_group(
            runtime,
            output,
            family_packets,
            result_live_bytes,
            pending_groups,
            merged_results,
        )?;
    }
    if pending_groups.is_empty() {
        return runtime.batch_scratch();
    }
    if let Some(scratch) = runtime.try_batch_scratch()? {
        return Ok(scratch);
    }
    finish_oldest_full_rgb_group(
        runtime,
        output,
        family_packets,
        result_live_bytes,
        pending_groups,
        merged_results,
    )?;
    runtime.batch_scratch()
}

/// Checks (or grows, for a resizable output) the caller's buffer for every
/// packet shape before any group is submitted.
#[cfg(target_os = "macos")]
fn check_grouped_rgb_output<P: FastSubsampledMetal>(
    runtime: &MetalRuntime,
    output: &crate::MetalBatchOutputBuffer,
    family_packets: &[&P],
    tile_count: usize,
) -> Result<(), Error> {
    for packet in family_packets {
        let out_stride = packet.dimensions().0 as usize * PixelFormat::Rgb8.bytes_per_pixel();
        let out_tile_len = out_stride * packet.dimensions().1 as usize;
        batch_output_buffer_or_new(
            runtime,
            Some(output),
            packet.dimensions(),
            tile_count,
            out_stride,
            out_tile_len,
        )?;
    }
    Ok(())
}

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
        check_grouped_rgb_output(runtime, output, family_packets, requests.len())?;
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
    let mut pending_groups: Vec<PendingFullRgbGroup<'_, '_, P>> =
        result_budget.try_vec(2, "JPEG Metal pending full RGB groups")?;
    for group_indices in groups {
        let scratch = lease_scratch_for_next_full_rgb_group(
            runtime,
            output,
            family_packets,
            result_budget.live_bytes(),
            &mut pending_groups,
            &mut merged_results,
        )?;
        let pending_live_bytes =
            pending_full_rgb_group_live_bytes(result_budget.live_bytes(), &pending_groups)?;
        let mut group_budget = crate::batch_allocation::BatchMetadataBudget::with_external_live(
            "JPEG Metal grouped full RGB sub-batch",
            pending_live_bytes,
        );
        group_budget.account_capacity::<usize>(group_indices.capacity())?;
        let mut group_requests =
            group_budget.try_vec(group_indices.len(), "JPEG Metal grouped full RGB requests")?;
        group_requests.extend(group_indices.iter().map(|&index| requests[index].clone()));
        let mut group_packets =
            group_budget.try_vec(group_indices.len(), "JPEG Metal grouped full RGB packets")?;
        group_packets.extend(group_indices.iter().map(|&index| family_packets[index]));
        batch::stamp_execution_owner_baseline(&mut group_requests, 0, group_budget.live_bytes());

        let Some(pending) = try_submit_compatible_full_rgb_batch::<P>(
            runtime,
            &group_requests,
            &group_packets,
            decode_mode,
            None,
            scratch,
        )?
        else {
            return Ok(None);
        };
        pending_groups.push(PendingFullRgbGroup {
            indices: group_indices,
            requests: group_requests,
            pending,
        });
    }
    while !pending_groups.is_empty() {
        finish_oldest_full_rgb_group(
            runtime,
            output,
            family_packets,
            result_budget.live_bytes(),
            &mut pending_groups,
            &mut merged_results,
        )?;
    }

    ordered_grouped_results(
        &mut result_budget,
        merged_results,
        "JPEG Metal ordered grouped full RGB results",
        |index| {
            format!(
                "JPEG Metal grouped {} buffer result for tile {index} was missing",
                P::FAMILY_NAME
            )
        },
    )
    .map(Some)
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
