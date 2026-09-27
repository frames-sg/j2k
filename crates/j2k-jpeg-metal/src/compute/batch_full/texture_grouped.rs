// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::scratch_pool::BatchScratchLease;
use super::super::{
    batch, validate_rgba_texture_batch_output, Error, FastBatchDecodeMode, FastSubsampledMetal,
    MetalRuntime, PixelFormat, PlaneMode,
};
use super::ordered_grouped_results;
use super::texture::{try_submit_compatible_full_rgba_texture_batch, PendingTextureBatch};

struct PendingTextureGroup<'runtime> {
    indices: Vec<usize>,
    requests: Vec<batch::QueuedRequest>,
    output: crate::MetalBatchTextureOutput,
    pending: PendingTextureBatch<'runtime>,
}

impl PendingTextureGroup<'_> {
    fn account_retained_metadata(
        &self,
        budget: &mut crate::batch_allocation::BatchMetadataBudget,
    ) -> Result<(), Error> {
        budget.account_capacity::<usize>(self.indices.capacity())?;
        budget.account_capacity::<batch::QueuedRequest>(self.requests.capacity())?;
        self.output.account_texture_handle_capacity(budget)
    }

    fn finish<P: FastSubsampledMetal>(
        mut self,
        merged: &mut [Option<Result<crate::MetalTextureTile, Error>>],
        external_live_bytes: usize,
    ) -> Result<(), Error> {
        batch::stamp_execution_owner_baseline(&mut self.requests, 0, external_live_bytes);
        let results = self.pending.finish(&self.requests, &self.output)?;
        if results.len() != self.indices.len() {
            return Err(Error::MetalKernel {
                message: format!(
                    "JPEG Metal grouped {} texture result count mismatch",
                    P::FAMILY_NAME
                ),
            });
        }
        for (index, result) in self.indices.into_iter().zip(results) {
            merged[index] = Some(result);
        }
        Ok(())
    }
}

fn pending_texture_metadata_budget(
    phase: &'static str,
    result_live_bytes: usize,
    pending_groups: &[PendingTextureGroup<'_>],
) -> Result<crate::batch_allocation::BatchMetadataBudget, Error> {
    let mut budget =
        crate::batch_allocation::BatchMetadataBudget::with_external_live(phase, result_live_bytes);
    budget.preflight(&[])?;
    for group in pending_groups {
        group.account_retained_metadata(&mut budget)?;
    }
    Ok(budget)
}

fn finish_oldest_pending_texture_group<P: FastSubsampledMetal>(
    pending_groups: &mut Vec<PendingTextureGroup<'_>>,
    merged: &mut [Option<Result<crate::MetalTextureTile, Error>>],
    result_live_bytes: usize,
) -> Result<(), Error> {
    let finish_budget = pending_texture_metadata_budget(
        "JPEG Metal pending full texture group completion",
        result_live_bytes,
        pending_groups,
    )?;
    pending_groups
        .remove(0)
        .finish::<P>(merged, finish_budget.live_bytes())
}

/// Keeps at most two groups in flight and leases scratch for the next one.
/// Never waits for another lease while retaining one: concurrent batches may
/// each own one of the pool's two slots, so the oldest group finishes first.
#[cfg(target_os = "macos")]
fn lease_scratch_for_next_texture_group<'runtime, P: FastSubsampledMetal>(
    runtime: &'runtime MetalRuntime,
    pending_groups: &mut Vec<PendingTextureGroup<'runtime>>,
    merged: &mut [Option<Result<crate::MetalTextureTile, Error>>],
    result_live_bytes: usize,
) -> Result<BatchScratchLease<'runtime>, Error> {
    if pending_groups.len() == 2 {
        finish_oldest_pending_texture_group::<P>(pending_groups, merged, result_live_bytes)?;
    }
    if pending_groups.is_empty() {
        return runtime.batch_scratch();
    }
    if let Some(scratch) = runtime.try_batch_scratch()? {
        return Ok(scratch);
    }
    finish_oldest_pending_texture_group::<P>(pending_groups, merged, result_live_bytes)?;
    runtime.batch_scratch()
}

/// Checks the caller's texture set against every packet shape before any group
/// is submitted.
#[cfg(target_os = "macos")]
fn check_grouped_texture_output<P: FastSubsampledMetal>(
    output: &crate::MetalBatchTextureOutput,
    family_packets: &[&P],
    tile_count: usize,
) -> Result<(), Error> {
    for packet in family_packets {
        let out_stride = packet.dimensions().0 as usize * PixelFormat::Rgba8.bytes_per_pixel();
        let out_tile_len = out_stride * packet.dimensions().1 as usize;
        validate_rgba_texture_batch_output(output, packet.dimensions(), tile_count, out_tile_len)?;
    }
    Ok(())
}

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
    check_grouped_texture_output(output, family_packets, requests.len())?;

    let mut result_budget = crate::plan_owner_ledger::batch_execution_budget(
        "JPEG Metal grouped full texture results",
        requests,
    )?;
    let mut merged_results = result_budget.try_filled(
        requests.len(),
        None,
        "JPEG Metal grouped full texture result slots",
    )?;
    let mut pending_groups: Vec<PendingTextureGroup<'_>> =
        result_budget.try_vec(2, "JPEG Metal pending full texture groups")?;
    for group_indices in groups {
        let scratch = lease_scratch_for_next_texture_group::<P>(
            runtime,
            &mut pending_groups,
            &mut merged_results,
            result_budget.live_bytes(),
        )?;
        let mut retained_budget = pending_texture_metadata_budget(
            "JPEG Metal grouped full texture retained sub-batch",
            result_budget.live_bytes(),
            &pending_groups,
        )?;
        retained_budget.account_capacity::<usize>(group_indices.capacity())?;
        retained_budget.preflight(&[
            crate::batch_allocation::BatchMetadataRequest::of::<crate::metal_types::Texture>(
                group_indices.len(),
            ),
            crate::batch_allocation::BatchMetadataRequest::of::<batch::QueuedRequest>(
                group_indices.len(),
            ),
        ])?;
        let group_output = output.clone_slots(&group_indices)?;
        group_output.account_texture_handle_capacity(&mut retained_budget)?;
        let mut group_requests = retained_budget.try_vec(
            group_indices.len(),
            "JPEG Metal grouped full texture requests",
        )?;
        group_requests.extend(group_indices.iter().map(|&index| requests[index].clone()));
        let mut submission_budget =
            crate::batch_allocation::BatchMetadataBudget::with_external_live(
                "JPEG Metal grouped full texture submission",
                retained_budget.live_bytes(),
            );
        let mut group_packets = submission_budget.try_vec(
            group_indices.len(),
            "JPEG Metal grouped full texture packets",
        )?;
        group_packets.extend(group_indices.iter().map(|&index| family_packets[index]));
        batch::stamp_execution_owner_baseline(
            &mut group_requests,
            0,
            submission_budget.live_bytes(),
        );

        let Some(pending) = try_submit_compatible_full_rgba_texture_batch::<P>(
            runtime,
            &group_requests,
            &group_packets,
            family_modes[group_indices[0]],
            &group_output,
            decode_mode,
            scratch,
        )?
        else {
            return Ok(None);
        };
        drop(group_packets);
        batch::stamp_execution_owner_baseline(&mut group_requests, 0, retained_budget.live_bytes());
        pending_groups.push(PendingTextureGroup {
            indices: group_indices,
            requests: group_requests,
            output: group_output,
            pending,
        });
    }
    while !pending_groups.is_empty() {
        finish_oldest_pending_texture_group::<P>(
            &mut pending_groups,
            &mut merged_results,
            result_budget.live_bytes(),
        )?;
    }

    ordered_grouped_results(
        &mut result_budget,
        merged_results,
        "JPEG Metal ordered grouped full texture results",
        |index| {
            format!(
                "JPEG Metal grouped {} texture result for tile {index} was missing",
                P::FAMILY_NAME
            )
        },
    )
    .map(Some)
}
