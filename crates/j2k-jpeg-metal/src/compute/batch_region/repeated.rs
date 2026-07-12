// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::{
    batch, BatchedFastPacket, Error, MetalRuntime, Surface, REGION_SCALED_BATCH_CHUNK,
};
use super::common::decode_region_scaled_packet_surface;

#[cfg(target_os = "macos")]
fn requests_share_one_input(requests: &[batch::QueuedRequest]) -> bool {
    let Some(first) = requests.first() else {
        return false;
    };
    requests
        .iter()
        .all(|request| j2k_jpeg::adapter::SharedJpegInput::ptr_eq(&request.input, &first.input))
}

#[cfg(target_os = "macos")]
fn requests_share_one_region_scaled_work(requests: &[batch::QueuedRequest]) -> bool {
    let Some(first) = requests.first() else {
        return false;
    };
    requests_share_one_input(requests)
        && requests.iter().all(|request| {
            request.fmt == first.fmt && request.backend == first.backend && request.op == first.op
        })
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn try_decode_repeated_region_scaled_batch_to_surfaces(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    packets: &[BatchedFastPacket<'_>],
) -> Result<Option<Vec<Result<Surface, Error>>>, Error> {
    if requests.len() <= REGION_SCALED_BATCH_CHUNK
        || !requests_share_one_input(requests)
        || !requests
            .iter()
            .all(|request| matches!(request.op, batch::BatchOp::RegionScaled { .. }))
    {
        return Ok(None);
    }

    if requests_share_one_region_scaled_work(requests) {
        let surface = decode_region_scaled_packet_surface(runtime, &requests[0], &packets[0])?;
        let mut budget = crate::plan_owner_ledger::batch_execution_budget(
            "JPEG Metal repeated region-scaled results",
            requests,
        )?;
        return Ok(Some(budget.try_filled(
            requests.len(),
            Ok(surface),
            "JPEG Metal repeated region-scaled result slots",
        )?));
    }

    let mut budget = crate::plan_owner_ledger::batch_execution_budget(
        "JPEG Metal repeated-input region-scaled results",
        requests,
    )?;
    let mut results = budget.try_vec(
        requests.len(),
        "JPEG Metal repeated-input region-scaled result slots",
    )?;
    for (request, packet) in requests.iter().zip(packets.iter()) {
        results.push(decode_region_scaled_packet_surface(
            runtime, request, packet,
        ));
    }

    Ok(Some(results))
}
