// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::{
    batch, BatchedFastPacket, CpuDecoder, Error, MetalRuntime, Surface, REGION_SCALED_BATCH_CHUNK,
};
use super::common::decode_region_scaled_packet_surface;

#[cfg(target_os = "macos")]
fn requests_share_one_input(requests: &[batch::QueuedRequest]) -> bool {
    let Some(first) = requests.first() else {
        return false;
    };
    requests.iter().all(|request| {
        request.input.as_ptr() == first.input.as_ptr() && request.input.len() == first.input.len()
    })
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

    let decoder = CpuDecoder::new(requests[0].input.as_ref())?;
    if requests_share_one_region_scaled_work(requests) {
        let surface =
            decode_region_scaled_packet_surface(runtime, &decoder, &requests[0], &packets[0])?;
        return Ok(Some(
            (0..requests.len())
                .map(|_| Ok(surface.clone()))
                .collect::<Vec<_>>(),
        ));
    }

    let mut results = Vec::with_capacity(requests.len());
    for (request, packet) in requests.iter().zip(packets.iter()) {
        results.push(decode_region_scaled_packet_surface(
            runtime, &decoder, request, packet,
        ));
    }

    Ok(Some(results))
}
