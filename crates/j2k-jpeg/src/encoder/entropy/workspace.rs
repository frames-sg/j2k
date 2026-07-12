// SPDX-License-Identifier: MIT OR Apache-2.0

//! Exact host-live capacity for parallel restart entropy orchestration.

use core::mem::size_of;

use j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES;

use super::entropy_mcu_layout;
use super::restart::{
    parallel_entropy_chunk_count, restart_chunk_entropy_capacity, restart_chunk_segment_bounds,
};
use crate::adapter::{checked_encode_host_live_bytes, JpegBaselineSampling};
use crate::encoder::JpegEncodeError;

#[derive(Clone, Copy)]
pub(in crate::encoder) struct RestartEntropyWorkspacePlan {
    pub(in crate::encoder) chunk_count: usize,
    pub(in crate::encoder) chunk_capacity_bytes: usize,
    pub(in crate::encoder) result_metadata_bytes: usize,
    pub(in crate::encoder) output_capacity: usize,
}

impl RestartEntropyWorkspacePlan {
    pub(in crate::encoder) fn live_bytes(self) -> Result<usize, JpegEncodeError> {
        checked_encode_host_live_bytes([
            self.output_capacity,
            self.chunk_capacity_bytes,
            self.result_metadata_bytes,
        ])
    }
}

pub(in crate::encoder) fn entropy_host_workspace_bytes(
    width: u32,
    height: u32,
    sampling: JpegBaselineSampling,
    restart_interval: Option<u16>,
    entropy_capacity: usize,
) -> Result<usize, JpegEncodeError> {
    let Some(restart_interval) = restart_interval else {
        return checked_encode_host_live_bytes([entropy_capacity]);
    };
    if restart_interval == 0 {
        return Err(JpegEncodeError::InvalidRestartInterval);
    }

    let (_, total_mcus) = entropy_mcu_layout(width, height, sampling)?;
    if total_mcus == 0 {
        return Ok(0);
    }
    restart_entropy_workspace_plan(total_mcus, sampling, restart_interval, entropy_capacity)?
        .live_bytes()
}

pub(in crate::encoder) fn restart_entropy_workspace_plan(
    total_mcus: u32,
    sampling: JpegBaselineSampling,
    restart_interval: u16,
    entropy_capacity: usize,
) -> Result<RestartEntropyWorkspacePlan, JpegEncodeError> {
    if restart_interval == 0 {
        return Err(JpegEncodeError::InvalidRestartInterval);
    }
    let restart_interval = u32::from(restart_interval);
    let segment_count = total_mcus.div_ceil(restart_interval);
    let chunk_count = parallel_entropy_chunk_count(segment_count)?;
    let result_metadata_bytes = chunk_count
        .checked_mul(size_of::<super::restart::RestartChunkJob>())
        .ok_or_else(cap_overflow)?;
    let mut chunk_capacity_bytes = 0usize;
    for chunk_index in 0..chunk_count {
        let (start_segment, end_segment) =
            restart_chunk_segment_bounds(segment_count, chunk_index, chunk_count)?;
        let chunk_capacity = restart_chunk_entropy_capacity(
            total_mcus,
            restart_interval,
            start_segment,
            end_segment,
            sampling,
        )?;
        chunk_capacity_bytes =
            checked_encode_host_live_bytes([chunk_capacity_bytes, chunk_capacity])?;
    }

    let plan = RestartEntropyWorkspacePlan {
        chunk_count,
        chunk_capacity_bytes,
        result_metadata_bytes,
        output_capacity: entropy_capacity,
    };
    plan.live_bytes()?;
    Ok(plan)
}

fn cap_overflow() -> JpegEncodeError {
    JpegEncodeError::MemoryCapExceeded {
        requested: usize::MAX,
        cap: DEFAULT_MAX_HOST_ALLOCATION_BYTES,
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn workspace_module_stays_focused() {
        const SOURCE: &str = include_str!("workspace.rs");
        assert!(SOURCE.lines().count() <= 130);
    }
}
