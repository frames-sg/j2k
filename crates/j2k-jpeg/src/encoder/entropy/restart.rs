// SPDX-License-Identifier: MIT OR Apache-2.0

//! Restart-segment chunk planning, preallocation, and parallel fill.

use alloc::borrow::Cow;
use alloc::vec::Vec;

use j2k_core::try_host_vec_with_capacity;
use rayon::prelude::*;

use crate::adapter::{
    checked_encode_host_live_bytes, jpeg_baseline_entropy_capacity_for_mcus,
    JpegBaselineHuffmanTable, JpegBaselineSampling,
};
use crate::encoded_output::{checked_jpeg_baseline_frame_capacity, CappedBytes};

use super::workspace::restart_entropy_workspace_plan;
use super::{encode_entropy_mcu_range_into, entropy_mcu_layout, BitWriter, JpegEncodeError};

pub(in crate::encoder) const MAX_PARALLEL_ENTROPY_CHUNKS: usize = 64;

pub(super) struct RestartChunkJob {
    start_segment: u32,
    end_segment: u32,
    writer: BitWriter,
    error: Option<JpegEncodeError>,
}

#[derive(Clone, Copy)]
struct RestartJobPlan {
    total_mcus: u32,
    segment_count: u32,
    restart_interval: u32,
    sampling: JpegBaselineSampling,
    external_live_bytes: usize,
    output_capacity: usize,
    chunk_count: usize,
}

struct PreparedRestartJobs {
    jobs: Vec<RestartChunkJob>,
    metadata_bytes: usize,
    chunk_live_bytes: usize,
}

#[expect(
    clippy::too_many_arguments,
    reason = "private JPEG entropy hot path keeps scalar arguments for optimized codegen"
)]
pub(in crate::encoder) fn encode_entropy_restart_segments(
    planes: &[Cow<'_, [u8]>],
    width: u32,
    height: u32,
    sampling: JpegBaselineSampling,
    q_luma: &[u8; 64],
    q_chroma: &[u8; 64],
    dc_tables: [&JpegBaselineHuffmanTable; 2],
    ac_tables: [&JpegBaselineHuffmanTable; 2],
    cosine: &[[f64; 8]; 8],
    restart_interval: u16,
    entropy_capacity: usize,
    external_live_bytes: usize,
) -> Result<Vec<u8>, JpegEncodeError> {
    if restart_interval == 0 {
        return Err(JpegEncodeError::InvalidRestartInterval);
    }
    let (mcus_per_row, total_mcus) = entropy_mcu_layout(width, height, sampling)?;
    if total_mcus == 0 {
        return Ok(Vec::new());
    }
    checked_jpeg_baseline_frame_capacity(entropy_capacity)?;
    let restart_interval_u32 = u32::from(restart_interval);
    let segment_count = total_mcus.div_ceil(restart_interval_u32);
    let workspace =
        restart_entropy_workspace_plan(total_mcus, sampling, restart_interval, entropy_capacity)?;
    checked_encode_host_live_bytes([external_live_bytes, workspace.live_bytes()?])?;
    let PreparedRestartJobs {
        mut jobs,
        metadata_bytes: job_metadata_bytes,
        chunk_live_bytes,
    } = prepare_restart_jobs(RestartJobPlan {
        total_mcus,
        segment_count,
        restart_interval: restart_interval_u32,
        sampling,
        external_live_bytes,
        output_capacity: workspace.output_capacity,
        chunk_count: workspace.chunk_count,
    })?;

    jobs.par_iter_mut().for_each(|job| {
        job.error = encode_entropy_restart_chunk_into(
            planes,
            width,
            height,
            sampling,
            q_luma,
            q_chroma,
            dc_tables,
            ac_tables,
            cosine,
            mcus_per_row,
            total_mcus,
            restart_interval_u32,
            job.start_segment,
            job.end_segment,
            &mut job.writer,
        )
        .err();
    });

    let mut out = CappedBytes::try_with_capacity(entropy_capacity, entropy_capacity)?;
    checked_encode_host_live_bytes([
        external_live_bytes,
        job_metadata_bytes,
        chunk_live_bytes,
        out.capacity(),
    ])?;
    for job in jobs {
        if let Some(error) = job.error {
            return Err(error);
        }
        let chunk = job.writer.into_bytes()?;
        out.extend_from_slice(&chunk)?;
    }
    Ok(out.into_vec())
}

fn prepare_restart_jobs(plan: RestartJobPlan) -> Result<PreparedRestartJobs, JpegEncodeError> {
    let mut jobs: Vec<RestartChunkJob> =
        try_host_vec_with_capacity(plan.chunk_count).map_err(|error| {
            JpegEncodeError::HostAllocationFailed {
                bytes: error.requested_bytes(),
            }
        })?;
    let metadata_bytes = jobs
        .capacity()
        .checked_mul(core::mem::size_of::<RestartChunkJob>())
        .ok_or(JpegEncodeError::MemoryCapExceeded {
            requested: usize::MAX,
            cap: j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        })?;
    let mut chunk_live_bytes = 0usize;
    for chunk_index in 0..plan.chunk_count {
        let (start_segment, end_segment) =
            restart_chunk_segment_bounds(plan.segment_count, chunk_index, plan.chunk_count)?;
        let chunk_capacity = restart_chunk_entropy_capacity(
            plan.total_mcus,
            plan.restart_interval,
            start_segment,
            end_segment,
            plan.sampling,
        )?;
        checked_restart_chunk_preallocation_live_bytes(
            plan.external_live_bytes,
            plan.output_capacity,
            metadata_bytes,
            chunk_live_bytes,
            chunk_capacity,
        )?;
        let writer = BitWriter::try_with_max_bytes(chunk_capacity)?;
        chunk_live_bytes = chunk_live_bytes
            .checked_add(writer.capacity_bytes())
            .ok_or(JpegEncodeError::MemoryCapExceeded {
                requested: usize::MAX,
                cap: j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
            })?;
        checked_restart_chunk_preallocation_live_bytes(
            plan.external_live_bytes,
            plan.output_capacity,
            metadata_bytes,
            chunk_live_bytes,
            0,
        )?;
        jobs.push(RestartChunkJob {
            start_segment,
            end_segment,
            writer,
            error: None,
        });
    }
    Ok(PreparedRestartJobs {
        jobs,
        metadata_bytes,
        chunk_live_bytes,
    })
}

pub(in crate::encoder) fn parallel_entropy_chunk_count(
    segment_count: u32,
) -> Result<usize, JpegEncodeError> {
    let segment_count =
        usize::try_from(segment_count).map_err(|_| JpegEncodeError::InternalInvariant {
            reason: "JPEG restart segment count exceeds usize",
        })?;
    Ok(segment_count.clamp(1, MAX_PARALLEL_ENTROPY_CHUNKS))
}

pub(super) fn restart_chunk_segment_bounds(
    segment_count: u32,
    chunk_index: usize,
    chunk_count: usize,
) -> Result<(u32, u32), JpegEncodeError> {
    let segment_count = u64::from(segment_count);
    let chunk_index =
        u64::try_from(chunk_index).map_err(|_| JpegEncodeError::InternalInvariant {
            reason: "JPEG entropy chunk index exceeds u64",
        })?;
    let chunk_count =
        u64::try_from(chunk_count).map_err(|_| JpegEncodeError::InternalInvariant {
            reason: "JPEG entropy chunk count exceeds u64",
        })?;
    let start = segment_count
        .checked_mul(chunk_index)
        .and_then(|value| value.checked_div(chunk_count))
        .ok_or(JpegEncodeError::InternalInvariant {
            reason: "JPEG entropy chunk arithmetic overflow",
        })?;
    let end = segment_count
        .checked_mul(chunk_index + 1)
        .and_then(|value| value.checked_div(chunk_count))
        .ok_or(JpegEncodeError::InternalInvariant {
            reason: "JPEG entropy chunk arithmetic overflow",
        })?;
    Ok((
        u32::try_from(start).map_err(|_| JpegEncodeError::InternalInvariant {
            reason: "JPEG entropy chunk start exceeds u32",
        })?,
        u32::try_from(end).map_err(|_| JpegEncodeError::InternalInvariant {
            reason: "JPEG entropy chunk end exceeds u32",
        })?,
    ))
}

pub(super) fn restart_chunk_entropy_capacity(
    total_mcus: u32,
    restart_interval: u32,
    start_segment: u32,
    end_segment: u32,
    sampling: JpegBaselineSampling,
) -> Result<usize, JpegEncodeError> {
    let start_mcu = u64::from(start_segment)
        .checked_mul(u64::from(restart_interval))
        .ok_or(JpegEncodeError::InternalInvariant {
            reason: "JPEG restart MCU offset overflow",
        })?;
    let end_mcu = u64::from(end_segment)
        .checked_mul(u64::from(restart_interval))
        .ok_or(JpegEncodeError::InternalInvariant {
            reason: "JPEG restart MCU offset overflow",
        })?
        .min(u64::from(total_mcus));
    let marker_count =
        u64::from(end_segment - start_segment).saturating_sub(u64::from(start_segment == 0));
    jpeg_baseline_entropy_capacity_for_mcus(end_mcu - start_mcu, sampling, marker_count)
}

fn checked_restart_chunk_preallocation_live_bytes(
    external_live_bytes: usize,
    output_capacity: usize,
    metadata_bytes: usize,
    allocated_chunk_bytes: usize,
    next_chunk_capacity: usize,
) -> Result<usize, JpegEncodeError> {
    checked_encode_host_live_bytes([
        external_live_bytes,
        output_capacity,
        metadata_bytes,
        allocated_chunk_bytes,
        next_chunk_capacity,
    ])
}

#[expect(
    clippy::too_many_arguments,
    reason = "private JPEG entropy hot path keeps scalar arguments for optimized codegen"
)]
pub(super) fn encode_entropy_restart_chunk(
    planes: &[Cow<'_, [u8]>],
    width: u32,
    height: u32,
    sampling: JpegBaselineSampling,
    q_luma: &[u8; 64],
    q_chroma: &[u8; 64],
    dc_tables: [&JpegBaselineHuffmanTable; 2],
    ac_tables: [&JpegBaselineHuffmanTable; 2],
    cosine: &[[f64; 8]; 8],
    mcus_per_row: u32,
    total_mcus: u32,
    restart_interval: u32,
    start_segment: u32,
    end_segment: u32,
    max_bytes: usize,
    external_live_bytes: usize,
) -> Result<Vec<u8>, JpegEncodeError> {
    checked_encode_host_live_bytes([external_live_bytes, max_bytes])?;
    let mut writer = BitWriter::try_with_max_bytes(max_bytes)?;
    checked_encode_host_live_bytes([external_live_bytes, writer.capacity_bytes()])?;
    encode_entropy_restart_chunk_into(
        planes,
        width,
        height,
        sampling,
        q_luma,
        q_chroma,
        dc_tables,
        ac_tables,
        cosine,
        mcus_per_row,
        total_mcus,
        restart_interval,
        start_segment,
        end_segment,
        &mut writer,
    )?;
    writer.into_bytes()
}

#[expect(
    clippy::too_many_arguments,
    reason = "private JPEG entropy hot path keeps scalar arguments for optimized codegen"
)]
fn encode_entropy_restart_chunk_into(
    planes: &[Cow<'_, [u8]>],
    width: u32,
    height: u32,
    sampling: JpegBaselineSampling,
    q_luma: &[u8; 64],
    q_chroma: &[u8; 64],
    dc_tables: [&JpegBaselineHuffmanTable; 2],
    ac_tables: [&JpegBaselineHuffmanTable; 2],
    cosine: &[[f64; 8]; 8],
    mcus_per_row: u32,
    total_mcus: u32,
    restart_interval: u32,
    start_segment: u32,
    end_segment: u32,
    writer: &mut BitWriter,
) -> Result<(), JpegEncodeError> {
    for segment_index in start_segment..end_segment {
        if segment_index > 0 {
            let restart_index = u8::try_from((segment_index - 1) & 0x07).map_err(|_| {
                JpegEncodeError::InternalInvariant {
                    reason: "JPEG restart marker index exceeds u8",
                }
            })?;
            writer.write_restart_marker(0xD0 + restart_index)?;
        }
        let start_mcu = segment_index.checked_mul(restart_interval).ok_or(
            JpegEncodeError::InternalInvariant {
                reason: "JPEG restart MCU offset overflow",
            },
        )?;
        let end_mcu = start_mcu
            .checked_add(restart_interval)
            .ok_or(JpegEncodeError::InternalInvariant {
                reason: "JPEG restart MCU end overflow",
            })?
            .min(total_mcus);
        encode_entropy_mcu_range_into(
            planes,
            width,
            height,
            sampling,
            q_luma,
            q_chroma,
            dc_tables,
            ac_tables,
            cosine,
            mcus_per_row,
            start_mcu,
            end_mcu,
            writer,
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{checked_restart_chunk_preallocation_live_bytes, JpegEncodeError};
    use j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES;

    #[test]
    fn restart_chunk_preallocation_stops_before_the_next_writer_exceeds_the_cap() {
        let retained = DEFAULT_MAX_HOST_ALLOCATION_BYTES - 4;
        assert_eq!(
            checked_restart_chunk_preallocation_live_bytes(retained, 1, 1, 1, 1)
                .expect("exact preallocation boundary"),
            DEFAULT_MAX_HOST_ALLOCATION_BYTES
        );
        assert!(matches!(
            checked_restart_chunk_preallocation_live_bytes(retained, 1, 1, 1, 2),
            Err(JpegEncodeError::MemoryCapExceeded { requested, cap })
                if requested == DEFAULT_MAX_HOST_ALLOCATION_BYTES + 1
                    && cap == DEFAULT_MAX_HOST_ALLOCATION_BYTES
        ));
    }

    #[test]
    fn restart_entropy_module_stays_focused() {
        const SOURCE: &str = include_str!("restart.rs");
        assert!(SOURCE.lines().count() <= 420);
    }
}
