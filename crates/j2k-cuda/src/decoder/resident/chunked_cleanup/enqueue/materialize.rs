// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_core::HtGpuJobChunkEntry;
use j2k_cuda_j2k_engine::CudaHtj2kCodeBlockJob;

use super::super::super::super::{CudaComponentDecodeWork, Error};
use super::super::planning::{
    job_at, sort_selected_jobs_for_coalesced_targets, Htj2kJobLocation, SelectedHtj2kChunkJob,
};
use super::super::Htj2kChunkJobIdentity;
use super::chunk_plan_invariant_error;
use crate::allocation::HostPhaseBudget;

pub(super) struct LogicalPayloadPart<'a> {
    bytes: &'a [u8],
    logical_start: usize,
    logical_end: usize,
}

pub(super) struct LogicalPayload<'a> {
    parts: Vec<LogicalPayloadPart<'a>>,
    len: usize,
}

pub(super) struct MaterializedHtj2kChunk<'a> {
    pub(super) payload_parts: Vec<&'a [u8]>,
    pub(super) jobs: Vec<CudaHtj2kCodeBlockJob>,
    pub(super) identities: Vec<Htj2kChunkJobIdentity>,
}

#[derive(Clone, Copy)]
struct LastPayloadSpan {
    part_index: usize,
    start: usize,
    end: usize,
}

pub(super) fn build_logical_payload<'a>(
    payload_parts: &[&'a [u8]],
    budget: &mut HostPhaseBudget,
) -> Result<LogicalPayload<'a>, Error> {
    let mut parts = budget.try_vec_with_capacity(payload_parts.len())?;
    let mut logical_start = 0usize;
    for &bytes in payload_parts {
        let logical_end = logical_start
            .checked_add(bytes.len())
            .ok_or_else(chunk_plan_invariant_error)?;
        parts.push(LogicalPayloadPart {
            bytes,
            logical_start,
            logical_end,
        });
        logical_start = logical_end;
    }
    Ok(LogicalPayload {
        parts,
        len: logical_start,
    })
}

pub(super) fn select_chunk_jobs(
    entries: &[HtGpuJobChunkEntry],
    locations: &[Htj2kJobLocation],
    budget: &mut HostPhaseBudget,
) -> Result<Vec<SelectedHtj2kChunkJob>, Error> {
    let mut selected = budget.try_vec_with_capacity(entries.len())?;
    for entry in entries {
        let location = *locations
            .get(entry.original_job_index())
            .ok_or_else(chunk_plan_invariant_error)?;
        if location.source != entry.source_index() {
            return Err(chunk_plan_invariant_error());
        }
        selected.push(SelectedHtj2kChunkJob {
            location,
            original_job_index: entry.original_job_index(),
            source_index: entry.source_index(),
        });
    }
    sort_selected_jobs_for_coalesced_targets(&mut selected);
    Ok(selected)
}

pub(super) fn materialize_chunk_payload<'a>(
    logical_payload: &LogicalPayload<'a>,
    component_work: &[CudaComponentDecodeWork],
    selected: &[SelectedHtj2kChunkJob],
    payload_bytes: usize,
    budget: &mut HostPhaseBudget,
) -> Result<MaterializedHtj2kChunk<'a>, Error> {
    materialize_chunk_jobs(
        logical_payload,
        selected.iter().map(|selected| {
            Ok((
                *job_at(component_work, selected.location)?,
                Htj2kChunkJobIdentity::new(selected.original_job_index, selected.source_index),
            ))
        }),
        payload_bytes,
        budget,
    )
}

fn materialize_chunk_jobs<'a>(
    logical_payload: &LogicalPayload<'a>,
    input_jobs: impl ExactSizeIterator<
        Item = Result<(CudaHtj2kCodeBlockJob, Htj2kChunkJobIdentity), Error>,
    >,
    payload_bytes: usize,
    budget: &mut HostPhaseBudget,
) -> Result<MaterializedHtj2kChunk<'a>, Error> {
    let job_count = input_jobs.len();
    let mut payload_parts = budget.try_vec_with_capacity(job_count)?;
    let mut jobs = budget.try_vec_with_capacity(job_count)?;
    let mut identities = budget.try_vec_with_capacity(job_count)?;
    let mut materialized_bytes = 0usize;
    let mut last_span = None;
    for input in input_jobs {
        let (mut job, identity) = input?;
        let start =
            usize::try_from(job.payload_offset).map_err(|_| chunk_plan_invariant_error())?;
        let end = start
            .checked_add(job.payload_len as usize)
            .ok_or_else(chunk_plan_invariant_error)?;
        if end > logical_payload.len {
            return Err(chunk_plan_invariant_error());
        }
        job.payload_offset =
            u64::try_from(materialized_bytes).map_err(|_| chunk_plan_invariant_error())?;
        append_payload_range(
            logical_payload,
            start,
            end,
            &mut payload_parts,
            &mut last_span,
            budget,
        )?;
        materialized_bytes = materialized_bytes
            .checked_add(job.payload_len as usize)
            .ok_or_else(chunk_plan_invariant_error)?;
        jobs.push(job);
        identities.push(identity);
    }
    if materialized_bytes != payload_bytes {
        return Err(chunk_plan_invariant_error());
    }
    Ok(MaterializedHtj2kChunk {
        payload_parts,
        jobs,
        identities,
    })
}

fn append_payload_range<'a>(
    logical_payload: &LogicalPayload<'a>,
    start: usize,
    end: usize,
    payload_parts: &mut Vec<&'a [u8]>,
    last_span: &mut Option<LastPayloadSpan>,
    budget: &mut HostPhaseBudget,
) -> Result<(), Error> {
    if start == end {
        return Ok(());
    }
    let mut cursor = start;
    while cursor < end {
        let part_index = logical_payload
            .parts
            .partition_point(|part| part.logical_end <= cursor);
        let part = logical_payload
            .parts
            .get(part_index)
            .ok_or_else(chunk_plan_invariant_error)?;
        if cursor < part.logical_start {
            return Err(chunk_plan_invariant_error());
        }
        let span_logical_end = end.min(part.logical_end);
        let span_start = cursor - part.logical_start;
        let span_end = span_logical_end - part.logical_start;
        append_payload_span(
            part,
            part_index,
            span_start,
            span_end,
            payload_parts,
            last_span,
            budget,
        )?;
        cursor = span_logical_end;
    }
    Ok(())
}

fn append_payload_span<'a>(
    part: &LogicalPayloadPart<'a>,
    part_index: usize,
    start: usize,
    end: usize,
    payload_parts: &mut Vec<&'a [u8]>,
    last_span: &mut Option<LastPayloadSpan>,
    budget: &mut HostPhaseBudget,
) -> Result<(), Error> {
    if let Some(previous) = last_span.as_mut() {
        if previous.part_index == part_index && previous.end == start {
            let last = payload_parts
                .last_mut()
                .ok_or_else(chunk_plan_invariant_error)?;
            *last = part
                .bytes
                .get(previous.start..end)
                .ok_or_else(chunk_plan_invariant_error)?;
            previous.end = end;
            return Ok(());
        }
    }
    let bytes = part
        .bytes
        .get(start..end)
        .ok_or_else(chunk_plan_invariant_error)?;
    budget.try_vec_push(payload_parts, bytes)?;
    *last_span = Some(LastPayloadSpan {
        part_index,
        start,
        end,
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        build_logical_payload, materialize_chunk_jobs, CudaHtj2kCodeBlockJob, Htj2kChunkJobIdentity,
    };
    use crate::allocation::HostPhaseBudget;

    #[test]
    fn borrowed_payload_spans_cross_parts_and_coalesce_adjacent_ranges() {
        let mut budget = HostPhaseBudget::new("test borrowed HT chunk payload");
        let first = b"abcd".as_slice();
        let second = b"EFGH".as_slice();
        let logical = build_logical_payload(&[first, second], &mut budget).unwrap();
        let jobs = [job(1, 2), job(3, 3)];
        let materialized = materialize_chunk_jobs(
            &logical,
            jobs.into_iter()
                .enumerate()
                .map(|(index, job)| Ok((job, Htj2kChunkJobIdentity::new(index, index + 10)))),
            5,
            &mut budget,
        )
        .unwrap();

        assert_eq!(materialized.payload_parts, [b"bcd".as_slice(), b"EF"]);
        assert_eq!(materialized.payload_parts[0].as_ptr(), first[1..].as_ptr());
        assert_eq!(materialized.payload_parts[1].as_ptr(), second.as_ptr());
        assert_eq!(
            materialized
                .jobs
                .iter()
                .map(|job| job.payload_offset)
                .collect::<Vec<_>>(),
            [0, 2]
        );
        assert_eq!(
            materialized.identities,
            [
                Htj2kChunkJobIdentity::new(0, 10),
                Htj2kChunkJobIdentity::new(1, 11),
            ]
        );
    }

    #[test]
    fn borrowed_payload_spans_preserve_gap_and_noncontiguous_job_order() {
        let mut budget = HostPhaseBudget::new("test noncontiguous HT chunk payload");
        let logical = build_logical_payload(&[b"abcdef"], &mut budget).unwrap();
        let jobs = [job(4, 2), job(1, 2)];
        let materialized = materialize_chunk_jobs(
            &logical,
            jobs.into_iter()
                .enumerate()
                .map(|(index, job)| Ok((job, Htj2kChunkJobIdentity::new(index, index)))),
            4,
            &mut budget,
        )
        .unwrap();

        assert_eq!(materialized.payload_parts, [b"ef".as_slice(), b"bc"]);
        assert_eq!(
            materialized
                .jobs
                .iter()
                .map(|job| job.payload_offset)
                .collect::<Vec<_>>(),
            [0, 2]
        );
    }

    #[test]
    fn zero_length_jobs_share_the_current_rebased_offset_without_payload_spans() {
        let mut budget = HostPhaseBudget::new("test zero length HT chunk payload");
        let logical = build_logical_payload(&[b"abc", b""], &mut budget).unwrap();
        let jobs = [job(3, 0), job(1, 0)];
        let materialized = materialize_chunk_jobs(
            &logical,
            jobs.into_iter()
                .enumerate()
                .map(|(index, job)| Ok((job, Htj2kChunkJobIdentity::new(index, index)))),
            0,
            &mut budget,
        )
        .unwrap();

        assert_eq!(materialized.payload_parts, [] as [&[u8]; 0]);
        assert!(materialized.jobs.iter().all(|job| job.payload_offset == 0));
    }

    #[test]
    fn invalid_logical_ranges_and_payload_totals_are_rejected() {
        let mut budget = HostPhaseBudget::new("test invalid HT chunk payload");
        let logical = build_logical_payload(&[b"abc", b"def"], &mut budget).unwrap();
        let identity = Htj2kChunkJobIdentity::new(0, 0);

        assert!(materialize_chunk_jobs(
            &logical,
            [Ok((job(5, 2), identity))].into_iter(),
            2,
            &mut budget,
        )
        .is_err());
        assert!(materialize_chunk_jobs(
            &logical,
            [Ok((job(0, 2), identity))].into_iter(),
            3,
            &mut budget,
        )
        .is_err());
    }

    fn job(payload_offset: u64, payload_len: u32) -> CudaHtj2kCodeBlockJob {
        CudaHtj2kCodeBlockJob {
            payload_offset,
            width: 1,
            height: 1,
            payload_len,
            cleanup_length: payload_len,
            refinement_length: 0,
            missing_bit_planes: 0,
            num_bitplanes: 1,
            roi_shift: 0,
            number_of_coding_passes: 1,
            output_stride: 1,
            output_offset: 0,
            dequantization_step: 1.0,
            stripe_causal: false,
            irreversible_midpoint: false,
        }
    }
}
