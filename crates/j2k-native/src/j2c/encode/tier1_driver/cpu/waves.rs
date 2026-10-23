// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::super::allocation::checked_add_bytes;
use super::super::super::tier1_allocation::Tier1PhaseTracker;
use super::super::super::{
    bitplane_encode, ht_block_encode, internal_sub_band_type, J2kTier1CodeBlockEncodeJob,
    NativeEncodePipelineError, NativeEncodePipelineResult, Vec, HT_CPU_PARALLEL_FALLBACK_MIN_JOBS,
};
use super::super::scratch::{check_classic_wave, check_ht_wave, cpu_worker_limit};
use super::validate_ht_cpu_jobs;
#[cfg(feature = "parallel")]
use rayon::prelude::{
    IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator,
};

pub(in crate::j2c::encode::tier1_driver) type Tier1CpuSlot =
    Option<crate::EncodeResult<bitplane_encode::EncodedCodeBlock>>;

pub(in crate::j2c::encode::tier1_driver) fn encode_ht_cpu_results_accounted(
    jobs: &[crate::J2kHtCodeBlockEncodeJob<'_>],
    tracker: &mut Tier1PhaseTracker<'_, '_>,
    fixed: [usize; 4],
) -> NativeEncodePipelineResult<Vec<Tier1CpuSlot>> {
    validate_ht_cpu_jobs(jobs).map_err(NativeEncodePipelineError::unsupported)?;
    let (mut encoded, outer_bytes) = tracker.try_vec::<Tier1CpuSlot>(
        jobs.len(),
        fixed,
        "bounded CPU HT Tier-1 result owners",
    )?;
    encoded.resize_with(jobs.len(), || None);

    if jobs.is_empty() {
        return Ok(encoded);
    }

    let parallel = cfg!(feature = "parallel") && jobs.len() >= HT_CPU_PARALLEL_FALLBACK_MIN_JOBS;
    let wave_size = cpu_worker_limit(jobs.len(), parallel).max(1);
    let (mut workspaces, workspace_owner_bytes) = tracker
        .try_vec::<ht_block_encode::HtEncodeWorkspace>(
            wave_size,
            fixed.into_iter().chain([outer_bytes]),
            "bounded CPU HT Tier-1 workspace owners",
        )?;
    let workspace_bytes = checked_ht_workspace_bytes(wave_size)?;
    let mut retained_payload_bytes = 0usize;
    #[cfg(feature = "parallel")]
    if parallel {
        let full_fixed = [
            fixed[0],
            fixed[1],
            fixed[2],
            fixed[3],
            outer_bytes,
            workspace_owner_bytes,
        ];
        // Charging every job's output and every worker's scratch makes this
        // independent of Rayon scheduling. If that frontier is too large, the
        // bounded worker-sized waves below remain available.
        if try_check_full_ht_wave(jobs, wave_size, tracker, &full_fixed)? {
            try_fill_ht_workspaces(&mut workspaces, wave_size)?;
            encode_ht_parallel_with_workspaces(jobs, &mut encoded, &mut workspaces);
            retained_payload_bytes = checked_wave_payload_bytes(
                retained_payload_bytes,
                &mut encoded,
                "bounded CPU HT Tier-1 payload",
            )?;
            tracker.check(
                fixed.into_iter().chain([
                    outer_bytes,
                    workspace_owner_bytes,
                    workspace_bytes,
                    retained_payload_bytes,
                ]),
                "bounded CPU HT Tier-1 output",
            )?;
            return Ok(encoded);
        }
    }
    for (job_wave, slot_wave) in jobs.chunks(wave_size).zip(encoded.chunks_mut(wave_size)) {
        let wave_fixed = [
            fixed[0],
            fixed[1],
            fixed[2],
            fixed[3],
            outer_bytes,
            retained_payload_bytes,
            workspace_owner_bytes,
            checked_ht_workspace_bytes(wave_size - job_wave.len())?,
        ];
        check_ht_wave(job_wave, tracker, &wave_fixed, wave_size)?;

        if workspaces.is_empty() {
            try_fill_ht_workspaces(&mut workspaces, wave_size)?;
        }

        #[cfg(feature = "parallel")]
        if parallel {
            encode_ht_parallel_with_workspaces(job_wave, slot_wave, &mut workspaces);
        } else {
            encode_ht_wave_serial(job_wave, slot_wave, &mut workspaces[0]);
        }
        #[cfg(not(feature = "parallel"))]
        encode_ht_wave_serial(job_wave, slot_wave, &mut workspaces[0]);

        retained_payload_bytes = checked_wave_payload_bytes(
            retained_payload_bytes,
            slot_wave,
            "bounded CPU HT Tier-1 payload",
        )?;
        tracker.check(
            fixed.into_iter().chain([
                outer_bytes,
                workspace_owner_bytes,
                workspace_bytes,
                retained_payload_bytes,
            ]),
            "bounded CPU HT Tier-1 output",
        )?;
    }
    Ok(encoded)
}

#[cfg(any(feature = "parallel", test))]
fn try_check_full_ht_wave(
    jobs: &[crate::J2kHtCodeBlockEncodeJob<'_>],
    workers: usize,
    tracker: &mut Tier1PhaseTracker<'_, '_>,
    fixed_and_retained_output: &[usize],
) -> NativeEncodePipelineResult<bool> {
    match check_ht_wave(jobs, tracker, fixed_and_retained_output, workers) {
        Ok(_) => Ok(true),
        Err(NativeEncodePipelineError::Typed(crate::EncodeError::AllocationTooLarge {
            ..
        })) => Ok(false),
        Err(error) => Err(error),
    }
}

pub(in crate::j2c::encode::tier1_driver) fn encode_classic_cpu_results_accounted(
    jobs: &[J2kTier1CodeBlockEncodeJob<'_>],
    tracker: &mut Tier1PhaseTracker<'_, '_>,
    fixed: [usize; 4],
) -> NativeEncodePipelineResult<Vec<Tier1CpuSlot>> {
    let (mut encoded, outer_bytes) = tracker.try_vec::<Tier1CpuSlot>(
        jobs.len(),
        fixed,
        "bounded CPU classic Tier-1 result owners",
    )?;
    encoded.resize_with(jobs.len(), || None);

    let parallel = cfg!(feature = "parallel");
    let workers = cpu_worker_limit(jobs.len(), parallel).max(1);
    let mut retained_payload_bytes = 0usize;
    let mut start = 0;
    while start < jobs.len() {
        let wave_fixed = [
            fixed[0],
            fixed[1],
            fixed[2],
            fixed[3],
            outer_bytes,
            retained_payload_bytes,
        ];
        let end = start + longest_classic_wave(&jobs[start..], workers, tracker, &wave_fixed)?;
        let (job_wave, slot_wave) = (&jobs[start..end], &mut encoded[start..end]);
        start = end;

        #[cfg(feature = "parallel")]
        if parallel {
            slot_wave
                .par_iter_mut()
                .zip(job_wave.par_iter())
                .for_each(|(slot, job)| {
                    *slot = Some(bitplane_encode::try_encode_code_block(
                        job.coefficients,
                        job.width,
                        job.height,
                        internal_sub_band_type(job.sub_band_type),
                        job.total_bitplanes,
                    ));
                });
        } else {
            encode_classic_wave_serial(job_wave, slot_wave);
        }
        #[cfg(not(feature = "parallel"))]
        encode_classic_wave_serial(job_wave, slot_wave);

        retained_payload_bytes = checked_wave_payload_bytes(
            retained_payload_bytes,
            slot_wave,
            "bounded CPU classic Tier-1 payload",
        )?;
        tracker.check(
            fixed
                .into_iter()
                .chain([outer_bytes, retained_payload_bytes]),
            "bounded CPU classic Tier-1 output",
        )?;
    }
    Ok(encoded)
}

/// The longest prefix of `jobs`, doubling from one block per worker, whose
/// worst-case frontier fits the cap. Each wave ends at a barrier, so fewer and
/// longer waves leave workers idle less often on unequal blocks and cores.
fn longest_classic_wave(
    jobs: &[J2kTier1CodeBlockEncodeJob<'_>],
    workers: usize,
    tracker: &mut Tier1PhaseTracker<'_, '_>,
    fixed_and_retained_output: &[usize],
) -> NativeEncodePipelineResult<usize> {
    let mut len = workers.min(jobs.len());
    check_classic_wave(&jobs[..len], tracker, fixed_and_retained_output, workers)?;
    while len < jobs.len() {
        let next = len.saturating_mul(2).min(jobs.len());
        match check_classic_wave(&jobs[..next], tracker, fixed_and_retained_output, workers) {
            Ok(_) => len = next,
            Err(NativeEncodePipelineError::Typed(crate::EncodeError::AllocationTooLarge {
                ..
            })) => break,
            Err(error) => return Err(error),
        }
    }
    Ok(len)
}

fn try_fill_ht_workspaces(
    workspaces: &mut Vec<ht_block_encode::HtEncodeWorkspace>,
    count: usize,
) -> NativeEncodePipelineResult<()> {
    for _ in 0..count {
        workspaces.push(ht_block_encode::HtEncodeWorkspace::try_new()?);
    }
    Ok(())
}

fn checked_ht_workspace_bytes(count: usize) -> NativeEncodePipelineResult<usize> {
    count
        .checked_mul(ht_block_encode::HtEncodeWorkspace::ALLOCATION_BYTES)
        .ok_or(crate::EncodeError::ArithmeticOverflow {
            what: "HTJ2K CPU workspace bytes",
        })
        .map_err(Into::into)
}

#[cfg(feature = "parallel")]
fn encode_ht_parallel_with_workspaces(
    jobs: &[crate::J2kHtCodeBlockEncodeJob<'_>],
    slots: &mut [Tier1CpuSlot],
    workspaces: &mut [ht_block_encode::HtEncodeWorkspace],
) {
    // Workers pull small batches so slower cores and denser bands do not
    // hold the others up; one fixed chunk per worker scaled 1.9x on 10 cores.
    let batch = (jobs.len() / (4 * workspaces.len())).clamp(1, 8);
    let batches = std::sync::Mutex::new(slots.chunks_mut(batch).zip(jobs.chunks(batch)));
    workspaces
        .par_iter_mut()
        .with_max_len(1)
        .for_each(|workspace| {
            while let Some((slots, jobs)) = batches.lock().ok().and_then(|mut b| b.next()) {
                encode_ht_wave_serial(jobs, slots, workspace);
            }
        });
}

fn encode_ht_wave_serial(
    jobs: &[crate::J2kHtCodeBlockEncodeJob<'_>],
    slots: &mut [Tier1CpuSlot],
    workspace: &mut ht_block_encode::HtEncodeWorkspace,
) {
    for (slot, job) in slots.iter_mut().zip(jobs) {
        *slot = Some(
            ht_block_encode::try_encode_code_block_with_passes_in_workspace(
                job.coefficients,
                job.width,
                job.height,
                job.total_bitplanes,
                job.target_coding_passes,
                workspace,
            ),
        );
    }
}

fn encode_classic_wave_serial(jobs: &[J2kTier1CodeBlockEncodeJob<'_>], slots: &mut [Tier1CpuSlot]) {
    for (slot, job) in slots.iter_mut().zip(jobs) {
        *slot = Some(bitplane_encode::try_encode_code_block(
            job.coefficients,
            job.width,
            job.height,
            internal_sub_band_type(job.sub_band_type),
            job.total_bitplanes,
        ));
    }
}

fn checked_wave_payload_bytes(
    mut retained: usize,
    slots: &mut [Tier1CpuSlot],
    what: &'static str,
) -> NativeEncodePipelineResult<usize> {
    for slot in slots {
        if matches!(slot, Some(Err(_))) {
            let Some(Err(error)) = slot.take() else {
                return Err(crate::EncodeError::InternalInvariant {
                    what: "Tier-1 worker error slot changed during collection",
                }
                .into());
            };
            return Err(error.into());
        }
        match slot.as_ref() {
            Some(Ok(block)) => {
                retained = checked_add_bytes(retained, block.data.capacity(), what)?;
            }
            Some(Err(_)) => {
                return Err(crate::EncodeError::InternalInvariant {
                    what: "Tier-1 worker error slot survived extraction",
                }
                .into())
            }
            None => {
                return Err(crate::EncodeError::InternalInvariant {
                    what: "Tier-1 worker wave left a result slot empty",
                }
                .into());
            }
        }
    }
    Ok(retained)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::j2c::encode::{NativeEncodeRetainedInput, NativeEncodeSession};

    #[test]
    fn accounted_ht_cpu_results_match_fresh_workspaces_across_many_blocks() {
        let coefficient_blocks: Vec<Vec<i32>> = (0..64usize)
            .map(|seed| {
                let side = if seed.is_multiple_of(3) { 8 } else { 64 };
                (0..side * side)
                    .map(|index| {
                        if (index + seed).is_multiple_of(13) {
                            0
                        } else {
                            i32::try_from(((index * 29) ^ (seed * 11)) & 0x01ff)
                                .expect("masked coefficient fits i32")
                                - 255
                        }
                    })
                    .collect()
            })
            .collect();
        let jobs: Vec<_> = coefficient_blocks
            .iter()
            .enumerate()
            .map(|(index, coefficients)| {
                let side = if index.is_multiple_of(3) { 8 } else { 64 };
                crate::J2kHtCodeBlockEncodeJob {
                    coefficients,
                    width: side,
                    height: side,
                    total_bitplanes: 9,
                    target_coding_passes: 1,
                }
            })
            .collect();
        let expected: Vec<_> = jobs
            .iter()
            .map(|job| {
                ht_block_encode::try_encode_code_block_with_passes(
                    job.coefficients,
                    job.width,
                    job.height,
                    job.total_bitplanes,
                    job.target_coding_passes,
                )
                .expect("fresh-workspace encode")
            })
            .collect();
        let session = NativeEncodeSession::try_new(NativeEncodeRetainedInput::none())
            .expect("HT CPU session");
        let mut tracker = Tier1PhaseTracker::new(&session, 0);

        let actual = encode_ht_cpu_results_accounted(&jobs, &mut tracker, [0; 4])
            .expect("accounted HT CPU encode");

        for (slot, expected) in actual.into_iter().zip(expected) {
            let actual = slot.expect("worker slot").expect("reused-workspace encode");
            assert_eq!(actual.data, expected.data);
            assert_eq!(actual.num_coding_passes, expected.num_coding_passes);
            assert_eq!(actual.num_zero_bitplanes, expected.num_zero_bitplanes);
            assert_eq!(actual.ht_cleanup_length, expected.ht_cleanup_length);
            assert_eq!(actual.ht_refinement_length, expected.ht_refinement_length);
        }
    }

    #[test]
    fn classic_waves_double_while_their_frontier_fits_the_cap() {
        let coefficients = [1_i32; 16];
        let jobs = [J2kTier1CodeBlockEncodeJob {
            coefficients: &coefficients,
            width: 4,
            height: 4,
            sub_band_type: crate::J2kSubBandType::LowLow,
            total_bitplanes: 1,
            style: crate::j2c::encode::api_helpers::default_public_code_block_style(),
        }; 5];
        let worker =
            bitplane_encode::classic_worker_allocation(4, 4, 1).expect("worker allocation");
        // Rows: workers, blocks whose output and running scratch fit, expected wave.
        for (workers, fitting, expected) in [
            (1, 1, 1),
            (1, 3, 2),
            (1, 4, 4),
            (1, 5, 5),
            (2, 2, 2),
            (2, 3, 2),
            (2, 5, 5),
            (5, 5, 5),
        ] {
            let cap = fitting * worker.output_bytes + workers.min(fitting) * worker.scratch_bytes;
            let session = NativeEncodeSession::try_with_cap(NativeEncodeRetainedInput::none(), cap)
                .expect("session");
            let mut tracker = Tier1PhaseTracker::new(&session, 0);
            let actual = longest_classic_wave(&jobs, workers, &mut tracker, &[])
                .expect("one worker wave fits");
            assert_eq!(actual, expected, "{workers} workers, {fitting} fitting");
        }
    }

    #[test]
    fn full_ht_wave_falls_back_when_only_one_worker_frontier_fits() {
        let coefficients = [1_i32; 16];
        let jobs = [
            crate::J2kHtCodeBlockEncodeJob {
                coefficients: &coefficients,
                width: 4,
                height: 4,
                total_bitplanes: 1,
                target_coding_passes: 1,
            },
            crate::J2kHtCodeBlockEncodeJob {
                coefficients: &coefficients,
                width: 4,
                height: 4,
                total_bitplanes: 1,
                target_coding_passes: 1,
            },
        ];
        let worker = ht_block_encode::ht_worker_allocation(4, 4, 1)
            .expect("worker allocation")
            .total_bytes()
            .expect("worker frontier");
        let session = NativeEncodeSession::try_with_cap(NativeEncodeRetainedInput::none(), worker)
            .expect("one-worker session");
        let mut tracker = Tier1PhaseTracker::new(&session, 0);

        assert!(!try_check_full_ht_wave(&jobs, 2, &mut tracker, &[])
            .expect("full-wave capacity fallback should be typed"));
        check_ht_wave(&jobs[..1], &mut tracker, &[], 1)
            .expect("one worker remains within the same cap");
    }

    #[test]
    fn full_ht_wave_charges_every_output_but_one_scratch_per_worker() {
        let coefficients = [1_i32; 16];
        let jobs = [crate::J2kHtCodeBlockEncodeJob {
            coefficients: &coefficients,
            width: 4,
            height: 4,
            total_bitplanes: 1,
            target_coding_passes: 1,
        }; 3];
        let worker = ht_block_encode::ht_worker_allocation(4, 4, 1).expect("worker allocation");
        for workers in [1, 2, 3] {
            let frontier = 3 * worker.output_bytes + workers * worker.scratch_bytes;
            for (cap, fits) in [(frontier, true), (frontier - 1, false)] {
                let session =
                    NativeEncodeSession::try_with_cap(NativeEncodeRetainedInput::none(), cap)
                        .expect("session");
                let mut tracker = Tier1PhaseTracker::new(&session, 0);
                let actual = try_check_full_ht_wave(&jobs, workers, &mut tracker, &[])
                    .expect("capacity fallback should be typed");
                assert_eq!(actual, fits, "{workers} workers, cap {cap}");
            }
        }
    }
}
