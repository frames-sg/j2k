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

    let parallel = cfg!(feature = "parallel") && jobs.len() >= HT_CPU_PARALLEL_FALLBACK_MIN_JOBS;
    let wave_size = cpu_worker_limit(jobs.len(), parallel).max(1);
    let mut retained_payload_bytes = 0usize;
    #[cfg(feature = "parallel")]
    if parallel {
        let full_fixed = [fixed[0], fixed[1], fixed[2], fixed[3], outer_bytes];
        // Charging every job's output and scratch makes this deliberately
        // independent of Rayon scheduling. If that conservative frontier is
        // too large, the bounded worker-sized waves below remain available.
        if try_check_full_ht_wave(jobs, tracker, &full_fixed)? {
            encoded
                .par_iter_mut()
                .zip(jobs.par_iter())
                .for_each(|(slot, job)| {
                    *slot = Some(ht_block_encode::try_encode_code_block_with_passes(
                        job.coefficients,
                        job.width,
                        job.height,
                        job.total_bitplanes,
                        job.target_coding_passes,
                    ));
                });
            retained_payload_bytes = checked_wave_payload_bytes(
                retained_payload_bytes,
                &mut encoded,
                "bounded CPU HT Tier-1 payload",
            )?;
            tracker.check(
                fixed
                    .into_iter()
                    .chain([outer_bytes, retained_payload_bytes]),
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
        ];
        check_ht_wave(job_wave, tracker, &wave_fixed, wave_size)?;

        #[cfg(feature = "parallel")]
        if parallel {
            slot_wave
                .par_iter_mut()
                .zip(job_wave.par_iter())
                .for_each(|(slot, job)| {
                    *slot = Some(ht_block_encode::try_encode_code_block_with_passes(
                        job.coefficients,
                        job.width,
                        job.height,
                        job.total_bitplanes,
                        job.target_coding_passes,
                    ));
                });
        } else {
            encode_ht_wave_serial(job_wave, slot_wave);
        }
        #[cfg(not(feature = "parallel"))]
        encode_ht_wave_serial(job_wave, slot_wave);

        retained_payload_bytes = checked_wave_payload_bytes(
            retained_payload_bytes,
            slot_wave,
            "bounded CPU HT Tier-1 payload",
        )?;
        tracker.check(
            fixed
                .into_iter()
                .chain([outer_bytes, retained_payload_bytes]),
            "bounded CPU HT Tier-1 output",
        )?;
    }
    Ok(encoded)
}

#[cfg(any(feature = "parallel", test))]
fn try_check_full_ht_wave(
    jobs: &[crate::J2kHtCodeBlockEncodeJob<'_>],
    tracker: &mut Tier1PhaseTracker<'_, '_>,
    fixed_and_retained_output: &[usize],
) -> NativeEncodePipelineResult<bool> {
    match check_ht_wave(jobs, tracker, fixed_and_retained_output, jobs.len()) {
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
    let wave_size = cpu_worker_limit(jobs.len(), parallel).max(1);
    let mut retained_payload_bytes = 0usize;
    for (job_wave, slot_wave) in jobs.chunks(wave_size).zip(encoded.chunks_mut(wave_size)) {
        let wave_fixed = [
            fixed[0],
            fixed[1],
            fixed[2],
            fixed[3],
            outer_bytes,
            retained_payload_bytes,
        ];
        check_classic_wave(job_wave, tracker, &wave_fixed, wave_size)?;

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

fn encode_ht_wave_serial(jobs: &[crate::J2kHtCodeBlockEncodeJob<'_>], slots: &mut [Tier1CpuSlot]) {
    for (slot, job) in slots.iter_mut().zip(jobs) {
        *slot = Some(ht_block_encode::try_encode_code_block_with_passes(
            job.coefficients,
            job.width,
            job.height,
            job.total_bitplanes,
            job.target_coding_passes,
        ));
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

        assert!(!try_check_full_ht_wave(&jobs, &mut tracker, &[])
            .expect("full-wave capacity fallback should be typed"));
        check_ht_wave(&jobs[..1], &mut tracker, &[], 1)
            .expect("one worker remains within the same cap");
    }
}
