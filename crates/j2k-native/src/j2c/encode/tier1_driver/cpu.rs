// SPDX-License-Identifier: MIT OR Apache-2.0

//! Bounded CPU Tier-1 execution and test parity helpers.

#[cfg(test)]
use super::super::{bitplane_encode, ht_block_encode, Vec};

#[cfg(all(feature = "parallel", test))]
use rayon::prelude::IntoParallelRefMutIterator;
#[cfg(all(feature = "parallel", test))]
use rayon::prelude::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

pub(super) fn coefficients_fit_i32(coefficients: &[i64]) -> bool {
    coefficients
        .iter()
        .all(|&coefficient| i32::try_from(coefficient).is_ok())
}

mod direct_i64;
mod waves;
pub(super) use direct_i64::encode_classic_i64_direct;
pub(super) use waves::{encode_classic_cpu_results_accounted, encode_ht_cpu_results_accounted};

#[cfg(test)]
pub(in crate::j2c::encode) fn encode_all_ht_code_blocks_serial_cpu(
    jobs: &[crate::J2kHtCodeBlockEncodeJob<'_>],
) -> Result<Vec<bitplane_encode::EncodedCodeBlock>, &'static str> {
    validate_ht_cpu_jobs(jobs)?;
    let mut encoded = Vec::new();
    encoded
        .try_reserve_exact(jobs.len())
        .map_err(|_| "HT Tier-1 result owner allocation failed")?;
    for job in jobs {
        encoded.push(ht_block_encode::encode_code_block_with_passes(
            job.coefficients,
            job.width,
            job.height,
            job.total_bitplanes,
            job.target_coding_passes,
        )?);
    }
    Ok(encoded)
}

#[cfg(all(feature = "parallel", test))]
pub(in crate::j2c::encode) fn encode_all_ht_code_blocks_parallel(
    jobs: &[crate::J2kHtCodeBlockEncodeJob<'_>],
) -> Result<Vec<bitplane_encode::EncodedCodeBlock>, &'static str> {
    validate_ht_cpu_jobs(jobs)?;
    let mut slots = Vec::new();
    slots
        .try_reserve_exact(jobs.len())
        .map_err(|_| "HT Tier-1 parallel result owner allocation failed")?;
    slots.resize_with(jobs.len(), || None);
    slots
        .par_iter_mut()
        .zip(jobs.par_iter())
        .for_each(|(slot, job)| {
            *slot = Some(ht_block_encode::encode_code_block_with_passes(
                job.coefficients,
                job.width,
                job.height,
                job.total_bitplanes,
                job.target_coding_passes,
            ));
        });
    collect_parallel_slots(slots, "parallel HT Tier-1 result is missing")
}

#[cfg(all(not(feature = "parallel"), test))]
pub(in crate::j2c::encode) fn encode_all_ht_code_blocks_parallel(
    jobs: &[crate::J2kHtCodeBlockEncodeJob<'_>],
) -> Result<Vec<bitplane_encode::EncodedCodeBlock>, &'static str> {
    encode_all_ht_code_blocks_serial_cpu(jobs)
}

#[cfg(all(feature = "parallel", test))]
fn collect_parallel_slots<T>(
    slots: Vec<Option<Result<T, &'static str>>>,
    missing: &'static str,
) -> Result<Vec<T>, &'static str> {
    let mut encoded = Vec::new();
    encoded
        .try_reserve_exact(slots.len())
        .map_err(|_| "Tier-1 ordered result owner allocation failed")?;
    for slot in slots {
        encoded.push(slot.ok_or(missing)??);
    }
    Ok(encoded)
}

fn validate_ht_cpu_jobs(jobs: &[crate::J2kHtCodeBlockEncodeJob<'_>]) -> Result<(), &'static str> {
    if jobs
        .iter()
        .any(|job| !(1..=3).contains(&job.target_coding_passes))
    {
        return Err("CPU HTJ2K code-block fallback supports at most three HT coding passes");
    }
    Ok(())
}
