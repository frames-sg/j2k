// SPDX-License-Identifier: MIT OR Apache-2.0

//! Schedule-independent Tier-1 worker-wave accounting.

use super::super::allocation::checked_add_bytes;
use super::super::tier1_allocation::Tier1PhaseTracker;
use super::super::{
    bitplane_encode, ht_block_encode, J2kTier1CodeBlockEncodeJob, NativeEncodePipelineResult,
};
use crate::{EncodeError, EncodeResult};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct WorkerAllocation {
    output_bytes: usize,
    scratch_bytes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct WorkerFrontier {
    pub(super) output_bytes: usize,
    pub(super) scratch_bytes: usize,
}

impl WorkerFrontier {
    #[cfg(test)]
    fn total_bytes(self) -> EncodeResult<usize> {
        checked_add_bytes(
            self.output_bytes,
            self.scratch_bytes,
            "Tier-1 worker frontier",
        )
    }
}

pub(super) fn check_classic_wave(
    jobs: &[J2kTier1CodeBlockEncodeJob<'_>],
    tracker: &mut Tier1PhaseTracker<'_, '_>,
    fixed_and_retained_output: &[usize],
    active_worker_limit: usize,
) -> NativeEncodePipelineResult<WorkerFrontier> {
    check_worker_wave(
        jobs.iter().map(|job| {
            bitplane_encode::classic_worker_allocation(
                job.width as usize,
                job.height as usize,
                job.total_bitplanes,
            )
            .map(|plan| WorkerAllocation {
                output_bytes: plan.output_bytes,
                scratch_bytes: plan.scratch_bytes,
            })
        }),
        jobs.len(),
        active_worker_limit,
        tracker,
        fixed_and_retained_output,
        "classic Tier-1 CPU worker wave",
    )
}

pub(super) fn check_ht_wave(
    jobs: &[crate::J2kHtCodeBlockEncodeJob<'_>],
    tracker: &mut Tier1PhaseTracker<'_, '_>,
    fixed_and_retained_output: &[usize],
    active_worker_limit: usize,
) -> NativeEncodePipelineResult<WorkerFrontier> {
    check_worker_wave(
        jobs.iter().map(|job| {
            ht_block_encode::ht_worker_allocation(
                job.width as usize,
                job.height as usize,
                job.target_coding_passes,
            )
            .map(|plan| WorkerAllocation {
                output_bytes: plan.output_bytes,
                scratch_bytes: plan.scratch_bytes,
            })
        }),
        jobs.len(),
        active_worker_limit,
        tracker,
        fixed_and_retained_output,
        "HTJ2K Tier-1 CPU worker wave",
    )
}

pub(super) fn cpu_worker_limit(job_count: usize, parallel: bool) -> usize {
    if job_count == 0 {
        return 0;
    }
    #[cfg(feature = "parallel")]
    if parallel {
        return rayon::current_num_threads().min(job_count);
    }
    let _ = parallel;
    1
}

fn check_worker_wave(
    plans: impl IntoIterator<Item = EncodeResult<WorkerAllocation>>,
    job_count: usize,
    active_worker_limit: usize,
    tracker: &mut Tier1PhaseTracker<'_, '_>,
    fixed_and_retained_output: &[usize],
    what: &'static str,
) -> NativeEncodePipelineResult<WorkerFrontier> {
    if job_count > active_worker_limit {
        return Err(EncodeError::InternalInvariant {
            what: "Tier-1 worker wave exceeds the active-worker limit",
        }
        .into());
    }
    let frontier = plans.into_iter().try_fold(
        WorkerFrontier {
            output_bytes: 0,
            scratch_bytes: 0,
        },
        |frontier, plan| {
            let plan = plan?;
            Ok::<WorkerFrontier, EncodeError>(WorkerFrontier {
                output_bytes: checked_add_bytes(
                    frontier.output_bytes,
                    plan.output_bytes,
                    "Tier-1 worker-wave output bound",
                )?,
                scratch_bytes: checked_add_bytes(
                    frontier.scratch_bytes,
                    plan.scratch_bytes,
                    "Tier-1 worker-wave scratch bound",
                )?,
            })
        },
    )?;
    tracker.check(
        fixed_and_retained_output
            .iter()
            .copied()
            .chain([frontier.output_bytes, frontier.scratch_bytes]),
        what,
    )?;
    Ok(frontier)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::j2c::encode::{NativeEncodeRetainedInput, NativeEncodeSession};

    fn synthetic_frontier() -> WorkerFrontier {
        [
            WorkerAllocation {
                output_bytes: 10,
                scratch_bytes: 1,
            },
            WorkerAllocation {
                output_bytes: 20,
                scratch_bytes: 9,
            },
            WorkerAllocation {
                output_bytes: 30,
                scratch_bytes: 4,
            },
        ]
        .into_iter()
        .try_fold(
            WorkerFrontier {
                output_bytes: 0,
                scratch_bytes: 0,
            },
            |frontier, plan| {
                Ok::<WorkerFrontier, EncodeError>(WorkerFrontier {
                    output_bytes: checked_add_bytes(
                        frontier.output_bytes,
                        plan.output_bytes,
                        "synthetic output",
                    )?,
                    scratch_bytes: checked_add_bytes(
                        frontier.scratch_bytes,
                        plan.scratch_bytes,
                        "synthetic scratch",
                    )?,
                })
            },
        )
        .expect("synthetic frontier")
    }

    #[test]
    fn worker_wave_sums_every_simultaneously_live_output_and_scratch_owner() {
        let frontier = synthetic_frontier();
        assert_eq!(frontier.output_bytes, 60);
        assert_eq!(frontier.scratch_bytes, 14);
        assert_eq!(frontier.total_bytes().expect("frontier total"), 74);
    }

    #[test]
    fn exact_frontier_cap_passes_and_one_byte_less_is_typed() {
        let frontier = synthetic_frontier();
        let exact_cap = frontier.total_bytes().expect("frontier total");
        NativeEncodeSession::try_with_cap(NativeEncodeRetainedInput::none(), exact_cap)
            .expect("exact cap session")
            .checked_phase(exact_cap, "test Tier-1 frontier")
            .expect("exact frontier is accepted");

        let cap = exact_cap - 1;
        let error = NativeEncodeSession::try_with_cap(NativeEncodeRetainedInput::none(), cap)
            .expect("under-cap session itself is valid")
            .checked_phase(exact_cap, "test Tier-1 frontier")
            .err()
            .expect("one byte below frontier is rejected");
        assert_eq!(
            error,
            EncodeError::AllocationTooLarge {
                what: "test Tier-1 frontier",
                requested: exact_cap,
                cap,
            }
        );
    }
}
