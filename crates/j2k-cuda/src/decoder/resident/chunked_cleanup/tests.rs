// SPDX-License-Identifier: MIT OR Apache-2.0

use super::planning::{
    sort_selected_jobs_for_coalesced_targets, Htj2kJobLocation, SelectedHtj2kChunkJob,
};
use super::{map_chunk_status_error, Htj2kChunkJobIdentity};
use crate::Error;
use j2k_cuda_runtime::CudaError;

#[test]
fn chunk_local_kernel_failure_maps_to_original_job_and_source() {
    let identities = [
        Htj2kChunkJobIdentity::new(11, 4),
        Htj2kChunkJobIdentity::new(29, 8),
    ];
    let mapped = map_chunk_status_error(
        CudaError::KernelJobStatus {
            kernel: "injected",
            job_index: 1,
            code: 7,
            detail: 13,
        },
        &identities,
    );
    assert!(matches!(
        mapped,
        Error::CudaTier1JobFailed {
            source_index: 8,
            original_job_index: 29,
            ..
        }
    ));
}

#[test]
fn chunk_jobs_for_one_coefficient_band_are_adjacent_for_one_runtime_target() {
    let mut selected = [
        selected_job(0, 2, 1, 7),
        selected_job(0, 1, 4, 3),
        selected_job(0, 2, 0, 6),
        selected_job(1, 0, 0, 8),
    ];

    sort_selected_jobs_for_coalesced_targets(&mut selected);

    assert_eq!(
        selected.map(|selected| (
            selected.location.work,
            selected.location.pending,
            selected.location.job,
        )),
        [(0, 1, 4), (0, 2, 0), (0, 2, 1), (1, 0, 0)]
    );
}

fn selected_job(
    work_index: usize,
    pending_index: usize,
    job_index: usize,
    original_job_index: usize,
) -> SelectedHtj2kChunkJob {
    SelectedHtj2kChunkJob {
        location: Htj2kJobLocation {
            work: work_index,
            pending: pending_index,
            job: job_index,
            source: 5,
        },
        original_job_index,
        source_index: 5,
    }
}
