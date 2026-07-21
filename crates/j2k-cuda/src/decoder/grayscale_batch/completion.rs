// SPDX-License-Identifier: MIT OR Apache-2.0

//! Grayscale status identity mapping and pending output ownership.

use super::{
    cuda_error, Arc, CudaDeviceBufferRange, CudaHtj2kProfileReport, CudaQueuedJ2kStoreBatch, Error,
    HostPhaseBudget, PendingCleanup, PendingDecodeCompletion, Surface,
};

pub(crate) struct GrayscaleOwnedBatch {
    pub(crate) surfaces: Vec<Surface>,
    pub(crate) buffer: Arc<j2k_cuda_runtime::CudaDeviceBuffer>,
    pub(crate) ranges: Vec<CudaDeviceBufferRange>,
}

pub(super) enum GrayscaleBatchOutput {
    Owned(GrayscaleOwnedBatch),
    External(Vec<CudaDeviceBufferRange>),
}

pub(super) struct StoredGrayscaleBatch {
    pub(super) output: GrayscaleBatchOutput,
    pub(super) queued: Option<CudaQueuedJ2kStoreBatch>,
}

pub(super) type GrayscalePendingCompletion = PendingDecodeCompletion<GrayscaleHtj2kCleanup>;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct GrayscaleJobIdentity {
    pub(super) source_index: usize,
    pub(super) original_job_index: usize,
}

pub(super) struct GrayscaleHtj2kCleanup {
    queued: Option<j2k_cuda_runtime::CudaQueuedHtj2kCleanup>,
    identities: Vec<GrayscaleJobIdentity>,
}

impl GrayscaleHtj2kCleanup {
    pub(super) fn new(
        queued: j2k_cuda_runtime::CudaQueuedHtj2kCleanup,
        identities: Vec<GrayscaleJobIdentity>,
    ) -> Self {
        Self {
            queued: Some(queued),
            identities,
        }
    }

    fn has_status_readback(&self) -> bool {
        self.queued
            .as_ref()
            .is_some_and(|queued| queued.status_count() != 0)
    }

    fn finish(mut self) -> Result<(), Error> {
        let Some(queued) = self.queued.take() else {
            return Ok(());
        };
        queued
            .finish()
            .map(|_| ())
            .map_err(|error| map_grayscale_status_error(error, &self.identities))
    }
}

impl PendingCleanup for GrayscaleHtj2kCleanup {
    fn has_status_readback(&self) -> bool {
        Self::has_status_readback(self)
    }

    fn finish(self) -> Result<(), Error> {
        Self::finish(self)
    }
}

impl Drop for GrayscaleHtj2kCleanup {
    fn drop(&mut self) {
        if let Some(queued) = self.queued.take() {
            let _ = queued.finish();
        }
    }
}

/// Pending external grayscale batch whose internal codec resources remain live
/// until the asynchronous final store completes.
pub(crate) struct SubmittedGrayscaleExternalBatch {
    pub(super) ranges: Vec<CudaDeviceBufferRange>,
    pub(super) report: CudaHtj2kProfileReport,
    pub(super) completion: Option<GrayscalePendingCompletion>,
}

pub(crate) struct SubmittedGrayscaleResidentBatch {
    pub(super) output: Option<GrayscaleOwnedBatch>,
    pub(super) report: CudaHtj2kProfileReport,
    pub(super) completion: Option<GrayscalePendingCompletion>,
}

impl SubmittedGrayscaleResidentBatch {
    pub(crate) fn is_complete(&self) -> Result<bool, Error> {
        self.completion
            .as_ref()
            .map_or(Ok(true), GrayscalePendingCompletion::is_complete)
    }

    pub(crate) fn finish(mut self) -> Result<(GrayscaleOwnedBatch, CudaHtj2kProfileReport), Error> {
        if let Some(mut completion) = self.completion.take() {
            if let Err(error) = completion.complete() {
                if error.completion_is_uncertain() {
                    if let Some(output) = self.output.take() {
                        core::mem::forget(output);
                    }
                }
                return Err(error);
            }
        }
        let output = self.output.take().ok_or(Error::UnsupportedCudaRequest {
            reason: "CUDA resident grayscale submission lost its output owner",
        })?;
        Ok((output, self.report.clone()))
    }
}

impl Drop for SubmittedGrayscaleResidentBatch {
    fn drop(&mut self) {
        let result = self
            .completion
            .take()
            .map_or(Ok(()), |mut completion| completion.complete());
        if result.is_err_and(|error| error.completion_is_uncertain()) {
            if let Some(output) = self.output.take() {
                core::mem::forget(output);
            }
        }
    }
}

impl SubmittedGrayscaleExternalBatch {
    pub(crate) fn ranges(&self) -> &[CudaDeviceBufferRange] {
        &self.ranges
    }

    pub(crate) fn is_complete(&self) -> Result<bool, Error> {
        self.completion
            .as_ref()
            .map_or(Ok(true), GrayscalePendingCompletion::is_complete)
    }

    pub(crate) fn finish(
        mut self,
    ) -> Result<(Vec<CudaDeviceBufferRange>, CudaHtj2kProfileReport), Error> {
        if let Some(mut completion) = self.completion.take() {
            completion.complete()?;
        }
        Ok((self.ranges, self.report))
    }
}

pub(super) fn grayscale_htj2k_job_identities(
    component_work: &[super::super::CudaComponentDecodeWork],
    source_indices: &[usize],
    live_host_bytes: usize,
) -> Result<Vec<GrayscaleJobIdentity>, Error> {
    if component_work.len() != source_indices.len() {
        return Err(Error::UnsupportedCudaRequest {
            reason: "CUDA grayscale source identity count does not match component work",
        });
    }
    let job_count = component_work
        .iter()
        .flat_map(|work| &work.pending_dequant_bands)
        .try_fold(0usize, |count, pending| {
            count.checked_add(pending.jobs.len())
        })
        .ok_or(Error::HostAllocationFailed {
            bytes: usize::MAX,
            what: "CUDA grayscale HTJ2K status identities",
        })?;
    let mut budget = HostPhaseBudget::with_live_bytes(
        "CUDA grayscale HTJ2K status identities",
        live_host_bytes,
    )?;
    let mut identities = budget.try_vec_with_capacity(job_count)?;
    for (work, source_index) in component_work.iter().zip(source_indices.iter().copied()) {
        for pending in &work.pending_dequant_bands {
            for _ in &pending.jobs {
                identities.push(GrayscaleJobIdentity {
                    source_index,
                    original_job_index: identities.len(),
                });
            }
        }
    }
    Ok(identities)
}

pub(super) fn map_grayscale_status_error(
    error: j2k_cuda_runtime::CudaError,
    identities: &[GrayscaleJobIdentity],
) -> Error {
    let Some(job_index) = error.kernel_job_index() else {
        return cuda_error(error);
    };
    let Some(identity) = identities.get(job_index) else {
        return cuda_error(error);
    };
    Error::CudaTier1JobFailed {
        source_index: identity.source_index,
        original_job_index: identity.original_job_index,
        source: error,
    }
}
