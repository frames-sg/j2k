// SPDX-License-Identifier: MIT OR Apache-2.0

mod enqueue;
mod planning;

#[cfg(test)]
mod tests;

pub(in crate::decoder) use enqueue::enqueue_chunked_htj2k_cleanup_dequant;

use j2k_cuda_runtime::{CudaHtj2kDecodeResources, CudaQueuedHtj2kCleanupGroup};

use super::super::{combine_cuda_cleanup_errors, cuda_error, Error};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Htj2kChunkJobIdentity {
    original_job_index: usize,
    source_index: usize,
}

impl Htj2kChunkJobIdentity {
    const fn new(original_job_index: usize, source_index: usize) -> Self {
        Self {
            original_job_index,
            source_index,
        }
    }
}

/// Pending pass-homogeneous HTJ2K chunks retained through final-store completion.
pub(in crate::decoder) struct ChunkedHtj2kCleanup {
    pub(super) group: Option<CudaQueuedHtj2kCleanupGroup>,
    pub(super) resources: Vec<CudaHtj2kDecodeResources>,
    pub(super) identities: Vec<Htj2kChunkJobIdentity>,
    pub(super) chunk_count: usize,
    pub(super) dequant_chunk_count: usize,
}

impl ChunkedHtj2kCleanup {
    pub(in crate::decoder) fn has_status_readback(&self) -> bool {
        self.group
            .as_ref()
            .is_some_and(|group| group.status_count() != 0)
    }

    /// Number of bounded compressed/descriptor arenas submitted for this group.
    #[cfg(test)]
    pub(in crate::decoder) fn chunk_count(&self) -> usize {
        self.chunk_count
    }

    pub(in crate::decoder) fn finish(mut self) -> Result<(), Error> {
        self.finish_remaining()
    }

    fn finish_remaining(&mut self) -> Result<(), Error> {
        let Some(group) = self.group.take() else {
            self.resources.clear();
            return Ok(());
        };
        match group.finish() {
            Ok(_) => {
                self.resources.clear();
                Ok(())
            }
            Err(error) => {
                if error.completion_is_uncertain() {
                    for resources in self.resources.drain(..) {
                        core::mem::forget(resources);
                    }
                } else {
                    self.resources.clear();
                }
                Err(map_chunk_status_error(error, &self.identities))
            }
        }
    }

    pub(super) fn finish_after_error(mut self, primary: Error) -> Error {
        match self.finish_remaining() {
            Ok(()) => primary,
            Err(cleanup) => combine_cuda_cleanup_errors(primary, cleanup),
        }
    }
}

impl Drop for ChunkedHtj2kCleanup {
    fn drop(&mut self) {
        let _ = self.finish_remaining();
    }
}

fn map_chunk_status_error(
    error: j2k_cuda_runtime::CudaError,
    identities: &[Htj2kChunkJobIdentity],
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
