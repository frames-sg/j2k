// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    ChunkedHtj2kCleanup, CudaDeviceBufferRange, CudaHtj2kProfileReport, CudaQueuedJ2kStoreBatch,
    Error, NativeColorOwnedBatch, PendingDecodeCompletion,
};
use crate::decoder::color_batch::offset_color_batch_error;

pub(super) enum NativeColorBatchOutput {
    Owned(NativeColorOwnedBatch),
    External {
        ranges: Vec<CudaDeviceBufferRange>,
        execution: j2k_cuda_runtime::CudaExecutionStats,
    },
}

pub(super) struct StoredNativeColorBatch {
    pub(super) output: NativeColorBatchOutput,
    pub(super) queued: Option<CudaQueuedJ2kStoreBatch>,
}

pub(super) type NativeColorPendingCompletion = PendingDecodeCompletion<ChunkedHtj2kCleanup>;

/// Code-block jobs that precede a pending chunk within its whole colour group.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct ColorJobIndexBases {
    pub(super) ht: usize,
    pub(super) classic: usize,
}

impl ColorJobIndexBases {
    /// Rebase a chunk-local job failure; source indices are already global.
    pub(super) fn rebase(self, error: Error) -> Error {
        offset_color_batch_error(error, 0, self.ht, self.classic)
    }
}

pub(crate) struct SubmittedNativeColorExternalBatch {
    pub(super) ranges: Vec<CudaDeviceBufferRange>,
    pub(super) execution: j2k_cuda_runtime::CudaExecutionStats,
    pub(super) report: CudaHtj2kProfileReport,
    pub(super) completion: Option<NativeColorPendingCompletion>,
    pub(super) job_index_bases: ColorJobIndexBases,
}

pub(crate) struct SubmittedNativeColorResidentBatch {
    pub(super) output: Option<NativeColorOwnedBatch>,
    pub(super) report: CudaHtj2kProfileReport,
    pub(super) completion: Option<NativeColorPendingCompletion>,
    pub(super) job_index_bases: ColorJobIndexBases,
}

impl SubmittedNativeColorResidentBatch {
    pub(crate) fn is_complete(&self) -> Result<bool, Error> {
        self.completion
            .as_ref()
            .map_or(Ok(true), NativeColorPendingCompletion::is_complete)
            .map_err(|error| self.job_index_bases.rebase(error))
    }

    pub(crate) fn finish(
        mut self,
    ) -> Result<(NativeColorOwnedBatch, CudaHtj2kProfileReport), Error> {
        if let Some(mut completion) = self.completion.take() {
            if let Err(error) = completion.complete() {
                let error = self.job_index_bases.rebase(error);
                if error.completion_is_uncertain() {
                    if let Some(output) = self.output.take() {
                        core::mem::forget(output);
                    }
                }
                return Err(error);
            }
        }
        let output = self.output.take().ok_or(Error::capability_rejected(
            j2k_core::CapabilityRejection::contract_violation(
                "CUDA resident RGB submission lost its output owner",
            ),
        ))?;
        Ok((output, self.report.clone()))
    }
}

impl Drop for SubmittedNativeColorResidentBatch {
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

impl SubmittedNativeColorExternalBatch {
    pub(crate) fn ranges(&self) -> &[CudaDeviceBufferRange] {
        &self.ranges
    }

    pub(crate) fn is_complete(&self) -> Result<bool, Error> {
        self.completion
            .as_ref()
            .map_or(Ok(true), NativeColorPendingCompletion::is_complete)
            .map_err(|error| self.job_index_bases.rebase(error))
    }

    pub(crate) fn finish(
        mut self,
    ) -> Result<(Vec<CudaDeviceBufferRange>, CudaHtj2kProfileReport), Error> {
        if let Some(mut completion) = self.completion.take() {
            completion
                .complete()
                .map_err(|error| self.job_index_bases.rebase(error))?;
        }
        Ok((self.ranges, self.report))
    }
}
