// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    ChunkedHtj2kCleanup, CudaDeviceBufferRange, CudaHtj2kProfileReport, CudaQueuedJ2kStoreBatch,
    Error, NativeColorOwnedBatch, PendingDecodeCompletion,
};

pub(super) enum NativeColorBatchOutput {
    Owned(NativeColorOwnedBatch),
    External(Vec<CudaDeviceBufferRange>),
}

pub(super) struct StoredNativeColorBatch {
    pub(super) output: NativeColorBatchOutput,
    pub(super) queued: Option<CudaQueuedJ2kStoreBatch>,
}

pub(super) type NativeColorPendingCompletion = PendingDecodeCompletion<ChunkedHtj2kCleanup>;

pub(crate) struct SubmittedNativeColorExternalBatch {
    pub(super) ranges: Vec<CudaDeviceBufferRange>,
    pub(super) report: CudaHtj2kProfileReport,
    pub(super) completion: Option<NativeColorPendingCompletion>,
}

pub(crate) struct SubmittedNativeColorResidentBatch {
    pub(super) output: Option<NativeColorOwnedBatch>,
    pub(super) report: CudaHtj2kProfileReport,
    pub(super) completion: Option<NativeColorPendingCompletion>,
}

impl SubmittedNativeColorResidentBatch {
    pub(crate) fn is_complete(&self) -> Result<bool, Error> {
        self.completion
            .as_ref()
            .map_or(Ok(true), NativeColorPendingCompletion::is_complete)
    }

    pub(crate) fn finish(
        mut self,
    ) -> Result<(NativeColorOwnedBatch, CudaHtj2kProfileReport), Error> {
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
            reason: "CUDA resident RGB submission lost its output owner",
        })?;
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
