// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared lifetime state for asynchronous resident and external batch completion.

use j2k_cuda_runtime::{CudaHtj2kDecodeResources, CudaQueuedJ2kStoreBatch};

use super::resident::{ChunkedHtj2kCleanup, QueuedComponentClassicDecode};
use super::{
    combine_cuda_cleanup_errors, cuda_error, CudaDecodedComponent, CudaQueuedIdwtBatch, Error,
};

pub(in crate::decoder) trait PendingCleanup: Sized {
    fn has_status_readback(&self) -> bool;
    fn finish(self) -> Result<(), Error>;
}

impl PendingCleanup for ChunkedHtj2kCleanup {
    fn has_status_readback(&self) -> bool {
        Self::has_status_readback(self)
    }

    fn finish(self) -> Result<(), Error> {
        Self::finish(self)
    }
}

pub(in crate::decoder) fn retire_decode_after_error<C: PendingCleanup>(
    error: Error,
    idwt: Option<CudaQueuedIdwtBatch>,
    cleanup: Option<C>,
    classic: Option<QueuedComponentClassicDecode>,
) -> Error {
    let mut completion_error = Some(error);
    if let Some(idwt) = idwt {
        if let Err(error) = idwt.finish() {
            accumulate_completion_error(&mut completion_error, error);
        }
    }
    if let Some(cleanup) = cleanup {
        if let Err(error) = cleanup.finish() {
            accumulate_completion_error(&mut completion_error, error);
        }
    }
    if let Some(classic) = classic {
        if let Err(error) = classic.finish() {
            accumulate_completion_error(&mut completion_error, error);
        }
    }
    completion_error.expect("primary decode error is always present")
}

pub(in crate::decoder) fn finish_decode_statuses<C: PendingCleanup>(
    cleanup: Option<C>,
    classic: Option<QueuedComponentClassicDecode>,
) -> Result<(), Error> {
    let mut completion_error = None;
    if let Some(cleanup) = cleanup {
        if let Err(error) = cleanup.finish() {
            accumulate_completion_error(&mut completion_error, error);
        }
    }
    if let Some(classic) = classic {
        if let Err(error) = classic.finish() {
            accumulate_completion_error(&mut completion_error, error);
        }
    }
    completion_error.map_or(Ok(()), Err)
}

pub(in crate::decoder) struct PendingDecodeCompletion<C: PendingCleanup> {
    store: Option<CudaQueuedJ2kStoreBatch>,
    idwt: Option<CudaQueuedIdwtBatch>,
    cleanup: Option<C>,
    classic: Option<QueuedComponentClassicDecode>,
    decoded: Option<Vec<CudaDecodedComponent>>,
    resources: Option<CudaHtj2kDecodeResources>,
}

impl<C: PendingCleanup> PendingDecodeCompletion<C> {
    pub(in crate::decoder) fn new(
        store: Option<CudaQueuedJ2kStoreBatch>,
        idwt: Option<CudaQueuedIdwtBatch>,
        cleanup: Option<C>,
        classic: Option<QueuedComponentClassicDecode>,
        decoded: Vec<CudaDecodedComponent>,
        resources: Option<CudaHtj2kDecodeResources>,
    ) -> Self {
        Self {
            store,
            idwt,
            cleanup,
            classic,
            decoded: Some(decoded),
            resources,
        }
    }

    pub(in crate::decoder) fn is_complete(&self) -> Result<bool, Error> {
        self.store
            .as_ref()
            .map_or(Ok(true), CudaQueuedJ2kStoreBatch::is_complete)
            .map_err(cuda_error)
    }

    fn abandon_remaining(&mut self) {
        if let Some(idwt) = self.idwt.take() {
            core::mem::forget(idwt);
        }
        if let Some(cleanup) = self.cleanup.take() {
            core::mem::forget(cleanup);
        }
        if let Some(classic) = self.classic.take() {
            core::mem::forget(classic);
        }
        if let Some(decoded) = self.decoded.take() {
            core::mem::forget(decoded);
        }
        if let Some(resources) = self.resources.take() {
            core::mem::forget(resources);
        }
    }

    pub(in crate::decoder) fn complete(&mut self) -> Result<(), Error> {
        let has_status_readback = self
            .cleanup
            .as_ref()
            .is_some_and(PendingCleanup::has_status_readback)
            || self
                .classic
                .as_ref()
                .is_some_and(QueuedComponentClassicDecode::has_status_readback);
        let mut completion_error = None;
        if let Some(cleanup) = self.cleanup.take() {
            if let Err(error) = cleanup.finish() {
                accumulate_completion_error(&mut completion_error, error);
            }
        }
        if let Some(classic) = self.classic.take() {
            if let Err(error) = classic.finish() {
                accumulate_completion_error(&mut completion_error, error);
            }
        }
        let status_established_completion = has_status_readback
            && completion_error
                .as_ref()
                .is_none_or(|error: &Error| !error.completion_is_uncertain());
        if let Some(store) = self.store.take() {
            let store_result = if status_established_completion {
                // SAFETY: the group status operation is ordered after the
                // final store on the same stream and established completion.
                unsafe {
                    store.release_after_stream_completion();
                }
                Ok(())
            } else {
                store.finish().map(|_| ()).map_err(cuda_error)
            };
            if let Err(error) = store_result {
                accumulate_completion_error(&mut completion_error, error);
                self.abandon_remaining();
                return Err(completion_error.expect("store error was recorded"));
            }
        }
        if let Some(mut idwt) = self.idwt.take() {
            if let Err(error) = idwt.release_after_completion() {
                accumulate_completion_error(&mut completion_error, error);
            }
        }
        self.decoded.take();
        self.resources.take();
        completion_error.map_or(Ok(()), Err)
    }
}

fn accumulate_completion_error(current: &mut Option<Error>, error: Error) {
    *current = Some(match current.take() {
        Some(primary) => combine_cuda_cleanup_errors(primary, error),
        None => error,
    });
}

impl<C: PendingCleanup> Drop for PendingDecodeCompletion<C> {
    fn drop(&mut self) {
        let _ = self.complete();
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    };

    use super::{retire_decode_after_error, PendingCleanup};
    use crate::Error;

    struct FailingCleanup {
        finished: Arc<AtomicBool>,
    }

    impl PendingCleanup for FailingCleanup {
        fn has_status_readback(&self) -> bool {
            false
        }

        fn finish(self) -> Result<(), Error> {
            self.finished.store(true, Ordering::Relaxed);
            Err(Error::UnsupportedCudaRequest {
                reason: "cleanup failure",
            })
        }
    }

    #[test]
    fn error_retirement_attempts_cleanup_and_preserves_both_errors() {
        let finished = Arc::new(AtomicBool::new(false));
        let error = retire_decode_after_error(
            Error::UnsupportedCudaRequest {
                reason: "primary failure",
            },
            None,
            Some(FailingCleanup {
                finished: finished.clone(),
            }),
            None,
        );

        assert!(finished.load(Ordering::Relaxed));
        let Error::CudaCleanupFailed { primary, cleanup } = error else {
            panic!("retirement must preserve primary and cleanup errors")
        };
        assert!(matches!(
            *primary,
            Error::UnsupportedCudaRequest {
                reason: "primary failure"
            }
        ));
        assert!(matches!(
            *cleanup,
            Error::UnsupportedCudaRequest {
                reason: "cleanup failure"
            }
        ));
    }
}
