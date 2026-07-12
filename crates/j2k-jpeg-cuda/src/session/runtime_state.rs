// SPDX-License-Identifier: MIT OR Apache-2.0

//! Clone-shared lazy CUDA context and output-pool ownership.

use std::sync::{
    atomic::{AtomicBool, Ordering},
    Mutex, TryLockError,
};

use j2k_cuda_runtime::{CudaBufferPool, CudaContext};

use crate::{runtime::cuda_error, Error};

#[derive(Default)]
pub(super) struct SharedCudaRuntimeState {
    state: Mutex<CudaRuntimeState>,
    initialized: AtomicBool,
}

#[derive(Default)]
struct CudaRuntimeState {
    context: Option<CudaContext>,
    owned_output_pool: Option<CudaBufferPool>,
}

impl SharedCudaRuntimeState {
    pub(super) fn is_initialized(&self) -> bool {
        self.initialized.load(Ordering::Acquire)
    }

    pub(super) fn context(&self) -> Result<CudaContext, Error> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| Error::CudaSessionRuntimePoisoned)?;
        if state.context.is_none() {
            let context = CudaContext::system_default().map_err(cuda_error)?;
            state.context = Some(context);
            self.initialized.store(true, Ordering::Release);
        }
        state.context.clone().ok_or(Error::CudaUnavailable)
    }

    pub(super) fn bind_context(&self, context: &CudaContext) -> Result<CudaContext, Error> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| Error::CudaSessionRuntimePoisoned)?;
        if let Some(existing) = &state.context {
            if !existing.is_same_context(context) {
                return Err(Error::UnsupportedCudaRequest {
                    reason: "CUDA JPEG session is already bound to a different CUDA context",
                });
            }
            return Ok(existing.clone());
        }
        state.context = Some(context.clone());
        self.initialized.store(true, Ordering::Release);
        Ok(context.clone())
    }

    pub(super) fn existing_context(&self) -> Result<Option<CudaContext>, Error> {
        self.state
            .lock()
            .map_err(|_| Error::CudaSessionRuntimePoisoned)
            .map(|state| state.context.clone())
    }

    pub(super) fn owned_output_pool(&self) -> Result<CudaBufferPool, Error> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| Error::CudaSessionRuntimePoisoned)?;
        if state.context.is_none() {
            let context = CudaContext::system_default().map_err(cuda_error)?;
            state.context = Some(context);
            self.initialized.store(true, Ordering::Release);
        }
        if let Some(pool) = &state.owned_output_pool {
            return Ok(pool.clone());
        }
        let pool = state
            .context
            .as_ref()
            .ok_or(Error::CudaUnavailable)?
            .buffer_pool();
        state.owned_output_pool = Some(pool.clone());
        Ok(pool)
    }

    pub(super) fn retained_output_buffers(&self) -> Result<usize, Error> {
        let state = self
            .state
            .lock()
            .map_err(|_| Error::CudaSessionRuntimePoisoned)?;
        state
            .owned_output_pool
            .as_ref()
            .map_or(Ok(0), |pool| pool.cached_count().map_err(cuda_error))
    }

    pub(super) fn try_retained_output_buffers(&self) -> Result<Option<usize>, Error> {
        match self.state.try_lock() {
            Ok(state) => state
                .owned_output_pool
                .as_ref()
                .map_or(Ok(Some(0)), |pool| {
                    pool.cached_count().map(Some).map_err(cuda_error)
                }),
            Err(TryLockError::WouldBlock) => Ok(None),
            Err(TryLockError::Poisoned(_)) => Err(Error::CudaSessionRuntimePoisoned),
        }
    }
}
