// SPDX-License-Identifier: MIT OR Apache-2.0

mod cache_policy;
mod readback;
mod reuse_guard;
mod size_buckets;

use self::cache_policy::{
    checked_deferred_bytes, observe_deferred_high_water, CudaBufferPoolMetrics,
};
pub use self::cache_policy::{CudaBufferPoolDiagnostics, CudaBufferPoolLimits};
#[cfg(test)]
pub(crate) use self::readback::copy_pooled_bytes_to_vec_uninit;
pub(crate) use self::readback::copy_pooled_bytes_to_vec_uninit_with_budget;
pub(crate) use self::reuse_guard::CudaBufferPoolReuseGuard;
use self::size_buckets::CudaBufferPoolSizeBuckets;
use super::{pinned_staging::select_pinned_upload_result, CudaDeviceBuffer};
use crate::{
    allocation::host_allocation_error,
    bytes::{f32_slice_as_bytes, i16_slice_as_bytes},
    context::CudaContext,
    error::CudaError,
};
use std::{
    ffi::c_void,
    sync::{Arc, Mutex},
};

/// Reusable CUDA device-buffer pool for repeated adapter dispatches.
#[derive(Clone, Debug)]
pub struct CudaBufferPool {
    pub(crate) inner: Arc<CudaBufferPoolInner>,
}

#[derive(Debug)]
pub(crate) struct CudaBufferPoolInner {
    pub(crate) context: CudaContext,
    pub(crate) limits: CudaBufferPoolLimits,
    pub(crate) state: Mutex<CudaBufferPoolState>,
}

#[derive(Debug)]
pub(crate) struct CudaBufferPoolState {
    pub(crate) free: CudaBufferPoolFree,
    pub(crate) deferred: Vec<CudaDeviceBuffer>,
    pub(crate) deferred_bytes: usize,
    pub(crate) reuse_holds: usize,
    metrics: CudaBufferPoolMetrics,
}

#[derive(Debug)]
pub(crate) enum CudaBufferPoolFree {
    FirstFit(Vec<CudaDeviceBuffer>),
    SizeBuckets(CudaBufferPoolSizeBuckets),
}

impl CudaBufferPoolInner {
    fn recycle_buffer(&self, buffer: CudaDeviceBuffer) -> Result<(), CudaError> {
        if !buffer.is_owned_by(&self.context) {
            return Err(CudaError::InvalidArgument {
                message: "CUDA buffer must belong to the pool's context".to_string(),
            });
        }
        if let Err(error) = self.context.inner.ensure_resource_lifetime_available() {
            drop(buffer);
            return Err(error);
        }
        let mut state = match self.state.lock() {
            Ok(state) => state,
            Err(error) => {
                // Pool invariants are unknown after poisoning. Retain the
                // allocation instead of letting it be freed while queued CUDA
                // work may still reference it.
                std::mem::forget(buffer);
                return Err(CudaError::StatePoisoned {
                    message: error.to_string(),
                });
            }
        };
        if state.reuse_holds != 0 {
            let deferred_bytes =
                match checked_deferred_bytes(state.deferred_bytes, buffer.byte_len()) {
                    Ok(bytes) => bytes,
                    Err(error) => {
                        // Queued work may still reference the allocation. Preserve
                        // the token when even the safety ledger cannot represent it.
                        std::mem::forget(buffer);
                        return Err(error);
                    }
                };
            if state.deferred.try_reserve(1).is_err() {
                let error = host_allocation_error::<CudaDeviceBuffer>(
                    state.deferred.len().saturating_add(1),
                );
                // Queued CUDA work may still reference this allocation. If
                // retention metadata cannot grow, intentionally leak the
                // device token instead of running its destructor early.
                std::mem::forget(buffer);
                return Err(error);
            }
            state.deferred.push(buffer);
            state.deferred_bytes = deferred_bytes;
            observe_deferred_high_water(&mut state);
            return Ok(());
        }
        drop(state);
        self.recycle_completed_buffer(buffer)
    }

    fn release_reuse_hold(&self) -> Result<(), CudaError> {
        let deferred = {
            let mut state = self
                .state
                .lock()
                .map_err(|error| CudaError::StatePoisoned {
                    message: error.to_string(),
                })?;
            release_reuse_hold_state(&mut state)?
        };
        if let Some(deferred) = deferred {
            // Completion is established before the final hold is released.
            // Admission and device release both happen outside the state lock.
            for buffer in deferred {
                self.recycle_completed_buffer(buffer)?;
            }
        }
        Ok(())
    }
}

fn acquire_reuse_hold(state: &mut CudaBufferPoolState) -> Result<(), CudaError> {
    state.reuse_holds =
        state
            .reuse_holds
            .checked_add(1)
            .ok_or_else(|| CudaError::InvalidArgument {
                message: "CUDA buffer pool reuse hold count overflow".to_string(),
            })?;
    Ok(())
}

fn release_reuse_hold_state(
    state: &mut CudaBufferPoolState,
) -> Result<Option<Vec<CudaDeviceBuffer>>, CudaError> {
    if state.reuse_holds == 0 {
        return Err(CudaError::InvalidArgument {
            message: "CUDA buffer pool reuse hold is already released".to_string(),
        });
    }
    state.reuse_holds -= 1;
    if state.reuse_holds == 0 {
        let deferred = std::mem::take(&mut state.deferred);
        state.deferred_bytes = 0;
        return Ok(Some(deferred));
    }
    Ok(None)
}

#[cfg(test)]
mod reuse_hold_tests {
    use super::*;

    #[test]
    fn nested_pool_reuse_holds_release_only_at_zero() {
        let mut state = CudaBufferPoolState {
            free: CudaBufferPoolFree::FirstFit(Vec::new()),
            deferred: Vec::new(),
            deferred_bytes: 0,
            reuse_holds: 0,
            metrics: CudaBufferPoolMetrics::default(),
        };

        acquire_reuse_hold(&mut state).expect("first reuse hold");
        acquire_reuse_hold(&mut state).expect("nested reuse hold");
        assert_eq!(state.reuse_holds, 2);

        assert!(release_reuse_hold_state(&mut state)
            .expect("release nested hold")
            .is_none());
        assert_eq!(state.reuse_holds, 1);
        assert!(release_reuse_hold_state(&mut state)
            .expect("release final hold")
            .is_some());
        assert_eq!(state.reuse_holds, 0);
        assert_eq!(state.deferred_bytes, 0);
        assert!(matches!(
            release_reuse_hold_state(&mut state),
            Err(CudaError::InvalidArgument { .. })
        ));
    }
}

#[doc(hidden)]
/// Diagnostics for one traced [`CudaBufferPool`] acquisition.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CudaBufferPoolTakeTrace {
    /// Requested byte length for the checkout.
    pub requested_len: usize,
    /// Number of cached free buffers before the checkout.
    pub free_count_before: usize,
    /// Number of cached entries examined while finding a reusable buffer or allocating.
    pub scanned_count: usize,
    /// Whether the checkout reused a cached allocation.
    pub reused: bool,
    /// Actual allocation byte length backing the checkout.
    pub allocation_byte_len: usize,
}

impl CudaBufferPool {
    /// Acquire a device buffer with at least `len` bytes.
    pub fn take(&self, len: usize) -> Result<CudaPooledDeviceBuffer, CudaError> {
        self.inner
            .context
            .inner
            .ensure_resource_lifetime_available()?;
        let mut state = self
            .inner
            .state
            .lock()
            .map_err(|error| CudaError::StatePoisoned {
                message: error.to_string(),
            })?;
        let (reusable_buffer, _) = pool_take_fit_buffer(&mut state.free, len);
        let buffer = if let Some(buffer) = reusable_buffer {
            buffer
        } else {
            drop(state);
            self.inner.context.allocate(len)?
        };
        Ok(CudaPooledDeviceBuffer {
            buffer: Some(buffer),
            requested_len: len,
            pool: self.inner.clone(),
        })
    }

    /// Return a raw device buffer to this pool.
    pub fn recycle(&self, buffer: CudaDeviceBuffer) -> Result<(), CudaError> {
        self.inner.recycle_buffer(buffer)
    }

    #[doc(hidden)]
    /// Acquire a device buffer with diagnostics for profiling pool behavior.
    pub fn take_with_trace(
        &self,
        len: usize,
    ) -> Result<(CudaPooledDeviceBuffer, CudaBufferPoolTakeTrace), CudaError> {
        self.inner
            .context
            .inner
            .ensure_resource_lifetime_available()?;
        let mut state = self
            .inner
            .state
            .lock()
            .map_err(|error| CudaError::StatePoisoned {
                message: error.to_string(),
            })?;
        let free_count_before = state.free.cached_count();
        let (reusable_buffer, scanned_count) = pool_take_fit_buffer(&mut state.free, len);
        let reused = reusable_buffer.is_some();
        let buffer = if let Some(buffer) = reusable_buffer {
            buffer
        } else {
            drop(state);
            self.inner.context.allocate(len)?
        };
        let allocation_byte_len = buffer.byte_len();
        let trace = CudaBufferPoolTakeTrace {
            requested_len: len,
            free_count_before,
            scanned_count,
            reused,
            allocation_byte_len,
        };
        Ok((
            CudaPooledDeviceBuffer {
                buffer: Some(buffer),
                requested_len: len,
                pool: self.inner.clone(),
            },
            trace,
        ))
    }

    /// Upload host bytes into a pooled device buffer.
    pub fn upload(&self, bytes: &[u8]) -> Result<CudaPooledDeviceBuffer, CudaError> {
        let buffer = self.take(bytes.len())?;
        if !bytes.is_empty() {
            self.inner
                .context
                .inner
                .with_current_resource_operation(|| {
                    // SAFETY: `buffer` is a live device allocation with at
                    // least `bytes.len()` bytes, and `bytes` covers that many
                    // host bytes while the lifecycle gate is held.
                    let result = unsafe {
                        (self.inner.context.inner.driver.cu_memcpy_htod)(
                            buffer.device_ptr(),
                            bytes.as_ptr().cast::<c_void>(),
                            bytes.len(),
                        )
                    };
                    self.inner
                        .context
                        .inner
                        .driver
                        .check("cuMemcpyHtoD_v2", result)
                })?;
            self.inner.context.record_host_to_device_copy(bytes.len());
        }
        Ok(buffer)
    }

    /// Upload host bytes through temporary page-locked staging into a pooled device buffer.
    pub fn upload_pinned(&self, bytes: &[u8]) -> Result<CudaPooledDeviceBuffer, CudaError> {
        if bytes.is_empty() {
            return self.upload(bytes);
        }

        let operation = self.inner.context.begin_pinned_upload_operation()?;
        let buffer = self.take(bytes.len())?;
        let mut staging = operation.prepare_upload(bytes.len())?;
        staging.copy_from_slice(bytes)?;
        let staging_bytes = staging.as_slice()?;
        let upload_result = self
            .inner
            .context
            .inner
            .with_current_resource_operation(|| {
                // SAFETY: `buffer` is a live device allocation with at least
                // `bytes.len()` bytes, the pinned staging slice covers that
                // range, and the lifecycle gate is held.
                let result = unsafe {
                    (self.inner.context.inner.driver.cu_memcpy_htod)(
                        buffer.device_ptr(),
                        staging_bytes.as_ptr().cast::<c_void>(),
                        bytes.len(),
                    )
                };
                self.inner
                    .context
                    .inner
                    .driver
                    .check("cuMemcpyHtoD_v2", result)
            });
        if upload_result.is_ok() {
            self.inner.context.record_host_to_device_copy(bytes.len());
        }
        let recycle_result = staging.recycle();
        select_pinned_upload_result(upload_result.map(|()| buffer), recycle_result)
    }

    /// Upload host `f32` samples into a pooled device buffer.
    pub fn upload_f32(&self, samples: &[f32]) -> Result<CudaPooledDeviceBuffer, CudaError> {
        self.upload(f32_slice_as_bytes(samples))
    }

    /// Upload host `f32` samples through pinned staging into a pooled device buffer.
    pub fn upload_f32_pinned(&self, samples: &[f32]) -> Result<CudaPooledDeviceBuffer, CudaError> {
        self.upload_pinned(f32_slice_as_bytes(samples))
    }

    #[doc(hidden)]
    /// Upload host `i16` samples into a pooled device buffer.
    pub fn upload_i16(&self, samples: &[i16]) -> Result<CudaPooledDeviceBuffer, CudaError> {
        self.upload(i16_slice_as_bytes(samples))
    }

    #[doc(hidden)]
    /// Upload host `i16` samples through pinned staging into a pooled device buffer.
    pub fn upload_i16_pinned(&self, samples: &[i16]) -> Result<CudaPooledDeviceBuffer, CudaError> {
        self.upload_pinned(i16_slice_as_bytes(samples))
    }

    /// Number of free buffers currently cached by the pool.
    pub fn cached_count(&self) -> Result<usize, CudaError> {
        self.inner
            .context
            .inner
            .ensure_resource_lifetime_available()?;
        Ok(self
            .inner
            .state
            .lock()
            .map_err(|error| CudaError::StatePoisoned {
                message: error.to_string(),
            })?
            .free
            .cached_count())
    }

    pub(crate) fn defer_reuse(&self) -> Result<CudaBufferPoolReuseGuard, CudaError> {
        self.inner
            .context
            .inner
            .ensure_resource_lifetime_available()?;
        let mut state = self
            .inner
            .state
            .lock()
            .map_err(|error| CudaError::StatePoisoned {
                message: error.to_string(),
            })?;
        acquire_reuse_hold(&mut state)?;
        drop(state);
        Ok(CudaBufferPoolReuseGuard {
            pool: self.inner.clone(),
            active: true,
        })
    }

    pub(crate) fn is_owned_by(&self, context: &CudaContext) -> bool {
        self.inner.context.is_same_context(context)
    }
}

impl CudaBufferPoolFree {
    fn cached_count(&self) -> usize {
        match self {
            Self::FirstFit(free) => free.len(),
            Self::SizeBuckets(free) => free.cached_count(),
        }
    }
}

pub(crate) fn pool_take_fit_buffer(
    free: &mut CudaBufferPoolFree,
    len: usize,
) -> (Option<CudaDeviceBuffer>, usize) {
    match free {
        CudaBufferPoolFree::FirstFit(free) => pool_take_first_fit_buffer(free, len),
        CudaBufferPoolFree::SizeBuckets(free) => free.take(len),
    }
}

pub(crate) fn pool_take_first_fit_buffer(
    free: &mut Vec<CudaDeviceBuffer>,
    len: usize,
) -> (Option<CudaDeviceBuffer>, usize) {
    let mut examined = 0usize;
    for (index, buffer) in free.iter().enumerate() {
        examined = examined.saturating_add(1);
        if buffer.byte_len() >= len {
            return (Some(free.remove(index)), examined);
        }
    }
    (None, examined)
}

#[cfg(test)]
pub(crate) fn pool_fit_buffer_index_by_len<I>(lengths: I, len: usize) -> Option<usize>
where
    I: IntoIterator<Item = (usize, usize)>,
{
    let lengths = lengths.into_iter().collect::<Vec<_>>();
    let mut left = 0usize;
    let mut right = lengths.len();
    while left < right {
        let mid = left + (right - left) / 2;
        if lengths[mid].1 < len {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    (left < lengths.len()).then_some(lengths[left].0)
}

/// Device buffer borrowed from a [`CudaBufferPool`].
#[derive(Debug)]
pub struct CudaPooledDeviceBuffer {
    pub(crate) buffer: Option<CudaDeviceBuffer>,
    pub(crate) requested_len: usize,
    pub(crate) pool: Arc<CudaBufferPoolInner>,
}

impl CudaPooledDeviceBuffer {
    /// Raw CUDA device pointer value for kernel argument binding.
    pub fn device_ptr(&self) -> u64 {
        self.buffer.as_ref().map_or(0, CudaDeviceBuffer::device_ptr)
    }

    /// Requested byte length for the current checkout.
    pub fn byte_len(&self) -> usize {
        self.requested_len
    }

    /// Actual device allocation byte length.
    pub fn allocation_byte_len(&self) -> usize {
        self.buffer.as_ref().map_or(0, CudaDeviceBuffer::byte_len)
    }

    /// Borrow the underlying device buffer while the checkout is live.
    pub fn as_device_buffer(&self) -> Option<&CudaDeviceBuffer> {
        self.buffer.as_ref()
    }

    /// Detach and return the underlying device buffer instead of recycling it
    /// when this checkout is dropped.
    pub fn into_device_buffer(mut self) -> Result<CudaDeviceBuffer, CudaError> {
        self.buffer
            .take()
            .ok_or_else(|| CudaError::InvalidArgument {
                message: "pooled CUDA buffer checkout is empty".to_string(),
            })
    }

    /// Copy the requested bytes for this checkout into caller-owned host output.
    pub fn copy_to_host(&self, out: &mut [u8]) -> Result<(), CudaError> {
        if out.len() < self.requested_len {
            return Err(CudaError::OutputTooSmall {
                required: self.requested_len,
                have: out.len(),
            });
        }
        if self.requested_len == 0 {
            return Ok(());
        }
        let buffer = self
            .buffer
            .as_ref()
            .ok_or_else(|| CudaError::InvalidArgument {
                message: "pooled CUDA buffer checkout is empty".to_string(),
            })?;
        buffer.context.inner.with_current_resource_operation(|| {
            // SAFETY: `buffer.ptr` is a live allocation with at least
            // `requested_len` bytes and `out` covers that range while the
            // lifecycle gate is held.
            let result = unsafe {
                (buffer.context.inner.driver.cu_memcpy_dtoh)(
                    out.as_mut_ptr().cast::<c_void>(),
                    buffer.ptr,
                    self.requested_len,
                )
            };
            buffer.context.inner.driver.check("cuMemcpyDtoH_v2", result)
        })?;
        buffer
            .context
            .record_device_to_host_copy(self.requested_len);
        Ok(())
    }
}

impl Drop for CudaPooledDeviceBuffer {
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            let _ = self.pool.recycle_buffer(buffer);
        }
    }
}

pub(crate) fn pooled_device_buffer(
    buffer: &CudaPooledDeviceBuffer,
) -> Result<&CudaDeviceBuffer, CudaError> {
    buffer
        .as_device_buffer()
        .ok_or_else(|| CudaError::InvalidArgument {
            message: "pooled CUDA buffer checkout is empty".to_string(),
        })
}
