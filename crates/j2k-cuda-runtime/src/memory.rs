// SPDX-License-Identifier: MIT OR Apache-2.0

mod pinned_staging;
mod pool;
mod ranges;

pub(crate) use self::pinned_staging::PinnedUploadStagingPool;
pub use self::pinned_staging::{
    CudaPinnedUploadOperationGuard, CudaPinnedUploadStagingCheckout,
    CudaPinnedUploadStagingPoolDiagnostics, CudaPinnedUploadStagingPoolLimits,
};
#[cfg(test)]
pub(crate) use self::pool::copy_pooled_bytes_to_vec_uninit;
#[cfg(test)]
pub(crate) use self::pool::pool_fit_buffer_index_by_len;
pub(crate) use self::pool::{
    copy_pooled_bytes_to_vec_uninit_with_budget, pooled_device_buffer, CudaBufferPoolReuseGuard,
};
pub use self::pool::{
    CudaBufferPool, CudaBufferPoolDiagnostics, CudaBufferPoolLimits, CudaBufferPoolTakeTrace,
    CudaPooledDeviceBuffer,
};
pub(crate) use self::ranges::CheckedDeviceBufferRanges;

#[cfg(test)]
use crate::context::validate_non_null_pinned_host_allocation;
use crate::{
    bytes::f32_slice_as_bytes, context::CudaContext, driver::CuDevicePtr, error::CudaError,
};
use std::ffi::c_void;

impl CudaContext {
    /// Upload host bytes into a CUDA device buffer.
    pub fn upload(&self, bytes: &[u8]) -> Result<CudaDeviceBuffer, CudaError> {
        let mut ptr = 0;
        let buffer = if bytes.is_empty() {
            self.inner.set_current()?;
            CudaDeviceBuffer {
                context: self.clone(),
                ptr,
                len: bytes.len(),
            }
        } else {
            self.inner.with_current_stateful_operation(|| {
                // SAFETY: CUDA writes a device pointer for the requested byte
                // size while this context's lifecycle gate is held.
                self.inner.driver.check("cuMemAlloc_v2", unsafe {
                    (self.inner.driver.cu_mem_alloc)(&raw mut ptr, bytes.len())
                })?;
                crate::context::validate_device_allocation(ptr, bytes.len())
            })?;

            CudaDeviceBuffer {
                context: self.clone(),
                ptr,
                len: bytes.len(),
            }
        };

        if !bytes.is_empty() {
            self.inner.with_current_resource_operation(|| {
                // SAFETY: ptr is a valid device allocation of bytes.len(), the
                // host pointer covers that length, and the lifecycle gate is held.
                self.inner.driver.check("cuMemcpyHtoD_v2", unsafe {
                    (self.inner.driver.cu_memcpy_htod)(
                        ptr,
                        bytes.as_ptr().cast::<c_void>(),
                        bytes.len(),
                    )
                })
            })?;
        }

        Ok(buffer)
    }

    /// Upload host `f32` samples into a CUDA device buffer.
    pub fn upload_f32(&self, samples: &[f32]) -> Result<CudaDeviceBuffer, CudaError> {
        self.upload(f32_slice_as_bytes(samples))
    }

    /// Allocate an uninitialized CUDA device buffer.
    pub fn allocate(&self, len: usize) -> Result<CudaDeviceBuffer, CudaError> {
        let mut ptr = 0;
        if len != 0 {
            self.inner.with_current_stateful_operation(|| {
                // SAFETY: CUDA writes a device pointer for the requested byte
                // size while this context's lifecycle gate is held.
                self.inner.driver.check("cuMemAlloc_v2", unsafe {
                    (self.inner.driver.cu_mem_alloc)(&raw mut ptr, len)
                })?;
                crate::context::validate_device_allocation(ptr, len)
            })?;
        } else {
            self.inner.set_current()?;
        }
        Ok(CudaDeviceBuffer {
            context: self.clone(),
            ptr,
            len,
        })
    }

    /// Allocate page-locked host memory for host-to-device staging.
    #[cfg(test)]
    pub(crate) fn pinned_host_buffer(&self, len: usize) -> Result<CudaPinnedHostBuffer, CudaError> {
        let mut ptr = std::ptr::null_mut();
        if len != 0 {
            self.inner.with_current_stateful_operation(|| {
                // SAFETY: CUDA writes a page-locked host pointer for the requested
                // byte length. The allocation is freed by CudaPinnedHostBuffer.
                self.inner.driver.check("cuMemHostAlloc", unsafe {
                    (self.inner.driver.cu_mem_host_alloc)(&raw mut ptr, len, 0)
                })?;
                validate_non_null_pinned_host_allocation(ptr.cast::<u8>(), len).map(|_| ())
            })?;
        } else {
            self.inner.set_current()?;
        }
        Ok(CudaPinnedHostBuffer {
            context: self.clone(),
            ptr: ptr.cast::<u8>(),
            len,
        })
    }

    /// Create a reusable device-buffer pool for this context.
    pub fn buffer_pool(&self) -> CudaBufferPool {
        CudaBufferPool::new(self.clone())
    }

    /// Create a reusable best-fit device-buffer pool for workloads with many
    /// same-sized intermediate buffers.
    pub fn best_fit_buffer_pool(&self) -> CudaBufferPool {
        CudaBufferPool::new_size_buckets(self.clone())
    }
}

/// Page-locked host staging buffer.
#[cfg(test)]
#[derive(Debug)]
pub(crate) struct CudaPinnedHostBuffer {
    pub(crate) context: CudaContext,
    pub(crate) ptr: *mut u8,
    pub(crate) len: usize,
}

#[cfg(test)]
impl CudaPinnedHostBuffer {
    /// Immutable byte view of the pinned allocation.
    pub(crate) fn as_slice(&self) -> &[u8] {
        if self.len == 0 {
            &[]
        } else {
            // SAFETY: ptr is a live pinned allocation of len bytes.
            unsafe { std::slice::from_raw_parts(self.ptr.cast_const(), self.len) }
        }
    }

    /// Mutable byte view of the pinned allocation.
    pub(crate) fn as_mut_slice(&mut self) -> &mut [u8] {
        if self.len == 0 {
            &mut []
        } else {
            // SAFETY: ptr is uniquely borrowed through &mut self and covers len
            // bytes allocated by CUDA.
            unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
        }
    }
}

#[cfg(test)]
impl Drop for CudaPinnedHostBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            let free_result = self.context.inner.with_current_stateful_operation(|| {
                // SAFETY: ptr was returned by cuMemHostAlloc for this process,
                // and the context lifecycle gate is held during destruction.
                self.context.inner.driver.check("cuMemFreeHost", unsafe {
                    (self.context.inner.driver.cu_mem_free_host)(self.ptr.cast())
                })
            });
            if free_result.is_err() {
                std::mem::forget(self.context.clone());
            }
        }
    }
}

// SAFETY: The pinned allocation is owned by this value and CUDA frees it on
// drop. Mutable access still requires &mut self.
#[cfg(test)]
unsafe impl Send for CudaPinnedHostBuffer {}

/// Owned CUDA device buffer.
#[derive(Debug)]
pub struct CudaDeviceBuffer {
    pub(crate) context: CudaContext,
    pub(crate) ptr: CuDevicePtr,
    pub(crate) len: usize,
}

#[doc(hidden)]
/// Typed immutable device buffer view.
#[derive(Clone, Copy, Debug)]
pub struct CudaDeviceBufferView<'a, T> {
    pub(crate) ptr: CuDevicePtr,
    pub(crate) len: usize,
    pub(crate) _marker: std::marker::PhantomData<&'a T>,
}

impl<T> CudaDeviceBufferView<'_, T> {
    /// Raw CUDA device pointer value for kernel argument binding.
    pub fn device_ptr(&self) -> u64 {
        self.ptr
    }

    /// Number of typed elements in this view.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether this view has no elements.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

#[doc(hidden)]
/// Typed mutable device buffer view.
#[derive(Debug)]
pub struct CudaDeviceBufferViewMut<'a, T> {
    pub(crate) ptr: CuDevicePtr,
    pub(crate) len: usize,
    pub(crate) _marker: std::marker::PhantomData<&'a mut T>,
}

/// Lifetime-bound mutable view of CUDA memory owned by another runtime.
///
/// This value never frees the allocation. Its lifetime is tied to an exclusive
/// borrow of the external runtime's managed-resource guard.
#[doc(hidden)]
#[derive(Debug)]
pub struct CudaExternalDeviceBufferViewMut<'a> {
    context: CudaContext,
    ptr: CuDevicePtr,
    len: usize,
    _exclusive: std::marker::PhantomData<&'a mut ()>,
}

impl<'a> CudaExternalDeviceBufferViewMut<'a> {
    /// Construct a non-owning external device-buffer view.
    ///
    /// # Safety
    ///
    /// `ptr..ptr+len` must be a live CUDA allocation range represented by
    /// `_managed_owner`. The exclusive owner borrow must exclude every
    /// overlapping host or device mutation for this view's lifetime. The
    /// allocation must remain valid and must not be freed by the caller until
    /// the view is dropped.
    pub unsafe fn from_raw_parts<Owner>(
        context: &CudaContext,
        ptr: u64,
        len: usize,
        required_alignment: usize,
        _managed_owner: &'a mut Owner,
    ) -> Result<Self, CudaError> {
        if len == 0 {
            return Err(CudaError::InvalidArgument {
                message: "external CUDA buffer must not be empty".to_string(),
            });
        }
        if ptr == 0 {
            return Err(CudaError::InvalidArgument {
                message: "external CUDA buffer pointer must not be null".to_string(),
            });
        }
        if required_alignment == 0 || !required_alignment.is_power_of_two() {
            return Err(CudaError::InvalidArgument {
                message: "external CUDA buffer alignment must be a nonzero power of two"
                    .to_string(),
            });
        }
        if !ptr.is_multiple_of(required_alignment as u64) {
            return Err(CudaError::InvalidArgument {
                message: format!(
                    "external CUDA buffer pointer {ptr:#x} is not aligned to {required_alignment} bytes"
                ),
            });
        }
        let len_u64 = u64::try_from(len).map_err(|_| CudaError::LengthTooLarge { len })?;
        ptr.checked_add(len_u64)
            .ok_or(CudaError::LengthTooLarge { len })?;
        context.inner.validate_pointer_context(ptr)?;
        Ok(Self {
            context: context.clone(),
            ptr,
            len,
            _exclusive: std::marker::PhantomData,
        })
    }

    /// Context that owns the external allocation.
    pub fn context(&self) -> &CudaContext {
        &self.context
    }

    /// Raw device pointer.
    pub fn device_ptr(&self) -> u64 {
        self.ptr
    }

    /// External allocation range length in bytes.
    pub fn byte_len(&self) -> usize {
        self.len
    }
}

impl<T> CudaDeviceBufferViewMut<'_, T> {
    /// Raw CUDA device pointer value for kernel argument binding.
    pub fn device_ptr(&self) -> u64 {
        self.ptr
    }

    /// Number of typed elements in this view.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether this view has no elements.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

#[doc(hidden)]
/// One byte range inside a contiguous CUDA batch output allocation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaDeviceBufferRange {
    /// Byte offset from the start of the contiguous allocation.
    pub offset: usize,
    /// Byte length for this output item.
    pub len: usize,
}

impl CudaDeviceBuffer {
    pub(crate) fn is_owned_by(&self, context: &CudaContext) -> bool {
        self.context.is_same_context(context)
    }

    /// CUDA context that owns this allocation.
    pub fn context(&self) -> CudaContext {
        self.context.clone()
    }

    /// Raw CUDA device pointer value.
    pub fn device_ptr(&self) -> u64 {
        self.ptr
    }

    /// Device allocation length in bytes.
    pub fn byte_len(&self) -> usize {
        self.len
    }

    #[doc(hidden)]
    /// Borrow this allocation as a typed immutable device view.
    pub fn typed_view<T>(&self) -> Result<CudaDeviceBufferView<'_, T>, CudaError> {
        let element_size = std::mem::size_of::<T>();
        if element_size == 0 || !self.len.is_multiple_of(element_size) {
            return Err(CudaError::LengthNotElementAligned {
                bytes: self.len,
                element_size,
            });
        }
        Ok(CudaDeviceBufferView {
            ptr: self.ptr,
            len: self.len / element_size,
            _marker: std::marker::PhantomData,
        })
    }

    #[doc(hidden)]
    /// Borrow this allocation as a typed mutable device view.
    pub fn typed_view_mut<T>(&mut self) -> Result<CudaDeviceBufferViewMut<'_, T>, CudaError> {
        let element_size = std::mem::size_of::<T>();
        if element_size == 0 || !self.len.is_multiple_of(element_size) {
            return Err(CudaError::LengthNotElementAligned {
                bytes: self.len,
                element_size,
            });
        }
        Ok(CudaDeviceBufferViewMut {
            ptr: self.ptr,
            len: self.len / element_size,
            _marker: std::marker::PhantomData,
        })
    }

    /// Copy device bytes into caller-owned host output.
    pub fn copy_to_host(&self, out: &mut [u8]) -> Result<(), CudaError> {
        if out.len() < self.len {
            return Err(CudaError::OutputTooSmall {
                required: self.len,
                have: out.len(),
            });
        }
        if self.len == 0 {
            return Ok(());
        }

        self.context.inner.with_current_resource_operation(|| {
            // SAFETY: ptr is a live device allocation of self.len bytes, out
            // covers that range, and the context lifecycle gate is held.
            self.context.inner.driver.check("cuMemcpyDtoH_v2", unsafe {
                (self.context.inner.driver.cu_memcpy_dtoh)(
                    out.as_mut_ptr().cast::<c_void>(),
                    self.ptr,
                    self.len,
                )
            })
        })
    }

    /// Copy a byte range from this device buffer into caller-owned host output.
    pub fn copy_range_to_host(&self, offset: usize, out: &mut [u8]) -> Result<(), CudaError> {
        self.copy_byte_range_to_host_elements(offset, out)
    }

    /// Copy a byte range from this device buffer into uninitialized host output.
    pub fn copy_range_to_host_uninit(
        &self,
        offset: usize,
        out: &mut [std::mem::MaybeUninit<u8>],
    ) -> Result<(), CudaError> {
        self.copy_byte_range_to_host_elements(offset, out)
    }

    fn copy_byte_range_to_host_elements<T>(
        &self,
        offset: usize,
        out: &mut [T],
    ) -> Result<(), CudaError> {
        let byte_len = out
            .len()
            .checked_mul(std::mem::size_of::<T>())
            .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
        let end = offset
            .checked_add(byte_len)
            .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
        if end > self.len {
            return Err(CudaError::OutputTooSmall {
                required: end,
                have: self.len,
            });
        }
        if byte_len == 0 {
            return Ok(());
        }

        let source = self
            .ptr
            .checked_add(
                u64::try_from(offset).map_err(|_| CudaError::LengthTooLarge { len: offset })?,
            )
            .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
        self.context.inner.with_current_resource_operation(|| {
            // SAFETY: `source` is inside this live device allocation, `out`
            // covers exactly `byte_len` bytes, and the lifecycle gate is held.
            self.context.inner.driver.check("cuMemcpyDtoH_v2", unsafe {
                (self.context.inner.driver.cu_memcpy_dtoh)(
                    out.as_mut_ptr().cast::<c_void>(),
                    source,
                    byte_len,
                )
            })
        })
    }
}

impl Drop for CudaDeviceBuffer {
    fn drop(&mut self) {
        if self.ptr != 0 {
            let free_result = self.context.inner.with_current_stateful_operation(|| {
                // SAFETY: ptr was allocated by this CUDA context. The context
                // lifetime gate is held while the allocation is destroyed.
                let status = unsafe { (self.context.inner.driver.cu_mem_free)(self.ptr) };
                self.context.inner.driver.check("cuMemFree_v2", status)
            });
            if free_result.is_err() {
                // Retain the context so neither this allocation nor any
                // potentially in-flight work is torn down after completion
                // became uncertain.
                std::mem::forget(self.context.clone());
            }
        }
    }
}

pub(crate) fn checked_image_words(
    width: u32,
    height: u32,
    channels: usize,
) -> Result<usize, CudaError> {
    width
        .try_into()
        .ok()
        .and_then(|width: usize| width.checked_mul(height as usize))
        .and_then(|pixels| pixels.checked_mul(channels))
        .ok_or(CudaError::ImageTooLarge {
            width,
            height,
            channels,
        })
}
