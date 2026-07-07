use crate::{
    build_flags::PINNED_UPLOAD_STAGING_POOL_MAX,
    bytes::{f32_slice_as_bytes, i16_slice_as_bytes, i32_slice_as_bytes},
    context::{CudaContext, PinnedUploadStaging},
    driver::CuDevicePtr,
    error::CudaError,
};
use std::{
    collections::BTreeMap,
    ffi::c_void,
    sync::{Arc, Mutex},
};

impl CudaContext {
    /// Upload host bytes into a CUDA device buffer.
    pub fn upload(&self, bytes: &[u8]) -> Result<CudaDeviceBuffer, CudaError> {
        self.inner.set_current()?;

        let mut ptr = 0;
        let buffer = if bytes.is_empty() {
            CudaDeviceBuffer {
                context: self.clone(),
                ptr,
                len: bytes.len(),
            }
        } else {
            // SAFETY: CUDA writes a device pointer for the requested byte size.
            self.inner.driver.check("cuMemAlloc_v2", unsafe {
                (self.inner.driver.cu_mem_alloc)(&raw mut ptr, bytes.len())
            })?;

            CudaDeviceBuffer {
                context: self.clone(),
                ptr,
                len: bytes.len(),
            }
        };

        if !bytes.is_empty() {
            // SAFETY: ptr is a valid device allocation of bytes.len(), and the
            // host pointer is valid for bytes.len().
            self.inner.driver.check("cuMemcpyHtoD_v2", unsafe {
                (self.inner.driver.cu_memcpy_htod)(
                    ptr,
                    bytes.as_ptr().cast::<c_void>(),
                    bytes.len(),
                )
            })?;
        }

        Ok(buffer)
    }

    /// Upload host bytes through a temporary page-locked staging buffer.
    pub fn upload_pinned(&self, bytes: &[u8]) -> Result<CudaDeviceBuffer, CudaError> {
        if bytes.is_empty() {
            return self.upload(bytes);
        }
        let mut staging = self.take_pinned_upload_staging(bytes.len())?;
        staging.as_mut_slice()[..bytes.len()].copy_from_slice(bytes);
        let upload_result = self.upload(&staging.as_slice()[..bytes.len()]);
        let recycle_result = self.recycle_pinned_upload_staging(staging);
        match (upload_result, recycle_result) {
            (Ok(buffer), Ok(())) => Ok(buffer),
            (Err(error), _) | (_, Err(error)) => Err(error),
        }
    }

    pub(crate) fn take_pinned_upload_staging(
        &self,
        len: usize,
    ) -> Result<PinnedUploadStaging, CudaError> {
        self.inner.set_current()?;
        let mut staging =
            self.inner
                .pinned_upload_staging
                .lock()
                .map_err(|error| CudaError::StatePoisoned {
                    message: error.to_string(),
                })?;
        if let Some(index) = staging.iter().position(|buffer| buffer.len >= len) {
            return Ok(staging.swap_remove(index));
        }
        drop(staging);

        let mut ptr = std::ptr::null_mut();
        // SAFETY: CUDA writes a page-locked host pointer for the requested byte
        // length. The allocation is freed by the context's staging pool cleanup.
        self.inner.driver.check("cuMemHostAlloc", unsafe {
            (self.inner.driver.cu_mem_host_alloc)(&raw mut ptr, len, 0)
        })?;
        Ok(PinnedUploadStaging {
            ptr: ptr.cast::<u8>(),
            len,
        })
    }

    pub(crate) fn recycle_pinned_upload_staging(
        &self,
        staging: PinnedUploadStaging,
    ) -> Result<(), CudaError> {
        let mut pool =
            self.inner
                .pinned_upload_staging
                .lock()
                .map_err(|error| CudaError::StatePoisoned {
                    message: error.to_string(),
                })?;
        if pool.len() < PINNED_UPLOAD_STAGING_POOL_MAX {
            pool.push(staging);
            return Ok(());
        }
        drop(pool);
        self.inner.set_current()?;
        staging.free(&self.inner.driver)
    }

    /// Upload host `f32` samples into a CUDA device buffer.
    pub fn upload_f32(&self, samples: &[f32]) -> Result<CudaDeviceBuffer, CudaError> {
        self.upload(f32_slice_as_bytes(samples))
    }

    /// Upload host `f32` samples through a temporary page-locked staging buffer.
    pub fn upload_f32_pinned(&self, samples: &[f32]) -> Result<CudaDeviceBuffer, CudaError> {
        self.upload_pinned(f32_slice_as_bytes(samples))
    }

    /// Upload host `i32` samples through a temporary page-locked staging buffer.
    pub(crate) fn upload_i32_pinned(&self, samples: &[i32]) -> Result<CudaDeviceBuffer, CudaError> {
        self.upload_pinned(i32_slice_as_bytes(samples))
    }

    /// Allocate an uninitialized CUDA device buffer.
    pub fn allocate(&self, len: usize) -> Result<CudaDeviceBuffer, CudaError> {
        self.inner.set_current()?;
        let mut ptr = 0;
        if len != 0 {
            // SAFETY: CUDA writes a device pointer for the requested byte size.
            self.inner.driver.check("cuMemAlloc_v2", unsafe {
                (self.inner.driver.cu_mem_alloc)(&raw mut ptr, len)
            })?;
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
        self.inner.set_current()?;
        let mut ptr = std::ptr::null_mut();
        if len != 0 {
            // SAFETY: CUDA writes a page-locked host pointer for the requested
            // byte length. The allocation is freed by CudaPinnedHostBuffer.
            self.inner.driver.check("cuMemHostAlloc", unsafe {
                (self.inner.driver.cu_mem_host_alloc)(&raw mut ptr, len, 0)
            })?;
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
            let _ = self.context.inner.set_current();
            // SAFETY: ptr was returned by cuMemHostAlloc for this process.
            let _ = unsafe { (self.context.inner.driver.cu_mem_free_host)(self.ptr.cast()) };
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

/// Reusable CUDA device-buffer pool for repeated adapter dispatches.
#[derive(Clone, Debug)]
pub struct CudaBufferPool {
    pub(crate) inner: Arc<CudaBufferPoolInner>,
}

#[derive(Debug)]
pub(crate) struct CudaBufferPoolInner {
    pub(crate) context: CudaContext,
    pub(crate) free: Mutex<CudaBufferPoolFree>,
}

#[derive(Debug)]
pub(crate) enum CudaBufferPoolFree {
    FirstFit(Vec<CudaDeviceBuffer>),
    SizeBuckets(BTreeMap<usize, Vec<CudaDeviceBuffer>>),
}

impl CudaBufferPoolInner {
    fn recycle_buffer(&self, buffer: CudaDeviceBuffer) -> Result<(), CudaError> {
        let mut free = self.free.lock().map_err(|error| CudaError::StatePoisoned {
            message: error.to_string(),
        })?;
        match &mut *free {
            CudaBufferPoolFree::FirstFit(free) => free.push(buffer),
            CudaBufferPoolFree::SizeBuckets(free) => {
                free.entry(buffer.byte_len()).or_default().push(buffer);
            }
        }
        Ok(())
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
    /// Create a new pool for `context`.
    pub fn new(context: CudaContext) -> Self {
        Self {
            inner: Arc::new(CudaBufferPoolInner {
                context,
                free: Mutex::new(CudaBufferPoolFree::FirstFit(Vec::new())),
            }),
        }
    }

    fn new_size_buckets(context: CudaContext) -> Self {
        Self {
            inner: Arc::new(CudaBufferPoolInner {
                context,
                free: Mutex::new(CudaBufferPoolFree::SizeBuckets(BTreeMap::new())),
            }),
        }
    }

    /// Acquire a device buffer with at least `len` bytes.
    pub fn take(&self, len: usize) -> Result<CudaPooledDeviceBuffer, CudaError> {
        let mut free = self
            .inner
            .free
            .lock()
            .map_err(|error| CudaError::StatePoisoned {
                message: error.to_string(),
            })?;
        let (reusable_buffer, _) = pool_take_fit_buffer(&mut free, len);
        let buffer = if let Some(buffer) = reusable_buffer {
            buffer
        } else {
            drop(free);
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
        let mut free = self
            .inner
            .free
            .lock()
            .map_err(|error| CudaError::StatePoisoned {
                message: error.to_string(),
            })?;
        let free_count_before = free.cached_count();
        let (reusable_buffer, scanned_count) = pool_take_fit_buffer(&mut free, len);
        let reused = reusable_buffer.is_some();
        let buffer = if let Some(buffer) = reusable_buffer {
            buffer
        } else {
            drop(free);
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
            self.inner.context.inner.set_current()?;
            // SAFETY: `buffer` is a live device allocation with at least
            // `bytes.len()` bytes for this checkout, and `bytes` is valid for
            // that many host bytes.
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
                .check("cuMemcpyHtoD_v2", result)?;
        }
        Ok(buffer)
    }

    /// Upload host bytes through temporary page-locked staging into a pooled device buffer.
    pub fn upload_pinned(&self, bytes: &[u8]) -> Result<CudaPooledDeviceBuffer, CudaError> {
        if bytes.is_empty() {
            return self.upload(bytes);
        }

        let buffer = self.take(bytes.len())?;
        let mut staging = self.inner.context.take_pinned_upload_staging(bytes.len())?;
        staging.as_mut_slice()[..bytes.len()].copy_from_slice(bytes);
        self.inner.context.inner.set_current()?;
        // SAFETY: `buffer` is a live device allocation with at least
        // `bytes.len()` bytes, and the pinned staging slice covers that range.
        let upload_result = unsafe {
            (self.inner.context.inner.driver.cu_memcpy_htod)(
                buffer.device_ptr(),
                staging.as_slice()[..bytes.len()].as_ptr().cast::<c_void>(),
                bytes.len(),
            )
        };
        let upload_result = self
            .inner
            .context
            .inner
            .driver
            .check("cuMemcpyHtoD_v2", upload_result);
        let recycle_result = self.inner.context.recycle_pinned_upload_staging(staging);
        match (upload_result, recycle_result) {
            (Ok(()), Ok(())) => Ok(buffer),
            (Err(error), _) | (_, Err(error)) => Err(error),
        }
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
        Ok(self
            .inner
            .free
            .lock()
            .map_err(|error| CudaError::StatePoisoned {
                message: error.to_string(),
            })?
            .cached_count())
    }
}

impl CudaBufferPoolFree {
    fn cached_count(&self) -> usize {
        match self {
            Self::FirstFit(free) => free.len(),
            Self::SizeBuckets(free) => free.values().map(Vec::len).sum(),
        }
    }
}

pub(crate) fn pool_take_fit_buffer(
    free: &mut CudaBufferPoolFree,
    len: usize,
) -> (Option<CudaDeviceBuffer>, usize) {
    match free {
        CudaBufferPoolFree::FirstFit(free) => pool_take_first_fit_buffer(free, len),
        CudaBufferPoolFree::SizeBuckets(free) => pool_take_size_bucket_buffer(free, len),
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
            return (Some(free.swap_remove(index)), examined);
        }
    }
    (None, examined)
}

pub(crate) fn pool_take_size_bucket_buffer(
    free: &mut BTreeMap<usize, Vec<CudaDeviceBuffer>>,
    len: usize,
) -> (Option<CudaDeviceBuffer>, usize) {
    let Some(size) = free.range(len..).next().map(|(size, _)| *size) else {
        return (None, usize::from(!free.is_empty()));
    };
    let buffer = free
        .get_mut(&size)
        .expect("selected CUDA buffer pool size bucket must exist")
        .pop();
    if free.get(&size).is_some_and(Vec::is_empty) {
        free.remove(&size);
    }
    (buffer, 1)
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
        buffer.context.inner.set_current()?;
        // SAFETY: `buffer.ptr` is a live allocation with at least
        // `requested_len` bytes for this checkout, and `out` was validated.
        let result = unsafe {
            (buffer.context.inner.driver.cu_memcpy_dtoh)(
                out.as_mut_ptr().cast::<c_void>(),
                buffer.ptr,
                self.requested_len,
            )
        };
        buffer
            .context
            .inner
            .driver
            .check("cuMemcpyDtoH_v2", result)?;
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

#[doc(hidden)]
/// One byte range inside a contiguous CUDA batch output allocation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaDeviceBufferRange {
    /// Byte offset from the start of the contiguous allocation.
    pub offset: usize,
    /// Byte length for this output item.
    pub len: usize,
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

pub(crate) fn copy_pooled_bytes_to_vec_uninit(
    buffer: &CudaPooledDeviceBuffer,
    byte_len: usize,
) -> Result<Vec<u8>, CudaError> {
    let mut out = Vec::with_capacity(byte_len);
    pooled_device_buffer(buffer)?.copy_range_to_host_uninit(0, out.spare_capacity_mut())?;
    // SAFETY: copy_range_to_host_uninit returned success after writing exactly
    // byte_len initialized bytes into the Vec spare capacity.
    unsafe {
        out.set_len(byte_len);
    }
    Ok(out)
}

impl CudaDeviceBuffer {
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

        self.context.inner.set_current()?;
        // SAFETY: ptr is a live device allocation of self.len bytes, and out is
        // valid for at least self.len bytes.
        self.context.inner.driver.check("cuMemcpyDtoH_v2", unsafe {
            (self.context.inner.driver.cu_memcpy_dtoh)(
                out.as_mut_ptr().cast::<c_void>(),
                self.ptr,
                self.len,
            )
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

        self.context.inner.set_current()?;
        let source = self
            .ptr
            .checked_add(
                u64::try_from(offset).map_err(|_| CudaError::LengthTooLarge { len: offset })?,
            )
            .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
        // SAFETY: `source` is inside this live device allocation, and `out` is
        // a writable host slice covering exactly `byte_len` bytes.
        self.context.inner.driver.check("cuMemcpyDtoH_v2", unsafe {
            (self.context.inner.driver.cu_memcpy_dtoh)(
                out.as_mut_ptr().cast::<c_void>(),
                source,
                byte_len,
            )
        })
    }
}

impl Drop for CudaDeviceBuffer {
    fn drop(&mut self) {
        if self.ptr != 0 {
            let _ = self.context.inner.set_current();
            // SAFETY: ptr was allocated by this CUDA context. Drop cannot
            // surface errors, so failures are ignored during cleanup.
            let _ = unsafe { (self.context.inner.driver.cu_mem_free)(self.ptr) };
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
