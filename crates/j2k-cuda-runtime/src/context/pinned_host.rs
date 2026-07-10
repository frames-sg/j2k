// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{driver::Driver, error::CudaError};

pub(crate) struct PinnedUploadStaging {
    pub(crate) ptr: *mut u8,
    pub(crate) len: usize,
}

impl PinnedUploadStaging {
    pub(crate) fn as_slice(&self) -> &[u8] {
        if self.len == 0 {
            &[]
        } else {
            // SAFETY: ptr is a live pinned allocation of len bytes.
            unsafe { std::slice::from_raw_parts(self.ptr.cast_const(), self.len) }
        }
    }

    pub(crate) fn as_mut_slice(&mut self) -> &mut [u8] {
        if self.len == 0 {
            &mut []
        } else {
            // SAFETY: ptr is uniquely borrowed through &mut self and covers len
            // bytes allocated by CUDA.
            unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
        }
    }

    pub(crate) fn free(self, driver: &Driver) -> Result<(), CudaError> {
        if self.ptr.is_null() {
            return Ok(());
        }
        // SAFETY: ptr was returned by cuMemHostAlloc for this process.
        driver.check("cuMemFreeHost", unsafe {
            (driver.cu_mem_free_host)(self.ptr.cast())
        })
    }
}

// SAFETY: The pinned allocation is owned by this value. Mutable access requires
// &mut self, and freeing is explicitly coordinated by the owning CudaContext.
unsafe impl Send for PinnedUploadStaging {}
