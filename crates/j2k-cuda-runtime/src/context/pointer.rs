// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::error::CudaError;

use super::ContextInner;

const CU_MEMORYTYPE_DEVICE: u32 = 2;

fn validate_stream_ordered_pointer(
    memory_type: u32,
    device_ordinal: i32,
    expected_ordinal: i32,
    mempool_handle: *mut std::ffi::c_void,
    device_ptr: crate::driver::CuDevicePtr,
) -> Result<crate::driver::CuDevicePtr, CudaError> {
    if memory_type != CU_MEMORYTYPE_DEVICE
        || device_ordinal != expected_ordinal
        || mempool_handle.is_null()
        || device_ptr == 0
    {
        return Err(CudaError::InvalidArgument {
            message: "external CUDA pointer is not an accessible stream-ordered allocation on this device"
                .to_string(),
        });
    }
    Ok(device_ptr)
}

impl ContextInner {
    pub(crate) fn resolve_pointer_for_context(
        &self,
        ptr: crate::driver::CuDevicePtr,
    ) -> Result<crate::driver::CuDevicePtr, CudaError> {
        const CU_POINTER_ATTRIBUTE_CONTEXT: i32 = 1;
        const CU_POINTER_ATTRIBUTE_MEMORY_TYPE: i32 = 2;
        const CU_POINTER_ATTRIBUTE_DEVICE_POINTER: i32 = 3;
        const CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL: i32 = 9;
        const CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE: i32 = 17;
        let mut pointer_context: crate::driver::CuContext = std::ptr::null_mut();
        self.with_current_resource_operation(|| {
            // SAFETY: CUDA writes one CUcontext value to pointer_context for
            // the live device pointer supplied by the caller.
            self.driver.check("cuPointerGetAttribute", unsafe {
                (self.driver.cu_pointer_get_attribute)(
                    (&raw mut pointer_context).cast(),
                    CU_POINTER_ATTRIBUTE_CONTEXT,
                    ptr,
                )
            })
        })?;
        if !pointer_context.is_null() {
            if pointer_context != self.context {
                return Err(CudaError::InvalidArgument {
                    message: "external CUDA pointer belongs to a different context".to_string(),
                });
            }
            return Ok(ptr);
        }

        let mut memory_type = 0u32;
        let mut device_ordinal = -1i32;
        let mut mempool_handle: *mut std::ffi::c_void = std::ptr::null_mut();
        let mut device_ptr = 0;
        self.with_current_resource_operation(|| {
            // SAFETY: CUDA writes each requested pointer attribute to a value
            // of the documented ABI type. A null allocation context is the
            // expected provenance for stream-ordered pool allocations.
            self.driver.check("cuPointerGetAttribute", unsafe {
                (self.driver.cu_pointer_get_attribute)(
                    (&raw mut memory_type).cast(),
                    CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
                    ptr,
                )
            })?;
            // SAFETY: CUDA writes one device ordinal to device_ordinal.
            self.driver.check("cuPointerGetAttribute", unsafe {
                (self.driver.cu_pointer_get_attribute)(
                    (&raw mut device_ordinal).cast(),
                    CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
                    ptr,
                )
            })?;
            // SAFETY: CUDA writes one CUmemoryPool handle to mempool_handle.
            self.driver.check("cuPointerGetAttribute", unsafe {
                (self.driver.cu_pointer_get_attribute)(
                    (&raw mut mempool_handle).cast(),
                    CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE,
                    ptr,
                )
            })?;
            // SAFETY: CUDA writes the device pointer through which kernels in
            // the current context may access this allocation, or rejects it.
            self.driver.check("cuPointerGetAttribute", unsafe {
                (self.driver.cu_pointer_get_attribute)(
                    (&raw mut device_ptr).cast(),
                    CU_POINTER_ATTRIBUTE_DEVICE_POINTER,
                    ptr,
                )
            })
        })?;

        let expected_ordinal =
            i32::try_from(self.device_ordinal).map_err(|_| CudaError::InvalidArgument {
                message: "CUDA context device ordinal exceeds i32".to_string(),
            })?;
        validate_stream_ordered_pointer(
            memory_type,
            device_ordinal,
            expected_ordinal,
            mempool_handle,
            device_ptr,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{validate_stream_ordered_pointer, CU_MEMORYTYPE_DEVICE};

    const POOL: *mut std::ffi::c_void = std::ptr::dangling_mut::<std::ffi::c_void>();

    #[test]
    fn stream_ordered_pointer_requires_complete_matching_provenance() {
        assert_eq!(
            validate_stream_ordered_pointer(CU_MEMORYTYPE_DEVICE, 2, 2, POOL, 0x1000)
                .expect("matching stream-ordered allocation"),
            0x1000
        );

        for result in [
            validate_stream_ordered_pointer(1, 2, 2, POOL, 0x1000),
            validate_stream_ordered_pointer(CU_MEMORYTYPE_DEVICE, 1, 2, POOL, 0x1000),
            validate_stream_ordered_pointer(
                CU_MEMORYTYPE_DEVICE,
                2,
                2,
                std::ptr::null_mut(),
                0x1000,
            ),
            validate_stream_ordered_pointer(CU_MEMORYTYPE_DEVICE, 2, 2, POOL, 0),
        ] {
            assert!(result.is_err());
        }
    }
}
