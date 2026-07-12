// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{driver::CuDevicePtr, error::CudaError};

pub(crate) fn validate_device_allocation(ptr: CuDevicePtr, len: usize) -> Result<(), CudaError> {
    if len != 0 && ptr == 0 {
        return Err(CudaError::InternalInvariant {
            what: "CUDA returned null for a nonzero device allocation",
        });
    }
    Ok(())
}

pub(crate) fn validate_resource_handle<T>(
    ptr: *mut T,
    what: &'static str,
) -> Result<(), CudaError> {
    if ptr.is_null() {
        return Err(CudaError::InternalInvariant { what });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{validate_device_allocation, validate_resource_handle};
    use crate::error::CudaError;
    use std::ffi::c_void;

    #[test]
    fn nonzero_null_device_allocation_is_rejected() {
        assert!(matches!(
            validate_device_allocation(0, 1),
            Err(CudaError::InternalInvariant { .. })
        ));
        validate_device_allocation(0, 0).expect("zero-byte allocation may use the null sentinel");
        validate_device_allocation(1, 1).expect("nonzero device allocation");
    }

    #[test]
    fn null_resource_handle_is_rejected() {
        assert!(matches!(
            validate_resource_handle::<c_void>(
                std::ptr::null_mut(),
                "CUDA returned a null test handle",
            ),
            Err(CudaError::InternalInvariant { .. })
        ));
        validate_resource_handle(
            std::ptr::without_provenance_mut::<c_void>(1),
            "CUDA returned a null test handle",
        )
        .expect("non-null resource handle");
    }
}
