// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::error::CudaError;

use super::ContextInner;

mod stream_ordered;

impl ContextInner {
    pub(crate) fn resolve_pointer_for_context(
        &self,
        ptr: crate::driver::CuDevicePtr,
    ) -> Result<crate::driver::CuDevicePtr, CudaError> {
        const CU_POINTER_ATTRIBUTE_CONTEXT: i32 = 1;
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
        stream_ordered::resolve(self, ptr)
    }
}
