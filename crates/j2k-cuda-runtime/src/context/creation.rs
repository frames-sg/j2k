// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use crate::{driver::Driver, error::CudaError, memory::PinnedUploadStagingPool};

use super::{
    diagnostics::{CudaContextDiagnosticsState, CudaEventPoolState},
    host_budget::SharedCudaHostBudget,
    inner::{ContextInner, ContextOwnership},
    validate_resource_handle, ContextResourceLifecycle, CudaContext,
};

pub(super) fn create_context(
    device_ordinal: usize,
    retain_primary: bool,
) -> Result<CudaContext, CudaError> {
    let driver = Driver::load()?;
    // SAFETY: cuInit is the CUDA Driver API process initializer.
    driver.check("cuInit", unsafe { (driver.cu_init)(0) })?;

    let mut count = 0;
    // SAFETY: CUDA writes one integer device count to the provided pointer.
    driver.check("cuDeviceGetCount", unsafe {
        (driver.cu_device_get_count)(&raw mut count)
    })?;
    if count <= 0 {
        return Err(CudaError::Unavailable {
            message: "no CUDA devices reported by driver".to_string(),
        });
    }

    let ordinal = i32::try_from(device_ordinal).map_err(|_| CudaError::InvalidArgument {
        message: "CUDA device ordinal exceeds i32".to_string(),
    })?;
    if ordinal >= count {
        return Err(CudaError::InvalidArgument {
            message: format!(
                "CUDA device ordinal {device_ordinal} is out of range for {count} devices"
            ),
        });
    }
    let mut device = 0;
    // SAFETY: the ordinal was checked against cuDeviceGetCount above.
    driver.check("cuDeviceGet", unsafe {
        (driver.cu_device_get)(&raw mut device, ordinal)
    })?;

    let mut context = std::ptr::null_mut();
    let ownership = if retain_primary {
        // SAFETY: CUDA writes the retained primary context for a valid device.
        driver.check("cuDevicePrimaryCtxRetain", unsafe {
            (driver.cu_device_primary_ctx_retain)(&raw mut context, device)
        })?;
        ContextOwnership::RetainedPrimary { device }
    } else {
        // SAFETY: CUDA writes a newly-created context handle for a valid device.
        driver.check("cuCtxCreate_v2", unsafe {
            (driver.cu_ctx_create)(&raw mut context, 0, device)
        })?;
        ContextOwnership::Owned
    };
    if let Err(error) = validate_resource_handle(
        context,
        "CUDA returned a null context after successful creation",
    ) {
        match ownership {
            ContextOwnership::Owned => {
                // SAFETY: balance successful creation before ownership can escape.
                let _ = unsafe { (driver.cu_ctx_destroy)(context) };
            }
            ContextOwnership::RetainedPrimary { device } => {
                // SAFETY: this balances the successful primary-context retain
                // when its returned handle fails validation.
                let _ = unsafe { (driver.cu_device_primary_ctx_release)(device) };
            }
        }
        return Err(error);
    }

    Ok(CudaContext {
        inner: Arc::new(ContextInner {
            driver,
            context,
            ownership,
            device_ordinal,
            modules: Mutex::new(HashMap::new()),
            event_pool: Mutex::new(CudaEventPoolState::default()),
            diagnostics: CudaContextDiagnosticsState::default(),
            pinned_upload_operation: Mutex::new(()),
            pinned_upload_staging: Mutex::new(PinnedUploadStagingPool::new()),
            host_budget: SharedCudaHostBudget::new(),
            resource_lifecycle: ContextResourceLifecycle::new(),
        }),
    })
}
