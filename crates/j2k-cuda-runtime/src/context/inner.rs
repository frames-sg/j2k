// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{collections::HashMap, sync::Mutex};

use crate::{
    driver::{CuContext, Driver},
    error::CudaError,
};

use super::{CompiledKernel, CompiledKernelKey, PinnedUploadStaging};

pub(crate) struct ContextInner {
    pub(crate) driver: Driver,
    pub(crate) context: CuContext,
    pub(crate) modules: Mutex<HashMap<CompiledKernelKey, CompiledKernel>>,
    pub(crate) pinned_upload_staging: Mutex<Vec<PinnedUploadStaging>>,
}

impl ContextInner {
    pub(crate) fn set_current(&self) -> Result<(), CudaError> {
        // SAFETY: context is created by cuCtxCreate_v2 and remains valid while
        // ContextInner is alive.
        self.driver.check("cuCtxSetCurrent", unsafe {
            (self.driver.cu_ctx_set_current)(self.context)
        })
    }
}

impl Drop for ContextInner {
    fn drop(&mut self) {
        if !self.context.is_null() {
            let _ = self.set_current();
            let pinned_upload_staging = match self.pinned_upload_staging.get_mut() {
                Ok(pinned_upload_staging) => pinned_upload_staging,
                Err(poisoned) => poisoned.into_inner(),
            };
            for staging in pinned_upload_staging.drain(..) {
                let _ = staging.free(&self.driver);
            }
            let modules = match self.modules.get_mut() {
                Ok(modules) => modules,
                Err(poisoned) => poisoned.into_inner(),
            };
            for compiled in modules.drain().map(|(_, compiled)| compiled) {
                // SAFETY: modules were loaded into this CUDA context. Drop
                // cannot surface errors, so cleanup failures are ignored.
                let _ = unsafe { (self.driver.cu_module_unload)(compiled.module) };
            }
            // SAFETY: context was created by this ContextInner and cached
            // modules have already been unloaded.
            let _ = unsafe { (self.driver.cu_ctx_destroy)(self.context) };
        }
    }
}

// SAFETY: ContextInner owns an opaque CUDA context handle and synchronizes its
// Rust-side mutable caches with mutexes.
unsafe impl Send for ContextInner {}

// SAFETY: All shared Rust state is mutex-protected, and CUDA operations set the
// current context before touching context-owned resources.
unsafe impl Sync for ContextInner {}
