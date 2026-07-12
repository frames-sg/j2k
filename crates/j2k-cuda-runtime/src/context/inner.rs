// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{collections::HashMap, sync::Mutex};

use crate::{
    driver::{CuContext, Driver},
    memory::PinnedUploadStagingPool,
};

use super::{CompiledKernel, CompiledKernelKey, ContextResourceLifecycle};

pub(crate) struct ContextInner {
    pub(crate) driver: Driver,
    pub(crate) context: CuContext,
    pub(crate) modules: Mutex<HashMap<CompiledKernelKey, CompiledKernel>>,
    pub(crate) pinned_upload_operation: Mutex<()>,
    pub(crate) pinned_upload_staging: Mutex<PinnedUploadStagingPool>,
    pub(crate) resource_lifecycle: ContextResourceLifecycle,
}

impl Drop for ContextInner {
    fn drop(&mut self) {
        if !self.context.is_null() {
            let can_release_individually = self.resource_lifecycle.can_release_individually()
                && self.set_current_for_resource_release().is_ok();
            if can_release_individually {
                let pinned_upload_staging = match self.pinned_upload_staging.get_mut() {
                    Ok(pinned_upload_staging) => pinned_upload_staging,
                    Err(poisoned) => poisoned.into_inner(),
                };
                for mut staging in pinned_upload_staging.drain_cached() {
                    let _ = staging.free(&self.driver);
                }
                for mut staging in pinned_upload_staging.drain_uncertain() {
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
            }
            // SAFETY: context was created by this ContextInner and cached
            // resources are either already released or left for context-wide
            // destruction after a failed/uncertain completion state.
            let _ = unsafe { (self.driver.cu_ctx_destroy)(self.context) };
        }
    }
}

// SAFETY: ContextInner owns an opaque CUDA context handle and synchronizes its
// Rust-side mutable caches with mutexes.
unsafe impl Send for ContextInner {}

// SAFETY: All shared Rust state is mutex-protected. Destructive, launch, and
// completion-sensitive CUDA operations are serialized by the resource
// lifetime gate and bind this context while holding that gate.
unsafe impl Sync for ContextInner {}
