// SPDX-License-Identifier: MIT OR Apache-2.0

use super::inner::ContextInner;
use super::validate_resource_handle;
use crate::allocation::host_allocation_error;
use crate::driver::{CuFunction, CuModule};
use crate::error::{select_resource_release_error, CudaError};
#[cfg(j2k_cuda_oxide_enabled)]
use crate::kernels;
#[cfg(j2k_cuda_oxide_enabled)]
use crate::kernels::CudaKernel;
use crate::CudaKernelSpec;
use std::ffi::{c_char, c_void};

#[derive(Debug)]
pub(crate) struct CompiledKernel {
    pub(crate) module: CuModule,
    pub(crate) function: CuFunction,
}
fn resolve_loaded_kernel_function(
    module: CuModule,
    lookup: impl FnOnce(CuModule) -> Result<CuFunction, CudaError>,
    unload: impl FnOnce(CuModule) -> Result<(), CudaError>,
) -> Result<CompiledKernel, CudaError> {
    match lookup(module).and_then(|function| {
        validate_resource_handle(
            function,
            "CUDA returned a null function after successful lookup",
        )?;
        Ok(function)
    }) {
        Ok(function) => Ok(CompiledKernel { module, function }),
        Err(error) => match unload(module) {
            Ok(()) => Err(error),
            Err(unload_error) => Err(select_resource_release_error(error, unload_error)),
        },
    }
}
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum CompiledKernelKey {
    Runtime(CudaKernelSpec),
    #[cfg(feature = "cuda-oxide-copy-u8")]
    CudaOxideCopyU8,
}
impl ContextInner {
    pub(in crate::context) fn kernel_function_from_key(
        &self,
        key: CompiledKernelKey,
    ) -> Result<CuFunction, CudaError> {
        match key {
            CompiledKernelKey::Runtime(_) => {}
            #[cfg(feature = "cuda-oxide-copy-u8")]
            CompiledKernelKey::CudaOxideCopyU8 => {}
        }
        self.set_current()?;
        let mut modules = self
            .modules
            .lock()
            .map_err(|error| CudaError::StatePoisoned {
                message: error.to_string(),
            })?;
        if let Some(compiled) = modules.get(&key) {
            return Ok(compiled.function);
        }

        modules.try_reserve(1).map_err(|_| {
            host_allocation_error::<(CompiledKernelKey, CompiledKernel)>(
                modules.len().saturating_add(1),
            )
        })?;
        let compiled = CompiledKernel::load(self, key)?;
        let function = compiled.function;
        modules.insert(key, compiled);
        Ok(function)
    }

    pub(crate) fn kernel_function_from_spec(
        &self,
        spec: CudaKernelSpec,
    ) -> Result<CuFunction, CudaError> {
        self.kernel_function_from_key(CompiledKernelKey::Runtime(spec))
    }
}

impl CompiledKernelKey {
    pub(crate) fn ptx(self) -> &'static [u8] {
        match self {
            Self::Runtime(spec) => spec.ptx(),
            #[cfg(feature = "cuda-oxide-copy-u8")]
            Self::CudaOxideCopyU8 => kernels::cuda_oxide_copy_u8_ptx(),
        }
    }

    pub(crate) fn entrypoint(self) -> &'static [u8] {
        match self {
            Self::Runtime(spec) => spec.entrypoint(),
            #[cfg(feature = "cuda-oxide-copy-u8")]
            Self::CudaOxideCopyU8 => CudaKernel::CopyU8.entrypoint(),
        }
    }
}

impl CompiledKernel {
    pub(crate) fn load(context: &ContextInner, key: CompiledKernelKey) -> Result<Self, CudaError> {
        let module = context.with_current_stateful_operation(|| {
            let mut module = std::ptr::null_mut();
            // SAFETY: image is a NUL-terminated PTX string. CUDA copies or
            // parses it while this context's lifecycle gate is held.
            context.driver.check("cuModuleLoadData", unsafe {
                (context.driver.cu_module_load_data)(
                    &raw mut module,
                    key.ptx().as_ptr().cast::<c_void>(),
                )
            })?;
            validate_resource_handle(module, "CUDA returned a null module after successful load")?;
            Ok(module)
        })?;
        resolve_loaded_kernel_function(
            module,
            |module| {
                context.with_current_resource_operation(|| {
                    let mut function = std::ptr::null_mut();
                    // SAFETY: name is a NUL-terminated kernel symbol in this
                    // live module, and the context lifecycle gate is held.
                    context.driver.check("cuModuleGetFunction", unsafe {
                        (context.driver.cu_module_get_function)(
                            &raw mut function,
                            module,
                            key.entrypoint().as_ptr().cast::<c_char>(),
                        )
                    })?;
                    Ok(function)
                })
            },
            |module| {
                context.with_current_stateful_operation(|| {
                    // SAFETY: module was loaded successfully above, no function
                    // from it was launched, and the lifecycle gate is held.
                    // Stateful unload failure quarantines the context.
                    context.driver.check("cuModuleUnload", unsafe {
                        (context.driver.cu_module_unload)(module)
                    })
                })
            },
        )
    }
}

#[cfg(test)]
mod tests;

// SAFETY: CompiledKernel stores opaque CUDA module/function handles. Lifetime
// and unloading are coordinated by ContextInner's module cache mutex.
unsafe impl Send for CompiledKernel {}
