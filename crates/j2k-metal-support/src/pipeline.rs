// SPDX-License-Identifier: MIT OR Apache-2.0

use metal::{
    foreign_types::ForeignType,
    objc::{
        rc::StrongPtr,
        runtime::{Class, Object, Sel},
        Message,
    },
    CompileOptions, ComputePipelineState, Device, DeviceRef, Library, MTLCompileOptions,
    MTLComputePipelineState, MTLLibrary,
};

use crate::MetalSupportError;

unsafe fn objective_c_error_message(error: *mut Object) -> String {
    if error.is_null() {
        return "Objective-C returned no error diagnostic".to_string();
    }
    // SAFETY: `error` is non-null and was returned through an NSError** out
    // parameter by the immediately preceding synchronous Metal call.
    let description: *mut Object =
        match unsafe { (&*error).send_message(Sel::register("localizedDescription"), ()) } {
            Ok(description) => description,
            Err(dispatch_error) => {
                return format!("Objective-C error diagnostic failed: {dispatch_error}");
            }
        };
    if description.is_null() {
        return "Objective-C returned an error without a description".to_string();
    }
    // SAFETY: `description` is a non-null NSString returned by NSError.
    let utf8: *const core::ffi::c_char =
        match unsafe { (&*description).send_message(Sel::register("UTF8String"), ()) } {
            Ok(utf8) => utf8,
            Err(dispatch_error) => {
                return format!("Objective-C error UTF-8 conversion failed: {dispatch_error}");
            }
        };
    if utf8.is_null() {
        "Objective-C returned a non-UTF-8 error description".to_string()
    } else {
        // SAFETY: UTF8String returns a NUL-terminated pointer valid while the
        // NSString remains alive in the current autorelease context.
        unsafe { std::ffi::CStr::from_ptr(utf8) }
            .to_string_lossy()
            .into_owned()
    }
}

pub(crate) unsafe fn checked_compile_options_from_retained_ptr(
    raw: *mut MTLCompileOptions,
) -> Result<CompileOptions, MetalSupportError> {
    if raw.is_null() {
        Err(MetalSupportError::ShaderLibrary {
            message: "Metal compile-options allocation returned nil".to_string(),
        })
    } else {
        // SAFETY: The caller guarantees that a non-null pointer is a retained
        // MTLCompileOptions result whose ownership transfers here.
        Ok(unsafe { CompileOptions::from_ptr(raw) })
    }
}

fn checked_compile_options() -> Result<CompileOptions, MetalSupportError> {
    let class =
        Class::get("MTLCompileOptions").ok_or_else(|| MetalSupportError::ShaderLibrary {
            message: "MTLCompileOptions class is unavailable".to_string(),
        })?;
    // SAFETY: `new` takes no arguments and returns a retained
    // MTLCompileOptions pointer, which is checked before wrapping.
    let raw: *mut MTLCompileOptions = unsafe {
        class
            .send_message(Sel::register("new"), ())
            .map_err(|error| MetalSupportError::ShaderLibrary {
                message: format!("Metal compile-options allocation failed: {error}"),
            })?
    };
    // SAFETY: `new` returns either nil or a retained MTLCompileOptions pointer.
    unsafe { checked_compile_options_from_retained_ptr(raw) }
}

fn checked_nsstring(source: &str) -> Result<StrongPtr, MetalSupportError> {
    const UTF8_ENCODING: usize = 4;

    let class = Class::get("NSString").ok_or_else(|| MetalSupportError::ShaderLibrary {
        message: "NSString class is unavailable".to_string(),
    })?;
    // SAFETY: `alloc` takes no arguments and returns a retained object pointer.
    let allocated: *mut Object = unsafe {
        class
            .send_message(Sel::register("alloc"), ())
            .map_err(|error| MetalSupportError::ShaderLibrary {
                message: format!("Metal shader source allocation failed: {error}"),
            })?
    };
    if allocated.is_null() {
        return Err(MetalSupportError::ShaderLibrary {
            message: "Metal shader source allocation returned nil".to_string(),
        });
    }
    // SAFETY: `allocated` is a retained NSString allocation. The source bytes
    // remain live for the synchronous initializer call.
    let initialized: *mut Object = match unsafe {
        (&*allocated).send_message(
            Sel::register("initWithBytes:length:encoding:"),
            (
                source.as_ptr().cast::<core::ffi::c_void>(),
                source.len(),
                UTF8_ENCODING,
            ),
        )
    } {
        Ok(initialized) => initialized,
        Err(error) => {
            // SAFETY: `alloc` returned this object at +1, and the initializer
            // dispatch did not complete to consume or replace it.
            unsafe { metal::objc::runtime::objc_release(allocated) };
            return Err(MetalSupportError::ShaderLibrary {
                message: format!("Metal shader source initialization failed: {error}"),
            });
        }
    };
    if initialized.is_null() {
        Err(MetalSupportError::ShaderLibrary {
            message: "Metal shader source initialization returned nil".to_string(),
        })
    } else {
        // SAFETY: A successful `init` returns an initialized object at +1.
        Ok(unsafe { StrongPtr::new(initialized) })
    }
}

/// Compile a Metal shader source string with default compile options.
///
/// # Errors
///
/// Returns [`MetalSupportError::ShaderLibrary`] when Metal rejects the source.
pub fn shader_library(device: &DeviceRef, source: &str) -> Result<Library, MetalSupportError> {
    let options = checked_compile_options()?;
    let source = checked_nsstring(source)?;
    let mut error: *mut Object = core::ptr::null_mut();
    // SAFETY: The selector ABI matches MTLDevice. Source/options remain live
    // for the synchronous compile, and both returned pointers are checked.
    let raw: *mut MTLLibrary = unsafe {
        device
            .send_message(
                Sel::register("newLibraryWithSource:options:error:"),
                (*source, options.as_ptr(), core::ptr::from_mut(&mut error)),
            )
            .map_err(|dispatch_error| MetalSupportError::ShaderLibrary {
                message: format!("Metal shader compilation dispatch failed: {dispatch_error}"),
            })?
    };
    if raw.is_null() {
        let message = if error.is_null() {
            "Metal shader compilation returned nil without an error".to_string()
        } else {
            // SAFETY: The non-null NSError came from the synchronous compile.
            unsafe { objective_c_error_message(error) }
        };
        return Err(MetalSupportError::ShaderLibrary { message });
    }
    if !error.is_null() {
        // SAFETY: The non-null NSError came from the synchronous compile and a
        // non-null library indicates compiler warnings rather than failure.
        log::warn!("Metal shader compiler warning: {}", unsafe {
            objective_c_error_message(error)
        });
    }
    // SAFETY: The raw selector returned a non-null retained MTLLibrary.
    Ok(unsafe { Library::from_ptr(raw) })
}

/// Load a named compute pipeline from an already compiled shader library.
///
/// # Errors
///
/// Returns [`MetalSupportError::PipelineFunction`] when the function is absent,
/// or [`MetalSupportError::PipelineState`] when pipeline construction fails.
pub fn named_pipeline(
    device: &DeviceRef,
    library: &Library,
    function_name: &str,
) -> Result<ComputePipelineState, MetalSupportError> {
    let function = library
        .get_function(function_name, None)
        .map_err(|message| MetalSupportError::PipelineFunction {
            function_name: function_name.to_string(),
            message,
        })?;
    let mut error: *mut Object = core::ptr::null_mut();
    // SAFETY: The selector ABI matches MTLDevice, the retained function remains
    // live for the synchronous call, and the result is checked before wrapping.
    let raw: *mut MTLComputePipelineState = unsafe {
        device
            .send_message(
                Sel::register("newComputePipelineStateWithFunction:error:"),
                (function.as_ptr(), core::ptr::from_mut(&mut error)),
            )
            .map_err(|dispatch_error| MetalSupportError::PipelineState {
                function_name: function_name.to_string(),
                message: format!("Metal pipeline dispatch failed: {dispatch_error}"),
            })?
    };
    if !error.is_null() {
        if !raw.is_null() {
            // SAFETY: A non-null raw pipeline is retained even on a diagnostic
            // path; wrap and immediately drop it to balance ownership.
            drop(unsafe { ComputePipelineState::from_ptr(raw) });
        }
        // SAFETY: The non-null NSError came from the synchronous pipeline call.
        let message = unsafe { objective_c_error_message(error) };
        return Err(MetalSupportError::PipelineState {
            function_name: function_name.to_string(),
            message,
        });
    }
    if raw.is_null() {
        Err(MetalSupportError::PipelineState {
            function_name: function_name.to_string(),
            message: "Metal returned nil without a pipeline error".to_string(),
        })
    } else {
        // SAFETY: The raw selector returned a non-null retained pipeline.
        Ok(unsafe { ComputePipelineState::from_ptr(raw) })
    }
}

/// Convenience loader for many pipelines from one Metal shader library.
pub struct MetalPipelineLoader {
    device: Device,
    library: Library,
}

impl MetalPipelineLoader {
    /// Compile `source` and keep the resulting library for named pipeline loads.
    ///
    /// # Errors
    ///
    /// Returns [`MetalSupportError::ShaderLibrary`] when Metal rejects the source.
    pub fn new(device: &DeviceRef, source: &str) -> Result<Self, MetalSupportError> {
        Ok(Self {
            device: device.to_owned(),
            library: shader_library(device, source)?,
        })
    }

    /// Load one compute pipeline from the cached shader library.
    ///
    /// # Errors
    ///
    /// Returns a typed lookup or pipeline-construction error.
    pub fn pipeline(&self, function_name: &str) -> Result<ComputePipelineState, MetalSupportError> {
        named_pipeline(&self.device, &self.library, function_name)
    }

    /// Borrow the compiled shader library.
    #[must_use]
    pub fn library(&self) -> &Library {
        &self.library
    }
}
