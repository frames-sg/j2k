// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{cell::Cell, ffi::c_void};

use super::*;

#[test]
fn failed_function_lookup_unloads_loaded_module_and_preserves_primary_error() {
    let module = std::ptr::without_provenance_mut::<c_void>(1);
    let unloaded = Cell::new(std::ptr::null_mut());
    let result = resolve_loaded_kernel_function(
        module,
        |_| {
            Err(CudaError::InvalidArgument {
                message: "missing entrypoint".to_string(),
            })
        },
        |module| {
            unloaded.set(module);
            Ok(())
        },
    );

    assert!(matches!(
        result,
        Err(CudaError::InvalidArgument { message }) if message == "missing entrypoint"
    ));
    assert_eq!(unloaded.get(), module);
}

#[test]
fn failed_function_lookup_retains_rollback_failure_diagnostic() {
    let module = std::ptr::without_provenance_mut::<c_void>(1);
    let result = resolve_loaded_kernel_function(
        module,
        |_| {
            Err(CudaError::InvalidArgument {
                message: "missing entrypoint".to_string(),
            })
        },
        |_| {
            Err(CudaError::InvalidArgument {
                message: "module unload failed".to_string(),
            })
        },
    );

    let error = result.expect_err("lookup and rollback failures must be retained");
    assert!(matches!(error, CudaError::ResourceReleaseFailed { .. }));
    let message = error.to_string();
    assert!(message.contains("missing entrypoint"));
    assert!(message.contains("module unload failed"));
}

#[test]
fn successful_null_function_lookup_unloads_loaded_module() {
    let module = std::ptr::without_provenance_mut::<c_void>(1);
    let unloaded = Cell::new(std::ptr::null_mut());
    let result = resolve_loaded_kernel_function(
        module,
        |_| Ok(std::ptr::null_mut()),
        |module| {
            unloaded.set(module);
            Ok(())
        },
    );

    assert!(matches!(
        result,
        Err(CudaError::InternalInvariant { what })
            if what == "CUDA returned a null function after successful lookup"
    ));
    assert_eq!(unloaded.get(), module);
}
