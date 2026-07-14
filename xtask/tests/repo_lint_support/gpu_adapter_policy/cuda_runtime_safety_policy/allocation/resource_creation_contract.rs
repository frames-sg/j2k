// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{fs, path::Path};

use super::super::super::super::{assert_pattern_checks, PatternCheck};

pub(super) fn assert_policy(root: &Path) {
    let read = |relative: &str| {
        fs::read_to_string(root.join(relative))
            .unwrap_or_else(|error| panic!("read {relative}: {error}"))
    };
    let validation = read("crates/j2k-cuda-runtime/src/context/resource_creation.rs");
    let memory = read("crates/j2k-cuda-runtime/src/memory.rs");
    let context_creation = read("crates/j2k-cuda-runtime/src/context/creation.rs");
    let events = read("crates/j2k-cuda-runtime/src/execution/events.rs");
    let kernel_cache = read("crates/j2k-cuda-runtime/src/context/kernel_cache.rs");
    let kernel_tests = read("crates/j2k-cuda-runtime/src/context/kernel_cache/tests.rs");

    assert_pattern_checks(&[
        PatternCheck::new("CUDA resource creation validators", &validation).required(&[
            "pub(crate) fn validate_device_allocation(",
            "len != 0 && ptr == 0",
            "pub(crate) fn validate_resource_handle<T>(",
            "ptr.is_null()",
            "nonzero_null_device_allocation_is_rejected",
            "null_resource_handle_is_rejected",
        ]),
        PatternCheck::new("CUDA device allocation promotion", &memory).required(&[
            "cu_mem_alloc)(&raw mut ptr, bytes.len())",
            "validate_device_allocation(ptr, bytes.len())",
            "cu_mem_alloc)(&raw mut ptr, len)",
            "validate_device_allocation(ptr, len)",
        ]),
        PatternCheck::new("CUDA context promotion", &context_creation).required(&[
            "cu_ctx_create)(&raw mut context, 0, device)",
            "cu_device_primary_ctx_retain)(&raw mut context, device)",
            "validate_resource_handle(",
            "CUDA returned a null context after successful creation",
        ]),
        PatternCheck::new("CUDA stream and event promotion", &events).required(&[
            "cu_stream_create)(&raw mut stream, 0)",
            "CUDA returned a null stream after successful creation",
            "cu_event_create)(&raw mut event, 0)",
            "CUDA returned a null event after successful creation",
        ]),
        PatternCheck::new("CUDA module and function promotion", &kernel_cache).required(&[
            "cu_module_load_data)(",
            "CUDA returned a null module after successful load",
            "lookup(module).and_then(|function|",
            "CUDA returned a null function after successful lookup",
        ]),
        PatternCheck::new("CUDA module rollback regression", &kernel_tests).required(&[
            "successful_null_function_lookup_unloads_loaded_module",
            "assert_eq!(unloaded.get(), module);",
        ]),
    ]);
}
