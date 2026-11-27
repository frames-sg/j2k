// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared Metal runtime, allocation, access, and dispatch helpers for J2K
//! adapter crates.
//!
//! This crate owns the Metal runtime for every J2K Metal adapter. Owned
//! Objective-C objects are `Retained<ProtocolObject<dyn MTL...>>`, and
//! borrowed values in the expert API are `&ProtocolObject<dyn MTL...>`.
//! `metal-rs` types are not accepted.

#![warn(unreachable_pub)]
#![warn(clippy::non_send_fields_in_send_ty)]
mod completion_slots;
mod error;
mod route;
mod submission_queue;

#[doc(hidden)]
pub use completion_slots::{CompletionSlot, CompletionSlots};
pub use error::{MetalCommandEncoderKind, MetalSupportError};
pub use route::{
    cpu_host_route, metal_kernel_route, metal_unavailable_route, reject_explicit_metal_route,
    reject_unsupported_backend_route, MetalRouteProfileLabels,
};
#[doc(hidden)]
pub use submission_queue::FallibleSubmissionQueue;

#[cfg(target_os = "macos")]
mod allocation;
#[cfg(target_os = "macos")]
mod buffer_access;
#[cfg(target_os = "macos")]
mod dispatch;
#[cfg(target_os = "macos")]
mod pipeline;
#[cfg(target_os = "macos")]
mod resident;
#[cfg(target_os = "macos")]
mod runtime;

#[cfg(target_os = "macos")]
pub use allocation::{
    checked_heap_shared_buffer, checked_private_buffer, checked_private_buffer_for_len,
    checked_shared_buffer, checked_shared_buffer_for_len, checked_shared_buffer_with_bytes,
    checked_shared_buffer_with_slice, checked_shared_heap, checked_texture,
};
#[cfg(target_os = "macos")]
pub use buffer_access::{
    checked_buffer_fill_bytes, checked_buffer_read, checked_buffer_read_vec, checked_buffer_write,
    with_checked_buffer_slice_mut,
};
#[cfg(target_os = "macos")]
pub use dispatch::{
    dispatch_1d_pipeline, dispatch_2d_pipeline, dispatch_3d_pipeline, dispatch_single_thread,
    mtl_size, one_d_threads_per_group, two_d_threads_per_group,
};
#[cfg(target_os = "macos")]
pub use pipeline::{named_pipeline, shader_library, MetalPipelineLoader};
#[cfg(target_os = "macos")]
pub use resident::{
    MetalImageDestination, MetalImageLayout, ResidentMetalImage, SubmittedMetalImages,
};
#[cfg(target_os = "macos")]
pub use runtime::{
    checked_blit_command_encoder, checked_command_buffer, checked_command_queue,
    checked_compute_command_encoder, checked_event, checked_shared_event, commit_and_wait,
    ensure_completed, system_default_device, wait_for_completion, MetalRuntimeSession,
};

#[cfg(all(test, target_os = "macos"))]
mod tests;
