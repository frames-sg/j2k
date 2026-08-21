// SPDX-License-Identifier: MIT OR Apache-2.0

//! Public JPEG Metal batch semantics and source-neutral planning.
//!
//! Queue grouping, flushing, and completion remain owned by [`crate::batch`].

#[cfg(target_os = "macos")]
mod buffer_target;
#[cfg(target_os = "macos")]
mod inspect;
#[cfg(target_os = "macos")]
mod owner_accounting;
#[cfg(target_os = "macos")]
mod plan;
#[cfg(target_os = "macos")]
mod request;
#[cfg(target_os = "macos")]
mod source;
mod submit;
#[cfg(target_os = "macos")]
mod texture_target;

#[cfg(target_os = "macos")]
pub use request::{
    MetalBufferBatchTarget, MetalTextureBatchTarget, Rgb8MetalBatchOp, Rgb8MetalBatchRequest,
    Rgb8MetalBatchSource,
};
