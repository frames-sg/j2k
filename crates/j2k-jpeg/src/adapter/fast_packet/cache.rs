// SPDX-License-Identifier: MIT OR Apache-2.0

//! Backend-neutral ownership and retention cache for accelerator JPEG plans.

use alloc::collections::TryReserveError;

use super::FastPacketError;
use crate::JpegError;

mod build;
mod packet;
mod plan;
mod shared_allocation;
mod shared_input;
mod store;

pub use packet::{JpegFastPacket, SharedJpegFastPacket};
pub use plan::{JpegCachedPlan, JpegFastPacketState};
pub use shared_input::SharedJpegInput;
pub use store::{
    JpegPlanCache, JpegPlanCacheDiagnostics, JpegPlanCacheInsert, DEFAULT_JPEG_PLAN_CACHE_ENTRIES,
    DEFAULT_JPEG_PLAN_CACHE_HOST_BYTES,
};

#[derive(Debug, Clone, thiserror::Error)]
#[doc(hidden)]
/// Hard failure while constructing or updating a shared JPEG accelerator cache.
pub enum JpegPlanCacheError {
    /// A requested owner graph exceeds its explicit host-memory limit.
    #[error(
        "host allocation limit exceeded for {what}: requested {requested} bytes, cap {cap} bytes"
    )]
    Limit {
        /// Logical allocation purpose.
        what: &'static str,
        /// Requested logical or allocator-reported bytes.
        requested: usize,
        /// Maximum permitted retained bytes.
        cap: usize,
    },
    /// A size-dependent host allocation failed after checked preflight.
    #[error("host allocation failed for {bytes} bytes while allocating {what}: {source}")]
    Allocation {
        /// Stable allocation context.
        what: &'static str,
        /// Requested bytes before allocator capacity rounding.
        bytes: usize,
        /// Concrete allocator failure.
        #[source]
        source: TryReserveError,
    },
    /// Cache ownership or byte-accounting state violated an internal invariant.
    #[error("JPEG accelerator plan cache invariant failed: {0}")]
    Invariant(&'static str),
}

#[derive(Debug, Clone, thiserror::Error)]
#[doc(hidden)]
/// Hard failure while building an inspect-once cached JPEG accelerator plan.
pub enum JpegCachedPlanBuildError {
    /// JPEG parsing or decoder construction failed.
    #[error(transparent)]
    Decode(#[from] JpegError),
    /// Fast-packet inspection or materialization failed.
    #[error(transparent)]
    FastPacket(#[from] FastPacketError),
    /// Shared ownership or cache accounting failed.
    #[error(transparent)]
    Cache(#[from] JpegPlanCacheError),
}

impl JpegPlanCacheError {
    pub(super) fn allocation(what: &'static str, bytes: usize, source: TryReserveError) -> Self {
        Self::Allocation {
            what,
            bytes,
            source,
        }
    }
}

#[cfg(test)]
mod tests;
