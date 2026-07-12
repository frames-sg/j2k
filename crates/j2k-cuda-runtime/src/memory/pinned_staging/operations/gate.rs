// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::context::CudaContext;
use std::sync::MutexGuard;
use std::{cell::Cell, marker::PhantomData};

#[doc(hidden)]
/// Clone-shared transaction guard for one context's page-locked upload staging.
///
/// Holding this guard keeps staging-pool diagnostics, host admission, upload,
/// and recycle in one serialized operation across every clone of the context.
#[must_use = "the pinned-upload transaction ends when this guard is dropped"]
pub struct CudaPinnedUploadOperationGuard<'a> {
    pub(super) context: &'a CudaContext,
    pub(super) _gate: MutexGuard<'a, ()>,
    pub(super) _not_sync: PhantomData<Cell<()>>,
}
