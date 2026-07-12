// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::{sync::Arc, vec::Vec};

mod accounting;
mod construction;

struct SharedJpegInputInner {
    bytes: Vec<u8>,
}

#[derive(Clone)]
enum SharedJpegInputStorage {
    Copied(Arc<SharedJpegInputInner>),
    ArcSlice(Arc<[u8]>),
}

#[derive(Clone)]
#[doc(hidden)]
/// JPEG input shared cheaply between a request and its cache.
///
/// Borrowed slices are copied into a fallibly reserved `Vec`; caller-owned
/// immutable `Arc<[u8]>` payloads can instead move in without another payload
/// copy. Vector owners are accounted by allocator-reported capacity and Arc
/// slices by their fixed length. Stable Rust exposes neither fallible Arc
/// allocation nor allocator usable-size for its control block, so diagnostics
/// add a two-counter estimate but cannot observe fixed-allocation rounding.
pub struct SharedJpegInput(SharedJpegInputStorage);

impl SharedJpegInput {
    /// Borrow the complete copied JPEG input.
    #[must_use]
    pub fn as_slice(&self) -> &[u8] {
        match &self.0 {
            SharedJpegInputStorage::Copied(input) => input.bytes.as_slice(),
            SharedJpegInputStorage::ArcSlice(input) => input.as_ref(),
        }
    }

    /// Borrow the complete copied JPEG input.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        self.as_slice()
    }
}

impl AsRef<[u8]> for SharedJpegInput {
    fn as_ref(&self) -> &[u8] {
        self.as_slice()
    }
}

impl core::fmt::Debug for SharedJpegInput {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("SharedJpegInput")
            .field("len", &self.as_slice().len())
            .field("capacity", &self.data_capacity())
            .finish_non_exhaustive()
    }
}
