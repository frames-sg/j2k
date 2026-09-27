// SPDX-License-Identifier: MIT OR Apache-2.0

//! Reuse of batch output buffers whose surfaces have all been dropped.
//!
//! Batch decodes that return [`Surface`](crate::Surface)s write every tile into
//! one shared buffer, and each surface retains that buffer. Allocating a fresh
//! buffer per call costs the driver about 0.26 ms of command-buffer scheduling
//! for 16 RGB tiles of 512x512 (mapping the new pages), which is a quarter of
//! the call. The pool hands a buffer out again only when it holds the sole
//! reference: every live surface and every in-flight command buffer that used
//! the buffer retains it, so no output a caller can still read is overwritten.

use std::sync::Mutex;

use objc2::runtime::NSObjectProtocol as _;
use objc2_metal::MTLBuffer as _;

use crate::buffers::new_shared_buffer;
use crate::metal_types::{Buffer, DeviceRef};
use crate::Error;

/// Two slots let a caller keep one batch's surfaces while decoding the next.
const SURFACE_OUTPUT_POOL_SLOTS: usize = 2;
/// Smaller outputs are allocated per call, as before: the driver serves them
/// without a page-mapping cost, and reusing one measured slower (64 RGB tiles
/// of 16x16, 49 KiB: +9% to +17% per batch).
const SURFACE_OUTPUT_POOL_MIN_BUFFER_BYTES: usize = 1024 * 1024;
/// Larger outputs are allocated per call, as before, so an idle session
/// retains at most `SLOTS x 64 MiB` of output buffers (64 RGB tiles of 512x512
/// need 48 MiB).
const SURFACE_OUTPUT_POOL_MAX_BUFFER_BYTES: usize = 64 * 1024 * 1024;

#[derive(Default)]
pub(super) struct SurfaceOutputPool {
    slots: Mutex<Vec<Buffer>>,
}

impl SurfaceOutputPool {
    /// Returns a shared buffer of at least `bytes`, reusing a pooled buffer
    /// that nothing else references.
    pub(super) fn acquire(&self, device: &DeviceRef, bytes: usize) -> Result<Buffer, Error> {
        if !(SURFACE_OUTPUT_POOL_MIN_BUFFER_BYTES..=SURFACE_OUTPUT_POOL_MAX_BUFFER_BYTES)
            .contains(&bytes)
        {
            return new_shared_buffer(device, bytes);
        }
        let mut slots = self.slots.lock().map_err(|_| Error::MetalStatePoisoned {
            state: "JPEG Metal surface output pool",
        })?;
        // The count is read under the pool lock. New references to a pooled
        // buffer come only from this pool or from cloning a surface that
        // already holds one, so a count of one cannot rise concurrently.
        let is_free = |buffer: &Buffer| buffer.retainCount() == 1;
        if let Some(buffer) = slots
            .iter()
            .find(|buffer| is_free(buffer) && buffer.length() >= bytes)
        {
            return Ok(buffer.clone());
        }
        let buffer = new_shared_buffer(device, bytes)?;
        if slots.len() < SURFACE_OUTPUT_POOL_SLOTS {
            slots.push(buffer.clone());
        } else if let Some(slot) = slots.iter_mut().find(|buffer| is_free(buffer)) {
            // A free slot was too small; keep the larger buffer instead.
            *slot = buffer.clone();
        }
        Ok(buffer)
    }

    /// Forgets every pooled buffer. Surfaces keep their own references, so a
    /// buffer still in use is freed only when its last surface drops.
    pub(super) fn release(&self) -> Result<(), Error> {
        self.slots
            .lock()
            .map_err(|_| Error::MetalStatePoisoned {
                state: "JPEG Metal surface output pool",
            })?
            .clear();
        Ok(())
    }

    #[cfg(test)]
    pub(super) fn pooled_count_for_test(&self) -> usize {
        self.slots.lock().map_or(0, |slots| slots.len())
    }
}
