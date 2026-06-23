// SPDX-License-Identifier: MIT OR Apache-2.0

/// Caller-owned reusable scratch allocations for codec implementations.
pub trait ScratchPool: Send {
    /// Return the number of bytes currently retained by the pool.
    fn bytes_allocated(&self) -> usize;
    /// Clear reusable allocations or cached state held by the pool.
    fn reset(&mut self);
}
