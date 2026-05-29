// SPDX-License-Identifier: Apache-2.0

/// Caller-owned reusable allocation pool.
pub trait ScratchPool: Send {
    /// Return the approximate bytes currently retained by the pool.
    fn bytes_allocated(&self) -> usize;
    /// Clear reusable contents while keeping allocations for future decodes.
    fn reset(&mut self);
}
