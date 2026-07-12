// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(test)]
use std::cell::Cell;
use std::sync::Mutex;

use j2k_metal_support::{checked_private_buffer, checked_shared_buffer};
use metal::{Buffer, Device};

use crate::Error;

mod state;
#[cfg(test)]
mod tests;

pub use state::{MetalBufferPoolDiagnostics, MetalBufferPoolsDiagnostics};
use state::{PoolLimits, PoolState};

#[cfg(test)]
std::thread_local! {
    static PRIVATE_BUFFER_POOL_MISSES: Cell<usize> = const { Cell::new(0) };
    static SHARED_BUFFER_POOL_MISSES: Cell<usize> = const { Cell::new(0) };
}

#[cfg(test)]
pub(crate) fn reset_private_buffer_pool_misses_for_test() {
    PRIVATE_BUFFER_POOL_MISSES.with(|misses| misses.set(0));
}

#[cfg(test)]
pub(crate) fn private_buffer_pool_misses_for_test() -> usize {
    PRIVATE_BUFFER_POOL_MISSES.with(Cell::get)
}

#[cfg(test)]
pub(crate) fn reset_shared_buffer_pool_misses_for_test() {
    SHARED_BUFFER_POOL_MISSES.with(|misses| misses.set(0));
}

#[cfg(test)]
pub(crate) fn shared_buffer_pool_misses_for_test() -> usize {
    SHARED_BUFFER_POOL_MISSES.with(Cell::get)
}

#[cfg(test)]
fn record_private_buffer_pool_miss_for_test() {
    PRIVATE_BUFFER_POOL_MISSES.with(|misses| misses.set(misses.get() + 1));
}

#[cfg(test)]
fn record_shared_buffer_pool_miss_for_test() {
    SHARED_BUFFER_POOL_MISSES.with(|misses| misses.set(misses.get() + 1));
}

pub(crate) struct MetalBufferPools {
    private: Mutex<PoolState>,
    shared: Mutex<PoolState>,
}

impl MetalBufferPools {
    pub(crate) fn new(device: &Device) -> Self {
        let limits = PoolLimits::for_device(device);
        Self::with_limits(limits, limits)
    }

    fn with_limits(private: PoolLimits, shared: PoolLimits) -> Self {
        Self {
            private: Mutex::new(PoolState::new(private)),
            shared: Mutex::new(PoolState::new(shared)),
        }
    }

    pub(crate) fn take_private(&self, device: &Device, bytes: usize) -> Result<Buffer, Error> {
        let bytes = bytes.max(1);
        if let Some(buffer) = Self::take_from(&self.private, bytes, "private")? {
            return Ok(buffer);
        }
        #[cfg(test)]
        record_private_buffer_pool_miss_for_test();
        checked_private_buffer(device, bytes).map_err(|source| {
            crate::error::metal_kernel_support_error(
                "J2K Metal private buffer-pool allocation",
                source,
            )
        })
    }

    pub(crate) fn recycle_private(&self, bytes: usize, buffer: Buffer) -> Result<(), Error> {
        Self::recycle_into(&self.private, bytes.max(1), buffer, "private")
    }

    pub(crate) fn take_shared(&self, device: &Device, bytes: usize) -> Result<Buffer, Error> {
        let bytes = bytes.max(1);
        if let Some(buffer) = Self::take_from(&self.shared, bytes, "shared")? {
            return Ok(buffer);
        }
        #[cfg(test)]
        record_shared_buffer_pool_miss_for_test();
        checked_shared_buffer(device, bytes).map_err(|source| {
            crate::error::metal_kernel_support_error(
                "J2K Metal shared buffer-pool allocation",
                source,
            )
        })
    }

    pub(crate) fn recycle_shared(&self, bytes: usize, buffer: Buffer) -> Result<(), Error> {
        Self::recycle_into(&self.shared, bytes.max(1), buffer, "shared")
    }

    pub(crate) fn diagnostics(&self) -> Result<MetalBufferPoolsDiagnostics, Error> {
        Ok(MetalBufferPoolsDiagnostics {
            private: Self::pool_diagnostics(&self.private, "private")?,
            shared: Self::pool_diagnostics(&self.shared, "shared")?,
        })
    }

    fn take_from(
        pool: &Mutex<PoolState>,
        bytes: usize,
        state: &'static str,
    ) -> Result<Option<Buffer>, Error> {
        pool.lock()
            .map_err(|_| poisoned(state))?
            .take(bytes)
            .map_err(|reason| invariant(state, reason))
    }

    fn recycle_into(
        pool: &Mutex<PoolState>,
        bytes: usize,
        buffer: Buffer,
        state: &'static str,
    ) -> Result<(), Error> {
        pool.lock()
            .map_err(|_| poisoned(state))?
            .recycle(bytes, buffer)
            .map_err(|reason| invariant(state, reason))
    }

    fn pool_diagnostics(
        pool: &Mutex<PoolState>,
        state: &'static str,
    ) -> Result<MetalBufferPoolDiagnostics, Error> {
        Ok(pool.lock().map_err(|_| poisoned(state))?.diagnostics())
    }

    #[cfg(test)]
    fn with_limits_for_test(private: PoolLimits, shared: PoolLimits) -> Self {
        Self::with_limits(private, shared)
    }

    #[cfg(test)]
    fn private_diagnostics(&self) -> Result<MetalBufferPoolDiagnostics, Error> {
        Self::pool_diagnostics(&self.private, "private")
    }

    #[cfg(test)]
    fn shared_diagnostics(&self) -> Result<MetalBufferPoolDiagnostics, Error> {
        Self::pool_diagnostics(&self.shared, "shared")
    }

    #[cfg(test)]
    fn fail_next_private_metadata_reserve_for_test(&self) {
        self.private
            .lock()
            .expect("private pool test lock")
            .fail_next_metadata_reserve();
    }
}

fn poisoned(state: &'static str) -> Error {
    Error::MetalStatePoisoned {
        state: pool_state_name(state),
    }
}

fn invariant(state: &'static str, reason: &'static str) -> Error {
    Error::MetalStateInvariant {
        state: pool_state_name(state),
        reason,
    }
}

fn pool_state_name(state: &'static str) -> &'static str {
    match state {
        "private" => "j2k metal private buffer pool",
        "shared" => "j2k metal shared buffer pool",
        _ => "j2k metal buffer pool",
    }
}
