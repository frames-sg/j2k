// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(test)]
use std::cell::Cell;
use std::{collections::HashMap, sync::Mutex};

use j2k_metal_support::{private_buffer, shared_buffer};
use metal::{Buffer, Device};

use crate::Error;

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
    private: Mutex<HashMap<usize, Vec<Buffer>>>,
    shared: Mutex<HashMap<usize, Vec<Buffer>>>,
}

impl MetalBufferPools {
    pub(crate) fn new() -> Self {
        Self {
            private: Mutex::new(HashMap::new()),
            shared: Mutex::new(HashMap::new()),
        }
    }

    pub(crate) fn take_private(&self, device: &Device, bytes: usize) -> Result<Buffer, Error> {
        let bytes = bytes.max(1);
        let mut pool = self.private.lock().map_err(|_| Error::MetalStatePoisoned {
            state: "j2k metal private buffer pool",
        })?;
        if let Some(buffer) = pool.get_mut(&bytes).and_then(Vec::pop) {
            Ok(buffer)
        } else {
            #[cfg(test)]
            record_private_buffer_pool_miss_for_test();
            Ok(private_buffer(device, bytes))
        }
    }

    pub(crate) fn recycle_private(&self, bytes: usize, buffer: Buffer) -> Result<(), Error> {
        let bytes = bytes.max(1);
        self.private
            .lock()
            .map_err(|_| Error::MetalStatePoisoned {
                state: "j2k metal private buffer pool",
            })?
            .entry(bytes)
            .or_default()
            .push(buffer);
        Ok(())
    }

    pub(crate) fn take_shared(&self, device: &Device, bytes: usize) -> Result<Buffer, Error> {
        let bytes = bytes.max(1);
        let mut pool = self.shared.lock().map_err(|_| Error::MetalStatePoisoned {
            state: "j2k metal shared buffer pool",
        })?;
        if let Some(buffer) = pool.get_mut(&bytes).and_then(Vec::pop) {
            Ok(buffer)
        } else {
            #[cfg(test)]
            record_shared_buffer_pool_miss_for_test();
            Ok(shared_buffer(device, bytes))
        }
    }

    pub(crate) fn recycle_shared(&self, bytes: usize, buffer: Buffer) -> Result<(), Error> {
        let bytes = bytes.max(1);
        self.shared
            .lock()
            .map_err(|_| Error::MetalStatePoisoned {
                state: "j2k metal shared buffer pool",
            })?
            .entry(bytes)
            .or_default()
            .push(buffer);
        Ok(())
    }
}
