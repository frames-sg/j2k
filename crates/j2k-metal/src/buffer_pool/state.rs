// SPDX-License-Identifier: MIT OR Apache-2.0

use metal::{Buffer, Device};

const DEFAULT_RETAINED_BYTES_PER_POOL: usize = 256 * 1024 * 1024;
const DEFAULT_RETAINED_BUFFERS_PER_POOL: usize = 64;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct PoolLimits {
    retained_bytes: usize,
    retained_buffers: usize,
}

impl PoolLimits {
    pub(super) fn for_device(device: &Device) -> Self {
        let device_limit =
            usize::try_from(device.max_buffer_length()).map_or(usize::MAX, |bytes| bytes);
        Self {
            retained_bytes: device_limit.min(DEFAULT_RETAINED_BYTES_PER_POOL),
            retained_buffers: DEFAULT_RETAINED_BUFFERS_PER_POOL,
        }
    }

    #[cfg(test)]
    pub(super) const fn new(retained_bytes: usize, retained_buffers: usize) -> Self {
        Self {
            retained_bytes,
            retained_buffers,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
/// Retention and high-water counters for one Metal scratch-buffer pool.
pub struct MetalBufferPoolDiagnostics {
    /// Bytes currently retained in completed reusable buffers.
    pub cached_bytes: usize,
    /// Completed reusable buffers currently retained.
    pub cached_buffers: usize,
    /// Allocator-reported capacity of the flat host metadata owner.
    pub metadata_capacity: usize,
    /// Highest completed-buffer byte count retained by this pool.
    pub peak_cached_bytes: usize,
    /// Highest completed-buffer count retained by this pool.
    pub peak_cached_buffers: usize,
    /// Oldest completed buffers evicted to admit more useful entries.
    pub evictions: usize,
    /// Completed buffers deliberately declined instead of retained.
    pub rejections: usize,
    /// Metadata reservations rejected by the host allocator.
    pub metadata_failures: usize,
    /// Caller size records that disagreed with the Metal allocation length.
    pub size_mismatches: usize,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
/// Separate diagnostics for private and shared Metal scratch retention.
pub struct MetalBufferPoolsDiagnostics {
    /// Private-storage scratch pool counters.
    pub private: MetalBufferPoolDiagnostics,
    /// Shared-storage scratch pool counters.
    pub shared: MetalBufferPoolDiagnostics,
}

#[derive(Default)]
struct PoolCounters {
    peak_cached_bytes: usize,
    peak_cached_buffers: usize,
    evictions: usize,
    rejections: usize,
    metadata_failures: usize,
    size_mismatches: usize,
}

struct PooledBuffer {
    bytes: usize,
    buffer: Buffer,
}

pub(super) struct PoolState {
    entries: Vec<PooledBuffer>,
    retained_bytes: usize,
    limits: PoolLimits,
    counters: PoolCounters,
    #[cfg(test)]
    fail_next_metadata_reserve: bool,
}

impl PoolState {
    pub(super) fn new(limits: PoolLimits) -> Self {
        Self {
            entries: Vec::new(),
            retained_bytes: 0,
            limits,
            counters: PoolCounters::default(),
            #[cfg(test)]
            fail_next_metadata_reserve: false,
        }
    }

    pub(super) fn take(&mut self, bytes: usize) -> Result<Option<Buffer>, &'static str> {
        let Some(index) = self.entries.iter().rposition(|entry| entry.bytes == bytes) else {
            return Ok(None);
        };
        let entry = self.entries.remove(index);
        self.retained_bytes = self
            .retained_bytes
            .checked_sub(entry.bytes)
            .ok_or("retained byte count underflow while taking a buffer")?;
        Ok(Some(entry.buffer))
    }

    pub(super) fn recycle(
        &mut self,
        expected_bytes: usize,
        buffer: Buffer,
    ) -> Result<(), &'static str> {
        let actual_bytes = usize::try_from(buffer.length())
            .map_err(|_| "Metal buffer length does not fit usize")?;
        if actual_bytes != expected_bytes {
            self.counters.size_mismatches = self
                .counters
                .size_mismatches
                .checked_add(1)
                .ok_or("size-mismatch counter overflow")?;
            return Err("recorded buffer size differs from the Metal allocation length");
        }
        if actual_bytes > self.limits.retained_bytes || self.limits.retained_buffers == 0 {
            self.record_rejection()?;
            return Ok(());
        }
        if !self.reserve_metadata_slot()? {
            return Ok(());
        }

        loop {
            let next_bytes = self
                .retained_bytes
                .checked_add(actual_bytes)
                .ok_or("retained byte count overflow while recycling a buffer")?;
            if self.entries.len() < self.limits.retained_buffers
                && next_bytes <= self.limits.retained_bytes
            {
                self.retained_bytes = next_bytes;
                break;
            }
            self.evict_oldest()?;
        }

        self.entries.push(PooledBuffer {
            bytes: actual_bytes,
            buffer,
        });
        self.counters.peak_cached_bytes = self.counters.peak_cached_bytes.max(self.retained_bytes);
        self.counters.peak_cached_buffers =
            self.counters.peak_cached_buffers.max(self.entries.len());
        Ok(())
    }

    fn reserve_metadata_slot(&mut self) -> Result<bool, &'static str> {
        #[cfg(test)]
        if std::mem::take(&mut self.fail_next_metadata_reserve) {
            self.record_metadata_failure()?;
            return Ok(false);
        }
        if self.entries.len() < self.entries.capacity() {
            return Ok(true);
        }
        if self.entries.try_reserve_exact(1).is_err() {
            self.record_metadata_failure()?;
            return Ok(false);
        }
        Ok(true)
    }

    fn evict_oldest(&mut self) -> Result<(), &'static str> {
        let evicted = self.entries.remove(0);
        self.retained_bytes = self
            .retained_bytes
            .checked_sub(evicted.bytes)
            .ok_or("retained byte count underflow while evicting a buffer")?;
        self.counters.evictions = self
            .counters
            .evictions
            .checked_add(1)
            .ok_or("eviction counter overflow")?;
        Ok(())
    }

    fn record_rejection(&mut self) -> Result<(), &'static str> {
        self.counters.rejections = self
            .counters
            .rejections
            .checked_add(1)
            .ok_or("rejection counter overflow")?;
        Ok(())
    }

    fn record_metadata_failure(&mut self) -> Result<(), &'static str> {
        self.counters.metadata_failures = self
            .counters
            .metadata_failures
            .checked_add(1)
            .ok_or("metadata-failure counter overflow")?;
        self.record_rejection()
    }

    pub(super) fn diagnostics(&self) -> MetalBufferPoolDiagnostics {
        MetalBufferPoolDiagnostics {
            cached_bytes: self.retained_bytes,
            cached_buffers: self.entries.len(),
            metadata_capacity: self.entries.capacity(),
            peak_cached_bytes: self.counters.peak_cached_bytes,
            peak_cached_buffers: self.counters.peak_cached_buffers,
            evictions: self.counters.evictions,
            rejections: self.counters.rejections,
            metadata_failures: self.counters.metadata_failures,
            size_mismatches: self.counters.size_mismatches,
        }
    }

    #[cfg(test)]
    pub(super) fn fail_next_metadata_reserve(&mut self) {
        self.fail_next_metadata_reserve = true;
    }
}
