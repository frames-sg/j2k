// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::atomic::{AtomicU64, Ordering};

use crate::{driver::CuEvent, error::CudaError};

use super::CudaContext;

#[derive(Debug, Default)]
pub(crate) struct CudaEventPoolState {
    cached: Vec<usize>,
    driver_allocations: u64,
    reuses: u64,
}

#[derive(Debug, Default)]
pub(crate) struct CudaContextDiagnosticsState {
    host_to_device_operations: AtomicU64,
    host_to_device_bytes: AtomicU64,
    device_to_host_operations: AtomicU64,
    device_to_host_bytes: AtomicU64,
    status_device_to_host_operations: AtomicU64,
    status_device_to_host_bytes: AtomicU64,
    kernel_launches: AtomicU64,
    device_allocation_operations: AtomicU64,
    device_allocation_bytes: AtomicU64,
    live_device_allocations: AtomicU64,
    live_device_bytes: AtomicU64,
    peak_live_device_allocations: AtomicU64,
    peak_live_device_bytes: AtomicU64,
    event_host_synchronizations: AtomicU64,
    context_host_synchronizations: AtomicU64,
}

impl CudaContextDiagnosticsState {
    fn record_host_to_device_copy(&self, byte_len: usize) {
        atomic_saturating_add(&self.host_to_device_operations, 1);
        atomic_saturating_add(&self.host_to_device_bytes, usize_as_u64(byte_len));
    }

    fn record_kernel_launch(&self) {
        atomic_saturating_add(&self.kernel_launches, 1);
    }

    fn record_device_allocation(&self, byte_len: usize) {
        atomic_saturating_add(&self.device_allocation_operations, 1);
        atomic_saturating_add(&self.device_allocation_bytes, usize_as_u64(byte_len));
        let live_allocations = atomic_saturating_add(&self.live_device_allocations, 1);
        let live_bytes = atomic_saturating_add(&self.live_device_bytes, usize_as_u64(byte_len));
        self.peak_live_device_allocations
            .fetch_max(live_allocations, Ordering::Relaxed);
        self.peak_live_device_bytes
            .fetch_max(live_bytes, Ordering::Relaxed);
    }

    fn record_device_free(&self, byte_len: usize) {
        atomic_saturating_sub(&self.live_device_allocations, 1);
        atomic_saturating_sub(&self.live_device_bytes, usize_as_u64(byte_len));
    }
}

impl CudaEventPoolState {
    pub(crate) fn take(&mut self) -> Option<CuEvent> {
        let event = self.cached.pop()?;
        self.reuses = self.reuses.saturating_add(1);
        Some(event as CuEvent)
    }

    pub(crate) fn record_driver_allocation(&mut self) {
        self.driver_allocations = self.driver_allocations.saturating_add(1);
    }

    pub(crate) fn recycle(&mut self, event: CuEvent) -> Result<(), ()> {
        self.cached.try_reserve(1).map_err(|_| ())?;
        self.cached.push(event as usize);
        Ok(())
    }

    pub(crate) fn drain(&mut self) -> impl Iterator<Item = CuEvent> + '_ {
        self.cached.drain(..).map(|event| event as CuEvent)
    }
}

/// Monotonic runtime-work counters and the current event-cache size for one CUDA context.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CudaContextDiagnostics {
    /// Successful host-to-device copy operations submitted through this runtime.
    pub host_to_device_operations: u64,
    /// Successful host-to-device bytes copied through this runtime.
    pub host_to_device_bytes: u64,
    /// Successful device-to-host copy operations submitted through this runtime.
    pub device_to_host_operations: u64,
    /// Successful device-to-host bytes copied through this runtime.
    pub device_to_host_bytes: u64,
    /// Device-to-host operations used specifically to validate codec statuses.
    pub status_device_to_host_operations: u64,
    /// Device-to-host bytes used specifically to validate codec statuses.
    pub status_device_to_host_bytes: u64,
    /// Successful kernel submissions made through this runtime.
    pub kernel_launches: u64,
    /// Successful runtime-owned CUDA device allocation operations.
    pub device_allocation_operations: u64,
    /// Cumulative bytes requested by successful runtime-owned device allocations.
    pub device_allocation_bytes: u64,
    /// Runtime-owned CUDA device allocations that have not been successfully freed.
    pub live_device_allocations: u64,
    /// Runtime-owned CUDA device bytes that have not been successfully freed.
    pub live_device_bytes: u64,
    /// Highest observed number of simultaneous runtime-owned device allocations.
    pub peak_live_device_allocations: u64,
    /// Highest observed runtime-owned device byte total.
    pub peak_live_device_bytes: u64,
    /// CUDA event handles created by the driver for this context.
    pub event_driver_allocations: u64,
    /// Event checkouts satisfied from this context's re-recordable event cache.
    pub event_reuses: u64,
    /// Event handles currently retained for reuse by this context.
    pub cached_events: usize,
    /// Successful host waits on individual CUDA events.
    pub event_host_synchronizations: u64,
    /// Successful context-wide host synchronization operations.
    pub context_host_synchronizations: u64,
}

impl CudaContext {
    /// Snapshot runtime-work diagnostics without synchronizing CUDA work.
    #[doc(hidden)]
    pub fn diagnostics(&self) -> Result<CudaContextDiagnostics, CudaError> {
        self.inner.ensure_resource_lifetime_available()?;
        let events = self
            .inner
            .event_pool
            .lock()
            .map_err(|error| CudaError::StatePoisoned {
                message: error.to_string(),
            })?;
        Ok(CudaContextDiagnostics {
            host_to_device_operations: self
                .inner
                .diagnostics
                .host_to_device_operations
                .load(Ordering::Relaxed),
            host_to_device_bytes: self
                .inner
                .diagnostics
                .host_to_device_bytes
                .load(Ordering::Relaxed),
            device_to_host_operations: self
                .inner
                .diagnostics
                .device_to_host_operations
                .load(Ordering::Relaxed),
            device_to_host_bytes: self
                .inner
                .diagnostics
                .device_to_host_bytes
                .load(Ordering::Relaxed),
            status_device_to_host_operations: self
                .inner
                .diagnostics
                .status_device_to_host_operations
                .load(Ordering::Relaxed),
            status_device_to_host_bytes: self
                .inner
                .diagnostics
                .status_device_to_host_bytes
                .load(Ordering::Relaxed),
            kernel_launches: self
                .inner
                .diagnostics
                .kernel_launches
                .load(Ordering::Relaxed),
            device_allocation_operations: self
                .inner
                .diagnostics
                .device_allocation_operations
                .load(Ordering::Relaxed),
            device_allocation_bytes: self
                .inner
                .diagnostics
                .device_allocation_bytes
                .load(Ordering::Relaxed),
            live_device_allocations: self
                .inner
                .diagnostics
                .live_device_allocations
                .load(Ordering::Relaxed),
            live_device_bytes: self
                .inner
                .diagnostics
                .live_device_bytes
                .load(Ordering::Relaxed),
            peak_live_device_allocations: self
                .inner
                .diagnostics
                .peak_live_device_allocations
                .load(Ordering::Relaxed),
            peak_live_device_bytes: self
                .inner
                .diagnostics
                .peak_live_device_bytes
                .load(Ordering::Relaxed),
            event_driver_allocations: events.driver_allocations,
            event_reuses: events.reuses,
            cached_events: events.cached.len(),
            event_host_synchronizations: self
                .inner
                .diagnostics
                .event_host_synchronizations
                .load(Ordering::Relaxed),
            context_host_synchronizations: self
                .inner
                .diagnostics
                .context_host_synchronizations
                .load(Ordering::Relaxed),
        })
    }

    pub(crate) fn record_device_to_host_copy(&self, byte_len: usize) {
        atomic_saturating_add(&self.inner.diagnostics.device_to_host_operations, 1);
        atomic_saturating_add(
            &self.inner.diagnostics.device_to_host_bytes,
            u64::try_from(byte_len).unwrap_or(u64::MAX),
        );
    }

    pub(crate) fn record_host_to_device_copy(&self, byte_len: usize) {
        self.inner.diagnostics.record_host_to_device_copy(byte_len);
    }

    pub(crate) fn record_kernel_launch(&self) {
        self.inner.diagnostics.record_kernel_launch();
    }

    pub(crate) fn record_device_allocation(&self, byte_len: usize) {
        self.inner.diagnostics.record_device_allocation(byte_len);
    }

    pub(crate) fn record_device_free(&self, byte_len: usize) {
        self.inner.diagnostics.record_device_free(byte_len);
    }

    pub(crate) fn record_status_device_to_host_copy(&self, byte_len: usize) {
        atomic_saturating_add(&self.inner.diagnostics.status_device_to_host_operations, 1);
        atomic_saturating_add(
            &self.inner.diagnostics.status_device_to_host_bytes,
            u64::try_from(byte_len).unwrap_or(u64::MAX),
        );
    }

    pub(crate) fn record_event_host_synchronization(&self) {
        atomic_saturating_add(&self.inner.diagnostics.event_host_synchronizations, 1);
    }

    pub(crate) fn record_context_host_synchronization(&self) {
        atomic_saturating_add(&self.inner.diagnostics.context_host_synchronizations, 1);
    }
}

fn usize_as_u64(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

fn atomic_saturating_add(counter: &AtomicU64, increment: u64) -> u64 {
    counter
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
            Some(current.saturating_add(increment))
        })
        .map_or_else(
            |current| current,
            |previous| previous.saturating_add(increment),
        )
}

fn atomic_saturating_sub(counter: &AtomicU64, decrement: u64) -> u64 {
    counter
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
            Some(current.saturating_sub(decrement))
        })
        .map_or_else(
            |current| current,
            |previous| previous.saturating_sub(decrement),
        )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagnostics_state_tracks_successful_runtime_work_and_live_allocation_peaks() {
        let state = CudaContextDiagnosticsState::default();

        state.record_host_to_device_copy(17);
        state.record_host_to_device_copy(5);
        state.record_kernel_launch();
        state.record_kernel_launch();
        state.record_device_allocation(64);
        state.record_device_allocation(96);
        state.record_device_free(64);
        state.record_device_allocation(32);

        assert_eq!(state.host_to_device_operations.load(Ordering::Relaxed), 2);
        assert_eq!(state.host_to_device_bytes.load(Ordering::Relaxed), 22);
        assert_eq!(state.kernel_launches.load(Ordering::Relaxed), 2);
        assert_eq!(
            state.device_allocation_operations.load(Ordering::Relaxed),
            3
        );
        assert_eq!(state.device_allocation_bytes.load(Ordering::Relaxed), 192);
        assert_eq!(state.live_device_allocations.load(Ordering::Relaxed), 2);
        assert_eq!(state.live_device_bytes.load(Ordering::Relaxed), 128);
        assert_eq!(
            state.peak_live_device_allocations.load(Ordering::Relaxed),
            2
        );
        assert_eq!(state.peak_live_device_bytes.load(Ordering::Relaxed), 160);
    }
}
