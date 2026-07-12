// SPDX-License-Identifier: MIT OR Apache-2.0

//! One synchronized neutral cache plus clone-shared in-flight owner ledger.

use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Mutex, TryLockError,
};

use j2k_jpeg::adapter::{
    decoder_retained_allocation_bytes, JpegCachedPlan, JpegCachedPlanBuildError, JpegPlanCache,
    JpegPlanCacheDiagnostics, SharedJpegFastPacket,
};
use j2k_jpeg::Decoder as CpuDecoder;

use super::host_ledger::{HostCacheTransactionError, HostOwnerLease, SharedHostLedger};
use crate::Error;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[expect(
    clippy::struct_field_names,
    reason = "the byte-unit suffix keeps cache and owner metrics explicit at the aggregation seam"
)]
pub(super) struct PacketHostMemoryDiagnostics {
    pub(super) cache_retained_bytes: usize,
    pub(super) active_owner_bytes: usize,
    pub(super) current_combined_bytes: usize,
    pub(super) peak_active_owner_bytes: usize,
    pub(super) peak_combined_bytes: usize,
}

pub(super) struct OwnedPacketPlanCache {
    pub(super) state: Mutex<JpegPlanCache>,
    last_coherent_entries: AtomicUsize,
    ledger: Arc<SharedHostLedger>,
}

pub(crate) struct LeasedOwnedPacket {
    pub(crate) packet: SharedJpegFastPacket,
    _lease: HostOwnerLease,
}

impl OwnedPacketPlanCache {
    #[cfg(test)]
    pub(super) fn with_limits(entry_limit: usize, host_byte_limit: usize) -> Self {
        Self {
            state: Mutex::new(JpegPlanCache::with_limits(entry_limit, host_byte_limit)),
            last_coherent_entries: AtomicUsize::new(0),
            ledger: SharedHostLedger::new(),
        }
    }

    pub(super) fn resolve_packet(&self, input: &[u8]) -> Result<Option<LeasedOwnedPacket>, Error> {
        self.resolve_packet_inner(None, |cache, external_live_bytes| {
            cache.resolve_with_external_live(input, external_live_bytes)
        })
    }

    pub(super) fn resolve_packet_from_decoder(
        &self,
        decoder: &CpuDecoder<'_>,
    ) -> Result<Option<LeasedOwnedPacket>, Error> {
        let decoder_bytes = decoder_retained_allocation_bytes(decoder)?;
        self.resolve_packet_inner(Some(decoder_bytes), |cache, external_live_bytes| {
            cache.resolve_from_decoder_with_external_live(decoder, external_live_bytes)
        })
    }

    fn resolve_packet_inner(
        &self,
        existing_decoder_bytes: Option<usize>,
        resolve: impl FnOnce(
            &mut JpegPlanCache,
            usize,
        ) -> Result<JpegCachedPlan, JpegCachedPlanBuildError>,
    ) -> Result<Option<LeasedOwnedPacket>, Error> {
        let _allocation = self.ledger.lock_allocations()?;
        let mut cache = self
            .state
            .lock()
            .map_err(|_| Error::OwnedPacketCachePoisoned)?;
        let cache_before = cache.diagnostics().retained_bytes;
        let result = self.ledger.transact_cache(cache_before, |owners_before| {
            let result = resolve(&mut cache, owners_before);
            (result, cache.diagnostics().retained_bytes)
        });
        let diagnostics = cache.diagnostics();
        self.record_coherent_entries(diagnostics);
        let current_combined = self.ledger.combined_bytes(diagnostics.retained_bytes)?;
        self.ledger.observe_combined(current_combined)?;
        let plan = match result {
            Ok(plan) => plan,
            Err(HostCacheTransactionError::Operation(error)) => {
                return Err(cached_plan_error(error));
            }
            Err(HostCacheTransactionError::Accounting(error)) => return Err(error),
            Err(HostCacheTransactionError::OperationAndAccounting {
                operation,
                accounting,
            }) => {
                return Err(Error::OperationAndHostAccountingFailed {
                    primary: Box::new(cached_plan_error(operation)),
                    accounting: Box::new(accounting),
                });
            }
        };
        let Some(packet) = plan.fast_packet().cloned() else {
            return Ok(None);
        };
        let packet_bytes = packet.retained_cache_bytes()?;
        let owner_bytes = packet_bytes
            .checked_add(existing_decoder_bytes.unwrap_or(0))
            .ok_or(Error::HostAllocationTooLarge {
                requested: usize::MAX,
                cap: j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
                what: "CUDA JPEG packet and existing decoder owners",
            })?;
        let lease = self.ledger.reserve(
            diagnostics.retained_bytes,
            owner_bytes,
            j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
            "CUDA JPEG cache, pinned staging, and in-flight host owners",
        )?;
        Ok(Some(LeasedOwnedPacket {
            packet,
            _lease: lease,
        }))
    }

    pub(super) fn allocate_host_owner<T>(
        &self,
        allocate: impl FnOnce(usize) -> Result<(T, usize), Error>,
    ) -> Result<(T, HostOwnerLease), Error> {
        let _allocation = self.ledger.lock_allocations()?;
        let cache = self
            .state
            .lock()
            .map_err(|_| Error::OwnedPacketCachePoisoned)?;
        let diagnostics = cache.diagnostics();
        self.ledger
            .allocate_host_owner(diagnostics.retained_bytes, allocate)
    }

    pub(super) fn reserve_existing_host_owner(
        &self,
        bytes: usize,
    ) -> Result<HostOwnerLease, Error> {
        let _allocation = self.ledger.lock_allocations()?;
        let cache = self
            .state
            .lock()
            .map_err(|_| Error::OwnedPacketCachePoisoned)?;
        self.ledger.reserve(
            cache.diagnostics().retained_bytes,
            bytes,
            j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
            "CUDA JPEG cache, pinned staging, and in-flight host owners",
        )
    }

    pub(super) fn host_live_bytes(&self) -> Result<usize, Error> {
        let _allocation = self.ledger.lock_allocations()?;
        let cache = self
            .state
            .lock()
            .map_err(|_| Error::OwnedPacketCachePoisoned)?;
        self.ledger
            .combined_bytes(cache.diagnostics().retained_bytes)
    }

    pub(super) fn host_live_bytes_without_pinned(&self) -> Result<usize, Error> {
        let _allocation = self.ledger.lock_allocations()?;
        let cache = self
            .state
            .lock()
            .map_err(|_| Error::OwnedPacketCachePoisoned)?;
        self.ledger
            .combined_bytes_without_pinned(cache.diagnostics().retained_bytes)
    }

    pub(super) fn host_memory_diagnostics(&self) -> Result<PacketHostMemoryDiagnostics, Error> {
        let _allocation = self.ledger.lock_allocations()?;
        let cache = self
            .state
            .lock()
            .map_err(|_| Error::OwnedPacketCachePoisoned)?;
        let cache_retained_bytes = cache.diagnostics().retained_bytes;
        let ledger = self.ledger.diagnostics()?;
        let current_combined_bytes = cache_retained_bytes
            .checked_add(ledger.active_bytes)
            .and_then(|bytes| bytes.checked_add(ledger.pinned_retained_bytes))
            .ok_or(Error::HostAllocationTooLarge {
                requested: usize::MAX,
                cap: j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
                what: "CUDA JPEG cache, in-flight owners, and pinned staging",
            })?;
        self.ledger.observe_combined(current_combined_bytes)?;
        Ok(PacketHostMemoryDiagnostics {
            cache_retained_bytes,
            active_owner_bytes: ledger.active_bytes,
            current_combined_bytes,
            peak_active_owner_bytes: ledger.peak_active_bytes,
            peak_combined_bytes: ledger.peak_combined_bytes.max(current_combined_bytes),
        })
    }

    pub(super) fn bind_context(
        &self,
        context: &j2k_cuda_runtime::CudaContext,
    ) -> Result<(), Error> {
        self.ledger.bind_context(context)
    }

    pub(super) fn ensure_context(
        &self,
        context: &j2k_cuda_runtime::CudaContext,
    ) -> Result<(), Error> {
        self.ledger.ensure_context(context)
    }

    pub(super) fn diagnostics(&self) -> Result<JpegPlanCacheDiagnostics, Error> {
        let cache = self
            .state
            .lock()
            .map_err(|_| Error::OwnedPacketCachePoisoned)?;
        let diagnostics = cache.diagnostics();
        self.record_coherent_entries(diagnostics);
        Ok(diagnostics)
    }

    pub(super) fn try_diagnostics(&self) -> Result<Option<JpegPlanCacheDiagnostics>, Error> {
        match self.state.try_lock() {
            Ok(cache) => {
                let diagnostics = cache.diagnostics();
                self.record_coherent_entries(diagnostics);
                Ok(Some(diagnostics))
            }
            Err(TryLockError::WouldBlock) => Ok(None),
            Err(TryLockError::Poisoned(_)) => Err(Error::OwnedPacketCachePoisoned),
        }
    }

    pub(super) fn last_coherent_entries(&self) -> usize {
        self.last_coherent_entries.load(Ordering::Acquire)
    }

    #[cfg(test)]
    pub(super) fn active_host_bytes(&self) -> Result<usize, Error> {
        self.ledger.active_bytes()
    }

    #[cfg(test)]
    pub(super) fn reserve_with_cap_for_test(
        &self,
        bytes: usize,
        cap: usize,
    ) -> Result<HostOwnerLease, Error> {
        let _allocation = self.ledger.lock_allocations()?;
        let cache = self.state.lock().unwrap();
        self.ledger.reserve(
            cache.diagnostics().retained_bytes,
            bytes,
            cap,
            "CUDA JPEG test host-owner reservation",
        )
    }

    fn record_coherent_entries(&self, diagnostics: JpegPlanCacheDiagnostics) {
        self.last_coherent_entries
            .store(diagnostics.entries, Ordering::Release);
    }
}

impl Default for OwnedPacketPlanCache {
    fn default() -> Self {
        Self {
            state: Mutex::new(JpegPlanCache::new()),
            last_coherent_entries: AtomicUsize::new(0),
            ledger: SharedHostLedger::new(),
        }
    }
}

fn cached_plan_error(error: JpegCachedPlanBuildError) -> Error {
    match error {
        JpegCachedPlanBuildError::Decode(error) => Error::Decode(error),
        JpegCachedPlanBuildError::FastPacket(error) => Error::FastPacket(error),
        JpegCachedPlanBuildError::Cache(error) => Error::JpegPlanCache(error),
    }
}
