// SPDX-License-Identifier: MIT OR Apache-2.0

use metal::Buffer;

use super::{
    invariant, poisoned, MetalBufferPoolDiagnostics, MetalBufferPools, PoolLimits, PooledBuffer,
};

impl MetalBufferPools {
    pub(super) fn with_limits_for_test(private: PoolLimits, shared: PoolLimits) -> Self {
        Self::with_limits(private, shared)
    }

    pub(super) fn private_diagnostics(&self) -> Result<MetalBufferPoolDiagnostics, crate::Error> {
        Self::pool_diagnostics(&self.private, "private")
    }

    pub(super) fn shared_diagnostics(&self) -> Result<MetalBufferPoolDiagnostics, crate::Error> {
        Self::pool_diagnostics(&self.shared, "shared")
    }

    pub(super) fn fail_next_private_metadata_reserve_for_test(&self) {
        self.private
            .lock()
            .expect("private pool test lock")
            .fail_next_metadata_reserve();
    }

    pub(super) fn recycle_private_checked(
        &self,
        bytes: usize,
        buffer: Buffer,
    ) -> Result<(), crate::Error> {
        let owner = match PooledBuffer::new_checked(bytes.max(1), buffer) {
            Ok(owner) => owner,
            Err(reason) => {
                self.private
                    .lock()
                    .map_err(|_| poisoned("private"))?
                    .record_size_mismatch()
                    .map_err(|counter_error| invariant("private", counter_error))?;
                return Err(invariant("private", reason));
            }
        };
        self.recycle_private(owner)
    }

    pub(super) fn recycle_shared_checked(
        &self,
        bytes: usize,
        buffer: Buffer,
    ) -> Result<(), crate::Error> {
        let owner = PooledBuffer::new_checked(bytes.max(1), buffer)
            .map_err(|reason| invariant("shared", reason))?;
        self.recycle_shared(owner)
    }
}
