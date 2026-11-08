// SPDX-License-Identifier: MIT OR Apache-2.0

//! Per-operation native policy, never retained as an immutable source fact.

use alloc::sync::Arc;
use core::sync::atomic::{AtomicBool, Ordering};

use crate::{DecodeError, DecodingError, Result, DEFAULT_MAX_DECODE_BYTES};

/// Native working-memory and cooperative cancellation policy for one operation.
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct DecodeOperationControl {
    limit: usize,
    cancellation: Option<Arc<AtomicBool>>,
    retained_bytes: usize,
}

impl Default for DecodeOperationControl {
    fn default() -> Self {
        Self::new(None, None)
    }
}

impl DecodeOperationControl {
    /// Construct a policy without raising the engine's built-in allocation cap.
    #[must_use]
    pub fn new(limit: Option<usize>, cancellation: Option<Arc<AtomicBool>>) -> Self {
        Self {
            limit: limit
                .unwrap_or(DEFAULT_MAX_DECODE_BYTES)
                .min(DEFAULT_MAX_DECODE_BYTES),
            cancellation,
            retained_bytes: 0,
        }
    }

    /// Stop before starting another bounded unit of work.
    ///
    /// # Errors
    /// Returns cancellation when the operation's flag has been set.
    pub fn check(&self) -> Result<()> {
        if self
            .cancellation
            .as_ref()
            .is_some_and(|flag| flag.load(Ordering::Relaxed))
        {
            return Err(DecodingError::Cancelled.into());
        }
        Ok(())
    }

    /// Reservation that lowers the existing aggregate allocation allowance.
    #[must_use]
    pub fn working_baseline(&self) -> usize {
        DEFAULT_MAX_DECODE_BYTES - self.limit
    }

    /// Compose actual retained owners with this operation's allowance.
    ///
    /// # Errors
    /// Returns cancellation or an allocation error when the byte count overflows.
    pub fn baseline(&self, retained: usize) -> Result<usize> {
        self.check()?;
        retained
            .checked_add(self.retained_bytes)
            .and_then(|bytes| bytes.checked_add(self.working_baseline()))
            .ok_or(DecodeError::AllocationTooLarge {
                what: "decode operation retained owners",
                requested: usize::MAX,
                cap: DEFAULT_MAX_DECODE_BYTES,
            })
    }

    /// Reserve an actual codec-owned buffer while this operation runs.
    ///
    /// This reservation belongs to the scoped operation, never to cached image metadata.
    /// # Errors
    /// Returns an allocation error if the retained owners cannot fit the allowance.
    pub fn beside_retained_bytes(mut self, bytes: usize) -> Result<Self> {
        self.retained_bytes =
            self.retained_bytes
                .checked_add(bytes)
                .ok_or(DecodeError::AllocationTooLarge {
                    what: "decode operation retained owners",
                    requested: usize::MAX,
                    cap: self.limit,
                })?;
        let baseline = self.baseline(0)?;
        if baseline > DEFAULT_MAX_DECODE_BYTES {
            return Err(DecodeError::AllocationTooLarge {
                what: "decode operation retained owners",
                requested: baseline,
                cap: DEFAULT_MAX_DECODE_BYTES,
            });
        }
        Ok(self)
    }

    pub(crate) fn has_cancellation(&self) -> bool {
        self.cancellation.is_some()
    }

    pub(crate) fn cancellation(&self) -> Option<Arc<AtomicBool>> {
        self.cancellation.clone()
    }
}
