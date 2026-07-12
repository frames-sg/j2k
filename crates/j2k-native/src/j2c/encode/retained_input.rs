// SPDX-License-Identifier: MIT OR Apache-2.0

//! Lifetime-bound accounting for caller-owned native encode inputs.

use alloc::vec::Vec;
use core::marker::PhantomData;

use super::allocation::{checked_add_bytes, checked_element_bytes, EncodeAllocationLedger};
use crate::{EncodeError, EncodeResult, DEFAULT_MAX_CODEC_BYTES};

mod child_session;

/// Opaque accounting token for caller-owned allocations retained during encode.
///
/// The token borrows every owner used to build it, so those owners cannot be
/// mutated or reallocated while the encode session is live. It is deliberately
/// move-only: copying a byte count would sever that ownership contract.
#[derive(Debug)]
pub(crate) struct NativeEncodeRetainedInput<'a> {
    bytes: usize,
    owners: PhantomData<&'a ()>,
}

impl NativeEncodeRetainedInput<'static> {
    /// Construct the zero baseline used by ordinary public encode entrypoints.
    #[must_use]
    pub(crate) const fn none() -> Self {
        Self {
            bytes: 0,
            owners: PhantomData,
        }
    }
}

impl<'a> NativeEncodeRetainedInput<'a> {
    pub(crate) fn from_owner_bytes<O: ?Sized>(_owner: &'a O, bytes: usize) -> Self {
        Self {
            bytes,
            owners: PhantomData,
        }
    }

    pub(crate) const fn bytes(&self) -> usize {
        self.bytes
    }
}

/// One native encode invocation and the caller-owned baseline it retains.
pub(crate) struct NativeEncodeSession<'a> {
    retained_input: NativeEncodeRetainedInput<'a>,
    cap: usize,
}

/// Checked bytes retained by one encode phase before accepting backend output.
pub(crate) struct NativeEncodePhase<'session, 'input> {
    session: &'session NativeEncodeSession<'input>,
    phase_bytes: usize,
    retained_bytes: usize,
}

pub(crate) type NativeEncodePipelineResult<T> = core::result::Result<T, NativeEncodePipelineError>;

#[derive(Debug)]
pub(crate) enum NativeEncodePipelineError {
    InvalidInput(&'static str),
    Unsupported(&'static str),
    ArithmeticOverflow(&'static str),
    InternalInvariant(&'static str),
    Typed(EncodeError),
}

impl NativeEncodePipelineError {
    pub(crate) fn into_encode_error(self) -> EncodeError {
        match self {
            Self::InvalidInput(what) => EncodeError::InvalidInput { what },
            Self::Unsupported(what) => EncodeError::Unsupported { what },
            Self::ArithmeticOverflow(what) => EncodeError::ArithmeticOverflow { what },
            Self::InternalInvariant(what) => EncodeError::InternalInvariant { what },
            Self::Typed(error) => error,
        }
    }

    pub(crate) const fn invalid_input(what: &'static str) -> Self {
        Self::InvalidInput(what)
    }

    pub(crate) const fn unsupported(what: &'static str) -> Self {
        Self::Unsupported(what)
    }

    pub(crate) const fn arithmetic_overflow(what: &'static str) -> Self {
        Self::ArithmeticOverflow(what)
    }

    pub(crate) const fn internal_invariant(what: &'static str) -> Self {
        Self::InternalInvariant(what)
    }
}

impl From<EncodeError> for NativeEncodePipelineError {
    fn from(error: EncodeError) -> Self {
        Self::Typed(error)
    }
}

impl<'a> NativeEncodeSession<'a> {
    pub(crate) fn try_new(retained_input: NativeEncodeRetainedInput<'a>) -> EncodeResult<Self> {
        Self::try_with_cap(retained_input, DEFAULT_MAX_CODEC_BYTES)
    }

    /// Construct a session whose cap can only be lower than the process-wide
    /// codec ceiling. Cross-codec pipelines use this to reserve part of the
    /// shared host budget for owners that remain live outside native encode.
    pub(crate) fn try_with_lowered_cap(
        retained_input: NativeEncodeRetainedInput<'a>,
        requested_cap: usize,
    ) -> EncodeResult<Self> {
        Self::try_with_cap(retained_input, requested_cap.min(DEFAULT_MAX_CODEC_BYTES))
    }

    pub(crate) fn try_with_cap(
        retained_input: NativeEncodeRetainedInput<'a>,
        cap: usize,
    ) -> EncodeResult<Self> {
        // Validate before transform or Tier-1 work starts. Packet production
        // later creates a phase ledger seeded from this same baseline because
        // its final handoff seals that ledger.
        EncodeAllocationLedger::with_cap(retained_input.bytes(), cap)?;
        Ok(Self {
            retained_input,
            cap,
        })
    }

    pub(crate) fn checked_phase(
        &self,
        phase_bytes: usize,
        what: &'static str,
    ) -> EncodeResult<NativeEncodePhase<'_, 'a>> {
        let retained_bytes = checked_add_bytes(self.retained_input.bytes(), phase_bytes, what)?;
        EncodeAllocationLedger::with_phase_cap(retained_bytes, self.cap, what)?;
        Ok(NativeEncodePhase {
            session: self,
            phase_bytes,
            retained_bytes,
        })
    }

    pub(crate) fn checked_phase_retained_bytes(
        &self,
        phase_bytes: usize,
        what: &'static str,
    ) -> EncodeResult<usize> {
        Ok(self.checked_phase(phase_bytes, what)?.retained_bytes())
    }
}

impl NativeEncodePhase<'_, '_> {
    pub(crate) const fn retained_bytes(&self) -> usize {
        self.retained_bytes
    }

    pub(crate) fn reconcile_accelerator_output_bytes(
        &self,
        output_bytes: usize,
        what: &'static str,
    ) -> EncodeResult<()> {
        let phase_bytes = checked_add_bytes(self.phase_bytes, output_bytes, what)?;
        self.session.checked_phase(phase_bytes, what)?;
        Ok(())
    }

    pub(crate) fn reconcile_accelerator_vec<T>(
        &self,
        output: &Vec<T>,
        what: &'static str,
    ) -> EncodeResult<()> {
        self.reconcile_accelerator_output_bytes(
            checked_element_bytes::<T>(output.capacity(), what)?,
            what,
        )
    }
}

#[cfg(test)]
mod tests;
