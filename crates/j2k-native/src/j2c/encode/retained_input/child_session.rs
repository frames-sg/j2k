// SPDX-License-Identifier: MIT OR Apache-2.0

//! Nested session derivation for parent-owned encode phases.

use super::{NativeEncodeRetainedInput, NativeEncodeSession};
use crate::EncodeResult;

impl NativeEncodeSession<'_> {
    /// Derive a nested encode session whose baseline includes owners retained
    /// by the current phase.
    ///
    /// The returned session borrows this session, preserving the lifetime of
    /// the original caller-owned graph while a child tile or other nested
    /// encode runs. `phase_owners` must contain borrows of every owner counted
    /// by `additional_bytes`; the original retained input is included exactly
    /// once.
    pub(crate) fn checked_child_session<'session, O: ?Sized>(
        &'session self,
        phase_owners: &'session O,
        additional_bytes: usize,
        what: &'static str,
    ) -> EncodeResult<NativeEncodeSession<'session>> {
        let retained_bytes = self.checked_phase(additional_bytes, what)?.retained_bytes();
        Ok(NativeEncodeSession {
            retained_input: NativeEncodeRetainedInput::from_owner_bytes(
                phase_owners,
                retained_bytes,
            ),
            cap: self.cap,
        })
    }
}
