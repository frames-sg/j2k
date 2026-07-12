// SPDX-License-Identifier: MIT OR Apache-2.0

//! Terminal marker observation and real-versus-synthetic bit accounting.

use super::{BitReader, ACC_BITS};

impl BitReader<'_> {
    pub(crate) fn observed_marker(&self) -> Option<u8> {
        self.marker
    }

    pub(crate) fn observe_marker(&mut self) {
        if self.marker.is_none() {
            let _ = self.refill_one_byte();
        }
    }

    pub(crate) fn unread_real_bits(&self) -> u8 {
        self.bits - self.synthetic_bits
    }

    pub(crate) fn unread_real_bits_are_ones(&self) -> bool {
        let bits = self.unread_real_bits();
        if bits == 0 {
            return true;
        }
        let value = self.acc >> (ACC_BITS - bits);
        let expected = if bits == ACC_BITS {
            u64::MAX
        } else {
            (1u64 << bits) - 1
        };
        value == expected
    }
}
