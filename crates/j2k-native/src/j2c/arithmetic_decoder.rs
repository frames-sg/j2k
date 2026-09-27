//! The arithmetic decoder, described in Annex C.
//!
//! The arithmetic decoder keeps track of some state and continuously receives
//! context labels as input, each time yielding a new bit from the original data
//! as output.

use super::mq::PACKED_DECODER_STATES;

/// Tier-1 passes copy the decoder into a local, decode, and write it back so
/// the MQ registers stay in machine registers; `Clone` (not `Copy`) keeps
/// those copies explicit.
#[derive(Clone)]
pub(crate) struct ArithmeticDecoder<'a> {
    /// The underlying encoded data.
    data: &'a [u8],
    /// The C-register (see Table C.1).
    c: u32,
    /// The A-register (see Table C.1).
    a: u32,
    /// The pointer to the current byte.
    base_pointer: u32,
    /// The bit shift counter.
    shift_count: u32,
}

impl<'a> ArithmeticDecoder<'a> {
    pub(crate) fn new(data: &'a [u8]) -> Self {
        let mut decoder = ArithmeticDecoder {
            data,
            c: 0,
            a: 0,
            base_pointer: 0,
            shift_count: 0,
        };

        decoder.initialize();

        decoder
    }

    /// Read the next bit using the given context label.
    #[expect(
        clippy::inline_always,
        reason = "MQ state transitions are measured per-symbol hot paths"
    )]
    #[inline(always)]
    pub(crate) fn read_bit(&mut self, context: &mut ArithmeticDecoderContext) -> u32 {
        self.decode(context)
    }

    /// The INITDEC procedure from C.3.5.
    ///
    /// We use the version from Annex G in <https://www.itu.int/rec/T-REC-T.88-201808-I>.
    pub(crate) fn initialize(&mut self) {
        self.c = (u32::from(self.current_byte()) ^ 0xff) << 16;
        self.read_byte();

        self.c <<= 7;
        self.shift_count -= 7;
        self.a = 0x8000;
    }

    /// The BYTEIN procedure from C.3.4.
    ///
    /// We use the version from Annex G from <https://www.itu.int/rec/T-REC-T.88-201808-I>.
    #[expect(
        clippy::inline_always,
        reason = "MQ state transitions are measured per-symbol hot paths"
    )]
    #[inline(always)]
    fn read_byte(&mut self) {
        if self.current_byte() == 0xff {
            let b1 = self.next_byte();

            if b1 > 0x8f {
                self.shift_count = 8;
            } else {
                self.base_pointer += 1;
                self.c = self
                    .c
                    .wrapping_add(0xfe00)
                    .wrapping_sub(u32::from(self.current_byte()) << 9);
                self.shift_count = 7;
            }
        } else {
            self.base_pointer += 1;
            self.c = self
                .c
                .wrapping_add(0xff00)
                .wrapping_sub(u32::from(self.current_byte()) << 8);
            self.shift_count = 8;
        }
    }

    /// The RENORMD procedure from C.3.3.
    #[expect(
        clippy::inline_always,
        reason = "MQ state transitions are measured per-symbol hot paths"
    )]
    #[inline(always)]
    fn renormalize(&mut self) {
        // Original code:
        // loop {
        //     if self.shift_count == 0 {
        //         self.read_byte();
        //     }
        //
        //     self.a <<= 1;
        //     self.c <<= 1;
        //     self.shift_count -= 1;
        //
        //     if self.a & 0x8000 != 0 {
        //         break;
        //     }
        // }

        // Optimization: Batch shifts.
        while self.a & 0x8000 == 0 {
            if self.shift_count == 0 {
                self.read_byte();
            }

            let shifts_needed = self.a.leading_zeros() - 16;
            let batch = shifts_needed.min(self.shift_count);
            self.a <<= batch;
            self.c <<= batch;
            self.shift_count -= batch;
        }
    }

    /// The DECODE procedure from C.3.2.
    ///
    /// We use the version from Annex G from <https://www.itu.int/rec/T-REC-T.88-201808-I>.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::inline_always,
        clippy::similar_names,
        reason = "packed MQ fields are explicitly bounded and MPS/LPS are paired domain terms on a measured per-symbol hot path"
    )]
    #[inline(always)]
    pub(crate) fn decode(&mut self, context: &mut ArithmeticDecoderContext) -> u32 {
        let state = PACKED_DECODER_STATES[context.0 as usize];
        let qe = state as u32 & 0xffff;
        let mps = u32::from((state >> 16) & 1 != 0);

        let a = self.a - qe;

        // Branchless: whether the symbol is the MPS and whether A needs
        // renormalizing are data-dependent and poorly predicted, and the
        // caller already branches on the decoded bit. The formulas below
        // reduce to "return the MPS, change nothing" in the fast case.
        //
        // In the Annex C.3.2 procedures, the only difference between
        // MPS_EXCHANGE and LPS_EXCHANGE is that LPS flips the role of cond:
        //   exchange_mps: d = mps ^ cond,       flip when cond,      index = cond*nlps + inv*nmps
        //   exchange_lps: d = mps ^ inv_cond,   flip when inv_cond,  index = cond*nmps + inv*nlps
        // so both are handled by XOR-ing cond with is_lps.
        let is_lps = u32::from((self.c >> 16) >= a);
        // A probability-state transition happens exactly when A renormalizes:
        // on every LPS, and on an MPS that leaves A below 0x8000.
        let transition = is_lps | u32::from(a < 0x8000);

        // LPS: C -= A << 16 and A = Qe; MPS: A stays A - Qe.
        let lps_mask = is_lps.wrapping_neg();
        self.c -= (a << 16) & lps_mask;
        self.a = (a & !lps_mask) | (qe & lps_mask);

        // Same condition as in exchange_mps / exchange_lps. Without a
        // transition, A >= 0x8000 > Qe, so this is zero and d is the MPS.
        let cond = u32::from(a < qe);
        let pick_nlps = cond ^ is_lps;
        let d = mps ^ pick_nlps;

        let pick_mask = u8::from(pick_nlps != 0).wrapping_neg();
        let next_mps = (state >> 24) as u8;
        let next_lps = (state >> 32) as u8;
        let next = next_mps ^ ((next_mps ^ next_lps) & pick_mask);
        let keep_mask = u8::from(transition == 0).wrapping_neg();
        context.0 = next ^ ((next ^ context.0) & keep_mask);

        // Shift A back to at least 0x8000 in one step when the buffered bits
        // suffice (zero shift in the fast case); otherwise take the byte-in
        // loop.
        let shifts_needed = self.a.leading_zeros() - 16;
        if shifts_needed <= self.shift_count {
            self.a <<= shifts_needed;
            self.c <<= shifts_needed;
            self.shift_count -= shifts_needed;
        } else {
            self.renormalize();
        }

        d
    }

    #[expect(
        clippy::inline_always,
        reason = "MQ state transitions are measured per-symbol hot paths"
    )]
    #[inline(always)]
    fn current_byte(&self) -> u8 {
        self.data
            .get(self.base_pointer as usize)
            .copied()
            // "The number of bytes corresponding to the coding passes is
            // specified in the packet header. Often at that point there are
            // more symbols to be decoded. Therefore, the decoder shall extend
            // the input bit stream to the arithmetic coder with 0xFF bytes,
            // as necessary, until all symbols have been decoded."
            .unwrap_or(0xFF)
    }

    #[expect(
        clippy::inline_always,
        reason = "MQ state transitions are measured per-symbol hot paths"
    )]
    #[inline(always)]
    fn next_byte(&self) -> u8 {
        self.data
            .get((self.base_pointer + 1) as usize)
            .copied()
            .unwrap_or(0xFF)
    }
}

// Previously, we stored the context as 2 u32's, but doing it with a bit-packed
// u8 seems to be slightly better (though it doesn't make that huge of a
// difference).
/// Bits 0-6 = index (0-46).
/// Bit 7 = mps (0 or 1).
#[derive(Copy, Clone, Debug, Default)]
pub(crate) struct ArithmeticDecoderContext(u8);

impl ArithmeticDecoderContext {
    pub(crate) const fn empty() -> Self {
        Self(0)
    }

    #[expect(
        clippy::inline_always,
        reason = "MQ state transitions are measured per-symbol hot paths"
    )]
    #[cfg(test)]
    #[inline(always)]
    pub(crate) fn index(self) -> u32 {
        u32::from(self.0 & 0x7F)
    }

    #[expect(
        clippy::inline_always,
        reason = "MQ state transitions are measured per-symbol hot paths"
    )]
    #[cfg(test)]
    #[inline(always)]
    pub(crate) fn mps(self) -> u32 {
        u32::from(self.0 >> 7)
    }

    #[expect(
        clippy::inline_always,
        reason = "MQ state transitions are measured per-symbol hot paths"
    )]
    #[inline(always)]
    pub(crate) fn reset(&mut self) {
        self.0 = 0;
    }

    #[expect(
        clippy::inline_always,
        reason = "MQ state transitions are measured per-symbol hot paths"
    )]
    #[inline(always)]
    pub(crate) fn reset_with_index(&mut self, index: u8) {
        self.0 = index;
    }
}
