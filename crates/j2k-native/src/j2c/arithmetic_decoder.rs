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
    /// The C-register (see Table C.1) in bits 32-63: its upper half, which
    /// DECODE compares with A, in bits 48-63, followed by `shift_count` bits
    /// of code bytes read ahead.
    c: u64,
    /// The A-register (see Table C.1).
    a: u32,
    /// The pointer to the current byte.
    base_pointer: u32,
    /// The number of read-ahead bits below the upper half of C.
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
        self.c = u64::from(self.current_byte() ^ 0xff) << 48;
        self.read_bytes();

        self.c <<= 7;
        self.shift_count -= 7;
        self.a = 0x8000;
    }

    /// The BYTEIN procedure from C.3.4, repeated while another byte fits
    /// below the upper half of C.
    ///
    /// We use the version from Annex G from <https://www.itu.int/rec/T-REC-T.88-201808-I>.
    /// A byte after 0xFF may carry into the bits above it, so it is only read
    /// once no read-ahead bits remain, as in the procedure; any other byte
    /// lands on zero bits, so reading it early changes nothing DECODE sees.
    #[expect(
        clippy::inline_always,
        reason = "MQ state transitions are measured per-symbol hot paths"
    )]
    #[inline(always)]
    fn read_bytes(&mut self) {
        while self.shift_count <= 40 {
            if self.current_byte() == 0xff {
                let b1 = self.next_byte();

                if b1 > 0x8f {
                    self.shift_count += 8;
                } else if self.shift_count == 0 {
                    self.base_pointer += 1;
                    self.c = self
                        .c
                        .wrapping_add(0x7f << 41)
                        .wrapping_sub(u64::from(b1) << 41);
                    self.shift_count = 7;
                } else {
                    break;
                }
            } else {
                self.base_pointer += 1;
                self.c += u64::from(self.current_byte() ^ 0xff) << (40 - self.shift_count);
                self.shift_count += 8;
            }
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
        //         BYTEIN
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

        // Optimization: Batch shifts over several bytes read ahead.
        while self.a & 0x8000 == 0 {
            self.read_bytes();

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
        reason = "packed MQ fields are explicitly bounded on a measured per-symbol hot path"
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
        let is_lps = u32::from((self.c >> 48) as u32 >= a);
        // A probability-state transition happens exactly when A renormalizes:
        // on every LPS, and on an MPS that leaves A below 0x8000.
        let transition = is_lps | u32::from(a < 0x8000);

        // LPS: C -= A << 16 and A = Qe; MPS: A stays A - Qe.
        let lps_mask = is_lps.wrapping_neg();
        self.c -= (u64::from(a) << 48) & u64::from(is_lps).wrapping_neg();
        self.a = (a & !lps_mask) | (qe & lps_mask);

        // Same condition as in exchange_mps / exchange_lps. Without a
        // transition, A >= 0x8000 > Qe, so this is zero and d is the MPS.
        let cond = u32::from(a < qe);
        let pick_nlps = cond ^ is_lps;
        let d = mps ^ pick_nlps;

        // The next MPS state is byte 3 of `state` and the next LPS state byte 4.
        let next = (state >> (24 + 8 * pick_nlps)) as u8;
        context.0 = if transition == 0 { context.0 } else { next };

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

    /// How many decisions in `context`, from the current registers, take the
    /// MPS path of C.3.2 without renormalizing: each leaves C and the
    /// context's state as they are and lowers A by Qe, which holds while
    /// A - k * Qe stays at least 0x8000 and above the upper half of C.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::inline_always,
        reason = "the upper half of C is 16 bits; checked on a measured per-column hot path"
    )]
    #[inline(always)]
    pub(crate) fn mps_run(&self, context: ArithmeticDecoderContext) -> u32 {
        let qe = PACKED_DECODER_STATES[context.0 as usize] as u32 & 0xffff;
        let floor = ((self.c >> 48) as u32 + 1).max(0x8000);
        self.a.saturating_sub(floor) / qe
    }

    /// Decode `count` decisions in `context` that [`Self::mps_run`] covers:
    /// all of them yield the MPS and only A changes.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::inline_always,
        reason = "Qe is the low 16 bits of the packed state; per-column hot path"
    )]
    #[inline(always)]
    pub(crate) fn skip_mps_run(&mut self, context: ArithmeticDecoderContext, count: u32) {
        debug_assert!(count <= self.mps_run(context));
        let qe = PACKED_DECODER_STATES[context.0 as usize] as u32 & 0xffff;
        self.a -= count * qe;
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

#[cfg(test)]
mod tests {
    use super::{ArithmeticDecoder, ArithmeticDecoderContext};
    use crate::j2c::arithmetic_encoder::{ArithmeticEncoder, ArithmeticEncoderContext};
    use alloc::vec::Vec;

    /// Symbols of one context that are 1 about once in `one_in`.
    fn sparse_symbols(count: usize, one_in: u32, seed: u32) -> Vec<u32> {
        let mut state = seed;
        (0..count)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                u32::from((state >> 8).is_multiple_of(one_in))
            })
            .collect()
    }

    fn registers(decoder: &ArithmeticDecoder<'_>) -> (u64, u32, u32, u32) {
        (
            decoder.c,
            decoder.a,
            decoder.base_pointer,
            decoder.shift_count,
        )
    }

    #[test]
    fn skipping_mps_runs_matches_decoding_each_decision() {
        for (one_in, start_index, seed) in [
            (40, 3, 1),
            (300, 3, 2),
            (7, 0, 3),
            (2000, 0, 4),
            (3, 3, 5),
            (5, 46, 6),
        ] {
            let symbols = sparse_symbols(20_000, one_in, seed);
            let mut encoder = ArithmeticEncoder::new();
            let mut encoder_context = ArithmeticEncoderContext::default();
            encoder_context.reset_with_index(start_index);
            for &symbol in &symbols {
                encoder.encode(symbol, &mut encoder_context);
            }
            let encoded = encoder.finish();

            let mut each = ArithmeticDecoder::new(&encoded);
            let mut each_context = ArithmeticDecoderContext::empty();
            each_context.reset_with_index(start_index);
            let mut runs = ArithmeticDecoder::new(&encoded);
            let mut runs_context = each_context;
            let mut skipped = 0usize;
            let mut position = 0;
            while position < symbols.len() {
                let run = if runs_context.mps() == 0 {
                    runs.mps_run(runs_context)
                } else {
                    0
                };
                if run == 0 {
                    assert_eq!(runs.read_bit(&mut runs_context), symbols[position]);
                    assert_eq!(each.read_bit(&mut each_context), symbols[position]);
                    position += 1;
                } else {
                    let count = run.min(u32::try_from(symbols.len() - position).expect("fits"));
                    for _ in 0..count {
                        let before = each_context.0;
                        assert_eq!(each.read_bit(&mut each_context), 0);
                        assert_eq!(each_context.0, before, "a skipped decision changed state");
                        assert_eq!(symbols[position], 0);
                        position += 1;
                    }
                    runs.skip_mps_run(runs_context, count);
                    skipped += count as usize;
                }
                assert_eq!(registers(&runs), registers(&each));
                assert_eq!(runs_context.0, each_context.0);
            }
            // State 46 (the uniform context) never adapts, so its runs are short.
            if one_in >= 40 {
                assert!(skipped > 0, "no run skipped for one_in {one_in}");
            }
        }
    }
}
