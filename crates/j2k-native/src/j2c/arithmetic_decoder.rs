//! The arithmetic decoder, described in Annex C.
//!
//! The arithmetic decoder keeps track of some state and continuously receives
//! context labels as input, each time yielding a new bit from the original data
//! as output.

use super::mq::QE_TABLE;

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
        clippy::inline_always,
        reason = "MQ state transitions are measured per-symbol hot paths"
    )]
    #[inline(always)]
    pub(crate) fn decode(&mut self, context: &mut ArithmeticDecoderContext) -> u32 {
        let qe_entry = &QE_TABLE[context.index() as usize];

        self.a -= qe_entry.qe;

        // This is a faster version that reduces branching, which has shown
        // itself to be the main limiting factor for better performance.
        // We short-circuit the case where just the most probably symbol is
        // returned, and otherwise use a code path that works for both,
        // MPS_EXCHANGE and LPS_EXCHANGE.

        if (self.c >> 16) < self.a && self.a & 0x8000 != 0 {
            return context.mps();
        }

        // Unified branchless MPS_EXCHANGE / LPS_EXCHANGE. In the Annex C.3.2
        // procedures, the only difference is that LPS flips the role of cond:
        //   exchange_mps: d = mps ^ cond,       flip when cond,      index = cond*nlps + inv*nmps
        //   exchange_lps: d = mps ^ inv_cond,   flip when inv_cond,  index = cond*nmps + inv*nlps
        //
        // This is equivalent to XOR-ing cond with is_lps, so we can handle
        // both paths with a single branchless computation.
        //
        // As can be seen above, renormalization is always performed.
        let is_lps = u32::from((self.c >> 16) >= self.a);

        // LPS: C -= A << 16 (no-op when MPS).
        let lps_mask = is_lps.wrapping_neg(); // 0xFFFF_FFFF if LPS, 0 if MPS
        self.c -= (self.a << 16) & lps_mask;

        // Same condition as in exchange_mps / exchange_lps.
        let cond = u32::from(self.a < qe_entry.qe);

        // LPS: a = qe (no-op when MPS, a stays as a - qe).
        self.a = (self.a & !lps_mask) | (qe_entry.qe & lps_mask);

        // exchange_mps: d = mps ^ cond       →  cond ^ 0
        // exchange_lps: d = mps ^ inv_cond   →  cond ^ 1
        // unified:      d = mps ^ (cond ^ is_lps)
        let d = context.mps() ^ cond ^ is_lps;

        // exchange_mps: flip mps when cond & switch       →  (cond ^ 0) & switch
        // exchange_lps: flip mps when inv_cond & switch   →  (cond ^ 1) & switch
        // unified:      flip mps when (cond ^ is_lps) & switch
        context.xor_mps((cond ^ is_lps) & u32::from(qe_entry.switch));

        // exchange_mps: index = cond * nlps + inv_cond * nmps
        // exchange_lps: index = cond * nmps + inv_cond * nlps  (swapped)
        // unified: the result is always exactly nmps or nlps —
        //          pick nlps when (cond ^ is_lps) == 1, nmps otherwise.
        let pick_nlps = u8::from(cond ^ is_lps != 0).wrapping_neg(); // 0xFF or 0x00
        context.set_index(qe_entry.nmps ^ ((qe_entry.nmps ^ qe_entry.nlps) & pick_nlps));

        self.renormalize();

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
    #[cfg_attr(not(test), allow(dead_code))]
    #[expect(
        clippy::inline_always,
        reason = "MQ state transitions are measured per-symbol hot paths"
    )]
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
    fn set_index(&mut self, index: u8) {
        self.0 = (self.0 & 0x80) | index;
    }

    #[expect(
        clippy::inline_always,
        reason = "MQ state transitions are measured per-symbol hot paths"
    )]
    #[inline(always)]
    fn xor_mps(&mut self, val: u32) {
        self.0 ^= u8::from(val & 1 != 0) << 7;
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
