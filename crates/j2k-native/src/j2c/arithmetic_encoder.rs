//! MQ arithmetic encoder for JPEG 2000 (ITU-T T.800 Annex C).
//!
//! This is the encoding counterpart of `arithmetic_decoder.rs`.
//! It uses the same QE probability table and context state machine.

use alloc::vec::Vec;

use super::mq::QE_TABLE;

/// MQ arithmetic encoder context (identical layout to decoder context).
///
/// Bits 0-6: state index (0-46) into the QE table.
/// Bit 7: MPS (Most Probable Symbol, 0 or 1).
#[derive(Copy, Clone, Debug, Default)]
pub(crate) struct ArithmeticEncoderContext(u8);

impl ArithmeticEncoderContext {
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
    pub(crate) fn reset_with_index(&mut self, index: u8) {
        self.0 = index;
    }
}

/// MQ arithmetic encoder (ITU-T T.800 Annex C).
///
/// Uses the direct buffer approach with proper 7-bit stuffing after 0xFF
/// to match the decoder's Annex G byte input convention.
pub(crate) struct ArithmeticEncoder {
    /// Output byte stream. Index 0 is a sentinel byte (0x00).
    data: Vec<u8>,
    /// A-register (interval size), 16-bit precision.
    a: u32,
    /// C-register (code value), 28-bit + carry at bit 27.
    c: u32,
    /// Bit shift counter (12 initially, then 7 after 0xFF, 8 otherwise).
    ct: u32,
}

impl ArithmeticEncoder {
    pub(crate) fn new() -> Self {
        Self::with_capacity(1)
    }

    pub(crate) fn with_capacity(capacity: usize) -> Self {
        let mut data = Vec::with_capacity(capacity.max(1));
        data.push(0x00);

        Self {
            data, // Sentinel byte at index 0
            a: 0x8000,
            c: 0,
            ct: 12,
        }
    }

    /// Encode a single symbol (0 or 1) with the given context (C.2.6 ENCODE).
    #[expect(
        clippy::inline_always,
        reason = "MQ state transitions are measured per-symbol hot paths"
    )]
    #[inline(always)]
    pub(crate) fn encode(&mut self, bit: u32, context: &mut ArithmeticEncoderContext) {
        let qe_entry = &QE_TABLE[context.index() as usize];
        self.a -= qe_entry.qe;

        if bit == context.mps() {
            // MPS path (CODEMPS, C.2.6)
            if self.a & 0x8000 != 0 {
                // No renormalization needed: fast path
                self.c += qe_entry.qe;
                return;
            }
            if self.a < qe_entry.qe {
                // Conditional exchange: MPS coded in lower sub-interval
                // C stays (don't add Qe), A takes the larger Qe value
                self.a = qe_entry.qe;
            } else {
                // Normal: MPS coded in upper sub-interval
                self.c += qe_entry.qe;
            }
            context.set_index(qe_entry.nmps);
        } else {
            // LPS path (CODELPS, C.2.6)
            if self.a < qe_entry.qe {
                // Conditional exchange: LPS coded in upper sub-interval
                self.c += qe_entry.qe;
            } else {
                // Normal: LPS coded in lower sub-interval, A = Qe
                self.a = qe_entry.qe;
            }
            if qe_entry.switch {
                context.xor_mps(1);
            }
            context.set_index(qe_entry.nlps);
        }

        self.renormalize();
    }

    /// Renormalize the encoder (C.2.7 RENORME).
    fn renormalize(&mut self) {
        loop {
            self.a <<= 1;
            self.c <<= 1;
            self.ct -= 1;
            if self.ct == 0 {
                self.byte_out();
            }
            if self.a & 0x8000 != 0 {
                break;
            }
        }
    }

    /// Output a byte with carry propagation and bit stuffing (C.2.8 BYTEOUT).
    ///
    /// After 0xFF, only 7 bits are extracted (bit stuffing) to prevent
    /// marker-like byte sequences in the output.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "MQ register masks bound every emitted chunk to one byte"
    )]
    fn byte_out(&mut self) {
        let last_byte = *self.data.last().unwrap();
        if last_byte == 0xFF {
            // 7-bit mode after 0xFF (bit stuffing)
            let b = (self.c >> 20) as u8;
            self.data.push(b);
            self.c &= 0x000F_FFFF;
            self.ct = 7;
        } else if self.c & 0x0800_0000 == 0 {
            // No carry: normal 8-bit output
            let b = (self.c >> 19) as u8;
            self.data.push(b);
            self.c &= 0x0007_FFFF;
            self.ct = 8;
        } else {
            // Carry occurred (bit 27 set): propagate into last byte
            let last = self.data.last_mut().unwrap();
            *last += 1;
            self.c &= 0x07FF_FFFF; // Clear carry bit
            if *last == 0xFF {
                // Carry made last byte 0xFF: switch to 7-bit mode
                let b = (self.c >> 20) as u8;
                self.data.push(b);
                self.c &= 0x000F_FFFF;
                self.ct = 7;
            } else {
                let b = (self.c >> 19) as u8;
                self.data.push(b);
                self.c &= 0x0007_FFFF;
                self.ct = 8;
            }
        }
    }

    /// SETBITS procedure (C.2.9).
    fn set_bits(&mut self) {
        let temp = self.c + self.a;
        self.c |= 0xFFFF;
        if self.c >= temp {
            self.c -= 0x8000;
        }
    }

    /// Flush the encoder state (C.2.9 FLUSH).
    pub(crate) fn flush(&mut self) {
        self.set_bits();
        self.c <<= self.ct;
        self.byte_out();
        self.c <<= self.ct;
        self.byte_out();
    }

    /// Return the encoded data (excluding sentinel), consuming the encoder.
    pub(crate) fn finish(mut self) -> Vec<u8> {
        self.flush();
        // Remove sentinel byte at index 0
        self.data.drain(..1);
        self.data
    }
}

#[cfg(test)]
mod tests {
    use super::{ArithmeticEncoder, ArithmeticEncoderContext};
    use crate::j2c::arithmetic_decoder::{ArithmeticDecoder, ArithmeticDecoderContext};
    use alloc::{vec, vec::Vec};

    #[test]
    #[cfg_attr(
        test,
        expect(clippy::similar_names, reason = "paired encoder/decoder state")
    )]
    fn test_encode_decode_round_trip() {
        let symbols: Vec<u32> = vec![0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1];
        let mut encoder = ArithmeticEncoder::new();
        let mut enc_ctx = ArithmeticEncoderContext::default();

        for &s in &symbols {
            encoder.encode(s, &mut enc_ctx);
        }
        let encoded = encoder.finish();

        // Decode and verify (new() already calls initialize())
        let mut decoder = ArithmeticDecoder::new(&encoded);
        let mut dec_ctx = ArithmeticDecoderContext::default();

        let mut decoded = Vec::new();
        for _ in 0..symbols.len() {
            decoded.push(decoder.decode(&mut dec_ctx));
        }

        assert_eq!(symbols, decoded);
    }

    #[test]
    #[cfg_attr(
        test,
        expect(clippy::similar_names, reason = "paired encoder/decoder state")
    )]
    fn test_encode_all_mps() {
        let mut encoder = ArithmeticEncoder::new();
        let mut ctx = ArithmeticEncoderContext::default();
        for _ in 0..100 {
            encoder.encode(0, &mut ctx);
        }
        let encoded = encoder.finish();

        let mut decoder = ArithmeticDecoder::new(&encoded);
        let mut dec_ctx = ArithmeticDecoderContext::default();
        for _ in 0..100 {
            assert_eq!(decoder.decode(&mut dec_ctx), 0);
        }
    }

    #[test]
    #[cfg_attr(
        test,
        expect(clippy::similar_names, reason = "paired encoder/decoder state")
    )]
    fn with_capacity_preserves_round_trip_encoding() {
        let mut encoder = ArithmeticEncoder::with_capacity(128);
        let mut enc_ctx = ArithmeticEncoderContext::default();
        let symbols = [0u32, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1];

        for &symbol in &symbols {
            encoder.encode(symbol, &mut enc_ctx);
        }
        let encoded = encoder.finish();

        let mut decoder = ArithmeticDecoder::new(&encoded);
        let mut dec_ctx = ArithmeticDecoderContext::default();
        for &symbol in &symbols {
            assert_eq!(decoder.decode(&mut dec_ctx), symbol);
        }
    }

    #[test]
    #[cfg_attr(
        test,
        expect(clippy::similar_names, reason = "paired encoder/decoder state")
    )]
    fn test_encode_all_lps() {
        let mut encoder = ArithmeticEncoder::new();
        let mut ctx = ArithmeticEncoderContext::default();
        for _ in 0..50 {
            encoder.encode(1, &mut ctx);
        }
        let encoded = encoder.finish();

        let mut decoder = ArithmeticDecoder::new(&encoded);
        let mut dec_ctx = ArithmeticDecoderContext::default();
        for _ in 0..50 {
            assert_eq!(decoder.decode(&mut dec_ctx), 1);
        }
    }

    #[test]
    #[cfg_attr(
        test,
        expect(clippy::similar_names, reason = "paired encoder/decoder state")
    )]
    fn test_multiple_contexts() {
        let symbols_a = [0u32, 1, 0, 0, 1, 1, 0, 1];
        let symbols_b = [1u32, 1, 0, 1, 0, 0, 1, 0];

        let mut encoder = ArithmeticEncoder::new();
        let mut ctx_a = ArithmeticEncoderContext::default();
        let mut ctx_b = ArithmeticEncoderContext::default();

        for i in 0..8 {
            encoder.encode(symbols_a[i], &mut ctx_a);
            encoder.encode(symbols_b[i], &mut ctx_b);
        }
        let encoded = encoder.finish();

        let mut decoder = ArithmeticDecoder::new(&encoded);
        let mut dec_ctx_a = ArithmeticDecoderContext::default();
        let mut dec_ctx_b = ArithmeticDecoderContext::default();

        for i in 0..8 {
            assert_eq!(decoder.decode(&mut dec_ctx_a), symbols_a[i]);
            assert_eq!(decoder.decode(&mut dec_ctx_b), symbols_b[i]);
        }
    }

    #[test]
    #[cfg_attr(
        test,
        expect(clippy::similar_names, reason = "paired encoder/decoder state")
    )]
    fn test_many_context_round_trip() {
        let mut state = 0x1234_5678u32;
        let mut symbols = Vec::new();
        let mut labels = Vec::new();
        let mut encoder = ArithmeticEncoder::new();
        let mut enc_contexts = [ArithmeticEncoderContext::default(); 19];
        enc_contexts[0].reset_with_index(4);
        enc_contexts[17].reset_with_index(3);
        enc_contexts[18].reset_with_index(46);

        for _ in 0..100_000 {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let label = (state % 19) as usize;
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let bit = (state >> 31) & 1;
            encoder.encode(bit, &mut enc_contexts[label]);
            labels.push(label);
            symbols.push(bit);
        }

        let encoded = encoder.finish();
        let mut decoder = ArithmeticDecoder::new(&encoded);
        let mut dec_contexts = [ArithmeticDecoderContext::default(); 19];
        dec_contexts[0].reset_with_index(4);
        dec_contexts[17].reset_with_index(3);
        dec_contexts[18].reset_with_index(46);

        for (index, (&label, &symbol)) in labels.iter().zip(symbols.iter()).enumerate() {
            let decoded = decoder.decode(&mut dec_contexts[label]);
            assert_eq!(decoded, symbol, "mismatch at symbol {index}");
        }
    }

    #[test]
    #[cfg_attr(
        test,
        expect(clippy::similar_names, reason = "paired encoder/decoder state")
    )]
    fn test_context_state_identical() {
        let mut enc_ctx = ArithmeticEncoderContext::default();
        let mut dec_ctx = ArithmeticDecoderContext::default();

        let bits = [0u32, 0, 1, 0, 1, 1, 0, 0];
        let mut encoder = ArithmeticEncoder::new();
        for &b in &bits {
            encoder.encode(b, &mut enc_ctx);
        }
        let encoded = encoder.finish();

        let mut decoder = ArithmeticDecoder::new(&encoded);
        for &b in &bits {
            let decoded = decoder.decode(&mut dec_ctx);
            assert_eq!(decoded, b);
        }

        // Both contexts should be in same state
        assert_eq!(enc_ctx.index(), dec_ctx.index());
        assert_eq!(enc_ctx.mps(), dec_ctx.mps());
    }
}
