// SPDX-License-Identifier: MIT OR Apache-2.0

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

#[test]
fn checked_payload_limit_is_sticky_and_never_underflows_renormalization() {
    let mut encoder =
        ArithmeticEncoder::try_with_byte_limit(0).expect("sentinel allocation succeeds");
    let mut context = ArithmeticEncoderContext::default();
    for index in 0..10_000 {
        encoder.encode(index & 1, &mut context);
    }
    assert_eq!(
        encoder
            .finish_checked()
            .expect_err("zero-byte payload plan is exceeded"),
        crate::EncodeError::InternalInvariant {
            what: "MQ encoder exceeded its checked payload plan",
        }
    );
}
