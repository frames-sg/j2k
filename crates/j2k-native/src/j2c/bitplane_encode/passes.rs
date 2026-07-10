// SPDX-License-Identifier: MIT OR Apache-2.0

// Classic EBCOT cleanup, significance-propagation, and refinement kernels.

use alloc::vec;
use alloc::vec::Vec;

use super::super::arithmetic_encoder::{ArithmeticEncoder, ArithmeticEncoderContext};
use super::super::build::SubBandType;
use super::super::codestream::CodeBlockStyle;
use crate::writer::BitWriter;

/// Coefficient state flags.
pub(super) const SIGNIFICANT: u8 = 1 << 7;
const MAGNITUDE_REFINED: u8 = 1 << 6;
pub(super) const CODED_IN_CURRENT_PASS: u8 = 1 << 5;
pub(super) const NEGATIVE: u8 = 1 << 4;

/// Context labels for zero coding (Table D.1).
/// Index into 256-entry lookup tables by neighbor significance pattern.
#[rustfmt::skip]
const ZERO_CTX_LL_LH: [u8; 256] = [
    0, 3, 1, 3, 5, 7, 6, 7, 1, 3, 2, 3, 6, 7, 6, 7, 5, 7, 6, 7, 8, 8, 8, 8, 6, 7, 6, 7, 8, 8, 8, 8,
    1, 3, 2, 3, 6, 7, 6, 7, 2, 3, 2, 3, 6, 7, 6, 7, 6, 7, 6, 7, 8, 8, 8, 8, 6, 7, 6, 7, 8, 8, 8, 8,
    3, 4, 3, 4, 7, 7, 7, 7, 3, 4, 3, 4, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 7, 7, 7, 7, 8, 8, 8, 8,
    3, 4, 3, 4, 7, 7, 7, 7, 3, 4, 3, 4, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 7, 7, 7, 7, 8, 8, 8, 8,
    1, 3, 2, 3, 6, 7, 6, 7, 2, 3, 2, 3, 6, 7, 6, 7, 6, 7, 6, 7, 8, 8, 8, 8, 6, 7, 6, 7, 8, 8, 8, 8,
    2, 3, 2, 3, 6, 7, 6, 7, 2, 3, 2, 3, 6, 7, 6, 7, 6, 7, 6, 7, 8, 8, 8, 8, 6, 7, 6, 7, 8, 8, 8, 8,
    3, 4, 3, 4, 7, 7, 7, 7, 3, 4, 3, 4, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 7, 7, 7, 7, 8, 8, 8, 8,
    3, 4, 3, 4, 7, 7, 7, 7, 3, 4, 3, 4, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 7, 7, 7, 7, 8, 8, 8, 8,
];

#[rustfmt::skip]
const ZERO_CTX_HL: [u8; 256] = [
    0, 5, 1, 6, 3, 7, 3, 7, 1, 6, 2, 6, 3, 7, 3, 7, 3, 7, 3, 7, 4, 7, 4, 7, 3,
    7, 3, 7, 4, 7, 4, 7, 1, 6, 2, 6, 3, 7, 3, 7, 2, 6, 2, 6, 3, 7, 3, 7, 3, 7,
    3, 7, 4, 7, 4, 7, 3, 7, 3, 7, 4, 7, 4, 7, 5, 8, 6, 8, 7, 8, 7, 8, 6, 8, 6,
    8, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8, 6, 8, 6, 8,
    7, 8, 7, 8, 6, 8, 6, 8, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8, 7,
    8, 7, 8, 1, 6, 2, 6, 3, 7, 3, 7, 2, 6, 2, 6, 3, 7, 3, 7, 3, 7, 3, 7, 4, 7,
    4, 7, 3, 7, 3, 7, 4, 7, 4, 7, 2, 6, 2, 6, 3, 7, 3, 7, 2, 6, 2, 6, 3, 7, 3,
    7, 3, 7, 3, 7, 4, 7, 4, 7, 3, 7, 3, 7, 4, 7, 4, 7, 6, 8, 6, 8, 7, 8, 7, 8,
    6, 8, 6, 8, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8, 6,
    8, 6, 8, 7, 8, 7, 8, 6, 8, 6, 8, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8,
    7, 8, 7, 8, 7, 8,
];

#[rustfmt::skip]
const ZERO_CTX_HH: [u8; 256] = [
    0, 1, 3, 4, 1, 2, 4, 5, 3, 4, 6, 7, 4, 5, 7, 7, 1, 2, 4, 5, 2, 2, 5, 5, 4,
    5, 7, 7, 5, 5, 7, 7, 3, 4, 6, 7, 4, 5, 7, 7, 6, 7, 8, 8, 7, 7, 8, 8, 4, 5,
    7, 7, 5, 5, 7, 7, 7, 7, 8, 8, 7, 7, 8, 8, 1, 2, 4, 5, 2, 2, 5, 5, 4, 5, 7,
    7, 5, 5, 7, 7, 2, 2, 5, 5, 2, 2, 5, 5, 5, 5, 7, 7, 5, 5, 7, 7, 4, 5, 7, 7,
    5, 5, 7, 7, 7, 7, 8, 8, 7, 7, 8, 8, 5, 5, 7, 7, 5, 5, 7, 7, 7, 7, 8, 8, 7,
    7, 8, 8, 3, 4, 6, 7, 4, 5, 7, 7, 6, 7, 8, 8, 7, 7, 8, 8, 4, 5, 7, 7, 5, 5,
    7, 7, 7, 7, 8, 8, 7, 7, 8, 8, 6, 7, 8, 8, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    8, 7, 7, 8, 8, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 4, 5, 7, 7, 5, 5, 7, 7,
    7, 7, 8, 8, 7, 7, 8, 8, 5, 5, 7, 7, 5, 5, 7, 7, 7, 7, 8, 8, 7, 7, 8, 8, 7,
    7, 8, 8, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 7, 7, 8, 8, 7, 7, 8, 8, 8, 8,
    8, 8, 8, 8, 8, 8,
];

/// Sign coding context lookup (Table D.2), matching the decoder's convention.
///
/// The index is built by combining significance and sign of the 4 cardinal
/// neighbors into a merged byte:
///   1. `significances = neighbor_byte & 0b01010101` (keep T(6), L(4), R(2), B(0))
///   2. `signs = (top_sign << 6) | (left_sign << 4) | (right_sign << 2) | bottom_sign`
///   3. `negative_sigs = significances & signs`
///   4. `positive_sigs = significances & !signs`
///   5. `merged = (negative_sigs << 1) | positive_sigs`
///
/// Each entry is (`context_label`, `xor_bit`). (0,0) represents impossible combinations.
#[rustfmt::skip]
const SIGN_CONTEXT_LOOKUP: [(u8, u8); 256] = [
    (9,0), (10,0), (10,1), (0,0), (12,0), (13,0), (11,0), (0,0), (12,1), (11,1),
    (13,1), (0,0), (0,0), (0,0), (0,0), (0,0), (12,0), (13,0), (11,0), (0,0),
    (12,0), (13,0), (11,0), (0,0), (9,0), (10,0), (10,1), (0,0), (0,0), (0,0),
    (0,0), (0,0), (12,1), (11,1), (13,1), (0,0), (9,0), (10,0), (10,1), (0,0),
    (12,1), (11,1), (13,1), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0),
    (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0),
    (0,0), (0,0), (0,0), (10,0), (10,0), (9,0), (0,0), (13,0), (13,0), (12,0),
    (0,0), (11,1), (11,1), (12,1), (0,0), (0,0), (0,0), (0,0), (0,0), (13,0),
    (13,0), (12,0), (0,0), (13,0), (13,0), (12,0), (0,0), (10,0), (10,0), (9,0),
    (0,0), (0,0), (0,0), (0,0), (0,0), (11,1), (11,1), (12,1), (0,0), (10,0),
    (10,0), (9,0), (0,0), (11,1), (11,1), (12,1), (0,0), (0,0), (0,0), (0,0),
    (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0),
    (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (10,1), (9,0), (10,1), (0,0),
    (11,0), (12,0), (11,0), (0,0), (13,1), (12,1), (13,1), (0,0), (0,0), (0,0),
    (0,0), (0,0), (11,0), (12,0), (11,0), (0,0), (11,0), (12,0), (11,0), (0,0),
    (10,1), (9,0), (10,1), (0,0), (0,0), (0,0), (0,0), (0,0), (13,1), (12,1),
    (13,1), (0,0), (10,1), (9,0), (10,1), (0,0), (13,1), (12,1), (13,1), (0,0),
    (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0),
    (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0),
    (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0),
    (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0),
    (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0),
    (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0),
    (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0),
    (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0),
];

pub(super) fn prepare_padded_coefficients(
    coefficients: &[i64],
    w: usize,
    h: usize,
    pw: usize,
) -> (Vec<u64>, Vec<u8>) {
    let mut magnitudes = vec![0u64; pw * (h + 2)];
    let mut states = vec![0u8; magnitudes.len()];

    for y in 0..h {
        for x in 0..w {
            let idx = (y + 1) * pw + (x + 1);
            let coeff = coefficients[y * w + x];
            magnitudes[idx] = coeff.unsigned_abs();
            if coeff < 0 {
                states[idx] = NEGATIVE;
            }
        }
    }

    (magnitudes, states)
}

pub(super) fn mark_coded_in_current_pass(
    idx: usize,
    states: &mut [u8],
    coded_indices: &mut Vec<usize>,
) {
    if states[idx] & CODED_IN_CURRENT_PASS == 0 {
        states[idx] |= CODED_IN_CURRENT_PASS;
        coded_indices.push(idx);
    }
}

pub(super) fn clear_coded_in_current_pass(states: &mut [u8], coded_indices: &mut Vec<usize>) {
    for idx in coded_indices.drain(..) {
        states[idx] &= !CODED_IN_CURRENT_PASS;
    }
}

/// Significance Propagation Pass (D.3.1)
#[expect(clippy::too_many_arguments, reason = "stable pass boundary")]
#[expect(clippy::trivially_copy_pass_by_ref, reason = "shared style borrow")]
pub(super) fn significance_propagation_pass(
    magnitudes: &[u64],
    states: &mut [u8],
    neighbors: &mut [u8],
    coded_indices: &mut Vec<usize>,
    encoder: &mut ArithmeticEncoder,
    contexts: &mut [ArithmeticEncoderContext; 19],
    w: usize,
    h: usize,
    pw: usize,
    bit_mask: u64,
    sub_band_type: SubBandType,
    style: &CodeBlockStyle,
) {
    if style.vertically_causal_context {
        significance_propagation_pass_impl::<true>(
            magnitudes,
            states,
            neighbors,
            coded_indices,
            encoder,
            contexts,
            w,
            h,
            pw,
            bit_mask,
            sub_band_type,
        );
    } else {
        significance_propagation_pass_impl::<false>(
            magnitudes,
            states,
            neighbors,
            coded_indices,
            encoder,
            contexts,
            w,
            h,
            pw,
            bit_mask,
            sub_band_type,
        );
    }
}

#[expect(clippy::too_many_arguments, reason = "explicit hot scan state")]
fn significance_propagation_pass_impl<const VERTICAL_CAUSAL: bool>(
    magnitudes: &[u64],
    states: &mut [u8],
    neighbors: &mut [u8],
    coded_indices: &mut Vec<usize>,
    encoder: &mut ArithmeticEncoder,
    contexts: &mut [ArithmeticEncoderContext; 19],
    w: usize,
    h: usize,
    pw: usize,
    bit_mask: u64,
    sub_band_type: SubBandType,
) {
    for y_base in (0..h).step_by(4) {
        for x in 0..w {
            let y_end = (y_base + 4).min(h);
            for y in y_base..y_end {
                let idx = (y + 1) * pw + (x + 1);
                let is_significant = states[idx] & SIGNIFICANT != 0;
                let neighbor_sig = effective_neighbor_sig::<VERTICAL_CAUSAL>(neighbors[idx], y, h);
                let has_sig_neighbors = neighbor_sig != 0;

                if !is_significant && has_sig_neighbors {
                    let ctx_label = zero_coding_ctx(neighbor_sig, sub_band_type);
                    let bit = u32::from(magnitudes[idx] & bit_mask != 0);
                    encoder.encode(bit, &mut contexts[ctx_label as usize]);
                    mark_coded_in_current_pass(idx, states, coded_indices);

                    if bit == 1 {
                        encode_sign::<VERTICAL_CAUSAL>(
                            idx, neighbors, states, encoder, contexts, pw, y, h,
                        );
                        set_significant(idx, states, neighbors, pw);
                    }
                }
            }
        }
    }
}

#[expect(clippy::too_many_arguments, reason = "stable raw pass boundary")]
#[expect(clippy::trivially_copy_pass_by_ref, reason = "shared style borrow")]
pub(super) fn significance_propagation_pass_raw(
    magnitudes: &[u64],
    states: &mut [u8],
    neighbors: &mut [u8],
    coded_indices: &mut Vec<usize>,
    writer: &mut BitWriter,
    w: usize,
    h: usize,
    pw: usize,
    bit_mask: u64,
    style: &CodeBlockStyle,
) {
    if style.vertically_causal_context {
        significance_propagation_pass_raw_impl::<true>(
            magnitudes,
            states,
            neighbors,
            coded_indices,
            writer,
            w,
            h,
            pw,
            bit_mask,
        );
    } else {
        significance_propagation_pass_raw_impl::<false>(
            magnitudes,
            states,
            neighbors,
            coded_indices,
            writer,
            w,
            h,
            pw,
            bit_mask,
        );
    }
}

#[expect(clippy::too_many_arguments, reason = "explicit hot scan state")]
fn significance_propagation_pass_raw_impl<const VERTICAL_CAUSAL: bool>(
    magnitudes: &[u64],
    states: &mut [u8],
    neighbors: &mut [u8],
    coded_indices: &mut Vec<usize>,
    writer: &mut BitWriter,
    w: usize,
    h: usize,
    pw: usize,
    bit_mask: u64,
) {
    for y_base in (0..h).step_by(4) {
        for x in 0..w {
            let y_end = (y_base + 4).min(h);
            for y in y_base..y_end {
                let idx = (y + 1) * pw + (x + 1);
                let is_significant = states[idx] & SIGNIFICANT != 0;
                let neighbor_sig = effective_neighbor_sig::<VERTICAL_CAUSAL>(neighbors[idx], y, h);
                if !is_significant && neighbor_sig != 0 {
                    let bit = u32::from(magnitudes[idx] & bit_mask != 0);
                    writer.write_bit(bit);
                    mark_coded_in_current_pass(idx, states, coded_indices);
                    if bit == 1 {
                        encode_sign_raw(idx, states, writer);
                        set_significant(idx, states, neighbors, pw);
                    }
                }
            }
        }
    }
}

/// Magnitude Refinement Pass (D.3.3)
#[expect(clippy::too_many_arguments, reason = "stable pass boundary")]
#[expect(clippy::trivially_copy_pass_by_ref, reason = "shared style borrow")]
pub(super) fn magnitude_refinement_pass(
    magnitudes: &[u64],
    states: &mut [u8],
    neighbors: &mut [u8],
    encoder: &mut ArithmeticEncoder,
    contexts: &mut [ArithmeticEncoderContext; 19],
    w: usize,
    h: usize,
    pw: usize,
    bit_mask: u64,
    style: &CodeBlockStyle,
) {
    if style.vertically_causal_context {
        magnitude_refinement_pass_impl::<true>(
            magnitudes, states, neighbors, encoder, contexts, w, h, pw, bit_mask,
        );
    } else {
        magnitude_refinement_pass_impl::<false>(
            magnitudes, states, neighbors, encoder, contexts, w, h, pw, bit_mask,
        );
    }
}

#[expect(clippy::too_many_arguments, reason = "explicit hot scan state")]
fn magnitude_refinement_pass_impl<const VERTICAL_CAUSAL: bool>(
    magnitudes: &[u64],
    states: &mut [u8],
    neighbors: &mut [u8],
    encoder: &mut ArithmeticEncoder,
    contexts: &mut [ArithmeticEncoderContext; 19],
    w: usize,
    h: usize,
    pw: usize,
    bit_mask: u64,
) {
    for y_base in (0..h).step_by(4) {
        for x in 0..w {
            let y_end = (y_base + 4).min(h);
            for y in y_base..y_end {
                let idx = (y + 1) * pw + (x + 1);
                let is_significant = states[idx] & SIGNIFICANT != 0;
                let coded_this_pass = states[idx] & CODED_IN_CURRENT_PASS != 0;

                if is_significant && !coded_this_pass {
                    let ctx_label = magnitude_refinement_ctx(
                        states[idx],
                        effective_neighbor_sig::<VERTICAL_CAUSAL>(neighbors[idx], y, h),
                    );
                    let bit = u32::from(magnitudes[idx] & bit_mask != 0);
                    encoder.encode(bit, &mut contexts[ctx_label as usize]);
                    states[idx] |= MAGNITUDE_REFINED;
                }
            }
        }
    }
}

#[expect(clippy::too_many_arguments, reason = "stable raw pass boundary")]
#[expect(clippy::trivially_copy_pass_by_ref, reason = "shared style borrow")]
pub(super) fn magnitude_refinement_pass_raw(
    magnitudes: &[u64],
    states: &mut [u8],
    neighbors: &mut [u8],
    writer: &mut BitWriter,
    w: usize,
    h: usize,
    pw: usize,
    bit_mask: u64,
    style: &CodeBlockStyle,
) {
    if style.vertically_causal_context {
        magnitude_refinement_pass_raw_impl::<true>(
            magnitudes, states, neighbors, writer, w, h, pw, bit_mask,
        );
    } else {
        magnitude_refinement_pass_raw_impl::<false>(
            magnitudes, states, neighbors, writer, w, h, pw, bit_mask,
        );
    }
}

#[expect(clippy::too_many_arguments, reason = "explicit hot scan state")]
fn magnitude_refinement_pass_raw_impl<const VERTICAL_CAUSAL: bool>(
    magnitudes: &[u64],
    states: &mut [u8],
    neighbors: &mut [u8],
    writer: &mut BitWriter,
    w: usize,
    h: usize,
    pw: usize,
    bit_mask: u64,
) {
    for y_base in (0..h).step_by(4) {
        for x in 0..w {
            let y_end = (y_base + 4).min(h);
            for y in y_base..y_end {
                let idx = (y + 1) * pw + (x + 1);
                let is_significant = states[idx] & SIGNIFICANT != 0;
                let coded_this_pass = states[idx] & CODED_IN_CURRENT_PASS != 0;
                let _neighbor_sig = effective_neighbor_sig::<VERTICAL_CAUSAL>(neighbors[idx], y, h);
                if is_significant && !coded_this_pass {
                    let bit = u32::from(magnitudes[idx] & bit_mask != 0);
                    writer.write_bit(bit);
                    states[idx] |= MAGNITUDE_REFINED;
                }
            }
        }
    }
}

/// Cleanup Pass (D.3.4)
#[expect(clippy::too_many_arguments, reason = "stable cleanup boundary")]
#[expect(clippy::trivially_copy_pass_by_ref, reason = "shared style borrow")]
pub(super) fn cleanup_pass(
    magnitudes: &[u64],
    states: &mut [u8],
    neighbors: &mut [u8],
    encoder: &mut ArithmeticEncoder,
    contexts: &mut [ArithmeticEncoderContext; 19],
    w: usize,
    h: usize,
    pw: usize,
    bit_mask: u64,
    sub_band_type: SubBandType,
    style: &CodeBlockStyle,
) {
    if style.vertically_causal_context {
        cleanup_pass_impl::<true>(
            magnitudes,
            states,
            neighbors,
            encoder,
            contexts,
            w,
            h,
            pw,
            bit_mask,
            sub_band_type,
        );
    } else {
        cleanup_pass_impl::<false>(
            magnitudes,
            states,
            neighbors,
            encoder,
            contexts,
            w,
            h,
            pw,
            bit_mask,
            sub_band_type,
        );
    }
}

#[expect(clippy::cast_possible_truncation, reason = "four-row run position")]
#[expect(clippy::too_many_arguments, reason = "explicit hot scan state")]
fn cleanup_pass_impl<const VERTICAL_CAUSAL: bool>(
    magnitudes: &[u64],
    states: &mut [u8],
    neighbors: &mut [u8],
    encoder: &mut ArithmeticEncoder,
    contexts: &mut [ArithmeticEncoderContext; 19],
    w: usize,
    h: usize,
    pw: usize,
    bit_mask: u64,
    sub_band_type: SubBandType,
) {
    for y_base in (0..h).step_by(4) {
        for x in 0..w {
            let y_end = (y_base + 4).min(h);
            let stripe_height = y_end - y_base;

            // Try run-length coding for full 4-row stripes
            if stripe_height == 4 {
                let mut all_zero_uncoded = true;
                for y in y_base..y_end {
                    let idx = (y + 1) * pw + (x + 1);
                    if states[idx] & (SIGNIFICANT | CODED_IN_CURRENT_PASS) != 0
                        || effective_neighbor_sig::<VERTICAL_CAUSAL>(neighbors[idx], y, h) != 0
                    {
                        all_zero_uncoded = false;
                        break;
                    }
                }

                if all_zero_uncoded {
                    // Check if any coefficient in this stripe becomes significant
                    let mut first_sig = None;
                    for (j, y) in (y_base..y_end).enumerate() {
                        let idx = (y + 1) * pw + (x + 1);
                        if magnitudes[idx] & bit_mask != 0 {
                            first_sig = Some(j);
                            break;
                        }
                    }

                    if let Some(pos) = first_sig {
                        // Not all zero: encode RLC=1, then position
                        encoder.encode(1, &mut contexts[17]); // RLC context
                        encoder.encode((pos >> 1) as u32 & 1, &mut contexts[18]); // UNIFORM
                        encoder.encode(pos as u32 & 1, &mut contexts[18]); // UNIFORM

                        // Encode sign for the first significant
                        let y = y_base + pos;
                        let idx = (y + 1) * pw + (x + 1);
                        encode_sign::<VERTICAL_CAUSAL>(
                            idx, neighbors, states, encoder, contexts, pw, y, h,
                        );
                        set_significant(idx, states, neighbors, pw);

                        // Continue cleanup for remaining samples in stripe
                        for y in (y_base + pos + 1)..y_end {
                            let idx = (y + 1) * pw + (x + 1);
                            if states[idx] & (SIGNIFICANT | CODED_IN_CURRENT_PASS) == 0 {
                                let ctx_label = zero_coding_ctx(
                                    effective_neighbor_sig::<VERTICAL_CAUSAL>(neighbors[idx], y, h),
                                    sub_band_type,
                                );
                                let bit = u32::from(magnitudes[idx] & bit_mask != 0);
                                encoder.encode(bit, &mut contexts[ctx_label as usize]);
                                if bit == 1 {
                                    encode_sign::<VERTICAL_CAUSAL>(
                                        idx, neighbors, states, encoder, contexts, pw, y, h,
                                    );
                                    set_significant(idx, states, neighbors, pw);
                                }
                            }
                        }
                        continue;
                    }

                    // All zero: encode RLC=0
                    encoder.encode(0, &mut contexts[17]);
                    continue;
                }
            }

            // Non-RLC: process each sample individually
            for y in y_base..y_end {
                let idx = (y + 1) * pw + (x + 1);
                if states[idx] & (SIGNIFICANT | CODED_IN_CURRENT_PASS) == 0 {
                    let ctx_label = zero_coding_ctx(
                        effective_neighbor_sig::<VERTICAL_CAUSAL>(neighbors[idx], y, h),
                        sub_band_type,
                    );
                    let bit = u32::from(magnitudes[idx] & bit_mask != 0);
                    encoder.encode(bit, &mut contexts[ctx_label as usize]);
                    if bit == 1 {
                        encode_sign::<VERTICAL_CAUSAL>(
                            idx, neighbors, states, encoder, contexts, pw, y, h,
                        );
                        set_significant(idx, states, neighbors, pw);
                    }
                }
            }
        }
    }
}

/// Encode the sign of a newly significant coefficient.
///
/// The sign context is computed exactly as the decoder does it:
/// combine significance and sign of the 4 cardinal neighbors into a
/// merged byte and look up `SIGN_CONTEXT_LOOKUP`.
#[expect(clippy::too_many_arguments, reason = "explicit hot sign state")]
fn encode_sign<const VERTICAL_CAUSAL: bool>(
    idx: usize,
    neighbors: &[u8],
    states: &[u8],
    encoder: &mut ArithmeticEncoder,
    contexts: &mut [ArithmeticEncoderContext; 19],
    pw: usize,
    y: usize,
    h: usize,
) {
    // Get cardinal-neighbor significances: T(6), L(4), R(2), B(0)
    let significances =
        effective_neighbor_sig::<VERTICAL_CAUSAL>(neighbors[idx], y, h) & 0b0101_0101;

    // Get sign of each cardinal neighbor (0=positive, 1=negative).
    // Only meaningful for significant neighbors; insignificant neighbors get 0.
    let top_sign = if states[idx - pw] & SIGNIFICANT != 0 {
        u8::from((states[idx - pw] & NEGATIVE) != 0)
    } else {
        0
    };
    let left_sign = if states[idx - 1] & SIGNIFICANT != 0 {
        u8::from((states[idx - 1] & NEGATIVE) != 0)
    } else {
        0
    };
    let right_sign = if states[idx + 1] & SIGNIFICANT != 0 {
        u8::from((states[idx + 1] & NEGATIVE) != 0)
    } else {
        0
    };
    let bottom_sign = if VERTICAL_CAUSAL && neighbor_in_next_stripe(y, h) {
        0
    } else if states[idx + pw] & SIGNIFICANT != 0 {
        u8::from((states[idx + pw] & NEGATIVE) != 0)
    } else {
        0
    };

    // Build sign bits at the same positions as significances
    let sign_bits = (top_sign << 6) | (left_sign << 4) | (right_sign << 2) | bottom_sign;

    // Split into negative-significant and positive-significant
    let negative_sigs = significances & sign_bits;
    let positive_sigs = significances & !sign_bits;
    // Merge: negative at (pos+1), positive at (pos) → 2-bit per neighbor
    let merged = (negative_sigs << 1) | positive_sigs;

    let (ctx_label, xor_bit) = SIGN_CONTEXT_LOOKUP[merged as usize];
    let sign_bit = u32::from((states[idx] & NEGATIVE) != 0);
    encoder.encode(
        sign_bit ^ u32::from(xor_bit),
        &mut contexts[ctx_label as usize],
    );
}

fn encode_sign_raw(idx: usize, states: &[u8], writer: &mut BitWriter) {
    let is_significant = states[idx] & SIGNIFICANT != 0;
    debug_assert!(!is_significant);
    writer.write_bit(u32::from((states[idx] & NEGATIVE) != 0));
}

#[inline]
fn neighbor_in_next_stripe(y: usize, height: usize) -> bool {
    y + 1 < height && ((y + 1) >> 2) > (y >> 2)
}

#[expect(clippy::inline_always, reason = "Tier-1 coefficient-loop hot path")]
#[inline(always)]
fn effective_neighbor_sig<const VERTICAL_CAUSAL: bool>(
    neighbor_sig: u8,
    y: usize,
    height: usize,
) -> u8 {
    if VERTICAL_CAUSAL && neighbor_in_next_stripe(y, height) {
        neighbor_sig & 0b1111_0100
    } else {
        neighbor_sig
    }
}

/// Get the zero-coding context label for a coefficient.
#[inline]
fn zero_coding_ctx(neighbor_sig: u8, sub_band_type: SubBandType) -> u8 {
    match sub_band_type {
        SubBandType::LowLow | SubBandType::LowHigh => ZERO_CTX_LL_LH[neighbor_sig as usize],
        SubBandType::HighLow => ZERO_CTX_HL[neighbor_sig as usize],
        SubBandType::HighHigh => ZERO_CTX_HH[neighbor_sig as usize],
    }
}

/// Get the magnitude refinement context label (Table D.4).
///
/// Matches the decoder: if already magnitude-refined → 16,
/// else if at least one neighbor is significant → 15, else 14.
#[inline]
fn magnitude_refinement_ctx(state: u8, neighbor_sig: u8) -> u8 {
    if state & MAGNITUDE_REFINED != 0 {
        16
    } else {
        14 + neighbor_sig.min(1)
    }
}

/// Mark a coefficient as significant and update neighbor significance maps.
fn set_significant(idx: usize, states: &mut [u8], neighbors: &mut [u8], pw: usize) {
    states[idx] |= SIGNIFICANT;

    // Update 8 neighbors
    // Neighbor bit layout: TL(7) T(6) TR(5) L(4) BL(3) R(2) BR(1) B(0)
    let top = idx - pw;
    let bottom = idx + pw;

    neighbors[top - 1] |= 1 << 1; // bottom-right of top-left
    neighbors[top] |= 1; // bottom of top
    neighbors[top + 1] |= 1 << 3; // bottom-left of top-right
    neighbors[idx - 1] |= 1 << 2; // right of left
    neighbors[idx + 1] |= 1 << 4; // left of right
    neighbors[bottom - 1] |= 1 << 5; // top-right of bottom-left
    neighbors[bottom] |= 1 << 6; // top of bottom
    neighbors[bottom + 1] |= 1 << 7; // top-left of bottom-right
}
