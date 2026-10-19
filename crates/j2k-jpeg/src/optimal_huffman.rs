// SPDX-License-Identifier: MIT OR Apache-2.0

//! Huffman tables built from symbol statistics (T.81 Annex K.2).
//!
//! The construction follows libjpeg's `jpeg_gen_optimal_table`, tie-breaking
//! included, so tables match `jpegtran -optimize` and `cjpeg -optimize` for
//! the same statistics. One code point is reserved so that no symbol is coded
//! with all ones, which T.81 forbids.

use crate::adapter::JpegBaselineHuffmanTable;
use crate::encoder::JpegEncodeError;

/// Symbols a JPEG Huffman table can code.
const SYMBOLS: usize = 256;
/// Longest code the unconstrained construction can produce: every merge
/// lengthens at least one code by one, and there are `SYMBOLS` merges.
const MAX_UNLIMITED_LEN: usize = SYMBOLS + 1;
/// Longest code JPEG allows.
const MAX_CODE_LEN: usize = 16;

/// Occurrence counts of each Huffman symbol for one table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct HuffmanFrequencies([u64; SYMBOLS]);

impl HuffmanFrequencies {
    pub(crate) const fn new() -> Self {
        Self([0; SYMBOLS])
    }

    pub(crate) fn add(&mut self, symbol: u8) {
        let count = &mut self.0[usize::from(symbol)];
        *count = count.saturating_add(1);
    }

    pub(crate) fn add_count(&mut self, symbol: u8, count: u64) {
        let slot = &mut self.0[usize::from(symbol)];
        *slot = slot.saturating_add(count);
    }

    pub(crate) fn count(&self, symbol: u8) -> u64 {
        self.0[usize::from(symbol)]
    }
}

impl Default for HuffmanFrequencies {
    fn default() -> Self {
        Self::new()
    }
}

/// A Huffman table as a DHT segment carries it: code counts per length and
/// the symbols in code order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct HuffmanSpec {
    pub(crate) bits: [u8; MAX_CODE_LEN],
    values: [u8; SYMBOLS],
    len: usize,
}

impl HuffmanSpec {
    /// Symbols in code order, as the DHT segment lists them.
    pub(crate) fn values(&self) -> &[u8] {
        &self.values[..self.len]
    }

    /// Canonical codes (T.81 Annex C) for every symbol in the table.
    pub(crate) fn table(&self) -> JpegBaselineHuffmanTable {
        let mut table = JpegBaselineHuffmanTable {
            codes: [0; SYMBOLS],
            lens: [0; SYMBOLS],
        };
        let mut code = 0_u32;
        let mut values = self.values().iter();
        for (index, &count) in self.bits.iter().enumerate() {
            let len = u8::try_from(index + 1).unwrap_or(u8::MAX);
            for _ in 0..count {
                if let Some(&symbol) = values.next() {
                    table.codes[usize::from(symbol)] = u16::try_from(code).unwrap_or(u16::MAX);
                    table.lens[usize::from(symbol)] = len;
                }
                code += 1;
            }
            code <<= 1;
        }
        table
    }
}

/// Build the optimal length-limited table for `frequencies`.
///
/// # Errors
/// Returns an invariant error when no symbol occurs: a JPEG table must code
/// at least one symbol.
pub(crate) fn optimal_huffman_spec(
    frequencies: &HuffmanFrequencies,
) -> Result<HuffmanSpec, JpegEncodeError> {
    if frequencies.0.iter().all(|&count| count == 0) {
        return Err(JpegEncodeError::InternalInvariant {
            reason: "optimal Huffman table has no symbols",
        });
    }
    let code_sizes = unlimited_code_sizes(frequencies);
    let mut bits = [0_usize; MAX_UNLIMITED_LEN + 1];
    for &size in &code_sizes {
        if size > 0 {
            bits[size] += 1;
        }
    }
    limit_code_lengths(&mut bits);
    // Drop the reserved code point, which holds the longest code.
    if let Some(longest) = (1..=MAX_CODE_LEN).rev().find(|&len| bits[len] > 0) {
        bits[longest] -= 1;
    }

    let mut spec = HuffmanSpec {
        bits: [0; MAX_CODE_LEN],
        values: [0; SYMBOLS],
        len: 0,
    };
    for (slot, &count) in spec.bits.iter_mut().zip(&bits[1..=MAX_CODE_LEN]) {
        *slot = u8::try_from(count).map_err(|_| JpegEncodeError::InternalInvariant {
            reason: "optimal Huffman code count exceeds u8",
        })?;
    }
    // Symbols in order of their unlimited code length, then symbol value;
    // length limiting preserves that order.
    for size in 1..=MAX_UNLIMITED_LEN {
        for (symbol, &symbol_size) in code_sizes[..SYMBOLS].iter().enumerate() {
            if symbol_size == size {
                spec.values[spec.len] = u8::try_from(symbol).unwrap_or(u8::MAX);
                spec.len += 1;
            }
        }
    }
    Ok(spec)
}

/// Code lengths of the unconstrained Huffman tree over every symbol plus one
/// reserved symbol of frequency one (T.81 Figure K.1).
fn unlimited_code_sizes(frequencies: &HuffmanFrequencies) -> [usize; SYMBOLS + 1] {
    const NONE: usize = usize::MAX;
    let mut freq = [0_u64; SYMBOLS + 1];
    freq[..SYMBOLS].copy_from_slice(&frequencies.0);
    freq[SYMBOLS] = 1;
    let mut code_size = [0_usize; SYMBOLS + 1];
    let mut others = [NONE; SYMBOLS + 1];
    while let Some(c1) = least_frequent(&freq, None) {
        let Some(c2) = least_frequent(&freq, Some(c1)) else {
            break;
        };
        freq[c1] = freq[c1].saturating_add(freq[c2]);
        freq[c2] = 0;
        let mut node = c1;
        code_size[node] += 1;
        while others[node] != NONE {
            node = others[node];
            code_size[node] += 1;
        }
        others[node] = c2;
        let mut node = c2;
        code_size[node] += 1;
        while others[node] != NONE {
            node = others[node];
            code_size[node] += 1;
        }
    }
    code_size
}

/// The least frequent nonzero entry other than `except`, preferring the
/// highest index on ties, as libjpeg does.
fn least_frequent(freq: &[u64; SYMBOLS + 1], except: Option<usize>) -> Option<usize> {
    let mut best: Option<(usize, u64)> = None;
    for (index, &count) in freq.iter().enumerate() {
        if count == 0 || Some(index) == except {
            continue;
        }
        if best.is_none_or(|(_, least)| count <= least) {
            best = Some((index, count));
        }
    }
    best.map(|(index, _)| index)
}

/// Move codes longer than 16 bits up the tree (T.81 Figure K.3): each step
/// takes two codes of the longest length, gives one a shorter prefix, and
/// splits a shorter code to make room.
fn limit_code_lengths(bits: &mut [usize; MAX_UNLIMITED_LEN + 1]) {
    for len in (MAX_CODE_LEN + 1..=MAX_UNLIMITED_LEN).rev() {
        while bits[len] > 0 {
            let mut shorter = len - 2;
            while bits[shorter] == 0 {
                shorter -= 1;
            }
            bits[len] -= 2;
            bits[len - 1] += 1;
            bits[shorter + 1] += 2;
            bits[shorter] -= 1;
        }
    }
}

#[cfg(test)]
mod tests;
