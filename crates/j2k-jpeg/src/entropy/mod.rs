// SPDX-License-Identifier: MIT OR Apache-2.0

//! Entropy decoding — Huffman tables and the per-MCU block decoder.

pub(crate) mod block;
pub(crate) mod huffman;
pub(crate) mod progressive;
pub(crate) mod sequential;

/// T.81 §A.3.6 zigzag order: the 8×8 coefficient scan order from DC to
/// highest-frequency AC. Coefficient `k` in the stream lands at linear
/// position `ZIGZAG[k]` in the 8×8 block (row-major).
pub(crate) const ZIGZAG: [u8; 64] = j2k_codec_math::jpeg::ZIGZAG;
