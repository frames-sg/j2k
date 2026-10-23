// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::ht_encode_tables::{
    HtUvlcTableEntry, HT_UVLC_ENCODE_TABLE, HT_VLC_ENCODE_TABLE0, HT_VLC_ENCODE_TABLE1,
};
use super::quad::{quad_pair, CodedQuad, QuadSink, RowPair};
use super::writers::{MagSgnEncoder, MelEncoder, VlcEncoder};

struct EncodeQuadSink<'a> {
    mel: &'a mut MelEncoder,
    vlc: &'a mut VlcEncoder,
    ms: &'a mut MagSgnEncoder,
    pair_vlc: PairVlc,
}

/// The VLC bits of one quad pair: its `CxtVLC` codewords, then its UVLC
/// prefixes and suffixes, written to the VLC stream in one call. At most
/// 2 * 7 + 2 * (3 + 5) bits.
#[derive(Default)]
struct PairVlc {
    bits: u64,
    len: u8,
}

impl PairVlc {
    #[expect(clippy::inline_always, reason = "runs several times per quad pair")]
    #[inline(always)]
    fn push(&mut self, codeword: u32, len: u8) {
        self.bits |= (u64::from(codeword) & ((1 << len) - 1)) << self.len;
        self.len += len;
    }
}

impl EncodeQuadSink<'_> {
    #[expect(clippy::inline_always, reason = "runs once per quad pair")]
    #[inline(always)]
    fn flush_pair_vlc(&mut self) -> Result<(), &'static str> {
        self.vlc.encode(self.pair_vlc.bits, self.pair_vlc.len)
    }
}

impl QuadSink for EncodeQuadSink<'_> {
    type Error = &'static str;

    #[expect(
        clippy::inline_always,
        reason = "the byte-emitting quad runs in the cleanup hot path"
    )]
    #[inline(always)]
    fn quad<const INITIAL: bool>(&mut self, quad: CodedQuad) -> Result<(), Self::Error> {
        let table = if INITIAL {
            &HT_VLC_ENCODE_TABLE0
        } else {
            &HT_VLC_ENCODE_TABLE1
        };
        let tuple = table[usize::from(quad.index)];
        self.pair_vlc
            .push(u32::from(tuple >> 8), ((tuple >> 4) & 0x7) as u8);
        let rho = quad.rho();
        if quad.context() == 0 {
            self.mel.encode(rho != 0)?;
        }
        encode_mag_signs(rho, quad.bound, tuple, &quad.values, self.ms)
    }

    #[expect(clippy::inline_always, reason = "erase encoder sink dispatch")]
    #[inline(always)]
    fn initial_uvlc_pair(&mut self, u_q0: i32, u_q1: i32) -> Result<(), Self::Error> {
        if u_q0 > 0 && u_q1 > 0 {
            self.mel.encode(u_q0.min(u_q1) > 2)?;
        }
        encode_uvlc(u_q0, u_q1, &mut self.pair_vlc);
        self.flush_pair_vlc()
    }

    #[expect(clippy::inline_always, reason = "erase encoder sink dispatch")]
    #[inline(always)]
    fn initial_uvlc_lone(&mut self, u_q0: i32) -> Result<(), Self::Error> {
        encode_uvlc(u_q0, 0, &mut self.pair_vlc);
        self.flush_pair_vlc()
    }

    #[expect(clippy::inline_always, reason = "erase encoder sink dispatch")]
    #[inline(always)]
    fn non_initial_uvlc(&mut self, u_q0: i32, u_q1: i32) -> Result<(), Self::Error> {
        encode_uvlc_non_initial(u_q0, u_q1, &mut self.pair_vlc);
        self.flush_pair_vlc()
    }
}

/// Code the quad pair at column `x` of the first (`INITIAL`) or a later row
/// pair into the three streams.
#[expect(clippy::inline_always, reason = "fuse the encoder quad walker")]
#[inline(always)]
pub(super) fn encode_quad_pair<const INITIAL: bool>(
    rows: &RowPair,
    width: usize,
    x: usize,
    mel: &mut MelEncoder,
    vlc: &mut VlcEncoder,
    ms: &mut MagSgnEncoder,
) -> Result<(), &'static str> {
    let pair_vlc = PairVlc::default();
    quad_pair::<INITIAL, _>(
        rows,
        width,
        x,
        &mut EncodeQuadSink {
            mel,
            vlc,
            ms,
            pair_vlc,
        },
    )
}

#[expect(
    clippy::inline_always,
    reason = "HT table exponents and Uq values are bounded bit counts in this per-sample hot path"
)]
#[inline(always)]
fn encode_mag_signs(
    rho: u8,
    u_q: u8,
    tuple: u16,
    s: &[u32; 4],
    ms: &mut MagSgnEncoder,
) -> Result<(), &'static str> {
    let (rho, u_q) = (u32::from(rho), u32::from(u_q));
    let e_k = u32::from(tuple & 0xF);
    // Payloads go out least significant first, as separate writes would. An
    // insignificant sample has a 0-bit payload, so packing needs no branches.
    // U <= 32 bounds a payload; with U <= 16 the quad fits one 64-bit word.
    let pack = |first: usize, count: usize| {
        let (mut word, mut word_bits) = (0_u64, 0_u32);
        for (sample, &value) in s.iter().enumerate().skip(first).take(count) {
            let significant = (rho >> sample) & 1;
            let magnitude_bits = significant * (u_q - ((e_k >> sample) & 1));
            let payload = u64::from(value) & ((1_u64 << magnitude_bits) - 1);
            word |= payload << word_bits;
            word_bits += magnitude_bits;
        }
        (word, word_bits)
    };
    if u_q <= 16 {
        let (word, word_bits) = pack(0, 4);
        return ms.encode(word, word_bits);
    }
    let (word, word_bits) = pack(0, 2);
    ms.encode(word, word_bits)?;
    let (word, word_bits) = pack(2, 2);
    ms.encode(word, word_bits)
}

#[expect(
    clippy::cast_sign_loss,
    reason = "branch guards and max operations make every UVLC table index and suffix nonnegative"
)]
#[expect(
    clippy::inline_always,
    reason = "runs once per quad pair in the cleanup hot path"
)]
#[inline(always)]
fn encode_uvlc(u_q0: i32, u_q1: i32, vlc: &mut PairVlc) {
    if u_q0 > 2 && u_q1 > 2 {
        let first = HT_UVLC_ENCODE_TABLE[(u_q0 - 2) as usize];
        let second = HT_UVLC_ENCODE_TABLE[(u_q1 - 2) as usize];
        encode_uvlc_pair(vlc, first, second);
    } else if u_q0 > 2 && u_q1 > 0 {
        let first = HT_UVLC_ENCODE_TABLE[u_q0 as usize];
        vlc.push(u32::from(first.pre), first.pre_len);
        vlc.push((u_q1 - 1) as u32, 1);
        vlc.push(u32::from(first.suf), first.suf_len);
    } else {
        let first = HT_UVLC_ENCODE_TABLE[u_q0.max(0) as usize];
        let second = HT_UVLC_ENCODE_TABLE[u_q1.max(0) as usize];
        encode_uvlc_pair(vlc, first, second);
    }
}

#[expect(
    clippy::cast_sign_loss,
    reason = "max with zero makes both non-initial UVLC table indices nonnegative"
)]
#[expect(
    clippy::inline_always,
    reason = "runs once per quad pair in the cleanup hot path"
)]
#[inline(always)]
fn encode_uvlc_non_initial(u_q0: i32, u_q1: i32, vlc: &mut PairVlc) {
    let first = HT_UVLC_ENCODE_TABLE[u_q0.max(0) as usize];
    let second = HT_UVLC_ENCODE_TABLE[u_q1.max(0) as usize];
    encode_uvlc_pair(vlc, first, second);
}

#[expect(
    clippy::inline_always,
    reason = "runs once per quad pair in the cleanup hot path"
)]
#[inline(always)]
fn encode_uvlc_pair(vlc: &mut PairVlc, first: HtUvlcTableEntry, second: HtUvlcTableEntry) {
    vlc.push(u32::from(first.pre), first.pre_len);
    vlc.push(u32::from(second.pre), second.pre_len);
    vlc.push(u32::from(first.suf), first.suf_len);
    vlc.push(u32::from(second.suf), second.suf_len);
}
