// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::ht_encode_tables::{
    HtUvlcTableEntry, HT_UVLC_ENCODE_TABLE, HT_VLC_ENCODE_TABLE0, HT_VLC_ENCODE_TABLE1,
};
use super::quad::{
    first_quad_pair, non_initial_quad_pair, FirstQuadPairRequest, InitialQuadRow,
    NonInitialQuadPairRequest, NonInitialQuadRow, QuadSink,
};
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
    fn flush_pair_vlc(&mut self) -> Result<(), &'static str> {
        self.vlc.encode(self.pair_vlc.bits, self.pair_vlc.len)
    }
}

impl QuadSink for EncodeQuadSink<'_> {
    type Error = &'static str;

    #[expect(clippy::inline_always, reason = "erase encoder sink dispatch")]
    #[inline(always)]
    fn quad_initial(&mut self, row: InitialQuadRow<'_>) -> Result<i32, Self::Error> {
        encode_quad_initial_row(row, self.mel, &mut self.pair_vlc, self.ms)
    }

    #[expect(clippy::inline_always, reason = "erase encoder sink dispatch")]
    #[inline(always)]
    fn quad_non_initial(&mut self, row: NonInitialQuadRow<'_>) -> Result<i32, Self::Error> {
        encode_quad_non_initial_row(row, self.mel, &mut self.pair_vlc, self.ms)
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

#[expect(clippy::inline_always, reason = "fuse the encoder quad walker")]
#[inline(always)]
pub(super) fn encode_first_quad_pair(
    request: FirstQuadPairRequest<'_>,
    mel: &mut MelEncoder,
    vlc: &mut VlcEncoder,
    ms: &mut MagSgnEncoder,
) -> Result<(), &'static str> {
    let pair_vlc = PairVlc::default();
    first_quad_pair(
        request,
        &mut EncodeQuadSink {
            mel,
            vlc,
            ms,
            pair_vlc,
        },
    )
}

#[expect(clippy::inline_always, reason = "fuse the encoder quad walker")]
#[inline(always)]
pub(super) fn encode_non_initial_quad_pair(
    request: NonInitialQuadPairRequest<'_>,
    mel: &mut MelEncoder,
    vlc: &mut VlcEncoder,
    ms: &mut MagSgnEncoder,
) -> Result<(), &'static str> {
    let pair_vlc = PairVlc::default();
    non_initial_quad_pair(
        request,
        &mut EncodeQuadSink {
            mel,
            vlc,
            ms,
            pair_vlc,
        },
    )
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::inline_always,
    reason = "packed quad fields are bounded by HT tables and this byte-emitting row is a hot path"
)]
#[inline(always)]
fn encode_quad_initial_row(
    row: InitialQuadRow<'_>,
    mel: &mut MelEncoder,
    vlc: &mut PairVlc,
    ms: &mut MagSgnEncoder,
) -> Result<i32, &'static str> {
    let InitialQuadRow {
        offset,
        c_q,
        rho,
        e_qmax,
        e_q,
        s,
        lep,
        lcxp,
        e_val,
        cx_val,
    } = row;
    let u_q = e_qmax.max(1) - 1;
    let eps = quad_eps(e_q, offset, e_qmax, u_q);

    e_val[lep] = e_val[lep].max(e_q[offset + 1] as u8);
    e_val[lep + 1] = e_q[offset + 3] as u8;
    cx_val[lcxp] |= ((rho & 2) >> 1) as u8;
    cx_val[lcxp + 1] = ((rho & 8) >> 3) as u8;

    let tuple = HT_VLC_ENCODE_TABLE0[(c_q << 8) | ((rho as usize) << 4) | eps as usize];
    vlc.push(u32::from(tuple >> 8), ((tuple >> 4) & 0x7) as u8);

    if c_q == 0 {
        mel.encode(rho != 0)?;
    }

    encode_mag_signs(rho, e_qmax.max(1), tuple, s, offset, ms)?;
    Ok(u_q)
}

#[expect(
    clippy::cast_sign_loss,
    clippy::inline_always,
    clippy::needless_pass_by_value,
    reason = "the by-value row matches the sink contract and its bounded packed fields index HT tables"
)]
#[inline(always)]
fn encode_quad_non_initial_row(
    row: NonInitialQuadRow<'_>,
    mel: &mut MelEncoder,
    vlc: &mut PairVlc,
    ms: &mut MagSgnEncoder,
) -> Result<i32, &'static str> {
    let NonInitialQuadRow {
        offset,
        c_q,
        rho,
        e_qmax,
        max_e,
        e_q,
        s,
    } = row;
    let kappa = if (rho & (rho - 1)) != 0 {
        max_e.max(1)
    } else {
        1
    };
    let u_q = e_qmax.max(kappa) - kappa;
    let eps = quad_eps(e_q, offset, e_qmax, u_q);

    let tuple = HT_VLC_ENCODE_TABLE1[(c_q << 8) | ((rho as usize) << 4) | eps as usize];
    vlc.push(u32::from(tuple >> 8), ((tuple >> 4) & 0x7) as u8);

    if c_q == 0 {
        mel.encode(rho != 0)?;
    }

    encode_mag_signs(rho, e_qmax.max(kappa), tuple, s, offset, ms)?;
    Ok(u_q)
}

/// The quad's `eps` mask: which samples reach `e_qmax`, or none when `u_q`
/// is 0. Computed without branches, since `u_q > 0` is data dependent.
#[expect(clippy::inline_always, reason = "runs once per quad")]
#[inline(always)]
fn quad_eps(e_q: &[i32; 8], offset: usize, e_qmax: i32, u_q: i32) -> u16 {
    let reaches = |sample: usize| u16::from(e_q[offset + sample] == e_qmax) << sample;
    let eps = reaches(0) | reaches(1) | reaches(2) | reaches(3);
    eps & (u16::from(u_q > 0) * 0xF)
}

#[expect(
    clippy::cast_sign_loss,
    clippy::inline_always,
    reason = "HT table exponents and Uq values are nonnegative bounded bit counts in this per-sample hot path"
)]
#[inline(always)]
fn encode_mag_signs(
    rho: i32,
    u_q: i32,
    tuple: u16,
    s: &[u32; 8],
    offset: usize,
    ms: &mut MagSgnEncoder,
) -> Result<(), &'static str> {
    let e_k = u32::from(tuple & 0xF);
    // Payloads go out least significant first, as separate writes would. An
    // insignificant sample has a 0-bit payload, so packing needs no branches.
    // U <= 32 bounds a payload; with U <= 16 the quad fits one 64-bit word.
    let pack = |first: usize, count: usize| {
        let (mut word, mut word_bits) = (0_u64, 0_u32);
        for sample in first..first + count {
            let significant = ((rho >> sample) & 1) as u32;
            let magnitude_bits = significant * (u_q as u32 - ((e_k >> sample) & 1));
            let payload = u64::from(s[offset + sample]) & ((1_u64 << magnitude_bits) - 1);
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
fn encode_uvlc_non_initial(u_q0: i32, u_q1: i32, vlc: &mut PairVlc) {
    let first = HT_UVLC_ENCODE_TABLE[u_q0.max(0) as usize];
    let second = HT_UVLC_ENCODE_TABLE[u_q1.max(0) as usize];
    encode_uvlc_pair(vlc, first, second);
}

fn encode_uvlc_pair(vlc: &mut PairVlc, first: HtUvlcTableEntry, second: HtUvlcTableEntry) {
    vlc.push(u32::from(first.pre), first.pre_len);
    vlc.push(u32::from(second.pre), second.pre_len);
    vlc.push(u32::from(first.suf), first.suf_len);
    vlc.push(u32::from(second.suf), second.suf_len);
}
