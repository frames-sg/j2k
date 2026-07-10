// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::ht_encode_tables::{
    HtUvlcTableEntry, HT_UVLC_ENCODE_TABLE, HT_VLC_ENCODE_TABLE0, HT_VLC_ENCODE_TABLE1,
};
use super::cleanup::CleanupCoefficientSource;
use super::quad::{
    first_quad_pair, non_initial_quad_pair, FirstQuadPairRequest, InitialQuadRow,
    NonInitialQuadPairRequest, NonInitialQuadRow, QuadSink,
};
use super::writers::{MagSgnEncoder, MelEncoder, VlcEncoder};

struct EncodeQuadSink<'a> {
    mel: &'a mut MelEncoder,
    vlc: &'a mut VlcEncoder,
    ms: &'a mut MagSgnEncoder,
}

impl QuadSink for EncodeQuadSink<'_> {
    type Error = &'static str;

    #[expect(clippy::inline_always, reason = "erase encoder sink dispatch")]
    #[inline(always)]
    fn quad_initial(&mut self, row: InitialQuadRow<'_>) -> Result<i32, Self::Error> {
        encode_quad_initial_row(row, self.mel, self.vlc, self.ms)
    }

    #[expect(clippy::inline_always, reason = "erase encoder sink dispatch")]
    #[inline(always)]
    fn quad_non_initial(&mut self, row: NonInitialQuadRow<'_>) -> Result<i32, Self::Error> {
        encode_quad_non_initial_row(row, self.mel, self.vlc, self.ms)
    }

    #[expect(clippy::inline_always, reason = "erase encoder sink dispatch")]
    #[inline(always)]
    fn initial_uvlc_pair(&mut self, u_q0: i32, u_q1: i32) -> Result<(), Self::Error> {
        if u_q0 > 0 && u_q1 > 0 {
            self.mel.encode(u_q0.min(u_q1) > 2)?;
        }
        encode_uvlc(u_q0, u_q1, self.vlc)
    }

    #[expect(clippy::inline_always, reason = "erase encoder sink dispatch")]
    #[inline(always)]
    fn initial_uvlc_lone(&mut self, u_q0: i32) -> Result<(), Self::Error> {
        encode_uvlc(u_q0, 0, self.vlc)
    }

    #[expect(clippy::inline_always, reason = "erase encoder sink dispatch")]
    #[inline(always)]
    fn non_initial_uvlc(&mut self, u_q0: i32, u_q1: i32) -> Result<(), Self::Error> {
        encode_uvlc_non_initial(u_q0, u_q1, self.vlc)
    }
}

#[expect(clippy::inline_always, reason = "fuse the encoder quad walker")]
#[inline(always)]
pub(super) fn encode_first_quad_pair<C: CleanupCoefficientSource + ?Sized>(
    request: FirstQuadPairRequest<'_, C>,
    mel: &mut MelEncoder,
    vlc: &mut VlcEncoder,
    ms: &mut MagSgnEncoder,
) -> Result<(), &'static str> {
    first_quad_pair(request, &mut EncodeQuadSink { mel, vlc, ms })
}

#[expect(clippy::inline_always, reason = "fuse the encoder quad walker")]
#[inline(always)]
pub(super) fn encode_non_initial_quad_pair<C: CleanupCoefficientSource + ?Sized>(
    request: NonInitialQuadPairRequest<'_, C>,
    mel: &mut MelEncoder,
    vlc: &mut VlcEncoder,
    ms: &mut MagSgnEncoder,
) -> Result<(), &'static str> {
    non_initial_quad_pair(request, &mut EncodeQuadSink { mel, vlc, ms })
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
    vlc: &mut VlcEncoder,
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
    let mut eps = 0u16;

    if u_q > 0 {
        eps |= u16::from(u8::from(e_q[offset] == e_qmax));
        eps |= u16::from(u8::from(e_q[offset + 1] == e_qmax)) << 1;
        eps |= u16::from(u8::from(e_q[offset + 2] == e_qmax)) << 2;
        eps |= u16::from(u8::from(e_q[offset + 3] == e_qmax)) << 3;
    }

    e_val[lep] = e_val[lep].max(e_q[offset + 1] as u8);
    e_val[lep + 1] = e_q[offset + 3] as u8;
    cx_val[lcxp] |= ((rho & 2) >> 1) as u8;
    cx_val[lcxp + 1] = ((rho & 8) >> 3) as u8;

    let tuple = HT_VLC_ENCODE_TABLE0[(c_q << 8) | ((rho as usize) << 4) | eps as usize];
    vlc.encode(u32::from(tuple >> 8), ((tuple >> 4) & 0x7) as u8)?;

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
    vlc: &mut VlcEncoder,
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
    let mut eps = 0u16;

    if u_q > 0 {
        eps |= u16::from(u8::from(e_q[offset] == e_qmax));
        eps |= u16::from(u8::from(e_q[offset + 1] == e_qmax)) << 1;
        eps |= u16::from(u8::from(e_q[offset + 2] == e_qmax)) << 2;
        eps |= u16::from(u8::from(e_q[offset + 3] == e_qmax)) << 3;
    }

    let tuple = HT_VLC_ENCODE_TABLE1[(c_q << 8) | ((rho as usize) << 4) | eps as usize];
    vlc.encode(u32::from(tuple >> 8), ((tuple >> 4) & 0x7) as u8)?;

    if c_q == 0 {
        mel.encode(rho != 0)?;
    }

    encode_mag_signs(rho, e_qmax.max(kappa), tuple, s, offset, ms)?;
    Ok(u_q)
}

#[expect(
    clippy::cast_possible_wrap,
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
    let e_k = tuple & 0xF;
    let mut encode = |bit: i32, shift: u32, sample_offset: usize| -> Result<(), &'static str> {
        let sample_mask = 1 << bit;
        if (rho & sample_mask) == 0 {
            return Ok(());
        }

        let reduction = ((u32::from(e_k) >> shift) & 1) as i32;
        let magnitude_bits = (u_q - reduction) as u32;
        let payload = match magnitude_bits {
            0 => 0,
            32.. => s[offset + sample_offset],
            bits => s[offset + sample_offset] & ((1u32 << bits) - 1),
        };
        ms.encode(payload, magnitude_bits)
    };

    encode(0, 0, 0)?;
    encode(1, 1, 1)?;
    encode(2, 2, 2)?;
    encode(3, 3, 3)?;

    Ok(())
}

#[expect(
    clippy::cast_sign_loss,
    reason = "branch guards and max operations make every UVLC table index and suffix nonnegative"
)]
fn encode_uvlc(u_q0: i32, u_q1: i32, vlc: &mut VlcEncoder) -> Result<(), &'static str> {
    if u_q0 > 2 && u_q1 > 2 {
        let first = HT_UVLC_ENCODE_TABLE[(u_q0 - 2) as usize];
        let second = HT_UVLC_ENCODE_TABLE[(u_q1 - 2) as usize];
        encode_uvlc_pair(vlc, first, second)
    } else if u_q0 > 2 && u_q1 > 0 {
        let first = HT_UVLC_ENCODE_TABLE[u_q0 as usize];
        vlc.encode(u32::from(first.pre), first.pre_len)?;
        vlc.encode((u_q1 - 1) as u32, 1)?;
        vlc.encode(u32::from(first.suf), first.suf_len)
    } else {
        let first = HT_UVLC_ENCODE_TABLE[u_q0.max(0) as usize];
        let second = HT_UVLC_ENCODE_TABLE[u_q1.max(0) as usize];
        encode_uvlc_pair(vlc, first, second)
    }
}

#[expect(
    clippy::cast_sign_loss,
    reason = "max with zero makes both non-initial UVLC table indices nonnegative"
)]
fn encode_uvlc_non_initial(u_q0: i32, u_q1: i32, vlc: &mut VlcEncoder) -> Result<(), &'static str> {
    let first = HT_UVLC_ENCODE_TABLE[u_q0.max(0) as usize];
    let second = HT_UVLC_ENCODE_TABLE[u_q1.max(0) as usize];
    encode_uvlc_pair(vlc, first, second)
}

fn encode_uvlc_pair(
    vlc: &mut VlcEncoder,
    first: HtUvlcTableEntry,
    second: HtUvlcTableEntry,
) -> Result<(), &'static str> {
    vlc.encode(u32::from(first.pre), first.pre_len)?;
    vlc.encode(u32::from(second.pre), second.pre_len)?;
    vlc.encode(u32::from(first.suf), first.suf_len)?;
    vlc.encode(u32::from(second.suf), second.suf_len)
}
