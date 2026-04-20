//! HTJ2K block encoder — port of `ojph_block_encoder.cpp`.
//!
//! Encodes a codeblock of samples (u32 or u64) into an HTJ2K-compliant coded
//! byte buffer consisting of three concatenated segments: MagSgn + MEL + VLC.

use crate::arch::{count_leading_zeros, count_leading_zeros_u64};
use crate::error::{OjphError, Result};
use crate::types::{ojph_max, ojph_min};

use super::common::{encoder_tables, UvlcEncEntry};

// ---------------------------------------------------------------------------
// Result type
// ---------------------------------------------------------------------------

/// Output of a codeblock encoding operation.
pub(crate) struct EncodeResult {
    /// Coded byte buffer (MagSgn ‖ MEL ‖ VLC).
    pub data: Vec<u8>,
    /// Total number of valid bytes in `data`.
    pub length: u32,
}

// ---------------------------------------------------------------------------
// MEL encoder
// ---------------------------------------------------------------------------

/// MEL exponent table (Table A.6 in the spec).
const MEL_EXP: [i32; 13] = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 5];

/// Modified Exponential-Golomb run-length encoder.
struct MelEncoder {
    buf: Vec<u8>,
    pos: u32,
    remaining_bits: i32,
    tmp: i32,
    run: i32,
    k: i32,
    threshold: i32,
}

impl MelEncoder {
    fn new(size: usize) -> Self {
        Self {
            buf: vec![0u8; size],
            pos: 0,
            remaining_bits: 8,
            tmp: 0,
            run: 0,
            k: 0,
            threshold: 1, // 1 << mel_exp[0]
        }
    }

    /// Emits a single bit into the MEL bitstream with 0xFF byte stuffing.
    #[inline]
    fn emit_bit(&mut self, v: i32) -> Result<()> {
        debug_assert!(v == 0 || v == 1);
        self.tmp = (self.tmp << 1) + v;
        self.remaining_bits -= 1;
        if self.remaining_bits == 0 {
            if self.pos as usize >= self.buf.len() {
                return Err(OjphError::Codec {
                    code: 0x00020001,
                    message: "mel encoder's buffer is full".into(),
                });
            }
            self.buf[self.pos as usize] = self.tmp as u8;
            self.pos += 1;
            self.remaining_bits = if self.tmp == 0xFF { 7 } else { 8 };
            self.tmp = 0;
        }
        Ok(())
    }

    /// Encodes a single MEL event (significance bit).
    #[inline]
    fn encode(&mut self, bit: bool) -> Result<()> {
        if !bit {
            self.run += 1;
            if self.run >= self.threshold {
                self.emit_bit(1)?;
                self.run = 0;
                self.k = ojph_min(12, self.k + 1);
                self.threshold = 1 << MEL_EXP[self.k as usize];
            }
        } else {
            self.emit_bit(0)?;
            let mut t = MEL_EXP[self.k as usize];
            while t > 0 {
                t -= 1;
                self.emit_bit((self.run >> t) & 1)?;
            }
            self.run = 0;
            self.k = ojph_max(0, self.k - 1);
            self.threshold = 1 << MEL_EXP[self.k as usize];
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// VLC encoder (backward-growing bitstream)
// ---------------------------------------------------------------------------

/// Variable-Length Code encoder that writes backward from the end of its
/// buffer, with byte-stuffing to prevent 0xFF bytes in the output.
struct VlcEncoder {
    buf: Vec<u8>,
    pos: u32,
    used_bits: i32,
    tmp: i32,
    last_greater_than_8f: bool,
}

impl VlcEncoder {
    fn new(size: usize) -> Self {
        let mut buf = vec![0u8; size];
        // buf[size-1] is the logical "first" byte; it is pre-set to 0xFF.
        buf[size - 1] = 0xFF;
        Self {
            buf,
            pos: 1,
            used_bits: 4,
            tmp: 0x0F,
            last_greater_than_8f: true,
        }
    }

    /// Encodes `cwd_len` bits of `cwd` into the backward VLC stream.
    #[inline]
    fn encode(&mut self, mut cwd: i32, mut cwd_len: i32) -> Result<()> {
        while cwd_len > 0 {
            if self.pos as usize >= self.buf.len() {
                return Err(OjphError::Codec {
                    code: 0x00020002,
                    message: "vlc encoder's buffer is full".into(),
                });
            }

            let avail_bits = 8 - (self.last_greater_than_8f as i32) - self.used_bits;
            let t = ojph_min(avail_bits, cwd_len);
            self.tmp |= (cwd & ((1 << t) - 1)) << self.used_bits;
            self.used_bits += t;
            let avail_bits = avail_bits - t;
            cwd_len -= t;
            cwd >>= t;
            if avail_bits == 0 {
                if self.last_greater_than_8f && self.tmp != 0x7F {
                    self.last_greater_than_8f = false;
                    continue; // one empty bit remaining
                }
                let write_pos = self.buf.len() - 1 - self.pos as usize;
                self.buf[write_pos] = self.tmp as u8;
                self.pos += 1;
                self.last_greater_than_8f = self.tmp > 0x8F;
                self.tmp = 0;
                self.used_bits = 0;
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// MagSgn encoder (forward-growing bitstream)
// ---------------------------------------------------------------------------

/// Magnitude-Sign encoder with 0xFF byte stuffing.
struct MsEncoder {
    buf: Vec<u8>,
    pos: u32,
    max_bits: i32,
    used_bits: i32,
    tmp: u32,
}

impl MsEncoder {
    fn new(size: usize) -> Self {
        Self {
            buf: vec![0u8; size],
            pos: 0,
            max_bits: 8,
            used_bits: 0,
            tmp: 0,
        }
    }

    /// Encodes up to 32 bits of magnitude-sign data.
    #[inline]
    fn encode(&mut self, mut cwd: u32, mut cwd_len: i32) -> Result<()> {
        while cwd_len > 0 {
            if self.pos as usize >= self.buf.len() {
                return Err(OjphError::Codec {
                    code: 0x00020005,
                    message: "magnitude sign encoder's buffer is full".into(),
                });
            }
            let t = ojph_min(self.max_bits - self.used_bits, cwd_len);
            self.tmp |= (cwd & ((1u32 << t) - 1)) << self.used_bits;
            self.used_bits += t;
            cwd >>= t;
            cwd_len -= t;
            if self.used_bits >= self.max_bits {
                self.buf[self.pos as usize] = self.tmp as u8;
                self.pos += 1;
                self.max_bits = if self.tmp == 0xFF { 7 } else { 8 };
                self.tmp = 0;
                self.used_bits = 0;
            }
        }
        Ok(())
    }

    /// Encodes up to 64 bits of magnitude-sign data (for the 64-bit path).
    #[allow(dead_code)]
    #[inline]
    fn encode64(&mut self, mut cwd: u64, mut cwd_len: i32) -> Result<()> {
        while cwd_len > 0 {
            if self.pos as usize >= self.buf.len() {
                return Err(OjphError::Codec {
                    code: 0x00020005,
                    message: "magnitude sign encoder's buffer is full".into(),
                });
            }
            let t = ojph_min(self.max_bits - self.used_bits, cwd_len);
            self.tmp |= ((cwd & ((1u64 << t) - 1)) << self.used_bits) as u32;
            self.used_bits += t;
            cwd >>= t;
            cwd_len -= t;
            if self.used_bits >= self.max_bits {
                self.buf[self.pos as usize] = self.tmp as u8;
                self.pos += 1;
                self.max_bits = if self.tmp == 0xFF { 7 } else { 8 };
                self.tmp = 0;
                self.used_bits = 0;
            }
        }
        Ok(())
    }

    /// Flushes the MagSgn encoder, padding remaining bits.
    #[inline]
    fn terminate(&mut self) -> Result<()> {
        if self.used_bits != 0 {
            let t = self.max_bits - self.used_bits;
            self.tmp |= (0xFF & ((1u32 << t) - 1)) << self.used_bits;
            self.used_bits += t;
            if self.tmp != 0xFF {
                if self.pos as usize >= self.buf.len() {
                    return Err(OjphError::Codec {
                        code: 0x00020006,
                        message: "magnitude sign encoder's buffer is full".into(),
                    });
                }
                self.buf[self.pos as usize] = self.tmp as u8;
                self.pos += 1;
            }
        } else if self.max_bits == 7 {
            self.pos -= 1;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// MEL/VLC termination & merge
// ---------------------------------------------------------------------------

/// Terminates the MEL and VLC encoders and fuses the final byte when possible.
fn terminate_mel_vlc(mel: &mut MelEncoder, vlc: &mut VlcEncoder) -> Result<()> {
    if mel.run > 0 {
        mel.emit_bit(1)?;
    }

    mel.tmp <<= mel.remaining_bits;
    let mel_mask = (0xFF << mel.remaining_bits) & 0xFF;
    let vlc_mask = 0xFF >> (8 - vlc.used_bits);

    if (mel_mask | vlc_mask) == 0 {
        return Ok(());
    }

    if mel.pos as usize >= mel.buf.len() {
        return Err(OjphError::Codec {
            code: 0x00020003,
            message: "mel encoder's buffer is full".into(),
        });
    }

    let fuse = mel.tmp | vlc.tmp;
    if (((fuse ^ mel.tmp) & mel_mask) | ((fuse ^ vlc.tmp) & vlc_mask)) == 0
        && fuse != 0xFF
        && vlc.pos > 1
    {
        mel.buf[mel.pos as usize] = fuse as u8;
        mel.pos += 1;
    } else {
        if vlc.pos as usize >= vlc.buf.len() {
            return Err(OjphError::Codec {
                code: 0x00020004,
                message: "vlc encoder's buffer is full".into(),
            });
        }
        // mel.tmp cannot be 0xFF
        mel.buf[mel.pos as usize] = mel.tmp as u8;
        mel.pos += 1;
        let write_pos = vlc.buf.len() - 1 - vlc.pos as usize;
        vlc.buf[write_pos] = vlc.tmp as u8;
        vlc.pos += 1;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Output assembly helper
// ---------------------------------------------------------------------------

/// Assembles the final coded buffer from the three sub-streams.
///
/// Layout: `[MS bytes] [MEL bytes] [VLC bytes (reversed)]`
/// The last 12 bits of the buffer carry the interface-locator word.
fn assemble_output(ms: &MsEncoder, mel: &MelEncoder, vlc: &VlcEncoder) -> EncodeResult {
    let total_len = (ms.pos + mel.pos + vlc.pos) as usize;
    let mut data = vec![0u8; total_len];

    // MagSgn data (forward)
    data[..ms.pos as usize].copy_from_slice(&ms.buf[..ms.pos as usize]);

    // MEL data (forward)
    let mel_start = ms.pos as usize;
    data[mel_start..mel_start + mel.pos as usize].copy_from_slice(&mel.buf[..mel.pos as usize]);

    // VLC data (the backward buffer stores bytes in reverse order)
    let vlc_start = mel_start + mel.pos as usize;
    let vlc_buf_end = vlc.buf.len() - 1; // logical position 0
    data[vlc_start..vlc_start + vlc.pos as usize]
        .copy_from_slice(&vlc.buf[vlc_buf_end + 1 - vlc.pos as usize..=vlc_buf_end]);

    // Interface locator word (12 bits at end of buffer)
    let num_bytes = mel.pos + vlc.pos;
    if total_len >= 2 {
        data[total_len - 1] = (num_bytes >> 4) as u8;
        data[total_len - 2] = (data[total_len - 2] & 0xF0) | (num_bytes & 0xF) as u8;
    }

    EncodeResult {
        data,
        length: total_len as u32,
    }
}

// ---------------------------------------------------------------------------
// 32-bit sample preparation helper
// ---------------------------------------------------------------------------

/// Extracts significance, exponent, and v_n from a single 32-bit sample.
///
/// Returns `(is_significant, e_q, v_n)`.
#[inline]
fn prepare_sample32(t: u32, p: u32) -> (bool, i32, u32) {
    let mut val = t.wrapping_add(t); // multiply by 2, discard sign
    val >>= p;
    val &= !1u32; // 2 * mu_p
    if val != 0 {
        val -= 1; // 2*mu_p - 1
        let e_q = 32 - count_leading_zeros(val) as i32;
        val -= 1;
        let s = val.wrapping_add(t >> 31); // v_n = 2*(mu_p-1) + sign
        (true, e_q, s)
    } else {
        (false, 0, 0)
    }
}

/// Extracts significance, exponent, and v_n from a single 64-bit sample.
#[allow(dead_code)]
#[inline]
fn prepare_sample64(t: u64, p: u32) -> (bool, i32, u64) {
    let mut val = t.wrapping_add(t);
    val >>= p;
    val &= !1u64;
    if val != 0 {
        val -= 1;
        let e_q = 64 - count_leading_zeros_u64(val) as i32;
        val -= 1;
        let s = val.wrapping_add(t >> 63);
        (true, e_q, s)
    } else {
        (false, 0, 0)
    }
}

// ---------------------------------------------------------------------------
// Per-quad MagSgn encoding helpers
// ---------------------------------------------------------------------------

/// Encodes the four MagSgn values for a quad (32-bit path).
#[inline]
fn encode_quad_ms32(
    ms: &mut MsEncoder,
    rho: i32,
    uq: i32,
    tuple: u16,
    s: &[u32; 8],
    base: usize,
) -> Result<()> {
    for bit_idx in 0..4 {
        let bit = 1 << bit_idx;
        let m = if (rho & bit) != 0 {
            uq - (((tuple as i32) & bit) >> bit_idx)
        } else {
            0
        };
        ms.encode(s[base + bit_idx] & ((1u32 << m) - 1), m)?;
    }
    Ok(())
}

/// Encodes the four MagSgn values for a quad (64-bit path).
#[allow(dead_code)]
#[inline]
fn encode_quad_ms64(
    ms: &mut MsEncoder,
    rho: i32,
    uq: i32,
    tuple: u16,
    s: &[u64; 8],
    base: usize,
) -> Result<()> {
    for bit_idx in 0..4 {
        let bit = 1 << bit_idx;
        let m = if (rho & bit) != 0 {
            uq - (((tuple as i32) & bit) >> bit_idx)
        } else {
            0
        };
        ms.encode64(s[base + bit_idx] & ((1u64 << m) - 1), m)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Epsilon (EMB) computation helper
// ---------------------------------------------------------------------------

/// Computes the EMB (epsilon) bitmask for a quad.
#[inline]
fn compute_eps(u_q: i32, e_q: &[i32], base: usize, e_qmax: i32) -> i32 {
    if u_q > 0 {
        let mut eps = 0;
        eps |= (e_q[base] == e_qmax) as i32;
        eps |= ((e_q[base + 1] == e_qmax) as i32) << 1;
        eps |= ((e_q[base + 2] == e_qmax) as i32) << 2;
        eps |= ((e_q[base + 3] == e_qmax) as i32) << 3;
        eps
    } else {
        0
    }
}

// ---------------------------------------------------------------------------
// Public encoding functions
// ---------------------------------------------------------------------------

/// Encodes a 32-bit codeblock into an HTJ2K coded byte buffer.
///
/// # Arguments
///
/// * `buf`          — sample buffer (row-major, `stride` elements per row)
/// * `missing_msbs` — number of missing most-significant bit-planes
/// * `num_passes`   — always 1 for HTJ2K
/// * `width`        — codeblock width in samples
/// * `height`       — codeblock height in samples
/// * `stride`       — number of elements between consecutive rows
pub(crate) fn encode_codeblock32(
    buf: &[u32],
    missing_msbs: u32,
    num_passes: u32,
    width: u32,
    height: u32,
    stride: u32,
) -> Result<EncodeResult> {
    debug_assert_eq!(num_passes, 1);
    let _ = num_passes;

    let tables = encoder_tables();
    let vlc_tbl0 = &tables.vlc_tbl0;
    let vlc_tbl1 = &tables.vlc_tbl1;
    let uvlc_tbl = &tables.uvlc_tbl;

    // Buffer sizes (generous upper bounds)
    const MS_SIZE: usize = (16384_usize * 16).div_ceil(15);
    const MEL_VLC_SIZE: usize = 3072;
    const MEL_SIZE: usize = 192;
    const VLC_SIZE: usize = MEL_VLC_SIZE - MEL_SIZE;

    let mut mel = MelEncoder::new(MEL_SIZE);
    let mut vlc = VlcEncoder::new(VLC_SIZE);
    let mut ms = MsEncoder::new(MS_SIZE);

    let p = 30 - missing_msbs;

    // Per-quad-row state: e_val stores max E for bottom samples,
    // cx_val stores context (significance) from the row above.
    let mut e_val = vec![0u8; 513];
    let mut cx_val = vec![0u8; 513];

    // -----------------------------------------------------------------------
    // Initial row of quads (y = 0)
    // -----------------------------------------------------------------------
    {
        let mut c_q0: i32 = 0;

        let mut lep: usize = 0; // index into e_val
        let mut lcxp: usize = 0; // index into cx_val
        e_val[0] = 0;
        cx_val[0] = 0;

        let mut sp: usize = 0; // index into buf (row 0)

        let mut x: u32 = 0;
        while x < width {
            // Reset per-pair state
            let mut e_qmax = [0i32; 2];
            let mut e_q = [0i32; 8];
            let mut rho = [0i32; 2];
            let mut s = [0u32; 8];

            // --- Quad 0: samples (x, 0), (x, 1), (x+1, 0), (x+1, 1) ---

            // sample (x, row 0)
            let t = buf[sp];
            let (sig, eq, sv) = prepare_sample32(t, p);
            if sig {
                rho[0] = 1;
                e_q[0] = eq;
                e_qmax[0] = eq;
                s[0] = sv;
            }

            // sample (x, row 1)
            let t = if height > 1 {
                buf[sp + stride as usize]
            } else {
                0
            };
            sp += 1;
            let (sig, eq, sv) = prepare_sample32(t, p);
            if sig {
                rho[0] += 2;
                e_q[1] = eq;
                e_qmax[0] = ojph_max(e_qmax[0], eq);
                s[1] = sv;
            }

            if x + 1 < width {
                // sample (x+1, row 0)
                let t = buf[sp];
                let (sig, eq, sv) = prepare_sample32(t, p);
                if sig {
                    rho[0] += 4;
                    e_q[2] = eq;
                    e_qmax[0] = ojph_max(e_qmax[0], eq);
                    s[2] = sv;
                }

                // sample (x+1, row 1)
                let t = if height > 1 {
                    buf[sp + stride as usize]
                } else {
                    0
                };
                sp += 1;
                let (sig, eq, sv) = prepare_sample32(t, p);
                if sig {
                    rho[0] += 8;
                    e_q[3] = eq;
                    e_qmax[0] = ojph_max(e_qmax[0], eq);
                    s[3] = sv;
                }
            }

            let uq0 = ojph_max(e_qmax[0], 1); // kappa = 1
            let u_q0 = uq0 - 1;
            let mut u_q1: i32 = 0;

            let eps0 = compute_eps(u_q0, &e_q, 0, e_qmax[0]);

            e_val[lep] = ojph_max(e_val[lep] as i32, e_q[1]) as u8;
            lep += 1;
            e_val[lep] = e_q[3] as u8;
            cx_val[lcxp] |= ((rho[0] & 2) >> 1) as u8;
            lcxp += 1;
            cx_val[lcxp] = ((rho[0] & 8) >> 3) as u8;

            let tuple0 = vlc_tbl0[((c_q0 << 8) + (rho[0] << 4) + eps0) as usize];
            vlc.encode((tuple0 >> 8) as i32, ((tuple0 >> 4) & 7) as i32)?;

            if c_q0 == 0 {
                mel.encode(rho[0] != 0)?;
            }

            encode_quad_ms32(&mut ms, rho[0], uq0, tuple0, &s, 0)?;

            // --- Quad 1: samples (x+2, 0), (x+2, 1), (x+3, 0), (x+3, 1) ---
            if x + 2 < width {
                let t = buf[sp];
                let (sig, eq, sv) = prepare_sample32(t, p);
                if sig {
                    rho[1] = 1;
                    e_q[4] = eq;
                    e_qmax[1] = eq;
                    s[4] = sv;
                }

                let t = if height > 1 {
                    buf[sp + stride as usize]
                } else {
                    0
                };
                sp += 1;
                let (sig, eq, sv) = prepare_sample32(t, p);
                if sig {
                    rho[1] += 2;
                    e_q[5] = eq;
                    e_qmax[1] = ojph_max(e_qmax[1], eq);
                    s[5] = sv;
                }

                if x + 3 < width {
                    let t = buf[sp];
                    let (sig, eq, sv) = prepare_sample32(t, p);
                    if sig {
                        rho[1] += 4;
                        e_q[6] = eq;
                        e_qmax[1] = ojph_max(e_qmax[1], eq);
                        s[6] = sv;
                    }

                    let t = if height > 1 {
                        buf[sp + stride as usize]
                    } else {
                        0
                    };
                    sp += 1;
                    let (sig, eq, sv) = prepare_sample32(t, p);
                    if sig {
                        rho[1] += 8;
                        e_q[7] = eq;
                        e_qmax[1] = ojph_max(e_qmax[1], eq);
                        s[7] = sv;
                    }
                }

                let c_q1 = (rho[0] >> 1) | (rho[0] & 1);
                let uq1 = ojph_max(e_qmax[1], 1); // kappa = 1
                u_q1 = uq1 - 1;

                let eps1 = compute_eps(u_q1, &e_q, 4, e_qmax[1]);

                e_val[lep] = ojph_max(e_val[lep] as i32, e_q[5]) as u8;
                lep += 1;
                e_val[lep] = e_q[7] as u8;
                cx_val[lcxp] |= ((rho[1] & 2) >> 1) as u8;
                lcxp += 1;
                cx_val[lcxp] = ((rho[1] & 8) >> 3) as u8;

                let tuple1 = vlc_tbl0[((c_q1 << 8) + (rho[1] << 4) + eps1) as usize];
                vlc.encode((tuple1 >> 8) as i32, ((tuple1 >> 4) & 7) as i32)?;

                if c_q1 == 0 {
                    mel.encode(rho[1] != 0)?;
                }

                encode_quad_ms32(&mut ms, rho[1], uq1, tuple1, &s, 4)?;
            }

            // UVLC encoding
            if u_q0 > 0 && u_q1 > 0 {
                mel.encode(ojph_min(u_q0, u_q1) > 2)?;
            }

            encode_uvlc32(&mut vlc, uvlc_tbl, u_q0, u_q1)?;

            // Prepare for next quad pair
            c_q0 = (rho[1] >> 1) | (rho[1] & 1);

            x += 4;
        }

        e_val[lep + 1] = 0;
    }

    // -----------------------------------------------------------------------
    // Non-initial quad rows (y = 2, 4, 6, ...)
    // -----------------------------------------------------------------------
    {
        let mut y: u32 = 2;
        while y < height {
            let mut lep: usize = 0;
            let mut max_e = ojph_max(e_val[0] as i32, e_val[1] as i32) - 1;
            e_val[0] = 0;

            let mut lcxp: usize = 0;
            let mut c_q0: i32 = cx_val[0] as i32 + ((cx_val[1] as i32) << 2);
            cx_val[0] = 0;

            let row_offset = (y * stride) as usize;

            let mut sp: usize = row_offset;

            let mut x: u32 = 0;
            while x < width {
                let mut e_qmax = [0i32; 2];
                let mut e_q = [0i32; 8];
                let mut rho = [0i32; 2];
                let mut s = [0u32; 8];

                // --- Quad 0 ---
                let t = buf[sp];
                let (sig, eq, sv) = prepare_sample32(t, p);
                if sig {
                    rho[0] = 1;
                    e_q[0] = eq;
                    e_qmax[0] = eq;
                    s[0] = sv;
                }

                let t = if y + 1 < height {
                    buf[sp + stride as usize]
                } else {
                    0
                };
                sp += 1;
                let (sig, eq, sv) = prepare_sample32(t, p);
                if sig {
                    rho[0] += 2;
                    e_q[1] = eq;
                    e_qmax[0] = ojph_max(e_qmax[0], eq);
                    s[1] = sv;
                }

                if x + 1 < width {
                    let t = buf[sp];
                    let (sig, eq, sv) = prepare_sample32(t, p);
                    if sig {
                        rho[0] += 4;
                        e_q[2] = eq;
                        e_qmax[0] = ojph_max(e_qmax[0], eq);
                        s[2] = sv;
                    }

                    let t = if y + 1 < height {
                        buf[sp + stride as usize]
                    } else {
                        0
                    };
                    sp += 1;
                    let (sig, eq, sv) = prepare_sample32(t, p);
                    if sig {
                        rho[0] += 8;
                        e_q[3] = eq;
                        e_qmax[0] = ojph_max(e_qmax[0], eq);
                        s[3] = sv;
                    }
                }

                let kappa = if (rho[0] & (rho[0] - 1)) != 0 {
                    ojph_max(1, max_e)
                } else {
                    1
                };
                let uq0 = ojph_max(e_qmax[0], kappa);
                let u_q0 = uq0 - kappa;
                let mut u_q1: i32 = 0;

                let eps0 = compute_eps(u_q0, &e_q, 0, e_qmax[0]);

                e_val[lep] = ojph_max(e_val[lep] as i32, e_q[1]) as u8;
                lep += 1;
                max_e = ojph_max(e_val[lep] as i32, e_val[lep + 1] as i32) - 1;
                e_val[lep] = e_q[3] as u8;

                cx_val[lcxp] |= ((rho[0] & 2) >> 1) as u8;
                lcxp += 1;
                let mut c_q1: i32 = cx_val[lcxp] as i32 + ((cx_val[lcxp + 1] as i32) << 2);
                cx_val[lcxp] = ((rho[0] & 8) >> 3) as u8;

                let tuple0 = vlc_tbl1[((c_q0 << 8) + (rho[0] << 4) + eps0) as usize];
                vlc.encode((tuple0 >> 8) as i32, ((tuple0 >> 4) & 7) as i32)?;

                if c_q0 == 0 {
                    mel.encode(rho[0] != 0)?;
                }

                encode_quad_ms32(&mut ms, rho[0], uq0, tuple0, &s, 0)?;

                // --- Quad 1 ---
                if x + 2 < width {
                    let t = buf[sp];
                    let (sig, eq, sv) = prepare_sample32(t, p);
                    if sig {
                        rho[1] = 1;
                        e_q[4] = eq;
                        e_qmax[1] = eq;
                        s[4] = sv;
                    }

                    let t = if y + 1 < height {
                        buf[sp + stride as usize]
                    } else {
                        0
                    };
                    sp += 1;
                    let (sig, eq, sv) = prepare_sample32(t, p);
                    if sig {
                        rho[1] += 2;
                        e_q[5] = eq;
                        e_qmax[1] = ojph_max(e_qmax[1], eq);
                        s[5] = sv;
                    }

                    if x + 3 < width {
                        let t = buf[sp];
                        let (sig, eq, sv) = prepare_sample32(t, p);
                        if sig {
                            rho[1] += 4;
                            e_q[6] = eq;
                            e_qmax[1] = ojph_max(e_qmax[1], eq);
                            s[6] = sv;
                        }

                        let t = if y + 1 < height {
                            buf[sp + stride as usize]
                        } else {
                            0
                        };
                        sp += 1;
                        let (sig, eq, sv) = prepare_sample32(t, p);
                        if sig {
                            rho[1] += 8;
                            e_q[7] = eq;
                            e_qmax[1] = ojph_max(e_qmax[1], eq);
                            s[7] = sv;
                        }
                    }

                    let kappa = if (rho[1] & (rho[1] - 1)) != 0 {
                        ojph_max(1, max_e)
                    } else {
                        1
                    };
                    c_q1 |= ((rho[0] & 4) >> 1) | ((rho[0] & 8) >> 2);
                    let uq1 = ojph_max(e_qmax[1], kappa);
                    u_q1 = uq1 - kappa;

                    let eps1 = compute_eps(u_q1, &e_q, 4, e_qmax[1]);

                    e_val[lep] = ojph_max(e_val[lep] as i32, e_q[5]) as u8;
                    lep += 1;
                    max_e = ojph_max(e_val[lep] as i32, e_val[lep + 1] as i32) - 1;
                    e_val[lep] = e_q[7] as u8;

                    cx_val[lcxp] |= ((rho[1] & 2) >> 1) as u8;
                    lcxp += 1;
                    c_q0 = cx_val[lcxp] as i32 + ((cx_val[lcxp + 1] as i32) << 2);
                    cx_val[lcxp] = ((rho[1] & 8) >> 3) as u8;

                    let tuple1 = vlc_tbl1[((c_q1 << 8) + (rho[1] << 4) + eps1) as usize];
                    vlc.encode((tuple1 >> 8) as i32, ((tuple1 >> 4) & 7) as i32)?;

                    if c_q1 == 0 {
                        mel.encode(rho[1] != 0)?;
                    }

                    encode_quad_ms32(&mut ms, rho[1], uq1, tuple1, &s, 4)?;
                }

                // UVLC (non-initial rows always use direct encoding)
                vlc.encode(
                    uvlc_tbl[u_q0 as usize].pre as i32,
                    uvlc_tbl[u_q0 as usize].pre_len as i32,
                )?;
                vlc.encode(
                    uvlc_tbl[u_q1 as usize].pre as i32,
                    uvlc_tbl[u_q1 as usize].pre_len as i32,
                )?;
                vlc.encode(
                    uvlc_tbl[u_q0 as usize].suf as i32,
                    uvlc_tbl[u_q0 as usize].suf_len as i32,
                )?;
                vlc.encode(
                    uvlc_tbl[u_q1 as usize].suf as i32,
                    uvlc_tbl[u_q1 as usize].suf_len as i32,
                )?;

                // Prepare for next quad pair
                c_q0 |= ((rho[1] & 4) >> 1) | ((rho[1] & 8) >> 2);

                x += 4;
            }

            y += 2;
        }
    }

    terminate_mel_vlc(&mut mel, &mut vlc)?;
    ms.terminate()?;

    Ok(assemble_output(&ms, &mel, &vlc))
}

/// UVLC encoding for initial rows (32-bit path).
///
/// Three cases based on u_q0 and u_q1.
#[inline]
fn encode_uvlc32(
    vlc: &mut VlcEncoder,
    uvlc_tbl: &[UvlcEncEntry; 75],
    u_q0: i32,
    u_q1: i32,
) -> Result<()> {
    if u_q0 > 2 && u_q1 > 2 {
        let i0 = (u_q0 - 2) as usize;
        let i1 = (u_q1 - 2) as usize;
        vlc.encode(uvlc_tbl[i0].pre as i32, uvlc_tbl[i0].pre_len as i32)?;
        vlc.encode(uvlc_tbl[i1].pre as i32, uvlc_tbl[i1].pre_len as i32)?;
        vlc.encode(uvlc_tbl[i0].suf as i32, uvlc_tbl[i0].suf_len as i32)?;
        vlc.encode(uvlc_tbl[i1].suf as i32, uvlc_tbl[i1].suf_len as i32)?;
    } else if u_q0 > 2 && u_q1 > 0 {
        let i0 = u_q0 as usize;
        vlc.encode(uvlc_tbl[i0].pre as i32, uvlc_tbl[i0].pre_len as i32)?;
        vlc.encode(u_q1 - 1, 1)?;
        vlc.encode(uvlc_tbl[i0].suf as i32, uvlc_tbl[i0].suf_len as i32)?;
    } else {
        let i0 = u_q0 as usize;
        let i1 = u_q1 as usize;
        vlc.encode(uvlc_tbl[i0].pre as i32, uvlc_tbl[i0].pre_len as i32)?;
        vlc.encode(uvlc_tbl[i1].pre as i32, uvlc_tbl[i1].pre_len as i32)?;
        vlc.encode(uvlc_tbl[i0].suf as i32, uvlc_tbl[i0].suf_len as i32)?;
        vlc.encode(uvlc_tbl[i1].suf as i32, uvlc_tbl[i1].suf_len as i32)?;
    }
    Ok(())
}

/// UVLC encoding for initial rows (64-bit path) — includes ext fields.
#[allow(dead_code)]
#[inline]
fn encode_uvlc64(
    vlc: &mut VlcEncoder,
    uvlc_tbl: &[UvlcEncEntry; 75],
    u_q0: i32,
    u_q1: i32,
) -> Result<()> {
    if u_q0 > 2 && u_q1 > 2 {
        let i0 = (u_q0 - 2) as usize;
        let i1 = (u_q1 - 2) as usize;
        vlc.encode(uvlc_tbl[i0].pre as i32, uvlc_tbl[i0].pre_len as i32)?;
        vlc.encode(uvlc_tbl[i1].pre as i32, uvlc_tbl[i1].pre_len as i32)?;
        vlc.encode(uvlc_tbl[i0].suf as i32, uvlc_tbl[i0].suf_len as i32)?;
        vlc.encode(uvlc_tbl[i1].suf as i32, uvlc_tbl[i1].suf_len as i32)?;
        vlc.encode(uvlc_tbl[i0].ext as i32, uvlc_tbl[i0].ext_len as i32)?;
        vlc.encode(uvlc_tbl[i1].ext as i32, uvlc_tbl[i1].ext_len as i32)?;
    } else if u_q0 > 2 && u_q1 > 0 {
        let i0 = u_q0 as usize;
        vlc.encode(uvlc_tbl[i0].pre as i32, uvlc_tbl[i0].pre_len as i32)?;
        vlc.encode(u_q1 - 1, 1)?;
        vlc.encode(uvlc_tbl[i0].suf as i32, uvlc_tbl[i0].suf_len as i32)?;
        vlc.encode(uvlc_tbl[i0].ext as i32, uvlc_tbl[i0].ext_len as i32)?;
    } else {
        let i0 = u_q0 as usize;
        let i1 = u_q1 as usize;
        vlc.encode(uvlc_tbl[i0].pre as i32, uvlc_tbl[i0].pre_len as i32)?;
        vlc.encode(uvlc_tbl[i1].pre as i32, uvlc_tbl[i1].pre_len as i32)?;
        vlc.encode(uvlc_tbl[i0].suf as i32, uvlc_tbl[i0].suf_len as i32)?;
        vlc.encode(uvlc_tbl[i1].suf as i32, uvlc_tbl[i1].suf_len as i32)?;
        vlc.encode(uvlc_tbl[i0].ext as i32, uvlc_tbl[i0].ext_len as i32)?;
        vlc.encode(uvlc_tbl[i1].ext as i32, uvlc_tbl[i1].ext_len as i32)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// 64-bit codeblock encoder
// ---------------------------------------------------------------------------

/// Encodes a 64-bit codeblock into an HTJ2K coded byte buffer.
///
/// Same algorithm as `encode_codeblock32` but operates on 64-bit samples,
/// uses `ms_encode64`, and includes UVLC extension fields.
#[allow(dead_code)]
pub(crate) fn encode_codeblock64(
    buf: &[u64],
    missing_msbs: u32,
    num_passes: u32,
    width: u32,
    height: u32,
    stride: u32,
) -> Result<EncodeResult> {
    debug_assert_eq!(num_passes, 1);
    let _ = num_passes;

    let tables = encoder_tables();
    let vlc_tbl0 = &tables.vlc_tbl0;
    let vlc_tbl1 = &tables.vlc_tbl1;
    let uvlc_tbl = &tables.uvlc_tbl;

    const MS_SIZE: usize = (22528_usize * 16).div_ceil(15);
    const MEL_VLC_SIZE: usize = 3072;
    const MEL_SIZE: usize = 192;
    const VLC_SIZE: usize = MEL_VLC_SIZE - MEL_SIZE;

    let mut mel = MelEncoder::new(MEL_SIZE);
    let mut vlc = VlcEncoder::new(VLC_SIZE);
    let mut ms = MsEncoder::new(MS_SIZE);

    let p = 62 - missing_msbs;

    let mut e_val = vec![0u8; 513];
    let mut cx_val = vec![0u8; 513];

    // -----------------------------------------------------------------------
    // Initial row of quads (y = 0)
    // -----------------------------------------------------------------------
    {
        let mut c_q0: i32 = 0;

        let mut lep: usize = 0;
        let mut lcxp: usize = 0;
        e_val[0] = 0;
        cx_val[0] = 0;

        let mut sp: usize = 0;

        let mut x: u32 = 0;
        while x < width {
            let mut e_qmax = [0i32; 2];
            let mut e_q = [0i32; 8];
            let mut rho = [0i32; 2];
            let mut s = [0u64; 8];

            // --- Quad 0 ---
            let t = buf[sp];
            let (sig, eq, sv) = prepare_sample64(t, p);
            if sig {
                rho[0] = 1;
                e_q[0] = eq;
                e_qmax[0] = eq;
                s[0] = sv;
            }

            let t = if height > 1 {
                buf[sp + stride as usize]
            } else {
                0
            };
            sp += 1;
            let (sig, eq, sv) = prepare_sample64(t, p);
            if sig {
                rho[0] += 2;
                e_q[1] = eq;
                e_qmax[0] = ojph_max(e_qmax[0], eq);
                s[1] = sv;
            }

            if x + 1 < width {
                let t = buf[sp];
                let (sig, eq, sv) = prepare_sample64(t, p);
                if sig {
                    rho[0] += 4;
                    e_q[2] = eq;
                    e_qmax[0] = ojph_max(e_qmax[0], eq);
                    s[2] = sv;
                }

                let t = if height > 1 {
                    buf[sp + stride as usize]
                } else {
                    0
                };
                sp += 1;
                let (sig, eq, sv) = prepare_sample64(t, p);
                if sig {
                    rho[0] += 8;
                    e_q[3] = eq;
                    e_qmax[0] = ojph_max(e_qmax[0], eq);
                    s[3] = sv;
                }
            }

            let uq0 = ojph_max(e_qmax[0], 1);
            let u_q0 = uq0 - 1;
            let mut u_q1: i32 = 0;

            let eps0 = compute_eps(u_q0, &e_q, 0, e_qmax[0]);

            e_val[lep] = ojph_max(e_val[lep] as i32, e_q[1]) as u8;
            lep += 1;
            e_val[lep] = e_q[3] as u8;
            cx_val[lcxp] |= ((rho[0] & 2) >> 1) as u8;
            lcxp += 1;
            cx_val[lcxp] = ((rho[0] & 8) >> 3) as u8;

            let tuple0 = vlc_tbl0[((c_q0 << 8) + (rho[0] << 4) + eps0) as usize];
            vlc.encode((tuple0 >> 8) as i32, ((tuple0 >> 4) & 7) as i32)?;

            if c_q0 == 0 {
                mel.encode(rho[0] != 0)?;
            }

            encode_quad_ms64(&mut ms, rho[0], uq0, tuple0, &s, 0)?;

            // --- Quad 1 ---
            if x + 2 < width {
                let t = buf[sp];
                let (sig, eq, sv) = prepare_sample64(t, p);
                if sig {
                    rho[1] = 1;
                    e_q[4] = eq;
                    e_qmax[1] = eq;
                    s[4] = sv;
                }

                let t = if height > 1 {
                    buf[sp + stride as usize]
                } else {
                    0
                };
                sp += 1;
                let (sig, eq, sv) = prepare_sample64(t, p);
                if sig {
                    rho[1] += 2;
                    e_q[5] = eq;
                    e_qmax[1] = ojph_max(e_qmax[1], eq);
                    s[5] = sv;
                }

                if x + 3 < width {
                    let t = buf[sp];
                    let (sig, eq, sv) = prepare_sample64(t, p);
                    if sig {
                        rho[1] += 4;
                        e_q[6] = eq;
                        e_qmax[1] = ojph_max(e_qmax[1], eq);
                        s[6] = sv;
                    }

                    let t = if height > 1 {
                        buf[sp + stride as usize]
                    } else {
                        0
                    };
                    sp += 1;
                    let (sig, eq, sv) = prepare_sample64(t, p);
                    if sig {
                        rho[1] += 8;
                        e_q[7] = eq;
                        e_qmax[1] = ojph_max(e_qmax[1], eq);
                        s[7] = sv;
                    }
                }

                let c_q1 = (rho[0] >> 1) | (rho[0] & 1);
                let uq1 = ojph_max(e_qmax[1], 1);
                u_q1 = uq1 - 1;

                let eps1 = compute_eps(u_q1, &e_q, 4, e_qmax[1]);

                e_val[lep] = ojph_max(e_val[lep] as i32, e_q[5]) as u8;
                lep += 1;
                e_val[lep] = e_q[7] as u8;
                cx_val[lcxp] |= ((rho[1] & 2) >> 1) as u8;
                lcxp += 1;
                cx_val[lcxp] = ((rho[1] & 8) >> 3) as u8;

                let tuple1 = vlc_tbl0[((c_q1 << 8) + (rho[1] << 4) + eps1) as usize];
                vlc.encode((tuple1 >> 8) as i32, ((tuple1 >> 4) & 7) as i32)?;

                if c_q1 == 0 {
                    mel.encode(rho[1] != 0)?;
                }

                encode_quad_ms64(&mut ms, rho[1], uq1, tuple1, &s, 4)?;
            }

            // UVLC (initial row, 64-bit: includes ext)
            if u_q0 > 0 && u_q1 > 0 {
                mel.encode(ojph_min(u_q0, u_q1) > 2)?;
            }

            encode_uvlc64(&mut vlc, uvlc_tbl, u_q0, u_q1)?;

            c_q0 = (rho[1] >> 1) | (rho[1] & 1);

            x += 4;
        }

        e_val[lep + 1] = 0;
    }

    // -----------------------------------------------------------------------
    // Non-initial quad rows
    // -----------------------------------------------------------------------
    {
        let mut y: u32 = 2;
        while y < height {
            let mut lep: usize = 0;
            let mut max_e = ojph_max(e_val[0] as i32, e_val[1] as i32) - 1;
            e_val[0] = 0;

            let mut lcxp: usize = 0;
            let mut c_q0: i32 = cx_val[0] as i32 + ((cx_val[1] as i32) << 2);
            cx_val[0] = 0;

            let row_offset = (y as usize) * (stride as usize);

            let mut sp: usize = row_offset;

            let mut x: u32 = 0;
            while x < width {
                let mut e_qmax = [0i32; 2];
                let mut e_q = [0i32; 8];
                let mut rho = [0i32; 2];
                let mut s = [0u64; 8];

                // --- Quad 0 ---
                let t = buf[sp];
                let (sig, eq, sv) = prepare_sample64(t, p);
                if sig {
                    rho[0] = 1;
                    e_q[0] = eq;
                    e_qmax[0] = eq;
                    s[0] = sv;
                }

                let t = if y + 1 < height {
                    buf[sp + stride as usize]
                } else {
                    0
                };
                sp += 1;
                let (sig, eq, sv) = prepare_sample64(t, p);
                if sig {
                    rho[0] += 2;
                    e_q[1] = eq;
                    e_qmax[0] = ojph_max(e_qmax[0], eq);
                    s[1] = sv;
                }

                if x + 1 < width {
                    let t = buf[sp];
                    let (sig, eq, sv) = prepare_sample64(t, p);
                    if sig {
                        rho[0] += 4;
                        e_q[2] = eq;
                        e_qmax[0] = ojph_max(e_qmax[0], eq);
                        s[2] = sv;
                    }

                    let t = if y + 1 < height {
                        buf[sp + stride as usize]
                    } else {
                        0
                    };
                    sp += 1;
                    let (sig, eq, sv) = prepare_sample64(t, p);
                    if sig {
                        rho[0] += 8;
                        e_q[3] = eq;
                        e_qmax[0] = ojph_max(e_qmax[0], eq);
                        s[3] = sv;
                    }
                }

                let kappa = if (rho[0] & (rho[0] - 1)) != 0 {
                    ojph_max(1, max_e)
                } else {
                    1
                };
                let uq0 = ojph_max(e_qmax[0], kappa);
                let u_q0 = uq0 - kappa;
                let mut u_q1: i32 = 0;

                let eps0 = compute_eps(u_q0, &e_q, 0, e_qmax[0]);

                e_val[lep] = ojph_max(e_val[lep] as i32, e_q[1]) as u8;
                lep += 1;
                max_e = ojph_max(e_val[lep] as i32, e_val[lep + 1] as i32) - 1;
                e_val[lep] = e_q[3] as u8;

                cx_val[lcxp] |= ((rho[0] & 2) >> 1) as u8;
                lcxp += 1;
                let mut c_q1: i32 = cx_val[lcxp] as i32 + ((cx_val[lcxp + 1] as i32) << 2);
                cx_val[lcxp] = ((rho[0] & 8) >> 3) as u8;

                let tuple0 = vlc_tbl1[((c_q0 << 8) + (rho[0] << 4) + eps0) as usize];
                vlc.encode((tuple0 >> 8) as i32, ((tuple0 >> 4) & 7) as i32)?;

                if c_q0 == 0 {
                    mel.encode(rho[0] != 0)?;
                }

                encode_quad_ms64(&mut ms, rho[0], uq0, tuple0, &s, 0)?;

                // --- Quad 1 ---
                if x + 2 < width {
                    let t = buf[sp];
                    let (sig, eq, sv) = prepare_sample64(t, p);
                    if sig {
                        rho[1] = 1;
                        e_q[4] = eq;
                        e_qmax[1] = eq;
                        s[4] = sv;
                    }

                    let t = if y + 1 < height {
                        buf[sp + stride as usize]
                    } else {
                        0
                    };
                    sp += 1;
                    let (sig, eq, sv) = prepare_sample64(t, p);
                    if sig {
                        rho[1] += 2;
                        e_q[5] = eq;
                        e_qmax[1] = ojph_max(e_qmax[1], eq);
                        s[5] = sv;
                    }

                    if x + 3 < width {
                        let t = buf[sp];
                        let (sig, eq, sv) = prepare_sample64(t, p);
                        if sig {
                            rho[1] += 4;
                            e_q[6] = eq;
                            e_qmax[1] = ojph_max(e_qmax[1], eq);
                            s[6] = sv;
                        }

                        let t = if y + 1 < height {
                            buf[sp + stride as usize]
                        } else {
                            0
                        };
                        sp += 1;
                        let (sig, eq, sv) = prepare_sample64(t, p);
                        if sig {
                            rho[1] += 8;
                            e_q[7] = eq;
                            e_qmax[1] = ojph_max(e_qmax[1], eq);
                            s[7] = sv;
                        }
                    }

                    let kappa = if (rho[1] & (rho[1] - 1)) != 0 {
                        ojph_max(1, max_e)
                    } else {
                        1
                    };
                    c_q1 |= ((rho[0] & 4) >> 1) | ((rho[0] & 8) >> 2);
                    let uq1 = ojph_max(e_qmax[1], kappa);
                    u_q1 = uq1 - kappa;

                    let eps1 = compute_eps(u_q1, &e_q, 4, e_qmax[1]);

                    e_val[lep] = ojph_max(e_val[lep] as i32, e_q[5]) as u8;
                    lep += 1;
                    max_e = ojph_max(e_val[lep] as i32, e_val[lep + 1] as i32) - 1;
                    e_val[lep] = e_q[7] as u8;

                    cx_val[lcxp] |= ((rho[1] & 2) >> 1) as u8;
                    lcxp += 1;
                    c_q0 = cx_val[lcxp] as i32 + ((cx_val[lcxp + 1] as i32) << 2);
                    cx_val[lcxp] = ((rho[1] & 8) >> 3) as u8;

                    let tuple1 = vlc_tbl1[((c_q1 << 8) + (rho[1] << 4) + eps1) as usize];
                    vlc.encode((tuple1 >> 8) as i32, ((tuple1 >> 4) & 7) as i32)?;

                    if c_q1 == 0 {
                        mel.encode(rho[1] != 0)?;
                    }

                    encode_quad_ms64(&mut ms, rho[1], uq1, tuple1, &s, 4)?;
                }

                // UVLC (non-initial rows, 64-bit: includes ext)
                vlc.encode(
                    uvlc_tbl[u_q0 as usize].pre as i32,
                    uvlc_tbl[u_q0 as usize].pre_len as i32,
                )?;
                vlc.encode(
                    uvlc_tbl[u_q1 as usize].pre as i32,
                    uvlc_tbl[u_q1 as usize].pre_len as i32,
                )?;
                vlc.encode(
                    uvlc_tbl[u_q0 as usize].suf as i32,
                    uvlc_tbl[u_q0 as usize].suf_len as i32,
                )?;
                vlc.encode(
                    uvlc_tbl[u_q1 as usize].suf as i32,
                    uvlc_tbl[u_q1 as usize].suf_len as i32,
                )?;
                vlc.encode(
                    uvlc_tbl[u_q0 as usize].ext as i32,
                    uvlc_tbl[u_q0 as usize].ext_len as i32,
                )?;
                vlc.encode(
                    uvlc_tbl[u_q1 as usize].ext as i32,
                    uvlc_tbl[u_q1 as usize].ext_len as i32,
                )?;

                // Prepare for next quad pair
                c_q0 |= ((rho[1] & 4) >> 1) | ((rho[1] & 8) >> 2);

                x += 4;
            }

            y += 2;
        }
    }

    terminate_mel_vlc(&mut mel, &mut vlc)?;
    ms.terminate()?;

    Ok(assemble_output(&ms, &mel, &vlc))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// All-zero samples should produce a minimal coded buffer.
    #[test]
    fn encode_zeros_32() {
        let width = 4u32;
        let height = 4u32;
        let stride = width;
        let buf = vec![0u32; (stride * height) as usize];
        let result = encode_codeblock32(&buf, 0, 1, width, height, stride).unwrap();
        assert!(result.length > 0);
        assert_eq!(result.data.len(), result.length as usize);
    }

    /// All-zero samples should produce a minimal coded buffer (64-bit).
    #[test]
    fn encode_zeros_64() {
        let width = 4u32;
        let height = 4u32;
        let stride = width;
        let buf = vec![0u64; (stride * height) as usize];
        let result = encode_codeblock64(&buf, 0, 1, width, height, stride).unwrap();
        assert!(result.length > 0);
        assert_eq!(result.data.len(), result.length as usize);
    }

    /// Single pixel codeblock.
    #[test]
    fn encode_1x1_32() {
        let buf = vec![0x40000000u32];
        let result = encode_codeblock32(&buf, 0, 1, 1, 1, 1).unwrap();
        assert!(result.length > 0);
    }

    /// 2x2 codeblock with non-zero data.
    #[test]
    fn encode_2x2_32() {
        let buf = vec![0x10000000u32, 0x20000000, 0x30000000, 0x40000000];
        let result = encode_codeblock32(&buf, 0, 1, 2, 2, 2).unwrap();
        assert!(result.length > 0);
    }

    /// Smoke test: encoding wider blocks (8 pixels).
    #[test]
    fn encode_8x4_32() {
        let width = 8u32;
        let height = 4u32;
        let stride = width;
        let mut buf = vec![0u32; (stride * height) as usize];
        for (i, v) in buf.iter_mut().enumerate() {
            *v = ((i as u32) << 20) | (((i & 1) as u32) << 31);
        }
        let result = encode_codeblock32(&buf, 0, 1, width, height, stride).unwrap();
        assert!(result.length > 0);
    }

    /// MEL encoder produces correct byte-stuffing after 0xFF.
    #[test]
    fn mel_byte_stuffing() {
        let mut mel = MelEncoder::new(32);
        for _ in 0..8 {
            mel.emit_bit(1).unwrap();
        }
        assert_eq!(mel.buf[0], 0xFF);
        assert_eq!(mel.remaining_bits, 7);
    }

    /// VLC encoder writes backward correctly.
    #[test]
    fn vlc_backward_write() {
        let vlc = VlcEncoder::new(16);
        assert_eq!(vlc.buf[15], 0xFF);
        assert_eq!(vlc.pos, 1);
    }

    /// MagSgn termination edge case: no bits written.
    #[test]
    fn ms_terminate_empty() {
        let mut ms = MsEncoder::new(16);
        ms.terminate().unwrap();
        assert_eq!(ms.pos, 0);
    }

    /// MagSgn termination after 0xFF.
    #[test]
    fn ms_terminate_after_ff() {
        let mut ms = MsEncoder::new(16);
        ms.encode(0xFF, 8).unwrap();
        assert_eq!(ms.buf[0], 0xFF);
        assert_eq!(ms.max_bits, 7);
        ms.terminate().unwrap();
        assert_eq!(ms.pos, 0);
    }
}
