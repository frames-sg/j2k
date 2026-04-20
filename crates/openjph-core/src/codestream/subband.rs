//! Subband processing.
//!
//! Port of `ojph_subband.h/cpp`. A subband represents one frequency band
//! (LL, HL, LH, or HH) at a particular resolution level.

use super::codeblock::*;
use crate::coding::decoder32::decode_codeblock32;
use crate::coding::encoder::encode_codeblock32;
use crate::error::Result;
use crate::types::*;

/// Subband types: LL (lowpass-lowpass), HL, LH, HH.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum SubbandType {
    /// LL subband (only at the lowest resolution).
    LL = 0,
    /// HL subband (horizontal highpass, vertical lowpass).
    HL = 1,
    /// LH subband (horizontal lowpass, vertical highpass).
    LH = 2,
    /// HH subband (both highpass).
    HH = 3,
}

impl SubbandType {
    #[allow(dead_code)]
    pub fn from_index(i: u32) -> Self {
        match i {
            0 => Self::LL,
            1 => Self::HL,
            2 => Self::LH,
            3 => Self::HH,
            _ => Self::LL,
        }
    }
}

/// A subband within a resolution level.
///
/// Contains the geometry and quantization parameters for one frequency band.
#[derive(Debug, Clone)]
pub struct Subband {
    /// Subband type (LL, HL, LH, HH).
    pub band_type: SubbandType,
    /// Resolution level that owns this subband (0 = lowest).
    pub resolution_num: u32,
    /// Subband rectangle in the coordinate space of the resolution.
    pub band_rect: Rect,
    /// Log2 of the codeblock dimensions used in this subband.
    #[allow(dead_code)]
    pub log_block_dims: Size,
    /// Number of missing MSBs (Kmax from quantization).
    pub k_max: u32,
    /// Quantization step size delta (irreversible only).
    pub delta: f32,
    /// True if reversible coding.
    pub reversible: bool,
    /// Number of codeblocks in x direction.
    pub num_blocks_x: u32,
    /// Number of codeblocks in y direction.
    pub num_blocks_y: u32,
    /// Codeblocks in raster order (row by row).
    pub codeblocks: Vec<Codeblock>,
    /// Coefficient data (row-major, width × height).
    pub coeffs: Vec<i32>,
    /// Float coefficient data for irreversible decode (dequantized, normalized).
    pub coeffs_f32: Vec<f32>,
}

impl Default for Subband {
    fn default() -> Self {
        Self {
            band_type: SubbandType::LL,
            resolution_num: 0,
            band_rect: Rect::new(Point::new(0, 0), Size::new(0, 0)),
            log_block_dims: Size::new(0, 0),
            k_max: 0,
            delta: 1.0,
            reversible: true,
            num_blocks_x: 0,
            num_blocks_y: 0,
            codeblocks: Vec::new(),
            coeffs: Vec::new(),
            coeffs_f32: Vec::new(),
        }
    }
}

impl Subband {
    /// Create a new subband with basic parameters.
    pub fn new(
        band_type: SubbandType,
        resolution_num: u32,
        band_rect: Rect,
        log_block_dims: Size,
    ) -> Self {
        let bw = 1u32 << log_block_dims.w;
        let bh = 1u32 << log_block_dims.h;
        let num_blocks_x = if band_rect.siz.w > 0 {
            div_ceil(band_rect.org.x + band_rect.siz.w, bw) - (band_rect.org.x / bw)
        } else {
            0
        };
        let num_blocks_y = if band_rect.siz.h > 0 {
            div_ceil(band_rect.org.y + band_rect.siz.h, bh) - (band_rect.org.y / bh)
        } else {
            0
        };

        // Create codeblock objects
        let first_x = (band_rect.org.x / bw) * bw;
        let first_y = (band_rect.org.y / bh) * bh;
        let mut codeblocks = Vec::with_capacity((num_blocks_x * num_blocks_y) as usize);
        for by in 0..num_blocks_y {
            for bx in 0..num_blocks_x {
                let cb_x0 = (first_x + bx * bw).max(band_rect.org.x);
                let cb_y0 = (first_y + by * bh).max(band_rect.org.y);
                let cb_x1 = (first_x + (bx + 1) * bw).min(band_rect.org.x + band_rect.siz.w);
                let cb_y1 = (first_y + (by + 1) * bh).min(band_rect.org.y + band_rect.siz.h);
                let rect = Rect::new(
                    Point::new(cb_x0, cb_y0),
                    Size::new(cb_x1 - cb_x0, cb_y1 - cb_y0),
                );
                codeblocks.push(Codeblock::new(rect, log_block_dims));
            }
        }

        Self {
            band_type,
            resolution_num,
            band_rect,
            log_block_dims,
            num_blocks_x,
            num_blocks_y,
            codeblocks,
            ..Default::default()
        }
    }

    /// Width of the subband.
    #[inline]
    pub fn width(&self) -> u32 {
        self.band_rect.siz.w
    }

    /// Height of the subband.
    #[inline]
    pub fn height(&self) -> u32 {
        self.band_rect.siz.h
    }

    /// True if this subband has zero area.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.band_rect.siz.w == 0 || self.band_rect.siz.h == 0
    }

    /// Total number of codeblocks.
    #[inline]
    #[allow(dead_code)]
    pub fn total_blocks(&self) -> u32 {
        self.num_blocks_x * self.num_blocks_y
    }

    /// Encode all codeblocks in this subband using the HTJ2K block encoder.
    pub fn encode_codeblocks(&mut self) -> Result<()> {
        let sb_w = self.width();
        if self.is_empty() {
            return Ok(());
        }

        for cb in &mut self.codeblocks {
            let cb_w = cb.width();
            let cb_h = cb.height();
            if cb_w == 0 || cb_h == 0 {
                continue;
            }

            let nominal_w = 1u32 << cb.log_block_dims.w;
            let nominal_h = 1u32 << cb.log_block_dims.h;
            let stride = (nominal_w + 7) & !7;

            // Extract and convert coefficients: i32 → u32 (sign<<31 | magnitude)
            let cb_x0 = cb.cb_rect.org.x - self.band_rect.org.x;
            let cb_y0 = cb.cb_rect.org.y - self.band_rect.org.y;
            let mut u32_buf = vec![0u32; (stride * nominal_h) as usize];
            let mut max_val = 0u32;

            if self.reversible {
                let shift = 31u32.saturating_sub(self.k_max);
                u32_buf.fill(0);
                for y in 0..cb_h {
                    for x in 0..cb_w {
                        let coeff = self.coeffs[((cb_y0 + y) * sb_w + (cb_x0 + x)) as usize];
                        let sign = if coeff < 0 { 0x8000_0000 } else { 0 };
                        let mag = coeff.unsigned_abs() << shift;
                        let val = sign | mag;
                        u32_buf[(y * stride + x) as usize] = val;
                        max_val |= mag;
                    }
                }
            } else {
                for y in 0..cb_h {
                    for x in 0..cb_w {
                        let coeff = self.coeffs[((cb_y0 + y) * sb_w + (cb_x0 + x)) as usize];
                        let sign = if coeff < 0 { 0x8000_0000 } else { 0 };
                        let mag = coeff.unsigned_abs();
                        let val = sign | mag;
                        u32_buf[(y * stride + x) as usize] = val;
                        max_val |= mag;
                    }
                }
            }

            if max_val == 0 {
                cb.enc_state = Some(CodeblockEncState {
                    pass1_bytes: 0,
                    pass2_bytes: 0,
                    num_passes: 0,
                    missing_msbs: 0,
                    has_data: false,
                });
                continue;
            }

            let missing_msbs = self.k_max.saturating_sub(1);

            let result = encode_codeblock32(&u32_buf, missing_msbs, 1, cb_w, cb_h, stride)?;

            cb.enc_state = Some(CodeblockEncState {
                pass1_bytes: result.length,
                pass2_bytes: 0,
                num_passes: 1,
                missing_msbs,
                has_data: true,
            });
            cb.coded_data = result.data[..result.length as usize].to_vec();
        }
        Ok(())
    }

    /// Decode all codeblocks and place coefficients into self.coeffs.
    pub fn decode_codeblocks(&mut self) -> Result<()> {
        let sb_w = self.width();
        let sb_h = self.height();
        if self.is_empty() {
            return Ok(());
        }

        self.coeffs = vec![0i32; (sb_w * sb_h) as usize];
        if !self.reversible {
            self.coeffs_f32 = vec![0f32; (sb_w * sb_h) as usize];
        }

        for cb in &mut self.codeblocks {
            let cb_w = cb.width();
            let cb_h = cb.height();
            if cb_w == 0 || cb_h == 0 {
                continue;
            }

            let dec = match &cb.dec_state {
                Some(d) => d.clone(),
                None => continue,
            };

            if dec.num_passes == 0 {
                continue;
            }

            let nominal_w = 1u32 << cb.log_block_dims.w;
            let nominal_h = 1u32 << cb.log_block_dims.h;
            let stride = (nominal_w + 7) & !7;
            let mut decoded = vec![0u32; (stride * nominal_h) as usize];

            let _ok = decode_codeblock32(
                &mut cb.coded_data,
                &mut decoded,
                dec.missing_msbs,
                dec.num_passes,
                dec.pass1_len,
                dec.pass2_len,
                cb_w,
                cb_h,
                stride,
                false,
            )?;
            let shift = 31u32.saturating_sub(self.k_max);
            let cb_x0 = cb.cb_rect.org.x - self.band_rect.org.x;
            let cb_y0 = cb.cb_rect.org.y - self.band_rect.org.y;
            for y in 0..cb_h {
                for x in 0..cb_w {
                    let val = decoded[(y * stride + x) as usize];
                    let sign = (val >> 31) & 1;
                    let idx = ((cb_y0 + y) * sb_w + (cb_x0 + x)) as usize;
                    if self.reversible {
                        let mag = (val & 0x7FFF_FFFF) >> shift;
                        self.coeffs[idx] = if sign != 0 { -(mag as i32) } else { mag as i32 };
                    } else {
                        // Irreversible: dequantize to float, keep full precision
                        let mag = (val & 0x7FFF_FFFF) as f32 * self.delta;
                        let float_val = if sign != 0 { -mag } else { mag };
                        self.coeffs_f32[idx] = float_val;
                    }
                }
            }
        }
        Ok(())
    }
}
