//! Resolution level processing.
//!
//! Port of `ojph_resolution.h/cpp`. A resolution level contains 1 or 3 subbands
//! and is divided into precincts.

#![allow(dead_code)]

use super::subband::{Subband, SubbandType};
use crate::types::*;

/// A resolution level within a tile-component.
///
/// Resolution 0 is the lowest (coarsest) level with a single LL subband.
/// Higher resolutions each have HL, LH, HH subbands.
#[derive(Debug, Clone)]
pub struct Resolution {
    /// Resolution index (0 = lowest).
    pub res_num: u32,
    /// Rectangle of this resolution in the image coordinate system.
    pub res_rect: Rect,
    /// Log2 of the precinct dimensions at this level.
    pub log_precinct_size: Size,
    /// Number of precincts in x direction.
    pub num_precincts_x: u32,
    /// Number of precincts in y direction.
    pub num_precincts_y: u32,
    /// Subbands at this resolution level.
    /// - res_num == 0: single LL subband
    /// - res_num > 0: HL, LH, HH subbands
    pub subbands: Vec<Subband>,
    /// Log2 of the codeblock dimensions.
    pub log_block_dims: Size,
    /// True if this is the lowest resolution.
    pub is_lowest: bool,
    /// Number of decomposition levels from the top.
    pub num_decomps: u32,
    /// True if reversible coding.
    pub reversible: bool,
}

impl Default for Resolution {
    fn default() -> Self {
        Self {
            res_num: 0,
            res_rect: Rect::new(Point::new(0, 0), Size::new(0, 0)),
            log_precinct_size: Size::new(15, 15),
            num_precincts_x: 0,
            num_precincts_y: 0,
            subbands: Vec::new(),
            log_block_dims: Size::new(0, 0),
            is_lowest: true,
            num_decomps: 0,
            reversible: true,
        }
    }
}

impl Resolution {
    /// Create a new resolution level with the given parameters.
    pub fn new(
        res_num: u32,
        num_decomps: u32,
        res_rect: Rect,
        log_precinct_size: Size,
        log_block_dims: Size,
        reversible: bool,
    ) -> Self {
        let is_lowest = res_num == 0;

        // Compute number of precincts
        let ppx = 1u32 << log_precinct_size.w;
        let ppy = 1u32 << log_precinct_size.h;
        let num_precincts_x = if res_rect.siz.w > 0 {
            div_ceil(res_rect.org.x + res_rect.siz.w, ppx) - (res_rect.org.x / ppx)
        } else {
            0
        };
        let num_precincts_y = if res_rect.siz.h > 0 {
            div_ceil(res_rect.org.y + res_rect.siz.h, ppy) - (res_rect.org.y / ppy)
        } else {
            0
        };

        // Create subbands
        let subbands = if is_lowest {
            // Single LL subband
            vec![Subband::new(
                SubbandType::LL,
                res_num,
                res_rect,
                log_block_dims,
            )]
        } else {
            // HL, LH, HH subbands with correct JPEG 2000 dimensions.
            // For resolution extent [ox, ox+w) × [oy, oy+h):
            //   Even (low) count:  ceil((o+s)/2) - ceil(o/2)
            //   Odd  (high) count: floor((o+s)/2) - floor(o/2) = (o+s)/2 - o/2
            let ox = res_rect.org.x;
            let oy = res_rect.org.y;
            let w = res_rect.siz.w;
            let h = res_rect.siz.h;

            let low_w = div_ceil(ox + w, 2) - div_ceil(ox, 2);
            let low_h = div_ceil(oy + h, 2) - div_ceil(oy, 2);
            let high_w = (ox + w) / 2 - ox / 2;
            let high_h = (oy + h) / 2 - oy / 2;

            // HL: horizontal high-pass, vertical low-pass
            let hl_rect = Rect::new(
                Point::new(ox / 2, div_ceil(oy, 2)),
                Size::new(high_w, low_h),
            );
            // LH: horizontal low-pass, vertical high-pass
            let lh_rect = Rect::new(
                Point::new(div_ceil(ox, 2), oy / 2),
                Size::new(low_w, high_h),
            );
            // HH: both high-pass
            let hh_rect = Rect::new(Point::new(ox / 2, oy / 2), Size::new(high_w, high_h));

            vec![
                Subband::new(SubbandType::HL, res_num, hl_rect, log_block_dims),
                Subband::new(SubbandType::LH, res_num, lh_rect, log_block_dims),
                Subband::new(SubbandType::HH, res_num, hh_rect, log_block_dims),
            ]
        };

        Self {
            res_num,
            res_rect,
            log_precinct_size,
            num_precincts_x,
            num_precincts_y,
            subbands,
            log_block_dims,
            is_lowest,
            num_decomps,
            reversible,
        }
    }

    /// Width of this resolution level.
    #[inline]
    pub fn width(&self) -> u32 {
        self.res_rect.siz.w
    }

    /// Height of this resolution level.
    #[inline]
    pub fn height(&self) -> u32 {
        self.res_rect.siz.h
    }

    /// Total number of precincts at this resolution.
    #[inline]
    pub fn num_precincts(&self) -> u32 {
        self.num_precincts_x * self.num_precincts_y
    }

    /// Number of subbands at this resolution level.
    #[inline]
    pub fn num_subbands(&self) -> usize {
        self.subbands.len()
    }

    /// True if this resolution has no area.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.res_rect.siz.w == 0 || self.res_rect.siz.h == 0
    }
}
