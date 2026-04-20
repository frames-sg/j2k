//! Code-block processing.
//!
//! Port of `ojph_codeblock.h/cpp`. A codeblock is the smallest coding unit
//! in the JPEG 2000 tile-component-resolution-subband hierarchy.

use crate::types::*;

/// State of a codeblock during encoding.
#[derive(Debug, Clone, Default)]
pub struct CodeblockEncState {
    /// Encoded byte data for pass 1.
    pub pass1_bytes: u32,
    /// Encoded byte data for pass 2.
    pub pass2_bytes: u32,
    /// Number of coding passes.
    pub num_passes: u32,
    /// Number of missing MSBs (zero bitplanes).
    pub missing_msbs: u32,
    /// True if this codeblock has any non-zero coefficients.
    pub has_data: bool,
}

/// State of a codeblock during decoding.
#[derive(Debug, Clone, Default)]
pub struct CodeblockDecState {
    /// Length of pass 1 data.
    pub pass1_len: u32,
    /// Length of pass 2 data.
    pub pass2_len: u32,
    /// Number of coding passes.
    pub num_passes: u32,
    /// Number of missing MSBs.
    pub missing_msbs: u32,
}

/// A codeblock in the JPEG 2000 hierarchy.
///
/// Contains the rectangular extent and coding state for one block.
#[derive(Debug, Clone)]
pub struct Codeblock {
    /// Position of this codeblock within the subband.
    pub cb_rect: Rect,
    /// Nominal block dimensions (log2).
    pub log_block_dims: Size,
    /// Encoding state (populated during encode).
    pub enc_state: Option<CodeblockEncState>,
    /// Decoding state (populated during decode).
    pub dec_state: Option<CodeblockDecState>,
    /// Compressed data for this block.
    pub coded_data: Vec<u8>,
}

impl Default for Codeblock {
    fn default() -> Self {
        Self {
            cb_rect: Rect::new(Point::new(0, 0), Size::new(0, 0)),
            log_block_dims: Size::new(0, 0),
            enc_state: None,
            dec_state: None,
            coded_data: Vec::new(),
        }
    }
}

impl Codeblock {
    /// Create a new codeblock at the given position/size.
    pub fn new(rect: Rect, log_dims: Size) -> Self {
        Self {
            cb_rect: rect,
            log_block_dims: log_dims,
            ..Default::default()
        }
    }

    /// Width of this codeblock.
    #[inline]
    pub fn width(&self) -> u32 {
        self.cb_rect.siz.w
    }

    /// Height of this codeblock.
    #[inline]
    pub fn height(&self) -> u32 {
        self.cb_rect.siz.h
    }

    /// True if this codeblock has zero area (empty).
    #[allow(dead_code)]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.cb_rect.siz.w == 0 || self.cb_rect.siz.h == 0
    }

    /// Clear encoded/decoded state (reset for reuse).
    #[allow(dead_code)]
    pub fn clear(&mut self) {
        self.enc_state = None;
        self.dec_state = None;
        self.coded_data.clear();
    }
}
