//! Tile processing.
//!
//! Port of `ojph_tile.h/cpp`. A tile is the top-level subdivision of the
//! image. Each tile contains tile-components.

#![allow(dead_code)]

use super::bitbuffer_write::BitBufferWrite;
use super::codeblock::CodeblockDecState;
use super::tile_comp::TileComp;
use crate::arch::count_leading_zeros;
use crate::error::{OjphError, Result};
use crate::file::{InfileBase, OutfileBase};
use crate::params::local::markers;
use crate::params::{ParamCod, ParamQcd, ParamSiz, ParamSot};
use crate::types::*;

/// A single tile within the codestream.
///
/// Contains the tile-components (one per image component).
#[derive(Debug, Clone)]
pub struct Tile {
    /// Tile index (raster order within the tile grid).
    pub tile_idx: u32,
    /// Tile rectangle in the image coordinate system.
    pub tile_rect: Rect,
    /// Tile components (one per image component).
    pub tile_comps: Vec<TileComp>,
    /// Number of components.
    pub num_comps: u32,
    /// Whether color transform is employed for this tile.
    pub employ_color_transform: bool,
    /// Number of quality layers in packet parsing.
    pub num_layers: u32,
    /// Number of tile parts.
    pub num_tileparts: u32,
    /// Whether SOP markers may appear before packet headers.
    pub may_use_sop: bool,
    /// Whether packet headers are followed by EPH markers.
    pub uses_eph: bool,
    /// SOT marker data for this tile.
    pub sot: ParamSot,
    /// Whether the tile has been fully read/written.
    pub is_complete: bool,
    /// Current line being processed within this tile.
    pub cur_line: u32,
    /// Number of lines in this tile.
    pub num_lines: u32,
    /// Skipped resolution levels for reconstruction.
    pub skipped_res_for_recon: u32,
}

impl Default for Tile {
    fn default() -> Self {
        Self {
            tile_idx: 0,
            tile_rect: Rect::new(Point::new(0, 0), Size::new(0, 0)),
            tile_comps: Vec::new(),
            num_comps: 0,
            employ_color_transform: false,
            num_layers: 1,
            num_tileparts: 1,
            may_use_sop: false,
            uses_eph: false,
            sot: ParamSot::default(),
            is_complete: false,
            cur_line: 0,
            num_lines: 0,
            skipped_res_for_recon: 0,
        }
    }
}

#[derive(Debug, Clone)]
struct PacketTagTree {
    widths: Vec<u32>,
    heights: Vec<u32>,
    levels: Vec<Vec<u32>>,
}

impl PacketTagTree {
    fn new(width: u32, height: u32, num_levels: u32, init_val: u32) -> Self {
        let mut widths = Vec::with_capacity((num_levels + 1) as usize);
        let mut heights = Vec::with_capacity((num_levels + 1) as usize);
        let mut levels = Vec::with_capacity((num_levels + 1) as usize);

        for lev in 0..num_levels {
            let w = (width + (1u32 << lev) - 1) >> lev;
            let h = (height + (1u32 << lev) - 1) >> lev;
            widths.push(w);
            heights.push(h);
            levels.push(vec![init_val; (w * h) as usize]);
        }

        widths.push(1);
        heights.push(1);
        levels.push(vec![0u32; 1]);

        Self {
            widths,
            heights,
            levels,
        }
    }

    fn get(&self, x: u32, y: u32, lev: u32) -> u32 {
        let lev = lev as usize;
        let idx = (x + y * self.widths[lev]) as usize;
        self.levels[lev][idx]
    }

    fn set(&mut self, x: u32, y: u32, lev: u32, value: u32) {
        let lev = lev as usize;
        let idx = (x + y * self.widths[lev]) as usize;
        self.levels[lev][idx] = value;
    }

    fn build_min_levels(&mut self, width: u32, height: u32, num_levels: u32) {
        for lev in 1..num_levels {
            let lev_w = (width + (1u32 << lev) - 1) >> lev;
            let lev_h = (height + (1u32 << lev) - 1) >> lev;
            for y in 0..lev_h {
                for x in 0..lev_w {
                    let child_x = x << 1;
                    let child_y = y << 1;
                    let child = |cx: u32, cy: u32| {
                        if cx < self.widths[(lev - 1) as usize]
                            && cy < self.heights[(lev - 1) as usize]
                        {
                            self.get(cx, cy, lev - 1)
                        } else {
                            0
                        }
                    };
                    let t1 = child(child_x, child_y).min(child(child_x + 1, child_y));
                    let t2 = child(child_x, child_y + 1).min(child(child_x + 1, child_y + 1));
                    self.set(x, y, lev, t1.min(t2));
                }
            }
        }
        self.set(0, 0, num_levels, 0);
    }
}

fn log2ceil(x: u32) -> u32 {
    let t = 31 - count_leading_zeros(x);
    t + u32::from((x & (x - 1)) != 0)
}

#[derive(Debug, Clone)]
struct PacketBitReader<'a> {
    data: &'a [u8],
    pos: usize,
    tmp: u8,
    avail_bits: u32,
    unstuff: bool,
    bytes_left: usize,
}

impl<'a> PacketBitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            pos: 0,
            tmp: 0,
            avail_bits: 0,
            unstuff: false,
            bytes_left: data.len(),
        }
    }

    fn refill(&mut self) -> bool {
        if self.bytes_left > 0 {
            let byte = self.data[self.pos];
            self.pos += 1;
            self.tmp = byte;
            self.avail_bits = 8 - u32::from(self.unstuff);
            self.unstuff = byte == 0xFF;
            self.bytes_left -= 1;
            true
        } else {
            self.tmp = 0;
            self.avail_bits = 8 - u32::from(self.unstuff);
            self.unstuff = false;
            false
        }
    }

    fn read_bit(&mut self, err: &'static str) -> Result<u32> {
        if self.avail_bits == 0 && !self.refill() {
            return Err(OjphError::Codec {
                code: 0x00060006,
                message: err.into(),
            });
        }
        self.avail_bits -= 1;
        Ok(((self.tmp >> self.avail_bits) & 1) as u32)
    }

    fn read_bits(&mut self, mut num_bits: u32, err: &'static str) -> Result<u32> {
        let mut bits = 0u32;
        while num_bits > 0 {
            if self.avail_bits == 0 && !self.refill() {
                return Err(OjphError::Codec {
                    code: 0x00060007,
                    message: err.into(),
                });
            }
            let tx_bits = self.avail_bits.min(num_bits);
            bits <<= tx_bits;
            self.avail_bits -= tx_bits;
            num_bits -= tx_bits;
            bits |= ((self.tmp >> self.avail_bits) as u32) & ((1u32 << tx_bits) - 1);
        }
        Ok(bits)
    }

    fn terminate(&mut self, uses_eph: bool) -> Result<()> {
        if self.unstuff {
            self.refill();
        }
        debug_assert!(!self.unstuff);
        if uses_eph {
            if self.bytes_left < 2 {
                return Err(OjphError::Codec {
                    code: 0x00060004,
                    message: "truncated packet header before EPH marker".into(),
                });
            }
            let eph = u16::from_be_bytes([self.data[self.pos], self.data[self.pos + 1]]);
            if eph != markers::EPH {
                return Err(OjphError::Codec {
                    code: 0x00060005,
                    message: format!("expected EPH marker (0xFF92), got 0x{:04X}", eph),
                });
            }
            self.pos += 2;
            self.bytes_left -= 2;
        }
        self.tmp = 0;
        self.avail_bits = 0;
        Ok(())
    }

    fn skip_optional_sop(&mut self) -> Result<()> {
        debug_assert_eq!(self.avail_bits, 0);
        debug_assert!(!self.unstuff);
        if self.bytes_left < 2 {
            return Ok(());
        }

        let marker = u16::from_be_bytes([self.data[self.pos], self.data[self.pos + 1]]);
        if marker != markers::SOP {
            return Ok(());
        }

        if self.bytes_left < 6 {
            return Err(OjphError::Codec {
                code: 0x00060008,
                message: "packet truncated inside SOP marker".into(),
            });
        }

        let lsop = u16::from_be_bytes([self.data[self.pos + 2], self.data[self.pos + 3]]);
        if lsop != 4 {
            return Err(OjphError::Codec {
                code: 0x00060009,
                message: format!("unexpected SOP length {lsop}, expected 4"),
            });
        }

        self.pos += 6;
        self.bytes_left -= 6;
        Ok(())
    }

    fn read_chunk(&mut self, num_bytes: usize) -> Vec<u8> {
        debug_assert_eq!(self.avail_bits, 0);
        debug_assert!(!self.unstuff);
        let bytes = num_bytes.min(self.bytes_left);
        let mut out = vec![0u8; num_bytes];
        out[..bytes].copy_from_slice(&self.data[self.pos..self.pos + bytes]);
        self.pos += bytes;
        self.bytes_left -= bytes;
        out
    }

    fn consumed(&self) -> usize {
        self.data.len() - self.bytes_left
    }
}

impl Tile {
    /// Create a new tile with the given parameters.
    pub fn new(
        tile_idx: u32,
        tile_rect: Rect,
        siz: &ParamSiz,
        cod: &ParamCod,
        qcd: &ParamQcd,
        skipped_res_for_recon: u32,
    ) -> Self {
        let num_comps = siz.get_num_components() as u32;
        let mut tile_comps = Vec::with_capacity(num_comps as usize);

        for c in 0..num_comps {
            let ds = siz.get_downsampling(c);
            let comp_rect = Rect::new(
                Point::new(
                    div_ceil(tile_rect.org.x, ds.x),
                    div_ceil(tile_rect.org.y, ds.y),
                ),
                Size::new(
                    div_ceil(tile_rect.org.x + tile_rect.siz.w, ds.x)
                        - div_ceil(tile_rect.org.x, ds.x),
                    div_ceil(tile_rect.org.y + tile_rect.siz.h, ds.y)
                        - div_ceil(tile_rect.org.y, ds.y),
                ),
            );

            let cp = cod.get_coc(c);
            let qp = qcd.get_qcc(c);
            let num_decomps = cp.get_num_decompositions() as u32;
            let log_block_dims = cp.get_log_block_dims();

            // Collect precinct sizes
            let mut log_pp = Vec::with_capacity((num_decomps + 1) as usize);
            for r in 0..=num_decomps {
                log_pp.push(cp.get_log_precinct_size(r));
            }

            let mut tile_comp = TileComp::new(
                c,
                comp_rect,
                num_decomps,
                log_block_dims,
                &log_pp,
                cp.is_reversible(),
                siz.get_bit_depth(c),
                siz.is_signed(c),
            );
            for res in &mut tile_comp.resolutions {
                for sb in &mut res.subbands {
                    let subband_num = sb.band_type as u32;
                    sb.k_max = qp.get_kmax(num_decomps, sb.resolution_num, subband_num);
                    sb.reversible = cp.is_reversible();
                    if !sb.reversible {
                        let d = qp.get_irrev_delta(num_decomps, sb.resolution_num, subband_num);
                        sb.delta = d / ((1u32 << (31 - sb.k_max)) as f32);
                    } else {
                        sb.delta = 1.0;
                    }
                }
            }
            tile_comps.push(tile_comp);
        }

        // Compute number of tile lines from highest resolution
        let num_lines = if skipped_res_for_recon > 0 {
            let ds = 1u32 << skipped_res_for_recon;
            div_ceil(tile_rect.siz.h, ds)
        } else {
            tile_rect.siz.h
        };

        Self {
            tile_idx,
            tile_rect,
            tile_comps,
            num_comps,
            employ_color_transform: cod.is_employing_color_transform(),
            num_layers: cod.get_num_layers() as u32,
            num_tileparts: 1,
            may_use_sop: cod.packets_may_use_sop(),
            uses_eph: cod.packets_use_eph(),
            sot: ParamSot::default(),
            is_complete: false,
            cur_line: 0,
            num_lines,
            skipped_res_for_recon,
        }
    }

    /// Width of this tile.
    #[inline]
    pub fn width(&self) -> u32 {
        self.tile_rect.siz.w
    }

    /// Height of this tile.
    #[inline]
    pub fn height(&self) -> u32 {
        self.tile_rect.siz.h
    }

    /// Returns a reference to the tile component at the given index.
    pub fn get_comp(&self, comp: u32) -> Option<&TileComp> {
        self.tile_comps.get(comp as usize)
    }

    /// Returns a mutable reference to the tile component at the given index.
    pub fn get_comp_mut(&mut self, comp: u32) -> Option<&mut TileComp> {
        self.tile_comps.get_mut(comp as usize)
    }

    /// Returns the number of tileparts for this tile, considering divisions.
    pub fn compute_num_tileparts(&self, tilepart_div: u32) -> u32 {
        if tilepart_div == 0 {
            return 1;
        }
        let mut tps = 1u32;
        if tilepart_div & super::super::params::local::TILEPART_COMPONENTS != 0 {
            tps *= self.num_comps;
        }
        if tilepart_div & super::super::params::local::TILEPART_RESOLUTIONS != 0 {
            // Use the max num_resolutions across all components
            let max_res = self
                .tile_comps
                .iter()
                .map(|tc| tc.num_resolutions)
                .max()
                .unwrap_or(1);
            tps *= max_res;
        }
        tps
    }

    // ===================================================================
    // Encode path
    // ===================================================================

    /// Push a line of image data to the specified component.
    pub fn push_line(&mut self, line: &[i32], comp_num: u32) -> Result<()> {
        let tc = self.tile_comps.get_mut(comp_num as usize).ok_or_else(|| {
            OjphError::InvalidParam(format!("component {} out of range", comp_num))
        })?;
        tc.push_line(line);
        Ok(())
    }

    /// Encode and write this tile's data to the output file.
    ///
    /// Performs DWT on all components, encodes codeblocks, then writes
    /// SOT + SOD + packet data.
    pub fn encode_and_write(&mut self, file: &mut dyn OutfileBase) -> Result<()> {
        // 1. DWT + codeblock encoding for each component
        for tc in &mut self.tile_comps {
            tc.perform_dwt()?;
            for res in &mut tc.resolutions {
                for sb in &mut res.subbands {
                    sb.encode_codeblocks()?;
                }
            }
        }

        // 2. Build all packet data into a buffer
        let packet_data = self.build_all_packets()?;

        // 3. Write SOT — payload_len is the tile data size AFTER the SOD marker
        self.sot.init(0, self.tile_idx as u16, 0, 1);
        self.sot.write(file, packet_data.len() as u32)?;

        // 4. Write SOD
        file.write(&markers::SOD.to_be_bytes())?;

        // 5. Write packet data
        file.write(&packet_data)?;

        self.is_complete = true;
        Ok(())
    }

    /// Build all packets for this tile in LRCP order.
    fn build_all_packets(&self) -> Result<Vec<u8>> {
        let mut data = Vec::new();
        let num_layers = self.num_layers;
        let max_res = self
            .tile_comps
            .iter()
            .map(|tc| tc.num_resolutions)
            .max()
            .unwrap_or(1);

        // LRCP: Layer → Resolution → Component → Position(precinct)
        for _layer in 0..num_layers {
            for r in 0..max_res {
                for c in 0..self.num_comps {
                    let tc = &self.tile_comps[c as usize];
                    if r >= tc.num_resolutions {
                        // Write empty packet for this component at this resolution
                        data.push(0x00);
                        continue;
                    }
                    let res = &tc.resolutions[r as usize];
                    // One packet per precinct. For simplicity, 1 precinct per resolution.
                    let packet = Self::write_packet(res)?;
                    data.extend_from_slice(&packet);
                }
            }
        }
        Ok(data)
    }

    /// Write a single packet (header + body) for one precinct at a resolution.
    #[allow(clippy::incompatible_msrv)]
    fn write_packet(res: &super::resolution::Resolution) -> Result<Vec<u8>> {
        let mut header = BitBufferWrite::new();
        let mut body: Vec<u8> = Vec::new();
        let mut packet_started = false;
        let mut skipped_empty_subbands = 0u32;

        for sb in &res.subbands {
            if sb.is_empty() || sb.num_blocks_x == 0 || sb.num_blocks_y == 0 {
                continue;
            }

            let num_levels = 1 + log2ceil(sb.num_blocks_x).max(log2ceil(sb.num_blocks_y));
            let mut inc_tag = PacketTagTree::new(sb.num_blocks_x, sb.num_blocks_y, num_levels, 255);
            let mut inc_flags = PacketTagTree::new(sb.num_blocks_x, sb.num_blocks_y, num_levels, 0);
            let mut mmsb_tag =
                PacketTagTree::new(sb.num_blocks_x, sb.num_blocks_y, num_levels, 255);
            let mut mmsb_flags =
                PacketTagTree::new(sb.num_blocks_x, sb.num_blocks_y, num_levels, 0);

            for y in 0..sb.num_blocks_y {
                for x in 0..sb.num_blocks_x {
                    let cb_idx = (y * sb.num_blocks_x + x) as usize;
                    let enc = sb.codeblocks[cb_idx].enc_state.as_ref();
                    let empty = enc.is_none_or(|e| !e.has_data || e.num_passes == 0);
                    inc_tag.set(x, y, 0, u32::from(empty));
                    mmsb_tag.set(x, y, 0, enc.map_or(0, |e| e.missing_msbs));
                }
            }
            inc_tag.build_min_levels(sb.num_blocks_x, sb.num_blocks_y, num_levels);
            mmsb_tag.build_min_levels(sb.num_blocks_x, sb.num_blocks_y, num_levels);
            inc_flags.set(0, 0, num_levels, 0);
            mmsb_flags.set(0, 0, num_levels, 0);

            if inc_tag.get(0, 0, num_levels - 1) != 0 {
                if packet_started {
                    header.write(0, 1);
                } else {
                    skipped_empty_subbands += 1;
                }
                continue;
            }

            if !packet_started {
                header.write(1, 1);
                if skipped_empty_subbands > 0 {
                    header.write(0, skipped_empty_subbands);
                }
                packet_started = true;
            }

            for y in 0..sb.num_blocks_y {
                for x in 0..sb.num_blocks_x {
                    let cb_idx = (y * sb.num_blocks_x + x) as usize;
                    let cb = &sb.codeblocks[cb_idx];
                    let enc = cb.enc_state.as_ref();

                    for cur_lev in (1..=num_levels).rev() {
                        let levm1 = cur_lev - 1;
                        if inc_flags.get(x >> levm1, y >> levm1, levm1) == 0 {
                            let skipped = inc_tag.get(x >> levm1, y >> levm1, levm1)
                                - inc_tag.get(x >> cur_lev, y >> cur_lev, cur_lev);
                            debug_assert!(skipped <= 1);
                            header.write(1 - skipped, 1);
                            inc_flags.set(x >> levm1, y >> levm1, levm1, 1);
                        }
                        if inc_tag.get(x >> levm1, y >> levm1, levm1) > 0 {
                            break;
                        }
                    }

                    let Some(enc) = enc else {
                        continue;
                    };
                    if !enc.has_data || enc.num_passes == 0 {
                        continue;
                    }

                    for cur_lev in (1..=num_levels).rev() {
                        let levm1 = cur_lev - 1;
                        if mmsb_flags.get(x >> levm1, y >> levm1, levm1) == 0 {
                            let num_zeros = mmsb_tag.get(x >> levm1, y >> levm1, levm1)
                                - mmsb_tag.get(x >> cur_lev, y >> cur_lev, cur_lev);
                            for _ in 0..num_zeros {
                                header.write(0, 1);
                            }
                            header.write(1, 1);
                            mmsb_flags.set(x >> levm1, y >> levm1, levm1, 1);
                        }
                    }

                    match enc.num_passes {
                        1 => header.write(0, 1),
                        2 => header.write(0b10, 2),
                        3 => header.write(0b1100, 4),
                        other => {
                            return Err(OjphError::Codec {
                                code: 0x00060020,
                                message: format!(
                                    "packet writer does not yet support {} coding passes",
                                    other
                                ),
                            })
                        }
                    }

                    let bits1 = 32 - enc.pass1_bytes.leading_zeros();
                    let extra_bit = u32::from(enc.num_passes > 2);
                    let bits2 = if enc.num_passes > 1 {
                        32 - enc.pass2_bytes.leading_zeros()
                    } else {
                        0
                    };
                    let bits = bits1.max(bits2.saturating_sub(extra_bit)).saturating_sub(3);
                    for _ in 0..bits {
                        header.write(1, 1);
                    }
                    header.write(0, 1);

                    header.write(enc.pass1_bytes, bits + 3);
                    if enc.num_passes > 1 {
                        header.write(enc.pass2_bytes, bits + 3 + extra_bit);
                    }

                    let data_len = (enc.pass1_bytes + enc.pass2_bytes) as usize;
                    if data_len > cb.coded_data.len() {
                        return Err(OjphError::Codec {
                            code: 0x00060021,
                            message: "encoded codeblock body shorter than packet header length"
                                .into(),
                        });
                    }
                    body.extend_from_slice(&cb.coded_data[..data_len]);
                }
            }
        }

        if !packet_started {
            header.write(0, 1);
        }

        header.finalize();
        let mut packet = header.into_data();
        packet.extend_from_slice(&body);
        Ok(packet)
    }

    // ===================================================================
    // Decode path
    // ===================================================================

    /// Read tile data from the file (after SOT marker has been consumed).
    /// Expects the file position to be at the SOT payload (after the 0xFF90 marker).
    pub fn read_tile_data(&mut self, file: &mut dyn InfileBase) -> Result<()> {
        // Read SOT payload
        self.sot.read(file, false)?;

        // Compute how many bytes remain (including SOD marker)
        let payload_including_sod = self.sot.get_payload_length();
        if payload_including_sod < 2 {
            return Err(OjphError::Codec {
                code: 0x00060001,
                message: "tile payload too small for SOD marker".into(),
            });
        }

        // Read SOD marker
        let mut marker_buf = [0u8; 2];
        file.read(&mut marker_buf)?;
        let marker = u16::from_be_bytes(marker_buf);
        if marker != markers::SOD {
            return Err(OjphError::Codec {
                code: 0x00060002,
                message: format!("expected SOD marker (0xFF93), got 0x{:04X}", marker),
            });
        }

        // Read tile data
        let data_len = (payload_including_sod - 2) as usize;
        let mut tile_data = vec![0u8; data_len];
        if data_len > 0 {
            let mut read_so_far = 0;
            while read_so_far < data_len {
                let n = file.read(&mut tile_data[read_so_far..])?;
                if n == 0 {
                    break;
                }
                read_so_far += n;
            }
        }

        // Parse packets from tile data
        self.parse_all_packets(&tile_data)?;

        // Decode codeblocks and perform IDWT
        for tc in &mut self.tile_comps {
            for res in &mut tc.resolutions {
                for sb in &mut res.subbands {
                    sb.decode_codeblocks()?;
                }
            }
            tc.perform_idwt()?;
        }

        self.is_complete = true;
        Ok(())
    }

    /// Parse all packets from a tile data buffer (LRCP order).
    fn parse_all_packets(&mut self, data: &[u8]) -> Result<()> {
        let mut pos = 0usize;
        let num_layers = self.num_layers;
        let max_res = self
            .tile_comps
            .iter()
            .map(|tc| tc.num_resolutions)
            .max()
            .unwrap_or(1);

        for _layer in 0..num_layers {
            for r in 0..max_res {
                for c in 0..self.num_comps {
                    let tc = &self.tile_comps[c as usize];
                    if r >= tc.num_resolutions {
                        // Parse empty packet
                        if pos < data.len() {
                            pos += 1; // skip the 0x00 byte
                        }
                        continue;
                    }

                    let consumed = self.parse_packet(c, r, &data[pos..])?;
                    pos += consumed;
                }
            }
        }
        Ok(())
    }

    /// Parse a single packet from data. Returns number of bytes consumed.
    fn parse_packet(&mut self, comp: u32, res_num: u32, data: &[u8]) -> Result<usize> {
        if data.is_empty() {
            return Ok(0);
        }
        let mut reader = PacketBitReader::new(data);
        if self.may_use_sop {
            reader.skip_optional_sop()?;
        }

        // Collect codeblock info for body parsing
        struct CbInfo {
            sb_idx: usize,
            cb_idx: usize,
            pass1_len: u32,
            pass2_len: u32,
            missing_msbs: u32,
            num_passes: u32,
        }
        let mut cb_infos: Vec<CbInfo> = Vec::new();

        let tc = &self.tile_comps[comp as usize];
        let res = &tc.resolutions[res_num as usize];
        let mut saw_non_empty_packet = false;
        let mut packet_is_empty = false;

        for (sb_idx, sb) in res.subbands.iter().enumerate() {
            if sb.is_empty() || sb.num_blocks_x == 0 || sb.num_blocks_y == 0 {
                continue;
            }

            if !saw_non_empty_packet {
                let bit = reader.read_bit("error reading from packet header p0")?;
                if bit == 0 {
                    packet_is_empty = true;
                    break;
                }
                saw_non_empty_packet = true;
            }

            let num_levels = 1 + log2ceil(sb.num_blocks_x).max(log2ceil(sb.num_blocks_y));
            let mut inc_tag = PacketTagTree::new(sb.num_blocks_x, sb.num_blocks_y, num_levels, 0);
            let mut inc_flags = PacketTagTree::new(sb.num_blocks_x, sb.num_blocks_y, num_levels, 0);
            let mut mmsb_tag = PacketTagTree::new(sb.num_blocks_x, sb.num_blocks_y, num_levels, 0);
            let mut mmsb_flags =
                PacketTagTree::new(sb.num_blocks_x, sb.num_blocks_y, num_levels, 0);

            for y in 0..sb.num_blocks_y {
                for x in 0..sb.num_blocks_x {
                    let cb_idx = (y * sb.num_blocks_x + x) as usize;

                    let mut empty_cb = false;
                    for cl in (1..=num_levels).rev() {
                        let cur_lev = cl - 1;
                        empty_cb = inc_tag.get(x >> cur_lev, y >> cur_lev, cur_lev) == 1;
                        if empty_cb {
                            break;
                        }
                        if inc_flags.get(x >> cur_lev, y >> cur_lev, cur_lev) == 0 {
                            let bit = reader.read_bit("error reading from packet header p1")?;
                            empty_cb = bit == 0;
                            inc_tag.set(x >> cur_lev, y >> cur_lev, cur_lev, 1 - bit);
                            inc_flags.set(x >> cur_lev, y >> cur_lev, cur_lev, 1);
                        }
                        if empty_cb {
                            break;
                        }
                    }

                    if empty_cb {
                        continue;
                    }

                    let mut missing_msbs = 0u32;
                    for levp1 in (1..=num_levels).rev() {
                        let cur_lev = levp1 - 1;
                        missing_msbs = mmsb_tag.get(x >> levp1, y >> levp1, levp1);
                        if mmsb_flags.get(x >> cur_lev, y >> cur_lev, cur_lev) == 0 {
                            let mut bit = 0u32;
                            while bit == 0 {
                                bit = reader.read_bit("error reading from packet header p2")?;
                                missing_msbs += 1 - bit;
                            }
                            mmsb_tag.set(x >> cur_lev, y >> cur_lev, cur_lev, missing_msbs);
                            mmsb_flags.set(x >> cur_lev, y >> cur_lev, cur_lev, 1);
                        }
                    }

                    let pass_bit = reader.read_bit("error reading from packet header p3")?;
                    let mut num_passes = if pass_bit == 0 {
                        1u32
                    } else {
                        let bit2 = reader.read_bit("error reading from packet header p4")?;
                        if bit2 == 0 {
                            2
                        } else {
                            let extra =
                                reader.read_bits(2, "error reading from packet header p5")?;
                            let mut passes = 3 + extra;
                            if extra == 3 {
                                let extra5 =
                                    reader.read_bits(5, "error reading from packet header p6")?;
                                passes = 6 + extra5;
                                if extra5 == 31 {
                                    let extra7 = reader
                                        .read_bits(7, "error reading from packet header p7")?;
                                    passes = 37 + extra7;
                                }
                            }
                            passes
                        }
                    };

                    let num_phld_passes = (num_passes - 1) / 3;
                    missing_msbs += num_phld_passes;
                    num_passes -= num_phld_passes * 3;

                    let mut lblock = 3u32;
                    loop {
                        let bit = reader.read_bit("error reading from packet header p8")?;
                        if bit == 0 {
                            break;
                        }
                        lblock += 1;
                    }

                    let pass1_len = reader.read_bits(
                        lblock + 31 - count_leading_zeros(num_phld_passes + 1),
                        "error reading from packet header p9",
                    )?;
                    let mut pass2_len = 0u32;
                    if num_passes > 1 {
                        pass2_len = reader.read_bits(
                            lblock + if num_passes > 2 { 1 } else { 0 },
                            "error reading from packet header p10",
                        )?;
                    }

                    cb_infos.push(CbInfo {
                        sb_idx,
                        cb_idx,
                        pass1_len,
                        pass2_len,
                        missing_msbs,
                        num_passes,
                    });
                }
            }
        }

        if !saw_non_empty_packet && !packet_is_empty {
            let _ = reader.read_bit("error reading from packet header p11")?;
        }

        reader.terminate(self.uses_eph)?;

        if packet_is_empty {
            return Ok(reader.consumed());
        }

        for info in &cb_infos {
            let len = (info.pass1_len + info.pass2_len) as usize;
            let coded_data = reader.read_chunk(len);

            let tc = &mut self.tile_comps[comp as usize];
            let sb = &mut tc.resolutions[res_num as usize].subbands[info.sb_idx];
            let cb = &mut sb.codeblocks[info.cb_idx];
            cb.coded_data = coded_data;
            cb.dec_state = Some(CodeblockDecState {
                pass1_len: info.pass1_len,
                pass2_len: info.pass2_len,
                num_passes: info.num_passes,
                missing_msbs: info.missing_msbs,
            });
        }

        Ok(reader.consumed())
    }

    /// Pull a decoded line for the given component.
    pub fn pull_line(&mut self, comp_num: u32) -> Option<Vec<i32>> {
        self.tile_comps
            .get_mut(comp_num as usize)
            .and_then(|tc| tc.pull_line())
    }
}
