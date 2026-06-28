//! JPEG 2000 codestream writer (ITU-T T.800 Annex A).
//!
//! Writes the complete codestream including all required markers:
//! SOC, SIZ, COD, QCD, SOT, SOD, EOC.

use alloc::vec::Vec;

use super::codestream::markers;
use super::encode::EncodeProgressionOrder;

const HT_RSIZ_CAPABILITY: u16 = 0x4000;

/// Per-component SIZ sample metadata for codestream writing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct EncodeComponentSampleInfo {
    pub(crate) bit_depth: u8,
    pub(crate) signed: bool,
}

/// Code-block coding mode for the codestream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BlockCodingMode {
    /// Classic JPEG 2000 Part 1 EBCOT block coding.
    Classic,
    /// High-throughput JPEG 2000 Part 15 block coding.
    HighThroughput,
}

/// Parameters for encoding a JPEG 2000 codestream.
#[derive(Debug, Clone)]
pub(crate) struct EncodeParams {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) tile_width: u32,
    pub(crate) tile_height: u32,
    pub(crate) num_components: u16,
    pub(crate) bit_depth: u8,
    pub(crate) signed: bool,
    pub(crate) component_sample_info: Vec<EncodeComponentSampleInfo>,
    pub(crate) component_quantization_step_sizes: Vec<Vec<(u16, u16)>>,
    pub(crate) num_decomposition_levels: u8,
    pub(crate) reversible: bool,
    pub(crate) code_block_width_exp: u8,
    pub(crate) code_block_height_exp: u8,
    pub(crate) num_layers: u8,
    pub(crate) use_mct: bool,
    pub(crate) guard_bits: u8,
    pub(crate) block_coding_mode: BlockCodingMode,
    pub(crate) progression_order: EncodeProgressionOrder,
    pub(crate) write_tlm: bool,
    pub(crate) write_plt: bool,
    pub(crate) write_plm: bool,
    pub(crate) write_ppm: bool,
    pub(crate) write_ppt: bool,
    pub(crate) write_sop: bool,
    pub(crate) write_eph: bool,
    pub(crate) terminate_coding_passes: bool,
    pub(crate) component_sampling: Vec<(u8, u8)>,
    pub(crate) roi_component_shifts: Vec<u8>,
    pub(crate) precinct_exponents: Vec<(u8, u8)>,
}

impl Default for EncodeParams {
    fn default() -> Self {
        Self {
            width: 0,
            height: 0,
            tile_width: 0,
            tile_height: 0,
            num_components: 1,
            bit_depth: 8,
            signed: false,
            component_sample_info: Vec::new(),
            component_quantization_step_sizes: Vec::new(),
            num_decomposition_levels: 5,
            reversible: true,
            code_block_width_exp: 4, // 2^(4+2) = 64
            code_block_height_exp: 4,
            num_layers: 1,
            use_mct: false,
            guard_bits: 1,
            block_coding_mode: BlockCodingMode::Classic,
            progression_order: EncodeProgressionOrder::Lrcp,
            write_tlm: false,
            write_plt: false,
            write_plm: false,
            write_ppm: false,
            write_ppt: false,
            write_sop: false,
            write_eph: false,
            terminate_coding_passes: false,
            component_sampling: Vec::new(),
            roi_component_shifts: Vec::new(),
            precinct_exponents: Vec::new(),
        }
    }
}

impl EncodeParams {
    fn sample_info_for_component(&self, component_index: u16) -> EncodeComponentSampleInfo {
        self.component_sample_info
            .get(usize::from(component_index))
            .copied()
            .unwrap_or(EncodeComponentSampleInfo {
                bit_depth: self.bit_depth,
                signed: self.signed,
            })
    }

    fn max_component_bit_depth(&self) -> u8 {
        (0..self.num_components)
            .map(|component_index| self.sample_info_for_component(component_index).bit_depth)
            .max()
            .unwrap_or(self.bit_depth)
    }
}

pub(crate) struct TilePartData<'a> {
    pub(crate) tile_index: u16,
    pub(crate) tile_part_index: u8,
    pub(crate) num_tile_parts: u8,
    pub(crate) data: &'a [u8],
    pub(crate) packet_lengths: &'a [u32],
    pub(crate) packet_headers: &'a [Vec<u8>],
}

/// Write the complete JPEG 2000 codestream.
pub(crate) fn write_codestream(
    params: &EncodeParams,
    tile_data: &[u8],
    quantization_step_sizes: &[(u16, u16)], // (exponent, mantissa)
) -> Vec<u8> {
    write_codestream_with_packet_lengths(params, tile_data, quantization_step_sizes, &[])
}

pub(crate) fn write_codestream_with_packet_lengths(
    params: &EncodeParams,
    tile_data: &[u8],
    quantization_step_sizes: &[(u16, u16)], // (exponent, mantissa)
    packet_lengths: &[u32],
) -> Vec<u8> {
    let tile = TilePartData {
        tile_index: 0,
        tile_part_index: 0,
        num_tile_parts: 1,
        data: tile_data,
        packet_lengths,
        packet_headers: &[],
    };
    write_codestream_tiles(params, &[tile], quantization_step_sizes)
}

pub(crate) fn write_codestream_tiles(
    params: &EncodeParams,
    tiles: &[TilePartData<'_>],
    quantization_step_sizes: &[(u16, u16)], // (exponent, mantissa)
) -> Vec<u8> {
    struct PreparedTilePart<'a> {
        tile_index: u16,
        tile_part_index: u8,
        num_tile_parts: u8,
        data: &'a [u8],
        markers: Vec<u8>,
        tile_part_len: u32,
    }

    let mut prepared_tiles = Vec::with_capacity(tiles.len());
    let mut main_header_packet_lengths = Vec::new();
    let mut main_header_packet_headers = Vec::new();
    let mut total_tile_bytes = 0usize;
    for tile in tiles {
        let mut markers = Vec::new();
        if params.write_plt && !tile.packet_lengths.is_empty() {
            write_plt_markers(&mut markers, tile.packet_lengths);
        }
        if params.write_ppt && !tile.packet_headers.is_empty() {
            write_ppt_markers(&mut markers, tile.packet_headers);
        }
        if params.write_plm {
            main_header_packet_lengths.extend_from_slice(tile.packet_lengths);
        }
        if params.write_ppm {
            main_header_packet_headers.extend_from_slice(tile.packet_headers);
        }
        let tile_part_len = 14
            + u32::try_from(markers.len()).unwrap_or(u32::MAX)
            + u32::try_from(tile.data.len()).unwrap_or(u32::MAX);
        total_tile_bytes = total_tile_bytes
            .saturating_add(markers.len())
            .saturating_add(tile.data.len())
            .saturating_add(14);
        prepared_tiles.push(PreparedTilePart {
            tile_index: tile.tile_index,
            tile_part_index: tile.tile_part_index,
            num_tile_parts: tile.num_tile_parts,
            data: tile.data,
            markers,
            tile_part_len,
        });
    }

    let mut out = Vec::with_capacity(total_tile_bytes + 256);

    // SOC (Start of codestream)
    write_marker(&mut out, markers::SOC);

    // SIZ (Image and tile sizes)
    write_siz_marker(&mut out, params);

    if params.block_coding_mode == BlockCodingMode::HighThroughput {
        write_cap_marker(&mut out, params);
    }

    // COD (Coding style defaults)
    write_cod_marker(&mut out, params);

    // QCD (Quantization defaults)
    write_qcd_marker(&mut out, params, quantization_step_sizes);
    write_qcc_markers(&mut out, params);

    write_rgn_markers(&mut out, params);

    if params.write_plm && !main_header_packet_lengths.is_empty() {
        write_plm_markers(&mut out, &main_header_packet_lengths);
    }

    if params.write_ppm && !main_header_packet_headers.is_empty() {
        write_ppm_markers(&mut out, &main_header_packet_headers);
    }

    if params.write_tlm {
        for tile in &prepared_tiles {
            write_tlm_marker(&mut out, tile.tile_index, tile.tile_part_len);
        }
    }

    for tile in prepared_tiles {
        write_sot_marker(
            &mut out,
            tile.tile_index,
            tile.tile_part_len - 2,
            tile.tile_part_index,
            tile.num_tile_parts,
        );
        out.extend_from_slice(&tile.markers);
        write_marker(&mut out, markers::SOD);
        out.extend_from_slice(tile.data);
    }

    // EOC (End of codestream)
    write_marker(&mut out, markers::EOC);

    out
}

fn write_marker(out: &mut Vec<u8>, marker: u8) {
    out.push(0xFF);
    out.push(marker);
}

/// Write SIZ marker segment (A.5.1).
fn write_siz_marker(out: &mut Vec<u8>, params: &EncodeParams) {
    write_marker(out, markers::SIZ);

    let num_comp = params.num_components;
    let marker_len = 38 + 3 * num_comp;

    // Lsiz
    out.extend_from_slice(&marker_len.to_be_bytes());
    // Rsiz (capabilities). Part 15 codestreams used in JPH files must set the
    // HT capability bit so strict HTJ2K decoders recognize the codestream.
    let rsiz = match params.block_coding_mode {
        BlockCodingMode::Classic => 0,
        BlockCodingMode::HighThroughput => HT_RSIZ_CAPABILITY,
    };
    out.extend_from_slice(&rsiz.to_be_bytes());
    // Xsiz (reference grid width)
    out.extend_from_slice(&params.width.to_be_bytes());
    // Ysiz (reference grid height)
    out.extend_from_slice(&params.height.to_be_bytes());
    // XOsiz (image area x offset)
    out.extend_from_slice(&0u32.to_be_bytes());
    // YOsiz (image area y offset)
    out.extend_from_slice(&0u32.to_be_bytes());
    let tile_width = if params.tile_width == 0 {
        params.width
    } else {
        params.tile_width
    };
    let tile_height = if params.tile_height == 0 {
        params.height
    } else {
        params.tile_height
    };
    // XTsiz (tile width)
    out.extend_from_slice(&tile_width.to_be_bytes());
    // YTsiz (tile height)
    out.extend_from_slice(&tile_height.to_be_bytes());
    // XTOsiz (tile x offset)
    out.extend_from_slice(&0u32.to_be_bytes());
    // YTOsiz (tile y offset)
    out.extend_from_slice(&0u32.to_be_bytes());
    // Csiz (number of components)
    out.extend_from_slice(&num_comp.to_be_bytes());

    // Per-component info
    for component_index in 0..params.num_components {
        let sample_info = params.sample_info_for_component(component_index);
        // Ssiz: bit depth - 1 (unsigned) or bit depth - 1 + 0x80 (signed)
        let ssiz = if sample_info.signed {
            (sample_info.bit_depth - 1) | 0x80
        } else {
            sample_info.bit_depth - 1
        };
        out.push(ssiz);
        let (x_rsiz, y_rsiz) = params
            .component_sampling
            .get(usize::from(component_index))
            .copied()
            .unwrap_or((1, 1));
        // XRsiz (horizontal sampling factor)
        out.push(x_rsiz);
        // YRsiz (vertical sampling factor)
        out.push(y_rsiz);
    }
}

/// Write CAP marker segment (Part 15 extended capabilities).
fn write_cap_marker(out: &mut Vec<u8>, params: &EncodeParams) {
    write_marker(out, markers::CAP);
    out.extend_from_slice(&8u16.to_be_bytes());
    out.extend_from_slice(&0x0002_0000u32.to_be_bytes());
    out.extend_from_slice(&ht_capability_word(params).to_be_bytes());
}

fn ht_capability_word(params: &EncodeParams) -> u16 {
    let magnitude_bits = u32::from(params.max_component_bit_depth().saturating_sub(1));
    let bp = if magnitude_bits <= 8 {
        0
    } else if magnitude_bits < 28 {
        magnitude_bits - 8
    } else {
        13 + (magnitude_bits >> 2)
    };

    let wavelet_flag = if params.reversible { 0u16 } else { 0x0020u16 };
    wavelet_flag | (bp as u16)
}

/// Write COD marker segment (A.6.1).
fn write_cod_marker(out: &mut Vec<u8>, params: &EncodeParams) {
    write_marker(out, markers::COD);

    let marker_len = 12u16
        + u16::try_from(params.precinct_exponents.len())
            .expect("precinct exponent count fits in COD marker length");
    out.extend_from_slice(&marker_len.to_be_bytes());

    // Scod (coding style flags)
    let mut scod = 0u8;
    if !params.precinct_exponents.is_empty() {
        scod |= 0x01;
    }
    if params.write_sop {
        scod |= 0x02;
    }
    if params.write_eph {
        scod |= 0x04;
    }
    out.push(scod);

    // SGcod: Progression order
    out.push(progression_order_byte(params.progression_order));
    // Number of layers
    out.extend_from_slice(&(params.num_layers as u16).to_be_bytes());
    // Multiple component transform
    out.push(if params.use_mct { 1 } else { 0 });

    // SPcod: Number of decomposition levels
    out.push(params.num_decomposition_levels);
    // Code-block width exponent - 2
    out.push(params.code_block_width_exp);
    // Code-block height exponent - 2
    out.push(params.code_block_height_exp);
    // Code-block style
    let code_block_style = u8::from(params.terminate_coding_passes) << 2
        | match params.block_coding_mode {
            BlockCodingMode::Classic => 0x00,
            BlockCodingMode::HighThroughput => 0x40,
        };
    out.push(code_block_style);
    // Wavelet transform: 0 = irreversible 9-7, 1 = reversible 5-3
    out.push(if params.reversible { 1 } else { 0 });

    for &(ppx, ppy) in &params.precinct_exponents {
        out.push((ppy << 4) | ppx);
    }
}

fn write_rgn_markers(out: &mut Vec<u8>, params: &EncodeParams) {
    for component_index in 0..params.num_components {
        let Some(&shift) = params
            .roi_component_shifts
            .get(usize::from(component_index))
        else {
            continue;
        };
        if shift == 0 {
            continue;
        }
        write_marker(out, markers::RGN);
        if params.num_components < 257 {
            out.extend_from_slice(&5u16.to_be_bytes());
            out.push(u8::try_from(component_index).expect("component index fits in Crgn byte"));
        } else {
            out.extend_from_slice(&6u16.to_be_bytes());
            out.extend_from_slice(&component_index.to_be_bytes());
        }
        out.push(0);
        out.push(shift);
    }
}

fn progression_order_byte(progression_order: EncodeProgressionOrder) -> u8 {
    match progression_order {
        EncodeProgressionOrder::Lrcp => 0x00,
        EncodeProgressionOrder::Rlcp => 0x01,
        EncodeProgressionOrder::Rpcl => 0x02,
        EncodeProgressionOrder::Pcrl => 0x03,
        EncodeProgressionOrder::Cprl => 0x04,
    }
}

/// Write TLM marker segment (A.7.1) for one tile-part.
fn write_tlm_marker(out: &mut Vec<u8>, tile_index: u16, tile_part_length: u32) {
    write_marker(out, markers::TLM);
    out.extend_from_slice(&10u16.to_be_bytes());
    out.push(0);
    out.push(0x22);
    out.extend_from_slice(&tile_index.to_be_bytes());
    out.extend_from_slice(&tile_part_length.to_be_bytes());
}

fn write_plt_markers(out: &mut Vec<u8>, packet_lengths: &[u32]) {
    let data = packet_length_bytes(packet_lengths);
    for (sequence_idx, chunk) in data.chunks(usize::from(u16::MAX) - 3).enumerate() {
        write_marker(out, markers::PLT);
        let marker_len = u16::try_from(3 + chunk.len()).expect("PLT marker chunk length fits");
        out.extend_from_slice(&marker_len.to_be_bytes());
        out.push(sequence_idx as u8);
        out.extend_from_slice(chunk);
    }
}

fn write_plm_markers(out: &mut Vec<u8>, packet_lengths: &[u32]) {
    let data = packet_length_bytes(packet_lengths);
    for (sequence_idx, chunk) in data.chunks(usize::from(u16::MAX) - 7).enumerate() {
        write_marker(out, markers::PLM);
        let marker_len = u16::try_from(7 + chunk.len()).expect("PLM marker chunk length fits");
        out.extend_from_slice(&marker_len.to_be_bytes());
        out.push(sequence_idx as u8);
        out.extend_from_slice(&u32::try_from(chunk.len()).unwrap_or(u32::MAX).to_be_bytes());
        out.extend_from_slice(chunk);
    }
}

const PACKET_HEADER_MARKER_PAYLOAD_LIMIT: usize = u16::MAX as usize - 3;
const PPM_PACKET_HEADER_LIMIT: usize = PACKET_HEADER_MARKER_PAYLOAD_LIMIT - 2;

fn write_ppm_markers(out: &mut Vec<u8>, packet_headers: &[Vec<u8>]) {
    let mut sequence_idx = 0usize;
    let mut start = 0usize;
    let mut payload_len = 0usize;

    for (idx, header) in packet_headers.iter().enumerate() {
        let entry_len = 2usize
            .checked_add(header.len())
            .expect("PPM packet header length fits usize");
        assert!(
            header.len() <= PPM_PACKET_HEADER_LIMIT,
            "PPM packet header length fits marker payload"
        );
        if payload_len > 0 && payload_len + entry_len > PACKET_HEADER_MARKER_PAYLOAD_LIMIT {
            write_ppm_marker_chunk(out, sequence_idx, &packet_headers[start..idx], payload_len);
            sequence_idx += 1;
            start = idx;
            payload_len = 0;
        }
        payload_len += entry_len;
    }

    if payload_len > 0 {
        write_ppm_marker_chunk(out, sequence_idx, &packet_headers[start..], payload_len);
    }
}

fn write_ppm_marker_chunk(
    out: &mut Vec<u8>,
    sequence_idx: usize,
    packet_headers: &[Vec<u8>],
    payload_len: usize,
) {
    write_marker(out, markers::PPM);
    let marker_len = u16::try_from(3 + payload_len).expect("PPM marker length fits");
    out.extend_from_slice(&marker_len.to_be_bytes());
    out.push(u8::try_from(sequence_idx).expect("PPM marker sequence index fits u8"));
    for header in packet_headers {
        let header_len = u16::try_from(header.len()).expect("PPM packet header length fits");
        out.extend_from_slice(&header_len.to_be_bytes());
        out.extend_from_slice(header);
    }
}

fn write_ppt_markers(out: &mut Vec<u8>, packet_headers: &[Vec<u8>]) {
    let mut chunk = Vec::new();
    let mut sequence_idx = 0usize;

    for header in packet_headers {
        let mut remaining = header.as_slice();
        while !remaining.is_empty() {
            let available = PACKET_HEADER_MARKER_PAYLOAD_LIMIT - chunk.len();
            let take = available.min(remaining.len());
            chunk.extend_from_slice(&remaining[..take]);
            remaining = &remaining[take..];
            if chunk.len() == PACKET_HEADER_MARKER_PAYLOAD_LIMIT {
                write_ppt_marker_chunk(out, sequence_idx, &chunk);
                sequence_idx += 1;
                chunk.clear();
            }
        }
    }

    if !chunk.is_empty() {
        write_ppt_marker_chunk(out, sequence_idx, &chunk);
    }
}

fn write_ppt_marker_chunk(out: &mut Vec<u8>, sequence_idx: usize, payload: &[u8]) {
    write_marker(out, markers::PPT);
    let marker_len = u16::try_from(3 + payload.len()).expect("PPT marker length fits");
    out.extend_from_slice(&marker_len.to_be_bytes());
    out.push(u8::try_from(sequence_idx).expect("PPT marker sequence index fits u8"));
    out.extend_from_slice(payload);
}

fn packet_length_bytes(packet_lengths: &[u32]) -> Vec<u8> {
    let mut out = Vec::new();

    for &packet_length in packet_lengths {
        let mut value = packet_length;
        let mut groups = Vec::new();
        groups.push((value & 0x7F) as u8);
        value >>= 7;

        while value > 0 {
            groups.push((value & 0x7F) as u8);
            value >>= 7;
        }

        for (idx, group) in groups.iter().rev().enumerate() {
            let continuation = idx + 1 != groups.len();
            out.push(if continuation { *group | 0x80 } else { *group });
        }
    }

    out
}

/// Write QCD marker segment (A.6.4).
fn write_qcd_marker(out: &mut Vec<u8>, params: &EncodeParams, step_sizes: &[(u16, u16)]) {
    write_marker(out, markers::QCD);

    if params.reversible {
        // No quantization: Sqcd = 0x00, then exponent bytes
        let marker_len = 3 + step_sizes.len() as u16;
        out.extend_from_slice(&marker_len.to_be_bytes());

        // Sqcd: no quantization (style 0), guard bits in upper 3 bits
        out.push(params.guard_bits << 5);

        // SPqcd: one byte per subband (exponent in upper 5 bits, mantissa = 0)
        for &(exp, _) in step_sizes {
            out.push((exp as u8) << 3);
        }
    } else {
        // Scalar expounded: Sqcd = 0x02, then 2 bytes per subband
        let marker_len = 3 + step_sizes.len() as u16 * 2;
        out.extend_from_slice(&marker_len.to_be_bytes());

        // Sqcd: scalar expounded quantization, guard bits
        out.push((params.guard_bits << 5) | 0x02);

        // SPqcd: two bytes per subband (5-bit exponent + 11-bit mantissa)
        for &(exp, mant) in step_sizes {
            let val = ((exp & 0x1F) << 11) | (mant & 0x7FF);
            out.extend_from_slice(&val.to_be_bytes());
        }
    }
}

fn write_qcc_markers(out: &mut Vec<u8>, params: &EncodeParams) {
    for component_index in 0..params.num_components {
        let Some(step_sizes) = params
            .component_quantization_step_sizes
            .get(usize::from(component_index))
        else {
            continue;
        };
        if step_sizes.is_empty() {
            continue;
        }
        write_qcc_marker(out, params, component_index, step_sizes);
    }
}

fn write_qcc_marker(
    out: &mut Vec<u8>,
    params: &EncodeParams,
    component_index: u16,
    step_sizes: &[(u16, u16)],
) {
    write_marker(out, markers::QCC);

    let component_index_len = if params.num_components < 257 {
        1_u16
    } else {
        2_u16
    };
    let step_bytes = if params.reversible {
        step_sizes.len() as u16
    } else {
        step_sizes.len() as u16 * 2
    };
    let marker_len = 3 + component_index_len + step_bytes;
    out.extend_from_slice(&marker_len.to_be_bytes());
    if params.num_components < 257 {
        out.push(component_index as u8);
    } else {
        out.extend_from_slice(&component_index.to_be_bytes());
    }

    if params.reversible {
        out.push(params.guard_bits << 5);
        for &(exp, _) in step_sizes {
            out.push((exp as u8) << 3);
        }
    } else {
        out.push((params.guard_bits << 5) | 0x02);
        for &(exp, mant) in step_sizes {
            let val = ((exp & 0x1F) << 11) | (mant & 0x7FF);
            out.extend_from_slice(&val.to_be_bytes());
        }
    }
}

/// Write SOT marker segment (A.4.2).
fn write_sot_marker(
    out: &mut Vec<u8>,
    tile_index: u16,
    tile_part_length: u32,
    tile_part_index: u8,
    num_tile_parts: u8,
) {
    write_marker(out, markers::SOT);

    // Lsot = 10
    out.extend_from_slice(&10u16.to_be_bytes());
    // Isot (tile index)
    out.extend_from_slice(&tile_index.to_be_bytes());
    // Psot (tile-part length including SOT marker)
    out.extend_from_slice(&(tile_part_length + 2).to_be_bytes()); // +2 for SOT marker bytes
                                                                  // TPsot (tile-part index)
    out.push(tile_part_index);
    // TNsot (number of tile-parts, 0 = unknown)
    out.push(num_tile_parts);
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::{vec, vec::Vec};

    fn find_marker_offset(codestream: &[u8], marker: u8) -> Option<usize> {
        codestream
            .windows(2)
            .position(|window| window == [0xFF, marker])
    }

    #[test]
    fn test_write_minimal_codestream() {
        let params = EncodeParams {
            width: 8,
            height: 8,
            num_components: 1,
            bit_depth: 8,
            num_decomposition_levels: 1,
            reversible: true,
            num_layers: 1,
            ..Default::default()
        };

        let tile_data = vec![0u8; 10];
        let step_sizes = vec![(9u16, 0u16), (8, 0), (8, 0), (7, 0)];
        let codestream = write_codestream(&params, &tile_data, &step_sizes);

        // Verify SOC marker
        assert_eq!(codestream[0], 0xFF);
        assert_eq!(codestream[1], markers::SOC);

        // Verify SIZ marker
        assert_eq!(codestream[2], 0xFF);
        assert_eq!(codestream[3], markers::SIZ);
        assert_eq!(&codestream[6..8], &[0x00, 0x00]);

        // Verify EOC marker
        let len = codestream.len();
        assert_eq!(codestream[len - 2], 0xFF);
        assert_eq!(codestream[len - 1], markers::EOC);
    }

    #[test]
    fn test_ht_capability_word_matches_fixture_examples() {
        let params = EncodeParams {
            bit_depth: 11,
            reversible: true,
            block_coding_mode: BlockCodingMode::HighThroughput,
            ..Default::default()
        };
        assert_eq!(ht_capability_word(&params), 0x0002);

        let params = EncodeParams {
            bit_depth: 12,
            reversible: true,
            block_coding_mode: BlockCodingMode::HighThroughput,
            ..Default::default()
        };
        assert_eq!(ht_capability_word(&params), 0x0003);

        let params = EncodeParams {
            bit_depth: 12,
            reversible: false,
            block_coding_mode: BlockCodingMode::HighThroughput,
            ..Default::default()
        };
        assert_eq!(ht_capability_word(&params), 0x0023);
    }

    #[test]
    fn ht_capability_word_uses_max_component_precision() {
        let params = EncodeParams {
            num_components: 2,
            bit_depth: 8,
            component_sample_info: vec![
                EncodeComponentSampleInfo {
                    bit_depth: 8,
                    signed: false,
                },
                EncodeComponentSampleInfo {
                    bit_depth: 12,
                    signed: true,
                },
            ],
            reversible: true,
            block_coding_mode: BlockCodingMode::HighThroughput,
            ..Default::default()
        };
        assert_eq!(ht_capability_word(&params), 0x0003);
    }

    #[test]
    fn write_siz_marker_uses_per_component_sample_info() {
        let params = EncodeParams {
            width: 4,
            height: 4,
            num_components: 3,
            bit_depth: 8,
            signed: false,
            component_sample_info: vec![
                EncodeComponentSampleInfo {
                    bit_depth: 8,
                    signed: false,
                },
                EncodeComponentSampleInfo {
                    bit_depth: 12,
                    signed: true,
                },
                EncodeComponentSampleInfo {
                    bit_depth: 38,
                    signed: false,
                },
            ],
            component_sampling: vec![(1, 1), (2, 1), (1, 2)],
            num_decomposition_levels: 0,
            reversible: true,
            num_layers: 1,
            ..Default::default()
        };

        let codestream = write_codestream(&params, &[0], &[(8, 0)]);
        let siz_offset = find_marker_offset(&codestream, markers::SIZ).expect("SIZ marker");
        let component_base = siz_offset + 40;
        assert_eq!(
            &codestream[component_base..component_base + 9],
            &[7, 1, 1, 0x8B, 2, 1, 37, 1, 2,]
        );
    }

    #[test]
    fn write_qcc_marker_for_component_quantization_override() {
        let params = EncodeParams {
            width: 4,
            height: 4,
            num_components: 2,
            bit_depth: 8,
            component_quantization_step_sizes: vec![Vec::new(), vec![(12, 0), (11, 0)]],
            num_decomposition_levels: 0,
            reversible: true,
            num_layers: 1,
            ..Default::default()
        };

        let codestream = write_codestream(&params, &[0], &[(8, 0), (7, 0)]);
        let qcc_offset = find_marker_offset(&codestream, markers::QCC).expect("QCC marker");
        assert_eq!(
            &codestream[qcc_offset..qcc_offset + 8],
            &[0xFF, markers::QCC, 0x00, 0x06, 0x01, 0x20, 0x60, 0x58]
        );
    }

    #[test]
    fn written_qcc_marker_overrides_component_quantization_on_parse() {
        let params = EncodeParams {
            width: 4,
            height: 4,
            num_components: 2,
            bit_depth: 8,
            component_quantization_step_sizes: vec![Vec::new(), vec![(12, 0)]],
            num_decomposition_levels: 0,
            reversible: true,
            num_layers: 1,
            ..Default::default()
        };

        let codestream = write_codestream(&params, &[0], &[(8, 0)]);
        let parsed = crate::j2c::parse_raw(&codestream, &crate::DecodeSettings::default())
            .expect("parse written codestream");
        assert_eq!(
            parsed.header.component_infos[0]
                .quantization_info
                .step_sizes[0]
                .exponent,
            8
        );
        assert_eq!(
            parsed.header.component_infos[1]
                .quantization_info
                .step_sizes[0]
                .exponent,
            12
        );
    }

    #[test]
    fn ppm_marker_writer_splits_at_packet_header_boundaries() {
        let headers = vec![
            vec![0x11; PPM_PACKET_HEADER_LIMIT],
            vec![0x22; 1],
            vec![0x33; 4],
        ];
        let mut out = Vec::new();

        write_ppm_markers(&mut out, &headers);

        let offsets = marker_offsets(&out, markers::PPM);
        assert_eq!(offsets.len(), 2);
        assert_eq!(marker_length(&out, offsets[0]), u16::MAX);
        assert_eq!(out[offsets[0] + 4], 0);
        assert_eq!(out[offsets[1] + 4], 1);
        assert_eq!(marker_length(&out, offsets[1]), 12);
        assert_eq!(
            &out[offsets[1] + 5..offsets[1] + 14],
            &[0, 1, 0x22, 0, 4, 0x33, 0x33, 0x33, 0x33]
        );
    }

    #[test]
    fn ppt_marker_writer_splits_large_payloads() {
        let headers = vec![vec![0x44; PACKET_HEADER_MARKER_PAYLOAD_LIMIT + 10]];
        let mut out = Vec::new();

        write_ppt_markers(&mut out, &headers);

        let offsets = marker_offsets(&out, markers::PPT);
        assert_eq!(offsets.len(), 2);
        assert_eq!(marker_length(&out, offsets[0]), u16::MAX);
        assert_eq!(out[offsets[0] + 4], 0);
        assert_eq!(marker_length(&out, offsets[1]), 13);
        assert_eq!(out[offsets[1] + 4], 1);
        assert!(out[offsets[1] + 5..offsets[1] + 15]
            .iter()
            .all(|byte| *byte == 0x44));
    }

    #[test]
    fn test_write_ht_lossless_codestream_headers() {
        let params = EncodeParams {
            width: 3,
            height: 5,
            num_components: 1,
            bit_depth: 12,
            num_decomposition_levels: 1,
            reversible: true,
            num_layers: 1,
            block_coding_mode: BlockCodingMode::HighThroughput,
            ..Default::default()
        };

        let tile_data = vec![0u8; 1];
        let step_sizes = vec![(12u16, 0u16), (13, 0), (13, 0), (14, 0)];
        let codestream = write_codestream(&params, &tile_data, &step_sizes);

        let siz_offset = find_marker_offset(&codestream, markers::SIZ).expect("SIZ marker");
        assert_eq!(
            &codestream[siz_offset + 4..siz_offset + 6],
            &HT_RSIZ_CAPABILITY.to_be_bytes()
        );

        let cap_offset = find_marker_offset(&codestream, markers::CAP).expect("CAP marker");
        let cap_len = u16::from_be_bytes([codestream[cap_offset + 2], codestream[cap_offset + 3]]);
        assert_eq!(cap_len, 8);
        assert_eq!(
            &codestream[cap_offset + 4..cap_offset + 10],
            &[0x00, 0x02, 0x00, 0x00, 0x00, 0x03]
        );

        let cod_offset = find_marker_offset(&codestream, markers::COD).expect("COD marker");
        assert_eq!(codestream[cod_offset + 12], 0x40);
        assert!(find_marker_offset(&codestream, markers::CPF).is_none());
    }

    #[test]
    fn test_write_rgb_codestream() {
        let params = EncodeParams {
            width: 16,
            height: 16,
            num_components: 3,
            bit_depth: 8,
            num_decomposition_levels: 2,
            reversible: true,
            use_mct: true,
            num_layers: 1,
            ..Default::default()
        };

        let tile_data = vec![0u8; 50];
        let step_sizes: Vec<(u16, u16)> = (0..7).map(|i| (9 - i / 3, 0)).collect();
        let codestream = write_codestream(&params, &tile_data, &step_sizes);

        // Should start with SOC and end with EOC
        assert_eq!(codestream[0], 0xFF);
        assert_eq!(codestream[1], markers::SOC);
        let len = codestream.len();
        assert_eq!(codestream[len - 2], 0xFF);
        assert_eq!(codestream[len - 1], markers::EOC);
    }

    fn marker_offsets(codestream: &[u8], marker: u8) -> Vec<usize> {
        codestream
            .windows(2)
            .enumerate()
            .filter_map(|(idx, window)| (window == [0xFF, marker]).then_some(idx))
            .collect()
    }

    fn marker_length(codestream: &[u8], offset: usize) -> u16 {
        u16::from_be_bytes([codestream[offset + 2], codestream[offset + 3]])
    }
}
