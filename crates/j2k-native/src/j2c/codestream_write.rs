// SPDX-License-Identifier: MIT OR Apache-2.0

//! JPEG 2000 codestream writer (ITU-T T.800 Annex A).
//!
//! Writes the complete codestream including all required markers:
//! SOC, SIZ, COD, QCD, SOT, SOD, EOC.

use alloc::vec::Vec;

use super::codestream::markers;
use super::encode::EncodeProgressionOrder;

mod accounting;
pub(crate) use self::accounting::AccountedCodestream;
use self::accounting::{codestream_tiles_output_len, tile_part_len};
mod packet_markers;
use self::packet_markers::{
    write_plm_markers, write_plt_markers, write_ppm_markers, write_ppt_markers,
};
#[cfg(test)]
use self::packet_markers::{PACKET_HEADER_MARKER_PAYLOAD_LIMIT, PPM_PACKET_HEADER_LIMIT};
use super::encode::allocation::host_allocation_failed;
use crate::{EncodeError, EncodeResult};

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
#[derive(Debug)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "independent codestream marker and coding switches remain explicit in the assembly job"
)]
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

/// Test oracle for writing a complete single-tile JPEG 2000 codestream.
#[cfg(test)]
pub(crate) fn write_codestream(
    params: &EncodeParams,
    tile_data: &[u8],
    quantization_step_sizes: &[(u16, u16)], // (exponent, mantissa)
) -> Result<Vec<u8>, &'static str> {
    write_codestream_with_packet_lengths(params, tile_data, quantization_step_sizes, &[])
}

/// Write one tile while reporting the peak of writer-owned heap capacities.
/// Borrowed parameters, quantization values, and tile bytes are intentionally
/// excluded so the calling encode session can count each retained owner once.
/// The peak check runs before reservation and again with allocator-returned
/// capacity before marker writing starts.
pub(crate) fn write_codestream_accounted_with_peak_check(
    params: &EncodeParams,
    tile_data: &[u8],
    quantization_step_sizes: &[(u16, u16)],
    check_writer_peak: impl FnMut(usize) -> EncodeResult<()>,
) -> EncodeResult<AccountedCodestream> {
    let tile = TilePartData {
        tile_index: 0,
        tile_part_index: 0,
        num_tile_parts: 1,
        data: tile_data,
        packet_lengths: &[],
        packet_headers: &[],
    };
    write_codestream_tiles_accounted_with_peak_check(
        params,
        &[tile],
        quantization_step_sizes,
        check_writer_peak,
    )
}

#[cfg(test)]
pub(crate) fn write_codestream_with_packet_lengths(
    params: &EncodeParams,
    tile_data: &[u8],
    quantization_step_sizes: &[(u16, u16)], // (exponent, mantissa)
    packet_lengths: &[u32],
) -> Result<Vec<u8>, &'static str> {
    let tile = TilePartData {
        tile_index: 0,
        tile_part_index: 0,
        num_tile_parts: 1,
        data: tile_data,
        packet_lengths,
        packet_headers: &[],
    };
    write_codestream_tiles_accounted_with_peak_check(
        params,
        &[tile],
        quantization_step_sizes,
        |_| Ok(()),
    )
    .map(|accounted| accounted.codestream)
    .map_err(legacy_writer_error)
}

/// Write any number of tile-parts under one pre-allocation and
/// allocator-returned writer peak contract.
pub(crate) fn write_codestream_tiles_accounted_with_peak_check(
    params: &EncodeParams,
    tiles: &[TilePartData<'_>],
    quantization_step_sizes: &[(u16, u16)],
    mut check_writer_peak: impl FnMut(usize) -> EncodeResult<()>,
) -> EncodeResult<AccountedCodestream> {
    let output_len = codestream_tiles_output_len(params, tiles, quantization_step_sizes)?;
    check_writer_peak(output_len)?;
    let mut out = Vec::new();
    out.try_reserve_exact(output_len)
        .map_err(|_| host_allocation_failed("codestream output", output_len))?;
    let output_capacity = out.capacity();
    check_writer_peak(output_capacity)?;

    write_main_header_prefix(&mut out, params, quantization_step_sizes)
        .map_err(|what| EncodeError::InvalidInput { what })?;
    if params.write_plm {
        write_plm_markers(&mut out, tiles)?;
    }
    if params.write_ppm {
        write_ppm_markers(&mut out, tiles)?;
    }
    if params.write_tlm {
        for tile in tiles {
            write_tlm_marker(&mut out, tile.tile_index, tile_part_len(params, tile)?);
        }
    }
    for tile in tiles {
        let tile_part_len = tile_part_len(params, tile)?;
        write_sot_marker(
            &mut out,
            tile.tile_index,
            tile_part_len - 2,
            tile.tile_part_index,
            tile.num_tile_parts,
        );
        if params.write_plt {
            write_plt_markers(&mut out, tile.packet_lengths)?;
        }
        if params.write_ppt {
            write_ppt_markers(&mut out, tile.packet_headers)?;
        }
        write_marker(&mut out, markers::SOD);
        out.extend_from_slice(tile.data);
    }
    write_marker(&mut out, markers::EOC);
    if out.len() != output_len || out.capacity() != output_capacity {
        return Err(EncodeError::InternalInvariant {
            what: "accounted codestream length changed after exact preflight",
        });
    }
    Ok(AccountedCodestream {
        codestream: out,
        writer_peak_bytes: output_capacity,
    })
}

#[cfg(test)]
fn legacy_writer_error(error: EncodeError) -> &'static str {
    match error {
        EncodeError::InvalidInput { what }
        | EncodeError::Unsupported { what }
        | EncodeError::ArithmeticOverflow { what }
        | EncodeError::InternalInvariant { what } => what,
        EncodeError::HostAllocationFailed { .. } => "codestream output allocation failed",
        EncodeError::AllocationTooLarge { .. } => "codestream output exceeds allocation cap",
        EncodeError::Accelerator { source, .. } => source.reason(),
        EncodeError::CodestreamValidation { detail } => detail,
    }
}

fn write_main_header_prefix(
    out: &mut Vec<u8>,
    params: &EncodeParams,
    quantization_step_sizes: &[(u16, u16)],
) -> Result<(), &'static str> {
    write_marker(out, markers::SOC);
    write_siz_marker(out, params);
    if params.block_coding_mode == BlockCodingMode::HighThroughput {
        write_cap_marker(out, params);
    }
    write_cod_marker(out, params);
    write_qcd_marker(out, params, quantization_step_sizes)?;
    write_qcc_markers(out, params)?;
    write_rgn_markers(out, params);
    Ok(())
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
    let magnitude_bits = u16::from(params.max_component_bit_depth().saturating_sub(1));
    let bp = if magnitude_bits <= 8 {
        0
    } else if magnitude_bits < 28 {
        magnitude_bits - 8
    } else {
        13 + (magnitude_bits >> 2)
    };

    let wavelet_flag = if params.reversible { 0u16 } else { 0x0020u16 };
    wavelet_flag | bp
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
    out.extend_from_slice(&u16::from(params.num_layers).to_be_bytes());
    // Multiple component transform
    out.push(u8::from(params.use_mct));

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
    out.push(u8::from(params.reversible));

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
    progression_order
        .packetization_order()
        .codestream_order_code()
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

/// Write QCD marker segment (A.6.4).
fn write_qcd_marker(
    out: &mut Vec<u8>,
    params: &EncodeParams,
    step_sizes: &[(u16, u16)],
) -> Result<(), &'static str> {
    if params.reversible {
        // No quantization: Sqcd = 0x00, then exponent bytes
        let step_count = u16::try_from(step_sizes.len())
            .map_err(|_| "QCD step-size count exceeds marker capacity")?;
        let marker_len = 3u16
            .checked_add(step_count)
            .ok_or("QCD marker length exceeds u16")?;
        if step_sizes.iter().any(|&(exponent, _)| exponent > 0x1f) {
            return Err("QCD exponent exceeds five bits");
        }
        write_marker(out, markers::QCD);
        out.extend_from_slice(&marker_len.to_be_bytes());

        // Sqcd: no quantization (style 0), guard bits in upper 3 bits
        out.push(params.guard_bits << 5);

        // SPqcd: one byte per subband (exponent in upper 5 bits, mantissa = 0)
        for &(exp, _) in step_sizes {
            let exponent = u8::try_from(exp).map_err(|_| "QCD exponent exceeds eight bits")?;
            out.push(exponent << 3);
        }
    } else {
        // Scalar expounded: Sqcd = 0x02, then 2 bytes per subband
        let step_bytes = u16::try_from(step_sizes.len())
            .map_err(|_| "QCD step-size count exceeds marker capacity")?
            .checked_mul(2)
            .ok_or("QCD step-size byte length exceeds u16")?;
        let marker_len = 3u16
            .checked_add(step_bytes)
            .ok_or("QCD marker length exceeds u16")?;
        write_marker(out, markers::QCD);
        out.extend_from_slice(&marker_len.to_be_bytes());

        // Sqcd: scalar expounded quantization, guard bits
        out.push((params.guard_bits << 5) | 0x02);

        // SPqcd: two bytes per subband (5-bit exponent + 11-bit mantissa)
        for &(exp, mant) in step_sizes {
            let val = ((exp & 0x1F) << 11) | (mant & 0x7FF);
            out.extend_from_slice(&val.to_be_bytes());
        }
    }
    Ok(())
}

fn write_qcc_markers(out: &mut Vec<u8>, params: &EncodeParams) -> Result<(), &'static str> {
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
        write_qcc_marker(out, params, component_index, step_sizes)?;
    }
    Ok(())
}

fn write_qcc_marker(
    out: &mut Vec<u8>,
    params: &EncodeParams,
    component_index: u16,
    step_sizes: &[(u16, u16)],
) -> Result<(), &'static str> {
    let component_index_len = if params.num_components < 257 {
        1_u16
    } else {
        2_u16
    };
    let step_count = u16::try_from(step_sizes.len())
        .map_err(|_| "QCC step-size count exceeds marker capacity")?;
    let step_bytes = if params.reversible {
        step_count
    } else {
        step_count
            .checked_mul(2)
            .ok_or("QCC step-size byte length exceeds u16")?
    };
    let marker_len = 3u16
        .checked_add(component_index_len)
        .and_then(|length| length.checked_add(step_bytes))
        .ok_or("QCC marker length exceeds u16")?;
    if params.reversible && step_sizes.iter().any(|&(exponent, _)| exponent > 0x1f) {
        return Err("QCC exponent exceeds five bits");
    }
    write_marker(out, markers::QCC);
    out.extend_from_slice(&marker_len.to_be_bytes());
    if params.num_components < 257 {
        out.push(u8::try_from(component_index).map_err(|_| "QCC component index exceeds u8")?);
    } else {
        out.extend_from_slice(&component_index.to_be_bytes());
    }

    if params.reversible {
        out.push(params.guard_bits << 5);
        for &(exp, _) in step_sizes {
            let exponent = u8::try_from(exp).map_err(|_| "QCC exponent exceeds eight bits")?;
            out.push(exponent << 3);
        }
    } else {
        out.push((params.guard_bits << 5) | 0x02);
        for &(exp, mant) in step_sizes {
            let val = ((exp & 0x1F) << 11) | (mant & 0x7FF);
            out.extend_from_slice(&val.to_be_bytes());
        }
    }
    Ok(())
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
        let codestream =
            write_codestream(&params, &tile_data, &step_sizes).expect("valid test codestream");

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
    fn reversible_quantization_errors_are_returned() {
        let params = EncodeParams {
            reversible: true,
            ..Default::default()
        };

        assert_eq!(
            write_codestream(&params, &[], &[(32, 0)]),
            Err("QCD exponent exceeds five bits")
        );

        let oversized_steps = vec![(1, 0); usize::from(u16::MAX)];
        assert_eq!(
            write_codestream(&params, &[], &oversized_steps),
            Err("QCD marker length exceeds u16")
        );
    }

    #[test]
    fn component_quantization_errors_are_returned() {
        let params = EncodeParams {
            reversible: true,
            component_quantization_step_sizes: vec![vec![(32, 0)]],
            ..Default::default()
        };

        assert_eq!(
            write_codestream(&params, &[], &[(1, 0)]),
            Err("QCC exponent exceeds five bits")
        );
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

        let codestream = write_codestream(&params, &[0], &[(8, 0)]).expect("valid test codestream");
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

        let codestream =
            write_codestream(&params, &[0], &[(8, 0), (7, 0)]).expect("valid test codestream");
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

        let codestream = write_codestream(&params, &[0], &[(8, 0)]).expect("valid test codestream");
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
        let tile = TilePartData {
            tile_index: 0,
            tile_part_index: 0,
            num_tile_parts: 1,
            data: &[],
            packet_lengths: &[],
            packet_headers: &headers,
        };

        write_ppm_markers(&mut out, &[tile]).expect("valid PPM markers");

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

        write_ppt_markers(&mut out, &headers).expect("valid PPT markers");

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
        let codestream =
            write_codestream(&params, &tile_data, &step_sizes).expect("valid test codestream");

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
        let codestream =
            write_codestream(&params, &tile_data, &step_sizes).expect("valid test codestream");

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
