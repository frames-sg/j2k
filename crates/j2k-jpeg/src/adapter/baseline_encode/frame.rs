// SPDX-License-Identifier: MIT OR Apache-2.0

//! Exact baseline JPEG marker, table-segment, and frame assembly.

use super::tables::{
    JPEG_BASELINE_ZIGZAG, STD_CHROMA_AC_BITS, STD_CHROMA_AC_VALUES, STD_CHROMA_DC_BITS,
    STD_CHROMA_DC_VALUES, STD_LUMA_AC_BITS, STD_LUMA_AC_VALUES, STD_LUMA_DC_BITS,
    STD_LUMA_DC_VALUES,
};
use super::types::{JpegBaselineEncodeTables, JpegBaselineSampling};
use super::validation::{
    validate_jpeg_baseline_dimensions, validate_jpeg_baseline_restart_interval,
};
use crate::encoded_output::{checked_jpeg_baseline_frame_capacity, CappedBytes};
use crate::encoder::{EncodedJpeg, JpegBackend, JpegEncodeError, JpegEncodeOptions};

/// Assemble a complete baseline JPEG codestream from entropy bytes and tables.
pub fn assemble_jpeg_baseline_frame(
    entropy: &[u8],
    width: u32,
    height: u32,
    tables: &JpegBaselineEncodeTables,
    options: JpegEncodeOptions,
    backend: JpegBackend,
) -> Result<EncodedJpeg, JpegEncodeError> {
    validate_jpeg_baseline_dimensions(width, height)?;
    validate_jpeg_baseline_restart_interval(options.restart_interval)?;

    let frame_capacity = checked_jpeg_baseline_frame_capacity(entropy.len())?;
    let mut out = CappedBytes::try_with_capacity(frame_capacity, frame_capacity)?;
    write_marker(&mut out, 0xD8)?;
    write_dqt(&mut out, 0, &tables.q_luma)?;
    if tables.sampling.components == 3 {
        write_dqt(&mut out, 1, &tables.q_chroma)?;
    }
    if let Some(restart_interval) = options.restart_interval {
        write_dri(&mut out, restart_interval)?;
    }
    write_sof0(&mut out, width, height, tables.sampling)?;
    write_dht(&mut out, 0, 0, &STD_LUMA_DC_BITS, &STD_LUMA_DC_VALUES)?;
    write_dht(&mut out, 1, 0, &STD_LUMA_AC_BITS, &STD_LUMA_AC_VALUES)?;
    if tables.sampling.components == 3 {
        write_dht(&mut out, 0, 1, &STD_CHROMA_DC_BITS, &STD_CHROMA_DC_VALUES)?;
        write_dht(&mut out, 1, 1, &STD_CHROMA_AC_BITS, &STD_CHROMA_AC_VALUES)?;
    }
    write_sos(&mut out, tables.sampling.components)?;
    out.extend_from_slice(entropy)?;
    write_marker(&mut out, 0xD9)?;

    Ok(EncodedJpeg {
        data: out.into_vec(),
        backend,
    })
}

pub(crate) fn assemble_jpeg_baseline_frame_with_quant_tables(
    entropy: &[u8],
    width: u32,
    height: u32,
    sampling: JpegBaselineSampling,
    q_luma: &[u8; 64],
    q_chroma: Option<&[u8; 64]>,
    backend: JpegBackend,
) -> Result<EncodedJpeg, JpegEncodeError> {
    validate_jpeg_baseline_dimensions(width, height)?;

    let frame_capacity = checked_jpeg_baseline_frame_capacity(entropy.len())?;
    let mut out = CappedBytes::try_with_capacity(frame_capacity, frame_capacity)?;
    write_marker(&mut out, 0xD8)?;
    write_dqt(&mut out, 0, q_luma)?;
    if sampling.components == 3 {
        let q_chroma = q_chroma.ok_or(JpegEncodeError::InternalInvariant {
            reason: "three-component DCT JPEG requires chroma quant table",
        })?;
        write_dqt(&mut out, 1, q_chroma)?;
    }
    write_sof0(&mut out, width, height, sampling)?;
    write_dht(&mut out, 0, 0, &STD_LUMA_DC_BITS, &STD_LUMA_DC_VALUES)?;
    write_dht(&mut out, 1, 0, &STD_LUMA_AC_BITS, &STD_LUMA_AC_VALUES)?;
    if sampling.components == 3 {
        write_dht(&mut out, 0, 1, &STD_CHROMA_DC_BITS, &STD_CHROMA_DC_VALUES)?;
        write_dht(&mut out, 1, 1, &STD_CHROMA_AC_BITS, &STD_CHROMA_AC_VALUES)?;
    }
    write_sos(&mut out, sampling.components)?;
    out.extend_from_slice(entropy)?;
    write_marker(&mut out, 0xD9)?;

    Ok(EncodedJpeg {
        data: out.into_vec(),
        backend,
    })
}

fn write_marker(out: &mut CappedBytes, marker: u8) -> Result<(), JpegEncodeError> {
    out.push(0xFF)?;
    out.push(marker)
}

fn write_segment(
    out: &mut CappedBytes,
    marker: u8,
    payload: &[u8],
    name: &'static str,
) -> Result<(), JpegEncodeError> {
    write_segment_header(out, marker, payload.len(), name)?;
    out.extend_from_slice(payload)
}

fn write_segment_header(
    out: &mut CappedBytes,
    marker: u8,
    payload_len: usize,
    name: &'static str,
) -> Result<(), JpegEncodeError> {
    let len = payload_len
        .checked_add(2)
        .and_then(|value| u16::try_from(value).ok())
        .ok_or(JpegEncodeError::SegmentTooLarge { name })?;
    write_marker(out, marker)?;
    out.extend_from_slice(&len.to_be_bytes())?;
    Ok(())
}

fn write_dqt(out: &mut CappedBytes, table_id: u8, quant: &[u8; 64]) -> Result<(), JpegEncodeError> {
    write_segment_header(out, 0xDB, 65, "DQT")?;
    out.push(table_id)?;
    for &natural_idx in &JPEG_BASELINE_ZIGZAG {
        out.push(quant[natural_idx as usize])?;
    }
    Ok(())
}

fn write_dri(out: &mut CappedBytes, restart_interval: u16) -> Result<(), JpegEncodeError> {
    write_segment(out, 0xDD, &restart_interval.to_be_bytes(), "DRI")
}

fn write_sof0(
    out: &mut CappedBytes,
    width: u32,
    height: u32,
    sampling: JpegBaselineSampling,
) -> Result<(), JpegEncodeError> {
    let height =
        u16::try_from(height).map_err(|_| JpegEncodeError::DimensionsTooLarge { width, height })?;
    let width = u16::try_from(width).map_err(|_| JpegEncodeError::DimensionsTooLarge {
        width,
        height: u32::from(height),
    })?;
    write_segment_header(out, 0xC0, 6 + usize::from(sampling.components) * 3, "SOF0")?;
    out.push(8)?;
    out.extend_from_slice(&height.to_be_bytes())?;
    out.extend_from_slice(&width.to_be_bytes())?;
    out.push(sampling.components)?;
    for component in 0..sampling.components as usize {
        let component_id =
            u8::try_from(component + 1).map_err(|_| JpegEncodeError::InternalInvariant {
                reason: "JPEG component id exceeds u8",
            })?;
        out.push(component_id)?;
        out.push((sampling.h[component] << 4) | sampling.v[component])?;
        out.push(u8::from(component != 0))?;
    }
    Ok(())
}

fn write_dht(
    out: &mut CappedBytes,
    class: u8,
    table_id: u8,
    bits: &[u8; 16],
    values: &[u8],
) -> Result<(), JpegEncodeError> {
    write_segment_header(out, 0xC4, 17 + values.len(), "DHT")?;
    out.push((class << 4) | table_id)?;
    out.extend_from_slice(bits)?;
    out.extend_from_slice(values)
}

fn write_sos(out: &mut CappedBytes, components: u8) -> Result<(), JpegEncodeError> {
    write_segment_header(out, 0xDA, 4 + usize::from(components) * 2, "SOS")?;
    out.push(components)?;
    for component in 0..components {
        out.push(component + 1)?;
        out.push(if component == 0 { 0x00 } else { 0x11 })?;
    }
    out.push(0)?;
    out.push(63)?;
    out.push(0)
}
