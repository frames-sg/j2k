// SPDX-License-Identifier: MIT OR Apache-2.0

//! Exact baseline JPEG marker, table-segment, and frame assembly.

use alloc::vec::Vec;

use super::tables::{
    JPEG_BASELINE_ZIGZAG, STD_CHROMA_AC_BITS, STD_CHROMA_AC_VALUES, STD_CHROMA_DC_BITS,
    STD_CHROMA_DC_VALUES, STD_LUMA_AC_BITS, STD_LUMA_AC_VALUES, STD_LUMA_DC_BITS,
    STD_LUMA_DC_VALUES,
};
use super::types::{JpegBaselineEncodeTables, JpegBaselineSampling};
use super::validation::{
    validate_jpeg_baseline_dimensions, validate_jpeg_baseline_restart_interval,
};
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

    let mut out = Vec::with_capacity(768usize.saturating_add(entropy.len()));
    write_marker(&mut out, 0xD8);
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
    out.extend_from_slice(entropy);
    write_marker(&mut out, 0xD9);

    Ok(EncodedJpeg { data: out, backend })
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

    let mut out = Vec::with_capacity(768usize.saturating_add(entropy.len()));
    write_marker(&mut out, 0xD8);
    write_dqt(&mut out, 0, q_luma)?;
    if sampling.components == 3 {
        let q_chroma = q_chroma.ok_or_else(|| {
            JpegEncodeError::Internal("three-component DCT JPEG requires chroma quant table".into())
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
    out.extend_from_slice(entropy);
    write_marker(&mut out, 0xD9);

    Ok(EncodedJpeg { data: out, backend })
}

fn write_marker(out: &mut Vec<u8>, marker: u8) {
    out.push(0xFF);
    out.push(marker);
}

fn write_segment(
    out: &mut Vec<u8>,
    marker: u8,
    payload: &[u8],
    name: &'static str,
) -> Result<(), JpegEncodeError> {
    let len = payload
        .len()
        .checked_add(2)
        .and_then(|value| u16::try_from(value).ok())
        .ok_or(JpegEncodeError::SegmentTooLarge { name })?;
    write_marker(out, marker);
    out.extend_from_slice(&len.to_be_bytes());
    out.extend_from_slice(payload);
    Ok(())
}

fn write_dqt(out: &mut Vec<u8>, table_id: u8, quant: &[u8; 64]) -> Result<(), JpegEncodeError> {
    let mut payload = Vec::with_capacity(65);
    payload.push(table_id);
    for &natural_idx in &JPEG_BASELINE_ZIGZAG {
        payload.push(quant[natural_idx as usize]);
    }
    write_segment(out, 0xDB, &payload, "DQT")
}

fn write_dri(out: &mut Vec<u8>, restart_interval: u16) -> Result<(), JpegEncodeError> {
    write_segment(out, 0xDD, &restart_interval.to_be_bytes(), "DRI")
}

fn write_sof0(
    out: &mut Vec<u8>,
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
    let mut payload = Vec::with_capacity(6 + sampling.components as usize * 3);
    payload.push(8);
    payload.extend_from_slice(&height.to_be_bytes());
    payload.extend_from_slice(&width.to_be_bytes());
    payload.push(sampling.components);
    for component in 0..sampling.components as usize {
        let component_id = u8::try_from(component + 1)
            .map_err(|_| JpegEncodeError::Internal("JPEG component id exceeds u8".into()))?;
        payload.push(component_id);
        payload.push((sampling.h[component] << 4) | sampling.v[component]);
        payload.push(u8::from(component != 0));
    }
    write_segment(out, 0xC0, &payload, "SOF0")
}

fn write_dht(
    out: &mut Vec<u8>,
    class: u8,
    table_id: u8,
    bits: &[u8; 16],
    values: &[u8],
) -> Result<(), JpegEncodeError> {
    let mut payload = Vec::with_capacity(17 + values.len());
    payload.push((class << 4) | table_id);
    payload.extend_from_slice(bits);
    payload.extend_from_slice(values);
    write_segment(out, 0xC4, &payload, "DHT")
}

fn write_sos(out: &mut Vec<u8>, components: u8) -> Result<(), JpegEncodeError> {
    let mut payload = Vec::with_capacity(4 + components as usize * 2);
    payload.push(components);
    for component in 0..components {
        payload.push(component + 1);
        payload.push(if component == 0 { 0x00 } else { 0x11 });
    }
    payload.push(0);
    payload.push(63);
    payload.push(0);
    write_segment(out, 0xDA, &payload, "SOS")
}
