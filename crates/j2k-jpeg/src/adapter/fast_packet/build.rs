// SPDX-License-Identifier: MIT OR Apache-2.0

mod gray;
mod materialization;

use super::allocation::{checked_color_packet_initial_live_bytes, checked_color_packet_live_bytes};
use super::checkpoints::{build_fast_entropy_checkpoints, inspect_fast_entropy_checkpoints};
use super::entropy::{
    extract_entropy_segments_from_layout, inspect_entropy_segments_allow_missing_eoi,
    EntropySegments,
};
use super::error::FastPacketError;
use super::header::{ColorFastHeader, FastLayout, FAST420_LAYOUT, FAST422_LAYOUT, FAST444_LAYOUT};
use super::types::{
    JpegEntropyCheckpointV1, JpegFast420PacketV1, JpegFast422PacketV1, JpegFast444PacketV1,
    JpegHuffmanTable,
};
use crate::adapter::device_plan::retained_decoder_allocation_bytes;
use crate::decoder::{Decoder, JpegView};
use crate::internal::checkpoint::validate_scan_bytes;
use alloc::borrow::Cow;
use alloc::vec::Vec;
pub use gray::build_gray_packet;
use j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES;
use materialization::scan_live_bytes;

#[derive(Debug, PartialEq, Eq)]
struct ColorFastPacketParts {
    dimensions: (u32, u32),
    mcus_per_row: u32,
    mcu_rows: u32,
    restart_interval_mcus: u32,
    restart_offsets: Vec<u32>,
    entropy_checkpoints: Vec<JpegEntropyCheckpointV1>,
    y_quant: [u16; 64],
    cb_quant: [u16; 64],
    cr_quant: [u16; 64],
    y_dc_table: JpegHuffmanTable,
    y_ac_table: JpegHuffmanTable,
    cb_dc_table: JpegHuffmanTable,
    cb_ac_table: JpegHuffmanTable,
    cr_dc_table: JpegHuffmanTable,
    cr_ac_table: JpegHuffmanTable,
    entropy_bytes: Vec<u8>,
}

macro_rules! impl_from_color_fast_packet_parts {
    ($packet:ty) => {
        impl From<ColorFastPacketParts> for $packet {
            fn from(parts: ColorFastPacketParts) -> Self {
                Self {
                    dimensions: parts.dimensions,
                    mcus_per_row: parts.mcus_per_row,
                    mcu_rows: parts.mcu_rows,
                    restart_interval_mcus: parts.restart_interval_mcus,
                    restart_offsets: parts.restart_offsets,
                    entropy_checkpoints: parts.entropy_checkpoints,
                    y_quant: parts.y_quant,
                    cb_quant: parts.cb_quant,
                    cr_quant: parts.cr_quant,
                    y_dc_table: parts.y_dc_table,
                    y_ac_table: parts.y_ac_table,
                    cb_dc_table: parts.cb_dc_table,
                    cb_ac_table: parts.cb_ac_table,
                    cr_dc_table: parts.cr_dc_table,
                    cr_ac_table: parts.cr_ac_table,
                    entropy_bytes: parts.entropy_bytes,
                }
            }
        }
    };
}

impl_from_color_fast_packet_parts!(JpegFast420PacketV1);
impl_from_color_fast_packet_parts!(JpegFast422PacketV1);
impl_from_color_fast_packet_parts!(JpegFast444PacketV1);

/// Build a 4:2:0 fast-path packet from JPEG bytes.
#[doc(hidden)]
pub fn build_fast420_packet(bytes: &[u8]) -> Result<JpegFast420PacketV1, FastPacketError> {
    build_color_fast_packet(bytes, FAST420_LAYOUT).map(Into::into)
}

/// Build a 4:4:4 fast-path packet from JPEG bytes.
#[doc(hidden)]
pub fn build_fast444_packet(bytes: &[u8]) -> Result<JpegFast444PacketV1, FastPacketError> {
    build_color_fast_packet(bytes, FAST444_LAYOUT).map(Into::into)
}

/// Build a 4:2:2 fast-path packet from JPEG bytes.
#[doc(hidden)]
pub fn build_fast422_packet(bytes: &[u8]) -> Result<JpegFast422PacketV1, FastPacketError> {
    build_color_fast_packet(bytes, FAST422_LAYOUT).map(Into::into)
}

fn build_color_fast_packet(
    bytes: &[u8],
    layout: FastLayout,
) -> Result<ColorFastPacketParts, FastPacketError> {
    let view = JpegView::parse(bytes)?;
    let header = ColorFastHeader::inspect(view.parsed_header(), layout)?;
    let decoder = Decoder::from_view(view)?;
    build_color_fast_packet_from_decoder(bytes, &decoder, header, 0)
}

pub(super) fn build_fast420_packet_from_decoder(
    bytes: &[u8],
    decoder: &Decoder<'_>,
    header: ColorFastHeader,
    external_live_bytes: usize,
) -> Result<JpegFast420PacketV1, FastPacketError> {
    build_color_fast_packet_from_decoder(bytes, decoder, header, external_live_bytes)
        .map(Into::into)
}

pub(super) fn build_fast422_packet_from_decoder(
    bytes: &[u8],
    decoder: &Decoder<'_>,
    header: ColorFastHeader,
    external_live_bytes: usize,
) -> Result<JpegFast422PacketV1, FastPacketError> {
    build_color_fast_packet_from_decoder(bytes, decoder, header, external_live_bytes)
        .map(Into::into)
}

pub(super) fn build_fast444_packet_from_decoder(
    bytes: &[u8],
    decoder: &Decoder<'_>,
    header: ColorFastHeader,
    external_live_bytes: usize,
) -> Result<JpegFast444PacketV1, FastPacketError> {
    build_color_fast_packet_from_decoder(bytes, decoder, header, external_live_bytes)
        .map(Into::into)
}

fn build_color_fast_packet_from_decoder(
    bytes: &[u8],
    decoder: &Decoder<'_>,
    header: ColorFastHeader,
    external_live_bytes: usize,
) -> Result<ColorFastPacketParts, FastPacketError> {
    let validated_scan = validate_scan_bytes(
        &bytes[header.entropy_offset..],
        header.restart_interval.is_some_and(|interval| interval > 0),
        header.entropy_offset,
    )?;
    let entropy_layout = inspect_entropy_segments_allow_missing_eoi(
        validated_scan.payload(),
        header.restart_interval,
    )?;
    let checkpoint_layout = inspect_fast_entropy_checkpoints(decoder, header.total_mcus);
    let retained_decoder_bytes = retained_decoder_allocation_bytes(decoder)?;
    let initial_live_bytes = checked_color_packet_initial_live_bytes(
        external_live_bytes,
        retained_decoder_bytes,
        DEFAULT_MAX_HOST_ALLOCATION_BYTES,
    )?;
    let terminated_copy_bytes =
        validated_scan.terminated_copy_len(DEFAULT_MAX_HOST_ALLOCATION_BYTES)?;
    checked_color_packet_live_bytes(
        initial_live_bytes,
        entropy_layout.entropy_len,
        entropy_layout.restart_count,
        checkpoint_layout.checkpoint_count,
        terminated_copy_bytes,
        DEFAULT_MAX_HOST_ALLOCATION_BYTES,
    )?;

    let mut live_bytes = initial_live_bytes;
    let entropy_checkpoints = build_fast_entropy_checkpoints(
        decoder,
        validated_scan,
        checkpoint_layout,
        &mut live_bytes,
        DEFAULT_MAX_HOST_ALLOCATION_BYTES,
    )?;
    let terminated_scan = validated_scan
        .terminated_with_live_budget(live_bytes, DEFAULT_MAX_HOST_ALLOCATION_BYTES)?;
    let mut materialization_live_bytes = scan_live_bytes(
        live_bytes,
        match &terminated_scan {
            Cow::Borrowed(_) => None,
            Cow::Owned(owned) => Some(owned.capacity()),
        },
    )?;
    let EntropySegments {
        entropy_bytes,
        restart_offsets,
    } = extract_entropy_segments_from_layout(
        terminated_scan.as_ref(),
        header.restart_interval,
        entropy_layout,
        &mut materialization_live_bytes,
        DEFAULT_MAX_HOST_ALLOCATION_BYTES,
    )?;
    let restart_interval_mcus = u32::from(header.restart_interval.unwrap_or(0));

    Ok(ColorFastPacketParts {
        dimensions: header.dimensions,
        mcus_per_row: header.mcus_per_row,
        mcu_rows: header.mcu_rows,
        restart_interval_mcus,
        restart_offsets,
        entropy_checkpoints,
        y_quant: header.y_quant,
        cb_quant: header.cb_quant,
        cr_quant: header.cr_quant,
        y_dc_table: header.y_dc_table,
        y_ac_table: header.y_ac_table,
        cb_dc_table: header.cb_dc_table,
        cb_ac_table: header.cb_ac_table,
        cr_dc_table: header.cr_dc_table,
        cr_ac_table: header.cr_ac_table,
        entropy_bytes,
    })
}
