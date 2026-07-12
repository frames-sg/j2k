// SPDX-License-Identifier: MIT OR Apache-2.0

//! Grayscale fast-packet inspection and materialization.

use alloc::borrow::Cow;

use super::super::allocation::checked_gray_packet_live_bytes;
use super::super::entropy::{
    extract_entropy_segments_from_layout, inspect_entropy_segments_allow_missing_eoi,
    EntropySegments,
};
use super::super::error::FastPacketError;
use super::super::header::GrayFastHeader;
use super::super::types::JpegGrayPacketV1;
use super::materialization::scan_live_bytes;
use crate::decoder::JpegView;
use crate::internal::checkpoint::validate_scan_bytes;
use j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES;

/// Build a grayscale fast-path packet from JPEG bytes.
#[doc(hidden)]
pub fn build_gray_packet(bytes: &[u8]) -> Result<JpegGrayPacketV1, FastPacketError> {
    let view = JpegView::parse(bytes)?;
    let header = GrayFastHeader::inspect(view.parsed_header())?;
    drop(view);
    let validated_scan = validate_scan_bytes(
        &bytes[header.entropy_offset..],
        header.restart_interval.is_some_and(|interval| interval > 0),
        header.entropy_offset,
    )?;
    let entropy_layout = inspect_entropy_segments_allow_missing_eoi(
        validated_scan.payload(),
        header.restart_interval,
    )?;
    let terminated_copy_bytes =
        validated_scan.terminated_copy_len(DEFAULT_MAX_HOST_ALLOCATION_BYTES)?;
    checked_gray_packet_live_bytes(
        entropy_layout.entropy_len,
        entropy_layout.restart_count,
        terminated_copy_bytes,
        DEFAULT_MAX_HOST_ALLOCATION_BYTES,
    )?;
    let terminated_scan =
        validated_scan.terminated_with_live_budget(0, DEFAULT_MAX_HOST_ALLOCATION_BYTES)?;
    let mut live_bytes = scan_live_bytes(
        0,
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
        &mut live_bytes,
        DEFAULT_MAX_HOST_ALLOCATION_BYTES,
    )?;
    let restart_interval_mcus = u32::from(header.restart_interval.unwrap_or(0));
    let (width, height) = header.dimensions;

    Ok(JpegGrayPacketV1 {
        dimensions: header.dimensions,
        mcus_per_row: width.div_ceil(8),
        mcu_rows: height.div_ceil(8),
        restart_interval_mcus,
        restart_offsets,
        y_quant: header.y_quant,
        y_dc_table: header.y_dc_table,
        y_ac_table: header.y_ac_table,
        entropy_bytes,
    })
}
