// SPDX-License-Identifier: MIT OR Apache-2.0

use metal::{Buffer, Device};

use crate::{
    abi::{
        JpegBaselineEncodeStatus, JpegDecodeStatus, FAST420_STATUS_HUFFMAN, FAST420_STATUS_OK,
        FAST420_STATUS_TRUNCATED, JPEG_BASELINE_ENCODE_STATUS_INVALID_PARAMS,
        JPEG_BASELINE_ENCODE_STATUS_MISSING_HUFFMAN, JPEG_BASELINE_ENCODE_STATUS_OVERFLOW,
    },
    buffers::{checked_buffer_slice, checked_fill_buffer_u8, new_shared_buffer},
    Error,
};

pub(super) fn jpeg_baseline_encode_status_error(status: JpegBaselineEncodeStatus) -> Error {
    let message = match status.code {
        JPEG_BASELINE_ENCODE_STATUS_OVERFLOW => {
            "JPEG Baseline Metal encode entropy output exceeded capacity".to_string()
        }
        JPEG_BASELINE_ENCODE_STATUS_MISSING_HUFFMAN => format!(
            "JPEG Baseline Metal encode missing Huffman code for symbol {}",
            status.detail
        ),
        JPEG_BASELINE_ENCODE_STATUS_INVALID_PARAMS => {
            "JPEG Baseline Metal encode received invalid kernel parameters".to_string()
        }
        other => format!("JPEG Baseline Metal encode failed with status {other}"),
    };
    Error::MetalKernel { message }
}

pub(super) fn fast_decode_status_error(status: JpegDecodeStatus) -> Error {
    let reason = match status.code {
        FAST420_STATUS_TRUNCATED => "truncated entropy stream",
        FAST420_STATUS_HUFFMAN => "invalid Huffman stream",
        _ => "unexpected Metal JPEG failure",
    };
    Error::MetalKernel {
        message: format!("{reason} at entropy byte {}", status.position),
    }
}

pub(super) fn decode_status_buffer(device: &Device, count: u32) -> Result<Buffer, Error> {
    let bytes = crate::batch_allocation::checked_count_product(
        count as usize,
        core::mem::size_of::<JpegDecodeStatus>(),
        "JPEG Metal decode status bytes",
    )?;
    let buffer = new_shared_buffer(device, bytes)?;
    checked_fill_buffer_u8(&buffer, bytes, 0, "initialize JPEG Metal decode statuses")?;
    Ok(buffer)
}

pub(super) fn first_decode_error_status(
    buffer: &Buffer,
    count: u32,
) -> Result<Option<JpegDecodeStatus>, Error> {
    let statuses =
        checked_buffer_slice::<JpegDecodeStatus>(buffer, count as usize, "decode statuses")?;
    Ok(statuses
        .iter()
        .copied()
        .find(|status| status.code != FAST420_STATUS_OK))
}

pub(super) fn fast422_status_error(status: JpegDecodeStatus) -> Error {
    Error::MetalKernel {
        message: format!(
            "unexpected Metal fast422 failure at entropy byte {}",
            status.position
        ),
    }
}
