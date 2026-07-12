// SPDX-License-Identifier: MIT OR Apache-2.0

//! Private host-allocation phases for CUDA-resident baseline JPEG encode.

use super::{
    encode_validation::{
        jpeg_encode_table_validation_host_bytes, jpeg_encode_validation_host_bytes,
    },
    CudaJpegBaselineEncodeParams, CudaJpegBaselineEncodeStatus,
};
use crate::{
    allocation::{host_element_bytes, HostPhaseBudget},
    error::CudaError,
};

fn checked_host_phase_bytes(
    byte_counts: impl IntoIterator<Item = usize>,
    what: &'static str,
) -> Result<usize, CudaError> {
    let mut budget = HostPhaseBudget::new(what);
    let mut requested = 0usize;
    for bytes in byte_counts {
        budget.account_bytes(bytes)?;
        requested = requested.saturating_add(bytes);
    }
    Ok(requested)
}

/// Check only simultaneously live Rust host owners. CUDA input, entropy,
/// status, parameter, quantization, and Huffman device buffers are governed by
/// the device allocator and are intentionally outside the 512 MiB host cap.
pub(super) fn checked_single_private_host_bytes(
    external_live_bytes: usize,
    entropy_capacity: usize,
) -> Result<usize, CudaError> {
    let validation =
        jpeg_encode_validation_host_bytes(1).max(jpeg_encode_table_validation_host_bytes());
    let validation_peak = checked_host_phase_bytes(
        [external_live_bytes, validation],
        "JPEG baseline single encode validation",
    )?;
    let output_peak = checked_host_phase_bytes(
        [external_live_bytes, entropy_capacity],
        "JPEG baseline single encode output",
    )?;
    Ok(validation_peak.max(output_peak))
}

/// Check the maximum validation or output host phase; the phases do not
/// overlap, while the caller-owned converted parameter vector spans both.
pub(super) fn checked_batch_private_host_bytes(
    external_live_bytes: usize,
    param_capacity: usize,
    tile_count: usize,
    status_capacity: usize,
    output_outer_capacity: usize,
    output_payload_capacity: usize,
) -> Result<usize, CudaError> {
    let params = host_element_bytes::<CudaJpegBaselineEncodeParams>(param_capacity);
    let validation = jpeg_encode_validation_host_bytes(tile_count)
        .max(jpeg_encode_table_validation_host_bytes());
    let statuses = host_element_bytes::<CudaJpegBaselineEncodeStatus>(status_capacity);
    let output_outer = host_element_bytes::<Vec<u8>>(output_outer_capacity);
    let validation_peak = checked_host_phase_bytes(
        [external_live_bytes, params, validation],
        "JPEG baseline batch encode validation",
    )?;
    let runtime_peak = checked_host_phase_bytes(
        [
            external_live_bytes,
            params,
            statuses,
            output_payload_capacity,
            output_outer,
        ],
        "JPEG baseline batch encode output",
    )?;
    Ok(validation_peak.max(runtime_peak))
}

#[cfg(test)]
mod tests;
