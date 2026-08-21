// SPDX-License-Identifier: MIT OR Apache-2.0

use super::CudaJpegBaselineEncodeParams;
use crate::{allocation::HostPhaseBudget, error::CudaError, kernels::CudaLaunchGeometry};

use self::{layout::validate_tile_layout, tables::validate_encode_tables};

#[cfg(any(feature = "cuda-oxide-jpeg-encode", test))]
pub(super) use self::tables::jpeg_encode_table_validation_host_bytes;
pub(super) use self::tables::CudaJpegBaselineEncodeTableRefs;

mod layout;
mod tables;

const U32_ADDRESSABLE_BYTES: u64 = 1u64 << 32;
pub(super) const JPEG_BATCH_GEOMETRY_EXCEEDS_LAUNCH_LIMITS: &str =
    "JPEG CUDA encode batch exceeds static CUDA launch limits";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct CudaJpegBaselineEncodeTileValidation {
    pub(super) input_ptr: u64,
    pub(super) entropy_offset: usize,
    pub(super) entropy_capacity: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct CudaJpegBaselineEncodeValidation {
    pub(super) tile_count: u32,
    pub(super) first_tile: CudaJpegBaselineEncodeTileValidation,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct EntropyRange {
    start: u64,
    end: u64,
    original_index: usize,
}

#[cfg(any(feature = "cuda-oxide-jpeg-encode", test))]
pub(super) fn jpeg_encode_validation_host_bytes(tile_count: usize) -> usize {
    crate::allocation::host_element_bytes::<EntropyRange>(tile_count)
}

fn invalid_request(message: impl Into<String>) -> CudaError {
    CudaError::InvalidArgument {
        message: message.into(),
    }
}

fn validate_disjoint_entropy_ranges(ranges: &mut [EntropyRange]) -> Result<(), CudaError> {
    ranges.sort_unstable_by_key(|range| (range.start, range.end, range.original_index));
    let Some(pair) = ranges.windows(2).find(|pair| pair[1].start < pair[0].end) else {
        return Ok(());
    };
    Err(invalid_request(format!(
        "JPEG CUDA encode entropy ranges for tiles {} and {} overlap",
        pair[0].original_index, pair[1].original_index
    )))
}

pub(super) fn validate_jpeg_encode_batch_launch(
    tile_count: u32,
) -> Result<CudaLaunchGeometry, CudaError> {
    CudaLaunchGeometry::new((tile_count, 1, 1), (1, 1, 1)).ok_or_else(|| {
        invalid_request(format!(
            "{JPEG_BATCH_GEOMETRY_EXCEEDS_LAUNCH_LIMITS}: tiles={tile_count}"
        ))
    })
}

pub(super) fn invalid_tile(index: usize, message: impl std::fmt::Display) -> CudaError {
    invalid_request(format!(
        "JPEG CUDA encode tile {index} has invalid parameters: {message}"
    ))
}

pub(super) fn validate_jpeg_baseline_encode_request(
    input_device_ptr: u64,
    input_byte_len: usize,
    bound_input_offset: usize,
    params: &[CudaJpegBaselineEncodeParams],
    entropy_byte_len: usize,
    tables: CudaJpegBaselineEncodeTableRefs<'_>,
    retained_host_bytes: usize,
) -> Result<CudaJpegBaselineEncodeValidation, CudaError> {
    if params.is_empty() {
        return Err(invalid_request(
            "JPEG CUDA encode validation requires at least one tile",
        ));
    }
    let tile_count =
        u32::try_from(params.len()).map_err(|_| CudaError::LengthTooLarge { len: params.len() })?;
    validate_encode_tables(params, tables, retained_host_bytes)?;
    let entropy_byte_len_u64 =
        u64::try_from(entropy_byte_len).map_err(|_| CudaError::LengthTooLarge {
            len: entropy_byte_len,
        })?;
    if entropy_byte_len_u64 > U32_ADDRESSABLE_BYTES {
        return Err(invalid_request(
            "JPEG CUDA encode entropy allocation exceeds u32 addressability",
        ));
    }

    let mut host_budget = HostPhaseBudget::new("JPEG baseline encode range validation");
    host_budget.account_bytes(retained_host_bytes)?;
    let mut entropy_ranges = host_budget.try_vec_with_capacity(params.len())?;
    let mut first_tile = None;
    for (index, params) in params.iter().copied().enumerate() {
        let input_span = validate_tile_layout(params, index)?;
        let parameter_input_offset = usize::try_from(params.input_offset_bytes)
            .map_err(|_| CudaError::LengthTooLarge { len: usize::MAX })?;
        let input_offset = bound_input_offset
            .checked_add(parameter_input_offset)
            .ok_or_else(|| invalid_tile(index, "input offset overflows usize"))?;
        let input_end = input_offset
            .checked_add(input_span)
            .ok_or_else(|| invalid_tile(index, "input offset and row footprint overflow usize"))?;
        if input_end > input_byte_len {
            return Err(invalid_tile(
                index,
                format_args!(
                    "input range ends at {input_end}, beyond allocation length {input_byte_len}"
                ),
            ));
        }
        let input_offset = u64::try_from(input_offset)
            .map_err(|_| CudaError::LengthTooLarge { len: input_offset })?;
        let input_span =
            u64::try_from(input_span).map_err(|_| CudaError::LengthTooLarge { len: input_span })?;
        let input_ptr = input_device_ptr
            .checked_add(input_offset)
            .ok_or_else(|| invalid_tile(index, "device input pointer offset overflows u64"))?;
        input_ptr
            .checked_add(input_span)
            .ok_or_else(|| invalid_tile(index, "device input pointer range overflows u64"))?;

        let entropy_offset = u64::from(params.entropy_offset_bytes);
        let entropy_capacity = u64::from(params.entropy_capacity);
        if entropy_capacity == 0 {
            return Err(invalid_tile(index, "entropy capacity must be nonzero"));
        }
        let entropy_end = entropy_offset
            .checked_add(entropy_capacity)
            .ok_or_else(|| invalid_tile(index, "entropy range overflows u64"))?;
        if entropy_end > U32_ADDRESSABLE_BYTES {
            return Err(invalid_tile(
                index,
                "entropy range is not addressable by u32 byte indexes",
            ));
        }
        if entropy_end > entropy_byte_len_u64 {
            return Err(invalid_tile(
                index,
                format_args!(
                    "entropy range ends at {entropy_end}, beyond allocation length {entropy_byte_len}"
                ),
            ));
        }
        let entropy_offset_usize = usize::try_from(entropy_offset)
            .map_err(|_| CudaError::LengthTooLarge { len: usize::MAX })?;
        let entropy_capacity_usize = usize::try_from(entropy_capacity)
            .map_err(|_| CudaError::LengthTooLarge { len: usize::MAX })?;
        first_tile.get_or_insert(CudaJpegBaselineEncodeTileValidation {
            input_ptr,
            entropy_offset: entropy_offset_usize,
            entropy_capacity: entropy_capacity_usize,
        });
        entropy_ranges.push(EntropyRange {
            start: entropy_offset,
            end: entropy_end,
            original_index: index,
        });
    }

    validate_disjoint_entropy_ranges(&mut entropy_ranges)?;

    let first_tile = first_tile.ok_or_else(|| {
        invalid_request("JPEG CUDA encode validation did not retain the first tile")
    })?;
    Ok(CudaJpegBaselineEncodeValidation {
        tile_count,
        first_tile,
    })
}

#[cfg(test)]
mod tests;
