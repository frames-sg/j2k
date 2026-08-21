// SPDX-License-Identifier: MIT OR Apache-2.0

use super::invalid_tile;
use crate::{error::CudaError, jpeg::CudaJpegBaselineEncodeFormat};

use crate::jpeg::CudaJpegBaselineEncodeParams;

fn validate_sampling(params: CudaJpegBaselineEncodeParams, index: usize) -> Result<u32, CudaError> {
    let gray = CudaJpegBaselineEncodeFormat::Gray8.abi();
    let rgb = CudaJpegBaselineEncodeFormat::Rgb8.abi();
    let layout = (
        params.format,
        params.components,
        params.max_h,
        params.max_v,
        params.h0,
        params.v0,
        params.h1,
        params.v1,
        params.h2,
        params.v2,
    );
    let supported = layout == (gray, 1, 1, 1, 1, 1, 0, 0, 0, 0)
        || layout == (rgb, 3, 1, 1, 1, 1, 1, 1, 1, 1)
        || layout == (rgb, 3, 2, 1, 2, 1, 1, 1, 1, 1)
        || layout == (rgb, 3, 2, 2, 2, 2, 1, 1, 1, 1);
    if !supported {
        return Err(invalid_tile(
            index,
            "format, component count, and sampling factors are inconsistent",
        ));
    }
    Ok(if params.format == gray { 1 } else { 3 })
}

fn validate_last_sample_axis(
    mcu_count: u32,
    max_sampling: u32,
    component_sampling: u32,
    index: usize,
    axis: &str,
) -> Result<(), CudaError> {
    let scale = max_sampling / component_sampling;
    let last_origin = (mcu_count - 1)
        .checked_mul(max_sampling)
        .and_then(|value| value.checked_mul(8))
        .ok_or_else(|| invalid_tile(index, format_args!("last MCU {axis} origin overflows u32")))?;
    let last_component_sample = (component_sampling - 1)
        .checked_mul(8)
        .and_then(|value| value.checked_add(7))
        .and_then(|value| value.checked_mul(scale))
        .and_then(|value| value.checked_add(scale - 1))
        .ok_or_else(|| {
            invalid_tile(
                index,
                format_args!("component {axis} sample index overflows u32"),
            )
        })?;
    last_origin
        .checked_add(last_component_sample)
        .ok_or_else(|| {
            invalid_tile(
                index,
                format_args!("last sample {axis} index overflows u32"),
            )
        })?;
    Ok(())
}

fn validate_kernel_index_products(
    params: CudaJpegBaselineEncodeParams,
    index: usize,
    bytes_per_pixel: u32,
) -> Result<usize, CudaError> {
    let last_row = (params.input_height - 1)
        .checked_mul(params.pitch_bytes)
        .ok_or_else(|| invalid_tile(index, "last input-row byte offset overflows u32"))?;
    let last_pixel = (params.input_width - 1)
        .checked_mul(bytes_per_pixel)
        .and_then(|value| value.checked_add(bytes_per_pixel - 1))
        .ok_or_else(|| invalid_tile(index, "last input-pixel byte offset overflows u32"))?;
    let last_input_byte = last_row
        .checked_add(last_pixel)
        .ok_or_else(|| invalid_tile(index, "input row footprint overflows u32 indexes"))?;

    let horizontal = [params.h0, params.h1, params.h2];
    let vertical = [params.v0, params.v1, params.v2];
    let component_count = usize::try_from(params.components)
        .map_err(|_| CudaError::LengthTooLarge { len: usize::MAX })?;
    for component in 0..component_count {
        validate_last_sample_axis(
            params.mcus_per_row,
            params.max_h,
            horizontal[component],
            index,
            "horizontal",
        )?;
        validate_last_sample_axis(
            params.mcu_rows,
            params.max_v,
            vertical[component],
            index,
            "vertical",
        )?;
        let _maximum_sum = (params.max_h / horizontal[component])
            .checked_mul(params.max_v / vertical[component])
            .and_then(|value| value.checked_mul(u32::from(u8::MAX)))
            .ok_or_else(|| invalid_tile(index, "component averaging sum overflows u32"))?;
    }

    usize::try_from(u64::from(last_input_byte) + 1)
        .map_err(|_| CudaError::LengthTooLarge { len: usize::MAX })
}

pub(super) fn validate_tile_layout(
    params: CudaJpegBaselineEncodeParams,
    index: usize,
) -> Result<usize, CudaError> {
    if params.input_width == 0
        || params.input_height == 0
        || params.output_width == 0
        || params.output_height == 0
    {
        return Err(invalid_tile(
            index,
            "input and output dimensions must be nonzero",
        ));
    }
    if params.output_width > u32::from(u16::MAX) || params.output_height > u32::from(u16::MAX) {
        return Err(invalid_tile(
            index,
            "output dimensions exceed baseline JPEG marker limits",
        ));
    }
    if params.input_width > params.output_width || params.input_height > params.output_height {
        return Err(invalid_tile(
            index,
            "input dimensions exceed the encoded frame dimensions",
        ));
    }
    if params.restart_interval_mcus > u32::from(u16::MAX) {
        return Err(invalid_tile(
            index,
            "restart interval exceeds the baseline JPEG marker limit",
        ));
    }

    let bytes_per_pixel = validate_sampling(params, index)?;
    let mcu_width = params
        .max_h
        .checked_mul(8)
        .ok_or_else(|| invalid_tile(index, "MCU width overflows u32"))?;
    let mcu_height = params
        .max_v
        .checked_mul(8)
        .ok_or_else(|| invalid_tile(index, "MCU height overflows u32"))?;
    let expected_mcus_per_row = params.output_width.div_ceil(mcu_width);
    let expected_mcu_rows = params.output_height.div_ceil(mcu_height);
    if params.mcus_per_row != expected_mcus_per_row || params.mcu_rows != expected_mcu_rows {
        return Err(invalid_tile(
            index,
            format_args!(
                "MCU geometry must be {expected_mcus_per_row}x{expected_mcu_rows}, got {}x{}",
                params.mcus_per_row, params.mcu_rows
            ),
        ));
    }
    let total_mcus = u64::from(params.mcus_per_row) * u64::from(params.mcu_rows);
    if total_mcus > u64::from(u32::MAX) {
        return Err(invalid_tile(index, "total MCU count exceeds u32"));
    }

    let row_bytes = params
        .input_width
        .checked_mul(bytes_per_pixel)
        .ok_or_else(|| invalid_tile(index, "input row byte count overflows u32"))?;
    if params.pitch_bytes < row_bytes {
        return Err(invalid_tile(
            index,
            format_args!(
                "pitch {} is smaller than row byte count {row_bytes}",
                params.pitch_bytes
            ),
        ));
    }
    validate_kernel_index_products(params, index, bytes_per_pixel)
}
