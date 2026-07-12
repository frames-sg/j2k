// SPDX-License-Identifier: MIT OR Apache-2.0

use super::huffman::validate_rgb8_huffman_tables;
use crate::{
    error::CudaError,
    jpeg::{
        CudaJpeg420Params, CudaJpegRgb8DecodePlan, CudaJpegRgb8Sampling, CudaJpegRgb8ValidatedPlan,
    },
    kernels::CudaLaunchGeometry,
};

mod checkpoints;
use self::checkpoints::validate_entropy_checkpoints;

const RGB8_CHANNELS: u32 = 3;
const U32_ADDRESSABLE_BYTES: u64 = u32::MAX as u64 + 1;

pub(crate) fn validate_jpeg_rgb8_plan(
    plan: &CudaJpegRgb8DecodePlan<'_>,
) -> Result<CudaJpegRgb8ValidatedPlan, CudaError> {
    let (width, height) = plan.dimensions;
    let out_stride = width
        .checked_mul(RGB8_CHANNELS)
        .ok_or(CudaError::ImageTooLarge {
            width,
            height,
            channels: RGB8_CHANNELS as usize,
        })?;
    validate_jpeg_rgb8_plan_with_pitch(plan, out_stride as usize)
}

pub(crate) fn validate_jpeg_rgb8_plan_with_pitch(
    plan: &CudaJpegRgb8DecodePlan<'_>,
    pitch_bytes: usize,
) -> Result<CudaJpegRgb8ValidatedPlan, CudaError> {
    let (width, height) = plan.dimensions;
    if width == 0 || height == 0 {
        return Err(invalid("decode dimensions must be nonzero"));
    }
    if plan.entropy_bytes.is_empty() {
        return Err(invalid("decode entropy payload must be nonempty"));
    }

    let entropy_len =
        u32::try_from(plan.entropy_bytes.len()).map_err(|_| CudaError::LengthTooLarge {
            len: plan.entropy_bytes.len(),
        })?;
    let checkpoint_count =
        u32::try_from(plan.entropy_checkpoints.len()).map_err(|_| CudaError::LengthTooLarge {
            len: plan.entropy_checkpoints.len(),
        })?;
    if checkpoint_count == 0 {
        return Err(invalid("decode requires at least one entropy checkpoint"));
    }
    let geometry = CudaLaunchGeometry::new((checkpoint_count, 1, 1), (1, 1, 1))
        .ok_or_else(|| invalid("decode checkpoint launch exceeds static CUDA limits"))?;

    let (mcu_width, mcu_height) = match plan.sampling {
        CudaJpegRgb8Sampling::Fast420 => (16, 16),
        CudaJpegRgb8Sampling::Fast422 => (16, 8),
        CudaJpegRgb8Sampling::Fast444 => (8, 8),
    };
    let expected_mcus_per_row = width.div_ceil(mcu_width);
    let expected_mcu_rows = height.div_ceil(mcu_height);
    if plan.mcus_per_row != expected_mcus_per_row || plan.mcu_rows != expected_mcu_rows {
        return Err(invalid(format_args!(
            "decode MCU grid {}x{} does not match the {:?} image grid {}x{}",
            plan.mcus_per_row,
            plan.mcu_rows,
            plan.sampling,
            expected_mcus_per_row,
            expected_mcu_rows
        )));
    }
    let total_mcus = expected_mcus_per_row
        .checked_mul(expected_mcu_rows)
        .ok_or_else(|| invalid("decode MCU count overflows the kernel ABI"))?;
    validate_entropy_checkpoints(plan.entropy_checkpoints, entropy_len, total_mcus)?;
    validate_quantization_tables(plan)?;
    validate_rgb8_huffman_tables(plan)?;

    let row_bytes = width
        .checked_mul(RGB8_CHANNELS)
        .ok_or(CudaError::ImageTooLarge {
            width,
            height,
            channels: RGB8_CHANNELS as usize,
        })?;
    if pitch_bytes < row_bytes as usize {
        return Err(invalid(format_args!(
            "decode pitch {pitch_bytes} is smaller than row byte count {row_bytes}"
        )));
    }
    let out_stride =
        u32::try_from(pitch_bytes).map_err(|_| CudaError::LengthTooLarge { len: pitch_bytes })?;
    let output_len_u64 = u64::from(out_stride)
        .checked_mul(u64::from(height - 1))
        .and_then(|prefix| prefix.checked_add(u64::from(row_bytes)))
        .ok_or(CudaError::ImageTooLarge {
            width,
            height,
            channels: RGB8_CHANNELS as usize,
        })?;
    if output_len_u64 > U32_ADDRESSABLE_BYTES {
        return Err(invalid(
            "decode output extent exceeds the kernel's u32 byte addressing",
        ));
    }
    let output_len = usize::try_from(output_len_u64).map_err(|_| CudaError::ImageTooLarge {
        width,
        height,
        channels: RGB8_CHANNELS as usize,
    })?;

    Ok(CudaJpegRgb8ValidatedPlan {
        params: CudaJpeg420Params {
            width,
            height,
            mcus_per_row: expected_mcus_per_row,
            mcu_rows: expected_mcu_rows,
            entropy_len,
            checkpoint_count,
            out_stride,
            reserved: 0,
        },
        output_len,
        geometry,
    })
}

fn validate_quantization_tables(plan: &CudaJpegRgb8DecodePlan<'_>) -> Result<(), CudaError> {
    for (label, table) in [
        ("Y", &plan.y_quant),
        ("Cb", &plan.cb_quant),
        ("Cr", &plan.cr_quant),
    ] {
        if let Some((index, value)) = table
            .iter()
            .copied()
            .enumerate()
            .find(|(_, value)| !(1..=255).contains(value))
        {
            return Err(invalid(format_args!(
                "{label} quantization entry {index} has unsupported baseline value {value}"
            )));
        }
    }
    Ok(())
}

pub(super) fn invalid(message: impl std::fmt::Display) -> CudaError {
    CudaError::InvalidArgument {
        message: format!("JPEG CUDA {message}"),
    }
}
