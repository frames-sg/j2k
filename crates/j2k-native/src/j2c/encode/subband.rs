// SPDX-License-Identifier: MIT OR Apache-2.0

use super::allocation::{checked_add_bytes, checked_element_bytes};
use super::code_block_metadata::validate_accelerated_ht_code_block;
use super::tier1_allocation::{
    prepared_subbands_ownership, public_ht_blocks_ownership, Tier1PhaseTracker,
};
#[cfg(test)]
use super::NativeEncodeRetainedInput;
use super::{
    quantize, BlockCodingMode, ComponentRoiEncodeRegion, J2kEncodeStageAccelerator,
    J2kHtSubbandEncodeJob, J2kQuantizeSubbandJob, NativeEncodePipelineError,
    NativeEncodePipelineResult, NativeEncodeSession, PreparedEncodeCodeBlock,
    PreparedEncodeSubband, QuantStepSize, SubBandType, Vec,
};

fn apply_roi_maxshift_encode_for_session(
    coefficients: &mut [i32],
    request: &F32SubbandEncodeRequest<'_, '_>,
    quantized_bytes: usize,
    tracker: &mut Tier1PhaseTracker<'_, '_>,
) -> NativeEncodePipelineResult<()> {
    if request.roi_shift == 0 {
        return Ok(());
    }
    let expected_len = (request.width as usize)
        .checked_mul(request.height as usize)
        .ok_or_else(|| NativeEncodePipelineError::arithmetic_overflow("ROI subband dimensions"))?;
    if coefficients.len() != expected_len {
        return Err(NativeEncodePipelineError::internal_invariant(
            "ROI subband coefficient length mismatch",
        ));
    }
    if request.roi_regions.is_empty() {
        for coefficient in coefficients {
            shift_roi_coefficient(coefficient, request.roi_shift)
                .map_err(NativeEncodePipelineError::arithmetic_overflow)?;
        }
        return Ok(());
    }

    let (mut selected, _) = tracker.try_vec::<u8>(
        coefficients.len(),
        [quantized_bytes],
        "ROI subband selection flags",
    )?;
    selected.resize(coefficients.len(), 0);
    for region in request.roi_regions {
        let Some((x0, y0, x1, y1)) =
            roi_region_subband_window(*region, request.width, request.height, request.roi_scale)
        else {
            continue;
        };
        for y in y0..y1 {
            for x in x0..x1 {
                let idx = (y as usize)
                    .checked_mul(request.width as usize)
                    .and_then(|row| row.checked_add(x as usize))
                    .ok_or_else(|| {
                        NativeEncodePipelineError::arithmetic_overflow("ROI subband index")
                    })?;
                if selected[idx] != 0 {
                    continue;
                }
                selected[idx] = 1;
                shift_roi_coefficient(&mut coefficients[idx], request.roi_shift)
                    .map_err(NativeEncodePipelineError::arithmetic_overflow)?;
            }
        }
    }
    Ok(())
}

fn shift_roi_coefficient(coefficient: &mut i32, roi_shift: u8) -> Result<(), &'static str> {
    *coefficient = coefficient
        .checked_shl(u32::from(roi_shift))
        .ok_or("ROI maxshift coefficient overflow")?;
    Ok(())
}

pub(super) fn shift_roi_coefficient_i64(
    coefficient: &mut i64,
    roi_shift: u8,
) -> Result<(), &'static str> {
    let factor = 1_i64
        .checked_shl(u32::from(roi_shift))
        .ok_or("ROI maxshift coefficient overflow")?;
    *coefficient = coefficient
        .checked_mul(factor)
        .ok_or("ROI maxshift coefficient overflow")?;
    Ok(())
}

pub(super) fn roi_region_subband_window(
    region: ComponentRoiEncodeRegion,
    width: u32,
    height: u32,
    roi_scale: u32,
) -> Option<(u32, u32, u32, u32)> {
    if width == 0 || height == 0 || roi_scale == 0 {
        return None;
    }
    let x1 = region.x.saturating_add(region.width);
    let y1 = region.y.saturating_add(region.height);
    let x0 = (region.x / roi_scale).min(width);
    let y0 = (region.y / roi_scale).min(height);
    let x1 = x1.div_ceil(roi_scale).min(width);
    let y1 = y1.div_ceil(roi_scale).min(height);
    if x0 >= x1 || y0 >= y1 {
        None
    } else {
        Some((x0, y0, x1, y1))
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "this codec boundary keeps geometry, state buffers, and validated options explicit without allocation or indirection"
)]
#[expect(
    clippy::trivially_copy_pass_by_ref,
    reason = "the stable codec boundary borrows shared Copy metadata used across nested calls"
)]
#[cfg(test)]
pub(super) fn prepare_subband(
    coefficients: &[f32],
    width: u32,
    height: u32,
    step_size: &QuantStepSize,
    bit_depth: u8,
    guard_bits: u8,
    reversible: bool,
    block_coding_mode: BlockCodingMode,
    cb_width: u32,
    cb_height: u32,
    sub_band_type: SubBandType,
    roi_shift: u8,
    roi_regions: &[ComponentRoiEncodeRegion],
    roi_scale: u32,
    ht_target_coding_passes: u8,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> crate::EncodeResult<PreparedEncodeSubband> {
    let session = NativeEncodeSession::try_new(NativeEncodeRetainedInput::none())?;
    let request = F32SubbandEncodeRequest {
        coefficients,
        width,
        height,
        step_size,
        bit_depth,
        guard_bits,
        reversible,
        block_coding_mode,
        cb_width,
        cb_height,
        sub_band_type,
        roi_shift,
        roi_regions,
        roi_scale,
        ht_target_coding_passes,
        session: &session,
        retained_base_bytes: 0,
    };
    prepare_subband_for_session(&request, accelerator)
        .map_err(NativeEncodePipelineError::into_encode_error)
}

pub(super) struct F32SubbandEncodeRequest<'a, 'input> {
    pub(super) coefficients: &'a [f32],
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) step_size: &'a QuantStepSize,
    pub(super) bit_depth: u8,
    pub(super) guard_bits: u8,
    pub(super) reversible: bool,
    pub(super) block_coding_mode: BlockCodingMode,
    pub(super) cb_width: u32,
    pub(super) cb_height: u32,
    pub(super) sub_band_type: SubBandType,
    pub(super) roi_shift: u8,
    pub(super) roi_regions: &'a [ComponentRoiEncodeRegion],
    pub(super) roi_scale: u32,
    pub(super) ht_target_coding_passes: u8,
    pub(super) session: &'a NativeEncodeSession<'input>,
    pub(super) retained_base_bytes: usize,
}

pub(super) fn prepare_subband_for_session(
    request: &F32SubbandEncodeRequest<'_, '_>,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> NativeEncodePipelineResult<PreparedEncodeSubband> {
    let mut tracker = Tier1PhaseTracker::new(request.session, request.retained_base_bytes);
    if request.width == 0 || request.height == 0 {
        return Ok(empty_prepared_subband(request));
    }

    let plan = F32SubbandPlan::try_new(request)?;
    if let Some(prepared) = try_prepare_fused_ht_subband(request, plan, &mut tracker, accelerator)?
    {
        return Ok(prepared);
    }

    let (mut quantized, quantized_bytes) =
        try_quantize_subband(request, plan, &mut tracker, accelerator)?;
    apply_roi_maxshift_encode_for_session(&mut quantized, request, quantized_bytes, &mut tracker)?;
    prepare_quantized_subband(request, plan, &quantized, quantized_bytes, &mut tracker)
}

#[derive(Clone, Copy)]
struct F32SubbandPlan {
    range_bits: u8,
    total_bitplanes: u8,
    num_cbs_x: u32,
    num_cbs_y: u32,
}

impl F32SubbandPlan {
    fn try_new(request: &F32SubbandEncodeRequest<'_, '_>) -> NativeEncodePipelineResult<Self> {
        let exponent = u8::try_from(request.step_size.exponent).map_err(|_| {
            NativeEncodePipelineError::internal_invariant(
                "quantization exponent exceeds supported range",
            )
        })?;
        let base_total_bitplanes = request
            .guard_bits
            .saturating_add(exponent)
            .saturating_sub(1);
        let total_bitplanes = base_total_bitplanes
            .checked_add(request.roi_shift)
            .ok_or_else(|| {
                NativeEncodePipelineError::unsupported(
                    "ROI maxshift exceeds supported coded bitplane count",
                )
            })?;
        Ok(Self {
            range_bits: subband_range_bits(request.bit_depth, request.sub_band_type),
            total_bitplanes,
            num_cbs_x: request.width.div_ceil(request.cb_width),
            num_cbs_y: request.height.div_ceil(request.cb_height),
        })
    }
}

fn empty_prepared_subband(request: &F32SubbandEncodeRequest<'_, '_>) -> PreparedEncodeSubband {
    PreparedEncodeSubband {
        code_blocks: Vec::new(),
        preencoded_ht_code_blocks: None,
        num_cbs_x: 0,
        num_cbs_y: 0,
        code_block_width: request.cb_width,
        code_block_height: request.cb_height,
        width: request.width,
        height: request.height,
        sub_band_type: request.sub_band_type,
        total_bitplanes: 0,
        block_coding_mode: request.block_coding_mode,
        ht_target_coding_passes: request.ht_target_coding_passes,
    }
}

fn try_prepare_fused_ht_subband(
    request: &F32SubbandEncodeRequest<'_, '_>,
    plan: F32SubbandPlan,
    tracker: &mut Tier1PhaseTracker<'_, '_>,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> NativeEncodePipelineResult<Option<PreparedEncodeSubband>> {
    if request.block_coding_mode != BlockCodingMode::HighThroughput
        || request.roi_shift != 0
        || request.ht_target_coding_passes != 1
    {
        return Ok(None);
    }
    let encoded = accelerator
        .encode_ht_subband(J2kHtSubbandEncodeJob {
            coefficients: request.coefficients,
            width: request.width,
            height: request.height,
            step_exponent: request.step_size.exponent,
            step_mantissa: request.step_size.mantissa,
            range_bits: plan.range_bits,
            reversible: request.reversible,
            code_block_width: request.cb_width,
            code_block_height: request.cb_height,
            total_bitplanes: plan.total_bitplanes,
        })
        .map_err(|source| crate::EncodeError::Accelerator {
            operation: "fused HT subband encode",
            source,
        })?;
    let Some(encoded) = encoded else {
        return Ok(None);
    };
    let expected_code_blocks = (plan.num_cbs_x as usize)
        .checked_mul(plan.num_cbs_y as usize)
        .ok_or_else(|| NativeEncodePipelineError::arithmetic_overflow("code-block count"))?;
    if encoded.len() != expected_code_blocks {
        return Err(crate::EncodeError::Accelerator {
            operation: "fused HT subband encode",
            source: crate::J2kEncodeStageError::internal_invariant(
                "accelerated HT subband code-block count mismatch",
            ),
        }
        .into());
    }
    for block in &encoded {
        validate_accelerated_ht_code_block(block, plan.total_bitplanes, 1).map_err(|detail| {
            crate::EncodeError::Accelerator {
                operation: "fused HT subband encode",
                source: crate::J2kEncodeStageError::internal_invariant(detail),
            }
        })?;
    }
    let encoded_bytes = public_ht_blocks_ownership(&encoded, encoded.capacity())?;
    tracker.check([encoded_bytes], "fused HT subband output")?;
    let code_blocks = code_block_shapes_for_session(
        request.width,
        request.height,
        request.cb_width,
        request.cb_height,
        encoded_bytes,
        tracker,
    )?;
    let prepared = prepared_subband(request, plan, code_blocks, Some(encoded));
    let prepared_bytes =
        prepared_subbands_ownership(core::slice::from_ref(&prepared), 0)?.total()?;
    tracker.check([prepared_bytes], "prepared fused HT subband")?;
    Ok(Some(prepared))
}

fn try_quantize_subband(
    request: &F32SubbandEncodeRequest<'_, '_>,
    plan: F32SubbandPlan,
    tracker: &mut Tier1PhaseTracker<'_, '_>,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> NativeEncodePipelineResult<(Vec<i32>, usize)> {
    let requested_bytes =
        checked_element_bytes::<i32>(request.coefficients.len(), "quantized subband coefficients")?;
    tracker.check([requested_bytes], "quantized subband coefficients")?;
    let quantized = accelerator
        .encode_quantize_subband(J2kQuantizeSubbandJob {
            coefficients: request.coefficients,
            step_exponent: request.step_size.exponent,
            step_mantissa: request.step_size.mantissa,
            range_bits: plan.range_bits,
            reversible: request.reversible,
        })
        .map_err(|source| crate::EncodeError::Accelerator {
            operation: "subband quantization",
            source,
        })?;
    let quantized = match quantized {
        Some(quantized) => {
            if quantized.len() != request.coefficients.len() {
                return Err(crate::EncodeError::Accelerator {
                    operation: "subband quantization",
                    source: crate::J2kEncodeStageError::internal_invariant(
                        "accelerated quantized subband length mismatch",
                    ),
                }
                .into());
            }
            quantized
        }
        None => quantize::try_quantize_subband(
            request.coefficients,
            request.step_size,
            plan.range_bits,
            request.reversible,
        )?,
    };
    let actual_bytes =
        checked_element_bytes::<i32>(quantized.capacity(), "quantized subband coefficients")?;
    tracker.check([actual_bytes], "quantized subband output")?;
    Ok((quantized, actual_bytes))
}

fn prepare_quantized_subband(
    request: &F32SubbandEncodeRequest<'_, '_>,
    plan: F32SubbandPlan,
    quantized: &[i32],
    quantized_bytes: usize,
    tracker: &mut Tier1PhaseTracker<'_, '_>,
) -> NativeEncodePipelineResult<PreparedEncodeSubband> {
    let code_block_count = usize::try_from(plan.num_cbs_x.checked_mul(plan.num_cbs_y).ok_or(
        crate::EncodeError::ArithmeticOverflow {
            what: "prepared code-block count",
        },
    )?)
    .map_err(|_| crate::EncodeError::ArithmeticOverflow {
        what: "prepared code-block count",
    })?;
    let (mut code_blocks, code_block_owner_bytes) = tracker.try_vec::<PreparedEncodeCodeBlock>(
        code_block_count,
        [quantized_bytes],
        "prepared code-block owners",
    )?;
    let mut prepared_coefficient_bytes = 0usize;

    for cby in 0..plan.num_cbs_y {
        for cbx in 0..plan.num_cbs_x {
            let x0 = cbx * request.cb_width;
            let y0 = cby * request.cb_height;
            let x1 = (x0 + request.cb_width).min(request.width);
            let y1 = (y0 + request.cb_height).min(request.height);
            let cbw = x1 - x0;
            let cbh = y1 - y0;

            let coefficient_count = (cbw as usize).checked_mul(cbh as usize).ok_or(
                crate::EncodeError::ArithmeticOverflow {
                    what: "prepared code-block coefficient count",
                },
            )?;
            let (mut cb_coeffs, cb_bytes) = tracker.try_vec::<i32>(
                coefficient_count,
                [
                    quantized_bytes,
                    code_block_owner_bytes,
                    prepared_coefficient_bytes,
                ],
                "prepared i32 code-block coefficients",
            )?;
            copy_code_block_coefficients_into(
                quantized,
                request.width as usize,
                x0 as usize,
                y0 as usize,
                cbw as usize,
                cbh as usize,
                &mut cb_coeffs,
            )?;
            prepared_coefficient_bytes = checked_add_bytes(
                prepared_coefficient_bytes,
                cb_bytes,
                "prepared i32 code-block coefficient graph",
            )?;

            code_blocks.push(PreparedEncodeCodeBlock {
                coefficients: super::PreparedCodeBlockCoefficients::I32(cb_coeffs),
                width: cbw,
                height: cbh,
            });
        }
    }

    let prepared = prepared_subband(request, plan, code_blocks, None);
    let prepared_bytes =
        prepared_subbands_ownership(core::slice::from_ref(&prepared), 0)?.total()?;
    tracker.check([prepared_bytes], "prepared quantized subband")?;
    Ok(prepared)
}

fn prepared_subband(
    request: &F32SubbandEncodeRequest<'_, '_>,
    plan: F32SubbandPlan,
    code_blocks: Vec<PreparedEncodeCodeBlock>,
    preencoded_ht_code_blocks: Option<Vec<super::EncodedHtJ2kCodeBlock>>,
) -> PreparedEncodeSubband {
    PreparedEncodeSubband {
        code_blocks,
        preencoded_ht_code_blocks,
        num_cbs_x: plan.num_cbs_x,
        num_cbs_y: plan.num_cbs_y,
        code_block_width: request.cb_width,
        code_block_height: request.cb_height,
        width: request.width,
        height: request.height,
        sub_band_type: request.sub_band_type,
        total_bitplanes: plan.total_bitplanes,
        block_coding_mode: request.block_coding_mode,
        ht_target_coding_passes: request.ht_target_coding_passes,
    }
}

#[derive(Clone, Copy)]
pub(super) struct I64SubbandEncodeSettings<'a> {
    pub(super) guard_bits: u8,
    pub(super) cb_width: u32,
    pub(super) cb_height: u32,
    pub(super) roi_shift: u8,
    pub(super) roi_regions: &'a [ComponentRoiEncodeRegion],
    pub(super) roi_scale: u32,
    pub(super) block_coding_mode: BlockCodingMode,
    pub(super) ht_target_coding_passes: u8,
}

fn code_block_shapes_for_session(
    width: u32,
    height: u32,
    cb_width: u32,
    cb_height: u32,
    encoded_bytes: usize,
    tracker: &mut Tier1PhaseTracker<'_, '_>,
) -> NativeEncodePipelineResult<Vec<PreparedEncodeCodeBlock>> {
    let num_cbs_x = width.div_ceil(cb_width);
    let num_cbs_y = height.div_ceil(cb_height);
    let count = usize::try_from(num_cbs_x.checked_mul(num_cbs_y).ok_or(
        crate::EncodeError::ArithmeticOverflow {
            what: "fused HT code-block shape count",
        },
    )?)
    .map_err(|_| crate::EncodeError::ArithmeticOverflow {
        what: "fused HT code-block shape count",
    })?;
    let (mut code_blocks, _) = tracker.try_vec::<PreparedEncodeCodeBlock>(
        count,
        [encoded_bytes],
        "fused HT code-block shape owners",
    )?;
    for cby in 0..num_cbs_y {
        for cbx in 0..num_cbs_x {
            let x0 = cbx * cb_width;
            let y0 = cby * cb_height;
            let x1 = (x0 + cb_width).min(width);
            let y1 = (y0 + cb_height).min(height);
            code_blocks.push(PreparedEncodeCodeBlock {
                coefficients: super::PreparedCodeBlockCoefficients::Empty,
                width: x1 - x0,
                height: y1 - y0,
            });
        }
    }
    Ok(code_blocks)
}

fn copy_code_block_coefficients_into(
    quantized: &[i32],
    width: usize,
    x0: usize,
    y0: usize,
    cbw: usize,
    cbh: usize,
    output: &mut Vec<i32>,
) -> NativeEncodePipelineResult<()> {
    let expected = cbw
        .checked_mul(cbh)
        .ok_or(crate::EncodeError::ArithmeticOverflow {
            what: "prepared code-block coefficient count",
        })?;
    if output.capacity() < expected || !output.is_empty() {
        return Err(crate::EncodeError::InternalInvariant {
            what: "prepared code-block coefficient owner does not match its plan",
        }
        .into());
    }
    for y in 0..cbh {
        let row_start = (y0 + y)
            .checked_mul(width)
            .and_then(|row| row.checked_add(x0))
            .ok_or(crate::EncodeError::ArithmeticOverflow {
                what: "prepared code-block coefficient range",
            })?;
        let row_end = row_start
            .checked_add(cbw)
            .ok_or(crate::EncodeError::ArithmeticOverflow {
                what: "prepared code-block coefficient range",
            })?;
        output.extend_from_slice(quantized.get(row_start..row_end).ok_or_else(|| {
            NativeEncodePipelineError::internal_invariant(
                "prepared code-block coefficient range is invalid",
            )
        })?);
    }
    if output.len() != expected {
        return Err(crate::EncodeError::InternalInvariant {
            what: "prepared code-block coefficient copy length changed",
        }
        .into());
    }
    Ok(())
}

fn subband_range_bits(bit_depth: u8, sub_band_type: SubBandType) -> u8 {
    let log_gain = match sub_band_type {
        SubBandType::LowLow => 0,
        SubBandType::LowHigh | SubBandType::HighLow => 1,
        SubBandType::HighHigh => 2,
    };

    bit_depth.saturating_add(log_gain)
}

#[cfg(test)]
mod tests;
