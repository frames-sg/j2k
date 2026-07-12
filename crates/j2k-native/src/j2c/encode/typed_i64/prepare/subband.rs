// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fallible packed-subband preparation for exact i64 Tier-1 input.

use alloc::vec::Vec;

use crate::j2c::encode::allocation::{
    checked_add_bytes, checked_element_bytes, host_allocation_failed,
};
use crate::j2c::encode::subband::{roi_region_subband_window, shift_roi_coefficient_i64};
use crate::j2c::encode::{
    I64SubbandEncodeSettings, NativeEncodePipelineError, NativeEncodePipelineResult,
    NativeEncodeSession, PreparedCodeBlockCoefficients, PreparedEncodeCodeBlock,
    PreparedEncodeSubband, QuantStepSize, SubBandType,
};
use crate::j2c::fdwt::PackedSubbandView;

mod plan;
use plan::{empty_prepared_subband, prepared_subband, PackedSubbandPlan};

pub(super) struct PackedSubbandRequest<'a, 'input> {
    pub(super) view: PackedSubbandView<'a, i64>,
    pub(super) step_size: &'a QuantStepSize,
    pub(super) sub_band_type: SubBandType,
    pub(super) settings: I64SubbandEncodeSettings<'a>,
    pub(super) retained_base_bytes: usize,
    pub(super) session: &'a NativeEncodeSession<'input>,
}

pub(super) fn prepare_packed_subband_i64(
    request: &PackedSubbandRequest<'_, '_>,
) -> NativeEncodePipelineResult<PreparedEncodeSubband> {
    let width = request.view.width();
    let height = request.view.height();
    if request.settings.cb_width == 0 || request.settings.cb_height == 0 {
        return Err(NativeEncodePipelineError::internal_invariant(
            "validated code-block dimensions must be non-zero",
        ));
    }
    if width == 0 || height == 0 {
        return Ok(empty_prepared_subband(request));
    }
    let plan = PackedSubbandPlan::try_new(request)?;
    let requested_outer = checked_element_bytes::<PreparedEncodeCodeBlock>(
        plan.block_count,
        "prepared i64 code-block owners",
    )?;
    request.session.checked_phase(
        checked_add_bytes(
            request.retained_base_bytes,
            requested_outer,
            "prepared i64 code-block owners",
        )?,
        "prepared i64 code-block owners",
    )?;
    let mut code_blocks = Vec::new();
    code_blocks
        .try_reserve_exact(plan.block_count)
        .map_err(|_| host_allocation_failed("prepared i64 code-block owners", requested_outer))?;
    let outer_bytes = checked_element_bytes::<PreparedEncodeCodeBlock>(
        code_blocks.capacity(),
        "prepared i64 code-block owners",
    )?;
    request.session.checked_phase(
        checked_add_bytes(
            request.retained_base_bytes,
            outer_bytes,
            "prepared i64 code-block owners",
        )?,
        "prepared i64 code-block owners",
    )?;

    let mut coefficient_bytes = 0usize;
    for cby in 0..plan.num_cbs_y {
        for cbx in 0..plan.num_cbs_x {
            let geometry = BlockGeometry::try_new(cbx, cby, plan, request.settings)?;
            let (code_block, actual) =
                try_prepare_code_block(request, geometry, outer_bytes, coefficient_bytes)?;
            coefficient_bytes = checked_add_bytes(
                coefficient_bytes,
                actual,
                "prepared i64 code-block coefficients",
            )?;
            code_blocks.push(code_block);
        }
    }

    Ok(prepared_subband(request, plan, code_blocks))
}

#[derive(Clone, Copy)]
struct BlockGeometry {
    x0: u32,
    y0: u32,
    x1: u32,
    y1: u32,
    width: u32,
    height: u32,
}

impl BlockGeometry {
    fn try_new(
        cbx: u32,
        cby: u32,
        plan: PackedSubbandPlan,
        settings: I64SubbandEncodeSettings<'_>,
    ) -> NativeEncodePipelineResult<Self> {
        let x0 = cbx
            .checked_mul(settings.cb_width)
            .ok_or_else(|| NativeEncodePipelineError::arithmetic_overflow("code-block x offset"))?;
        let y0 = cby
            .checked_mul(settings.cb_height)
            .ok_or_else(|| NativeEncodePipelineError::arithmetic_overflow("code-block y offset"))?;
        let x1 = x0.saturating_add(settings.cb_width).min(plan.width);
        let y1 = y0.saturating_add(settings.cb_height).min(plan.height);
        Ok(Self {
            x0,
            y0,
            x1,
            y1,
            width: x1 - x0,
            height: y1 - y0,
        })
    }
}

fn try_prepare_code_block(
    request: &PackedSubbandRequest<'_, '_>,
    geometry: BlockGeometry,
    outer_bytes: usize,
    prior_coefficient_bytes: usize,
) -> NativeEncodePipelineResult<(PreparedEncodeCodeBlock, usize)> {
    let count = usize::try_from(geometry.width)
        .map_err(|_| {
            NativeEncodePipelineError::arithmetic_overflow("code-block width exceeds usize")
        })?
        .checked_mul(usize::try_from(geometry.height).map_err(|_| {
            NativeEncodePipelineError::arithmetic_overflow("code-block height exceeds usize")
        })?)
        .ok_or_else(|| NativeEncodePipelineError::arithmetic_overflow("code-block area"))?;
    let requested = checked_element_bytes::<i64>(count, "prepared i64 code-block coefficients")?;
    check_block_peak(
        request.session,
        request.retained_base_bytes,
        outer_bytes,
        prior_coefficient_bytes,
        requested,
    )?;
    let mut coefficients = Vec::new();
    coefficients
        .try_reserve_exact(count)
        .map_err(|_| host_allocation_failed("prepared i64 code-block coefficients", requested))?;
    copy_block(request.view, geometry, &mut coefficients)?;
    apply_block_roi_shift(&mut coefficients, geometry, request)?;
    let actual = checked_element_bytes::<i64>(
        coefficients.capacity(),
        "prepared i64 code-block coefficients",
    )?;
    check_block_peak(
        request.session,
        request.retained_base_bytes,
        outer_bytes,
        prior_coefficient_bytes,
        actual,
    )?;
    Ok((
        PreparedEncodeCodeBlock {
            coefficients: PreparedCodeBlockCoefficients::I64(coefficients),
            width: geometry.width,
            height: geometry.height,
        },
        actual,
    ))
}

fn check_block_peak(
    session: &NativeEncodeSession<'_>,
    retained_base_bytes: usize,
    outer_bytes: usize,
    prior_coefficients: usize,
    new_coefficients: usize,
) -> NativeEncodePipelineResult<()> {
    let block_bytes = checked_add_bytes(
        outer_bytes,
        checked_add_bytes(
            prior_coefficients,
            new_coefficients,
            "prepared i64 code-block coefficients",
        )?,
        "prepared i64 subband",
    )?;
    session.checked_phase(
        checked_add_bytes(retained_base_bytes, block_bytes, "prepared i64 subband")?,
        "prepared i64 subband",
    )?;
    Ok(())
}

fn copy_block(
    view: PackedSubbandView<'_, i64>,
    geometry: BlockGeometry,
    output: &mut Vec<i64>,
) -> NativeEncodePipelineResult<()> {
    let x0 = usize::try_from(geometry.x0).map_err(|_| {
        NativeEncodePipelineError::arithmetic_overflow("code-block x offset exceeds usize")
    })?;
    let x1 = usize::try_from(geometry.x1).map_err(|_| {
        NativeEncodePipelineError::arithmetic_overflow("code-block x extent exceeds usize")
    })?;
    for y in geometry.y0..geometry.y1 {
        let row = view.row(y).ok_or_else(|| {
            NativeEncodePipelineError::internal_invariant("packed i64 subband row is out of range")
        })?;
        output.extend_from_slice(row.get(x0..x1).ok_or_else(|| {
            NativeEncodePipelineError::internal_invariant(
                "packed i64 code-block range is out of bounds",
            )
        })?);
    }
    Ok(())
}

fn apply_block_roi_shift(
    coefficients: &mut [i64],
    geometry: BlockGeometry,
    request: &PackedSubbandRequest<'_, '_>,
) -> NativeEncodePipelineResult<()> {
    if request.settings.roi_shift == 0 {
        return Ok(());
    }
    let block_width_usize = usize::try_from(geometry.width).map_err(|_| {
        NativeEncodePipelineError::arithmetic_overflow("ROI code-block width exceeds usize")
    })?;
    for local_y in 0..geometry.height {
        for local_x in 0..geometry.width {
            let x = geometry.x0 + local_x;
            let y = geometry.y0 + local_y;
            let selected = request.settings.roi_regions.is_empty()
                || request.settings.roi_regions.iter().any(|region| {
                    roi_region_subband_window(
                        *region,
                        request.view.width(),
                        request.view.height(),
                        request.settings.roi_scale,
                    )
                    .is_some_and(|(rx0, ry0, rx1, ry1)| x >= rx0 && x < rx1 && y >= ry0 && y < ry1)
                });
            if selected {
                let index = usize::try_from(local_y)
                    .ok()
                    .and_then(|row| row.checked_mul(block_width_usize))
                    .and_then(|row| {
                        usize::try_from(local_x)
                            .ok()
                            .and_then(|x| row.checked_add(x))
                    })
                    .ok_or_else(|| {
                        NativeEncodePipelineError::arithmetic_overflow("ROI code-block index")
                    })?;
                shift_roi_coefficient_i64(
                    coefficients.get_mut(index).ok_or_else(|| {
                        NativeEncodePipelineError::internal_invariant(
                            "ROI code-block index is out of range",
                        )
                    })?,
                    request.settings.roi_shift,
                )
                .map_err(NativeEncodePipelineError::arithmetic_overflow)?;
            }
        }
    }
    Ok(())
}
