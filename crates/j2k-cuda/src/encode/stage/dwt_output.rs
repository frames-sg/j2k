// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k::{
    J2kForwardDwt53Level, J2kForwardDwt53Output, J2kForwardDwt97Level, J2kForwardDwt97Output,
};
use j2k_cuda_runtime::{CudaDwt53LevelShape, CudaDwt53Output, CudaDwt97Output};

use crate::allocation::HostPhaseBudget;
use crate::encode::stage_error::{
    adapter_error, arithmetic_overflow, internal_invariant, CudaStageResult,
};

struct CudaDwtHostParts<T> {
    ll: Vec<f32>,
    levels: Vec<T>,
}

pub(in crate::encode) fn cuda_dwt53_output_to_j2k(
    output: &CudaDwt53Output,
) -> CudaStageResult<J2kForwardDwt53Output> {
    let (ll_width, ll_height) = output.ll_dimensions();
    let parts = cuda_dwt_output_parts(
        output.transformed(),
        output.levels(),
        (ll_width, ll_height),
        |shape, hl, lh, hh| J2kForwardDwt53Level {
            hl,
            lh,
            hh,
            width: shape.width,
            height: shape.height,
            low_width: shape.low_width,
            low_height: shape.low_height,
            high_width: shape.high_width,
            high_height: shape.high_height,
        },
    )?;
    Ok(J2kForwardDwt53Output {
        ll: parts.ll,
        ll_width,
        ll_height,
        levels: parts.levels,
    })
}

pub(super) fn cuda_dwt97_output_to_j2k(
    output: &CudaDwt97Output,
) -> CudaStageResult<J2kForwardDwt97Output> {
    let (ll_width, ll_height) = output.ll_dimensions();
    let parts = cuda_dwt_output_parts(
        output.transformed(),
        output.levels(),
        (ll_width, ll_height),
        |shape, hl, lh, hh| J2kForwardDwt97Level {
            hl,
            lh,
            hh,
            width: shape.width,
            height: shape.height,
            low_width: shape.low_width,
            low_height: shape.low_height,
            high_width: shape.high_width,
            high_height: shape.high_height,
        },
    )?;
    Ok(J2kForwardDwt97Output {
        ll: parts.ll,
        ll_width,
        ll_height,
        levels: parts.levels,
    })
}

fn cuda_dwt_output_parts<T>(
    transformed: &[f32],
    shapes: &[CudaDwt53LevelShape],
    ll_dimensions: (u32, u32),
    mut build_level: impl FnMut(&CudaDwt53LevelShape, Vec<f32>, Vec<f32>, Vec<f32>) -> T,
) -> CudaStageResult<CudaDwtHostParts<T>> {
    let (ll_width, ll_height) = ll_dimensions;
    let full_width = shapes.first().map_or(ll_width, |level| level.width) as usize;
    let mut host_budget = HostPhaseBudget::new("j2k CUDA DWT host output");
    let ll_capacity = (ll_width as usize)
        .checked_mul(ll_height as usize)
        .ok_or_else(|| arithmetic_overflow("CUDA DWT LL output capacity"))?;
    let mut ll = host_budget
        .try_vec_with_capacity(ll_capacity)
        .map_err(|error| adapter_error("allocate CUDA DWT LL output", error))?;
    for y in 0..ll_height as usize {
        let row_start = y
            .checked_mul(full_width)
            .ok_or_else(|| arithmetic_overflow("CUDA DWT LL row offset"))?;
        let row_end = row_start
            .checked_add(ll_width as usize)
            .ok_or_else(|| arithmetic_overflow("CUDA DWT LL row end"))?;
        ll.extend_from_slice(
            transformed.get(row_start..row_end).ok_or_else(|| {
                internal_invariant("CUDA DWT LL range exceeds transformed output")
            })?,
        );
    }

    let mut levels = host_budget
        .try_vec_with_capacity(shapes.len())
        .map_err(|error| adapter_error("allocate CUDA DWT level output", error))?;
    for shape in shapes {
        let hl = extract_cuda_subband(
            transformed,
            full_width,
            shape.low_width,
            0,
            shape.high_width,
            shape.low_height,
            &mut host_budget,
        )?;
        let lh = extract_cuda_subband(
            transformed,
            full_width,
            0,
            shape.low_height,
            shape.low_width,
            shape.high_height,
            &mut host_budget,
        )?;
        let hh = extract_cuda_subband(
            transformed,
            full_width,
            shape.low_width,
            shape.low_height,
            shape.high_width,
            shape.high_height,
            &mut host_budget,
        )?;
        levels.push(build_level(shape, hl, lh, hh));
    }
    levels.reverse();
    Ok(CudaDwtHostParts { ll, levels })
}

fn extract_cuda_subband(
    transformed: &[f32],
    full_width: usize,
    x0: u32,
    y0: u32,
    width: u32,
    height: u32,
    host_budget: &mut HostPhaseBudget,
) -> CudaStageResult<Vec<f32>> {
    let capacity = (width as usize)
        .checked_mul(height as usize)
        .ok_or_else(|| arithmetic_overflow("CUDA DWT subband output capacity"))?;
    let mut out = host_budget
        .try_vec_with_capacity(capacity)
        .map_err(|error| adapter_error("allocate CUDA DWT subband output", error))?;
    for y in 0..height as usize {
        let row_start = (y0 as usize)
            .checked_add(y)
            .and_then(|row| row.checked_mul(full_width))
            .and_then(|row| row.checked_add(x0 as usize))
            .ok_or_else(|| arithmetic_overflow("CUDA DWT subband offset"))?;
        let row_end = row_start
            .checked_add(width as usize)
            .ok_or_else(|| arithmetic_overflow("CUDA DWT subband row end"))?;
        out.extend_from_slice(transformed.get(row_start..row_end).ok_or_else(|| {
            internal_invariant("CUDA DWT subband range exceeds transformed output")
        })?);
    }
    Ok(out)
}
