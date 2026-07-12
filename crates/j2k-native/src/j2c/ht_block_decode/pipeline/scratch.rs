// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fallible HT phase scratch reservation.

use super::super::cleanup::cleanup_symbol_stride;
use super::super::significance::sigma_stride;
use super::super::state::HtBlockDecodeScratch;
use crate::error::{Result, ValidationError};
use crate::{try_reserve_decode_elements, DEFAULT_MAX_DECODE_BYTES};
use core::mem::size_of;

#[derive(Clone, Copy)]
struct HtWorkspaceLayout {
    coefficients: usize,
    cleanup_symbols: usize,
    v_n_words: usize,
    sigma_symbols: usize,
    previous_row_symbols: usize,
}

fn workspace_layout(width: u32, height: u32) -> Result<HtWorkspaceLayout> {
    let coefficient_len = (width as usize)
        .checked_mul(height as usize)
        .ok_or(ValidationError::ImageTooLarge)?;
    let cleanup_rows = (height.div_ceil(2) as usize)
        .checked_add(1)
        .ok_or(ValidationError::ImageTooLarge)?;
    let cleanup_len = cleanup_symbol_stride(width)
        .checked_mul(cleanup_rows)
        .ok_or(ValidationError::ImageTooLarge)?;
    let v_n_len = (width.div_ceil(2) as usize)
        .checked_add(2)
        .ok_or(ValidationError::ImageTooLarge)?;
    let sigma_rows = (height.div_ceil(4) as usize)
        .checked_add(1)
        .ok_or(ValidationError::ImageTooLarge)?;
    let sigma_len = sigma_stride(width)
        .checked_mul(sigma_rows)
        .ok_or(ValidationError::ImageTooLarge)?;
    let previous_row_len = (width.div_ceil(4) as usize)
        .checked_add(8)
        .ok_or(ValidationError::ImageTooLarge)?;
    Ok(HtWorkspaceLayout {
        coefficients: coefficient_len,
        cleanup_symbols: cleanup_len,
        v_n_words: v_n_len,
        sigma_symbols: sigma_len,
        previous_row_symbols: previous_row_len,
    })
}

pub(crate) fn ht_decode_workspace_bytes(width: u32, height: u32) -> Result<usize> {
    let layout = workspace_layout(width, height)?;
    let mut bytes = 0usize;
    include_elements::<u32>(&mut bytes, layout.coefficients)?;
    include_elements::<u16>(&mut bytes, layout.cleanup_symbols)?;
    include_elements::<u32>(&mut bytes, layout.v_n_words)?;
    include_elements::<u16>(&mut bytes, layout.sigma_symbols)?;
    include_elements::<u16>(&mut bytes, layout.previous_row_symbols)?;
    Ok(bytes)
}

fn include_elements<T>(bytes: &mut usize, count: usize) -> Result<()> {
    let additional = count
        .checked_mul(size_of::<T>())
        .ok_or(ValidationError::ImageTooLarge)?;
    *bytes = bytes
        .checked_add(additional)
        .ok_or(ValidationError::ImageTooLarge)?;
    if *bytes > DEFAULT_MAX_DECODE_BYTES {
        return Err(ValidationError::ImageTooLarge.into());
    }
    Ok(())
}

pub(crate) fn prepare_scratch(
    scratch: &mut HtBlockDecodeScratch,
    width: u32,
    height: u32,
) -> Result<()> {
    let layout = workspace_layout(width, height)?;

    try_reserve_decode_elements(&mut scratch.cleanup, layout.cleanup_symbols)?;
    try_reserve_decode_elements(&mut scratch.v_n, layout.v_n_words)?;
    try_reserve_decode_elements(&mut scratch.sigma, layout.sigma_symbols)?;
    try_reserve_decode_elements(&mut scratch.prev_row_sig, layout.previous_row_symbols)
}
