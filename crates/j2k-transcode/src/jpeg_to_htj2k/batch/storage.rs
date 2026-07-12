// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    flatten_integer_wavelet, float97_reference_coefficients, float_reference_coefficients,
    integer_reference_coefficients, j2k_dwt97_from_wavelet, j2k_dwt_from_integer_wavelet,
    rounded_wavelet97_i32, BatchComponentRef, ComponentWavelet97, Float97BatchTile,
    IntegerBatchTile, IntegerWavelet, JpegToHtj2kError, JpegToHtj2kOptions, JpegToHtj2kScratch,
    PrecomputedHtj2k53Component, PrecomputedHtj2k97Component,
};
use crate::allocation::try_extend_from_slice;

pub(in super::super) fn store_integer_batch_wavelet(
    component_ref: BatchComponentRef,
    wavelet: &IntegerWavelet,
    tiles: &mut [IntegerBatchTile],
    options: &JpegToHtj2kOptions,
    scratch: &mut JpegToHtj2kScratch,
) -> Result<(), JpegToHtj2kError> {
    let tile = &mut tiles[component_ref.tile_index];
    let component = &tile.jpeg.components[component_ref.component_index];
    let (x_rsiz, y_rsiz) = tile.component_sampling[component_ref.component_index];
    let actual_coefficients = flatten_integer_wavelet(wavelet)?;
    tile.precomputed_components[component_ref.component_index] =
        Some(PrecomputedHtj2k53Component {
            x_rsiz,
            y_rsiz,
            dwt: j2k_dwt_from_integer_wavelet(wavelet)?,
        });

    if options.validate_against_float_reference {
        try_extend_from_slice(&mut tile.float_validation_actual, &actual_coefficients)?;
        let expected = float_reference_coefficients(component, tile.decomposition_levels, scratch)?;
        try_extend_from_slice(&mut tile.float_validation_expected, &expected)?;
    }
    if options.validate_against_integer_reference {
        try_extend_from_slice(&mut tile.integer_validation_actual, &actual_coefficients)?;
        let expected = integer_reference_coefficients(component, tile.decomposition_levels)?;
        try_extend_from_slice(&mut tile.integer_validation_expected, &expected)?;
    }

    Ok(())
}

pub(in super::super) fn store_float97_batch_wavelet(
    component_ref: BatchComponentRef,
    wavelet: &ComponentWavelet97,
    tiles: &mut [Float97BatchTile],
    options: &JpegToHtj2kOptions,
    scratch: &mut JpegToHtj2kScratch,
) -> Result<(), JpegToHtj2kError> {
    let tile = &mut tiles[component_ref.tile_index];
    let component = &tile.jpeg.components[component_ref.component_index];
    let (x_rsiz, y_rsiz) = tile.component_sampling[component_ref.component_index];
    tile.precomputed_components[component_ref.component_index] =
        Some(PrecomputedHtj2k97Component {
            x_rsiz,
            y_rsiz,
            dwt: j2k_dwt97_from_wavelet(
                wavelet,
                component.width as usize,
                component.height as usize,
            )?,
        });

    if options.validate_against_float_reference {
        let actual_coefficients = rounded_wavelet97_i32(wavelet)?;
        try_extend_from_slice(&mut tile.float_validation_actual, &actual_coefficients)?;
        let expected =
            float97_reference_coefficients(component, tile.decomposition_levels, scratch)?;
        try_extend_from_slice(&mut tile.float_validation_expected, &expected)?;
    }

    Ok(())
}
