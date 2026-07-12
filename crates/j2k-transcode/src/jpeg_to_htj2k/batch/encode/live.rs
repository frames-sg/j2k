// SPDX-License-Identifier: MIT OR Apache-2.0

//! Retained-owner measurements for sequential batch encode admission.

use super::super::{
    Float97BatchTile, HostLiveBudget, IntegerBatchTile, JpegToHtj2kError,
    PrecomputedHtj2k53Component, PrecomputedHtj2k97Component, PreencodedHtj2k97CompactComponent,
    PreencodedHtj2k97Component, PrequantizedHtj2k97Component, TranscodeComponentReport,
};

pub(super) fn integer_tiles_nested_bytes(
    tiles: &[IntegerBatchTile],
) -> Result<usize, JpegToHtj2kError> {
    sum_tiles(tiles.iter().map(integer_tile_retained_bytes))
}

pub(super) fn float97_tiles_nested_bytes(
    tiles: &[Float97BatchTile],
) -> Result<usize, JpegToHtj2kError> {
    sum_tiles(tiles.iter().map(float97_tile_retained_bytes))
}

pub(super) fn checked_batch_live_bytes(
    fixed_bytes: usize,
    remaining_tiles: usize,
    completed_outputs: usize,
    cap: usize,
) -> Result<usize, JpegToHtj2kError> {
    let mut budget = HostLiveBudget::with_cap(cap);
    budget.add_bytes(fixed_bytes)?;
    budget.add_bytes(remaining_tiles)?;
    budget.add_bytes(completed_outputs)?;
    Ok(budget.live_bytes())
}

pub(in super::super) fn integer_tile_retained_bytes(
    tile: &IntegerBatchTile,
) -> Result<usize, JpegToHtj2kError> {
    let mut budget = common_tile_budget(
        tile.jpeg.retained_bytes()?,
        tile.component_sampling.capacity(),
        tile.component_reports.capacity(),
    )?;
    budget.add_capacity::<Option<PrecomputedHtj2k53Component>>(
        tile.precomputed_components.capacity(),
    )?;
    for component in tile.precomputed_components.iter().flatten() {
        add_dwt53(&mut budget, &component.dwt)?;
    }
    for capacity in [
        tile.float_validation_actual.capacity(),
        tile.float_validation_expected.capacity(),
        tile.integer_validation_actual.capacity(),
        tile.integer_validation_expected.capacity(),
    ] {
        budget.add_capacity::<i32>(capacity)?;
    }
    Ok(budget.live_bytes())
}

pub(in super::super) fn float97_tile_retained_bytes(
    tile: &Float97BatchTile,
) -> Result<usize, JpegToHtj2kError> {
    let mut budget = common_tile_budget(
        tile.jpeg.retained_bytes()?,
        tile.component_sampling.capacity(),
        tile.component_reports.capacity(),
    )?;
    budget.add_capacity::<Option<PrecomputedHtj2k97Component>>(
        tile.precomputed_components.capacity(),
    )?;
    for component in tile.precomputed_components.iter().flatten() {
        add_dwt97(&mut budget, &component.dwt)?;
    }
    budget.add_capacity::<u8>(tile.preencoded_compact_payload.capacity())?;
    budget.add_capacity::<Option<PreencodedHtj2k97CompactComponent>>(
        tile.preencoded_compact_components.capacity(),
    )?;
    for component in tile.preencoded_compact_components.iter().flatten() {
        add_compact_component(&mut budget, component)?;
    }
    budget.add_capacity::<Option<PreencodedHtj2k97Component>>(
        tile.preencoded_components.capacity(),
    )?;
    for component in tile.preencoded_components.iter().flatten() {
        add_preencoded_component(&mut budget, component)?;
    }
    budget.add_capacity::<Option<PrequantizedHtj2k97Component>>(
        tile.prequantized_components.capacity(),
    )?;
    for component in tile.prequantized_components.iter().flatten() {
        add_prequantized_component(&mut budget, component)?;
    }
    budget.add_capacity::<i32>(tile.float_validation_actual.capacity())?;
    budget.add_capacity::<i32>(tile.float_validation_expected.capacity())?;
    Ok(budget.live_bytes())
}

fn common_tile_budget(
    jpeg_bytes: usize,
    sampling_capacity: usize,
    report_capacity: usize,
) -> Result<HostLiveBudget, JpegToHtj2kError> {
    let mut budget = HostLiveBudget::process_cap();
    budget.add_bytes(jpeg_bytes)?;
    budget.add_capacity::<(u8, u8)>(sampling_capacity)?;
    budget.add_capacity::<TranscodeComponentReport>(report_capacity)?;
    Ok(budget)
}

fn add_dwt53(
    budget: &mut HostLiveBudget,
    output: &j2k::J2kForwardDwt53Output,
) -> Result<(), JpegToHtj2kError> {
    budget.add_capacity::<f32>(output.ll.capacity())?;
    budget.add_capacity::<j2k::J2kForwardDwt53Level>(output.levels.capacity())?;
    add_f32_levels(
        budget,
        output.levels.iter().map(|level| {
            [
                level.hl.capacity(),
                level.lh.capacity(),
                level.hh.capacity(),
            ]
        }),
    )
}

fn add_dwt97(
    budget: &mut HostLiveBudget,
    output: &j2k::J2kForwardDwt97Output,
) -> Result<(), JpegToHtj2kError> {
    budget.add_capacity::<f32>(output.ll.capacity())?;
    budget.add_capacity::<j2k::J2kForwardDwt97Level>(output.levels.capacity())?;
    add_f32_levels(
        budget,
        output.levels.iter().map(|level| {
            [
                level.hl.capacity(),
                level.lh.capacity(),
                level.hh.capacity(),
            ]
        }),
    )
}

fn add_f32_levels(
    budget: &mut HostLiveBudget,
    levels: impl Iterator<Item = [usize; 3]>,
) -> Result<(), JpegToHtj2kError> {
    for capacities in levels {
        for capacity in capacities {
            budget.add_capacity::<f32>(capacity)?;
        }
    }
    Ok(())
}

fn add_compact_component(
    budget: &mut HostLiveBudget,
    component: &PreencodedHtj2k97CompactComponent,
) -> Result<(), JpegToHtj2kError> {
    budget.add_capacity::<j2k::PreencodedHtj2k97CompactResolution>(
        component.resolutions.capacity(),
    )?;
    for resolution in &component.resolutions {
        budget
            .add_capacity::<j2k::PreencodedHtj2k97CompactSubband>(resolution.subbands.capacity())?;
        for subband in &resolution.subbands {
            budget.add_capacity::<j2k::PreencodedHtj2k97CompactCodeBlock>(
                subband.code_blocks.capacity(),
            )?;
        }
    }
    Ok(())
}

fn add_preencoded_component(
    budget: &mut HostLiveBudget,
    component: &PreencodedHtj2k97Component,
) -> Result<(), JpegToHtj2kError> {
    budget.add_capacity::<j2k::PreencodedHtj2k97Resolution>(component.resolutions.capacity())?;
    for resolution in &component.resolutions {
        budget.add_capacity::<j2k::PreencodedHtj2k97Subband>(resolution.subbands.capacity())?;
        for subband in &resolution.subbands {
            budget
                .add_capacity::<j2k::PreencodedHtj2k97CodeBlock>(subband.code_blocks.capacity())?;
            for block in &subband.code_blocks {
                budget.add_capacity::<u8>(block.encoded.data.capacity())?;
            }
        }
    }
    Ok(())
}

fn add_prequantized_component(
    budget: &mut HostLiveBudget,
    component: &PrequantizedHtj2k97Component,
) -> Result<(), JpegToHtj2kError> {
    budget.add_capacity::<j2k::PrequantizedHtj2k97Resolution>(component.resolutions.capacity())?;
    for resolution in &component.resolutions {
        budget.add_capacity::<j2k::PrequantizedHtj2k97Subband>(resolution.subbands.capacity())?;
        for subband in &resolution.subbands {
            budget.add_capacity::<j2k::PrequantizedHtj2k97CodeBlock>(
                subband.code_blocks.capacity(),
            )?;
            for block in &subband.code_blocks {
                budget.add_capacity::<i32>(block.coefficients.capacity())?;
            }
        }
    }
    Ok(())
}

fn sum_tiles(
    retained: impl Iterator<Item = Result<usize, JpegToHtj2kError>>,
) -> Result<usize, JpegToHtj2kError> {
    let mut budget = HostLiveBudget::process_cap();
    for bytes in retained {
        budget.add_bytes(bytes?)?;
    }
    Ok(budget.live_bytes())
}

#[cfg(test)]
mod tests {
    use super::checked_batch_live_bytes;
    use crate::JpegToHtj2kError;

    #[test]
    fn accumulated_batch_outputs_accept_exact_cap_and_reject_one_over() {
        assert!(matches!(checked_batch_live_bytes(4, 5, 7, 16), Ok(16)));
        assert!(matches!(
            checked_batch_live_bytes(4, 5, 8, 16),
            Err(JpegToHtj2kError::MemoryCapExceeded {
                requested: 17,
                cap: 16,
            })
        ));
    }
}
