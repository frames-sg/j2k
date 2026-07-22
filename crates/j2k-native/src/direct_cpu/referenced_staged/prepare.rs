// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::error::{bail, DecodingError, Result};
use crate::{
    J2kDirectGrayscalePlan, J2kDirectGrayscaleStep, J2kReferencedClassicPlan,
    J2kReferencedHtj2kPlan,
};

use super::super::{prepare_sub_band_output, J2kDirectCpuScratch, StagedDirectRoute};
use super::plan_access::{
    classic_component_count, classic_tile_components, ht_component_count, ht_tile_components,
};
use super::state::{begin_staged_image, prepare_staged_tile, EntropyRoute};
use super::J2kDirectCpuEntropyWorkspace;
use crate::direct_cpu::allocation::{
    max_referenced_classic_dimensions, max_referenced_ht_dimensions,
    prepare_referenced_classic_staged_scratch, prepare_referenced_htj2k_staged_scratch,
};

/// Prepare one retained worker workspace for HT jobs in `plan`.
#[doc(hidden)]
pub fn prepare_referenced_htj2k_entropy_workspace(
    plan: &J2kReferencedHtj2kPlan,
    worker_workspace: &mut J2kDirectCpuEntropyWorkspace,
) -> Result<()> {
    worker_workspace.prepare_ht_dimensions(max_referenced_ht_dimensions(plan))?;
    Ok(())
}

/// Prepare one retained worker workspace for classic jobs in `plan`.
#[doc(hidden)]
pub fn prepare_referenced_classic_entropy_workspace(
    plan: &J2kReferencedClassicPlan,
    worker_workspace: &mut J2kDirectCpuEntropyWorkspace,
) -> Result<()> {
    worker_workspace.prepare_classic_dimensions(max_referenced_classic_dimensions(plan))?;
    Ok(())
}

/// Prepare one image's retained coefficient owners for staged HT entropy work.
#[doc(hidden)]
pub fn prepare_referenced_htj2k_staged(
    plan: &J2kReferencedHtj2kPlan,
    image_scratch: &mut J2kDirectCpuScratch,
    worker_workspace: &mut J2kDirectCpuEntropyWorkspace,
) -> Result<()> {
    let budget = prepare_referenced_htj2k_staged_scratch(plan, image_scratch)?;
    worker_workspace.prepare_ht(max_referenced_ht_dimensions(plan), budget)?;
    begin_staged_image(
        image_scratch,
        StagedDirectRoute::Htj2k,
        plan.tiles().len(),
        plan.output_rect(),
        ht_component_count(plan),
    )
}

/// Prepare one image's retained coefficient owners for staged classic entropy work.
#[doc(hidden)]
pub fn prepare_referenced_classic_staged(
    plan: &J2kReferencedClassicPlan,
    image_scratch: &mut J2kDirectCpuScratch,
    worker_workspace: &mut J2kDirectCpuEntropyWorkspace,
) -> Result<()> {
    let budget = prepare_referenced_classic_staged_scratch(plan, image_scratch)?;
    worker_workspace.prepare_classic(max_referenced_classic_dimensions(plan), budget)?;
    begin_staged_image(
        image_scratch,
        StagedDirectRoute::Classic,
        plan.tiles().len(),
        plan.output_rect(),
        classic_component_count(plan),
    )
}

/// Prepare one HT tile's coefficient owners after the prior tile store has retired.
#[doc(hidden)]
pub fn prepare_referenced_htj2k_tile_staged(
    plan: &J2kReferencedHtj2kPlan,
    tile_index: usize,
    image_scratch: &mut J2kDirectCpuScratch,
) -> Result<()> {
    prepare_staged_tile(
        image_scratch,
        StagedDirectRoute::Htj2k,
        tile_index,
        ht_tile_components(plan, tile_index)?,
        EntropyRoute::Htj2k,
    )
}

/// Prepare one classic tile's coefficient owners after the prior tile store has retired.
#[doc(hidden)]
pub fn prepare_referenced_classic_tile_staged(
    plan: &J2kReferencedClassicPlan,
    tile_index: usize,
    image_scratch: &mut J2kDirectCpuScratch,
) -> Result<()> {
    prepare_staged_tile(
        image_scratch,
        StagedDirectRoute::Classic,
        tile_index,
        classic_tile_components(plan, tile_index)?,
        EntropyRoute::Classic,
    )
}

pub(super) fn prepare_entropy_bands(
    components: &[J2kDirectGrayscalePlan],
    route: EntropyRoute,
    scratch: &mut J2kDirectCpuScratch,
) -> Result<()> {
    if scratch.component_band_sets.len() < components.len() {
        bail!(DecodingError::CodeBlockDecodeFailure);
    }
    for (component_index, component) in components.iter().enumerate() {
        let bands = &mut scratch.component_band_sets[component_index];
        bands.reset();
        for step in &component.steps {
            let sub_band = match (route, step) {
                (EntropyRoute::Classic, J2kDirectGrayscaleStep::ClassicSubBand(sub_band)) => {
                    Some((
                        sub_band.band_id,
                        sub_band.rect,
                        sub_band.width,
                        sub_band.height,
                    ))
                }
                (EntropyRoute::Htj2k, J2kDirectGrayscaleStep::HtSubBand(sub_band)) => Some((
                    sub_band.band_id,
                    sub_band.rect,
                    sub_band.width,
                    sub_band.height,
                )),
                (EntropyRoute::Classic, J2kDirectGrayscaleStep::HtSubBand(_))
                | (EntropyRoute::Htj2k, J2kDirectGrayscaleStep::ClassicSubBand(_)) => {
                    bail!(DecodingError::CodeBlockDecodeFailure)
                }
                (_, J2kDirectGrayscaleStep::Idwt(_) | J2kDirectGrayscaleStep::Store(_)) => None,
            };
            if let Some((band_id, rect, width, height)) = sub_band {
                let _ = prepare_sub_band_output(bands, band_id, rect, width, height)?;
            }
        }
    }
    Ok(())
}
