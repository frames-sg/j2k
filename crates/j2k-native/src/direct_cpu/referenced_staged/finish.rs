// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::error::{bail, DecodingError, Result};
use crate::{
    J2kDirectGrayscalePlan, J2kDirectGrayscaleStep, J2kReferencedClassicPlan,
    J2kReferencedHtj2kPlan, J2kWaveletTransform,
};

use super::super::referenced::decoded_components as decoded_htj2k_components;
use super::super::referenced_classic::decoded_components as decoded_classic_components;
use super::super::{
    apply_inverse_mct_region, execute_idwt_step, store_component, J2kDirectCpuScratch,
    J2kDirectDecodedComponents, StagedDirectRoute,
};
use super::plan_access::{
    classic_tile_color_transform, classic_tile_components, ht_tile_color_transform,
    ht_tile_components,
};
use super::state::{finish_staged_image, finish_staged_tile};

/// Reconstruct and store one staged HT tile, then retire its coefficient bands.
#[doc(hidden)]
pub fn finish_referenced_htj2k_tile_staged(
    plan: &J2kReferencedHtj2kPlan,
    tile_index: usize,
    signed: bool,
    image_scratch: &mut J2kDirectCpuScratch,
) -> Result<()> {
    let components = ht_tile_components(plan, tile_index)?;
    let color_transform = ht_tile_color_transform(plan, tile_index)?;
    let destination = plan
        .tiles()
        .get(tile_index)
        .ok_or(DecodingError::CodeBlockDecodeFailure)?
        .destination_rect();
    finish_staged_tile(
        image_scratch,
        StagedDirectRoute::Htj2k,
        tile_index,
        components,
        color_transform,
        destination,
        signed,
    )
}

/// Reconstruct and store one staged classic tile, then retire its coefficient bands.
#[doc(hidden)]
pub fn finish_referenced_classic_tile_staged(
    plan: &J2kReferencedClassicPlan,
    tile_index: usize,
    signed: bool,
    image_scratch: &mut J2kDirectCpuScratch,
) -> Result<()> {
    let components = classic_tile_components(plan, tile_index)?;
    let color_transform = classic_tile_color_transform(plan, tile_index)?;
    let destination = plan
        .tiles()
        .get(tile_index)
        .ok_or(DecodingError::CodeBlockDecodeFailure)?
        .destination_rect();
    finish_staged_tile(
        image_scratch,
        StagedDirectRoute::Classic,
        tile_index,
        components,
        color_transform,
        destination,
        signed,
    )
}

/// Borrow one complete staged HT image after all tile stores complete.
#[doc(hidden)]
pub fn finish_referenced_htj2k_staged<'scratch>(
    plan: &J2kReferencedHtj2kPlan,
    signed: bool,
    image_scratch: &'scratch mut J2kDirectCpuScratch,
) -> Result<J2kDirectDecodedComponents<'scratch>> {
    let _ = signed;
    finish_staged_image(image_scratch, StagedDirectRoute::Htj2k, plan.tiles().len())?;
    decoded_htj2k_components(plan, image_scratch)
}

/// Borrow one complete staged classic image after all tile stores complete.
#[doc(hidden)]
pub fn finish_referenced_classic_staged<'scratch>(
    plan: &J2kReferencedClassicPlan,
    signed: bool,
    image_scratch: &'scratch mut J2kDirectCpuScratch,
) -> Result<J2kDirectDecodedComponents<'scratch>> {
    let _ = signed;
    finish_staged_image(
        image_scratch,
        StagedDirectRoute::Classic,
        plan.tiles().len(),
    )?;
    decoded_classic_components(plan, image_scratch)
}

pub(super) fn finish_tile_components(
    components: &[J2kDirectGrayscalePlan],
    color_transform: Option<([u8; 3], bool, J2kWaveletTransform)>,
    destination: crate::J2kRect,
    signed: bool,
    scratch: &mut J2kDirectCpuScratch,
) -> Result<()> {
    if scratch.component_band_sets.len() < components.len()
        || scratch.component_planes.len() < components.len()
    {
        bail!(DecodingError::CodeBlockDecodeFailure);
    }
    for (component_index, component) in components.iter().enumerate() {
        let bands = &mut scratch.component_band_sets[component_index];
        let output = &mut scratch.component_planes[component_index];
        let mut output_initialized = true;
        let mut stored = false;
        for step in &component.steps {
            match step {
                J2kDirectGrayscaleStep::ClassicSubBand(_)
                | J2kDirectGrayscaleStep::HtSubBand(_) => {}
                J2kDirectGrayscaleStep::Idwt(step) => execute_idwt_step(step, bands)?,
                J2kDirectGrayscaleStep::Store(store) => {
                    store_component(store, bands.active(), output, &mut output_initialized)?;
                    stored = true;
                }
            }
        }
        if !stored {
            bail!(DecodingError::CodeBlockDecodeFailure);
        }
    }
    if let Some((bit_depths, mct, transform)) = color_transform {
        if mct {
            let [plane0, plane1, plane2, ..] = scratch.component_planes.as_mut_slice() else {
                bail!(DecodingError::CodeBlockDecodeFailure);
            };
            apply_inverse_mct_region(
                transform,
                bit_depths,
                signed,
                destination,
                plane0,
                plane1,
                plane2,
            )?;
        }
    }
    Ok(())
}
