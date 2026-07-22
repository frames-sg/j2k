// SPDX-License-Identifier: MIT OR Apache-2.0

//! Full and retained grayscale direct-plan preparation.

use super::{
    append_referenced_classic_component_steps, append_referenced_htj2k_component_steps,
    finish_referenced_component_plan, prepare_classic_sub_band, prepare_classic_sub_band_groups,
    prepare_ht_sub_band, prepare_ht_sub_band_groups, validate_payload_record_span,
    validate_referenced_component_metadata, Arc, BandRequiredRegion, CpuTier1CoefficientCache,
    DirectTier1Mode, Error, J2kDirectGrayscalePlan, J2kDirectGrayscaleStep,
    J2kReferencedClassicPlan, J2kReferencedHtj2kPlan, PreparedDirectGrayscalePlan,
    PreparedDirectGrayscaleStep, PreparedDirectIdwt, ReferencedClassicPayloadCursor,
};

#[cfg(target_os = "macos")]
pub(crate) fn prepare_direct_grayscale_plan(
    plan: &J2kDirectGrayscalePlan,
) -> Result<PreparedDirectGrayscalePlan, Error> {
    prepare_direct_grayscale_plan_with_tier1_mode(plan, DirectTier1Mode::Metal)
}

#[cfg(target_os = "macos")]
pub(crate) fn prepare_referenced_htj2k_grayscale_plan(
    referenced: &J2kReferencedHtj2kPlan,
    input: &Arc<[u8]>,
) -> Result<PreparedDirectGrayscalePlan, Error> {
    let first = referenced
        .tiles()
        .first()
        .and_then(j2k_native::J2kReferencedTilePlan::grayscale_geometry)
        .ok_or(Error::UnsupportedMetalRequest {
            reason: "J2K Metal grayscale prepared path received a color HTJ2K plan",
        })?;
    let step_count = crate::batch_allocation::checked_count_sum(
        referenced.tiles().iter().map(|tile| {
            tile.grayscale_geometry()
                .map_or(0, |geometry| geometry.steps.len())
        }),
        "J2K MetalDirect referenced HTJ2K grayscale tile steps",
    )?;
    let mut budget = crate::batch_allocation::BatchMetadataBudget::new(
        "J2K MetalDirect referenced HTJ2K grayscale plan",
    );
    let mut steps = budget.try_vec(
        step_count,
        "J2K MetalDirect referenced HTJ2K grayscale tile steps",
    )?;
    let mut payload_cursor = 0usize;
    for tile in referenced.tiles() {
        let geometry = tile
            .grayscale_geometry()
            .ok_or(Error::UnsupportedMetalRequest {
                reason: "J2K Metal grayscale prepared path received a color HTJ2K tile",
            })?;
        validate_referenced_component_metadata(first, geometry)?;
        let expected_end = validate_payload_record_span(
            tile.payload_records(),
            payload_cursor,
            referenced.payloads().len(),
            "HTJ2K grayscale tile",
        )?;
        append_referenced_htj2k_component_steps(
            geometry,
            input,
            referenced.payloads(),
            &mut payload_cursor,
            &mut steps,
        )?;
        if payload_cursor != expected_end {
            return Err(Error::MetalStateInvariant {
                state: "HTJ2K grayscale tile payload traversal",
                reason: "tile geometry job count does not match its payload-record span",
            });
        }
    }
    if payload_cursor != referenced.payloads().len() {
        return Err(Error::MetalKernel {
            message: "HTJ2K referenced plan has unused payload ranges".to_string(),
        });
    }
    finish_referenced_component_plan(
        first.dimensions,
        first.bit_depth,
        steps,
        "J2K MetalDirect referenced HTJ2K grayscale plan",
    )
}

#[cfg(target_os = "macos")]
pub(crate) fn prepare_referenced_classic_grayscale_plan(
    referenced: &J2kReferencedClassicPlan,
    input: &[u8],
) -> Result<PreparedDirectGrayscalePlan, Error> {
    let first = referenced
        .tiles()
        .first()
        .and_then(j2k_native::J2kReferencedTilePlan::grayscale_geometry)
        .ok_or(Error::UnsupportedMetalRequest {
            reason: "J2K Metal grayscale prepared path received a color classic J2K plan",
        })?;
    let step_count = crate::batch_allocation::checked_count_sum(
        referenced.tiles().iter().map(|tile| {
            tile.grayscale_geometry()
                .map_or(0, |geometry| geometry.steps.len())
        }),
        "J2K MetalDirect referenced classic grayscale tile steps",
    )?;
    let mut budget = crate::batch_allocation::BatchMetadataBudget::new(
        "J2K MetalDirect referenced classic grayscale plan",
    );
    let mut steps = budget.try_vec(
        step_count,
        "J2K MetalDirect referenced classic grayscale tile steps",
    )?;
    let mut payloads = ReferencedClassicPayloadCursor::new(input, referenced);
    for tile in referenced.tiles() {
        let geometry = tile
            .grayscale_geometry()
            .ok_or(Error::UnsupportedMetalRequest {
                reason: "J2K Metal grayscale prepared path received a color classic J2K tile",
            })?;
        validate_referenced_component_metadata(first, geometry)?;
        let expected_end = validate_payload_record_span(
            tile.payload_records(),
            payloads.next_payload,
            referenced.payloads().len(),
            "classic grayscale tile",
        )?;
        append_referenced_classic_component_steps(
            geometry,
            &mut payloads,
            &mut steps,
            "J2K MetalDirect referenced classic grayscale plan",
        )?;
        if payloads.next_payload != expected_end {
            return Err(Error::MetalStateInvariant {
                state: "classic grayscale tile payload traversal",
                reason: "tile geometry job count does not match its payload-record span",
            });
        }
    }
    payloads.ensure_exhausted()?;
    finish_referenced_component_plan(
        first.dimensions,
        first.bit_depth,
        steps,
        "J2K MetalDirect referenced classic grayscale plan",
    )
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn prepare_direct_grayscale_plan_for_cpu_upload(
    plan: &J2kDirectGrayscalePlan,
) -> Result<PreparedDirectGrayscalePlan, Error> {
    prepare_direct_grayscale_plan_with_tier1_mode(plan, DirectTier1Mode::CpuUpload)
}

#[cfg(target_os = "macos")]
pub(super) fn prepare_direct_grayscale_plan_with_tier1_mode(
    plan: &J2kDirectGrayscalePlan,
    tier1_prepare_mode: DirectTier1Mode,
) -> Result<PreparedDirectGrayscalePlan, Error> {
    let mut budget = crate::batch_allocation::BatchMetadataBudget::new(
        "J2K MetalDirect prepared grayscale plan",
    );
    let mut steps = budget.try_vec(plan.steps.len(), "J2K MetalDirect prepared grayscale steps")?;
    for step in &plan.steps {
        match step {
            J2kDirectGrayscaleStep::ClassicSubBand(sub_band) => {
                steps.push(PreparedDirectGrayscaleStep::ClassicSubBand(
                    prepare_classic_sub_band(sub_band, tier1_prepare_mode)?,
                ));
            }
            J2kDirectGrayscaleStep::HtSubBand(sub_band) => {
                steps.push(PreparedDirectGrayscaleStep::HtSubBand(prepare_ht_sub_band(
                    sub_band,
                    tier1_prepare_mode,
                )?));
            }
            J2kDirectGrayscaleStep::Idwt(idwt) => {
                steps.push(PreparedDirectGrayscaleStep::Idwt(PreparedDirectIdwt {
                    step: *idwt,
                    output_window: BandRequiredRegion::full(idwt.rect.width(), idwt.rect.height()),
                }));
            }
            J2kDirectGrayscaleStep::Store(store) => {
                steps.push(PreparedDirectGrayscaleStep::Store(*store));
            }
        }
    }
    let classic_groups = prepare_classic_sub_band_groups(&steps, tier1_prepare_mode)?;
    let ht_groups = prepare_ht_sub_band_groups(&steps, tier1_prepare_mode)?;
    Ok(PreparedDirectGrayscalePlan {
        dimensions: plan.dimensions,
        bit_depth: plan.bit_depth,
        tier1_prepare_mode,
        steps,
        classic_groups,
        ht_groups,
        cpu_tier1_cache: Arc::new(CpuTier1CoefficientCache::default()),
    })
}
