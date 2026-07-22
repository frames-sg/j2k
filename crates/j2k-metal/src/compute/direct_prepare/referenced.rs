// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared retained-plan traversal and component assembly.

use super::{
    prepare_classic_sub_band_groups, prepare_ht_sub_band_groups,
    prepare_referenced_classic_sub_band, prepare_referenced_ht_sub_band, Arc, BandRequiredRegion,
    CpuTier1CoefficientCache, DirectTier1Mode, Error, HtCodeBlockPayloadRanges,
    J2kDirectGrayscaleStep, NativeGrayscalePlan, PreparedDirectGrayscalePlan,
    PreparedDirectGrayscaleStep, PreparedDirectIdwt, ReferencedClassicPayloadCursor,
};

pub(super) fn validate_payload_record_span(
    span: j2k_native::J2kReferencedPayloadRecordSpan,
    cursor: usize,
    payload_count: usize,
    state: &'static str,
) -> Result<usize, Error> {
    if span.first_record != cursor {
        return Err(Error::MetalStateInvariant {
            state,
            reason: "tile payload-record spans are not contiguous in geometry traversal order",
        });
    }
    let end = span.end_record().ok_or(Error::MetalStateInvariant {
        state,
        reason: "tile payload-record span overflowed",
    })?;
    if end > payload_count {
        return Err(Error::MetalStateInvariant {
            state,
            reason: "tile payload-record span exceeds the retained payload table",
        });
    }
    Ok(end)
}

#[cfg(target_os = "macos")]
pub(super) fn validate_referenced_component_metadata(
    first: &NativeGrayscalePlan,
    component: &NativeGrayscalePlan,
) -> Result<(), Error> {
    if component.dimensions != first.dimensions || component.bit_depth != first.bit_depth {
        return Err(Error::MetalStateInvariant {
            state: "referenced multi-tile component metadata",
            reason: "tile component dimensions or bit depth changed within one image",
        });
    }
    Ok(())
}

#[cfg(target_os = "macos")]
#[cfg(target_os = "macos")]
pub(super) fn append_referenced_classic_component_steps(
    geometry: &NativeGrayscalePlan,
    payloads: &mut ReferencedClassicPayloadCursor<'_>,
    steps: &mut Vec<PreparedDirectGrayscaleStep>,
    _allocation_context: &'static str,
) -> Result<(), Error> {
    for step in &geometry.steps {
        match step {
            J2kDirectGrayscaleStep::ClassicSubBand(sub_band) => {
                steps.push(PreparedDirectGrayscaleStep::ClassicSubBand(
                    prepare_referenced_classic_sub_band(sub_band, payloads)?,
                ));
            }
            J2kDirectGrayscaleStep::HtSubBand(_) => {
                return Err(Error::MetalStateInvariant {
                    state: "classic J2K referenced direct plan",
                    reason: "classic referenced plan contains an HT code-block step",
                });
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
    Ok(())
}

#[cfg(target_os = "macos")]
pub(super) fn finish_referenced_component_plan(
    dimensions: (u32, u32),
    bit_depth: u8,
    steps: Vec<PreparedDirectGrayscaleStep>,
    _allocation_context: &'static str,
) -> Result<PreparedDirectGrayscalePlan, Error> {
    let tier1_prepare_mode = DirectTier1Mode::Metal;
    let classic_groups = prepare_classic_sub_band_groups(&steps, tier1_prepare_mode)?;
    let ht_groups = prepare_ht_sub_band_groups(&steps, tier1_prepare_mode)?;
    Ok(PreparedDirectGrayscalePlan {
        dimensions,
        bit_depth,
        tier1_prepare_mode,
        steps,
        classic_groups,
        ht_groups,
        cpu_tier1_cache: Arc::new(CpuTier1CoefficientCache::default()),
    })
}

#[cfg(target_os = "macos")]
pub(super) fn append_referenced_htj2k_component_steps(
    geometry: &NativeGrayscalePlan,
    input: &Arc<[u8]>,
    payloads: &[HtCodeBlockPayloadRanges],
    payload_cursor: &mut usize,
    steps: &mut Vec<PreparedDirectGrayscaleStep>,
) -> Result<(), Error> {
    for step in &geometry.steps {
        match step {
            J2kDirectGrayscaleStep::ClassicSubBand(_) => {
                return Err(Error::MetalStateInvariant {
                    state: "HTJ2K referenced direct plan",
                    reason: "HTJ2K referenced plan contains a classic code-block step",
                });
            }
            J2kDirectGrayscaleStep::HtSubBand(sub_band) => {
                steps.push(PreparedDirectGrayscaleStep::HtSubBand(
                    prepare_referenced_ht_sub_band(sub_band, input, payloads, payload_cursor)?,
                ));
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
    Ok(())
}
