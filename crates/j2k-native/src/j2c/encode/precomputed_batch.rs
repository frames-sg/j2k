// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fallible preparation for shared Tier-1 precomputed 9/7 batches.

use super::precomputed::allocation::ConstructionTracker;
use super::precomputed::orchestrator::{self, Prepared97PacketPlan};
use super::precomputed::{precomputed_97_level_count, validate_precomputed_dwt97_geometry};
use super::tier1_allocation::prepared_subbands_ownership;
use super::{
    prepare_subband_for_session, BlockCodingMode, CpuOnlyJ2kEncodeStageAccelerator, EncodeOptions,
    F32SubbandEncodeRequest, NativeEncodePipelineError, NativeEncodePipelineResult,
    NativeEncodeSession, PrecomputedHtj2k97Component, PrecomputedHtj2k97Image,
    PreparedEncodeSubband, PreparedResolutionPacket, QuantStepSize, SubBandType, Vec,
    MAX_J2K_SPEC_COMPONENTS,
};

pub(super) fn prepare_precomputed_htj2k97_image_for_batch(
    image: &PrecomputedHtj2k97Image,
    options: &EncodeOptions,
    session: &NativeEncodeSession<'_>,
    retained_base_bytes: usize,
) -> NativeEncodePipelineResult<Prepared97PacketPlan> {
    validate_precomputed_request(image)?;
    validate_precomputed_dwt97_geometry(image).map_err(NativeEncodePipelineError::invalid_input)?;
    let num_levels = precomputed_97_level_count(&image.components)
        .map_err(NativeEncodePipelineError::invalid_input)?;
    let mut tracker = ConstructionTracker::new(session, retained_base_bytes);
    let metadata = orchestrator::try_metadata(
        image.width,
        image.height,
        image.bit_depth,
        image.signed,
        image
            .components
            .iter()
            .map(|component| (component.x_rsiz, component.y_rsiz)),
        num_levels,
        options,
        &mut tracker,
    )?;
    let code_block_width = code_block_dimension(
        options.code_block_width_exp,
        "code-block width exponent exceeds supported range",
    )
    .map_err(NativeEncodePipelineError::invalid_input)?;
    let code_block_height = code_block_dimension(
        options.code_block_height_exp,
        "code-block height exponent exceeds supported range",
    )
    .map_err(NativeEncodePipelineError::invalid_input)?;
    let mut component_packets = tracker.try_vec::<Vec<PreparedResolutionPacket>>(
        image.components.len(),
        "batch precomputed 9/7 component packet owners",
    )?;
    let mut cpu = CpuOnlyJ2kEncodeStageAccelerator;
    for (component_idx, component) in image.components.iter().enumerate() {
        component_packets.push(try_prepared_component(
            component_idx,
            component,
            &metadata.step_sizes,
            image.bit_depth,
            metadata.params.guard_bits,
            code_block_width,
            code_block_height,
            session,
            &mut tracker,
            &mut cpu,
        )?);
    }
    orchestrator::finish_plan(
        metadata,
        component_packets,
        options,
        session,
        retained_base_bytes,
    )
}

fn validate_precomputed_request(image: &PrecomputedHtj2k97Image) -> NativeEncodePipelineResult<()> {
    if image.width == 0 || image.height == 0 {
        return Err(NativeEncodePipelineError::invalid_input(
            "invalid dimensions",
        ));
    }
    if image.components.is_empty() {
        return Err(NativeEncodePipelineError::invalid_input(
            "component set must be non-empty",
        ));
    }
    if image.components.len() > usize::from(MAX_J2K_SPEC_COMPONENTS) {
        return Err(NativeEncodePipelineError::unsupported(
            "component count exceeds the JPEG 2000 Part 1 limit",
        ));
    }
    if image.bit_depth == 0 {
        return Err(NativeEncodePipelineError::invalid_input(
            "bit depth must be non-zero",
        ));
    }
    if image.bit_depth > 16 {
        return Err(NativeEncodePipelineError::unsupported(
            "precomputed 9/7 bit depth exceeds 16 bits",
        ));
    }
    if image
        .components
        .iter()
        .any(|component| component.x_rsiz == 0 || component.y_rsiz == 0)
    {
        return Err(NativeEncodePipelineError::invalid_input(
            "component sampling factors must be non-zero",
        ));
    }
    Ok(())
}

#[expect(
    clippy::too_many_arguments,
    reason = "the coefficient preparation boundary keeps image, coding, and budget state explicit"
)]
fn try_prepared_component(
    component_idx: usize,
    component: &PrecomputedHtj2k97Component,
    step_sizes: &[QuantStepSize],
    bit_depth: u8,
    guard_bits: u8,
    code_block_width: u32,
    code_block_height: u32,
    session: &NativeEncodeSession<'_>,
    tracker: &mut ConstructionTracker<'_, '_>,
    cpu: &mut CpuOnlyJ2kEncodeStageAccelerator,
) -> NativeEncodePipelineResult<Vec<PreparedResolutionPacket>> {
    let component_idx = u16::try_from(component_idx).map_err(|_| {
        NativeEncodePipelineError::arithmetic_overflow("component index exceeds u16")
    })?;
    let packet_count = component.dwt.levels.len().checked_add(1).ok_or(
        crate::EncodeError::ArithmeticOverflow {
            what: "batch precomputed 9/7 resolution count",
        },
    )?;
    let mut packets = tracker.try_vec::<PreparedResolutionPacket>(
        packet_count,
        "batch precomputed 9/7 prepared resolutions",
    )?;
    let mut ll_subbands =
        tracker.try_vec::<PreparedEncodeSubband>(1, "batch precomputed 9/7 LL subband owner")?;
    ll_subbands.push(try_prepared_subband(
        &component.dwt.ll,
        component.dwt.ll_width,
        component.dwt.ll_height,
        *step_sizes.first().ok_or_else(|| {
            NativeEncodePipelineError::internal_invariant("irreversible quantization step missing")
        })?,
        bit_depth,
        guard_bits,
        code_block_width,
        code_block_height,
        SubBandType::LowLow,
        session,
        tracker,
        cpu,
    )?);
    packets.push(PreparedResolutionPacket {
        component: component_idx,
        resolution: 0,
        precinct: 0,
        subbands: ll_subbands,
    });

    for (level_idx, level) in component.dwt.levels.iter().enumerate() {
        let step_base = level_idx
            .checked_mul(3)
            .and_then(|index| index.checked_add(1))
            .ok_or(crate::EncodeError::ArithmeticOverflow {
                what: "batch precomputed 9/7 step index",
            })?;
        let mut subbands = tracker
            .try_vec::<PreparedEncodeSubband>(3, "batch precomputed 9/7 detail subband owners")?;
        for (coefficients, width, height, step_index, sub_band_type) in [
            (
                level.hl.as_slice(),
                level.high_width,
                level.low_height,
                step_base,
                SubBandType::HighLow,
            ),
            (
                level.lh.as_slice(),
                level.low_width,
                level.high_height,
                step_base + 1,
                SubBandType::LowHigh,
            ),
            (
                level.hh.as_slice(),
                level.high_width,
                level.high_height,
                step_base + 2,
                SubBandType::HighHigh,
            ),
        ] {
            subbands.push(try_prepared_subband(
                coefficients,
                width,
                height,
                *step_sizes.get(step_index).ok_or_else(|| {
                    NativeEncodePipelineError::internal_invariant(
                        "irreversible quantization step missing",
                    )
                })?,
                bit_depth,
                guard_bits,
                code_block_width,
                code_block_height,
                sub_band_type,
                session,
                tracker,
                cpu,
            )?);
        }
        packets.push(PreparedResolutionPacket {
            component: component_idx,
            resolution: u32::try_from(level_idx + 1).map_err(|_| {
                NativeEncodePipelineError::arithmetic_overflow("resolution index exceeds u32")
            })?,
            precinct: 0,
            subbands,
        });
    }
    Ok(packets)
}

#[expect(
    clippy::too_many_arguments,
    reason = "the subband preparation boundary keeps codec geometry and retained ownership explicit"
)]
fn try_prepared_subband(
    coefficients: &[f32],
    width: u32,
    height: u32,
    step_size: QuantStepSize,
    bit_depth: u8,
    guard_bits: u8,
    code_block_width: u32,
    code_block_height: u32,
    sub_band_type: SubBandType,
    session: &NativeEncodeSession<'_>,
    tracker: &mut ConstructionTracker<'_, '_>,
    cpu: &mut CpuOnlyJ2kEncodeStageAccelerator,
) -> NativeEncodePipelineResult<PreparedEncodeSubband> {
    let retained_base_bytes = tracker.retained_bytes("batch precomputed 9/7 prepared graph")?;
    let request = F32SubbandEncodeRequest {
        coefficients,
        width,
        height,
        step_size: &step_size,
        bit_depth,
        guard_bits,
        reversible: false,
        block_coding_mode: BlockCodingMode::HighThroughput,
        cb_width: code_block_width,
        cb_height: code_block_height,
        sub_band_type,
        roi_shift: 0,
        roi_regions: &[],
        roi_scale: 1,
        ht_target_coding_passes: 1,
        session,
        retained_base_bytes,
    };
    let prepared = prepare_subband_for_session(&request, cpu)?;
    let retained = prepared_subbands_ownership(core::slice::from_ref(&prepared), 0)?.total()?;
    tracker.retain_existing(retained, "batch precomputed 9/7 prepared subband")?;
    Ok(prepared)
}

fn code_block_dimension(exponent: u8, what: &'static str) -> Result<u32, &'static str> {
    let exponent = exponent.checked_add(2).ok_or(what)?;
    1_u32.checked_shl(u32::from(exponent)).ok_or(what)
}

#[cfg(test)]
mod test_support;
#[cfg(test)]
pub(super) use self::test_support::{
    copy_code_block_coefficients, downcast_i64_coefficients_to_i32,
};
