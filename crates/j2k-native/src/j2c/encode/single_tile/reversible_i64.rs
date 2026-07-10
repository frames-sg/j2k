// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::{
    deinterleave_to_i64, encode_i64_component_resolution_packets, fdwt, forward_rct_i64,
    prepare_subband_i64, roi_subband_scale, vec, ComponentRoiEncodePlan, EncodeOptions,
    EncodeParams, I64CodestreamPacketRequest, I64PacketizeRequest, I64SubbandEncodeSettings,
    J2kEncodeStageAccelerator, PreparedResolutionPacket, QuantStepSize, SubBandType, Vec,
    MAX_CLASSIC_REVERSIBLE_MARKER_BITPLANES,
};

pub(super) struct ReversibleI64SingleTileRequest<'a, A: J2kEncodeStageAccelerator> {
    pub(super) pixels: &'a [u8],
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) num_pixels: usize,
    pub(super) num_components: u16,
    pub(super) bit_depth: u8,
    pub(super) signed: bool,
    pub(super) options: &'a EncodeOptions,
    pub(super) params: &'a EncodeParams,
    pub(super) quant_params: &'a [(u16, u16)],
    pub(super) step_sizes: &'a [QuantStepSize],
    pub(super) roi_plans: &'a [ComponentRoiEncodePlan],
    pub(super) use_mct: bool,
    pub(super) guard_bits: u8,
    pub(super) num_levels: u8,
    pub(super) cb_width: u32,
    pub(super) cb_height: u32,
    pub(super) ht_target_coding_passes: u8,
    pub(super) accelerator: &'a mut A,
}

#[expect(
    clippy::similar_names,
    reason = "paired axis, subband, and marker names follow JPEG 2000 specification notation"
)]
#[expect(
    clippy::too_many_lines,
    reason = "the ordered JPEG 2000 state machine stays cohesive to preserve marker, packet, pass, and sample order"
)]
pub(super) fn encode_reversible_i64_single_tile_codestream<A: J2kEncodeStageAccelerator>(
    request: ReversibleI64SingleTileRequest<'_, A>,
) -> Result<Vec<u8>, &'static str> {
    let ReversibleI64SingleTileRequest {
        pixels,
        width,
        height,
        num_pixels,
        num_components,
        bit_depth,
        signed,
        options,
        params,
        quant_params,
        step_sizes,
        roi_plans,
        use_mct,
        guard_bits,
        num_levels,
        cb_width,
        cb_height,
        ht_target_coding_passes,
        accelerator,
    } = request;
    let max_reversible_gain = if num_levels == 0 { 0 } else { 2 };
    if u16::from(bit_depth) + max_reversible_gain > MAX_CLASSIC_REVERSIBLE_MARKER_BITPLANES {
        return Err("25-38 bit reversible encode exceeds the current no-quantization guard/exponent signaling limit");
    }

    let mut components = deinterleave_to_i64(pixels, num_pixels, num_components, bit_depth, signed);
    if use_mct {
        forward_rct_i64(&mut components);
    }

    let decompositions = components
        .iter()
        .map(|component| fdwt::forward_dwt_i64(component, width, height, num_levels))
        .collect::<Vec<_>>();

    let mut component_resolution_packets = Vec::with_capacity(num_components as usize);
    for (component_idx, decomposition) in decompositions
        .iter()
        .take(num_components as usize)
        .enumerate()
    {
        let component = u16::try_from(component_idx).map_err(|_| "component index exceeds u16")?;
        let roi_shift = params
            .roi_component_shifts
            .get(component_idx)
            .copied()
            .unwrap_or(0);
        let roi_plan = roi_plans
            .get(component_idx)
            .ok_or("ROI plan count does not match component count")?;
        let mut packets = Vec::with_capacity(num_levels as usize + 1);

        let ll_roi_scale = roi_subband_scale(num_levels, None)?;
        let base_subband_settings = I64SubbandEncodeSettings {
            guard_bits,
            cb_width,
            cb_height,
            roi_shift,
            roi_regions: &roi_plan.regions,
            roi_scale: ll_roi_scale,
            block_coding_mode: params.block_coding_mode,
            ht_target_coding_passes,
        };
        let ll_subband = prepare_subband_i64(
            &decomposition.ll,
            decomposition.ll_width,
            decomposition.ll_height,
            step_sizes
                .first()
                .ok_or("reversible quantization step missing")?,
            SubBandType::LowLow,
            base_subband_settings,
        )?;
        packets.push(PreparedResolutionPacket {
            component,
            resolution: 0,
            precinct: 0,
            subbands: vec![ll_subband],
        });

        for (level_idx, level) in decomposition.levels.iter().enumerate() {
            let step_base = 1 + level_idx * 3;
            let level_roi_scale = roi_subband_scale(num_levels, Some(level_idx))?;
            let hl_subband = prepare_subband_i64(
                &level.hl,
                level.high_width,
                level.low_height,
                step_sizes
                    .get(step_base)
                    .ok_or("reversible quantization step missing")?,
                SubBandType::HighLow,
                I64SubbandEncodeSettings {
                    roi_scale: level_roi_scale,
                    ..base_subband_settings
                },
            )?;
            let lh_subband = prepare_subband_i64(
                &level.lh,
                level.low_width,
                level.high_height,
                step_sizes
                    .get(step_base + 1)
                    .ok_or("reversible quantization step missing")?,
                SubBandType::LowHigh,
                I64SubbandEncodeSettings {
                    roi_scale: level_roi_scale,
                    ..base_subband_settings
                },
            )?;
            let hh_subband = prepare_subband_i64(
                &level.hh,
                level.high_width,
                level.high_height,
                step_sizes
                    .get(step_base + 2)
                    .ok_or("reversible quantization step missing")?,
                SubBandType::HighHigh,
                I64SubbandEncodeSettings {
                    roi_scale: level_roi_scale,
                    ..base_subband_settings
                },
            )?;

            packets.push(PreparedResolutionPacket {
                component,
                resolution: u32::try_from(level_idx + 1)
                    .map_err(|_| "resolution index exceeds u32")?,
                precinct: 0,
                subbands: vec![hl_subband, lh_subband, hh_subband],
            });
        }
        component_resolution_packets.push(packets);
    }

    encode_i64_component_resolution_packets(
        component_resolution_packets,
        I64CodestreamPacketRequest {
            packetize: I64PacketizeRequest {
                width,
                height,
                num_components,
                num_levels,
                params,
                options,
                accelerator,
            },
            quant_params,
        },
    )
}
