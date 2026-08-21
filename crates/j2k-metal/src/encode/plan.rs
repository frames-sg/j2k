// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k::{J2kBlockCodingMode, J2kLosslessEncodeOptions, J2kProgressionOrder};
use j2k_native::{EncodeProgressionOrder, J2kSubBandType};
use j2k_types::encode_geometry::{
    code_block_exponent, encode_dwt_level_dimensions, lossless_decomposition_levels,
    reversible_subband_total_bitplanes, CodeBlockGeometryError, EncodeDwtLevelDimensions,
};

use crate::engine as compute;

#[derive(Clone, Copy)]
pub(super) struct LosslessSubbandPlan {
    pub(super) num_cbs_x: u32,
    pub(super) num_cbs_y: u32,
    pub(super) code_block_start: usize,
    pub(super) code_block_count: usize,
}

#[derive(Default)]
pub(super) struct LosslessResolutionPlan {
    pub(super) subbands: Vec<LosslessSubbandPlan>,
}

pub(super) struct LosslessDeviceEncodePlan {
    pub(super) components: u8,
    pub(super) bit_depth: u8,
    pub(super) block_coding_mode: J2kBlockCodingMode,
    pub(super) num_decomposition_levels: u8,
    pub(super) use_mct: bool,
    pub(super) guard_bits: u8,
    pub(super) code_block_width_exp: u8,
    pub(super) code_block_height_exp: u8,
    pub(super) code_blocks: Vec<compute::J2kLosslessDeviceCodeBlock>,
    pub(super) resolutions: Vec<LosslessResolutionPlan>,
    pub(super) progression_order: EncodeProgressionOrder,
    pub(super) write_tlm: bool,
}

impl LosslessDeviceEncodePlan {
    pub(super) fn take_code_blocks(&mut self) -> Vec<compute::J2kLosslessDeviceCodeBlock> {
        std::mem::take(&mut self.code_blocks)
    }
}

pub(super) const RESIDENT_CLASSIC_CODE_BLOCK_EDGE: u32 = 32;

#[derive(Clone, Copy)]
struct LosslessSubbandInput {
    component: u32,
    subband_x: u32,
    subband_y: u32,
    width: u32,
    height: u32,
    sub_band_type: J2kSubBandType,
    total_bitplanes: u8,
}

fn lossless_code_block_exp(edge: u32, axis: &str) -> Result<u8, crate::Error> {
    code_block_exponent(edge).map_err(|error| crate::Error::MetalKernel {
        message: match error {
            CodeBlockGeometryError::DimensionTooSmall
            | CodeBlockGeometryError::DimensionNotPowerOfTwo => format!(
                "J2K Metal resident encode {axis} code-block edge must be a power of two >= 4"
            ),
            CodeBlockGeometryError::StoredExponentTooLarge
            | CodeBlockGeometryError::AreaTooLarge => format!(
                "J2K Metal resident encode {axis} code-block edge exceeds JPEG 2000 COD range"
            ),
        },
    })
}

fn push_lossless_subband_plan(
    resolution: &mut LosslessResolutionPlan,
    code_blocks: &mut Vec<compute::J2kLosslessDeviceCodeBlock>,
    coefficient_offset: &mut u32,
    code_block_width: u32,
    code_block_height: u32,
    subband: LosslessSubbandInput,
) -> Result<(), crate::Error> {
    if subband.width == 0 || subband.height == 0 {
        resolution.subbands.push(LosslessSubbandPlan {
            num_cbs_x: 0,
            num_cbs_y: 0,
            code_block_start: code_blocks.len(),
            code_block_count: 0,
        });
        return Ok(());
    }
    let cb_width = code_block_width;
    let cb_height = code_block_height;
    let num_cbs_x = subband.width.div_ceil(cb_width);
    let num_cbs_y = subband.height.div_ceil(cb_height);
    let code_block_start = code_blocks.len();
    for cby in 0..num_cbs_y {
        for cbx in 0..num_cbs_x {
            let block_x = cbx * cb_width;
            let block_y = cby * cb_height;
            let block_width = (block_x + cb_width).min(subband.width) - block_x;
            let block_height = (block_y + cb_height).min(subband.height) - block_y;
            let coeff_count =
                block_width
                    .checked_mul(block_height)
                    .ok_or_else(|| crate::Error::MetalKernel {
                        message: "J2K Metal resident encode code-block size overflow".to_string(),
                    })?;
            code_blocks.push(compute::J2kLosslessDeviceCodeBlock {
                coefficient_offset: *coefficient_offset,
                component: subband.component,
                subband_x: subband.subband_x,
                subband_y: subband.subband_y,
                block_x,
                block_y,
                width: block_width,
                height: block_height,
                sub_band_type: subband.sub_band_type,
                total_bitplanes: subband.total_bitplanes,
            });
            *coefficient_offset = coefficient_offset.checked_add(coeff_count).ok_or_else(|| {
                crate::Error::MetalKernel {
                    message: "J2K Metal resident encode coefficient offset overflow".to_string(),
                }
            })?;
        }
    }
    resolution.subbands.push(LosslessSubbandPlan {
        num_cbs_x,
        num_cbs_y,
        code_block_start,
        code_block_count: code_blocks.len() - code_block_start,
    });
    Ok(())
}

fn lossless_dwt_level_plans(
    width: u32,
    height: u32,
    num_decomposition_levels: u8,
) -> Result<Vec<EncodeDwtLevelDimensions>, crate::Error> {
    let mut levels = crate::batch_allocation::try_vec(
        usize::from(num_decomposition_levels),
        "J2K Metal resident encode DWT level plans",
    )?;
    for level in encode_dwt_level_dimensions(width, height, num_decomposition_levels) {
        levels.push(level);
    }
    Ok(levels)
}

#[expect(
    clippy::too_many_lines,
    reason = "encode planning validates and derives one internally consistent layout"
)]
pub(super) fn lossless_device_encode_plan(
    width: u32,
    height: u32,
    components: u8,
    bit_depth: u8,
    options: J2kLosslessEncodeOptions,
    code_block_width: u32,
    code_block_height: u32,
) -> Result<Option<LosslessDeviceEncodePlan>, crate::Error> {
    if !matches!(
        options.block_coding_mode,
        J2kBlockCodingMode::Classic | J2kBlockCodingMode::HighThroughput
    ) {
        return Ok(None);
    }
    if code_block_width == 0 || code_block_height == 0 {
        return Err(crate::Error::MetalKernel {
            message: "J2K Metal resident encode code-block dimensions must be non-zero".to_string(),
        });
    }
    let code_block_width_exp = lossless_code_block_exp(code_block_width, "width")?;
    let code_block_height_exp = lossless_code_block_exp(code_block_height, "height")?;
    let num_decomposition_levels = lossless_decomposition_levels(
        width,
        height,
        options.progression.packetization_order(),
        options.max_decomposition_levels,
    );
    let progression_order = match options.progression {
        J2kProgressionOrder::Lrcp => EncodeProgressionOrder::Lrcp,
        J2kProgressionOrder::Rlcp => EncodeProgressionOrder::Rlcp,
        J2kProgressionOrder::Rpcl => EncodeProgressionOrder::Rpcl,
        J2kProgressionOrder::Pcrl => EncodeProgressionOrder::Pcrl,
        J2kProgressionOrder::Cprl => EncodeProgressionOrder::Cprl,
    };
    let use_mct = components >= 3;
    let guard_bits: u8 = if use_mct { 2 } else { 1 };
    let mut code_blocks = Vec::new();
    let mut coefficient_offset = 0u32;
    let mut component_resolutions = Vec::<Vec<LosslessResolutionPlan>>::new();
    for component in 0..components {
        let mut component_packets = Vec::new();
        let dwt_levels = lossless_dwt_level_plans(width, height, num_decomposition_levels)?;
        let mut base_packet = LosslessResolutionPlan {
            subbands: Vec::new(),
        };
        if num_decomposition_levels == 0 {
            push_lossless_subband_plan(
                &mut base_packet,
                &mut code_blocks,
                &mut coefficient_offset,
                code_block_width,
                code_block_height,
                LosslessSubbandInput {
                    component: u32::from(component),
                    subband_x: 0,
                    subband_y: 0,
                    width,
                    height,
                    sub_band_type: J2kSubBandType::LowLow,
                    total_bitplanes: reversible_subband_total_bitplanes(
                        bit_depth,
                        guard_bits,
                        J2kSubBandType::LowLow,
                    ),
                },
            )?;
            component_packets.push(base_packet);
        } else {
            let final_ll = dwt_levels.last().ok_or_else(|| crate::Error::MetalKernel {
                message: "J2K Metal resident encode DWT plan is missing its final LL level"
                    .to_string(),
            })?;
            push_lossless_subband_plan(
                &mut base_packet,
                &mut code_blocks,
                &mut coefficient_offset,
                code_block_width,
                code_block_height,
                LosslessSubbandInput {
                    component: u32::from(component),
                    subband_x: 0,
                    subband_y: 0,
                    width: final_ll.low_width,
                    height: final_ll.low_height,
                    sub_band_type: J2kSubBandType::LowLow,
                    total_bitplanes: reversible_subband_total_bitplanes(
                        bit_depth,
                        guard_bits,
                        J2kSubBandType::LowLow,
                    ),
                },
            )?;
            component_packets.push(base_packet);

            for level in dwt_levels.iter().rev().copied() {
                let mut detail_packet = LosslessResolutionPlan {
                    subbands: Vec::new(),
                };
                push_lossless_subband_plan(
                    &mut detail_packet,
                    &mut code_blocks,
                    &mut coefficient_offset,
                    code_block_width,
                    code_block_height,
                    LosslessSubbandInput {
                        component: u32::from(component),
                        subband_x: level.low_width,
                        subband_y: 0,
                        width: level.high_width,
                        height: level.low_height,
                        sub_band_type: J2kSubBandType::HighLow,
                        total_bitplanes: reversible_subband_total_bitplanes(
                            bit_depth,
                            guard_bits,
                            J2kSubBandType::HighLow,
                        ),
                    },
                )?;
                push_lossless_subband_plan(
                    &mut detail_packet,
                    &mut code_blocks,
                    &mut coefficient_offset,
                    code_block_width,
                    code_block_height,
                    LosslessSubbandInput {
                        component: u32::from(component),
                        subband_x: 0,
                        subband_y: level.low_height,
                        width: level.low_width,
                        height: level.high_height,
                        sub_band_type: J2kSubBandType::LowHigh,
                        total_bitplanes: reversible_subband_total_bitplanes(
                            bit_depth,
                            guard_bits,
                            J2kSubBandType::LowHigh,
                        ),
                    },
                )?;
                push_lossless_subband_plan(
                    &mut detail_packet,
                    &mut code_blocks,
                    &mut coefficient_offset,
                    code_block_width,
                    code_block_height,
                    LosslessSubbandInput {
                        component: u32::from(component),
                        subband_x: level.low_width,
                        subband_y: level.low_height,
                        width: level.high_width,
                        height: level.high_height,
                        sub_band_type: J2kSubBandType::HighHigh,
                        total_bitplanes: reversible_subband_total_bitplanes(
                            bit_depth,
                            guard_bits,
                            J2kSubBandType::HighHigh,
                        ),
                    },
                )?;
                component_packets.push(detail_packet);
            }
        }
        component_resolutions.push(component_packets);
    }

    let resolution_count = component_resolutions.first().map_or(0usize, Vec::len);
    let resolution_capacity = resolution_count
        .checked_mul(usize::from(components))
        .ok_or_else(|| crate::Error::MetalKernel {
            message: "J2K Metal resident encode resolution count overflow".to_string(),
        })?;
    let mut resolutions = crate::batch_allocation::try_vec(
        resolution_capacity,
        "J2K Metal resident encode resolution plans",
    )?;
    for resolution in 0..resolution_count {
        for component in &mut component_resolutions {
            resolutions.push(std::mem::take(&mut component[resolution]));
        }
    }

    Ok(Some(LosslessDeviceEncodePlan {
        components,
        bit_depth,
        block_coding_mode: options.block_coding_mode,
        num_decomposition_levels,
        use_mct,
        guard_bits,
        code_block_width_exp,
        code_block_height_exp,
        code_blocks,
        resolutions,
        progression_order,
        write_tlm: options.progression == J2kProgressionOrder::Rpcl,
    }))
}

#[cfg(test)]
mod tests {
    use super::lossless_device_encode_plan;
    use j2k::{
        j2k_lossless_decomposition_levels_for_options, J2kLosslessEncodeOptions,
        J2kLosslessSamples, J2kProgressionOrder,
    };

    fn assert_plan_matches_facade(
        width: u32,
        height: u32,
        progression: J2kProgressionOrder,
        maximum: Option<u8>,
    ) {
        let sample_count =
            usize::try_from(u64::from(width) * u64::from(height)).expect("fixture sample count");
        let pixels = vec![0; sample_count];
        let samples = J2kLosslessSamples::new(&pixels, width, height, 1, 8, false)
            .expect("valid fixture samples");
        let options = J2kLosslessEncodeOptions::default()
            .with_progression(progression)
            .with_max_decomposition_levels(maximum);
        let expected = j2k_lossless_decomposition_levels_for_options(samples, options);
        let plan = lossless_device_encode_plan(width, height, 1, 8, options, 32, 32)
            .expect("plan result")
            .expect("supported plan");
        assert_eq!(plan.num_decomposition_levels, expected);
    }

    #[test]
    fn resident_geometry_matches_facade_at_policy_boundaries() {
        for progression in [
            J2kProgressionOrder::Lrcp,
            J2kProgressionOrder::Rlcp,
            J2kProgressionOrder::Rpcl,
            J2kProgressionOrder::Pcrl,
            J2kProgressionOrder::Cprl,
        ] {
            assert_plan_matches_facade(63, 128, progression, Some(u8::MAX));
            assert_plan_matches_facade(64, 64, progression, None);
            assert_plan_matches_facade(128, 128, progression, Some(5));
            assert_plan_matches_facade(512, 512, progression, None);
        }
    }

    #[test]
    fn code_block_ownership_transfer_preserves_allocation_without_clone() {
        let mut plan = lossless_device_encode_plan(
            64,
            64,
            1,
            8,
            j2k::J2kLosslessEncodeOptions::default(),
            32,
            32,
        )
        .expect("plan result")
        .expect("supported plan");
        let original_ptr = plan.code_blocks.as_ptr();
        let original_capacity = plan.code_blocks.capacity();

        let code_blocks = plan.take_code_blocks();

        assert!(plan.code_blocks.is_empty());
        assert_eq!(plan.code_blocks.capacity(), 0);
        assert_eq!(code_blocks.as_ptr(), original_ptr);
        assert_eq!(code_blocks.capacity(), original_capacity);
    }
}
