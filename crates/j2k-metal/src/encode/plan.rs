// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k::{J2kBlockCodingMode, J2kLosslessEncodeOptions, J2kProgressionOrder};
use j2k_native::{EncodeProgressionOrder, J2kSubBandType};

use crate::compute;

#[derive(Clone, Copy)]
pub(super) struct LosslessSubbandPlan {
    pub(super) num_cbs_x: u32,
    pub(super) num_cbs_y: u32,
    pub(super) code_block_start: usize,
    pub(super) code_block_count: usize,
}

#[derive(Clone)]
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

pub(super) const RESIDENT_CLASSIC_CODE_BLOCK_EDGE: u32 = 32;

fn lossless_device_encode_levels(width: u32, height: u32, options: J2kLosslessEncodeOptions) -> u8 {
    const MIN_LOSSLESS_DWT_DIMENSION: u32 = 64;
    let levels = if matches!(
        options.progression,
        J2kProgressionOrder::Rpcl | J2kProgressionOrder::Pcrl | J2kProgressionOrder::Cprl
    ) {
        let mut levels = 0u8;
        let mut w = width;
        let mut h = height;
        let max_levels = if width.min(height) <= 1 {
            0
        } else {
            width.min(height).ilog2() as u8
        };
        while w.min(h) > MIN_LOSSLESS_DWT_DIMENSION && levels < max_levels {
            w = w.div_ceil(2);
            h = h.div_ceil(2);
            levels = levels.saturating_add(1);
        }
        levels
    } else {
        u8::from(width.min(height) >= MIN_LOSSLESS_DWT_DIMENSION)
    };

    options
        .max_decomposition_levels
        .map_or(levels, |requested| {
            let max_levels = if width.min(height) <= 1 {
                0
            } else {
                width.min(height).ilog2() as u8
            };
            requested.min(max_levels)
        })
}

#[derive(Clone, Copy)]
struct LosslessDwtLevelPlan {
    low_width: u32,
    low_height: u32,
    high_width: u32,
    high_height: u32,
}

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
    if edge < 4 || !edge.is_power_of_two() {
        return Err(crate::Error::MetalKernel {
            message: format!(
                "J2K Metal resident encode {axis} code-block edge must be a power of two >= 4"
            ),
        });
    }
    let exp = edge
        .trailing_zeros()
        .checked_sub(2)
        .ok_or_else(|| crate::Error::MetalKernel {
            message: format!("J2K Metal resident encode {axis} code-block exponent underflow"),
        })?;
    if exp > 8 {
        return Err(crate::Error::MetalKernel {
            message: format!(
                "J2K Metal resident encode {axis} code-block edge exceeds JPEG 2000 COD range"
            ),
        });
    }
    u8::try_from(exp).map_err(|_| crate::Error::MetalKernel {
        message: format!("J2K Metal resident encode {axis} code-block exponent exceeds u8"),
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
) -> Vec<LosslessDwtLevelPlan> {
    let mut levels = Vec::with_capacity(usize::from(num_decomposition_levels));
    let mut current_width = width;
    let mut current_height = height;
    for _ in 0..num_decomposition_levels {
        let low_width = current_width.div_ceil(2);
        let low_height = current_height.div_ceil(2);
        let high_width = current_width / 2;
        let high_height = current_height / 2;
        levels.push(LosslessDwtLevelPlan {
            low_width,
            low_height,
            high_width,
            high_height,
        });
        current_width = low_width;
        current_height = low_height;
    }
    levels
}

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
    let num_decomposition_levels = lossless_device_encode_levels(width, height, options);
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
        let dwt_levels = lossless_dwt_level_plans(width, height, num_decomposition_levels);
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
                    total_bitplanes: guard_bits.saturating_add(bit_depth).saturating_sub(1),
                },
            )?;
            component_packets.push(base_packet);
        } else {
            let final_ll = dwt_levels
                .last()
                .expect("nonzero DWT level count has a final LL level");
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
                    total_bitplanes: guard_bits.saturating_add(bit_depth).saturating_sub(1),
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
                        total_bitplanes: guard_bits.saturating_add(bit_depth),
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
                        total_bitplanes: guard_bits.saturating_add(bit_depth),
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
                        total_bitplanes: guard_bits.saturating_add(bit_depth).saturating_add(1),
                    },
                )?;
                component_packets.push(detail_packet);
            }
        }
        component_resolutions.push(component_packets);
    }

    let resolution_count = component_resolutions.first().map_or(0usize, Vec::len);
    let mut resolutions =
        Vec::with_capacity(resolution_count.saturating_mul(usize::from(components)));
    for resolution in 0..resolution_count {
        for component in &component_resolutions {
            resolutions.push(component[resolution].clone());
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
