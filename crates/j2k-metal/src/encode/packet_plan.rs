// SPDX-License-Identifier: Apache-2.0

use j2k::{
    EncodeBackendPreference, J2kBlockCodingMode, J2kEncodeValidation, J2kLosslessEncodeOptions,
    J2kProgressionOrder, ReversibleTransform,
};
use j2k_native::{
    EncodeProgressionOrder, EncodedHtJ2kCodeBlock, J2kHtj2kTileEncodeJob,
    J2kPacketizationBlockCodingMode, J2kPacketizationCodeBlock, J2kPacketizationPacketDescriptor,
    J2kPacketizationProgressionOrder, J2kPacketizationResolution, J2kPacketizationSubband,
};

use super::plan::LosslessDeviceEncodePlan;
use crate::compute;

const AUTO_HTJ2K_HOST_RESIDENT_MIN_PIXELS: usize = 1024 * 1024;

fn lossless_progression_from_packetization_order(
    order: J2kPacketizationProgressionOrder,
) -> J2kProgressionOrder {
    match order {
        J2kPacketizationProgressionOrder::Lrcp => J2kProgressionOrder::Lrcp,
        J2kPacketizationProgressionOrder::Rlcp => J2kProgressionOrder::Rlcp,
        J2kPacketizationProgressionOrder::Rpcl => J2kProgressionOrder::Rpcl,
        J2kPacketizationProgressionOrder::Pcrl => J2kProgressionOrder::Pcrl,
        J2kPacketizationProgressionOrder::Cprl => J2kProgressionOrder::Cprl,
    }
}

pub(super) fn lossless_options_for_resident_htj2k_tile_job(
    job: J2kHtj2kTileEncodeJob<'_>,
) -> Option<J2kLosslessEncodeOptions> {
    if job.num_components != 3
        || job.bit_depth != 8
        || job.signed
        || !job.reversible
        || !job.use_mct
        || job.guard_bits != 2
        || job.code_block_width != 64
        || job.code_block_height != 64
    {
        return None;
    }
    if job.component_sampling.len() != usize::from(job.num_components)
        || job
            .component_sampling
            .iter()
            .any(|&(x_sampling, y_sampling)| x_sampling != 1 || y_sampling != 1)
    {
        return None;
    }
    let expected_len = (job.width as usize)
        .checked_mul(job.height as usize)?
        .checked_mul(usize::from(job.num_components))?;
    if expected_len != job.pixels.len() {
        return None;
    }
    Some(J2kLosslessEncodeOptions::new(
        EncodeBackendPreference::Auto,
        J2kBlockCodingMode::HighThroughput,
        lossless_progression_from_packetization_order(job.progression_order),
        Some(job.num_decomposition_levels),
        ReversibleTransform::Rct53,
        J2kEncodeValidation::External,
    ))
}

pub(super) fn should_use_resident_htj2k_host_tile_for_auto(job: J2kHtj2kTileEncodeJob<'_>) -> bool {
    (job.width as usize).saturating_mul(job.height as usize) >= AUTO_HTJ2K_HOST_RESIDENT_MIN_PIXELS
}

pub(super) fn packet_descriptors_for_lossless_device_order(
    packet_count: usize,
    num_components: u8,
    progression_order: EncodeProgressionOrder,
) -> Result<Vec<J2kPacketizationPacketDescriptor>, crate::Error> {
    let component_count = usize::from(num_components).max(1);
    let mut descriptors = (0..packet_count)
        .map(|packet_index| {
            Ok(J2kPacketizationPacketDescriptor {
                packet_index: u32::try_from(packet_index).map_err(|_| {
                    crate::Error::MetalKernel {
                        message: "J2K Metal resident encode packet index exceeds u32".to_string(),
                    }
                })?,
                state_index: u32::try_from(packet_index).map_err(|_| {
                    crate::Error::MetalKernel {
                        message: "J2K Metal resident encode packet state index exceeds u32"
                            .to_string(),
                    }
                })?,
                layer: 0,
                resolution: u32::try_from(packet_index / component_count).map_err(|_| {
                    crate::Error::MetalKernel {
                        message: "J2K Metal resident encode packet resolution exceeds u32"
                            .to_string(),
                    }
                })?,
                component: u8::try_from(packet_index % component_count).map_err(|_| {
                    crate::Error::MetalKernel {
                        message: "J2K Metal resident encode packet component exceeds u8"
                            .to_string(),
                    }
                })?,
                precinct: 0,
            })
        })
        .collect::<Result<Vec<_>, crate::Error>>()?;
    sort_lossless_device_packet_descriptors(&mut descriptors, progression_order);
    Ok(descriptors)
}

fn sort_lossless_device_packet_descriptors(
    descriptors: &mut [J2kPacketizationPacketDescriptor],
    progression_order: EncodeProgressionOrder,
) {
    match progression_order {
        EncodeProgressionOrder::Lrcp => descriptors.sort_by_key(|descriptor| {
            (
                descriptor.layer,
                descriptor.resolution,
                descriptor.component,
                descriptor.precinct,
            )
        }),
        EncodeProgressionOrder::Rlcp => descriptors.sort_by_key(|descriptor| {
            (
                descriptor.resolution,
                descriptor.layer,
                descriptor.component,
                descriptor.precinct,
            )
        }),
        EncodeProgressionOrder::Rpcl => descriptors.sort_by_key(|descriptor| {
            (
                descriptor.resolution,
                descriptor.precinct,
                descriptor.component,
                descriptor.layer,
            )
        }),
        EncodeProgressionOrder::Pcrl => descriptors.sort_by_key(|descriptor| {
            (
                descriptor.precinct,
                descriptor.component,
                descriptor.resolution,
                descriptor.layer,
            )
        }),
        EncodeProgressionOrder::Cprl => descriptors.sort_by_key(|descriptor| {
            (
                descriptor.component,
                descriptor.precinct,
                descriptor.resolution,
                descriptor.layer,
            )
        }),
    }
}

pub(super) fn resident_packetization_resolutions_from_lossless_device_plan(
    plan: &LosslessDeviceEncodePlan,
) -> Result<Vec<compute::J2kResidentPacketizationResolution>, crate::Error> {
    plan.resolutions
        .iter()
        .map(|resolution| {
            let subbands = resolution
                .subbands
                .iter()
                .map(|subband| {
                    let code_block_end = subband
                        .code_block_start
                        .checked_add(subband.code_block_count)
                        .ok_or_else(|| crate::Error::MetalKernel {
                            message: "J2K Metal resident encode code-block range overflow"
                                .to_string(),
                        })?;
                    if code_block_end > plan.code_blocks.len() {
                        return Err(crate::Error::MetalKernel {
                            message: "J2K Metal resident encode code-block range out of bounds"
                                .to_string(),
                        });
                    }
                    Ok(compute::J2kResidentPacketizationSubband {
                        code_block_start: u32::try_from(subband.code_block_start).map_err(
                            |_| crate::Error::MetalKernel {
                                message: "J2K Metal resident encode code-block offset exceeds u32"
                                    .to_string(),
                            },
                        )?,
                        code_block_count: u32::try_from(subband.code_block_count).map_err(
                            |_| crate::Error::MetalKernel {
                                message: "J2K Metal resident encode code-block count exceeds u32"
                                    .to_string(),
                            },
                        )?,
                        num_cbs_x: subband.num_cbs_x,
                        num_cbs_y: subband.num_cbs_y,
                    })
                })
                .collect::<Result<Vec<_>, crate::Error>>()?;
            Ok(compute::J2kResidentPacketizationResolution { subbands })
        })
        .collect()
}

pub(super) fn packetization_progression_order(
    order: EncodeProgressionOrder,
) -> J2kPacketizationProgressionOrder {
    match order {
        EncodeProgressionOrder::Lrcp => J2kPacketizationProgressionOrder::Lrcp,
        EncodeProgressionOrder::Rlcp => J2kPacketizationProgressionOrder::Rlcp,
        EncodeProgressionOrder::Rpcl => J2kPacketizationProgressionOrder::Rpcl,
        EncodeProgressionOrder::Pcrl => J2kPacketizationProgressionOrder::Pcrl,
        EncodeProgressionOrder::Cprl => J2kPacketizationProgressionOrder::Cprl,
    }
}

pub(super) fn cpu_packetization_resolutions_from_lossless_device_plan<'a>(
    plan: &LosslessDeviceEncodePlan,
    encoded_blocks: &'a [EncodedHtJ2kCodeBlock],
) -> Result<Vec<J2kPacketizationResolution<'a>>, crate::Error> {
    if encoded_blocks.len() != plan.code_blocks.len() {
        return Err(crate::Error::MetalKernel {
            message: "J2K Metal resident hybrid HT block count mismatch".to_string(),
        });
    }
    plan.resolutions
        .iter()
        .map(|resolution| {
            let subbands = resolution
                .subbands
                .iter()
                .map(|subband| {
                    let code_block_end = subband
                        .code_block_start
                        .checked_add(subband.code_block_count)
                        .ok_or_else(|| crate::Error::MetalKernel {
                            message: "J2K Metal resident hybrid code-block range overflow"
                                .to_string(),
                        })?;
                    let encoded = encoded_blocks
                        .get(subband.code_block_start..code_block_end)
                        .ok_or_else(|| crate::Error::MetalKernel {
                            message: "J2K Metal resident hybrid code-block range out of bounds"
                                .to_string(),
                        })?;
                    let code_blocks = encoded
                        .iter()
                        .map(|block| J2kPacketizationCodeBlock {
                            data: block.data.as_slice(),
                            ht_cleanup_length: block.cleanup_length,
                            ht_refinement_length: block.refinement_length,
                            num_coding_passes: block.num_coding_passes,
                            num_zero_bitplanes: block.num_zero_bitplanes,
                            previously_included: false,
                            l_block: 3,
                            block_coding_mode: J2kPacketizationBlockCodingMode::HighThroughput,
                        })
                        .collect();
                    Ok(J2kPacketizationSubband {
                        code_blocks,
                        num_cbs_x: subband.num_cbs_x,
                        num_cbs_y: subband.num_cbs_y,
                    })
                })
                .collect::<Result<Vec<_>, crate::Error>>()?;
            Ok(J2kPacketizationResolution { subbands })
        })
        .collect()
}
