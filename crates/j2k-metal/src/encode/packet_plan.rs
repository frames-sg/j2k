// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k::{
    EncodeBackendPreference, J2kBlockCodingMode, J2kEncodeValidation, J2kLosslessEncodeOptions,
    J2kProgressionOrder, ReversibleTransform,
};
use j2k_native::{
    sort_packet_descriptors_for_progression, EncodeProgressionOrder, EncodedHtJ2kCodeBlock,
    J2kHtj2kTileEncodeJob, J2kPacketizationBlockCodingMode, J2kPacketizationCodeBlock,
    J2kPacketizationPacketDescriptor, J2kPacketizationProgressionOrder, J2kPacketizationResolution,
    J2kPacketizationSubband,
};

use super::plan::LosslessDeviceEncodePlan;
use crate::batch_allocation::{checked_count_sum, BatchMetadataBudget, BatchMetadataRequest};
use crate::compute;

const AUTO_HTJ2K_HOST_RESIDENT_MIN_PIXELS: usize = 512 * 512;

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
    if !matches!(job.num_components, 1 | 3)
        || job.bit_depth != 8
        || job.signed
        || !job.reversible
        || (job.num_components == 1 && job.use_mct)
        || (job.num_components == 3 && !job.use_mct)
        || (job.num_components == 1 && job.guard_bits != 1)
        || (job.num_components == 3 && job.guard_bits != 2)
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
        if job.use_mct {
            ReversibleTransform::Rct53
        } else {
            ReversibleTransform::None53
        },
        J2kEncodeValidation::External,
    ))
}

pub(super) fn should_use_resident_htj2k_host_shape_for_auto(width: u32, height: u32) -> bool {
    (width as usize).saturating_mul(height as usize) >= AUTO_HTJ2K_HOST_RESIDENT_MIN_PIXELS
}

pub(super) fn should_use_resident_htj2k_host_tile_for_auto(job: J2kHtj2kTileEncodeJob<'_>) -> bool {
    let _ = job;
    // The encode-stage tile callback represents one host-output frame. Keep
    // Auto on CPU here; callers that can amortize resident setup should use
    // the batch Metal-buffer APIs.
    false
}

pub(super) fn packet_descriptors_for_lossless_device_order(
    packet_count: usize,
    num_components: u8,
    progression_order: EncodeProgressionOrder,
) -> Result<Vec<J2kPacketizationPacketDescriptor>, crate::Error> {
    let component_count = usize::from(num_components).max(1);
    let mut budget = BatchMetadataBudget::new("J2K Metal packet descriptor metadata");
    let mut descriptors = budget.try_vec(packet_count, "J2K Metal packet descriptors")?;
    for packet_index in 0..packet_count {
        descriptors.push(J2kPacketizationPacketDescriptor {
            packet_index: u32::try_from(packet_index).map_err(|_| crate::Error::MetalKernel {
                message: "J2K Metal resident encode packet index exceeds u32".to_string(),
            })?,
            state_index: u32::try_from(packet_index).map_err(|_| crate::Error::MetalKernel {
                message: "J2K Metal resident encode packet state index exceeds u32".to_string(),
            })?,
            layer: 0,
            resolution: u32::try_from(packet_index / component_count).map_err(|_| {
                crate::Error::MetalKernel {
                    message: "J2K Metal resident encode packet resolution exceeds u32".to_string(),
                }
            })?,
            component: u16::try_from(packet_index % component_count).map_err(|_| {
                crate::Error::MetalKernel {
                    message: "J2K Metal resident encode packet component exceeds u16".to_string(),
                }
            })?,
            precinct: 0,
        });
    }
    sort_packet_descriptors_for_progression(
        &mut descriptors,
        packetization_progression_order(progression_order),
    );
    Ok(descriptors)
}

pub(super) fn resident_packetization_resolutions_from_lossless_device_plan(
    plan: &LosslessDeviceEncodePlan,
) -> Result<Vec<compute::J2kResidentPacketizationResolution>, crate::Error> {
    let subband_count = checked_count_sum(
        plan.resolutions
            .iter()
            .map(|resolution| resolution.subbands.len()),
        "J2K Metal resident packetization subband count",
    )?;
    for subband in plan
        .resolutions
        .iter()
        .flat_map(|resolution| &resolution.subbands)
    {
        let code_block_end = subband
            .code_block_start
            .checked_add(subband.code_block_count)
            .ok_or_else(|| crate::Error::MetalKernel {
                message: "J2K Metal resident encode code-block range overflow".to_string(),
            })?;
        if code_block_end > plan.code_blocks.len() {
            return Err(crate::Error::MetalKernel {
                message: "J2K Metal resident encode code-block range out of bounds".to_string(),
            });
        }
    }

    let mut budget = BatchMetadataBudget::new("J2K Metal resident packetization metadata");
    budget.preflight(&[
        BatchMetadataRequest::of::<compute::J2kResidentPacketizationResolution>(
            plan.resolutions.len(),
        ),
        BatchMetadataRequest::of::<compute::J2kResidentPacketizationSubband>(subband_count),
    ])?;
    let mut resolutions = budget.try_vec(
        plan.resolutions.len(),
        "J2K Metal resident packetization resolutions",
    )?;
    for resolution in &plan.resolutions {
        let mut subbands = budget.try_vec(
            resolution.subbands.len(),
            "J2K Metal resident packetization subbands",
        )?;
        for subband in &resolution.subbands {
            subbands.push(compute::J2kResidentPacketizationSubband {
                code_block_start: u32::try_from(subband.code_block_start).map_err(|_| {
                    crate::Error::MetalKernel {
                        message: "J2K Metal resident encode code-block offset exceeds u32"
                            .to_string(),
                    }
                })?,
                code_block_count: u32::try_from(subband.code_block_count).map_err(|_| {
                    crate::Error::MetalKernel {
                        message: "J2K Metal resident encode code-block count exceeds u32"
                            .to_string(),
                    }
                })?,
                num_cbs_x: subband.num_cbs_x,
                num_cbs_y: subband.num_cbs_y,
            });
        }
        resolutions.push(compute::J2kResidentPacketizationResolution { subbands });
    }
    Ok(resolutions)
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
    expected_code_block_count: usize,
    encoded_blocks: &'a [EncodedHtJ2kCodeBlock],
) -> Result<Vec<J2kPacketizationResolution<'a>>, crate::Error> {
    if encoded_blocks.len() != expected_code_block_count {
        return Err(crate::Error::MetalKernel {
            message: "J2K Metal resident hybrid HT block count mismatch".to_string(),
        });
    }
    let subband_count = checked_count_sum(
        plan.resolutions
            .iter()
            .map(|resolution| resolution.subbands.len()),
        "J2K Metal CPU packetization subband count",
    )?;
    let code_block_count = checked_count_sum(
        plan.resolutions
            .iter()
            .flat_map(|resolution| &resolution.subbands)
            .map(|subband| subband.code_block_count),
        "J2K Metal CPU packetization code-block count",
    )?;
    for subband in plan
        .resolutions
        .iter()
        .flat_map(|resolution| &resolution.subbands)
    {
        let code_block_end = subband
            .code_block_start
            .checked_add(subband.code_block_count)
            .ok_or_else(|| crate::Error::MetalKernel {
                message: "J2K Metal resident hybrid code-block range overflow".to_string(),
            })?;
        if code_block_end > encoded_blocks.len() {
            return Err(crate::Error::MetalKernel {
                message: "J2K Metal resident hybrid code-block range out of bounds".to_string(),
            });
        }
    }

    let mut budget = BatchMetadataBudget::new("J2K Metal CPU packetization metadata");
    budget.preflight(&[
        BatchMetadataRequest::of::<J2kPacketizationResolution<'a>>(plan.resolutions.len()),
        BatchMetadataRequest::of::<J2kPacketizationSubband<'a>>(subband_count),
        BatchMetadataRequest::of::<J2kPacketizationCodeBlock<'a>>(code_block_count),
    ])?;
    let mut resolutions = budget.try_vec(
        plan.resolutions.len(),
        "J2K Metal CPU packetization resolutions",
    )?;
    for resolution in &plan.resolutions {
        let mut subbands = budget.try_vec(
            resolution.subbands.len(),
            "J2K Metal CPU packetization subbands",
        )?;
        for subband in &resolution.subbands {
            let code_block_end = subband
                .code_block_start
                .checked_add(subband.code_block_count)
                .ok_or_else(|| crate::Error::MetalKernel {
                    message: "J2K Metal resident hybrid code-block range overflow".to_string(),
                })?;
            let encoded = encoded_blocks
                .get(subband.code_block_start..code_block_end)
                .ok_or_else(|| crate::Error::MetalKernel {
                    message: "J2K Metal resident hybrid code-block range out of bounds".to_string(),
                })?;
            let mut code_blocks =
                budget.try_vec(encoded.len(), "J2K Metal CPU packetization code blocks")?;
            for block in encoded {
                code_blocks.push(J2kPacketizationCodeBlock {
                    data: block.data.as_slice(),
                    ht_cleanup_length: block.cleanup_length,
                    ht_refinement_length: block.refinement_length,
                    num_coding_passes: block.num_coding_passes,
                    num_zero_bitplanes: block.num_zero_bitplanes,
                    previously_included: false,
                    l_block: 3,
                    block_coding_mode: J2kPacketizationBlockCodingMode::HighThroughput,
                });
            }
            subbands.push(J2kPacketizationSubband {
                code_blocks,
                num_cbs_x: subband.num_cbs_x,
                num_cbs_y: subband.num_cbs_y,
            });
        }
        resolutions.push(J2kPacketizationResolution { subbands });
    }
    Ok(resolutions)
}
