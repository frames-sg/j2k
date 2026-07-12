// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::{
    checked_element_sum, CudaBufferPool, CudaContext, CudaHtj2k97DeviceCodeblockBands,
    CudaHtj2kEncodeResources, CudaTranscodeError, Htj2k97CodeBlockOptions, J2kSubBandType,
    ResidentCompactPreencodedGroups, ResidentPreencodedGroups,
};
use super::output::{
    assemble_compact_preencoded_components_with_budget, assemble_preencoded_components_with_budget,
    split_resident_compact_subband_blocks, split_resident_subband_blocks, Htj2k97ComponentJob,
};
use super::planning::{
    build_resident_subband_group_plans, resident_group_block_count, resident_group_targets,
    resident_subband_encode_plan, resident_targets, ResidentDeviceGroup, ResidentMetadataBudget,
};
use super::{CompactResidentSubbands, ResidentSubbands};

pub(in crate::cuda) fn encode_resident_subbands(
    context: &CudaContext,
    resources: &CudaHtj2kEncodeResources,
    pool: &CudaBufferPool,
    bands: &CudaHtj2k97DeviceCodeblockBands,
    item_count: usize,
    options: Htj2k97CodeBlockOptions,
) -> Result<ResidentSubbands, CudaTranscodeError> {
    let mut budget = ResidentMetadataBudget::new("CUDA resident aggregate metadata");
    let plans = [
        resident_subband_encode_plan(
            &bands.ll,
            item_count,
            bands.low_width,
            bands.low_height,
            J2kSubBandType::LowLow,
            options,
            &mut budget,
        )?,
        resident_subband_encode_plan(
            &bands.hl,
            item_count,
            bands.high_width,
            bands.low_height,
            J2kSubBandType::HighLow,
            options,
            &mut budget,
        )?,
        resident_subband_encode_plan(
            &bands.lh,
            item_count,
            bands.low_width,
            bands.high_height,
            J2kSubBandType::LowHigh,
            options,
            &mut budget,
        )?,
        resident_subband_encode_plan(
            &bands.hh,
            item_count,
            bands.high_width,
            bands.high_height,
            J2kSubBandType::HighHigh,
            options,
            &mut budget,
        )?,
    ];
    let targets = resident_targets(&plans, &mut budget)?;
    let encoded = context
        .encode_htj2k_codeblocks_multi_resident_with_resources_and_pool_and_live_host_bytes(
            &targets,
            resources,
            pool,
            budget.live_bytes(),
        )
        .map_err(|error| {
            CudaTranscodeError::runtime("CUDA resident multi-input HTJ2K encode", error)
        })?;
    let expected_blocks = checked_element_sum(
        &[
            plans[0].jobs.len(),
            plans[1].jobs.len(),
            plans[2].jobs.len(),
            plans[3].jobs.len(),
        ],
        "CUDA resident encoded code-block count",
    )?;
    if encoded.code_blocks().len() != expected_blocks {
        return Err(CudaTranscodeError::Kernel(
            "CUDA resident multi-input HTJ2K encode returned wrong block count",
        ));
    }
    let ht_timings = encoded.stage_timings();
    let dispatches = encoded.execution().kernel_dispatches();
    let encoded_blocks = encoded.into_code_blocks();
    budget.account_vec(&encoded_blocks)?;
    let mut encoded_blocks = encoded_blocks.into_iter();

    let ll =
        split_resident_subband_blocks(&plans[0], item_count, &mut encoded_blocks, &mut budget)?;
    let hl =
        split_resident_subband_blocks(&plans[1], item_count, &mut encoded_blocks, &mut budget)?;
    let lh =
        split_resident_subband_blocks(&plans[2], item_count, &mut encoded_blocks, &mut budget)?;
    let hh =
        split_resident_subband_blocks(&plans[3], item_count, &mut encoded_blocks, &mut budget)?;
    if encoded_blocks.next().is_some() {
        return Err(CudaTranscodeError::Kernel(
            "CUDA resident multi-input HTJ2K output count mismatch",
        ));
    }

    Ok((ll, hl, lh, hh, ht_timings, dispatches))
}

pub(in crate::cuda) fn encode_resident_compact_subbands(
    context: &CudaContext,
    resources: &CudaHtj2kEncodeResources,
    pool: &CudaBufferPool,
    bands: &CudaHtj2k97DeviceCodeblockBands,
    item_count: usize,
    options: Htj2k97CodeBlockOptions,
) -> Result<CompactResidentSubbands, CudaTranscodeError> {
    let mut budget = ResidentMetadataBudget::new("CUDA compact resident aggregate metadata");
    let plans = [
        resident_subband_encode_plan(
            &bands.ll,
            item_count,
            bands.low_width,
            bands.low_height,
            J2kSubBandType::LowLow,
            options,
            &mut budget,
        )?,
        resident_subband_encode_plan(
            &bands.hl,
            item_count,
            bands.high_width,
            bands.low_height,
            J2kSubBandType::HighLow,
            options,
            &mut budget,
        )?,
        resident_subband_encode_plan(
            &bands.lh,
            item_count,
            bands.low_width,
            bands.high_height,
            J2kSubBandType::LowHigh,
            options,
            &mut budget,
        )?,
        resident_subband_encode_plan(
            &bands.hh,
            item_count,
            bands.high_width,
            bands.high_height,
            J2kSubBandType::HighHigh,
            options,
            &mut budget,
        )?,
    ];
    let targets = resident_targets(&plans, &mut budget)?;
    let encoded = context
        .encode_htj2k_codeblocks_multi_resident_compact_with_resources_and_pool_and_live_host_bytes(
            &targets,
            resources,
            pool,
            budget.live_bytes(),
        )
        .map_err(|error| {
            CudaTranscodeError::runtime("CUDA resident compact multi-input HTJ2K encode", error)
        })?;
    let expected_blocks = checked_element_sum(
        &[
            plans[0].jobs.len(),
            plans[1].jobs.len(),
            plans[2].jobs.len(),
            plans[3].jobs.len(),
        ],
        "CUDA compact resident encoded code-block count",
    )?;
    if encoded.code_blocks().len() != expected_blocks {
        return Err(CudaTranscodeError::Kernel(
            "CUDA resident compact multi-input HTJ2K encode returned wrong block count",
        ));
    }
    let ht_timings = encoded.stage_timings();
    let dispatches = encoded.execution().kernel_dispatches();
    let (payload, encoded_blocks) = encoded.into_payload_and_code_blocks();
    budget.account_vec(&payload)?;
    budget.account_vec(&encoded_blocks)?;
    let mut encoded_blocks = encoded_blocks.into_iter();

    let ll = split_resident_compact_subband_blocks(
        &plans[0],
        item_count,
        &mut encoded_blocks,
        &mut budget,
    )?;
    let hl = split_resident_compact_subband_blocks(
        &plans[1],
        item_count,
        &mut encoded_blocks,
        &mut budget,
    )?;
    let lh = split_resident_compact_subband_blocks(
        &plans[2],
        item_count,
        &mut encoded_blocks,
        &mut budget,
    )?;
    let hh = split_resident_compact_subband_blocks(
        &plans[3],
        item_count,
        &mut encoded_blocks,
        &mut budget,
    )?;
    if encoded_blocks.next().is_some() {
        return Err(CudaTranscodeError::Kernel(
            "CUDA resident compact multi-input HTJ2K output count mismatch",
        ));
    }

    Ok((payload, ll, hl, lh, hh, ht_timings, dispatches))
}

pub(in crate::cuda) fn device_band_groups_to_preencoded_components<J: Htj2k97ComponentJob>(
    context: &CudaContext,
    resources: &CudaHtj2kEncodeResources,
    pool: &CudaBufferPool,
    groups: &[ResidentDeviceGroup<'_, J>],
    options: Htj2k97CodeBlockOptions,
    live_metadata_bytes: usize,
) -> Result<ResidentPreencodedGroups, CudaTranscodeError> {
    let (group_plans, mut budget) =
        build_resident_subband_group_plans(groups, options, live_metadata_bytes)?;
    let targets = resident_group_targets(&group_plans, &mut budget)?;
    let encoded = context
        .encode_htj2k_codeblocks_multi_resident_with_resources_and_pool_and_live_host_bytes(
            &targets,
            resources,
            pool,
            budget.live_bytes(),
        )
        .map_err(|error| {
            CudaTranscodeError::runtime("CUDA grouped resident multi-input HTJ2K encode", error)
        })?;
    let expected_blocks = resident_group_block_count(&group_plans)?;
    if encoded.code_blocks().len() != expected_blocks {
        return Err(CudaTranscodeError::Kernel(
            "CUDA grouped resident multi-input HTJ2K encode returned wrong block count",
        ));
    }
    let ht_timings = encoded.stage_timings();
    let dispatches = encoded.execution().kernel_dispatches();
    let encoded_blocks = encoded.into_code_blocks();
    budget.account_vec(&encoded_blocks)?;
    let mut encoded_blocks = encoded_blocks.into_iter();
    let mut outputs =
        budget.try_vec_with_capacity(group_plans.len(), "CUDA grouped resident encoded outputs")?;

    for group in &group_plans {
        let item_count = group.jobs.len();
        let ll =
            split_resident_subband_blocks(&group.ll, item_count, &mut encoded_blocks, &mut budget)?;
        let hl =
            split_resident_subband_blocks(&group.hl, item_count, &mut encoded_blocks, &mut budget)?;
        let lh =
            split_resident_subband_blocks(&group.lh, item_count, &mut encoded_blocks, &mut budget)?;
        let hh =
            split_resident_subband_blocks(&group.hh, item_count, &mut encoded_blocks, &mut budget)?;
        let components =
            assemble_preencoded_components_with_budget(group.jobs, ll, hl, lh, hh, &mut budget)?;
        outputs.push((group.group_index, components));
    }
    if encoded_blocks.next().is_some() {
        return Err(CudaTranscodeError::Kernel(
            "CUDA grouped resident multi-input HTJ2K output count mismatch",
        ));
    }

    Ok((outputs, ht_timings, dispatches))
}

pub(in crate::cuda) fn device_band_groups_to_compact_preencoded_components<
    J: Htj2k97ComponentJob,
>(
    context: &CudaContext,
    resources: &CudaHtj2kEncodeResources,
    pool: &CudaBufferPool,
    groups: &[ResidentDeviceGroup<'_, J>],
    options: Htj2k97CodeBlockOptions,
    live_metadata_bytes: usize,
) -> Result<ResidentCompactPreencodedGroups, CudaTranscodeError> {
    let (group_plans, mut budget) =
        build_resident_subband_group_plans(groups, options, live_metadata_bytes)?;
    let targets = resident_group_targets(&group_plans, &mut budget)?;
    let encoded = context
        .encode_htj2k_codeblocks_multi_resident_compact_with_resources_and_pool_and_live_host_bytes(
            &targets,
            resources,
            pool,
            budget.live_bytes(),
        )
        .map_err(|error| {
            CudaTranscodeError::runtime(
                "CUDA grouped resident compact multi-input HTJ2K encode",
                error,
            )
        })?;
    let expected_blocks = resident_group_block_count(&group_plans)?;
    if encoded.code_blocks().len() != expected_blocks {
        return Err(CudaTranscodeError::Kernel(
            "CUDA grouped resident compact multi-input HTJ2K encode returned wrong block count",
        ));
    }
    let ht_timings = encoded.stage_timings();
    let dispatches = encoded.execution().kernel_dispatches();
    let (payload, encoded_blocks) = encoded.into_payload_and_code_blocks();
    budget.account_vec(&payload)?;
    budget.account_vec(&encoded_blocks)?;
    let mut encoded_blocks = encoded_blocks.into_iter();
    let mut outputs = budget.try_vec_with_capacity(
        group_plans.len(),
        "CUDA grouped compact resident encoded outputs",
    )?;

    for group in &group_plans {
        let item_count = group.jobs.len();
        let ll = split_resident_compact_subband_blocks(
            &group.ll,
            item_count,
            &mut encoded_blocks,
            &mut budget,
        )?;
        let hl = split_resident_compact_subband_blocks(
            &group.hl,
            item_count,
            &mut encoded_blocks,
            &mut budget,
        )?;
        let lh = split_resident_compact_subband_blocks(
            &group.lh,
            item_count,
            &mut encoded_blocks,
            &mut budget,
        )?;
        let hh = split_resident_compact_subband_blocks(
            &group.hh,
            item_count,
            &mut encoded_blocks,
            &mut budget,
        )?;
        let components = assemble_compact_preencoded_components_with_budget(
            group.jobs,
            ll,
            hl,
            lh,
            hh,
            &mut budget,
        )?;
        outputs.push((group.group_index, components));
    }
    if encoded_blocks.next().is_some() {
        return Err(CudaTranscodeError::Kernel(
            "CUDA grouped resident compact multi-input HTJ2K output count mismatch",
        ));
    }

    Ok((payload, outputs, ht_timings, dispatches))
}
