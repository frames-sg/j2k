// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::{
    accumulate_batch_timings, add_ht_encode_timings, append_i16_blocks, checked_element_product,
    device_band_groups_to_compact_preencoded_components,
    device_band_groups_to_preencoded_components, map_batch_timings, transcode_kernels_built,
    validate_htj2k97_codeblock_options, CudaBufferPool, CudaContext, CudaDwt97BatchGeometry,
    CudaHtj2k97I16CodeblockBatchWithPoolRequest, CudaHtj2k97QuantizeParams,
    CudaHtj2kEncodeResources, CudaHtj2kEncodeStageTimings, CudaTranscodeError,
    CudaTranscodeSession, DctGridI16ToHtj2k97CodeBlockBatch, DctGridI16ToHtj2k97CodeBlockJob,
    Dwt97BatchStageTimings, HostPhaseBudget, Htj2k97CodeBlockOptions,
    PreencodedHtj2k97CompactBatchGroups, PreencodedHtj2k97Component, ResidentDeviceGroup,
};
use super::{htj2k97_quantize_params, validate_i16_block_grid};

type EncodedGroups<C> = Vec<(usize, Vec<C>, Vec<u8>)>;
type GroupedSinkOutput<X, C> = (X, EncodedGroups<C>, CudaHtj2kEncodeStageTimings, usize);
type GroupedDispatchOutput<X, C> = (X, Vec<Vec<C>>, Vec<u8>, Dwt97BatchStageTimings);
type PreencodedGroupDispatchOutput = (
    Vec<Vec<PreencodedHtj2k97Component>>,
    Vec<u8>,
    Dwt97BatchStageTimings,
);
type CompactPreencodedGroupDispatchOutput = (
    PreencodedHtj2k97CompactBatchGroups,
    Vec<u8>,
    Dwt97BatchStageTimings,
);

fn flatten_required_magnitude_bounds(
    magnitude_bounds: Vec<Vec<u8>>,
    budget: &mut HostPhaseBudget,
) -> Result<Vec<u8>, CudaTranscodeError> {
    let count = magnitude_bounds
        .iter()
        .try_fold(0usize, |count, bounds| count.checked_add(bounds.len()))
        .ok_or(CudaTranscodeError::Kernel(
            "CUDA grouped 9/7 resident HT magnitude-bound count overflow",
        ))?;
    let mut flattened =
        budget.try_vec_with_capacity(count, "CUDA grouped resident magnitude-bound outputs")?;
    for bounds in magnitude_bounds {
        flattened.extend(bounds);
    }
    Ok(flattened)
}

fn store_encoded_groups<C>(
    encoded_groups: EncodedGroups<C>,
    outputs: &mut [Vec<C>],
    magnitude_bounds: &mut [Vec<u8>],
    output_present: &mut [bool],
    budget: &mut HostPhaseBudget,
) -> Result<(), CudaTranscodeError> {
    for (group_index, components, required_bounds) in encoded_groups {
        if components.len() != required_bounds.len() {
            return Err(CudaTranscodeError::Kernel(
                "CUDA grouped 9/7 resident HT magnitude-bound count mismatch",
            ));
        }
        budget.account_vec(&required_bounds)?;
        let Some(output) = outputs.get_mut(group_index) else {
            return Err(CudaTranscodeError::Kernel(
                "CUDA grouped 9/7 resident HT output group index is out of range",
            ));
        };
        let Some(output_bounds) = magnitude_bounds.get_mut(group_index) else {
            return Err(CudaTranscodeError::Kernel(
                "CUDA grouped 9/7 resident HT magnitude-bound group index is out of range",
            ));
        };
        let Some(present) = output_present.get_mut(group_index) else {
            return Err(CudaTranscodeError::Kernel(
                "CUDA grouped 9/7 resident HT output presence index is out of range",
            ));
        };
        if *present {
            return Err(CudaTranscodeError::Kernel(
                "CUDA grouped 9/7 resident HT output group was returned more than once",
            ));
        }
        *output = components;
        *output_bounds = required_bounds;
        *present = true;
    }
    Ok(())
}

fn live_staging_budget<T>(
    live_metadata_bytes: usize,
    staging_element_count: usize,
) -> Result<HostPhaseBudget, CudaTranscodeError> {
    let budget = HostPhaseBudget::with_live_bytes(
        "CUDA grouped resident dispatch and staging",
        live_metadata_bytes,
    )?;
    budget.preflight_capacity::<T>(staging_element_count)?;
    Ok(budget)
}

struct ResidentGroupStaging<'j, 'a> {
    output_present: Vec<bool>,
    device_groups: Vec<ResidentDeviceGroup<'j, DctGridI16ToHtj2k97CodeBlockJob<'a>>>,
    timings: Dwt97BatchStageTimings,
}

fn stage_resident_device_groups<'a, 'j>(
    context: &CudaContext,
    pool: &CudaBufferPool,
    groups: &[DctGridI16ToHtj2k97CodeBlockBatch<'a, 'j>],
    params: CudaHtj2k97QuantizeParams,
    live_metadata_bytes: usize,
    staging: &mut ResidentGroupStaging<'j, 'a>,
) -> Result<(), CudaTranscodeError> {
    for (group_index, group) in groups.iter().enumerate() {
        let Some(first) = group.jobs.first() else {
            staging.output_present[group_index] = true;
            continue;
        };
        let uniform = group.jobs.iter().all(|job| {
            job.block_cols == first.block_cols
                && job.block_rows == first.block_rows
                && job.width == first.width
                && job.height == first.height
        });
        if !uniform {
            return Err(CudaTranscodeError::UnsupportedJob(
                "CUDA grouped 9/7 resident HT i16 batches require uniform geometry inside each group",
            ));
        }

        for job in group.jobs {
            validate_i16_block_grid(job)?;
        }
        let staging_element_count = checked_element_product(
            &[group.jobs.len(), first.block_cols, first.block_rows, 64],
            "CUDA grouped resident i16 DCT staging",
        )?;
        let mut staging_budget =
            live_staging_budget::<i16>(live_metadata_bytes, staging_element_count)?;
        let mut blocks = staging_budget.try_vec_for_product::<i16>(
            &[group.jobs.len(), first.block_cols, first.block_rows, 64],
            "CUDA grouped resident i16 DCT staging",
        )?;
        for job in group.jobs {
            append_i16_blocks(job.dequantized_blocks, &mut blocks);
        }
        let (bands, group_timings) = context
            .j2k_transcode_htj2k97_codeblock_i16_batch_resident_with_pool(
                CudaHtj2k97I16CodeblockBatchWithPoolRequest {
                    blocks: &blocks,
                    geometry: CudaDwt97BatchGeometry {
                        item_count: group.jobs.len(),
                        block_cols: first.block_cols,
                        block_rows: first.block_rows,
                        width: first.width,
                        height: first.height,
                    },
                    params,
                    pool,
                },
            )
            .map_err(|error| {
                CudaTranscodeError::runtime("CUDA grouped 9/7 resident i16 batch dispatch", error)
            })?;
        drop(blocks);
        accumulate_batch_timings(&mut staging.timings, map_batch_timings(group_timings));
        staging.device_groups.push(ResidentDeviceGroup {
            group_index,
            bands,
            jobs: group.jobs,
        });
    }
    Ok(())
}

fn dispatch_with_sink<'a, 'g, 'j, C, X: Default>(
    session: &mut CudaTranscodeSession,
    groups: &'g [DctGridI16ToHtj2k97CodeBlockBatch<'a, 'j>],
    options: Htj2k97CodeBlockOptions,
    missing_group_error: &'static str,
    sink: impl FnOnce(
        &CudaContext,
        &CudaHtj2kEncodeResources,
        &CudaBufferPool,
        &[ResidentDeviceGroup<'j, DctGridI16ToHtj2k97CodeBlockJob<'a>>],
        Htj2k97CodeBlockOptions,
        usize,
    ) -> Result<GroupedSinkOutput<X, C>, CudaTranscodeError>,
) -> Result<GroupedDispatchOutput<X, C>, CudaTranscodeError> {
    if !transcode_kernels_built() {
        return Err(CudaTranscodeError::CudaUnavailable);
    }
    validate_htj2k97_codeblock_options(options)?;
    let context = session.context()?;
    let params = htj2k97_quantize_params(options)?;
    let pool = session.buffer_pool(&context);
    let mut host_budget = HostPhaseBudget::new("CUDA grouped resident dispatch metadata");
    let mut outputs = host_budget
        .try_vec_with_capacity::<Vec<C>>(groups.len(), "CUDA grouped resident output slots")?;
    outputs.resize_with(groups.len(), Vec::new);
    let mut magnitude_bounds = host_budget.try_vec_with_capacity::<Vec<u8>>(
        groups.len(),
        "CUDA grouped resident magnitude-bound slots",
    )?;
    magnitude_bounds.resize_with(groups.len(), Vec::new);
    let mut output_present = host_budget
        .try_vec_with_capacity::<bool>(groups.len(), "CUDA grouped resident output presence")?;
    output_present.resize(groups.len(), false);
    let device_groups = host_budget
        .try_vec_with_capacity(groups.len(), "CUDA grouped resident device-band metadata")?;
    let live_metadata_bytes = host_budget.live_bytes();
    let mut staging = ResidentGroupStaging {
        output_present,
        device_groups,
        timings: Dwt97BatchStageTimings::default(),
    };

    stage_resident_device_groups(
        &context,
        &pool,
        groups,
        params,
        live_metadata_bytes,
        &mut staging,
    )?;

    let mut extra = X::default();
    if !staging.device_groups.is_empty() {
        let resources = session.encode_resources(&context)?;
        let (sink_extra, encoded_groups, ht_timings, ht_dispatches) = sink(
            &context,
            resources.as_ref(),
            &pool,
            &staging.device_groups,
            options,
            live_metadata_bytes,
        )?;
        extra = sink_extra;
        add_ht_encode_timings(&mut staging.timings, ht_timings);
        staging.timings.ht_codeblock_dispatches = staging
            .timings
            .ht_codeblock_dispatches
            .saturating_add(ht_dispatches);
        store_encoded_groups(
            encoded_groups,
            &mut outputs,
            &mut magnitude_bounds,
            &mut staging.output_present,
            &mut host_budget,
        )?;
    }

    if staging.output_present.iter().any(|present| !present) {
        return Err(CudaTranscodeError::Kernel(missing_group_error));
    }

    let flattened_bounds = flatten_required_magnitude_bounds(magnitude_bounds, &mut host_budget)?;

    Ok((extra, outputs, flattened_bounds, staging.timings))
}

pub(crate) fn dispatch_htj2k97_preencoded_i16_batch_groups(
    session: &mut CudaTranscodeSession,
    groups: &[DctGridI16ToHtj2k97CodeBlockBatch<'_, '_>],
    options: Htj2k97CodeBlockOptions,
) -> Result<PreencodedGroupDispatchOutput, CudaTranscodeError> {
    let ((), outputs, required_magnitude_bounds, timings) = dispatch_with_sink(
        session,
        groups,
        options,
        "CUDA grouped 9/7 resident HT output group missing",
        |context, resources, pool, device_groups, options, live_metadata_bytes| {
            let (encoded_groups, ht_timings, ht_dispatches) =
                device_band_groups_to_preencoded_components(
                    context,
                    resources,
                    pool,
                    device_groups,
                    options,
                    live_metadata_bytes,
                )?;
            Ok(((), encoded_groups, ht_timings, ht_dispatches))
        },
    )?;
    Ok((outputs, required_magnitude_bounds, timings))
}

pub(crate) fn dispatch_htj2k97_compact_preencoded_i16_batch_groups(
    session: &mut CudaTranscodeSession,
    groups: &[DctGridI16ToHtj2k97CodeBlockBatch<'_, '_>],
    options: Htj2k97CodeBlockOptions,
) -> Result<CompactPreencodedGroupDispatchOutput, CudaTranscodeError> {
    let (payload, groups, required_magnitude_bounds, timings) = dispatch_with_sink(
        session,
        groups,
        options,
        "CUDA grouped 9/7 resident compact HT output group missing",
        |context, resources, pool, device_groups, options, live_metadata_bytes| {
            device_band_groups_to_compact_preencoded_components(
                context,
                resources,
                pool,
                device_groups,
                options,
                live_metadata_bytes,
            )
        },
    )?;
    Ok((
        PreencodedHtj2k97CompactBatchGroups { payload, groups },
        required_magnitude_bounds,
        timings,
    ))
}

#[cfg(test)]
mod tests {
    use super::live_staging_budget;
    use crate::CudaTranscodeError;
    use j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES;

    #[test]
    fn live_dispatch_metadata_is_counted_with_staging() {
        let half_plus_one = DEFAULT_MAX_HOST_ALLOCATION_BYTES / 2 + 1;
        assert!(matches!(
            live_staging_budget::<u8>(half_plus_one, half_plus_one),
            Err(CudaTranscodeError::HostAllocationTooLarge {
                what: "CUDA grouped resident dispatch and staging",
                ..
            })
        ));
    }
}
