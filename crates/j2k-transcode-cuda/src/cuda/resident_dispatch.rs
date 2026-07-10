// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    accumulate_batch_timings, add_ht_encode_timings, append_i16_blocks,
    assemble_compact_preencoded_components, assemble_preencoded_components,
    device_band_groups_to_compact_preencoded_components,
    device_band_groups_to_preencoded_components, encode_resident_compact_subbands,
    encode_resident_subbands, htj2k97_subband_delta, map_batch_timings, set_ht_encode_timings,
    transcode_kernels_built, validate_htj2k97_codeblock_options, CudaBufferPool, CudaContext,
    CudaDwt97BatchGeometry, CudaHtj2k97DeviceCodeblockBands,
    CudaHtj2k97I16CodeblockBatchWithPoolRequest, CudaHtj2k97QuantizeParams,
    CudaHtj2kEncodeResources, CudaHtj2kEncodeStageTimings, CudaTranscodeError,
    CudaTranscodeSession, DctGridI16ToHtj2k97CodeBlockBatch, DctGridI16ToHtj2k97CodeBlockJob,
    Dwt97BatchStageTimings, Htj2k97CodeBlockOptions, Htj2k97ComponentJob, J2kSubBandType,
    PreencodedHtj2k97CompactBatch, PreencodedHtj2k97CompactBatchGroups, PreencodedHtj2k97Component,
    ResidentDeviceGroup,
};

pub(super) fn dispatch_htj2k97_preencoded_i16_batch_with_sink<'a, 'j, R>(
    session: &mut CudaTranscodeSession,
    jobs: &'j [DctGridI16ToHtj2k97CodeBlockJob<'a>],
    options: Htj2k97CodeBlockOptions,
    empty: impl FnOnce() -> R,
    sink: impl FnOnce(
        &CudaContext,
        &CudaHtj2kEncodeResources,
        &CudaBufferPool,
        &CudaHtj2k97DeviceCodeblockBands,
        &'j [DctGridI16ToHtj2k97CodeBlockJob<'a>],
        Htj2k97CodeBlockOptions,
    ) -> Result<(R, CudaHtj2kEncodeStageTimings, usize), CudaTranscodeError>,
) -> Result<(R, Dwt97BatchStageTimings), CudaTranscodeError> {
    if !transcode_kernels_built() {
        return Err(CudaTranscodeError::CudaUnavailable);
    }
    validate_htj2k97_codeblock_options(options)?;
    let context = session.context()?;

    let Some(first) = jobs.first() else {
        return Ok((empty(), Dwt97BatchStageTimings::default()));
    };

    let uniform = jobs.iter().all(|job| {
        job.block_cols == first.block_cols
            && job.block_rows == first.block_rows
            && job.width == first.width
            && job.height == first.height
    });
    if !uniform {
        return Err(CudaTranscodeError::UnsupportedJob(
            "CUDA 9/7 resident HT i16 batch requires uniform job geometry",
        ));
    }

    let params = htj2k97_quantize_params(options)?;
    let mut blocks = Vec::with_capacity(jobs.len() * first.block_cols * first.block_rows * 64);
    for job in jobs {
        append_i16_blocks(job.dequantized_blocks, &mut blocks);
    }
    let pool = session.buffer_pool(&context);
    let (device_bands, cuda_timings) = context
        .j2k_transcode_htj2k97_codeblock_i16_batch_resident_with_pool(
            CudaHtj2k97I16CodeblockBatchWithPoolRequest {
                blocks: &blocks,
                geometry: CudaDwt97BatchGeometry {
                    item_count: jobs.len(),
                    block_cols: first.block_cols,
                    block_rows: first.block_rows,
                    width: first.width,
                    height: first.height,
                },
                params,
                pool: &pool,
            },
        )
        .map_err(|_| CudaTranscodeError::Kernel("CUDA 9/7 resident i16 batch dispatch failed"))?;
    let mut timings = map_batch_timings(cuda_timings);

    let resources = session.encode_resources(&context)?;
    let (output, ht_timings, ht_dispatches) = sink(
        &context,
        resources.as_ref(),
        &pool,
        &device_bands,
        jobs,
        options,
    )?;
    set_ht_encode_timings(&mut timings, ht_timings);
    timings.ht_codeblock_dispatches = ht_dispatches;
    Ok((output, timings))
}

pub(crate) fn dispatch_htj2k97_preencoded_i16_batch(
    session: &mut CudaTranscodeSession,
    jobs: &[DctGridI16ToHtj2k97CodeBlockJob<'_>],
    options: Htj2k97CodeBlockOptions,
) -> Result<(Vec<PreencodedHtj2k97Component>, Dwt97BatchStageTimings), CudaTranscodeError> {
    dispatch_htj2k97_preencoded_i16_batch_with_sink(
        session,
        jobs,
        options,
        Vec::new,
        |context, resources, pool, bands, jobs, options| {
            device_bands_to_preencoded_components(context, resources, pool, bands, jobs, options)
        },
    )
}

pub(crate) fn dispatch_htj2k97_compact_preencoded_i16_batch(
    session: &mut CudaTranscodeSession,
    jobs: &[DctGridI16ToHtj2k97CodeBlockJob<'_>],
    options: Htj2k97CodeBlockOptions,
) -> Result<(PreencodedHtj2k97CompactBatch, Dwt97BatchStageTimings), CudaTranscodeError> {
    dispatch_htj2k97_preencoded_i16_batch_with_sink(
        session,
        jobs,
        options,
        || PreencodedHtj2k97CompactBatch {
            payload: Vec::new(),
            components: Vec::new(),
        },
        |context, resources, pool, bands, jobs, options| {
            device_bands_to_compact_preencoded_batch(context, resources, pool, bands, jobs, options)
        },
    )
}

#[allow(clippy::type_complexity)]
pub(super) fn dispatch_htj2k97_preencoded_i16_batch_groups_with_sink<'a, 'g, 'j, C, X: Default>(
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
    ) -> Result<
        (X, Vec<(usize, Vec<C>)>, CudaHtj2kEncodeStageTimings, usize),
        CudaTranscodeError,
    >,
) -> Result<(X, Vec<Vec<C>>, Dwt97BatchStageTimings), CudaTranscodeError> {
    if !transcode_kernels_built() {
        return Err(CudaTranscodeError::CudaUnavailable);
    }
    validate_htj2k97_codeblock_options(options)?;
    let context = session.context()?;
    let params = htj2k97_quantize_params(options)?;
    let pool = session.buffer_pool(&context);
    let mut timings = Dwt97BatchStageTimings::default();
    let mut outputs = std::iter::repeat_with(|| None)
        .take(groups.len())
        .collect::<Vec<Option<Vec<C>>>>();
    let mut device_groups = Vec::new();

    for (group_index, group) in groups.iter().enumerate() {
        let Some(first) = group.jobs.first() else {
            outputs[group_index] = Some(Vec::new());
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

        let mut blocks =
            Vec::with_capacity(group.jobs.len() * first.block_cols * first.block_rows * 64);
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
                    pool: &pool,
                },
            )
            .map_err(|_| {
                CudaTranscodeError::Kernel("CUDA grouped 9/7 resident i16 batch dispatch failed")
            })?;
        accumulate_batch_timings(&mut timings, map_batch_timings(group_timings));
        device_groups.push(ResidentDeviceGroup {
            group_index,
            bands,
            jobs: group.jobs,
        });
    }

    let mut extra = X::default();
    if !device_groups.is_empty() {
        let resources = session.encode_resources(&context)?;
        let (sink_extra, encoded_groups, ht_timings, ht_dispatches) =
            sink(&context, resources.as_ref(), &pool, &device_groups, options)?;
        extra = sink_extra;
        add_ht_encode_timings(&mut timings, ht_timings);
        timings.ht_codeblock_dispatches = timings
            .ht_codeblock_dispatches
            .saturating_add(ht_dispatches);
        for (group_index, components) in encoded_groups {
            outputs[group_index] = Some(components);
        }
    }

    let outputs = outputs
        .into_iter()
        .map(|components| components.ok_or(CudaTranscodeError::Kernel(missing_group_error)))
        .collect::<Result<Vec<_>, _>>()?;

    Ok((extra, outputs, timings))
}

pub(crate) fn dispatch_htj2k97_preencoded_i16_batch_groups(
    session: &mut CudaTranscodeSession,
    groups: &[DctGridI16ToHtj2k97CodeBlockBatch<'_, '_>],
    options: Htj2k97CodeBlockOptions,
) -> Result<(Vec<Vec<PreencodedHtj2k97Component>>, Dwt97BatchStageTimings), CudaTranscodeError> {
    let ((), outputs, timings) = dispatch_htj2k97_preencoded_i16_batch_groups_with_sink(
        session,
        groups,
        options,
        "CUDA grouped 9/7 resident HT output group missing",
        |context, resources, pool, device_groups, options| {
            let (encoded_groups, ht_timings, ht_dispatches) =
                device_band_groups_to_preencoded_components(
                    context,
                    resources,
                    pool,
                    device_groups,
                    options,
                )?;
            Ok(((), encoded_groups, ht_timings, ht_dispatches))
        },
    )?;
    Ok((outputs, timings))
}

pub(crate) fn dispatch_htj2k97_compact_preencoded_i16_batch_groups(
    session: &mut CudaTranscodeSession,
    groups: &[DctGridI16ToHtj2k97CodeBlockBatch<'_, '_>],
    options: Htj2k97CodeBlockOptions,
) -> Result<(PreencodedHtj2k97CompactBatchGroups, Dwt97BatchStageTimings), CudaTranscodeError> {
    let (payload, groups, timings) = dispatch_htj2k97_preencoded_i16_batch_groups_with_sink(
        session,
        groups,
        options,
        "CUDA grouped 9/7 resident compact HT output group missing",
        |context, resources, pool, device_groups, options| {
            device_band_groups_to_compact_preencoded_components(
                context,
                resources,
                pool,
                device_groups,
                options,
            )
        },
    )?;
    Ok((
        PreencodedHtj2k97CompactBatchGroups { payload, groups },
        timings,
    ))
}

pub(super) fn htj2k97_quantize_params(
    options: Htj2k97CodeBlockOptions,
) -> Result<CudaHtj2k97QuantizeParams, CudaTranscodeError> {
    let (cb_width, cb_height) = validate_htj2k97_codeblock_options(options)?;
    let inv_delta =
        |sub: J2kSubBandType| -> f32 { (1.0 / htj2k97_subband_delta(options, sub)) as f32 };
    Ok(CudaHtj2k97QuantizeParams {
        inv_delta_ll: inv_delta(J2kSubBandType::LowLow),
        inv_delta_hl: inv_delta(J2kSubBandType::HighLow),
        inv_delta_lh: inv_delta(J2kSubBandType::LowHigh),
        inv_delta_hh: inv_delta(J2kSubBandType::HighHigh),
        cb_width,
        cb_height,
    })
}

#[allow(clippy::similar_names)]
pub(super) fn device_bands_to_preencoded_components<J: Htj2k97ComponentJob>(
    context: &CudaContext,
    resources: &CudaHtj2kEncodeResources,
    pool: &CudaBufferPool,
    bands: &CudaHtj2k97DeviceCodeblockBands,
    jobs: &[J],
    options: Htj2k97CodeBlockOptions,
) -> Result<
    (
        Vec<PreencodedHtj2k97Component>,
        CudaHtj2kEncodeStageTimings,
        usize,
    ),
    CudaTranscodeError,
> {
    if bands.item_count != jobs.len() {
        return Err(CudaTranscodeError::Kernel(
            "CUDA resident 9/7 band item count mismatch",
        ));
    }

    let (ll_subbands, hl_subbands, lh_subbands, hh_subbands, ht_timings, dispatches) =
        encode_resident_subbands(context, resources, pool, bands, bands.item_count, options)?;

    let components =
        assemble_preencoded_components(jobs, ll_subbands, hl_subbands, lh_subbands, hh_subbands)?;

    Ok((components, ht_timings, dispatches))
}

#[allow(clippy::similar_names)]
pub(super) fn device_bands_to_compact_preencoded_batch<J: Htj2k97ComponentJob>(
    context: &CudaContext,
    resources: &CudaHtj2kEncodeResources,
    pool: &CudaBufferPool,
    bands: &CudaHtj2k97DeviceCodeblockBands,
    jobs: &[J],
    options: Htj2k97CodeBlockOptions,
) -> Result<
    (
        PreencodedHtj2k97CompactBatch,
        CudaHtj2kEncodeStageTimings,
        usize,
    ),
    CudaTranscodeError,
> {
    if bands.item_count != jobs.len() {
        return Err(CudaTranscodeError::Kernel(
            "CUDA resident 9/7 band item count mismatch",
        ));
    }

    let (payload, ll_subbands, hl_subbands, lh_subbands, hh_subbands, ht_timings, dispatches) =
        encode_resident_compact_subbands(
            context,
            resources,
            pool,
            bands,
            bands.item_count,
            options,
        )?;

    let components = assemble_compact_preencoded_components(
        jobs,
        ll_subbands,
        hl_subbands,
        lh_subbands,
        hh_subbands,
    )?;

    Ok((
        PreencodedHtj2k97CompactBatch {
            payload,
            components,
        },
        ht_timings,
        dispatches,
    ))
}
