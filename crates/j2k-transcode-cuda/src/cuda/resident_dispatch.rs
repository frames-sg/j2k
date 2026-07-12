// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    append_i16_blocks, assemble_compact_preencoded_components, assemble_preencoded_components,
    checked_element_product, encode_resident_compact_subbands, encode_resident_subbands,
    htj2k97_subband_delta, map_batch_timings, set_ht_encode_timings, transcode_kernels_built,
    try_transcode_vec_for_product, validate_htj2k97_codeblock_options, CudaBufferPool, CudaContext,
    CudaDwt97BatchGeometry, CudaHtj2k97DeviceCodeblockBands,
    CudaHtj2k97I16CodeblockBatchWithPoolRequest, CudaHtj2k97QuantizeParams,
    CudaHtj2kEncodeResources, CudaHtj2kEncodeStageTimings, CudaTranscodeError,
    CudaTranscodeSession, DctGridI16ToHtj2k97CodeBlockJob, Dwt97BatchStageTimings,
    Htj2k97CodeBlockOptions, Htj2k97ComponentJob, J2kSubBandType, PreencodedHtj2k97CompactBatch,
    PreencodedHtj2k97Component,
};
mod grouped;
pub(crate) use self::grouped::{
    dispatch_htj2k97_compact_preencoded_i16_batch_groups,
    dispatch_htj2k97_preencoded_i16_batch_groups,
};

fn validate_i16_block_grid(
    job: &DctGridI16ToHtj2k97CodeBlockJob<'_>,
) -> Result<(), CudaTranscodeError> {
    let expected = checked_element_product(
        &[job.block_cols, job.block_rows],
        "CUDA resident i16 DCT block grid",
    )?;
    if job.dequantized_blocks.len() != expected {
        return Err(CudaTranscodeError::UnsupportedJob(
            "CUDA resident i16 DCT block slice does not match its grid",
        ));
    }
    Ok(())
}

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
    for job in jobs {
        validate_i16_block_grid(job)?;
    }
    let mut blocks = try_transcode_vec_for_product::<i16>(
        &[jobs.len(), first.block_cols, first.block_rows, 64],
        "CUDA resident i16 batch DCT staging",
    )?;
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
        .map_err(|error| {
            CudaTranscodeError::runtime("CUDA 9/7 resident i16 batch dispatch", error)
        })?;
    drop(blocks);
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

#[expect(
    clippy::cast_possible_truncation,
    reason = "the CUDA quantization ABI intentionally consumes f32 inverse deltas"
)]
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

#[expect(
    clippy::similar_names,
    reason = "LL, HL, LH, and HH are standard wavelet subband names"
)]
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

#[expect(
    clippy::similar_names,
    reason = "LL, HL, LH, and HH are standard wavelet subband names"
)]
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
        &payload,
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
