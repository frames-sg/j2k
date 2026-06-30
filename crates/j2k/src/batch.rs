// SPDX-License-Identifier: MIT OR Apache-2.0

use core::convert::Infallible;
use std::num::NonZeroUsize;

pub use j2k_core::TileBatchOptions;
use j2k_core::{
    collect_indexed_batch_results, tile_batch_worker_count, CompressedTransferSyntax,
    DecodeOutcome, DecoderContext, Downscale, IndexedBatchResult, PixelFormat, Rect,
    TileBatchDecode,
};
use j2k_native::{
    execute_direct_color_plan_rgb8_into, execute_direct_color_plan_rgba8_into,
    inspect_j2k_codestream_header, DecodeError as NativeDecodeError,
    DecodingError as NativeDecodingError, J2kDirectColorPlan, J2kDirectCpuScratch, J2kRect,
};

use crate::backend::{self, DecodeSettings};
use crate::decode::{validate_buffer, validate_region};
use crate::parse::parse_image_info;
use crate::{CpuDecodeParallelism, J2kCodec, J2kContext, J2kError, J2kScratchPool};

/// One full-tile decode request for [`decode_tiles_into`].
pub type TileDecodeJob<'i, 'o> = j2k_core::TileDecodeJob<'i, 'o>;

/// One ROI tile decode request for [`decode_tiles_region_into`].
pub type TileRegionDecodeJob<'i, 'o> = j2k_core::TileRegionDecodeJob<'i, 'o>;

/// One scaled tile decode request for [`decode_tiles_scaled_into`].
pub type TileScaledDecodeJob<'i, 'o> = j2k_core::TileScaledDecodeJob<'i, 'o>;

/// One ROI+scaled tile decode request for [`decode_tiles_region_scaled_into`].
pub type TileRegionScaledDecodeJob<'i, 'o> = j2k_core::TileRegionScaledDecodeJob<'i, 'o>;

/// Error returned by J2K CPU tile batches, annotated with the first failing
/// tile index from the caller's input order.
pub type TileBatchError = j2k_core::TileBatchError<J2kError>;

type BatchOutcome = DecodeOutcome<Infallible>;
type J2kIndexedBatchResult = IndexedBatchResult<BatchOutcome, J2kError>;

/// One-shot parse-plus-decode of an independent J2K/HTJ2K tile into the
/// caller's buffer, reusing both caller-owned [`DecoderContext`] and
/// caller-owned [`J2kScratchPool`].
pub fn decode_tile_into_in_context(
    bytes: &[u8],
    ctx: &mut DecoderContext<J2kContext>,
    pool: &mut J2kScratchPool,
    out: &mut [u8],
    stride: usize,
    fmt: PixelFormat,
) -> Result<BatchOutcome, J2kError> {
    <J2kCodec as TileBatchDecode>::decode_tile(ctx, pool, bytes, out, stride, fmt)
}

/// One-shot parse-plus-ROI-decode of an independent J2K/HTJ2K tile into the
/// caller's buffer, reusing both caller-owned [`DecoderContext`] and
/// caller-owned [`J2kScratchPool`].
pub fn decode_tile_region_into_in_context(
    bytes: &[u8],
    ctx: &mut DecoderContext<J2kContext>,
    pool: &mut J2kScratchPool,
    out: &mut [u8],
    stride: usize,
    fmt: PixelFormat,
    roi: Rect,
) -> Result<BatchOutcome, J2kError> {
    <J2kCodec as TileBatchDecode>::decode_tile_region(ctx, pool, bytes, out, stride, fmt, roi)
}

/// One-shot parse-plus-scaled-decode of an independent J2K/HTJ2K tile into the
/// caller's buffer, reusing both caller-owned [`DecoderContext`] and
/// caller-owned [`J2kScratchPool`].
pub fn decode_tile_scaled_into_in_context(
    bytes: &[u8],
    ctx: &mut DecoderContext<J2kContext>,
    pool: &mut J2kScratchPool,
    out: &mut [u8],
    stride: usize,
    fmt: PixelFormat,
    scale: Downscale,
) -> Result<BatchOutcome, J2kError> {
    <J2kCodec as TileBatchDecode>::decode_tile_scaled(ctx, pool, bytes, out, stride, fmt, scale)
}

/// One-shot parse-plus-ROI-scaled-decode of an independent J2K/HTJ2K tile
/// into the caller's buffer, reusing both caller-owned [`DecoderContext`] and
/// caller-owned [`J2kScratchPool`].
#[allow(clippy::too_many_arguments)]
pub fn decode_tile_region_scaled_into_in_context(
    bytes: &[u8],
    ctx: &mut DecoderContext<J2kContext>,
    pool: &mut J2kScratchPool,
    out: &mut [u8],
    stride: usize,
    fmt: PixelFormat,
    roi: Rect,
    scale: Downscale,
) -> Result<BatchOutcome, J2kError> {
    <J2kCodec as TileBatchDecode>::decode_tile_region_scaled(
        ctx, pool, bytes, out, stride, fmt, roi, scale,
    )
}

/// Decode independent J2K/HTJ2K tiles into caller-owned output buffers using
/// a scoped CPU worker pool.
///
/// Each worker owns one [`DecoderContext`] and one [`J2kScratchPool`]. Returned
/// outcomes preserve caller input order.
pub fn decode_tiles_into(
    jobs: &mut [TileDecodeJob<'_, '_>],
    fmt: PixelFormat,
    options: TileBatchOptions,
) -> Result<Vec<BatchOutcome>, TileBatchError> {
    if jobs.is_empty() {
        return Ok(Vec::new());
    }

    let job_count = jobs.len();
    let worker_count = tile_batch_worker_count(job_count, options, available_tile_batch_workers());
    let chunk_size = job_count.div_ceil(worker_count);
    let results =
        std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(worker_count);
            for (chunk_index, chunk) in jobs.chunks_mut(chunk_size).enumerate() {
                let start_index = chunk_index * chunk_size;
                let inner_parallelism = inner_parallelism_for_batch(job_count);
                handles.push(scope.spawn(move || {
                    decode_tile_job_chunk(start_index, chunk, fmt, inner_parallelism)
                }));
            }

            let mut results = Vec::with_capacity(job_count);
            for handle in handles {
                match handle.join() {
                    Ok(chunk_results) => results.extend(chunk_results),
                    Err(payload) => std::panic::resume_unwind(payload),
                }
            }
            results
        });

    collect_indexed_batch_results(job_count, results, |index, source| TileBatchError {
        index,
        source,
    })
}

/// Decode independent J2K/HTJ2K tile regions into caller-owned output buffers
/// using a scoped CPU worker pool.
pub fn decode_tiles_region_into(
    jobs: &mut [TileRegionDecodeJob<'_, '_>],
    fmt: PixelFormat,
    options: TileBatchOptions,
) -> Result<Vec<BatchOutcome>, TileBatchError> {
    if jobs.is_empty() {
        return Ok(Vec::new());
    }

    let job_count = jobs.len();
    let worker_count = tile_batch_worker_count(job_count, options, available_tile_batch_workers());
    let chunk_size = job_count.div_ceil(worker_count);
    let results = std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(worker_count);
        for (chunk_index, chunk) in jobs.chunks_mut(chunk_size).enumerate() {
            let start_index = chunk_index * chunk_size;
            let inner_parallelism = inner_parallelism_for_batch(job_count);
            handles.push(scope.spawn(move || {
                decode_tile_region_job_chunk(start_index, chunk, fmt, inner_parallelism)
            }));
        }

        let mut results = Vec::with_capacity(job_count);
        for handle in handles {
            match handle.join() {
                Ok(chunk_results) => results.extend(chunk_results),
                Err(payload) => std::panic::resume_unwind(payload),
            }
        }
        results
    });

    collect_indexed_batch_results(job_count, results, |index, source| TileBatchError {
        index,
        source,
    })
}

/// Decode independent J2K/HTJ2K tiles at reduced resolution into caller-owned
/// output buffers using a scoped CPU worker pool.
pub fn decode_tiles_scaled_into(
    jobs: &mut [TileScaledDecodeJob<'_, '_>],
    fmt: PixelFormat,
    options: TileBatchOptions,
) -> Result<Vec<BatchOutcome>, TileBatchError> {
    if jobs.is_empty() {
        return Ok(Vec::new());
    }

    let job_count = jobs.len();
    let worker_count = tile_batch_worker_count(job_count, options, available_tile_batch_workers());
    let chunk_size = job_count.div_ceil(worker_count);
    let results = std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(worker_count);
        for (chunk_index, chunk) in jobs.chunks_mut(chunk_size).enumerate() {
            let start_index = chunk_index * chunk_size;
            let inner_parallelism = inner_parallelism_for_batch(job_count);
            handles.push(scope.spawn(move || {
                decode_tile_scaled_job_chunk(start_index, chunk, fmt, inner_parallelism)
            }));
        }

        let mut results = Vec::with_capacity(job_count);
        for handle in handles {
            match handle.join() {
                Ok(chunk_results) => results.extend(chunk_results),
                Err(payload) => std::panic::resume_unwind(payload),
            }
        }
        results
    });

    collect_indexed_batch_results(job_count, results, |index, source| TileBatchError {
        index,
        source,
    })
}

/// Decode independent J2K/HTJ2K tile regions at reduced resolution into
/// caller-owned output buffers using a scoped CPU worker pool.
pub fn decode_tiles_region_scaled_into(
    jobs: &mut [TileRegionScaledDecodeJob<'_, '_>],
    fmt: PixelFormat,
    options: TileBatchOptions,
) -> Result<Vec<BatchOutcome>, TileBatchError> {
    if jobs.is_empty() {
        return Ok(Vec::new());
    }

    let job_count = jobs.len();
    let worker_count = tile_batch_worker_count(job_count, options, available_tile_batch_workers());
    let chunk_size = job_count.div_ceil(worker_count);
    let shared_direct_plan = build_repeated_direct_color_region_plan(jobs, fmt)
        .map_err(|source| TileBatchError { index: 0, source })?;
    let results = std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(worker_count);
        for (chunk_index, chunk) in jobs.chunks_mut(chunk_size).enumerate() {
            let start_index = chunk_index * chunk_size;
            let shared_direct_plan = shared_direct_plan.as_ref();
            handles.push(scope.spawn(move || {
                decode_tile_region_scaled_job_chunk(
                    start_index,
                    chunk,
                    fmt,
                    inner_parallelism_for_batch(job_count),
                    shared_direct_plan,
                )
            }));
        }

        let mut results = Vec::with_capacity(job_count);
        for handle in handles {
            match handle.join() {
                Ok(chunk_results) => results.extend(chunk_results),
                Err(payload) => std::panic::resume_unwind(payload),
            }
        }
        results
    });

    collect_indexed_batch_results(job_count, results, |index, source| TileBatchError {
        index,
        source,
    })
}

fn available_tile_batch_workers() -> usize {
    std::thread::available_parallelism().map_or(1, NonZeroUsize::get)
}

fn inner_parallelism_for_batch(batch_size: usize) -> CpuDecodeParallelism {
    if batch_size > 1 {
        CpuDecodeParallelism::Serial
    } else {
        CpuDecodeParallelism::Auto
    }
}

fn decode_tile_job_chunk(
    start_index: usize,
    jobs: &mut [TileDecodeJob<'_, '_>],
    fmt: PixelFormat,
    inner_parallelism: CpuDecodeParallelism,
) -> Vec<J2kIndexedBatchResult> {
    let mut ctx = DecoderContext::<J2kContext>::new();
    ctx.codec_mut()
        .set_cpu_decode_parallelism(inner_parallelism);
    let mut pool = J2kScratchPool::new();
    let mut results = Vec::with_capacity(jobs.len());
    for (local_index, job) in jobs.iter_mut().enumerate() {
        let outcome =
            decode_tile_into_in_context(job.input, &mut ctx, &mut pool, job.out, job.stride, fmt);
        results.push((start_index + local_index, outcome));
    }
    results
}

fn decode_tile_region_job_chunk(
    start_index: usize,
    jobs: &mut [TileRegionDecodeJob<'_, '_>],
    fmt: PixelFormat,
    inner_parallelism: CpuDecodeParallelism,
) -> Vec<J2kIndexedBatchResult> {
    let mut ctx = DecoderContext::<J2kContext>::new();
    ctx.codec_mut()
        .set_cpu_decode_parallelism(inner_parallelism);
    let mut pool = J2kScratchPool::new();
    let mut results = Vec::with_capacity(jobs.len());
    for (local_index, job) in jobs.iter_mut().enumerate() {
        let outcome = decode_tile_region_into_in_context(
            job.input, &mut ctx, &mut pool, job.out, job.stride, fmt, job.roi,
        );
        results.push((start_index + local_index, outcome));
    }
    results
}

fn decode_tile_scaled_job_chunk(
    start_index: usize,
    jobs: &mut [TileScaledDecodeJob<'_, '_>],
    fmt: PixelFormat,
    inner_parallelism: CpuDecodeParallelism,
) -> Vec<J2kIndexedBatchResult> {
    let mut ctx = DecoderContext::<J2kContext>::new();
    ctx.codec_mut()
        .set_cpu_decode_parallelism(inner_parallelism);
    let mut pool = J2kScratchPool::new();
    let mut results = Vec::with_capacity(jobs.len());
    for (local_index, job) in jobs.iter_mut().enumerate() {
        let outcome = decode_tile_scaled_into_in_context(
            job.input, &mut ctx, &mut pool, job.out, job.stride, fmt, job.scale,
        );
        results.push((start_index + local_index, outcome));
    }
    results
}

fn decode_tile_region_scaled_job_chunk(
    start_index: usize,
    jobs: &mut [TileRegionScaledDecodeJob<'_, '_>],
    fmt: PixelFormat,
    inner_parallelism: CpuDecodeParallelism,
    shared_direct_plan: Option<&DirectColorRegionCache>,
) -> Vec<J2kIndexedBatchResult> {
    let mut ctx = DecoderContext::<J2kContext>::new();
    ctx.codec_mut()
        .set_cpu_decode_parallelism(inner_parallelism);
    let mut pool = J2kScratchPool::new();
    let mut direct_scratch = J2kDirectCpuScratch::new();
    let mut direct_cache = None;
    let mut results = Vec::with_capacity(jobs.len());
    for (local_index, job) in jobs.iter_mut().enumerate() {
        let outcome = match decode_tile_region_scaled_shared_direct_color_u8_in_context(
            job,
            &mut ctx,
            fmt,
            &mut direct_scratch,
            shared_direct_plan,
        )
        .and_then(|outcome| {
            if outcome.is_some() {
                Ok(outcome)
            } else {
                decode_tile_region_scaled_direct_color_u8_in_context(
                    job,
                    &mut ctx,
                    fmt,
                    &mut direct_scratch,
                    &mut direct_cache,
                )
            }
        }) {
            Ok(Some(outcome)) => Ok(outcome),
            Ok(None) => decode_tile_region_scaled_into_in_context(
                job.input, &mut ctx, &mut pool, job.out, job.stride, fmt, job.roi, job.scale,
            ),
            Err(error) => Err(error),
        };
        results.push((start_index + local_index, outcome));
    }
    results
}

struct DirectColorRegionCache {
    input_ptr: usize,
    input_len: usize,
    roi: Rect,
    scale: Downscale,
    output_region: J2kRect,
    plan: J2kDirectColorPlan,
}

fn build_repeated_direct_color_region_plan(
    jobs: &[TileRegionScaledDecodeJob<'_, '_>],
    fmt: PixelFormat,
) -> Result<Option<DirectColorRegionCache>, J2kError> {
    if !is_direct_color_u8_format(fmt) {
        return Ok(None);
    }
    let Some(first) = jobs.first() else {
        return Ok(None);
    };
    if first.scale == Downscale::None {
        return Ok(None);
    }
    let key = DirectColorRegionKey {
        input_ptr: first.input.as_ptr() as usize,
        input_len: first.input.len(),
        roi: first.roi,
        scale: first.scale,
    };
    if !jobs.iter().all(|job| {
        job.input.as_ptr() as usize == key.input_ptr
            && job.input.len() == key.input_len
            && job.roi == key.roi
            && job.scale == key.scale
    }) {
        return Ok(None);
    }

    let Some((plan, output_region)) =
        build_direct_color_region_plan(first.input, first.roi, first.scale)?
    else {
        return Ok(None);
    };
    Ok(Some(DirectColorRegionCache {
        input_ptr: key.input_ptr,
        input_len: key.input_len,
        roi: key.roi,
        scale: key.scale,
        output_region,
        plan,
    }))
}

fn decode_tile_region_scaled_shared_direct_color_u8_in_context(
    job: &mut TileRegionScaledDecodeJob<'_, '_>,
    _ctx: &mut DecoderContext<J2kContext>,
    fmt: PixelFormat,
    scratch: &mut J2kDirectCpuScratch,
    shared_direct_plan: Option<&DirectColorRegionCache>,
) -> Result<Option<BatchOutcome>, J2kError> {
    let Some(shared_direct_plan) = shared_direct_plan else {
        return Ok(None);
    };
    if !is_direct_color_u8_format(fmt)
        || !shared_direct_plan.matches(DirectColorRegionKey {
            input_ptr: job.input.as_ptr() as usize,
            input_len: job.input.len(),
            roi: job.roi,
            scale: job.scale,
        })
    {
        return Ok(None);
    }

    let decoded = job.roi.scaled_covering(job.scale);
    validate_buffer((decoded.w, decoded.h), job.out.len(), job.stride, fmt)?;
    execute_direct_color_plan_u8_into(
        &shared_direct_plan.plan,
        shared_direct_plan.output_region,
        scratch,
        job.out,
        job.stride,
        fmt,
    )?;
    Ok(Some(DecodeOutcome::new(decoded, Vec::new())))
}

fn decode_tile_region_scaled_direct_color_u8_in_context(
    job: &mut TileRegionScaledDecodeJob<'_, '_>,
    _ctx: &mut DecoderContext<J2kContext>,
    fmt: PixelFormat,
    scratch: &mut J2kDirectCpuScratch,
    cache: &mut Option<DirectColorRegionCache>,
) -> Result<Option<BatchOutcome>, J2kError> {
    if !is_direct_color_u8_format(fmt) || job.scale == Downscale::None {
        return Ok(None);
    }

    let decoded = job.roi.scaled_covering(job.scale);
    validate_buffer((decoded.w, decoded.h), job.out.len(), job.stride, fmt)?;
    let key = DirectColorRegionKey {
        input_ptr: job.input.as_ptr() as usize,
        input_len: job.input.len(),
        roi: job.roi,
        scale: job.scale,
    };
    if !cache.as_ref().is_some_and(|cache| cache.matches(key)) {
        let Some((plan, output_region)) =
            build_direct_color_region_plan(job.input, job.roi, job.scale)?
        else {
            return Ok(None);
        };
        *cache = Some(DirectColorRegionCache {
            input_ptr: key.input_ptr,
            input_len: key.input_len,
            roi: key.roi,
            scale: key.scale,
            output_region,
            plan,
        });
    }

    let cache = cache
        .as_ref()
        .ok_or_else(|| J2kError::Backend("internal direct color plan cache missing".to_string()))?;
    execute_direct_color_plan_u8_into(
        &cache.plan,
        cache.output_region,
        scratch,
        job.out,
        job.stride,
        fmt,
    )?;
    Ok(Some(DecodeOutcome::new(decoded, Vec::new())))
}

#[derive(Clone, Copy)]
struct DirectColorRegionKey {
    input_ptr: usize,
    input_len: usize,
    roi: Rect,
    scale: Downscale,
}

impl DirectColorRegionCache {
    fn matches(&self, key: DirectColorRegionKey) -> bool {
        self.input_ptr == key.input_ptr
            && self.input_len == key.input_len
            && self.roi == key.roi
            && self.scale == key.scale
    }
}

fn is_direct_color_u8_format(fmt: PixelFormat) -> bool {
    matches!(fmt, PixelFormat::Rgb8 | PixelFormat::Rgba8)
}

fn execute_direct_color_plan_u8_into(
    plan: &J2kDirectColorPlan,
    output_region: J2kRect,
    scratch: &mut J2kDirectCpuScratch,
    out: &mut [u8],
    stride: usize,
    fmt: PixelFormat,
) -> Result<(), J2kError> {
    match fmt {
        PixelFormat::Rgb8 => {
            execute_direct_color_plan_rgb8_into(plan, output_region, scratch, out, stride)
        }
        PixelFormat::Rgba8 => {
            execute_direct_color_plan_rgba8_into(plan, output_region, scratch, out, stride)
        }
        _ => unreachable!("validated direct color output format"),
    }
    .map_err(|error| J2kError::Backend(error.to_string()))
}

fn build_direct_color_region_plan(
    input: &[u8],
    roi: Rect,
    scale: Downscale,
) -> Result<Option<(J2kDirectColorPlan, J2kRect)>, J2kError> {
    if !input_declares_htj2k(input) {
        return Ok(None);
    }

    let Ok(parsed) = parse_image_info(input) else {
        return Ok(None);
    };
    if !matches!(
        parsed.transfer_syntax,
        CompressedTransferSyntax::HtJpeg2000Lossless | CompressedTransferSyntax::HtJpeg2000Lossy
    ) {
        return Ok(None);
    }

    validate_region(roi, parsed.info.dimensions)?;
    let target_dims = (
        parsed.info.dimensions.0.div_ceil(scale.denominator()),
        parsed.info.dimensions.1.div_ceil(scale.denominator()),
    );
    let output_region = roi.scaled_covering(scale);
    let image = backend::image(
        input,
        DecodeSettings {
            target_resolution: Some(target_dims),
            ..DecodeSettings::default()
        },
    )?;
    validate_region(output_region, (image.width(), image.height()))?;

    let mut native_context = j2k_native::DecoderContext::default();
    match image.build_direct_color_plan_region_with_context(
        &mut native_context,
        (
            output_region.x,
            output_region.y,
            output_region.w,
            output_region.h,
        ),
    ) {
        Ok(plan) if direct_color_plan_uses_only_htj2k(&plan) => Ok(Some((
            plan,
            J2kRect {
                x0: output_region.x,
                y0: output_region.y,
                x1: output_region.x + output_region.w,
                y1: output_region.y + output_region.h,
            },
        ))),
        Ok(_) => Ok(None),
        Err(error) if is_unsupported_direct_color_plan_error(error) => Ok(None),
        Err(error) => Err(J2kError::Backend(error.to_string())),
    }
}

fn input_declares_htj2k(input: &[u8]) -> bool {
    crate::extract_j2k_codestream_payload(input)
        .ok()
        .and_then(|payload| inspect_j2k_codestream_header(payload.codestream()).ok())
        .is_some_and(|metadata| metadata.high_throughput)
}

fn direct_color_plan_uses_only_htj2k(plan: &J2kDirectColorPlan) -> bool {
    plan.component_plans.iter().all(|component| {
        component.steps.iter().any(|step| {
            matches!(
                step,
               j2k_native::J2kDirectGrayscaleStep::HtSubBand(sub_band)
                    if !sub_band.jobs.is_empty()
            )
        }) && component
            .steps
            .iter()
            .all(|step| !matches!(step, j2k_native::J2kDirectGrayscaleStep::ClassicSubBand(_)))
    })
}

fn is_unsupported_direct_color_plan_error(error: NativeDecodeError) -> bool {
    matches!(
        error,
        NativeDecodeError::Decoding(NativeDecodingError::UnsupportedFeature(_))
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use j2k_native::{encode, encode_htj2k, EncodeOptions};
    use j2k_test_support::wrap_codestream_jp2;

    fn encode_rgb_codestream(htj2k: bool) -> Vec<u8> {
        let pixels = (0..16 * 16 * 3)
            .map(|idx| ((idx * 11 + idx / 3) & 0xff) as u8)
            .collect::<Vec<_>>();
        let options = EncodeOptions {
            reversible: true,
            num_decomposition_levels: 2,
            ..EncodeOptions::default()
        };
        if htj2k {
            encode_htj2k(&pixels, 16, 16, 3, 8, false, &options).expect("encode HTJ2K")
        } else {
            encode(&pixels, 16, 16, 3, 8, false, &options).expect("encode J2K")
        }
    }

    #[test]
    fn htj2k_eligibility_accepts_raw_and_jp2_wrapped_inputs() {
        let raw_htj2k = encode_rgb_codestream(true);
        let jp2_htj2k = wrap_codestream_jp2(&raw_htj2k, 16, 16, 3, 8, 16);
        let raw_classic = encode_rgb_codestream(false);
        let jp2_classic = wrap_codestream_jp2(&raw_classic, 16, 16, 3, 8, 16);

        assert!(input_declares_htj2k(&raw_htj2k));
        assert!(input_declares_htj2k(&jp2_htj2k));
        assert!(!input_declares_htj2k(&raw_classic));
        assert!(!input_declares_htj2k(&jp2_classic));

        let mut malformed_jp2 = jp2_htj2k;
        malformed_jp2[11] = 0;
        assert!(!input_declares_htj2k(&malformed_jp2));
        assert!(!input_declares_htj2k(&malformed_jp2[..8]));
    }
}
