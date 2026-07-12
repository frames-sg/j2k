// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    decode_kakadu_once, decode_openjph_once, decode_tile_into_in_context,
    decode_tile_region_into_in_context, decode_tile_region_scaled_into_in_context,
    decode_tile_scaled_into_in_context, decode_tiles_into, decode_tiles_region_into,
    decode_tiles_region_scaled_into, decode_tiles_scaled_into, grok,
    is_openjpeg_region_scaled_noncomparable, openjpeg, reduce_factor, tile_batch_worker_count,
    BatchInputs, BenchmarkMode, CpuDecodeParallelism, DecoderContext, DecoderKind, FixtureCase,
    J2kContext, J2kScratchPool, MixedFixtureBatch, NonZeroUsize, Operation, OperationClass,
    PixelFormat, Rect, TileBatchOptions, TileDecodeJob, TileRegionDecodeJob,
    TileRegionScaledDecodeJob, TileScaledDecodeJob,
};

pub(super) fn decode_batch(
    benchmark_mode: BenchmarkMode,
    case: &FixtureCase,
    decoder: DecoderKind,
    batch_inputs: &BatchInputs,
    workers: Option<NonZeroUsize>,
) -> Result<Vec<u8>, String> {
    match decoder {
        DecoderKind::J2k => decode_j2k_batch(case, batch_inputs, workers),
        DecoderKind::OpenJpeg | DecoderKind::Grok | DecoderKind::OpenJph | DecoderKind::Kakadu => {
            decode_external_batch(benchmark_mode, case, decoder, batch_inputs, workers)
        }
    }
}

pub(super) fn decode_mixed_batch(
    benchmark_mode: BenchmarkMode,
    mixed_batch: &MixedFixtureBatch,
    decoder: DecoderKind,
    batch_size: usize,
    workers: Option<NonZeroUsize>,
) -> Result<Vec<u8>, String> {
    match decoder {
        DecoderKind::J2k => decode_j2k_mixed_batch(mixed_batch, batch_size, workers),
        DecoderKind::OpenJpeg | DecoderKind::Grok | DecoderKind::OpenJph | DecoderKind::Kakadu => {
            decode_external_mixed_batch(benchmark_mode, mixed_batch, decoder, batch_size, workers)
        }
    }
}

pub(super) fn decode_j2k_batch(
    case: &FixtureCase,
    batch_inputs: &BatchInputs,
    workers: Option<NonZeroUsize>,
) -> Result<Vec<u8>, String> {
    let output_len = case.output_len();
    let stride = case.output_stride();
    let batch_size = batch_inputs.len();
    if batch_size == 1 {
        return decode_j2k_single_case(case, batch_inputs.input(0));
    }
    let mut outputs = vec![vec![0_u8; output_len]; batch_size];
    match case.operation {
        Operation::Full => {
            let mut jobs = outputs
                .iter_mut()
                .enumerate()
                .map(|(index, out)| TileDecodeJob {
                    input: batch_inputs.input(index),
                    out: out.as_mut_slice(),
                    stride,
                })
                .collect::<Vec<_>>();
            decode_tiles_into(&mut jobs, case.format, TileBatchOptions { workers })
                .map_err(|error| format!("j2k full decode failed: {error}"))?;
        }
        Operation::Region(roi) => {
            let mut jobs = outputs
                .iter_mut()
                .enumerate()
                .map(|(index, out)| TileRegionDecodeJob {
                    input: batch_inputs.input(index),
                    out: out.as_mut_slice(),
                    stride,
                    roi,
                })
                .collect::<Vec<_>>();
            decode_tiles_region_into(&mut jobs, case.format, TileBatchOptions { workers })
                .map_err(|error| format!("j2k ROI decode failed: {error}"))?;
        }
        Operation::Scaled(scale) => {
            let mut jobs = outputs
                .iter_mut()
                .enumerate()
                .map(|(index, out)| TileScaledDecodeJob {
                    input: batch_inputs.input(index),
                    out: out.as_mut_slice(),
                    stride,
                    scale,
                })
                .collect::<Vec<_>>();
            decode_tiles_scaled_into(&mut jobs, case.format, TileBatchOptions { workers })
                .map_err(|error| format!("j2k scaled decode failed: {error}"))?;
        }
        Operation::RegionScaled { roi, scale } => {
            let mut jobs = outputs
                .iter_mut()
                .enumerate()
                .map(|(index, out)| TileRegionScaledDecodeJob {
                    input: batch_inputs.input(index),
                    out: out.as_mut_slice(),
                    stride,
                    roi,
                    scale,
                })
                .collect::<Vec<_>>();
            decode_tiles_region_scaled_into(&mut jobs, case.format, TileBatchOptions { workers })
                .map_err(|error| format!("j2k ROI+scaled decode failed: {error}"))?;
        }
    }
    Ok(flatten_outputs(outputs))
}

pub(super) fn decode_j2k_mixed_batch(
    mixed_batch: &MixedFixtureBatch,
    batch_size: usize,
    workers: Option<NonZeroUsize>,
) -> Result<Vec<u8>, String> {
    if batch_size == 1 {
        let case = mixed_case_at(mixed_batch, 0);
        return decode_j2k_single_case(case, case.bytes.as_slice());
    }
    let mut outputs = (0..batch_size)
        .map(|index| {
            let case = mixed_case_at(mixed_batch, index);
            vec![0_u8; case.output_len()]
        })
        .collect::<Vec<_>>();
    match mixed_batch.operation_class {
        OperationClass::Full => {
            let mut jobs = outputs
                .iter_mut()
                .enumerate()
                .map(|(index, out)| {
                    let case = mixed_case_at(mixed_batch, index);
                    TileDecodeJob {
                        input: case.bytes.as_slice(),
                        out: out.as_mut_slice(),
                        stride: case.output_stride(),
                    }
                })
                .collect::<Vec<_>>();
            decode_tiles_into(&mut jobs, mixed_batch.format, TileBatchOptions { workers })
                .map_err(|error| format!("j2k mixed full decode failed: {error}"))?;
        }
        OperationClass::Region => {
            let mut jobs = outputs
                .iter_mut()
                .enumerate()
                .map(|(index, out)| {
                    let case = mixed_case_at(mixed_batch, index);
                    let Operation::Region(roi) = case.operation else {
                        unreachable!("mixed operation class was validated");
                    };
                    TileRegionDecodeJob {
                        input: case.bytes.as_slice(),
                        out: out.as_mut_slice(),
                        stride: case.output_stride(),
                        roi,
                    }
                })
                .collect::<Vec<_>>();
            decode_tiles_region_into(&mut jobs, mixed_batch.format, TileBatchOptions { workers })
                .map_err(|error| format!("j2k mixed ROI decode failed: {error}"))?;
        }
        OperationClass::Scaled => {
            let mut jobs = outputs
                .iter_mut()
                .enumerate()
                .map(|(index, out)| {
                    let case = mixed_case_at(mixed_batch, index);
                    let Operation::Scaled(scale) = case.operation else {
                        unreachable!("mixed operation class was validated");
                    };
                    TileScaledDecodeJob {
                        input: case.bytes.as_slice(),
                        out: out.as_mut_slice(),
                        stride: case.output_stride(),
                        scale,
                    }
                })
                .collect::<Vec<_>>();
            decode_tiles_scaled_into(&mut jobs, mixed_batch.format, TileBatchOptions { workers })
                .map_err(|error| format!("j2k mixed scaled decode failed: {error}"))?;
        }
        OperationClass::RegionScaled => {
            let mut jobs = outputs
                .iter_mut()
                .enumerate()
                .map(|(index, out)| {
                    let case = mixed_case_at(mixed_batch, index);
                    let Operation::RegionScaled { roi, scale } = case.operation else {
                        unreachable!("mixed operation class was validated");
                    };
                    TileRegionScaledDecodeJob {
                        input: case.bytes.as_slice(),
                        out: out.as_mut_slice(),
                        stride: case.output_stride(),
                        roi,
                        scale,
                    }
                })
                .collect::<Vec<_>>();
            decode_tiles_region_scaled_into(
                &mut jobs,
                mixed_batch.format,
                TileBatchOptions { workers },
            )
            .map_err(|error| format!("j2k mixed ROI+scaled decode failed: {error}"))?;
        }
    }
    Ok(flatten_outputs(outputs))
}

pub(super) fn decode_j2k_single_case(case: &FixtureCase, input: &[u8]) -> Result<Vec<u8>, String> {
    let mut output = vec![0_u8; case.output_len()];
    let mut ctx = DecoderContext::<J2kContext>::new();
    ctx.codec_mut()
        .set_cpu_decode_parallelism(CpuDecodeParallelism::Serial);
    let mut pool = J2kScratchPool::new();
    match case.operation {
        Operation::Full => decode_tile_into_in_context(
            input,
            &mut ctx,
            &mut pool,
            j2k::TileDecodeOutput {
                out: &mut output,
                stride: case.output_stride(),
                fmt: case.format,
            },
        )
        .map_err(|error| format!("j2k serial full decode failed: {error}"))?,
        Operation::Region(roi) => decode_tile_region_into_in_context(
            input,
            &mut ctx,
            &mut pool,
            j2k::TileDecodeOutput {
                out: &mut output,
                stride: case.output_stride(),
                fmt: case.format,
            },
            roi,
        )
        .map_err(|error| format!("j2k serial ROI decode failed: {error}"))?,
        Operation::Scaled(scale) => decode_tile_scaled_into_in_context(
            input,
            &mut ctx,
            &mut pool,
            j2k::TileDecodeOutput {
                out: &mut output,
                stride: case.output_stride(),
                fmt: case.format,
            },
            scale,
        )
        .map_err(|error| format!("j2k serial scaled decode failed: {error}"))?,
        Operation::RegionScaled { roi, scale } => decode_tile_region_scaled_into_in_context(
            input,
            &mut ctx,
            &mut pool,
            j2k::TileDecodeOutput {
                out: &mut output,
                stride: case.output_stride(),
                fmt: case.format,
            },
            roi,
            scale,
        )
        .map_err(|error| format!("j2k serial ROI+scaled decode failed: {error}"))?,
    };
    Ok(output)
}

pub(super) fn decode_external_batch(
    benchmark_mode: BenchmarkMode,
    case: &FixtureCase,
    decoder: DecoderKind,
    batch_inputs: &BatchInputs,
    workers: Option<NonZeroUsize>,
) -> Result<Vec<u8>, String> {
    let batch_size = batch_inputs.len();
    let worker_count = tile_batch_worker_count(
        batch_size,
        TileBatchOptions { workers },
        std::thread::available_parallelism().map_or(1, NonZeroUsize::get),
    );
    let chunk_size = batch_size.div_ceil(worker_count);
    let chunks = (0..batch_size)
        .collect::<Vec<_>>()
        .chunks(chunk_size)
        .map(<[_]>::to_vec)
        .collect::<Vec<_>>();

    let outputs = std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(chunks.len());
        for chunk in chunks {
            handles.push(scope.spawn(move || {
                chunk
                    .iter()
                    .map(|index| {
                        decode_external_once(
                            benchmark_mode,
                            case,
                            decoder,
                            batch_inputs.input(*index),
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()
            }));
        }

        let mut outputs = Vec::with_capacity(batch_size);
        for handle in handles {
            match handle.join() {
                Ok(Ok(mut chunk_outputs)) => outputs.append(&mut chunk_outputs),
                Ok(Err(error)) => return Err(error),
                Err(payload) => std::panic::resume_unwind(payload),
            }
        }
        Ok(outputs)
    })?;
    Ok(flatten_outputs(outputs))
}

pub(super) fn decode_external_mixed_batch(
    benchmark_mode: BenchmarkMode,
    mixed_batch: &MixedFixtureBatch,
    decoder: DecoderKind,
    batch_size: usize,
    workers: Option<NonZeroUsize>,
) -> Result<Vec<u8>, String> {
    let worker_count = tile_batch_worker_count(
        batch_size,
        TileBatchOptions { workers },
        std::thread::available_parallelism().map_or(1, NonZeroUsize::get),
    );
    let chunk_size = batch_size.div_ceil(worker_count);
    let chunks = (0..batch_size)
        .collect::<Vec<_>>()
        .chunks(chunk_size)
        .map(<[_]>::to_vec)
        .collect::<Vec<_>>();

    let outputs = std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(chunks.len());
        for chunk in chunks {
            handles.push(scope.spawn(move || {
                chunk
                    .iter()
                    .map(|index| {
                        let case = mixed_case_at(mixed_batch, *index);
                        decode_external_once(benchmark_mode, case, decoder, case.bytes.as_slice())
                    })
                    .collect::<Result<Vec<_>, _>>()
            }));
        }

        let mut outputs = Vec::with_capacity(batch_size);
        for handle in handles {
            match handle.join() {
                Ok(Ok(mut chunk_outputs)) => outputs.append(&mut chunk_outputs),
                Ok(Err(error)) => return Err(error),
                Err(payload) => std::panic::resume_unwind(payload),
            }
        }
        Ok(outputs)
    })?;
    Ok(flatten_outputs(outputs))
}

pub(super) fn mixed_case_at(mixed_batch: &MixedFixtureBatch, index: usize) -> &FixtureCase {
    &mixed_batch.cases[index % mixed_batch.cases.len()]
}

pub(super) fn decode_external_once(
    benchmark_mode: BenchmarkMode,
    case: &FixtureCase,
    decoder: DecoderKind,
    input: &[u8],
) -> Result<Vec<u8>, String> {
    if should_emulate_region_scaled(benchmark_mode, decoder, case) {
        return decode_external_region_scaled_emulated_once(case, decoder, input);
    }

    let output = match (decoder, case.format, case.operation) {
        (DecoderKind::OpenJpeg, PixelFormat::Gray8, Operation::Full) => {
            openjpeg::decode_gray(input)
        }
        (DecoderKind::OpenJpeg, PixelFormat::Rgb8, Operation::Full) => openjpeg::decode_rgb(input),
        (DecoderKind::OpenJpeg, PixelFormat::Gray8, Operation::Region(roi)) => {
            openjpeg::decode_gray_region(input, roi)
        }
        (DecoderKind::OpenJpeg, PixelFormat::Rgb8, Operation::Region(roi)) => {
            openjpeg::decode_rgb_region(input, roi)
        }
        (DecoderKind::OpenJpeg, PixelFormat::Gray8, Operation::Scaled(scale)) => {
            openjpeg::decode_gray_scaled(input, reduce_factor(scale)?)
        }
        (DecoderKind::OpenJpeg, PixelFormat::Rgb8, Operation::Scaled(scale)) => {
            openjpeg::decode_rgb_scaled(input, reduce_factor(scale)?)
        }
        (DecoderKind::OpenJpeg, PixelFormat::Gray8, Operation::RegionScaled { roi, scale }) => {
            openjpeg::decode_gray_region_scaled(input, roi, reduce_factor(scale)?)
        }
        (DecoderKind::OpenJpeg, PixelFormat::Rgb8, Operation::RegionScaled { roi, scale }) => {
            openjpeg::decode_rgb_region_scaled(input, roi, reduce_factor(scale)?)
        }
        (DecoderKind::Grok, PixelFormat::Gray8, Operation::Full) => grok::decode_gray(input),
        (DecoderKind::Grok, PixelFormat::Rgb8, Operation::Full) => grok::decode_rgb(input),
        (DecoderKind::Grok, PixelFormat::Gray8, Operation::Region(roi)) => {
            grok::decode_gray_region(input, roi)
        }
        (DecoderKind::Grok, PixelFormat::Rgb8, Operation::Region(roi)) => {
            grok::decode_rgb_region(input, roi)
        }
        (DecoderKind::Grok, PixelFormat::Gray8, Operation::Scaled(scale)) => {
            grok::decode_gray_scaled(input, reduce_factor(scale)?)
        }
        (DecoderKind::Grok, PixelFormat::Rgb8, Operation::Scaled(scale)) => {
            grok::decode_rgb_scaled(input, reduce_factor(scale)?)
        }
        (DecoderKind::Grok, PixelFormat::Gray8, Operation::RegionScaled { roi, scale }) => {
            grok::decode_gray_region_scaled(input, roi, reduce_factor(scale)?)
        }
        (DecoderKind::Grok, PixelFormat::Rgb8, Operation::RegionScaled { roi, scale }) => {
            grok::decode_rgb_region_scaled(input, roi, reduce_factor(scale)?)
        }
        (
            DecoderKind::OpenJph,
            PixelFormat::Gray8 | PixelFormat::Rgb8,
            Operation::Full | Operation::Scaled(_),
        ) => decode_openjph_once(case, input),
        (
            DecoderKind::Kakadu,
            PixelFormat::Gray8 | PixelFormat::Rgb8,
            Operation::Full | Operation::Scaled(_),
        ) => decode_kakadu_once(case, input),
        (other, format, _) => Err(format!(
            "{} does not support {format:?} in fixture compare",
            other.label()
        )),
    }
    .map_err(|error| format!("{} {}: {error}", decoder.label(), case.name))?;

    let expected_len = case.output_len();
    if output.len() != expected_len {
        return Err(format!(
            "{} {}: decoded length {} != expected {expected_len}",
            decoder.label(),
            case.name,
            output.len()
        ));
    }
    Ok(output)
}

pub(super) fn should_emulate_region_scaled(
    benchmark_mode: BenchmarkMode,
    decoder: DecoderKind,
    case: &FixtureCase,
) -> bool {
    benchmark_mode == BenchmarkMode::PortableEmulated
        && decoder == DecoderKind::OpenJpeg
        && is_openjpeg_region_scaled_noncomparable(case)
}

pub(super) fn decode_method_label(
    benchmark_mode: BenchmarkMode,
    decoder: DecoderKind,
    case: &FixtureCase,
) -> &'static str {
    if decoder == DecoderKind::OpenJph {
        "openjph-cli-process-output-pnm"
    } else if decoder == DecoderKind::Kakadu {
        "kakadu-cli-process-output-pnm"
    } else if should_emulate_region_scaled(benchmark_mode, decoder, case) {
        "emulated-full-scaled-crop"
    } else {
        "native"
    }
}

pub(super) fn decode_external_region_scaled_emulated_once(
    case: &FixtureCase,
    decoder: DecoderKind,
    input: &[u8],
) -> Result<Vec<u8>, String> {
    let Operation::RegionScaled { roi, scale } = case.operation else {
        return Err(format!(
            "{} {}: emulation requested for non-ROI+scaled operation",
            decoder.label(),
            case.name
        ));
    };
    let full_scaled = match (decoder, case.format) {
        (DecoderKind::OpenJpeg, PixelFormat::Gray8) => {
            openjpeg::decode_gray_scaled(input, reduce_factor(scale)?)
        }
        (DecoderKind::OpenJpeg, PixelFormat::Rgb8) => {
            openjpeg::decode_rgb_scaled(input, reduce_factor(scale)?)
        }
        (DecoderKind::Grok, PixelFormat::Gray8) => {
            grok::decode_gray_scaled(input, reduce_factor(scale)?)
        }
        (DecoderKind::Grok, PixelFormat::Rgb8) => {
            grok::decode_rgb_scaled(input, reduce_factor(scale)?)
        }
        (other, format) => Err(format!(
            "{} does not support emulated {format:?} ROI+scaled fixture compare",
            other.label()
        )),
    }
    .map_err(|error| {
        format!(
            "{} {} emulated scaled decode: {error}",
            decoder.label(),
            case.name
        )
    })?;

    let full_scaled_dims = (
        case.dimensions.0.div_ceil(scale.denominator()),
        case.dimensions.1.div_ceil(scale.denominator()),
    );
    let scaled_roi = roi.scaled_covering(scale);
    crop_interleaved(&full_scaled, full_scaled_dims, scaled_roi, case.format)
        .map_err(|error| format!("{} {} emulated crop: {error}", decoder.label(), case.name))
}

pub(super) fn crop_interleaved(
    pixels: &[u8],
    dimensions: (u32, u32),
    roi: Rect,
    format: PixelFormat,
) -> Result<Vec<u8>, String> {
    if !roi.is_within(dimensions) {
        return Err(format!(
            "ROI {roi:?} exceeds scaled dimensions {dimensions:?}"
        ));
    }
    let bytes_per_pixel = format.bytes_per_pixel();
    let row_bytes = dimensions.0 as usize * bytes_per_pixel;
    let crop_row_bytes = roi.w as usize * bytes_per_pixel;
    let expected_len = row_bytes
        .checked_mul(dimensions.1 as usize)
        .ok_or_else(|| "scaled source dimensions overflow".to_string())?;
    if pixels.len() != expected_len {
        return Err(format!(
            "scaled source length {} != expected {expected_len}",
            pixels.len()
        ));
    }

    let mut out = Vec::with_capacity(crop_row_bytes * roi.h as usize);
    for y in roi.y..roi.y + roi.h {
        let start = y as usize * row_bytes + roi.x as usize * bytes_per_pixel;
        out.extend_from_slice(&pixels[start..start + crop_row_bytes]);
    }
    Ok(out)
}

pub(super) fn flatten_outputs(outputs: Vec<Vec<u8>>) -> Vec<u8> {
    let total_len = outputs.iter().map(Vec::len).sum();
    let mut flattened = Vec::with_capacity(total_len);
    for output in outputs {
        flattened.extend(output);
    }
    flattened
}

pub(super) fn pixel_format_label(format: PixelFormat) -> &'static str {
    match format {
        PixelFormat::Gray8 => "gray8",
        PixelFormat::Rgb8 => "rgb8",
        _ => "unsupported",
    }
}

#[cfg(test)]
mod tests;
