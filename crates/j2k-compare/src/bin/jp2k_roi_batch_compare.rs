// SPDX-License-Identifier: Apache-2.0

use std::num::NonZeroUsize;

use j2k::{
    decode_tiles_region_scaled_into, encode_j2k_lossless, EncodeBackendPreference,
    J2kBlockCodingMode, J2kEncodeValidation, J2kLosslessEncodeOptions, J2kLosslessSamples,
    TileBatchOptions, TileRegionScaledDecodeJob,
};
use j2k_compare::{grok, measure_repeated, parse_positive_usize, sample_stats, usize_to_f64};
use j2k_core::{tile_batch_worker_count, Downscale, PixelFormat, Rect};
use j2k_test_support::{patterned_rgb8, wrap_jp2_codestream};

const DEFAULT_REPEATS: usize = 9;
const DEFAULT_BATCH_SIZE: usize = 16;

struct CompareCase {
    name: &'static str,
    bytes: Vec<u8>,
    roi: Rect,
    scale: Downscale,
    batch_size: usize,
}

struct Measurement {
    decoder: &'static str,
    case_name: &'static str,
    repeats: usize,
    batch_size: usize,
    median_us: f64,
    mean_us: f64,
    tiles_per_second_median: f64,
    decoded_bytes_per_repeat: usize,
    samples_us: Vec<f64>,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    if !grok::is_available() {
        return Err(
            "in-process Grok is unavailable; install libgrokj2k or set J2K_GROK_SOURCE/J2K_GROK_ROOT"
                .to_string(),
        );
    }

    let repeats = std::env::var("J2K_ROI_COMPARE_REPEATS")
        .ok()
        .map(|value| parse_positive_usize(&value, "J2K_ROI_COMPARE_REPEATS"))
        .transpose()?
        .unwrap_or(DEFAULT_REPEATS);
    let workers = std::env::var("J2K_ROI_COMPARE_THREADS")
        .ok()
        .map(|value| parse_positive_usize(&value, "J2K_ROI_COMPARE_THREADS"))
        .transpose()?
        .map(|value| NonZeroUsize::new(value).expect("positive value was validated"));

    let cases = compare_cases()?;
    println!(
        "repeats\t{repeats}\nworkers\t{}\ngrok_available\t{}",
        workers.map_or_else(|| "auto".to_string(), |value| value.get().to_string()),
        grok::is_available()
    );
    println!(
        "decoder\tcase\tbatch_size\trepeats\tmedian_us\tmean_us\ttiles_per_second_median\tdecoded_bytes_per_repeat\tsamples_us"
    );

    for case in &cases {
        validate_case(case)?;
        emit_measurement(&measure_j2k(case, repeats, workers)?);
        emit_measurement(&measure_grok(case, repeats, workers)?);
    }
    Ok(())
}

fn compare_cases() -> Result<Vec<CompareCase>, String> {
    let raw_512 = encode_htj2k_rgb_codestream(512, 512)?;
    let jp2_512 = wrap_jp2_codestream(&raw_512, 512, 512, 3, 8, 16);
    let raw_256 = encode_htj2k_rgb_codestream(256, 256)?;
    let jp2_256 = wrap_jp2_codestream(&raw_256, 256, 256, 3, 8, 16);
    Ok(vec![
        CompareCase {
            name: "htj2k_raw_rgb8_512_roi256_q4_repeated_batch16",
            bytes: raw_512,
            roi: Rect {
                x: 128,
                y: 128,
                w: 256,
                h: 256,
            },
            scale: Downscale::Quarter,
            batch_size: DEFAULT_BATCH_SIZE,
        },
        CompareCase {
            name: "htj2k_jp2_rgb8_512_roi256_q4_repeated_batch16",
            bytes: jp2_512,
            roi: Rect {
                x: 128,
                y: 128,
                w: 256,
                h: 256,
            },
            scale: Downscale::Quarter,
            batch_size: DEFAULT_BATCH_SIZE,
        },
        CompareCase {
            name: "htj2k_jp2_rgb8_256_roi128_q4_repeated_batch16",
            bytes: jp2_256,
            roi: Rect {
                x: 64,
                y: 64,
                w: 128,
                h: 128,
            },
            scale: Downscale::Quarter,
            batch_size: DEFAULT_BATCH_SIZE,
        },
    ])
}

fn encode_htj2k_rgb_codestream(width: u32, height: u32) -> Result<Vec<u8>, String> {
    let pixels = patterned_rgb8(width, height);
    let samples = J2kLosslessSamples::new(&pixels, width, height, 3, 8, false)
        .map_err(|error| error.to_string())?;
    let options = J2kLosslessEncodeOptions::default()
        .with_backend(EncodeBackendPreference::CpuOnly)
        .with_block_coding_mode(J2kBlockCodingMode::HighThroughput)
        .with_max_decomposition_levels(Some(2))
        .with_validation(J2kEncodeValidation::External);
    Ok(encode_j2k_lossless(samples, &options)
        .map_err(|error| error.to_string())?
        .codestream)
}

fn validate_case(case: &CompareCase) -> Result<(), String> {
    let ours = decode_j2k_once(case)?;
    let theirs = decode_grok_once(case, None)?;
    if ours != theirs {
        return Err(format!(
            "{}: j2k/Grok ROI+scale output mismatch: {} vs {} bytes",
            case.name,
            ours.len(),
            theirs.len()
        ));
    }
    Ok(())
}

fn measure_j2k(
    case: &CompareCase,
    repeats: usize,
    workers: Option<NonZeroUsize>,
) -> Result<Measurement, String> {
    let (samples, decoded) = measure_repeated(repeats, 1_000_000.0, || {
        decode_j2k_once_with_workers(case, workers)
    })?;
    measurement("j2k", case, repeats, samples, decoded.len())
}

fn decode_j2k_once(case: &CompareCase) -> Result<Vec<u8>, String> {
    decode_j2k_once_with_workers(case, None)
}

fn decode_j2k_once_with_workers(
    case: &CompareCase,
    workers: Option<NonZeroUsize>,
) -> Result<Vec<u8>, String> {
    let output_len = output_len(case);
    let stride = output_stride(case);
    let mut outputs = vec![vec![0_u8; output_len]; case.batch_size];
    let mut jobs = outputs
        .iter_mut()
        .map(|out| TileRegionScaledDecodeJob {
            input: case.bytes.as_slice(),
            out: out.as_mut_slice(),
            stride,
            roi: case.roi,
            scale: case.scale,
        })
        .collect::<Vec<_>>();
    decode_tiles_region_scaled_into(&mut jobs, PixelFormat::Rgb8, TileBatchOptions { workers })
        .map_err(|error| error.to_string())?;
    Ok(outputs.into_iter().flatten().collect())
}

fn measure_grok(
    case: &CompareCase,
    repeats: usize,
    workers: Option<NonZeroUsize>,
) -> Result<Measurement, String> {
    let (samples, decoded) =
        measure_repeated(repeats, 1_000_000.0, || decode_grok_once(case, workers))?;
    measurement("grok", case, repeats, samples, decoded.len())
}

fn decode_grok_once(case: &CompareCase, workers: Option<NonZeroUsize>) -> Result<Vec<u8>, String> {
    let worker_count = tile_batch_worker_count(
        case.batch_size,
        TileBatchOptions { workers },
        std::thread::available_parallelism().map_or(1, NonZeroUsize::get),
    );
    let chunk_size = case.batch_size.div_ceil(worker_count);
    let reduce = reduce_factor(case.scale)?;
    let chunks = (0..case.batch_size)
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
                    .map(|_| grok::decode_rgb_region_scaled(&case.bytes, case.roi, reduce))
                    .collect::<Result<Vec<_>, _>>()
            }));
        }

        let mut outputs = Vec::with_capacity(case.batch_size);
        for handle in handles {
            match handle.join() {
                Ok(Ok(mut chunk_outputs)) => outputs.append(&mut chunk_outputs),
                Ok(Err(error)) => return Err(error),
                Err(payload) => std::panic::resume_unwind(payload),
            }
        }
        Ok(outputs)
    })?;
    Ok(outputs.into_iter().flatten().collect())
}

fn measurement(
    decoder: &'static str,
    case: &CompareCase,
    repeats: usize,
    samples: Vec<f64>,
    decoded_bytes_per_repeat: usize,
) -> Result<Measurement, String> {
    let stats = sample_stats(&samples)?;
    Ok(Measurement {
        decoder,
        case_name: case.name,
        repeats,
        batch_size: case.batch_size,
        median_us: stats.median,
        mean_us: stats.mean,
        tiles_per_second_median: usize_to_f64(case.batch_size) / (stats.median / 1_000_000.0),
        decoded_bytes_per_repeat,
        samples_us: samples,
    })
}

fn emit_measurement(row: &Measurement) {
    let samples = row
        .samples_us
        .iter()
        .map(|value| format!("{value:.3}"))
        .collect::<Vec<_>>()
        .join(",");
    println!(
        "{}\t{}\t{}\t{}\t{:.3}\t{:.3}\t{:.3}\t{}\t{}",
        row.decoder,
        row.case_name,
        row.batch_size,
        row.repeats,
        row.median_us,
        row.mean_us,
        row.tiles_per_second_median,
        row.decoded_bytes_per_repeat,
        samples
    );
}

fn output_stride(case: &CompareCase) -> usize {
    case.roi.scaled_covering(case.scale).w as usize * PixelFormat::Rgb8.bytes_per_pixel()
}

fn output_len(case: &CompareCase) -> usize {
    let scaled = case.roi.scaled_covering(case.scale);
    output_stride(case) * scaled.h as usize
}

fn reduce_factor(scale: Downscale) -> Result<u32, String> {
    match scale {
        Downscale::None => Ok(0),
        Downscale::Half => Ok(1),
        Downscale::Quarter => Ok(2),
        Downscale::Eighth => Ok(3),
        _ => Err(format!("unsupported downscale for Grok compare: {scale:?}")),
    }
}
