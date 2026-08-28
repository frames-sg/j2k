// SPDX-License-Identifier: MIT OR Apache-2.0

use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};

use j2k::{decode_tiles_into, J2kDecoder, PixelFormat, TileBatchOptions, TileDecodeJob};
use j2k_compare::{
    grok, measure_repeated, openhtj2k, openjpeg, openjph, parse_positive_usize, sample_stats,
    usize_to_f64,
};
use j2k_core::tile_batch_worker_count;

const DEFAULT_BATCH_SIZES: &[usize] = &[1, 16, 512, 1024];
const DEFAULT_REPEATS: usize = 3;

#[derive(Clone)]
struct TileInput {
    path: PathBuf,
    bytes: Vec<u8>,
    dimensions: (u32, u32),
    format: PixelFormat,
}

struct Measurement {
    decoder: &'static str,
    batch_size: usize,
    repeats: usize,
    sample_ms: Vec<f64>,
    median_ms: f64,
    mean_ms: f64,
    tiles_per_second_median: f64,
    decoded_bytes_per_repeat: usize,
}

struct RunOptions {
    tile_dir: PathBuf,
    batch_sizes: Vec<usize>,
    repeats: usize,
    workers: Option<NonZeroUsize>,
}

#[derive(Clone, Copy)]
enum ExternalDecoder {
    OpenJpeg,
    Grok,
    OpenHtj2k,
    OpenJph,
}

impl ExternalDecoder {
    const OPTIONAL: [Self; 3] = [Self::Grok, Self::OpenHtj2k, Self::OpenJph];

    const fn label(self) -> &'static str {
        match self {
            Self::OpenJpeg => "openjpeg",
            Self::Grok => "grok",
            Self::OpenHtj2k => "openhtj2k",
            Self::OpenJph => "openjph",
        }
    }

    const fn display_name(self) -> &'static str {
        match self {
            Self::OpenJpeg => "OpenJPEG",
            Self::Grok => "Grok",
            Self::OpenHtj2k => "OpenHTJ2K",
            Self::OpenJph => "OpenJPH",
        }
    }

    fn is_available(self) -> bool {
        match self {
            Self::OpenJpeg => openjpeg::is_available(),
            Self::Grok => grok::is_available(),
            Self::OpenHtj2k => openhtj2k::is_available(),
            Self::OpenJph => openjph::is_available(),
        }
    }
}

fn main() {
    if let Err(err) = run() {
        eprintln!("{err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let options = parse_run_options()?;
    let max_batch_size = options
        .batch_sizes
        .iter()
        .copied()
        .max()
        .ok_or_else(|| "no batch sizes requested".to_string())?;
    let (tiles, skipped) = load_tiles(&options.tile_dir, max_batch_size)?;
    if tiles.len() < max_batch_size {
        return Err(format!(
            "only loaded {} supported tiles from {}; need {max_batch_size}; skipped {skipped}",
            tiles.len(),
            options.tile_dir.display()
        ));
    }
    let format = tiles[0].format;
    if !tiles.iter().all(|tile| tile.format == format) {
        return Err("selected tiles do not share one output pixel format".to_string());
    }
    let (openhtj2k_max_abs_diff, openjph_max_abs_diff) = reference_parity(&tiles, format)?;
    emit_configuration(
        &options,
        &tiles,
        skipped,
        format,
        &openhtj2k_max_abs_diff,
        &openjph_max_abs_diff,
    );
    emit_all_measurements(&options, &tiles)
}

fn parse_run_options() -> Result<RunOptions, String> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.is_empty() {
        return Err("usage: jp2k_batch_compare <raw-tile-dir> [batch-size ...]".to_string());
    }

    let tile_dir = PathBuf::from(&args[0]);
    if !tile_dir.is_dir() {
        return Err(format!(
            "raw tile path is not a directory: {}",
            tile_dir.display()
        ));
    }
    let batch_sizes = if args.len() > 1 {
        args[1..]
            .iter()
            .map(|value| parse_positive_usize(value, "batch size"))
            .collect::<Result<Vec<_>, _>>()?
    } else {
        DEFAULT_BATCH_SIZES.to_vec()
    };
    let repeats = std::env::var("J2K_BATCH_COMPARE_REPEATS")
        .ok()
        .map(|value| parse_positive_usize(&value, "J2K_BATCH_COMPARE_REPEATS"))
        .transpose()?
        .unwrap_or(DEFAULT_REPEATS);
    let workers = std::env::var("J2K_BATCH_COMPARE_THREADS")
        .ok()
        .map(|value| parse_positive_usize(&value, "J2K_BATCH_COMPARE_THREADS"))
        .transpose()?
        .map(|value| NonZeroUsize::new(value).expect("positive value was validated"));

    Ok(RunOptions {
        tile_dir,
        batch_sizes,
        repeats,
        workers,
    })
}

fn reference_parity(tiles: &[TileInput], format: PixelFormat) -> Result<(String, String), String> {
    let allowed_reference_difference = std::env::var("J2K_BATCH_COMPARE_MAX_ABS_DIFF")
        .ok()
        .map(|value| {
            value.parse::<u8>().map_err(|error| {
                format!("invalid J2K_BATCH_COMPARE_MAX_ABS_DIFF {value:?}: {error}")
            })
        })
        .transpose()?
        .unwrap_or(1);
    let openhtj2k_max_abs_diff = reference_parity_result(
        tiles,
        format,
        ExternalDecoder::OpenHtj2k,
        allowed_reference_difference,
    )?;
    let openjph_max_abs_diff = reference_parity_result(
        tiles,
        format,
        ExternalDecoder::OpenJph,
        allowed_reference_difference,
    )?;

    Ok((openhtj2k_max_abs_diff, openjph_max_abs_diff))
}

fn emit_configuration(
    options: &RunOptions,
    tiles: &[TileInput],
    skipped: usize,
    format: PixelFormat,
    openhtj2k_max_abs_diff: &str,
    openjph_max_abs_diff: &str,
) {
    println!(
        "tile_dir\t{}\nloaded_tiles\t{}\nskipped_unsupported\t{}\nformat\t{:?}\nworkers\t{}\nopenjpeg_available\t{}\ngrok_available\t{}\nopenhtj2k_available\t{}\nopenhtj2k_version\t{}\nopenhtj2k_library\t{}\nopenhtj2k_max_abs_diff\t{}\nopenjph_available\t{}\nopenjph_version\t{}\nopenjph_library\t{}\nopenjph_max_abs_diff\t{}",
        options.tile_dir.display(),
        tiles.len(),
        skipped,
        format,
        options
            .workers
            .map_or_else(|| "auto".to_string(), |value| value.get().to_string()),
        openjpeg::is_available(),
        grok::is_available(),
        openhtj2k::is_available(),
        openhtj2k::version(),
        openhtj2k::library_path(),
        openhtj2k_max_abs_diff,
        openjph::is_available(),
        openjph::version(),
        openjph::library_path(),
        openjph_max_abs_diff,
    );
    println!(
        "decoder\tbatch_size\trepeats\tmedian_ms\tmean_ms\ttiles_per_second_median\tdecoded_bytes_per_repeat\tsamples_ms"
    );
}

fn emit_all_measurements(options: &RunOptions, tiles: &[TileInput]) -> Result<(), String> {
    for &batch_size in &options.batch_sizes {
        emit_measurement(measure_j2k(
            &tiles[..batch_size],
            options.repeats,
            options.workers,
        )?);
        emit_measurement(measure_external(
            &tiles[..batch_size],
            options.repeats,
            options.workers,
            ExternalDecoder::OpenJpeg,
        )?);
        for decoder in ExternalDecoder::OPTIONAL {
            if decoder.is_available() {
                emit_measurement(measure_external(
                    &tiles[..batch_size],
                    options.repeats,
                    options.workers,
                    decoder,
                )?);
            } else {
                println!(
                    "{}\t{batch_size}\t{}\tNA\tNA\tNA\tNA\tunavailable",
                    decoder.label(),
                    options.repeats
                );
            }
        }
    }

    Ok(())
}

fn emit_measurement(row: Measurement) {
    let Measurement {
        decoder,
        batch_size,
        repeats,
        sample_ms,
        median_ms,
        mean_ms,
        tiles_per_second_median,
        decoded_bytes_per_repeat,
    } = row;
    let samples = sample_ms
        .iter()
        .map(|value| format!("{value:.6}"))
        .collect::<Vec<_>>()
        .join(",");
    println!(
        "{decoder}\t{batch_size}\t{repeats}\t{median_ms:.6}\t{mean_ms:.6}\t{tiles_per_second_median:.3}\t{decoded_bytes_per_repeat}\t{samples}"
    );
}

fn load_tiles(dir: &Path, limit: usize) -> Result<(Vec<TileInput>, usize), String> {
    let mut paths = std::fs::read_dir(dir)
        .map_err(|err| format!("read tile dir {}: {err}", dir.display()))?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|err| format!("read tile dir entry: {err}"))?;
    paths.retain(|path| {
        path.extension()
            .and_then(|value| value.to_str())
            .is_some_and(|extension| {
                matches!(
                    extension.to_ascii_lowercase().as_str(),
                    "j2k" | "j2c" | "jp2" | "jph" | "jhc"
                )
            })
    });
    paths.sort();

    let mut tiles = Vec::with_capacity(limit);
    let mut skipped = 0usize;
    for path in paths {
        if tiles.len() == limit {
            break;
        }
        let bytes =
            std::fs::read(&path).map_err(|err| format!("read {}: {err}", path.display()))?;
        let Ok(info) = J2kDecoder::inspect(&bytes) else {
            skipped += 1;
            continue;
        };
        let Some(format) = pixel_format(info.components, info.bit_depth) else {
            skipped += 1;
            continue;
        };
        tiles.push(TileInput {
            path,
            bytes,
            dimensions: info.dimensions,
            format,
        });
    }
    Ok((tiles, skipped))
}

fn pixel_format(components: u16, bit_depth: u8) -> Option<PixelFormat> {
    match (components, bit_depth) {
        (1, 8) => Some(PixelFormat::Gray8),
        (3, 8) => Some(PixelFormat::Rgb8),
        _ => None,
    }
}

fn measure_j2k(
    tiles: &[TileInput],
    repeats: usize,
    workers: Option<NonZeroUsize>,
) -> Result<Measurement, String> {
    let format = tiles[0].format;
    let (samples, decoded_bytes_per_repeat) =
        measure_repeated(repeats, 1000.0, || decode_j2k_once(tiles, format, workers))?;
    measurement(
        "j2k",
        tiles.len(),
        repeats,
        samples,
        decoded_bytes_per_repeat,
    )
}

fn decode_j2k_once(
    tiles: &[TileInput],
    format: PixelFormat,
    workers: Option<NonZeroUsize>,
) -> Result<usize, String> {
    Ok(decode_j2k_outputs(tiles, format, workers)?
        .iter()
        .map(Vec::len)
        .sum())
}

fn decode_j2k_outputs(
    tiles: &[TileInput],
    format: PixelFormat,
    workers: Option<NonZeroUsize>,
) -> Result<Vec<Vec<u8>>, String> {
    let mut outputs = tiles
        .iter()
        .map(|tile| vec![0_u8; output_len(tile, format)])
        .collect::<Vec<_>>();
    let mut jobs = tiles
        .iter()
        .zip(outputs.iter_mut())
        .map(|(tile, out)| TileDecodeJob {
            input: tile.bytes.as_slice(),
            out: out.as_mut_slice(),
            stride: stride(tile, format),
        })
        .collect::<Vec<_>>();
    decode_tiles_into(&mut jobs, format, TileBatchOptions { workers })
        .map_err(|err| format!("j2k batch decode failed: {err}"))?;
    Ok(outputs)
}

fn reference_parity_result(
    tiles: &[TileInput],
    format: PixelFormat,
    decoder: ExternalDecoder,
    allowed_difference: u8,
) -> Result<String, String> {
    if !decoder.is_available() {
        return Ok("unavailable".to_string());
    }
    let observed = validate_reference_parity(tiles, format, decoder)?;
    if observed > allowed_difference {
        return Err(format!(
            "{} maximum absolute byte difference {observed} exceeds the configured {allowed_difference}",
            decoder.display_name()
        ));
    }
    Ok(observed.to_string())
}

fn validate_reference_parity(
    tiles: &[TileInput],
    format: PixelFormat,
    decoder: ExternalDecoder,
) -> Result<u8, String> {
    let native = decode_j2k_outputs(tiles, format, NonZeroUsize::new(1))?;
    tiles
        .iter()
        .zip(native)
        .try_fold(0_u8, |maximum, (tile, expected)| {
            let actual = decode_external_tile(tile, decoder)?;
            if actual.len() != expected.len() {
                return Err(format!(
                    "{}: {} decoded {} bytes, native decoded {}",
                    tile.path.display(),
                    decoder.label(),
                    actual.len(),
                    expected.len()
                ));
            }
            Ok(maximum.max(maximum_absolute_byte_difference(&actual, &expected)))
        })
}

fn maximum_absolute_byte_difference(left: &[u8], right: &[u8]) -> u8 {
    left.iter()
        .zip(right)
        .map(|(&left, &right)| left.abs_diff(right))
        .max()
        .unwrap_or(0)
}

fn measure_external(
    tiles: &[TileInput],
    repeats: usize,
    workers: Option<NonZeroUsize>,
    decoder: ExternalDecoder,
) -> Result<Measurement, String> {
    let decoder_name = decoder.label();
    let (samples, decoded_bytes_per_repeat) = measure_repeated(repeats, 1000.0, || {
        decode_external_once(tiles, workers, decoder)
    })?;
    measurement(
        decoder_name,
        tiles.len(),
        repeats,
        samples,
        decoded_bytes_per_repeat,
    )
}

fn decode_external_once(
    tiles: &[TileInput],
    workers: Option<NonZeroUsize>,
    decoder: ExternalDecoder,
) -> Result<usize, String> {
    let worker_count = tile_batch_worker_count(
        tiles.len(),
        TileBatchOptions { workers },
        std::thread::available_parallelism().map_or(1, NonZeroUsize::get),
    );
    let chunk_size = tiles.len().div_ceil(worker_count);
    let total_decoded = std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(worker_count);
        for chunk in tiles.chunks(chunk_size) {
            handles.push(scope.spawn(move || {
                chunk
                    .iter()
                    .map(|tile| decode_external_tile(tile, decoder))
                    .try_fold(0usize, |acc, decoded| decoded.map(|data| acc + data.len()))
            }));
        }
        let mut decoded_bytes = 0usize;
        for handle in handles {
            match handle.join() {
                Ok(Ok(bytes)) => decoded_bytes += bytes,
                Ok(Err(err)) => return Err(err),
                Err(payload) => std::panic::resume_unwind(payload),
            }
        }
        Ok(decoded_bytes)
    })?;
    Ok(total_decoded)
}

fn decode_external_tile(tile: &TileInput, decoder: ExternalDecoder) -> Result<Vec<u8>, String> {
    let result = match (decoder, tile.format) {
        (ExternalDecoder::OpenJpeg, PixelFormat::Gray8) => openjpeg::decode_gray(&tile.bytes),
        (ExternalDecoder::OpenJpeg, PixelFormat::Rgb8) => openjpeg::decode_rgb(&tile.bytes),
        (ExternalDecoder::Grok, PixelFormat::Gray8) => grok::decode_gray(&tile.bytes),
        (ExternalDecoder::Grok, PixelFormat::Rgb8) => grok::decode_rgb(&tile.bytes),
        (ExternalDecoder::OpenHtj2k, PixelFormat::Gray8) => {
            openhtj2k::decode_gray(&tile.bytes, 0, 1)
        }
        (ExternalDecoder::OpenHtj2k, PixelFormat::Rgb8) => openhtj2k::decode_rgb(&tile.bytes, 0, 1),
        (ExternalDecoder::OpenJph, PixelFormat::Gray8) => openjph::decode_gray(&tile.bytes, 0),
        (ExternalDecoder::OpenJph, PixelFormat::Rgb8) => openjph::decode_rgb(&tile.bytes, 0),
        (_, other) => Err(format!(
            "{other:?} is not implemented for external comparator"
        )),
    };
    result.map_err(|err| format!("{}: {err}", tile.path.display()))
}

fn measurement(
    decoder: &'static str,
    batch_size: usize,
    repeats: usize,
    samples: Vec<f64>,
    decoded_bytes_per_repeat: usize,
) -> Result<Measurement, String> {
    let stats = sample_stats(&samples)?;
    Ok(Measurement {
        decoder,
        batch_size,
        repeats,
        sample_ms: samples,
        median_ms: stats.median,
        mean_ms: stats.mean,
        tiles_per_second_median: usize_to_f64(batch_size) / (stats.median / 1000.0),
        decoded_bytes_per_repeat,
    })
}

fn stride(tile: &TileInput, format: PixelFormat) -> usize {
    tile.dimensions.0 as usize * format.bytes_per_pixel()
}

fn output_len(tile: &TileInput, format: PixelFormat) -> usize {
    stride(tile, format) * tile.dimensions.1 as usize
}

#[cfg(test)]
mod tests {
    use super::{maximum_absolute_byte_difference, ExternalDecoder};
    use j2k_compare::{parse_positive_usize, sample_stats};

    #[test]
    fn openhtj2k_has_a_distinct_in_process_benchmark_label() {
        assert_eq!(ExternalDecoder::OpenHtj2k.label(), "openhtj2k",);
        assert_eq!(ExternalDecoder::OpenJph.label(), "openjph");
    }

    #[test]
    fn maximum_absolute_byte_difference_covers_both_directions() {
        assert_eq!(maximum_absolute_byte_difference(&[0, 200], &[3, 190]), 10);
        assert_eq!(maximum_absolute_byte_difference(&[9], &[2]), 7);
    }

    #[test]
    fn shared_parse_positive_usize_rejects_zero() {
        assert_eq!(parse_positive_usize("3", "threads"), Ok(3));
        assert_eq!(
            parse_positive_usize("0", "threads"),
            Err("threads must be > 0".to_string())
        );
    }

    #[test]
    fn shared_sample_stats_reports_mean_and_median() {
        let stats = sample_stats(&[9.0, 1.0, 5.0]).expect("stats");
        assert!((stats.median - 5.0).abs() < f64::EPSILON);
        assert!((stats.mean - 5.0).abs() < f64::EPSILON);
    }
}
