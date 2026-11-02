// SPDX-License-Identifier: MIT OR Apache-2.0

use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use j2k::{
    decode_tiles_into, release_tile_batch_workspaces, CpuDecodeParallelism, J2kDecoder,
    J2kScratchPool, PixelFormat, TileBatchOptions, TileDecodeJob,
};
use j2k_compare::{
    grok, openhtj2k, openjpeg, openjph, parse_positive_usize, sample_stats, usize_to_f64,
    ExternalDecodeRequest,
};
use j2k_core::{tile_batch_worker_count, Rect};

#[path = "jp2k_batch_compare/viewport.rs"]
mod viewport;
use viewport::ViewportSpec;

const DEFAULT_BATCH_SIZES: &[usize] = &[1, 16, 512, 1024];
const DEFAULT_REPEATS: usize = 3;

#[derive(Clone)]
struct TileInput {
    path: PathBuf,
    /// Shared, so the requests of a viewport run all read one input buffer.
    bytes: Arc<[u8]>,
    dimensions: (u32, u32),
    format: PixelFormat,
    /// Source region to decode; `None` decodes the whole image.
    roi: Option<Rect>,
    /// Power-of-two resolution reduction applied to the region.
    reduce: u8,
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
    measured_decoder: Option<String>,
    /// Viewer-style region requests over the run's single input.
    viewport: Option<ViewportSpec>,
    /// Free j2k's retained batch workspaces before every timed j2k sample, so
    /// j2k allocates its scratch inside each sample as the other decoders do.
    j2k_cold: bool,
    /// External decoders left out entirely, for example `OpenJPH` on Part 1
    /// codestreams it cannot decode.
    excluded: Vec<ExternalDecoder>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ExternalDecoder {
    OpenJpeg,
    Grok,
    OpenHtj2k,
    OpenJph,
}

impl ExternalDecoder {
    const ALL: [Self; 4] = [Self::OpenJpeg, Self::Grok, Self::OpenHtj2k, Self::OpenJph];

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

    /// Whether the library can spread one image over several threads.
    /// `OpenJPH` decodes each image on the calling thread only.
    const fn supports_intra_image_threads(self) -> bool {
        !matches!(self, Self::OpenJph)
    }

    fn from_label(label: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|decoder| decoder.label() == label)
    }
}

/// Threads each decoder may use inside one image, given the batch length and
/// the run's thread budget. This mirrors j2k's own batch policy for an
/// explicit worker count: a lone image gets the whole budget, and images of a
/// longer batch decode serially, one per worker.
fn intra_image_threads(batch_len: usize, thread_budget: usize) -> u32 {
    if batch_len <= 1 {
        u32::try_from(thread_budget.max(1)).unwrap_or(u32::MAX)
    } else {
        1
    }
}

/// j2k's internal parallelism uses the global Rayon pool, so a lone image
/// would otherwise use every core whatever the requested budget. Require
/// `RAYON_NUM_THREADS` to match the budget so every decoder gets the same
/// number of threads.
fn check_rayon_threads(thread_budget: usize, rayon_env: Option<&str>) -> Result<(), String> {
    let rayon = rayon_env
        .ok_or_else(|| {
            format!("set RAYON_NUM_THREADS={thread_budget} so j2k uses the same thread budget as the other decoders")
        })?
        .trim()
        .parse::<usize>()
        .map_err(|err| format!("invalid RAYON_NUM_THREADS: {err}"))?;
    if rayon == thread_budget {
        Ok(())
    } else {
        Err(format!(
            "RAYON_NUM_THREADS={rayon} differs from the thread budget {thread_budget}; set them equal"
        ))
    }
}

fn parse_excluded(value: &str) -> Result<Vec<ExternalDecoder>, String> {
    value
        .split(',')
        .map(str::trim)
        .filter(|label| !label.is_empty())
        .map(|label| {
            ExternalDecoder::from_label(label)
                .ok_or_else(|| format!("unknown decoder {label:?} in J2K_BATCH_COMPARE_EXCLUDE"))
        })
        .collect()
}

fn main() {
    if let Err(err) = run() {
        eprintln!("{err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let options = parse_run_options()?;
    let pool_threads = options.workers.map_or_else(
        || std::thread::available_parallelism().map_or(1, NonZeroUsize::get),
        NonZeroUsize::get,
    );
    check_rayon_threads(
        pool_threads,
        std::env::var("RAYON_NUM_THREADS").ok().as_deref(),
    )?;
    if grok::is_available() {
        grok::initialize_with_threads(
            u32::try_from(pool_threads)
                .ok()
                .and_then(std::num::NonZeroU32::new)
                .ok_or("Grok pool thread count exceeds u32")?,
        )?;
    }
    let max_batch_size = options
        .batch_sizes
        .iter()
        .copied()
        .max()
        .ok_or_else(|| "no batch sizes requested".to_string())?;
    let load_limit = if options.viewport.is_some() {
        1
    } else {
        max_batch_size
    };
    let (tiles, skipped) = load_tiles(&options.tile_dir, load_limit)?;
    let tiles = match options.viewport {
        Some(spec) => {
            let [image] = tiles.as_slice() else {
                return Err("J2K_BATCH_COMPARE_VIEWPORT needs exactly one input".to_string());
            };
            spec.requests(image.dimensions)
                .into_iter()
                .map(|roi| TileInput {
                    roi: Some(roi),
                    reduce: spec.reduce,
                    ..image.clone()
                })
                .collect()
        }
        None => tiles,
    };
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
    // With J2K_BATCH_COMPARE_KEEP_GOING, a failure is a result: j2k failing ends
    // the cell with its status, and an external decoder that errors or returns
    // different pixels is reported as failed and left out of the timing.
    let keep_going = std::env::var_os("J2K_BATCH_COMPARE_KEEP_GOING").is_some();
    if let Err(err) = decode_j2k_outputs(&tiles, format, NonZeroUsize::new(1)) {
        if keep_going {
            println!("j2k_status\tfailed: {err}");
            return Ok(());
        }
        return Err(err);
    }
    println!("j2k_status\tok");
    let parity = reference_parity(&tiles, format, &options.excluded, keep_going)?;
    let mut options = options;
    for (decoder, status) in &parity {
        if status.starts_with("failed") && !options.excluded.contains(decoder) {
            options.excluded.push(*decoder);
        }
    }
    emit_configuration(&options, &tiles, skipped, format, &parity, pool_threads);
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
    let excluded = std::env::var("J2K_BATCH_COMPARE_EXCLUDE")
        .ok()
        .map(|value| parse_excluded(&value))
        .transpose()?
        .unwrap_or_default();
    let viewport = std::env::var("J2K_BATCH_COMPARE_VIEWPORT")
        .ok()
        .map(|value| ViewportSpec::parse(&value))
        .transpose()?;
    let mut excluded = excluded;
    if viewport.is_some() && !excluded.contains(&ExternalDecoder::OpenJph) {
        // OpenJPH has no region decode; decoding the whole image and cropping
        // would not be the same task, so viewport runs leave it out.
        excluded.push(ExternalDecoder::OpenJph);
    }
    let measured_decoder = std::env::var("J2K_BATCH_COMPARE_DECODER").ok();
    if let Some(label) = measured_decoder.as_deref() {
        if label != "j2k" {
            let decoder = ExternalDecoder::from_label(label)
                .ok_or_else(|| format!("unknown J2K_BATCH_COMPARE_DECODER {label:?}"))?;
            if !decoder.is_available() {
                return Err(format!("requested decoder {label:?} is unavailable"));
            }
            if excluded.contains(&decoder) {
                return Err(format!("requested decoder {label:?} is also excluded"));
            }
        }
    }

    Ok(RunOptions {
        tile_dir,
        batch_sizes,
        repeats,
        workers,
        measured_decoder,
        excluded,
        j2k_cold: std::env::var_os("J2K_BATCH_COMPARE_J2K_COLD").is_some(),
        viewport,
    })
}

fn reference_parity(
    tiles: &[TileInput],
    format: PixelFormat,
    excluded: &[ExternalDecoder],
    keep_going: bool,
) -> Result<Vec<(ExternalDecoder, String)>, String> {
    let allowed_reference_difference = std::env::var("J2K_BATCH_COMPARE_MAX_ABS_DIFF")
        .ok()
        .map(|value| {
            value.parse::<u8>().map_err(|error| {
                format!("invalid J2K_BATCH_COMPARE_MAX_ABS_DIFF {value:?}: {error}")
            })
        })
        .transpose()?
        .unwrap_or(1);
    ExternalDecoder::ALL
        .into_iter()
        .map(|decoder| {
            if excluded.contains(&decoder) {
                return Ok((decoder, "excluded".to_string()));
            }
            match reference_parity_result(tiles, format, decoder, allowed_reference_difference) {
                Ok(difference) => Ok((decoder, difference)),
                Err(err) if keep_going => Ok((decoder, format!("failed: {err}"))),
                Err(err) => Err(err),
            }
        })
        .collect()
}

fn emit_configuration(
    options: &RunOptions,
    tiles: &[TileInput],
    skipped: usize,
    format: PixelFormat,
    parity: &[(ExternalDecoder, String)],
    pool_threads: usize,
) {
    println!(
        "tile_dir\t{}\nloaded_tiles\t{}\nskipped_unsupported\t{}\nformat\t{:?}\nworkers\t{}",
        options.tile_dir.display(),
        tiles.len(),
        skipped,
        format,
        options
            .workers
            .map_or_else(|| "auto".to_string(), |value| value.get().to_string()),
    );
    for (decoder, difference) in parity {
        let name = decoder.label();
        let (version, library) = match decoder {
            ExternalDecoder::OpenJpeg => (openjpeg::version(), openjpeg::library_path()),
            ExternalDecoder::Grok => (grok::version().to_owned(), grok::library_path()),
            ExternalDecoder::OpenHtj2k => {
                (openhtj2k::version().to_owned(), openhtj2k::library_path())
            }
            ExternalDecoder::OpenJph => (openjph::version().to_owned(), openjph::library_path()),
        };
        println!(
            "{name}_available\t{}\n{name}_version\t{version}\n{name}_library\t{library}\n{name}_max_abs_diff\t{difference}",
            decoder.is_available()
        );
    }
    let excluded = options
        .excluded
        .iter()
        .map(|decoder| decoder.label().to_string())
        .collect::<Vec<_>>();
    println!(
        "excluded_decoders\t{}",
        j2k_compare::common::join_string_labels(&excluded)
    );
    println!("thread_budget\t{pool_threads}\nrayon_num_threads\t{pool_threads}");
    if let Some(spec) = options.viewport {
        println!(
            "viewport\treduce={} count={} output_size={} seed={}\nopenjph_region_support\tnone-excluded-from-viewport-runs",
            spec.reduce, spec.count, spec.size, spec.seed
        );
        for (index, tile) in tiles.iter().enumerate() {
            if let Some(roi) = tile.roi {
                println!(
                    "viewport_request_{index}\tx={} y={} w={} h={}",
                    roi.x, roi.y, roi.w, roi.h
                );
            }
        }
    }
    println!("intra_image_threads_policy\tbudget-for-a-one-image-batch-else-1-per-worker");
    for decoder in ExternalDecoder::ALL {
        println!(
            "{}_intra_image_threads_supported\t{}",
            decoder.label(),
            decoder.supports_intra_image_threads()
        );
    }
    if grok::is_available() {
        println!("grok_pool_threads\t{pool_threads}\ngrok_decode_threads\tintra_image_threads");
    }
    println!("timing_boundary\tin-process-parse-decode-pack-allocate-free-no-file-io");
    println!("sample_order\trotating-decoder-order");
    println!(
        "measured_decoder\t{}",
        options.measured_decoder.as_deref().unwrap_or("all")
    );
    println!("warmup_batches_per_decoder\t1");
    println!(
        "j2k_workspace_policy\t{}",
        if options.j2k_cold {
            "released-before-each-timed-sample"
        } else {
            "retained-between-calls"
        }
    );
    println!(
        "decoder\tbatch_size\trepeats\tmedian_ms\tmean_ms\ttiles_per_second_median\tdecoded_bytes_per_repeat\tsamples_ms"
    );
}

fn emit_all_measurements(options: &RunOptions, tiles: &[TileInput]) -> Result<(), String> {
    for &batch_size in &options.batch_sizes {
        let inputs = &tiles[..batch_size];
        let decoders = std::iter::once(None)
            .chain(
                ExternalDecoder::ALL
                    .into_iter()
                    .filter(|d| d.is_available() && !options.excluded.contains(d))
                    .map(Some),
            )
            .filter(|decoder| {
                options
                    .measured_decoder
                    .as_deref()
                    .is_none_or(|label| decoder.map_or("j2k", ExternalDecoder::label) == label)
            })
            .collect::<Vec<_>>();
        let mut samples = vec![Vec::with_capacity(options.repeats); decoders.len()];
        let mut decoded_bytes = vec![0; decoders.len()];
        for decoder in &decoders {
            std::hint::black_box(decode_once(inputs, options.workers, *decoder)?);
        }
        for round in 0..options.repeats {
            for offset in 0..decoders.len() {
                let index = (round + offset) % decoders.len();
                if options.j2k_cold && decoders[index].is_none() {
                    release_tile_batch_workspaces();
                }
                let started = Instant::now();
                let bytes = decode_once(inputs, options.workers, decoders[index])?;
                samples[index].push(started.elapsed().as_secs_f64() * 1000.0);
                decoded_bytes[index] = std::hint::black_box(bytes);
            }
        }
        for ((decoder, samples), bytes) in decoders.into_iter().zip(samples).zip(decoded_bytes) {
            emit_measurement(measurement(
                decoder.map_or("j2k", ExternalDecoder::label),
                batch_size,
                options.repeats,
                samples,
                bytes,
            )?);
        }
        for decoder in ExternalDecoder::ALL.into_iter().filter(|d| {
            options.measured_decoder.is_none() && !d.is_available() && !options.excluded.contains(d)
        }) {
            println!(
                "{}\t{batch_size}\t{}\tNA\tNA\tNA\tNA\tunavailable",
                decoder.label(),
                options.repeats
            );
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
            bytes: Arc::from(bytes),
            dimensions: info.dimensions,
            format,
            roi: None,
            reduce: 0,
        });
    }
    Ok((tiles, skipped))
}

fn pixel_format(components: u16, bit_depth: u8) -> Option<PixelFormat> {
    match (components, bit_depth) {
        (1, 8) => Some(PixelFormat::Gray8),
        (3, 8) => Some(PixelFormat::Rgb8),
        // 9- to 16-bit grayscale (CT, MR, mammography, radiography): every
        // decoder returns each sample's unscaled value as a u16.
        (1, 9..=16) => Some(PixelFormat::Gray16),
        _ => None,
    }
}

fn decode_once(
    tiles: &[TileInput],
    workers: Option<NonZeroUsize>,
    decoder: Option<ExternalDecoder>,
) -> Result<usize, String> {
    match decoder {
        Some(decoder) => decode_external_once(tiles, workers, decoder),
        None => decode_j2k_once(tiles, tiles[0].format, workers),
    }
}

fn decode_j2k_once(
    tiles: &[TileInput],
    format: PixelFormat,
    workers: Option<NonZeroUsize>,
) -> Result<usize, String> {
    let outputs = decode_j2k_outputs(tiles, format, workers)?;
    std::hint::black_box(&outputs);
    Ok(outputs.iter().map(Vec::len).sum())
}

fn decode_j2k_outputs(
    tiles: &[TileInput],
    format: PixelFormat,
    workers: Option<NonZeroUsize>,
) -> Result<Vec<Vec<u8>>, String> {
    if tiles.iter().any(|tile| tile.roi.is_some()) {
        return decode_j2k_viewport(tiles, format, workers);
    }
    let mut outputs = tiles
        .iter()
        .map(|tile| vec![0_u8; output_len(tile, format)])
        .collect::<Vec<_>>();
    let mut jobs = tiles
        .iter()
        .zip(outputs.iter_mut())
        .map(|(tile, out)| TileDecodeJob {
            input: &tile.bytes,
            out: out.as_mut_slice(),
            stride: stride(tile, format),
        })
        .collect::<Vec<_>>();
    decode_tiles_into(&mut jobs, format, TileBatchOptions { workers })
        .map_err(|err| format!("j2k batch decode failed: {err}"))?;
    Ok(outputs)
}

/// Region requests through j2k's large-image API, scheduled exactly like the
/// external decoders: static chunks on scoped worker threads, a fresh decoder
/// (header parse) and scratch pool per request, each request on one thread
/// unless the batch is a single request, which gets the whole budget.
fn decode_j2k_viewport(
    tiles: &[TileInput],
    format: PixelFormat,
    workers: Option<NonZeroUsize>,
) -> Result<Vec<Vec<u8>>, String> {
    let available = std::thread::available_parallelism().map_or(1, NonZeroUsize::get);
    let worker_count =
        tile_batch_worker_count(tiles.len(), TileBatchOptions { workers }, available);
    let threads = intra_image_threads(tiles.len(), workers.map_or(available, NonZeroUsize::get));
    let parallelism = if threads > 1 {
        CpuDecodeParallelism::Auto
    } else {
        CpuDecodeParallelism::Serial
    };
    run_static_chunks(tiles, worker_count, |tile| {
        let roi = tile.roi.unwrap_or_else(|| Rect::full(tile.dimensions));
        let mut view = J2kDecoder::new(&tile.bytes).map_err(|err| format!("j2k decoder: {err}"))?;
        view.set_cpu_decode_parallelism(parallelism);
        let mut out = vec![0_u8; output_len(tile, format)];
        view.decode_region_scaled_pow2_into(
            &mut J2kScratchPool::new(),
            &mut out,
            stride(tile, format),
            format,
            roi,
            tile.reduce,
        )
        .map_err(|err| format!("j2k region decode failed: {err}"))?;
        Ok(out)
    })
}

/// Decode every tile with `decode_one` as every decoder here is scheduled:
/// static chunks, one per worker, on scoped threads, in input order. A single
/// chunk runs on the calling thread.
fn run_static_chunks(
    tiles: &[TileInput],
    worker_count: usize,
    decode_one: impl Fn(&TileInput) -> Result<Vec<u8>, String> + Sync,
) -> Result<Vec<Vec<u8>>, String> {
    let chunk_size = tiles.len().div_ceil(worker_count.max(1)).max(1);
    if chunk_size >= tiles.len() {
        return tiles.iter().map(&decode_one).collect();
    }
    std::thread::scope(|scope| {
        let handles = tiles
            .chunks(chunk_size)
            .map(|chunk| {
                let decode_one = &decode_one;
                scope.spawn(move || chunk.iter().map(decode_one).collect::<Result<Vec<_>, _>>())
            })
            .collect::<Vec<_>>();
        let mut outputs = Vec::with_capacity(tiles.len());
        for handle in handles {
            match handle.join() {
                Ok(chunk) => outputs.append(&mut chunk?),
                Err(payload) => std::panic::resume_unwind(payload),
            }
        }
        Ok(outputs)
    })
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
            let actual = decode_external_tile(tile, decoder, 1)?;
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

fn decode_external_once(
    tiles: &[TileInput],
    workers: Option<NonZeroUsize>,
    decoder: ExternalDecoder,
) -> Result<usize, String> {
    let available = std::thread::available_parallelism().map_or(1, NonZeroUsize::get);
    let worker_count =
        tile_batch_worker_count(tiles.len(), TileBatchOptions { workers }, available);
    let threads = intra_image_threads(tiles.len(), workers.map_or(available, NonZeroUsize::get));
    let outputs = run_static_chunks(tiles, worker_count, |tile| {
        decode_external_tile(tile, decoder, threads)
    })?;
    std::hint::black_box(&outputs);
    Ok(outputs.iter().map(Vec::len).sum())
}

/// A comparator's decode of a request into a caller-allocated buffer.
type DecodeInto = fn(&[u8], ExternalDecodeRequest, &mut [u8]) -> Result<(), String>;

/// Decode one input with `threads` decoder-internal threads; `OpenJPH`
/// always decodes on the calling thread.
///
/// Every decoder writes its final pixels once into a buffer allocated the way
/// the native path allocates its own (`vec![0; len]` inside the timed region):
/// `OpenJPEG`'s wrapper packs straight into such a buffer, and the C/C++
/// wrappers receive one to write into, with no intermediate copy.
fn decode_external_tile(
    tile: &TileInput,
    decoder: ExternalDecoder,
    threads: u32,
) -> Result<Vec<u8>, String> {
    let request = match tile.format {
        PixelFormat::Gray8 => ExternalDecodeRequest::gray(),
        PixelFormat::Rgb8 => ExternalDecodeRequest::rgb(),
        PixelFormat::Gray16 => ExternalDecodeRequest::gray16(),
        other => {
            return Err(format!(
                "{other:?} is not implemented for external comparator"
            ))
        }
    };
    let request = ExternalDecodeRequest {
        region: tile.roi,
        reduce: (tile.reduce > 0).then_some(u32::from(tile.reduce)),
        ..request.with_threads(threads)
    };
    let into = |decode: DecodeInto| {
        let mut out = vec![0_u8; output_len(tile, tile.format)];
        decode(&tile.bytes, request, &mut out).map(|()| out)
    };
    let result = match decoder {
        ExternalDecoder::OpenJpeg => openjpeg::decode_request(&tile.bytes, request),
        ExternalDecoder::Grok => into(grok::decode_into),
        ExternalDecoder::OpenHtj2k => into(openhtj2k::decode_into),
        ExternalDecoder::OpenJph => into(openjph::decode_into),
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

/// Output size: the requested region (or the whole image) scaled down,
/// covering partial pixels, as every decoder here produces it.
fn out_dims(tile: &TileInput) -> (u32, u32) {
    let rect = tile.roi.unwrap_or_else(|| Rect::full(tile.dimensions));
    let denom = 1_u32 << tile.reduce;
    let span = |start: u32, len: u32| (start + len).div_ceil(denom) - start / denom;
    (span(rect.x, rect.w), span(rect.y, rect.h))
}

fn stride(tile: &TileInput, format: PixelFormat) -> usize {
    out_dims(tile).0 as usize * format.bytes_per_pixel()
}

fn output_len(tile: &TileInput, format: PixelFormat) -> usize {
    stride(tile, format) * out_dims(tile).1 as usize
}

#[cfg(test)]
mod tests {
    use super::{
        check_rayon_threads, intra_image_threads, maximum_absolute_byte_difference, parse_excluded,
        ExternalDecoder,
    };
    use j2k_compare::{parse_positive_usize, sample_stats};

    #[test]
    fn a_lone_image_gets_the_whole_thread_budget_and_batches_stay_serial() {
        assert_eq!(intra_image_threads(1, 10), 10);
        assert_eq!(intra_image_threads(1, 1), 1);
        assert_eq!(intra_image_threads(2, 10), 1);
        assert_eq!(intra_image_threads(32, 10), 1);
        assert_eq!(intra_image_threads(1, 0), 1);
    }

    #[test]
    fn rayon_pool_must_match_the_thread_budget() {
        assert!(check_rayon_threads(10, Some("10")).is_ok());
        assert!(check_rayon_threads(1, Some(" 1 ")).is_ok());
        assert!(check_rayon_threads(10, None).is_err());
        assert!(check_rayon_threads(10, Some("1")).is_err());
        assert!(check_rayon_threads(10, Some("ten")).is_err());
    }

    #[test]
    fn excluded_decoders_parse_known_labels_and_reject_unknown_ones() {
        assert_eq!(
            parse_excluded("openjph, openhtj2k"),
            Ok(vec![ExternalDecoder::OpenJph, ExternalDecoder::OpenHtj2k])
        );
        assert_eq!(parse_excluded(""), Ok(vec![]));
        assert!(parse_excluded("kakadu").is_err());
    }

    #[test]
    fn only_openjph_lacks_intra_image_threads() {
        assert!(!ExternalDecoder::OpenJph.supports_intra_image_threads());
        assert!(ExternalDecoder::OpenJpeg.supports_intra_image_threads());
        assert!(ExternalDecoder::Grok.supports_intra_image_threads());
        assert!(ExternalDecoder::OpenHtj2k.supports_intra_image_threads());
    }

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
