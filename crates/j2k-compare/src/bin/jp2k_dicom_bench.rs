// SPDX-License-Identifier: MIT OR Apache-2.0

//! Native-bit-depth decode throughput for DICOM-style JPEG 2000 / HTJ2K frames.
//!
//! Usage: `jp2k_dicom_bench <file-or-dir> [...]`
//!
//! Each codestream is parsed and decoded to native component planes, the way a
//! DICOM viewer decodes one frame, first with serial code-block decode and then
//! with the default internal parallelism. A final row decodes every input
//! concurrently with one frame per worker, as a multi-frame series decode does.
//!
//! Environment:
//! - `J2K_DICOM_BENCH_REPEATS`: timed repeats per row (default 15).
//! - `J2K_DICOM_BENCH_RAW_DIR`: directory of `<stem>.RAW` little-endian
//!   reference samples; matching inputs must decode to identical bytes.
//! - `J2K_DICOM_BENCH_THREADS`: workers for the series row (default: all cores).
//! - `J2K_DICOM_BENCH_SERIES_REPEAT`: times each input repeats in the series row
//!   (default 8).
//! - `J2K_DICOM_BENCH_MODE`: `serial`, `parallel`, or `series` to time only one
//!   kind of row (for profiling); default runs all three.
//! - `J2K_DICOM_BENCH_REUSE_DECODER=1`: time repeated decodes through one
//!   decoder instead of parsing each frame into a new one.

use std::path::{Path, PathBuf};
use std::time::Instant;

use j2k::{CpuDecodeParallelism, J2kDecodedNativeComponents, J2kDecoder};
use j2k_compare::{parse_positive_usize, sample_stats};

struct Input {
    path: PathBuf,
    bytes: Vec<u8>,
    pixels: u64,
    label: String,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("{err}");
        std::process::exit(1);
    }
}

#[expect(
    clippy::cast_precision_loss,
    reason = "pixel counts are far below 2^52, so throughput ratios are exact enough"
)]
fn run() -> Result<(), String> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.is_empty() {
        return Err("usage: jp2k_dicom_bench <file-or-dir> [...]".to_string());
    }
    let repeats = env_usize("J2K_DICOM_BENCH_REPEATS")?.unwrap_or(15);
    let series_repeat = env_usize("J2K_DICOM_BENCH_SERIES_REPEAT")?.unwrap_or(8);
    let threads = env_usize("J2K_DICOM_BENCH_THREADS")?
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, usize::from));
    let raw_dir = std::env::var_os("J2K_DICOM_BENCH_RAW_DIR").map(PathBuf::from);

    let mode = std::env::var("J2K_DICOM_BENCH_MODE").unwrap_or_default();
    let (run_serial, run_parallel, run_series) = match mode.as_str() {
        "" => (true, true, true),
        "serial" => (true, false, false),
        "parallel" => (false, true, false),
        "series" => (false, false, true),
        other => return Err(format!("unknown J2K_DICOM_BENCH_MODE {other:?}")),
    };

    let inputs = load_inputs(&args)?;
    println!("file\tshape\tserial_ms\tparallel_ms\tserial_mpix_s\tparallel_mpix_s\tverified");
    for input in inputs.iter().filter(|_| run_serial || run_parallel) {
        let verified = verify(input, raw_dir.as_deref())?;
        let serial = if run_serial {
            time_decode(input, repeats, CpuDecodeParallelism::Serial)?
        } else {
            f64::NAN
        };
        let parallel = if run_parallel {
            time_decode(input, repeats, CpuDecodeParallelism::Auto)?
        } else {
            f64::NAN
        };
        let mpix = input.pixels as f64 / 1.0e6;
        println!(
            "{}\t{}\t{serial:.3}\t{parallel:.3}\t{:.1}\t{:.1}\t{verified}",
            input.path.file_name().unwrap_or_default().to_string_lossy(),
            input.label,
            mpix / (serial / 1000.0),
            mpix / (parallel / 1000.0),
        );
    }
    if run_series {
        series_row(&inputs, threads, series_repeat, repeats.min(5))?;
    }
    Ok(())
}

fn env_usize(name: &str) -> Result<Option<usize>, String> {
    std::env::var(name)
        .ok()
        .map(|value| parse_positive_usize(&value, name))
        .transpose()
}

fn load_inputs(args: &[String]) -> Result<Vec<Input>, String> {
    let mut paths = Vec::new();
    for arg in args {
        let path = PathBuf::from(arg);
        if path.is_dir() {
            let mut entries = std::fs::read_dir(&path)
                .map_err(|err| format!("read dir {}: {err}", path.display()))?
                .map(|entry| entry.map(|entry| entry.path()))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|err| format!("read dir entry: {err}"))?;
            entries.retain(|p| has_codestream_extension(p));
            entries.sort();
            paths.extend(entries);
        } else {
            paths.push(path);
        }
    }
    paths
        .into_iter()
        .map(|path| {
            let bytes =
                std::fs::read(&path).map_err(|err| format!("read {}: {err}", path.display()))?;
            let info = J2kDecoder::inspect(&bytes)
                .map_err(|err| format!("inspect {}: {err}", path.display()))?;
            let (w, h) = info.dimensions;
            Ok(Input {
                label: format!("{w}x{h}x{}@{}", info.components, info.bit_depth),
                pixels: u64::from(w) * u64::from(h),
                path,
                bytes,
            })
        })
        .collect()
}

fn has_codestream_extension(path: &Path) -> bool {
    path.extension()
        .and_then(|value| value.to_str())
        .is_some_and(|ext| {
            matches!(
                ext.to_ascii_lowercase().as_str(),
                "j2k" | "j2c" | "jp2" | "jph" | "jhc"
            )
        })
}

fn decode_once(
    bytes: &[u8],
    parallelism: CpuDecodeParallelism,
) -> Result<J2kDecodedNativeComponents, String> {
    let mut decoder = J2kDecoder::new(bytes).map_err(|err| format!("parse: {err}"))?;
    decoder.set_cpu_decode_parallelism(parallelism);
    decoder
        .decode_native_components()
        .map_err(|err| format!("decode: {err}"))
}

fn verify(input: &Input, raw_dir: Option<&Path>) -> Result<&'static str, String> {
    let Some(raw_dir) = raw_dir else {
        return Ok("no-reference");
    };
    let stem = input
        .path
        .file_stem()
        .unwrap_or_default()
        .to_string_lossy()
        .into_owned();
    let raw_path = raw_dir.join(format!("{stem}.RAW"));
    let Ok(reference) = std::fs::read(&raw_path) else {
        return Ok("no-reference");
    };
    let decoded = decode_once(&input.bytes, CpuDecodeParallelism::Auto)?;
    // Reference files store interleaved samples; single-plane inputs compare
    // directly and multi-plane inputs are interleaved first.
    let decoded = if let [plane] = decoded.planes() {
        plane.data().to_vec()
    } else {
        let planes = decoded
            .planes()
            .iter()
            .map(|plane| (plane.data(), usize::from(plane.bytes_per_sample())))
            .collect::<Vec<_>>();
        interleave(&planes)
    };
    if decoded == reference {
        Ok("exact")
    } else if decoded.len() == reference.len() {
        Err(format!(
            "{}: decoded bytes differ from {}",
            input.path.display(),
            raw_path.display()
        ))
    } else {
        Ok("reference-size-mismatch")
    }
}

fn interleave(planes: &[(&[u8], usize)]) -> Vec<u8> {
    let sample_count = planes
        .iter()
        .map(|(data, bytes_per_sample)| {
            if *bytes_per_sample == 0 {
                0
            } else {
                data.len() / bytes_per_sample
            }
        })
        .min()
        .unwrap_or(0);
    let bytes_per_pixel = planes
        .iter()
        .map(|(_, bytes_per_sample)| bytes_per_sample)
        .sum::<usize>();
    let mut out = Vec::with_capacity(sample_count * bytes_per_pixel);
    for sample_index in 0..sample_count {
        for &(data, bytes_per_sample) in planes {
            let start = sample_index * bytes_per_sample;
            out.extend_from_slice(&data[start..start + bytes_per_sample]);
        }
    }
    out
}

fn time_decode(
    input: &Input,
    repeats: usize,
    parallelism: CpuDecodeParallelism,
) -> Result<f64, String> {
    decode_once(&input.bytes, parallelism)?;
    let mut samples = Vec::with_capacity(repeats);
    if std::env::var_os("J2K_DICOM_BENCH_REUSE_DECODER").is_some() {
        let mut decoder = J2kDecoder::new(&input.bytes).map_err(|err| format!("parse: {err}"))?;
        decoder.set_cpu_decode_parallelism(parallelism);
        decoder
            .decode_native_components()
            .map_err(|err| format!("decode: {err}"))?;
        for _ in 0..repeats {
            let start = Instant::now();
            let planes = decoder
                .decode_native_components()
                .map_err(|err| format!("decode: {err}"))?;
            samples.push(start.elapsed().as_secs_f64() * 1000.0);
            std::hint::black_box(planes);
        }
        return Ok(sample_stats(&samples)?.median);
    }
    for _ in 0..repeats {
        let start = Instant::now();
        let planes = decode_once(&input.bytes, parallelism)?;
        samples.push(start.elapsed().as_secs_f64() * 1000.0);
        std::hint::black_box(planes);
    }
    Ok(sample_stats(&samples)?.median)
}

#[expect(
    clippy::cast_precision_loss,
    reason = "pixel counts are far below 2^52, so throughput ratios are exact enough"
)]
fn series_row(
    inputs: &[Input],
    threads: usize,
    series_repeat: usize,
    repeats: usize,
) -> Result<(), String> {
    let jobs = inputs
        .iter()
        .flat_map(|input| std::iter::repeat_n(input, series_repeat))
        .collect::<Vec<_>>();
    let total_pixels = jobs.iter().map(|input| input.pixels).sum::<u64>();
    let mut samples = Vec::with_capacity(repeats);
    for _ in 0..=repeats {
        let next = std::sync::atomic::AtomicUsize::new(0);
        let start = Instant::now();
        std::thread::scope(|scope| -> Result<(), String> {
            let workers = (0..threads)
                .map(|_| {
                    scope.spawn(|| -> Result<(), String> {
                        loop {
                            let index = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            let Some(input) = jobs.get(index) else {
                                return Ok(());
                            };
                            std::hint::black_box(decode_once(
                                &input.bytes,
                                CpuDecodeParallelism::Serial,
                            )?);
                        }
                    })
                })
                .collect::<Vec<_>>();
            for worker in workers {
                worker
                    .join()
                    .map_err(|_| "series worker panicked".to_string())??;
            }
            Ok(())
        })?;
        samples.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    samples.remove(0);
    let median = sample_stats(&samples)?.median;
    println!(
        "series\t{} frames x{threads} workers\t{median:.3}\t-\t{:.1}\t-\t-",
        jobs.len(),
        total_pixels as f64 / 1.0e6 / (median / 1000.0)
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::interleave;

    #[test]
    fn interleave_keeps_wide_rgb_samples_together() {
        let red = [1, 2, 3, 4];
        let green = [11, 12, 13, 14];
        let blue = [21, 22, 23, 24];

        assert_eq!(
            interleave(&[(&red, 2), (&green, 2), (&blue, 2)]),
            [1, 2, 11, 12, 21, 22, 3, 4, 13, 14, 23, 24]
        );
    }
}
