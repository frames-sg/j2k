// SPDX-License-Identifier: MIT OR Apache-2.0

//! Compare CPU and Metal decode on DICOM-style JPEG 2000 / HTJ2K frames.
//!
//! For every codestream this times a single-image CPU decode, a single-image
//! Metal decode into a device surface, and CPU and Metal owned-batch decodes
//! of `J2K_METAL_BENCH_BATCH` copies (default 16). Metal output is read back
//! once, outside the timers, and must equal the CPU output byte for byte.
//! `J2K_METAL_BENCH_ONLY=metal-single` times only the single-image Metal
//! decode, for profiling it; `J2K_METAL_BENCH_ONLY=auto` reports the backend
//! `BackendRequest::Auto` selects for each file and times that decode.
//! `J2K_METAL_BENCH_ONLY=metal-into` times one reused decoder decoding into a
//! fresh surface against decoding into one reused caller-owned buffer, with
//! the two measurements interleaved.
//!
//! Run with:
//! `cargo run --release -p j2k-metal --example dicom_decode_bench -- <file-or-dir> [...]`

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("dicom_decode_bench requires macOS");
}

#[cfg(target_os = "macos")]
fn main() {
    if let Err(error) = bench::run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

#[cfg(target_os = "macos")]
mod bench {
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use std::time::Instant;

    use j2k::{
        BackendRequest, BatchDecodeOptions, BatchLayout, CpuBatchDecoder, CpuBatchSamples,
        DecodeRequest, EncodedImage, PixelFormat,
    };
    use j2k_metal::{
        J2kDecoder, MetalBackendSession, MetalBatchDecoder, MetalDecodeRequest,
        MetalImageDestination, MetalImageLayout,
    };

    pub(super) fn run() -> Result<(), String> {
        let paths = collect_paths(std::env::args().skip(1))?;
        if paths.is_empty() {
            return Err("usage: dicom_decode_bench <file-or-dir> [...]".to_string());
        }
        let repeats = env_usize("J2K_METAL_BENCH_REPEATS", 11)?;
        let batch = env_usize("J2K_METAL_BENCH_BATCH", 16)?;
        match std::env::var("J2K_METAL_BENCH_ONLY").as_deref() {
            Ok("metal-single") => return run_metal_single_only(&paths, repeats),
            Ok("auto") => return run_auto_only(&paths, repeats),
            Ok("metal-into") => return run_metal_into_only(&paths, repeats),
            _ => {}
        }
        let options = BatchDecodeOptions {
            layout: BatchLayout::Nhwc,
            ..BatchDecodeOptions::default()
        };
        let mut metal_batch = MetalBatchDecoder::system_default_with_options(options)
            .map_err(|error| format!("Metal batch decoder: {error}"))?;
        let mut cpu_batch = CpuBatchDecoder::new(options);

        println!(
            "file\tformat\tcpu_ms\tmetal_ms\tcpu_batch{batch}_ms\tmetal_batch{batch}_ms\tsingle\tbatch"
        );
        for path in paths {
            let bytes: Arc<[u8]> = std::fs::read(&path)
                .map_err(|error| format!("read {}: {error}", path.display()))?
                .into();
            let name = path.file_name().unwrap_or_default().to_string_lossy();
            let info = j2k::J2kDecoder::inspect(&bytes)
                .map_err(|error| format!("{name}: inspect: {error}"))?;
            let Some(format) = pixel_format(info.components, info.bit_depth) else {
                println!("{name}\tunsupported\t-\t-\t-\t-\t-\t-");
                continue;
            };

            let (cpu_ms, cpu_pixels) = time_cpu_single(&bytes, format, repeats)?;
            let (metal_ms, single_parity) = match time_metal_single(&bytes, format, repeats) {
                Ok((ms, pixels)) => (format!("{ms:.3}"), parity(&pixels, &cpu_pixels)),
                Err(error) => ("-".to_string(), format!("error: {error}")),
            };

            let inputs = || {
                (0..batch)
                    .map(|_| EncodedImage {
                        bytes: Arc::clone(&bytes),
                        request: DecodeRequest::Full,
                    })
                    .collect::<Vec<_>>()
            };
            let (cpu_batch_ms, cpu_batch_bytes) =
                time_cpu_batch(&mut cpu_batch, inputs(), repeats)?;
            let (metal_batch_ms, batch_parity) =
                match time_metal_batch(&mut metal_batch, inputs(), repeats) {
                    Ok((ms, bytes)) => (format!("{ms:.3}"), parity(&bytes, &cpu_batch_bytes)),
                    Err(error) => ("-".to_string(), format!("error: {error}")),
                };
            println!(
                "{name}\t{format:?}\t{cpu_ms:.3}\t{metal_ms}\t{cpu_batch_ms:.3}\t{metal_batch_ms}\t{single_parity}\t{batch_parity}"
            );
        }
        Ok(())
    }

    fn run_metal_single_only(paths: &[PathBuf], repeats: usize) -> Result<(), String> {
        println!("file\tformat\tmetal_ms");
        for path in paths {
            let bytes =
                std::fs::read(path).map_err(|error| format!("read {}: {error}", path.display()))?;
            let name = path.file_name().unwrap_or_default().to_string_lossy();
            let info = j2k::J2kDecoder::inspect(&bytes)
                .map_err(|error| format!("{name}: inspect: {error}"))?;
            let Some(format) = pixel_format(info.components, info.bit_depth) else {
                continue;
            };
            let metal_ms = if std::env::var_os("J2K_METAL_BENCH_REUSE").is_some() {
                // One decoder: its prepared plan and device payloads are reused.
                let mut decoder = J2kDecoder::new(&bytes).map_err(|error| error.to_string())?;
                let request = || MetalDecodeRequest::full(format, BackendRequest::Metal);
                decoder
                    .decode_request_to_device(request())
                    .map_err(|error| error.to_string())?;
                let mut samples = Vec::with_capacity(repeats);
                for _ in 0..repeats {
                    let start = Instant::now();
                    std::hint::black_box(
                        decoder
                            .decode_request_to_device(request())
                            .map_err(|error| error.to_string())?,
                    );
                    samples.push(start.elapsed().as_secs_f64() * 1000.0);
                }
                median(samples)
            } else {
                time_metal_single(&bytes, format, repeats)?.0
            };
            println!("{name}\t{format:?}\t{metal_ms:.3}");
        }
        Ok(())
    }

    fn run_metal_into_only(paths: &[PathBuf], repeats: usize) -> Result<(), String> {
        let session = MetalBackendSession::system_default().map_err(|error| error.to_string())?;
        println!("file\tformat\tsurface_ms\tinto_ms\tparity");
        for path in paths {
            let bytes =
                std::fs::read(path).map_err(|error| format!("read {}: {error}", path.display()))?;
            let name = path.file_name().unwrap_or_default().to_string_lossy();
            let info = j2k::J2kDecoder::inspect(&bytes)
                .map_err(|error| format!("{name}: inspect: {error}"))?;
            let Some(format) = pixel_format(info.components, info.bit_depth) else {
                continue;
            };
            let dimensions = info.dimensions;
            let pitch = usize::try_from(dimensions.0).map_err(|error| error.to_string())?
                * format.bytes_per_pixel();
            let len = pitch * usize::try_from(dimensions.1).map_err(|error| error.to_string())?;
            let buffer =
                j2k_metal_support::checked_shared_buffer_for_len::<u8>(session.device(), len)
                    .map_err(|error| error.to_string())?;
            let layout = MetalImageLayout::new(0, dimensions, pitch, format)
                .map_err(|error| error.to_string())?;
            // SAFETY: every decode below completes before `decode_into_metal`
            // returns, and the buffer is read only between decodes.
            let destination =
                unsafe { MetalImageDestination::from_exclusive_buffer(buffer.clone(), layout) }
                    .map_err(|error| error.to_string())?;

            let mut decoder = J2kDecoder::new(&bytes).map_err(|error| error.to_string())?;
            let request = || MetalDecodeRequest::full(format, BackendRequest::Metal);
            let surface = decoder
                .decode_request_to_device_with_session(request(), &session)
                .map_err(|error| error.to_string())?;
            let expected = surface
                .as_bytes()
                .map_err(|error| error.to_string())?
                .into_owned();
            drop(surface);
            decoder
                .decode_into_metal_with_session(&destination, &session)
                .map_err(|error| error.to_string())?;
            // SAFETY: the decode above has completed and none is in flight.
            let written =
                unsafe { j2k_metal_support::checked_buffer_read_vec::<u8>(&buffer, 0, len) }
                    .map_err(|error| error.to_string())?;

            let mut surface_samples = Vec::with_capacity(repeats);
            let mut into_samples = Vec::with_capacity(repeats);
            for _ in 0..repeats {
                let start = Instant::now();
                std::hint::black_box(
                    decoder
                        .decode_request_to_device_with_session(request(), &session)
                        .map_err(|error| error.to_string())?,
                );
                surface_samples.push(start.elapsed().as_secs_f64() * 1000.0);
                let start = Instant::now();
                decoder
                    .decode_into_metal_with_session(&destination, &session)
                    .map_err(|error| error.to_string())?;
                into_samples.push(start.elapsed().as_secs_f64() * 1000.0);
            }
            println!(
                "{name}\t{format:?}\t{:.3}\t{:.3}\t{}",
                median(surface_samples),
                median(into_samples),
                parity(&written, &expected)
            );
        }
        Ok(())
    }

    fn run_auto_only(paths: &[PathBuf], repeats: usize) -> Result<(), String> {
        println!("file\tformat\tauto_backend\tauto_ms\tcpu_ms\tmetal_ms");
        for path in paths {
            let bytes =
                std::fs::read(path).map_err(|error| format!("read {}: {error}", path.display()))?;
            let name = path.file_name().unwrap_or_default().to_string_lossy();
            let info = j2k::J2kDecoder::inspect(&bytes)
                .map_err(|error| format!("{name}: inspect: {error}"))?;
            let Some(format) = pixel_format(info.components, info.bit_depth) else {
                continue;
            };
            let decode_auto = || -> Result<j2k_metal::Surface, String> {
                let mut decoder = J2kDecoder::new(&bytes).map_err(|error| error.to_string())?;
                decoder
                    .decode_request_to_device(MetalDecodeRequest::full(
                        format,
                        BackendRequest::Auto,
                    ))
                    .map_err(|error| error.to_string())
            };
            let report = J2kDecoder::new(&bytes)
                .map_err(|error| error.to_string())?
                .decode_request_to_device_with_report(MetalDecodeRequest::full(
                    format,
                    BackendRequest::Auto,
                ))
                .map_err(|error| error.to_string())?
                .report;
            let mut samples = Vec::with_capacity(repeats);
            decode_auto()?;
            for _ in 0..repeats {
                let start = Instant::now();
                std::hint::black_box(decode_auto()?);
                samples.push(start.elapsed().as_secs_f64() * 1000.0);
            }
            let (cpu_ms, _) = time_cpu_single(&bytes, format, repeats)?;
            let (metal_ms, _) = time_metal_single(&bytes, format, repeats)?;
            println!(
                "{name}\t{format:?}\t{:?}\t{:.3}\t{cpu_ms:.3}\t{metal_ms:.3}",
                report.selected_backend,
                median(samples)
            );
        }
        Ok(())
    }

    fn collect_paths(args: impl Iterator<Item = String>) -> Result<Vec<PathBuf>, String> {
        let mut paths = Vec::new();
        for arg in args {
            let path = PathBuf::from(arg);
            if path.is_dir() {
                let mut entries = std::fs::read_dir(&path)
                    .map_err(|error| format!("read dir {}: {error}", path.display()))?
                    .map(|entry| entry.map(|entry| entry.path()))
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|error| format!("read dir entry: {error}"))?;
                entries.retain(|entry| is_codestream(entry));
                entries.sort();
                paths.extend(entries);
            } else {
                paths.push(path);
            }
        }
        Ok(paths)
    }

    fn is_codestream(path: &Path) -> bool {
        path.extension()
            .and_then(|ext| ext.to_str())
            .is_some_and(|ext| {
                matches!(
                    ext.to_ascii_lowercase().as_str(),
                    "j2k" | "j2c" | "jp2" | "jph" | "jhc"
                )
            })
    }

    fn env_usize(name: &str, default: usize) -> Result<usize, String> {
        std::env::var(name).map_or(Ok(default), |value| {
            value
                .parse::<usize>()
                .ok()
                .filter(|value| *value > 0)
                .ok_or_else(|| format!("{name} must be a positive integer"))
        })
    }

    fn pixel_format(components: u16, bit_depth: u8) -> Option<PixelFormat> {
        match (components, bit_depth) {
            (1, 1..=8) => Some(PixelFormat::Gray8),
            (1, 9..=16) => Some(PixelFormat::Gray16),
            (3, 1..=8) => Some(PixelFormat::Rgb8),
            (3, 9..=16) => Some(PixelFormat::Rgb16),
            _ => None,
        }
    }

    fn parity(actual: &[u8], expected: &[u8]) -> String {
        if actual == expected {
            "exact".to_string()
        } else if actual.len() != expected.len() {
            format!("length {} != {}", actual.len(), expected.len())
        } else {
            let differing = actual.iter().zip(expected).filter(|(a, b)| a != b).count();
            format!("{differing} bytes differ")
        }
    }

    fn median(mut samples: Vec<f64>) -> f64 {
        samples.sort_by(f64::total_cmp);
        samples[samples.len() / 2]
    }

    fn time_cpu_single(
        bytes: &[u8],
        format: PixelFormat,
        repeats: usize,
    ) -> Result<(f64, Vec<u8>), String> {
        let decode = || -> Result<Vec<u8>, String> {
            let mut decoder =
                j2k::J2kDecoder::new(bytes).map_err(|error| format!("CPU parse: {error}"))?;
            let (width, height) = decoder.info().dimensions;
            let stride = width as usize * format.bytes_per_pixel();
            let mut out = vec![0_u8; stride * height as usize];
            decoder
                .decode_into(&mut out, stride, format)
                .map_err(|error| format!("CPU decode: {error}"))?;
            Ok(out)
        };
        let pixels = decode()?;
        let mut samples = Vec::with_capacity(repeats);
        for _ in 0..repeats {
            let start = Instant::now();
            std::hint::black_box(decode()?);
            samples.push(start.elapsed().as_secs_f64() * 1000.0);
        }
        Ok((median(samples), pixels))
    }

    fn time_metal_single(
        bytes: &[u8],
        format: PixelFormat,
        repeats: usize,
    ) -> Result<(f64, Vec<u8>), String> {
        let decode = || -> Result<j2k_metal::Surface, String> {
            let mut decoder = J2kDecoder::new(bytes).map_err(|error| error.to_string())?;
            decoder
                .decode_request_to_device(MetalDecodeRequest::full(format, BackendRequest::Metal))
                .map_err(|error| error.to_string())
        };
        let pixels = decode()?
            .as_bytes()
            .map_err(|error| error.to_string())?
            .to_vec();
        let mut samples = Vec::with_capacity(repeats);
        for _ in 0..repeats {
            let start = Instant::now();
            std::hint::black_box(decode()?);
            samples.push(start.elapsed().as_secs_f64() * 1000.0);
        }
        Ok((median(samples), pixels))
    }

    fn cpu_samples_bytes(samples: &CpuBatchSamples) -> Vec<u8> {
        match samples {
            CpuBatchSamples::U8(values) => values.clone(),
            CpuBatchSamples::U16(values) => values.iter().flat_map(|v| v.to_le_bytes()).collect(),
            CpuBatchSamples::I16(values) => values.iter().flat_map(|v| v.to_le_bytes()).collect(),
            _ => Vec::new(),
        }
    }

    fn time_cpu_batch(
        decoder: &mut CpuBatchDecoder,
        inputs: Vec<EncodedImage>,
        repeats: usize,
    ) -> Result<(f64, Vec<u8>), String> {
        let prepared = decoder
            .prepare(inputs)
            .map_err(|error| format!("CPU batch prepare: {error}"))?;
        let result = decoder
            .decode_prepared(&prepared)
            .map_err(|error| format!("CPU batch decode: {error}"))?;
        if !result.errors().is_empty() {
            return Err(format!("CPU batch errors: {:?}", result.errors()));
        }
        let bytes = result
            .groups()
            .iter()
            .flat_map(|group| cpu_samples_bytes(group.samples()))
            .collect();
        let mut samples = Vec::with_capacity(repeats);
        for _ in 0..repeats {
            let start = Instant::now();
            std::hint::black_box(
                decoder
                    .decode_prepared(&prepared)
                    .map_err(|error| format!("CPU batch decode: {error}"))?,
            );
            samples.push(start.elapsed().as_secs_f64() * 1000.0);
        }
        Ok((median(samples), bytes))
    }

    fn time_metal_batch(
        decoder: &mut MetalBatchDecoder,
        inputs: Vec<EncodedImage>,
        repeats: usize,
    ) -> Result<(f64, Vec<u8>), String> {
        let prepared = decoder.prepare(inputs).map_err(|error| error.to_string())?;
        let result = decoder
            .decode_prepared(&prepared)
            .map_err(|error| error.to_string())?;
        if !result.errors().is_empty() || !result.group_errors().is_empty() {
            return Err(format!(
                "Metal batch errors: {:?} {:?}",
                result.errors(),
                result.group_errors()
            ));
        }
        let mut bytes = Vec::new();
        for group in result.groups() {
            for surface in group.surfaces() {
                bytes.extend_from_slice(&surface.as_bytes().map_err(|error| error.to_string())?);
            }
        }
        let mut samples = Vec::with_capacity(repeats);
        for _ in 0..repeats {
            let start = Instant::now();
            std::hint::black_box(
                decoder
                    .decode_prepared(&prepared)
                    .map_err(|error| error.to_string())?,
            );
            samples.push(start.elapsed().as_secs_f64() * 1000.0);
        }
        Ok((median(samples), bytes))
    }
}
