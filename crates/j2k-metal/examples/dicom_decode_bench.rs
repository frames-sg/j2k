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
//! the two measurements interleaved; with `J2K_METAL_BENCH_CAPTURE` set to an
//! absolute `.gputrace` path (and `MTL_CAPTURE_ENABLED=1`), one warmed-up
//! decode of the first file into the buffer is also captured there, for
//! Xcode's GPU profiler. `J2K_METAL_BENCH_ONLY=metal-batch` times
//! only the Metal owned-batch decode; with `J2K_METAL_BENCH_DISTINCT` its batch
//! holds distinct frames (the decoded image with a per-frame offset, re-encoded
//! as lossless HTJ2K) instead of copies of one codestream, which share one
//! decode.
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
            Ok("metal-batch") => return run_metal_batch_only(&paths, repeats, batch),
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

    fn run_metal_batch_only(paths: &[PathBuf], repeats: usize, batch: usize) -> Result<(), String> {
        let options = BatchDecodeOptions {
            layout: BatchLayout::Nhwc,
            ..BatchDecodeOptions::default()
        };
        let mut decoder = MetalBatchDecoder::system_default_with_options(options)
            .map_err(|error| format!("Metal batch decoder: {error}"))?;
        let session = MetalBackendSession::system_default().map_err(|error| error.to_string())?;
        println!("file\towned_batch{batch}_ms\tinto_batch{batch}_ms");
        for path in paths {
            let bytes: Arc<[u8]> = std::fs::read(path)
                .map_err(|error| format!("read {}: {error}", path.display()))?
                .into();
            let name = path.file_name().unwrap_or_default().to_string_lossy();
            let info = j2k::J2kDecoder::inspect(&bytes)
                .map_err(|error| format!("{name}: inspect: {error}"))?;
            let Some(format) = pixel_format(info.components, info.bit_depth) else {
                continue;
            };
            let inputs = if std::env::var_os("J2K_METAL_BENCH_DISTINCT").is_some() {
                distinct_frames(&bytes, &info, format, batch)?
            } else {
                (0..batch)
                    .map(|_| EncodedImage {
                        bytes: Arc::clone(&bytes),
                        request: DecodeRequest::Full,
                    })
                    .collect()
            };
            let prepared = decoder.prepare(inputs).map_err(|error| error.to_string())?;
            let [group] = prepared.groups() else {
                return Err(format!("{name}: expected one batch group"));
            };
            let dimensions = group.info().dimensions;
            let pitch = dimensions.0 as usize * format.bytes_per_pixel();
            let image_bytes = pitch * dimensions.1 as usize;
            let buffer = j2k_metal_support::checked_shared_buffer_for_len::<u8>(
                session.device(),
                image_bytes * batch,
            )
            .map_err(|error| error.to_string())?;
            let layout =
                MetalImageLayout::new_batch(0, dimensions, pitch, format, batch, image_bytes)
                    .map_err(|error| error.to_string())?;
            // SAFETY: each decode into the buffer completes before the call
            // returns, and nothing else accesses it.
            let destination =
                unsafe { MetalImageDestination::from_exclusive_buffer(buffer, layout) }
                    .map_err(|error| error.to_string())?;
            let into_supported = match decoder.decode_prepared_group_into(group, &destination) {
                Ok(()) => true,
                Err(error) => {
                    eprintln!("{name}: decode into a reused buffer unavailable: {error}");
                    false
                }
            };
            let checked = decoder
                .decode_prepared(&prepared)
                .map_err(|error| error.to_string())?;
            if !checked.errors().is_empty() || !checked.group_errors().is_empty() {
                return Err(format!(
                    "{name}: Metal batch errors: {:?} {:?}",
                    checked.errors(),
                    checked.group_errors()
                ));
            }
            drop(checked);
            std::hint::black_box(
                decoder
                    .decode_prepared(&prepared)
                    .map_err(|error| error.to_string())?,
            );
            let mut owned = Vec::with_capacity(repeats);
            let mut into_samples = Vec::with_capacity(repeats);
            for _ in 0..repeats {
                let start = Instant::now();
                std::hint::black_box(
                    decoder
                        .decode_prepared(&prepared)
                        .map_err(|error| error.to_string())?,
                );
                owned.push(start.elapsed().as_secs_f64() * 1000.0);
                if into_supported {
                    let start = Instant::now();
                    decoder
                        .decode_prepared_group_into(group, &destination)
                        .map_err(|error| error.to_string())?;
                    into_samples.push(start.elapsed().as_secs_f64() * 1000.0);
                }
            }
            let into_ms = if into_samples.is_empty() {
                "-".to_string()
            } else {
                format!("{:.3}", median(into_samples))
            };
            println!("{name}\t{:.3}\t{into_ms}", median(owned));
        }
        Ok(())
    }

    /// `count` lossless HTJ2K frames of the image in `bytes`, each with its
    /// samples offset by a different amount.
    fn distinct_frames(
        bytes: &[u8],
        info: &j2k_core::Info,
        format: PixelFormat,
        count: usize,
    ) -> Result<Vec<EncodedImage>, String> {
        let (width, height) = info.dimensions;
        let row_bytes = width as usize * format.bytes_per_pixel();
        let mut pixels = vec![0; row_bytes * height as usize];
        j2k::J2kDecoder::new(bytes)
            .map_err(|error| error.to_string())?
            .decode_into(&mut pixels, row_bytes, format)
            .map_err(|error| error.to_string())?;
        let bit_depth = info.bit_depth;
        let mask = u32::try_from((1_u64 << bit_depth) - 1).map_err(|error| error.to_string())?;
        let options = j2k_native::EncodeOptions {
            num_decomposition_levels: info.resolution_levels.saturating_sub(1).max(1),
            ..j2k_native::EncodeOptions::default()
        };
        (0..count)
            .map(|frame| {
                let offset = u32::try_from(frame * 7).map_err(|error| error.to_string())?;
                let frame_pixels: Vec<u8> = if format.bytes_per_sample() == 1 {
                    pixels
                        .iter()
                        .map(|&value| ((u32::from(value) + offset) & mask).to_le_bytes()[0])
                        .collect()
                } else {
                    pixels
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .flat_map(|&sample| {
                            let value = u32::from(u16::from_le_bytes(sample));
                            let shifted =
                                u16::try_from((value + offset) & mask).unwrap_or(u16::MAX);
                            shifted.to_le_bytes()
                        })
                        .collect()
                };
                let encoded = j2k_native::encode_htj2k(
                    &frame_pixels,
                    width,
                    height,
                    info.components,
                    bit_depth,
                    false,
                    &options,
                )
                .map_err(|error| format!("encode frame {frame}: {error}"))?;
                Ok(EncodedImage {
                    bytes: encoded.into(),
                    request: DecodeRequest::Full,
                })
            })
            .collect()
    }

    fn run_metal_into_only(paths: &[PathBuf], repeats: usize) -> Result<(), String> {
        let session = MetalBackendSession::system_default().map_err(|error| error.to_string())?;
        let mut capture_path = std::env::var_os("J2K_METAL_BENCH_CAPTURE").map(PathBuf::from);
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
            if let Some(trace) = capture_path.take() {
                capture_gpu_trace(session.device(), &trace, || {
                    decoder
                        .decode_into_metal_with_session(&destination, &session)
                        .map_err(|error| error.to_string())
                })?;
                eprintln!("{name}: captured one decode into {}", trace.display());
            }

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

    /// Runs `decode` inside a Metal GPU capture of `device`, written to the
    /// `.gputrace` document at `trace` for Xcode to open and profile.
    fn capture_gpu_trace(
        device: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDevice>,
        trace: &Path,
        decode: impl FnOnce() -> Result<(), String>,
    ) -> Result<(), String> {
        use objc2::runtime::AnyObject;
        use objc2_foundation::{NSString, NSURL};
        use objc2_metal::{MTLCaptureDescriptor, MTLCaptureDestination, MTLCaptureManager};

        if std::env::var("MTL_CAPTURE_ENABLED").as_deref() != Ok("1") {
            return Err("J2K_METAL_BENCH_CAPTURE needs MTL_CAPTURE_ENABLED=1".to_string());
        }
        if !trace.is_absolute()
            || trace.extension().and_then(std::ffi::OsStr::to_str) != Some("gputrace")
        {
            return Err(format!(
                "J2K_METAL_BENCH_CAPTURE must be an absolute .gputrace path: {}",
                trace.display()
            ));
        }
        if trace.exists() {
            return Err(format!(
                "refusing to overwrite existing GPU trace {}",
                trace.display()
            ));
        }
        // SAFETY: `sharedCaptureManager` returns Metal's process-wide
        // singleton, retained for as long as `manager` lives.
        let manager = unsafe { MTLCaptureManager::sharedCaptureManager() };
        if !manager.supportsDestination(MTLCaptureDestination::GPUTraceDocument) {
            return Err("Metal GPU trace documents are unavailable on this host".to_string());
        }
        let descriptor = MTLCaptureDescriptor::new();
        let capture_object: &AnyObject = device.as_ref();
        // SAFETY: a capture descriptor accepts an MTLDevice as its capture
        // object, and `device` outlives the capture.
        unsafe { descriptor.setCaptureObject(Some(capture_object)) };
        descriptor.setDestination(MTLCaptureDestination::GPUTraceDocument);
        let url = NSURL::fileURLWithPath(&NSString::from_str(&trace.to_string_lossy()));
        descriptor.setOutputURL(Some(&url));
        manager
            .startCaptureWithDescriptor_error(&descriptor)
            .map_err(|error| format!("start GPU capture: {error}"))?;
        let result = decode();
        manager.stopCapture();
        result
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
