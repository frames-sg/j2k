// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::similar_names)]

use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use j2k_core::{BackendKind, BackendRequest, DeviceSurface, Downscale, PixelFormat, Rect};
use j2k_metal::{DecodeOperation, J2kDecoder, MetalDecodeRequest, SurfaceResidency};
use j2k_native::{encode, encode_htj2k, EncodeOptions};
use j2k_test_support::{
    fnv1a64_hex, manifest_column, manifest_field, manifest_optional_value,
    optional_manifest_column, patterned_gray8, patterned_rgb8,
};

const GENERATED_DIMS: &[u32] = &[512, 1024];
const ITERS: usize = 5;
const METAL_DECODE_INPUT_DIRS_ENV: &str = "J2K_METAL_DECODE_INPUT_DIRS";
const METAL_DECODE_MANIFEST_ENV: &str = "J2K_METAL_DECODE_MANIFEST";
const METAL_DECODE_INCLUDE_GENERATED_ENV: &str = "J2K_METAL_DECODE_INCLUDE_GENERATED";
const REQUIRE_METAL_BENCH_ENV: &str = "J2K_REQUIRE_METAL_BENCH";
const METAL_DECODE_MANIFEST_LABEL: &str = "Metal decode manifest";

#[derive(Clone)]
struct DecodeBenchCase {
    id: String,
    bytes: Vec<u8>,
    fmt: PixelFormat,
    codec: String,
    container: String,
    source: String,
}

struct DecodeManifest {
    entries: HashMap<PathBuf, DecodeManifestEntry>,
}

struct DecodeManifestEntry {
    input_fnv1a64: String,
    codec: Option<String>,
    container: Option<String>,
}

struct TimedOutput<T> {
    duration: Duration,
    value: T,
}

type DecodeTimingTriplet = (TimedOutput<usize>, TimedOutput<usize>, TimedOutput<usize>);

#[test]
#[ignore = "benchmark harness; run explicitly with --ignored --nocapture"]
fn metal_decode_benchmark() {
    if !cfg!(target_os = "macos") {
        assert!(
            !env_truthy(REQUIRE_METAL_BENCH_ENV),
            "J2K Metal decode benchmark requires macOS"
        );
        emit_metadata(0, 0);
        println!("j2k_metal_decode_status\tskipped-not-macos");
        return;
    }

    let generated_cases = include_generated_decode_cases()
        .then(generated_decode_cases)
        .unwrap_or_default();
    let external_cases = external_decode_cases().unwrap_or_else(|error| panic!("{error}"));
    emit_metadata(generated_cases.len(), external_cases.len());

    for case in generated_cases.iter().chain(external_cases.iter()) {
        run_decode_case(case);
    }
}

fn generated_decode_cases() -> Vec<DecodeBenchCase> {
    let mut cases = Vec::new();
    for &dim in GENERATED_DIMS {
        let gray = patterned_gray8(dim, dim);
        cases.push(DecodeBenchCase {
            id: format!("generated_classic_gray8_{dim}"),
            bytes: encode_classic(&gray, dim, dim, 1),
            fmt: PixelFormat::Gray8,
            codec: "j2k".to_string(),
            container: "raw-codestream".to_string(),
            source: "generated".to_string(),
        });
        cases.push(DecodeBenchCase {
            id: format!("generated_htj2k_gray8_{dim}"),
            bytes: encode_ht(&gray, dim, dim, 1),
            fmt: PixelFormat::Gray8,
            codec: "htj2k".to_string(),
            container: "raw-codestream".to_string(),
            source: "generated".to_string(),
        });
        let rgb = patterned_rgb8(dim, dim);
        cases.push(DecodeBenchCase {
            id: format!("generated_classic_rgb8_{dim}"),
            bytes: encode_classic(&rgb, dim, dim, 3),
            fmt: PixelFormat::Rgb8,
            codec: "j2k".to_string(),
            container: "raw-codestream".to_string(),
            source: "generated".to_string(),
        });
    }
    cases
}

fn encode_classic(pixels: &[u8], width: u32, height: u32, components: u16) -> Vec<u8> {
    let options = encode_options();
    encode(pixels, width, height, components, 8, false, &options).expect("encode classic fixture")
}

fn encode_ht(pixels: &[u8], width: u32, height: u32, components: u16) -> Vec<u8> {
    let options = encode_options();
    encode_htj2k(pixels, width, height, components, 8, false, &options)
        .expect("encode HTJ2K fixture")
}

fn encode_options() -> EncodeOptions {
    EncodeOptions {
        reversible: true,
        num_decomposition_levels: 3,
        guard_bits: 2,
        ..EncodeOptions::default()
    }
}

fn run_decode_case(case: &DecodeBenchCase) {
    let probe = J2kDecoder::new(&case.bytes).expect("benchmark case parses");
    let dims = probe.inner().info().dimensions;
    let full = measure_decode(case, DecodeOperation::Full, dims);
    emit_decode_row(case, DecodeOperation::Full, dims, &full);

    let roi = benchmark_roi(dims);
    let scaled = roi.scaled_covering(Downscale::Half);
    let scaled_dims = (scaled.w, scaled.h);
    let region_scaled = measure_decode(case, DecodeOperation::RegionScaled, scaled_dims);
    emit_decode_row(
        case,
        DecodeOperation::RegionScaled,
        scaled_dims,
        &region_scaled,
    );
}

fn measure_decode(
    case: &DecodeBenchCase,
    operation: DecodeOperation,
    output_dims: (u32, u32),
) -> Result<DecodeTimingTriplet, String> {
    verify_decode_parity(case, operation, output_dims)?;

    let cpu = measure_result(|| decode_len_once(case, operation, BackendRequest::Cpu, false))?;
    let metal_resident =
        measure_result(|| decode_len_once(case, operation, BackendRequest::Metal, false))?;
    let metal_readback =
        measure_result(|| decode_len_once(case, operation, BackendRequest::Metal, true))?;

    if cpu.value != metal_resident.value || cpu.value != metal_readback.value {
        return Err(format!(
            "decoded byte lengths differ for {} {operation:?}: cpu={} metal_resident={} metal_readback={} dims={}x{}",
            case.id, cpu.value, metal_resident.value, metal_readback.value, output_dims.0, output_dims.1
        ));
    }

    Ok((cpu, metal_resident, metal_readback))
}

fn verify_decode_parity(
    case: &DecodeBenchCase,
    operation: DecodeOperation,
    output_dims: (u32, u32),
) -> Result<(), String> {
    let cpu = decode_bytes_once(case, operation, BackendRequest::Cpu)?;
    let metal = decode_bytes_once(case, operation, BackendRequest::Metal)?;
    if cpu != metal {
        return Err(format!(
            "decoded bytes differ for {} {operation:?}: cpu_len={} metal_len={} dims={}x{}",
            case.id,
            cpu.len(),
            metal.len(),
            output_dims.0,
            output_dims.1
        ));
    }
    Ok(())
}

fn decode_bytes_once(
    case: &DecodeBenchCase,
    operation: DecodeOperation,
    backend: BackendRequest,
) -> Result<Vec<u8>, String> {
    Ok(decode_surface_once(case, operation, backend)?
        .surface
        .as_bytes()
        .to_vec())
}

fn decode_len_once(
    case: &DecodeBenchCase,
    operation: DecodeOperation,
    backend: BackendRequest,
    readback: bool,
) -> Result<usize, String> {
    let decoded = decode_surface_once(case, operation, backend)?;
    if readback {
        Ok(decoded.surface.as_bytes().len())
    } else {
        Ok(decoded.surface.byte_len())
    }
}

fn decode_surface_once(
    case: &DecodeBenchCase,
    operation: DecodeOperation,
    backend: BackendRequest,
) -> Result<j2k_metal::DecodeSurfaceWithReport, String> {
    let mut decoder = J2kDecoder::new(&case.bytes).map_err(|error| error.to_string())?;
    let decoded = match operation {
        DecodeOperation::Full => decoder
            .decode_request_to_device_with_report(MetalDecodeRequest::full(case.fmt, backend))
            .map_err(|error| error.to_string())?,
        DecodeOperation::RegionScaled => {
            let dims = decoder.inner().info().dimensions;
            decoder
                .decode_request_to_device_with_report(MetalDecodeRequest::region_scaled(
                    case.fmt,
                    benchmark_roi(dims),
                    Downscale::Half,
                    backend,
                ))
                .map_err(|error| error.to_string())?
        }
        DecodeOperation::Region | DecodeOperation::Scaled => {
            return Err("benchmark covers full and region+scaled decode only".to_string());
        }
    };

    match backend {
        BackendRequest::Cpu => {
            if decoded.report.selected_backend != BackendKind::Cpu {
                return Err(format!(
                    "CPU decode selected unexpected backend {:?}",
                    decoded.report.selected_backend
                ));
            }
        }
        BackendRequest::Metal => {
            if decoded.report.selected_backend != BackendKind::Metal
                || decoded.report.surface_residency != SurfaceResidency::MetalResidentDecode
            {
                return Err(format!(
                    "strict Metal decode did not return resident Metal surface: {:?}",
                    decoded.report
                ));
            }
        }
        BackendRequest::Auto | BackendRequest::Cuda => {
            return Err(format!("unsupported benchmark backend {backend:?}"));
        }
    }
    Ok(decoded)
}

fn emit_decode_row(
    case: &DecodeBenchCase,
    operation: DecodeOperation,
    output_dims: (u32, u32),
    result: &Result<DecodeTimingTriplet, String>,
) {
    match result {
        Ok((cpu, metal_resident, metal_readback)) => println!(
            "j2k_metal_decode_bench case={} source={} codec={} container={} operation={} fmt={} size={}x{} cpu_ms={:.3} metal_resident_ms={:.3} metal_readback_ms={:.3} output_bytes={}",
            case.id,
            case.source,
            case.codec,
            case.container,
            operation_label(operation),
            format_label(case.fmt),
            output_dims.0,
            output_dims.1,
            millis(cpu.duration),
            millis(metal_resident.duration),
            millis(metal_readback.duration),
            cpu.value
        ),
        Err(error) => println!(
            "j2k_metal_decode_bench case={} source={} codec={} container={} operation={} fmt={} size={}x{} cpu_ms=skipped metal_resident_ms=skipped metal_readback_ms=skipped output_bytes=skipped error={}",
            case.id,
            case.source,
            case.codec,
            case.container,
            operation_label(operation),
            format_label(case.fmt),
            output_dims.0,
            output_dims.1,
            sanitize_error(error)
        ),
    }
}

fn measure_result<T>(mut run: impl FnMut() -> Result<T, String>) -> Result<TimedOutput<T>, String> {
    std::hint::black_box(run()?);
    let mut samples = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let started = Instant::now();
        let value = run()?;
        std::hint::black_box(&value);
        samples.push((started.elapsed(), value));
    }
    samples.sort_by_key(|(duration, _)| *duration);
    let median = samples.len() / 2;
    let (duration, value) = samples.swap_remove(median);
    Ok(TimedOutput { duration, value })
}

fn benchmark_roi(dims: (u32, u32)) -> Rect {
    let w = (dims.0 / 2).max(1);
    let h = (dims.1 / 2).max(1);
    Rect {
        x: (dims.0.saturating_sub(w)) / 2,
        y: (dims.1.saturating_sub(h)) / 2,
        w,
        h,
    }
}

fn external_decode_cases() -> Result<Vec<DecodeBenchCase>, String> {
    let dirs = external_decode_input_dirs();
    if dirs.is_empty() {
        return Ok(Vec::new());
    }
    let manifest = metal_decode_manifest()?;
    let mut cases = Vec::new();
    for dir in dirs {
        let mut paths = Vec::new();
        collect_decode_paths(&dir, &mut paths)?;
        if paths.is_empty() {
            return Err(format!(
                "{METAL_DECODE_INPUT_DIRS_ENV} entry {} contains no J2K/JP2/JPH fixtures",
                dir.display()
            ));
        }
        paths.sort();
        for path in paths {
            match load_external_decode_case(&path, manifest.as_ref()) {
                Ok(Some(case)) => cases.push(case),
                Ok(None) => {}
                Err(error) => return Err(error),
            }
        }
    }
    Ok(cases)
}

fn load_external_decode_case(
    path: &Path,
    manifest: Option<&DecodeManifest>,
) -> Result<Option<DecodeBenchCase>, String> {
    let bytes = fs::read(path).map_err(|error| format!("read {}: {error}", path.display()))?;
    let manifest_entry = validate_decode_manifest_entry(path, &bytes, manifest)?;
    let decoder = J2kDecoder::new(&bytes).map_err(|error| {
        format!(
            "external Metal decode input {} did not parse: {error}",
            path.display()
        )
    })?;
    let info = decoder.inner().info();
    let Some(fmt) = pixel_format_for_external(info.components, info.bit_depth) else {
        println!(
            "j2k_metal_decode_skipped_case path={} reason=unsupported_components_or_bit_depth components={} bit_depth={}",
            sanitized_path(path),
            info.components,
            info.bit_depth
        );
        return Ok(None);
    };
    let container = manifest_entry
        .and_then(|entry| entry.container.clone())
        .unwrap_or_else(|| infer_container(path, &bytes).to_string());
    if container != "raw-codestream" {
        println!(
            "j2k_metal_decode_skipped_case path={} reason=wrapper_container_not_claimed_for_metal_decode container={}",
            sanitized_path(path),
            container
        );
        return Ok(None);
    }
    Ok(Some(DecodeBenchCase {
        id: sanitized_stem(path),
        bytes,
        fmt,
        codec: manifest_entry
            .and_then(|entry| entry.codec.clone())
            .unwrap_or_else(|| "unknown".to_string()),
        container,
        source: format!("external:{}", sanitized_path(path)),
    }))
}

fn validate_decode_manifest_entry<'a>(
    path: &Path,
    bytes: &[u8],
    manifest: Option<&'a DecodeManifest>,
) -> Result<Option<&'a DecodeManifestEntry>, String> {
    let Some(manifest) = manifest else {
        return Ok(None);
    };
    let canonical_path = path.canonicalize().map_err(|error| {
        format!(
            "canonicalize external Metal decode input {}: {error}",
            path.display()
        )
    })?;
    let Some(entry) = manifest.entries.get(&canonical_path) else {
        return Err(format!(
            "external Metal decode input {} is not covered by {METAL_DECODE_MANIFEST_ENV}",
            path.display()
        ));
    };
    let actual_hash = fnv1a64_hex(bytes);
    if actual_hash != entry.input_fnv1a64 {
        return Err(format!(
            "external Metal decode input {} hash mismatch: manifest {} != actual {actual_hash}",
            path.display(),
            entry.input_fnv1a64
        ));
    }
    Ok(Some(entry))
}

fn metal_decode_manifest() -> Result<Option<DecodeManifest>, String> {
    let Some(path) = std::env::var_os(METAL_DECODE_MANIFEST_ENV).map(PathBuf::from) else {
        return Ok(None);
    };
    let text = fs::read_to_string(&path).map_err(|error| {
        format!(
            "read {METAL_DECODE_MANIFEST_ENV} {}: {error}",
            path.display()
        )
    })?;
    let mut lines = text.lines().filter(|line| !line.trim().is_empty());
    let header = lines
        .next()
        .ok_or_else(|| format!("Metal decode manifest {} is empty", path.display()))?;
    let headers = header.split('\t').collect::<Vec<_>>();
    let path_index = manifest_column(&headers, METAL_DECODE_MANIFEST_LABEL, "path")?;
    let hash_index = manifest_column(&headers, METAL_DECODE_MANIFEST_LABEL, "input_fnv1a64")?;
    let codec_index = optional_manifest_column(&headers, "codec");
    let container_index = optional_manifest_column(&headers, "container");
    let base = path.parent().unwrap_or_else(|| Path::new("."));
    let mut entries = HashMap::new();
    for (line_index, line) in lines.enumerate() {
        if line.trim_start().starts_with('#') {
            continue;
        }
        let fields = line.split('\t').collect::<Vec<_>>();
        let row_number = line_index + 2;
        let raw_path = manifest_field(
            &fields,
            METAL_DECODE_MANIFEST_LABEL,
            path_index,
            "path",
            row_number,
        )?;
        let resolved = if Path::new(raw_path).is_absolute() {
            PathBuf::from(raw_path)
        } else {
            base.join(raw_path)
        };
        let canonical = resolved.canonicalize().map_err(|error| {
            format!(
                "Metal decode manifest {} row {row_number} path {} cannot be canonicalized: {error}",
                path.display(),
                resolved.display()
            )
        })?;
        let entry = DecodeManifestEntry {
            input_fnv1a64: manifest_field(
                &fields,
                METAL_DECODE_MANIFEST_LABEL,
                hash_index,
                "input_fnv1a64",
                row_number,
            )?
            .to_string(),
            codec: manifest_optional_value(
                &fields,
                METAL_DECODE_MANIFEST_LABEL,
                codec_index,
                "codec",
                row_number,
            )?,
            container: manifest_optional_value(
                &fields,
                METAL_DECODE_MANIFEST_LABEL,
                container_index,
                "container",
                row_number,
            )?,
        };
        if entries.insert(canonical, entry).is_some() {
            return Err(format!(
                "Metal decode manifest {} row {row_number} duplicates path {raw_path}",
                path.display()
            ));
        }
    }
    Ok(Some(DecodeManifest { entries }))
}

fn collect_decode_paths(dir: &Path, paths: &mut Vec<PathBuf>) -> Result<(), String> {
    if !dir.is_dir() {
        return Err(format!(
            "{METAL_DECODE_INPUT_DIRS_ENV} entry is not a directory: {}",
            dir.display()
        ));
    }
    for entry in fs::read_dir(dir).map_err(|error| format!("read {}: {error}", dir.display()))? {
        let path = entry
            .map_err(|error| format!("read dir entry under {}: {error}", dir.display()))?
            .path();
        if path.is_dir() {
            collect_decode_paths(&path, paths)?;
        } else if is_decode_path(&path) {
            paths.push(path);
        }
    }
    Ok(())
}

fn is_decode_path(path: &Path) -> bool {
    path.extension()
        .and_then(|value| value.to_str())
        .is_some_and(|extension| {
            matches!(
                extension.to_ascii_lowercase().as_str(),
                "j2k" | "j2c" | "jhc" | "jp2" | "jph"
            )
        })
}

fn pixel_format_for_external(components: u16, bit_depth: u8) -> Option<PixelFormat> {
    match (components, bit_depth) {
        (1, 1..=8) => Some(PixelFormat::Gray8),
        (3, 1..=8) => Some(PixelFormat::Rgb8),
        (4, 1..=8) => Some(PixelFormat::Rgba8),
        (1, 9..=16) => Some(PixelFormat::Gray16),
        (3, 9..=16) => Some(PixelFormat::Rgb16),
        _ => None,
    }
}

fn infer_container(path: &Path, bytes: &[u8]) -> &'static str {
    if let Some(extension) = path.extension().and_then(|value| value.to_str()) {
        match extension.to_ascii_lowercase().as_str() {
            "jp2" => return "jp2",
            "jph" => return "jph",
            "jhc" => return "jhc",
            _ => {}
        }
    }
    if bytes.starts_with(&[0, 0, 0, 12, b'j', b'P', b' ', b' ']) {
        "jp2"
    } else {
        "raw-codestream"
    }
}

fn emit_metadata(generated_case_count: usize, external_case_count: usize) {
    println!(
        "j2k_metal_decode_generated_included\t{}",
        include_generated_decode_cases()
    );
    println!(
        "j2k_metal_decode_io_policy\tgenerated-fixtures-and-preloaded-external-codestreams;timed-full-rows-include-decode-work;metal_resident_ms-does-not-readback;metal_readback_ms-includes-host-visible-byte-access"
    );
    println!(
        "j2k_metal_decode_input_dirs\t{}",
        std::env::var(METAL_DECODE_INPUT_DIRS_ENV).unwrap_or_else(|_| "not set".to_string())
    );
    println!(
        "j2k_metal_decode_manifest\t{}",
        std::env::var(METAL_DECODE_MANIFEST_ENV).unwrap_or_else(|_| "not set".to_string())
    );
    println!("j2k_metal_decode_generated_case_count\t{generated_case_count}");
    println!("j2k_metal_decode_external_case_count\t{external_case_count}");
}

fn include_generated_decode_cases() -> bool {
    !env_falsey(METAL_DECODE_INCLUDE_GENERATED_ENV)
}

fn external_decode_input_dirs() -> Vec<PathBuf> {
    std::env::var_os(METAL_DECODE_INPUT_DIRS_ENV)
        .map(|paths| std::env::split_paths(&paths).collect())
        .unwrap_or_default()
}

fn env_truthy(name: &str) -> bool {
    std::env::var(name)
        .ok()
        .is_some_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "on"))
}

fn env_falsey(name: &str) -> bool {
    std::env::var(name)
        .ok()
        .is_some_and(|value| matches!(value.as_str(), "0" | "false" | "FALSE" | "no" | "off"))
}

fn operation_label(operation: DecodeOperation) -> &'static str {
    match operation {
        DecodeOperation::Full => "full",
        DecodeOperation::Region => "region",
        DecodeOperation::Scaled => "scaled",
        DecodeOperation::RegionScaled => "region_scaled",
    }
}

fn format_label(fmt: PixelFormat) -> &'static str {
    match fmt {
        PixelFormat::Gray8 => "gray8",
        PixelFormat::Rgb8 => "rgb8",
        PixelFormat::Rgba8 => "rgba8",
        PixelFormat::Gray16 => "gray16",
        PixelFormat::Rgb16 => "rgb16",
        PixelFormat::Rgba16 => "rgba16",
        _ => "unknown",
    }
}

fn sanitized_stem(path: &Path) -> String {
    path.file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("unnamed")
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

fn sanitized_path(path: &Path) -> String {
    path.display()
        .to_string()
        .chars()
        .map(|ch| {
            if ch.is_ascii_control() || ch.is_whitespace() {
                '_'
            } else {
                ch
            }
        })
        .collect()
}

fn sanitize_error(error: &str) -> String {
    error
        .chars()
        .map(|ch| {
            if ch.is_ascii_control() || ch.is_whitespace() {
                '_'
            } else {
                ch
            }
        })
        .collect()
}

fn millis(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}
