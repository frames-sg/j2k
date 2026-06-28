// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
    time::Duration,
};

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use j2k::{
    encode_j2k_lossless, EncodeBackendPreference, J2kBlockCodingMode, J2kEncodeValidation,
    J2kLosslessEncodeOptions, J2kLosslessSamples,
};
use j2k_core::BackendKind;
use j2k_cuda::encode_j2k_lossless_with_cuda;
use j2k_cuda_runtime::{
    CudaContext, CudaHtj2kEncodeCodeBlockJob, CudaHtj2kEncodeCodeBlockRegionJob,
    CudaHtj2kEncodeTables, CudaHtj2kEncodedCodeBlocks,
};
use j2k_native::{
    encode_ht_code_block_scalar, ht_uvlc_encode_table, ht_vlc_encode_table0, ht_vlc_encode_table1,
};
use j2k_test_support::read_pnm_image;

const TILE_DIM: u32 = 512;
const CODE_BLOCK_DIM: u32 = 64;
const CODE_BLOCK_BATCH: usize = 64;
const REGION_BLOCKS_X: u32 = 8;
const REGION_BLOCKS_Y: u32 = 8;
const ENCODE_SAMPLE_SIZE: usize = 10;
const ENCODE_WARM_UP: Duration = Duration::from_millis(500);
const ENCODE_MEASUREMENT: Duration = Duration::from_secs(1);

struct EncodeBenchCase {
    id: String,
    pixels: Vec<u8>,
    width: u32,
    height: u32,
    components: u8,
    input_source: String,
}

struct CudaEncodeManifest {
    entries: HashMap<PathBuf, CudaEncodeManifestEntry>,
}

struct CudaEncodeManifestEntry {
    input_fnv1a64: Option<String>,
}

fn bench_htj2k_encode(c: &mut Criterion) {
    let pixels = generate_gray_tile(TILE_DIM, TILE_DIM);
    let coefficients =
        generate_codeblock_coefficients(CODE_BLOCK_DIM, CODE_BLOCK_DIM, CODE_BLOCK_BATCH);
    let jobs = contiguous_jobs(CODE_BLOCK_DIM, CODE_BLOCK_DIM, CODE_BLOCK_BATCH);
    let region_width = CODE_BLOCK_DIM * REGION_BLOCKS_X;
    let region_height = CODE_BLOCK_DIM * REGION_BLOCKS_Y;
    let region_coefficients = generate_region_coefficients(region_width, region_height);
    let region_jobs = strided_region_jobs(CODE_BLOCK_DIM, REGION_BLOCKS_X, REGION_BLOCKS_Y);
    let cuda_available = cuda_encode_available(&pixels, &coefficients, &jobs);
    let external_cases = external_encode_cases();

    emit_encode_input_metadata(&external_cases);

    if include_generated_host_input() {
        bench_host_input(c, &pixels, cuda_available);
    }
    bench_external_host_input(c, &external_cases, cuda_available);
    bench_codeblock_microkernels(c, &coefficients, &jobs, cuda_available);
    bench_device_input_regions(c, &region_coefficients, &region_jobs, cuda_available);
}

fn bench_host_input(c: &mut Criterion, pixels: &[u8], cuda_available: bool) {
    let mut group = c.benchmark_group("j2k_cuda_htj2k_host_input_encode");
    group.bench_with_input(
        BenchmarkId::new("cpu_gray8", TILE_DIM),
        pixels,
        |b, pixels| {
            let options = cpu_htj2k_options();
            b.iter(|| {
                let samples = J2kLosslessSamples::new(
                    std::hint::black_box(pixels),
                    TILE_DIM,
                    TILE_DIM,
                    1,
                    8,
                    false,
                )
                .expect("valid gray8 samples");
                let encoded =
                    encode_j2k_lossless(samples, &options).expect("CPU HTJ2K lossless encode");
                assert_eq!(encoded.backend, BackendKind::Cpu);
                std::hint::black_box(encoded.codestream.len())
            });
        },
    );

    if cuda_available {
        group.bench_with_input(
            BenchmarkId::new("cuda_gray8", TILE_DIM),
            pixels,
            |b, pixels| {
                let options = cuda_htj2k_options();
                b.iter(|| {
                    let samples = J2kLosslessSamples::new(
                        std::hint::black_box(pixels),
                        TILE_DIM,
                        TILE_DIM,
                        1,
                        8,
                        false,
                    )
                    .expect("valid gray8 samples");
                    let encoded = encode_j2k_lossless_with_cuda(samples, &options)
                        .expect("CUDA HTJ2K lossless encode");
                    assert_eq!(encoded.backend, BackendKind::Cuda);
                    std::hint::black_box(encoded.codestream.len())
                });
            },
        );
    }
    group.finish();
}

fn bench_external_host_input(c: &mut Criterion, cases: &[EncodeBenchCase], cuda_available: bool) {
    if cases.is_empty() {
        return;
    }
    let mut group = c.benchmark_group("j2k_cuda_htj2k_external_host_input_encode");
    for case in cases {
        group.bench_with_input(
            BenchmarkId::new(format!("cpu_external_{}", case.id), dimensions_label(case)),
            case,
            |b, case| {
                let options = cpu_htj2k_options();
                b.iter(|| {
                    let encoded = encode_case_cpu(std::hint::black_box(case), options)
                        .expect("CPU external HTJ2K encode");
                    std::hint::black_box(encoded)
                });
            },
        );
        if cuda_available {
            group.bench_with_input(
                BenchmarkId::new(format!("cuda_external_{}", case.id), dimensions_label(case)),
                case,
                |b, case| {
                    let options = cuda_htj2k_options();
                    b.iter(|| {
                        let encoded = encode_case_cuda(std::hint::black_box(case), options)
                            .expect("CUDA external HTJ2K encode");
                        assert_eq!(encoded.backend, BackendKind::Cuda);
                        std::hint::black_box(encoded.codestream.len())
                    });
                },
            );
        }
    }
    group.finish();
}

fn encode_case_cpu(
    case: &EncodeBenchCase,
    options: J2kLosslessEncodeOptions,
) -> Result<usize, String> {
    let samples = J2kLosslessSamples::new(
        &case.pixels,
        case.width,
        case.height,
        u16::from(case.components),
        8,
        false,
    )
    .map_err(|error| error.to_string())?;
    let encoded = encode_j2k_lossless(samples, &options).map_err(|error| error.to_string())?;
    if encoded.backend != BackendKind::Cpu {
        return Err(format!(
            "external CPU encode case {} used {:?} backend",
            case.id, encoded.backend
        ));
    }
    Ok(encoded.codestream.len())
}

fn encode_case_cuda(
    case: &EncodeBenchCase,
    options: J2kLosslessEncodeOptions,
) -> Result<j2k::EncodedJ2k, String> {
    let samples = J2kLosslessSamples::new(
        &case.pixels,
        case.width,
        case.height,
        u16::from(case.components),
        8,
        false,
    )
    .map_err(|error| error.to_string())?;
    encode_j2k_lossless_with_cuda(samples, &options).map_err(|error| error.to_string())
}

fn bench_codeblock_microkernels(
    c: &mut Criterion,
    coefficients: &[i32],
    jobs: &[CudaHtj2kEncodeCodeBlockJob],
    cuda_available: bool,
) {
    let mut group = c.benchmark_group("j2k_cuda_htj2k_codeblock_microkernel");
    group.bench_with_input(
        BenchmarkId::new("cpu_scalar_cleanup", CODE_BLOCK_BATCH),
        &(coefficients, jobs),
        |b, (coefficients, jobs)| {
            b.iter(|| {
                let encoded_bytes = jobs
                    .iter()
                    .map(|job| {
                        let coefficients = contiguous_block(coefficients, *job);
                        encode_ht_code_block_scalar(
                            std::hint::black_box(coefficients),
                            job.width,
                            job.height,
                            job.total_bitplanes,
                        )
                        .expect("native scalar HT code-block encode")
                        .data
                        .len()
                    })
                    .sum::<usize>();
                std::hint::black_box(encoded_bytes)
            });
        },
    );

    if cuda_available {
        group.bench_with_input(
            BenchmarkId::new("cuda_host_staged_cleanup", CODE_BLOCK_BATCH),
            &(coefficients, jobs),
            |b, (coefficients, jobs)| {
                let context = CudaContext::system_default().expect("CUDA context");
                let uvlc_table = uvlc_encode_table_bytes();
                let resources = context
                    .upload_htj2k_encode_resources(cuda_encode_tables(&uvlc_table))
                    .expect("CUDA HTJ2K encode resources");
                b.iter(|| {
                    let encoded = context
                        .encode_htj2k_codeblocks_with_resources(
                            std::hint::black_box(coefficients),
                            std::hint::black_box(jobs),
                            &resources,
                        )
                        .expect("CUDA host-staged HTJ2K code-block encode");
                    std::hint::black_box(assert_cuda_batch(&encoded))
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("cuda_resident_cleanup", CODE_BLOCK_BATCH),
            &(coefficients, jobs),
            |b, (coefficients, jobs)| {
                let context = CudaContext::system_default().expect("CUDA context");
                let coefficient_bytes = coefficients_as_bytes(coefficients);
                let resident_coefficients = context
                    .upload(&coefficient_bytes)
                    .expect("resident quantized coefficients");
                let uvlc_table = uvlc_encode_table_bytes();
                let resources = context
                    .upload_htj2k_encode_resources(cuda_encode_tables(&uvlc_table))
                    .expect("CUDA HTJ2K encode resources");
                b.iter(|| {
                    let encoded = context
                        .encode_htj2k_codeblocks_resident_with_resources(
                            &resident_coefficients,
                            coefficients.len(),
                            std::hint::black_box(jobs),
                            &resources,
                        )
                        .expect("CUDA resident HTJ2K code-block encode");
                    std::hint::black_box(assert_cuda_batch(&encoded))
                });
            },
        );
    }
    group.finish();
}

fn bench_device_input_regions(
    c: &mut Criterion,
    coefficients: &[i32],
    jobs: &[CudaHtj2kEncodeCodeBlockRegionJob],
    cuda_available: bool,
) {
    let mut group = c.benchmark_group("j2k_cuda_htj2k_device_input_encode");
    group.bench_with_input(
        BenchmarkId::new("cpu_gather_scalar_cleanup", jobs.len()),
        &(coefficients, jobs),
        |b, (coefficients, jobs)| {
            b.iter(|| {
                let encoded_bytes = jobs
                    .iter()
                    .map(|job| {
                        let block = gather_region(coefficients, *job);
                        encode_ht_code_block_scalar(
                            std::hint::black_box(&block),
                            job.width,
                            job.height,
                            job.total_bitplanes,
                        )
                        .expect("native scalar strided HT encode")
                        .data
                        .len()
                    })
                    .sum::<usize>();
                std::hint::black_box(encoded_bytes)
            });
        },
    );

    if cuda_available {
        group.bench_with_input(
            BenchmarkId::new("cuda_resident_strided_cleanup", jobs.len()),
            &(coefficients, jobs),
            |b, (coefficients, jobs)| {
                let context = CudaContext::system_default().expect("CUDA context");
                let coefficient_bytes = coefficients_as_bytes(coefficients);
                let resident_coefficients = context
                    .upload(&coefficient_bytes)
                    .expect("resident strided quantized coefficients");
                let uvlc_table = uvlc_encode_table_bytes();
                let resources = context
                    .upload_htj2k_encode_resources(cuda_encode_tables(&uvlc_table))
                    .expect("CUDA HTJ2K encode resources");
                b.iter(|| {
                    let encoded = context
                        .encode_htj2k_codeblock_regions_resident_with_resources(
                            &resident_coefficients,
                            coefficients.len(),
                            std::hint::black_box(jobs),
                            &resources,
                        )
                        .expect("CUDA resident strided HTJ2K encode");
                    std::hint::black_box(assert_cuda_batch(&encoded))
                });
            },
        );
    }
    group.finish();
}

fn include_generated_host_input() -> bool {
    !env_falsey("J2K_CUDA_ENCODE_INCLUDE_GENERATED")
}

fn env_falsey(name: &str) -> bool {
    std::env::var(name)
        .ok()
        .is_some_and(|value| matches!(value.as_str(), "0" | "false" | "FALSE" | "no" | "off"))
}

fn external_encode_cases() -> Vec<EncodeBenchCase> {
    let dirs = external_encode_input_dirs();
    if dirs.is_empty() {
        return Vec::new();
    }
    let manifest = cuda_encode_manifest().unwrap_or_else(|error| panic!("{error}"));
    let mut cases = Vec::new();
    for dir in dirs {
        let mut paths = Vec::new();
        collect_pnm_paths(&dir, &mut paths)
            .unwrap_or_else(|error| panic!("collect external CUDA encode inputs: {error}"));
        paths.sort();
        assert!(
            !paths.is_empty(),
            "J2K_CUDA_ENCODE_INPUT_DIRS entry {} contains no staged .pgm/.ppm/.pnm files",
            dir.display()
        );
        for path in paths {
            cases.push(
                load_external_encode_case(&path, manifest.as_ref())
                    .unwrap_or_else(|error| panic!("{error}")),
            );
        }
    }
    cases
}

fn external_encode_input_dirs() -> Vec<PathBuf> {
    std::env::var_os("J2K_CUDA_ENCODE_INPUT_DIRS")
        .map(|paths| std::env::split_paths(&paths).collect())
        .unwrap_or_default()
}

fn collect_pnm_paths(dir: &Path, paths: &mut Vec<PathBuf>) -> Result<(), String> {
    if !dir.is_dir() {
        return Err(format!(
            "J2K_CUDA_ENCODE_INPUT_DIRS entry is not a directory: {}",
            dir.display()
        ));
    }
    for entry in fs::read_dir(dir).map_err(|error| format!("read {}: {error}", dir.display()))? {
        let path = entry
            .map_err(|error| format!("read dir entry under {}: {error}", dir.display()))?
            .path();
        if path.is_dir() {
            collect_pnm_paths(&path, paths)?;
        } else if is_pnm_path(&path) {
            paths.push(path);
        }
    }
    Ok(())
}

fn is_pnm_path(path: &Path) -> bool {
    path.extension()
        .and_then(|value| value.to_str())
        .is_some_and(|extension| {
            matches!(
                extension.to_ascii_lowercase().as_str(),
                "pgm" | "ppm" | "pnm"
            )
        })
}

fn load_external_encode_case(
    path: &Path,
    manifest: Option<&CudaEncodeManifest>,
) -> Result<EncodeBenchCase, String> {
    let image = read_pnm_image(path).map_err(|error| {
        format!(
            "read external CUDA encode staged PNM {}: {error}",
            path.display()
        )
    })?;
    let components = u8::try_from(image.channels)
        .map_err(|_| format!("{} channel count is too large", path.display()))?;
    if !matches!(components, 1 | 3) {
        return Err(format!(
            "{} has unsupported component count {components}; expected PGM or PPM",
            path.display()
        ));
    }
    validate_cuda_encode_manifest_entry(path, &image.pixels, manifest)?;
    Ok(EncodeBenchCase {
        id: sanitized_stem(path),
        pixels: image.pixels,
        width: image.width,
        height: image.height,
        components,
        input_source: external_source_label(path)?,
    })
}

fn cuda_encode_manifest() -> Result<Option<CudaEncodeManifest>, String> {
    let Some(path) = std::env::var_os("J2K_CUDA_ENCODE_MANIFEST").map(PathBuf::from) else {
        return Ok(None);
    };
    let text = fs::read_to_string(&path)
        .map_err(|error| format!("read J2K_CUDA_ENCODE_MANIFEST {}: {error}", path.display()))?;
    let base = path.parent().unwrap_or_else(|| Path::new("."));
    let relocation_roots = external_encode_input_dirs();
    let mut lines = text.lines().filter(|line| !line.trim().is_empty());
    let header = lines
        .next()
        .ok_or_else(|| format!("CUDA encode manifest {} is empty", path.display()))?;
    let headers = header.split('\t').collect::<Vec<_>>();
    let path_index = manifest_column(&headers, "path")?;
    let hash_index = optional_manifest_column(&headers, "input_fnv1a64");
    let mut entries = HashMap::new();
    for (line_index, line) in lines.enumerate() {
        if line.trim_start().starts_with('#') {
            continue;
        }
        let fields = line.split('\t').collect::<Vec<_>>();
        let row_number = line_index + 2;
        let raw_path = manifest_field(&fields, path_index, "path", row_number)?;
        let canonical_path = canonicalize_manifest_row_path(
            raw_path,
            base,
            &relocation_roots,
            "CUDA encode manifest",
            &path,
            row_number,
        )?;
        let entry = CudaEncodeManifestEntry {
            input_fnv1a64: manifest_optional_value(
                &fields,
                hash_index,
                "input_fnv1a64",
                row_number,
            )?,
        };
        if entries.insert(canonical_path, entry).is_some() {
            return Err(format!(
                "CUDA encode manifest {} row {row_number} duplicates path {raw_path}",
                path.display()
            ));
        }
    }
    Ok(Some(CudaEncodeManifest { entries }))
}

fn canonicalize_manifest_row_path(
    raw_path: &str,
    base: &Path,
    relocation_roots: &[PathBuf],
    manifest_label: &str,
    manifest_path: &Path,
    row_number: usize,
) -> Result<PathBuf, String> {
    let raw = Path::new(raw_path);
    let resolved_path = if raw.is_absolute() {
        raw.to_path_buf()
    } else {
        base.join(raw)
    };
    match resolved_path.canonicalize() {
        Ok(path) => Ok(path),
        Err(primary_error) => {
            let candidates = manifest_relocation_candidates(raw, relocation_roots);
            if candidates.len() == 1 {
                Ok(candidates[0].clone())
            } else if !candidates.is_empty() {
                Err(format!(
                    "{manifest_label} {} row {row_number} path {} is ambiguous after suffix remap: {}",
                    manifest_path.display(),
                    raw_path,
                    join_path_labels(&candidates)
                ))
            } else {
                Err(format!(
                    "{manifest_label} {} row {row_number} path {} cannot be canonicalized: {primary_error}; no suffix remap found under {}",
                    manifest_path.display(),
                    resolved_path.display(),
                    join_path_labels(relocation_roots)
                ))
            }
        }
    }
}

fn manifest_relocation_candidates(raw_path: &Path, relocation_roots: &[PathBuf]) -> Vec<PathBuf> {
    let suffixes = normal_path_suffixes(raw_path);
    let mut candidates = Vec::new();
    for root in relocation_roots {
        for suffix in &suffixes {
            let candidate = root.join(suffix);
            let Ok(canonical) = candidate.canonicalize() else {
                continue;
            };
            if !candidates.contains(&canonical) {
                candidates.push(canonical);
            }
        }
    }
    candidates
}

fn normal_path_suffixes(path: &Path) -> Vec<PathBuf> {
    let parts = path
        .components()
        .filter_map(|component| match component {
            std::path::Component::Normal(part) => Some(part.to_owned()),
            _ => None,
        })
        .collect::<Vec<_>>();
    let mut suffixes = Vec::new();
    for start in 0..parts.len() {
        let mut suffix = PathBuf::new();
        for part in &parts[start..] {
            suffix.push(part);
        }
        suffixes.push(suffix);
    }
    suffixes
}

fn join_path_labels(paths: &[PathBuf]) -> String {
    if paths.is_empty() {
        return "none".to_string();
    }
    paths
        .iter()
        .map(|path| path.display().to_string())
        .collect::<Vec<_>>()
        .join(",")
}

fn validate_cuda_encode_manifest_entry(
    path: &Path,
    pixels: &[u8],
    manifest: Option<&CudaEncodeManifest>,
) -> Result<(), String> {
    let Some(manifest) = manifest else {
        return Ok(());
    };
    let canonical_path = path.canonicalize().map_err(|error| {
        format!(
            "canonicalize external CUDA encode input {}: {error}",
            path.display()
        )
    })?;
    let Some(entry) = manifest.entries.get(&canonical_path) else {
        return Err(format!(
            "external CUDA encode input {} is not covered by J2K_CUDA_ENCODE_MANIFEST",
            path.display()
        ));
    };
    let Some(expected_hash) = &entry.input_fnv1a64 else {
        return Err(format!(
            "external CUDA encode input {} manifest row is missing input_fnv1a64",
            path.display()
        ));
    };
    let actual_hash = fnv1a64_hex(pixels);
    if actual_hash != *expected_hash {
        return Err(format!(
            "external CUDA encode input {} hash mismatch: manifest {expected_hash} != actual {actual_hash}",
            path.display()
        ));
    }
    Ok(())
}

fn manifest_column(headers: &[&str], name: &str) -> Result<usize, String> {
    optional_manifest_column(headers, name)
        .ok_or_else(|| format!("CUDA encode manifest is missing required {name:?} column"))
}

fn optional_manifest_column(headers: &[&str], name: &str) -> Option<usize> {
    headers.iter().position(|header| *header == name)
}

fn manifest_field<'a>(
    fields: &'a [&str],
    index: usize,
    name: &str,
    row_number: usize,
) -> Result<&'a str, String> {
    fields
        .get(index)
        .copied()
        .ok_or_else(|| format!("CUDA encode manifest row {row_number} is missing {name:?} field"))
}

fn manifest_optional_value(
    fields: &[&str],
    index: Option<usize>,
    name: &str,
    row_number: usize,
) -> Result<Option<String>, String> {
    let Some(index) = index else {
        return Ok(None);
    };
    let value = manifest_field(fields, index, name, row_number)?.trim();
    if value.is_empty() {
        return Ok(None);
    }
    if value.chars().any(char::is_control) {
        return Err(format!(
            "CUDA encode manifest row {row_number} field {name:?} contains a control character"
        ));
    }
    Ok(Some(value.to_string()))
}

fn external_source_label(path: &Path) -> Result<String, String> {
    let source_path = path.display().to_string();
    if source_path.chars().any(char::is_control) {
        return Err(format!(
            "external CUDA encode input path contains a control character: {}",
            source_path.escape_debug()
        ));
    }
    Ok(format!("external:{source_path}"))
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

fn emit_encode_input_metadata(external_cases: &[EncodeBenchCase]) {
    println!(
        "j2k_cuda_encode_generated_host_input_included\t{}",
        include_generated_host_input()
    );
    println!("j2k_cuda_encode_sample_size\t{}", encode_sample_size());
    println!("j2k_cuda_encode_warm_up_ms\t{}", ENCODE_WARM_UP.as_millis());
    println!(
        "j2k_cuda_encode_measurement_ms\t{}",
        ENCODE_MEASUREMENT.as_millis()
    );
    println!(
        "j2k_cuda_encode_io_policy\tstaged-pnm-pixels-preloaded-no-filesystem-io-in-timed-loop;cuda-host-input-rows-include-public-api-host-submission-and-device-encode-work"
    );
    println!(
        "j2k_cuda_encode_input_dirs\t{}",
        std::env::var("J2K_CUDA_ENCODE_INPUT_DIRS").unwrap_or_else(|_| "not set".to_string())
    );
    println!(
        "j2k_cuda_encode_manifest\t{}",
        std::env::var("J2K_CUDA_ENCODE_MANIFEST").unwrap_or_else(|_| "not set".to_string())
    );
    println!(
        "j2k_cuda_encode_external_case_count\t{}",
        external_cases.len()
    );
    println!("j2k_cuda_encode_external_input_format\tstaged-pnm-p5-p6");
    println!(
        "j2k_cuda_encode_external_case_sources\t{}",
        if external_cases.is_empty() {
            "none".to_string()
        } else {
            external_cases
                .iter()
                .map(|case| case.input_source.as_str())
                .collect::<Vec<_>>()
                .join(",")
        }
    );
}

fn cuda_encode_available(
    pixels: &[u8],
    coefficients: &[i32],
    jobs: &[CudaHtj2kEncodeCodeBlockJob],
) -> bool {
    let samples = J2kLosslessSamples::new(pixels, TILE_DIM, TILE_DIM, 1, 8, false)
        .expect("valid gray8 samples");
    let public_result = encode_j2k_lossless_with_cuda(samples, &cuda_htj2k_options());
    match public_result {
        Ok(encoded) if encoded.backend == BackendKind::Cuda => {}
        Ok(_) if std::env::var_os("J2K_REQUIRE_CUDA_BENCH").is_some() => {
            panic!("J2K_REQUIRE_CUDA_BENCH is set but CUDA HTJ2K encode was not resident")
        }
        Ok(_) => {
            eprintln!("skipping CUDA HTJ2K encode benches: device encode did not dispatch");
            return false;
        }
        Err(error) if std::env::var_os("J2K_REQUIRE_CUDA_BENCH").is_some() => {
            panic!("J2K_REQUIRE_CUDA_BENCH is set but CUDA HTJ2K encode failed: {error}")
        }
        Err(error) => {
            eprintln!("skipping CUDA HTJ2K encode benches: {error}");
            return false;
        }
    }

    let context = match CudaContext::system_default() {
        Ok(context) => context,
        Err(error) if std::env::var_os("J2K_REQUIRE_CUDA_BENCH").is_some() => {
            panic!("J2K_REQUIRE_CUDA_BENCH is set but CUDA context failed: {error}")
        }
        Err(error) => {
            eprintln!("skipping CUDA HTJ2K encode benches: {error}");
            return false;
        }
    };
    let uvlc_table = uvlc_encode_table_bytes();
    let resources = match context.upload_htj2k_encode_resources(cuda_encode_tables(&uvlc_table)) {
        Ok(resources) => resources,
        Err(error) if std::env::var_os("J2K_REQUIRE_CUDA_BENCH").is_some() => {
            panic!("J2K_REQUIRE_CUDA_BENCH is set but CUDA HTJ2K encode resource upload failed: {error}")
        }
        Err(error) => {
            eprintln!("skipping CUDA HTJ2K encode benches: {error}");
            return false;
        }
    };
    let result =
        context.encode_htj2k_codeblocks_with_resources(coefficients, &jobs[..1], &resources);
    match result {
        Ok(encoded) if encoded.execution().kernel_dispatches() > 0 => true,
        Ok(_) if std::env::var_os("J2K_REQUIRE_CUDA_BENCH").is_some() => {
            panic!(
                "J2K_REQUIRE_CUDA_BENCH is set but CUDA HTJ2K code-block encode did not dispatch"
            )
        }
        Ok(_) => {
            eprintln!("skipping CUDA HTJ2K encode benches: code-block kernel did not dispatch");
            false
        }
        Err(error) if std::env::var_os("J2K_REQUIRE_CUDA_BENCH").is_some() => {
            panic!("J2K_REQUIRE_CUDA_BENCH is set but CUDA HTJ2K code-block encode failed: {error}")
        }
        Err(error) => {
            eprintln!("skipping CUDA HTJ2K encode benches: {error}");
            false
        }
    }
}

fn cpu_htj2k_options() -> J2kLosslessEncodeOptions {
    J2kLosslessEncodeOptions::default()
        .with_backend(EncodeBackendPreference::CpuOnly)
        .with_block_coding_mode(J2kBlockCodingMode::HighThroughput)
        .with_max_decomposition_levels(Some(1))
        .with_validation(J2kEncodeValidation::External)
}

fn cuda_htj2k_options() -> J2kLosslessEncodeOptions {
    J2kLosslessEncodeOptions::default()
        .with_backend(EncodeBackendPreference::RequireDevice)
        .with_block_coding_mode(J2kBlockCodingMode::HighThroughput)
        .with_max_decomposition_levels(Some(1))
        .with_validation(J2kEncodeValidation::External)
}

fn cuda_encode_tables(uvlc_table: &[u8]) -> CudaHtj2kEncodeTables<'_> {
    CudaHtj2kEncodeTables {
        vlc_table0: ht_vlc_encode_table0(),
        vlc_table1: ht_vlc_encode_table1(),
        uvlc_table,
    }
}

fn uvlc_encode_table_bytes() -> Vec<u8> {
    ht_uvlc_encode_table()
        .iter()
        .flat_map(|entry| {
            [
                entry.pre,
                entry.pre_len,
                entry.suf,
                entry.suf_len,
                entry.ext,
                entry.ext_len,
            ]
        })
        .collect()
}

fn assert_cuda_batch(encoded: &CudaHtj2kEncodedCodeBlocks) -> usize {
    assert_eq!(encoded.execution().kernel_dispatches(), 1);
    assert!(encoded
        .code_blocks()
        .iter()
        .all(|block| block.status().is_ok()));
    encoded
        .code_blocks()
        .iter()
        .map(|block| block.data().len())
        .sum()
}

fn contiguous_block(coefficients: &[i32], job: CudaHtj2kEncodeCodeBlockJob) -> &[i32] {
    let start = usize::try_from(job.coefficient_offset).expect("job offset fits usize");
    let len = area_len(job.width, job.height);
    &coefficients[start..start + len]
}

fn gather_region(coefficients: &[i32], job: CudaHtj2kEncodeCodeBlockRegionJob) -> Vec<i32> {
    let start = usize::try_from(job.coefficient_offset).expect("job offset fits usize");
    let stride = usize::try_from(job.coefficient_stride).expect("job stride fits usize");
    let width = usize::try_from(job.width).expect("job width fits usize");
    let height = usize::try_from(job.height).expect("job height fits usize");
    let mut block = Vec::with_capacity(width * height);
    for y in 0..height {
        let row_start = start + y * stride;
        block.extend_from_slice(&coefficients[row_start..row_start + width]);
    }
    block
}

fn generate_gray_tile(width: u32, height: u32) -> Vec<u8> {
    let mut pixels = Vec::with_capacity(area_len(width, height));
    for y in 0..height {
        for x in 0..width {
            let value = (x * 17 + y * 31 + x.wrapping_mul(y) / 11) & 0xff;
            pixels.push(u8::try_from(value).expect("masked sample fits in u8"));
        }
    }
    pixels
}

fn generate_codeblock_coefficients(width: u32, height: u32, batch: usize) -> Vec<i32> {
    let mut coefficients = Vec::with_capacity(area_len(width, height) * batch);
    for block in 0..batch {
        let block = u32::try_from(block).expect("bench block index fits u32");
        for y in 0..height {
            for x in 0..width {
                coefficients.push(patterned_coefficient(x, y, block));
            }
        }
    }
    coefficients
}

fn generate_region_coefficients(width: u32, height: u32) -> Vec<i32> {
    let mut coefficients = Vec::with_capacity(area_len(width, height));
    for y in 0..height {
        for x in 0..width {
            coefficients.push(patterned_coefficient(
                x,
                y,
                x / CODE_BLOCK_DIM + y / CODE_BLOCK_DIM,
            ));
        }
    }
    coefficients
}

fn patterned_coefficient(x: u32, y: u32, block: u32) -> i32 {
    if (x + y + block).is_multiple_of(7) {
        return 0;
    }
    let raw = (x * 13 + y * 17 + block * 19 + (x ^ y)) & 0x1ff;
    (i32::try_from(raw).expect("masked coefficient fits i32") - 256) / 4
}

fn contiguous_jobs(width: u32, height: u32, batch: usize) -> Vec<CudaHtj2kEncodeCodeBlockJob> {
    let block_len = area_len(width, height);
    let mut jobs = Vec::with_capacity(batch);
    for block in 0..batch {
        let offset = block
            .checked_mul(block_len)
            .expect("bench coefficient offset fits usize");
        jobs.push(CudaHtj2kEncodeCodeBlockJob {
            coefficient_offset: u32::try_from(offset).expect("bench offset fits u32"),
            width,
            height,
            total_bitplanes: 8,
            target_coding_passes: 1,
        });
    }
    jobs
}

fn strided_region_jobs(
    block_dim: u32,
    blocks_x: u32,
    blocks_y: u32,
) -> Vec<CudaHtj2kEncodeCodeBlockRegionJob> {
    let stride = block_dim
        .checked_mul(blocks_x)
        .expect("bench stride fits u32");
    let mut jobs = Vec::with_capacity(area_len(blocks_x, blocks_y));
    for by in 0..blocks_y {
        for bx in 0..blocks_x {
            let row_offset = by
                .checked_mul(block_dim)
                .and_then(|value| value.checked_mul(stride))
                .expect("bench row offset fits u32");
            let column_offset = bx
                .checked_mul(block_dim)
                .expect("bench column offset fits u32");
            jobs.push(CudaHtj2kEncodeCodeBlockRegionJob {
                coefficient_offset: row_offset + column_offset,
                coefficient_stride: stride,
                width: block_dim,
                height: block_dim,
                total_bitplanes: 8,
                target_coding_passes: 1,
            });
        }
    }
    jobs
}

fn coefficients_as_bytes(coefficients: &[i32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(std::mem::size_of_val(coefficients));
    for coefficient in coefficients {
        bytes.extend_from_slice(&coefficient.to_ne_bytes());
    }
    bytes
}

fn dimensions_label(case: &EncodeBenchCase) -> String {
    format!("{}x{}", case.width, case.height)
}

fn fnv1a64_hex(bytes: &[u8]) -> String {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{hash:016x}")
}

fn area_len(width: u32, height: u32) -> usize {
    usize::try_from(width).expect("bench width fits usize")
        * usize::try_from(height).expect("bench height fits usize")
}

fn cuda_encode_criterion() -> Criterion {
    Criterion::default()
        .sample_size(encode_sample_size())
        .warm_up_time(ENCODE_WARM_UP)
        .measurement_time(ENCODE_MEASUREMENT)
}

fn encode_sample_size() -> usize {
    let Some(value) = std::env::var_os("J2K_CUDA_ENCODE_SAMPLE_SIZE") else {
        return ENCODE_SAMPLE_SIZE;
    };
    let value = value.to_string_lossy();
    let sample_size = value
        .parse::<usize>()
        .unwrap_or_else(|error| panic!("invalid J2K_CUDA_ENCODE_SAMPLE_SIZE `{value}`: {error}"));
    assert!(
        sample_size >= 10,
        "J2K_CUDA_ENCODE_SAMPLE_SIZE must be at least Criterion's minimum sample size of 10"
    );
    sample_size
}

criterion_group! {
    name = benches;
    config = cuda_encode_criterion();
    targets = bench_htj2k_encode
}
criterion_main!(benches);
