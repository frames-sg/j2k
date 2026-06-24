// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{
    collections::{HashMap, HashSet},
    fs,
    io::Write,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    time::Instant,
};

use j2k::{
    encode_j2k_lossless, EncodeBackendPreference, J2kBlockCodingMode, J2kDecoder,
    J2kEncodeValidation, J2kLosslessEncodeOptions, J2kLosslessSamples,
};
use j2k_compare::{grok, openjpeg, parse_positive_usize, sample_stats, usize_to_f64};
use j2k_core::PixelFormat;
use j2k_test_support::{patterned_gray8, patterned_rgb8, wrap_jp2_codestream};

const DEFAULT_REPEATS: usize = 5;
const DEFAULT_CASE_BATCH_SIZES: &[usize] = &[1];
const DEFAULT_MIXED_BATCH_SIZES: &[usize] = &[1, 16, 256];
const MIN_PUBLICATION_EXTERNAL_IMAGES: usize = 24;
const MIN_PUBLICATION_MIXED_DISTINCT_INPUTS: usize = 2;
const MIN_PUBLICATION_EXTERNAL_DIMENSIONS: usize = 3;
const MIN_PUBLICATION_EXTERNAL_SOURCE_FORMATS: usize = 2;

#[derive(Clone)]
struct ImageCase {
    name: String,
    input_source: String,
    corpus_category: String,
    corpus_name: String,
    license_status: String,
    source_command: String,
    manifest_status: String,
    source_format: String,
    width: u32,
    height: u32,
    components: u8,
    pixels: Vec<u8>,
    pnm_path: PathBuf,
}

impl ImageCase {
    fn format_label(&self) -> &'static str {
        match self.components {
            1 => "gray8",
            3 => "rgb8",
            _ => "unsupported",
        }
    }

    fn pixel_format(&self) -> Result<PixelFormat, String> {
        match self.components {
            1 => Ok(PixelFormat::Gray8),
            3 => Ok(PixelFormat::Rgb8),
            other => Err(format!(
                "{} has unsupported component count {other}",
                self.name
            )),
        }
    }

    fn input_digest(&self) -> String {
        fnv1a64_hex(&self.pixels)
    }
}

struct MixedImageBatch {
    name: String,
    cases: Vec<ImageCase>,
    components: u8,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum EncoderKind {
    J2k,
    OpenJpeg,
    Grok,
    Kakadu,
}

impl EncoderKind {
    const fn label(self) -> &'static str {
        match self {
            Self::J2k => "j2k",
            Self::OpenJpeg => "openjpeg",
            Self::Grok => "grok",
            Self::Kakadu => "kakadu",
        }
    }
}

#[derive(Clone)]
struct EncoderTool {
    kind: EncoderKind,
    program: PathBuf,
    available: bool,
}

struct Measurement {
    batch_size: usize,
    repeats: usize,
    median_us: f64,
    mean_us: f64,
    images_per_second_median: f64,
    encoded_bytes_per_repeat: usize,
    samples_us: Vec<f64>,
}

struct EncodeMeasurementState<'a> {
    tool: &'a EncoderTool,
    encoded_bytes_per_repeat: Option<usize>,
    samples_us: Vec<f64>,
}

struct EncodeManifest {
    entries: HashMap<PathBuf, EncodeManifestEntry>,
}

struct EncodeManifestEntry {
    corpus_category: String,
    corpus_name: String,
    license_status: String,
    source_command: String,
    input_fnv1a64: Option<String>,
}

struct ExternalImageMetadata {
    input_source: String,
    corpus_category: String,
    corpus_name: String,
    license_status: String,
    source_command: String,
    manifest_status: String,
}

#[derive(Clone, Copy)]
struct MetadataInput<'a> {
    args: &'a [String],
    repeats: usize,
    batch_sizes: &'a [usize],
    case_batch_sizes: &'a [usize],
    mixed_batch_sizes: &'a [usize],
    cases: &'a [ImageCase],
    mixed_batches: &'a [MixedImageBatch],
    selected_tools: &'a [EncoderTool],
    all_tools: &'a [EncoderTool],
    filters_empty: bool,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = std::env::args().collect::<Vec<_>>();
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        print_usage(&args[0]);
        return Ok(());
    }
    if args.get(1).is_some_and(|arg| arg == "--encode-one") {
        return encode_one(&args[2..]);
    }

    validate_tool_gates()?;
    let repeats = std::env::var("J2K_ENCODE_COMPARE_REPEATS")
        .ok()
        .map(|value| parse_positive_usize(&value, "J2K_ENCODE_COMPARE_REPEATS"))
        .transpose()?
        .unwrap_or(DEFAULT_REPEATS);
    let batch_sizes = batch_size_config_from_env()?;
    let combined_batch_sizes = combined_batch_sizes(
        &batch_sizes.case_batch_sizes,
        &batch_sizes.mixed_batch_sizes,
    );
    let filters = args.iter().skip(1).map(String::as_str).collect::<Vec<_>>();
    let work_dir = encode_work_dir()?;
    let cases = select_cases(all_image_cases(&work_dir)?, &filters)?;
    let mixed_batches = mixed_external_batches(&cases);
    let all_tools = all_encoder_tools()?;
    let tools = selected_encoder_tools(&all_tools)?;

    emit_metadata(MetadataInput {
        args: &args,
        repeats,
        batch_sizes: &combined_batch_sizes,
        case_batch_sizes: &batch_sizes.case_batch_sizes,
        mixed_batch_sizes: &batch_sizes.mixed_batch_sizes,
        cases: &cases,
        mixed_batches: &mixed_batches,
        selected_tools: &tools,
        all_tools: &all_tools,
        filters_empty: filters.is_empty(),
    });
    println!(
        "encoder\tcase\tbenchmark_mode\tencode_method\tinput_source\tcorpus_category\tcorpus_name\tlicense_status\tsource_command\tmanifest_status\tcodec\tcontainer\tformat\tdimensions\tbatch_size\trepeats\tinput_bytes\tinput_fnv1a64\tmedian_us\tmean_us\timages_per_second_median\tinput_mib_per_second_median\tencoded_bytes_per_repeat\tsamples_us\tskip_reason\tcommand_template"
    );

    for case in &cases {
        for &batch_size in &batch_sizes.case_batch_sizes {
            for row in measure_case_rows(case, &tools, repeats, batch_size, &work_dir)? {
                println!("{row}");
            }
        }
    }
    for mixed_batch in &mixed_batches {
        for &batch_size in &batch_sizes.mixed_batch_sizes {
            for row in measure_mixed_rows(mixed_batch, &tools, repeats, batch_size, &work_dir)? {
                println!("{row}");
            }
        }
    }
    println!("benchmark_complete\ttrue");
    Ok(())
}

fn print_usage(program: &str) {
    eprintln!("usage: {program} [case-name-filter ...]");
    eprintln!("       {program} --encode-one --input FILE.pnm --output FILE.jp2");
    eprintln!("Runs CLI-style lossless classic JPEG 2000 encoder benchmarks.");
}

fn encode_one(args: &[String]) -> Result<(), String> {
    let mut input = None;
    let mut output = None;
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--input" => {
                index += 1;
                input = args.get(index).map(PathBuf::from);
            }
            "--output" => {
                index += 1;
                output = args.get(index).map(PathBuf::from);
            }
            other => return Err(format!("unknown --encode-one argument `{other}`")),
        }
        index += 1;
    }
    let input = input.ok_or_else(|| "--encode-one requires --input".to_string())?;
    let output = output.ok_or_else(|| "--encode-one requires --output".to_string())?;
    let image = read_pnm(&input)?;
    let samples = J2kLosslessSamples::new(
        &image.pixels,
        image.width,
        image.height,
        image.components,
        8,
        false,
    )
    .map_err(|error| error.to_string())?;
    let options = J2kLosslessEncodeOptions::default()
        .with_backend(EncodeBackendPreference::CpuOnly)
        .with_block_coding_mode(J2kBlockCodingMode::Classic)
        .with_max_decomposition_levels(Some(2))
        .with_validation(J2kEncodeValidation::External);
    let encoded = encode_j2k_lossless(samples, &options).map_err(|error| error.to_string())?;
    let jp2 = wrap_jp2_codestream(
        &encoded.codestream,
        image.width,
        image.height,
        u16::from(image.components),
        8,
        16,
    );
    fs::write(&output, jp2).map_err(|error| format!("write {}: {error}", output.display()))
}

fn validate_tool_gates() -> Result<(), String> {
    let all_tools = all_encoder_tools()?;
    let selected_tools = selected_encoder_tools(&all_tools)?;
    if env_truthy("J2K_REQUIRE_OPENJPEG") && !tool_available(&all_tools, EncoderKind::OpenJpeg) {
        return Err("J2K_REQUIRE_OPENJPEG is set but opj_compress is unavailable".to_string());
    }
    if env_truthy("J2K_REQUIRE_GROK") && !tool_available(&all_tools, EncoderKind::Grok) {
        return Err("J2K_REQUIRE_GROK is set but grk_compress is unavailable".to_string());
    }
    if env_truthy("J2K_REQUIRE_KAKADU") && !tool_available(&all_tools, EncoderKind::Kakadu) {
        return Err(
            "J2K_REQUIRE_KAKADU is set but kdu_compress is unavailable; set J2K_KDU_COMPRESS_BIN"
                .to_string(),
        );
    }
    if env_truthy("J2K_REQUIRE_OPENJPEG")
        && !selected_tools
            .iter()
            .any(|tool| tool.kind == EncoderKind::OpenJpeg)
    {
        return Err(
            "J2K_REQUIRE_OPENJPEG is set but J2K_ENCODE_COMPARE_ENCODERS excludes openjpeg"
                .to_string(),
        );
    }
    if env_truthy("J2K_REQUIRE_GROK")
        && !selected_tools
            .iter()
            .any(|tool| tool.kind == EncoderKind::Grok)
    {
        return Err(
            "J2K_REQUIRE_GROK is set but J2K_ENCODE_COMPARE_ENCODERS excludes grok".to_string(),
        );
    }
    if env_truthy("J2K_REQUIRE_KAKADU")
        && !selected_tools
            .iter()
            .any(|tool| tool.kind == EncoderKind::Kakadu)
    {
        return Err(
            "J2K_REQUIRE_KAKADU is set but J2K_ENCODE_COMPARE_ENCODERS excludes kakadu".to_string(),
        );
    }
    Ok(())
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

fn include_generated_images() -> bool {
    !env_falsey("J2K_ENCODE_COMPARE_INCLUDE_GENERATED")
}

fn include_kakadu_encoder() -> bool {
    env_truthy("J2K_INCLUDE_KAKADU")
        || env_truthy("J2K_REQUIRE_KAKADU")
        || std::env::var("J2K_ENCODE_COMPARE_ENCODERS")
            .ok()
            .is_some_and(|value| {
                value
                    .split(',')
                    .map(str::trim)
                    .map(str::to_ascii_lowercase)
                    .any(|part| matches!(part.as_str(), "kakadu" | "kdu"))
            })
}

struct BatchSizeConfig {
    case_batch_sizes: Vec<usize>,
    mixed_batch_sizes: Vec<usize>,
}

fn batch_size_config_from_env() -> Result<BatchSizeConfig, String> {
    let case_batch_sizes = std::env::var("J2K_ENCODE_COMPARE_CASE_BATCH_SIZES").ok();
    let mixed_batch_sizes = std::env::var("J2K_ENCODE_COMPARE_MIXED_BATCH_SIZES").ok();
    let legacy = legacy_batch_sizes_from_env()?;
    batch_size_config_from_values(
        case_batch_sizes.as_deref(),
        mixed_batch_sizes.as_deref(),
        legacy,
    )
}

fn batch_size_config_from_values(
    case_batch_sizes: Option<&str>,
    mixed_batch_sizes: Option<&str>,
    legacy: Option<Vec<usize>>,
) -> Result<BatchSizeConfig, String> {
    let case_batch_sizes = if let Some(value) = case_batch_sizes {
        parse_batch_sizes(value, "J2K_ENCODE_COMPARE_CASE_BATCH_SIZES")?
    } else if let Some(values) = &legacy {
        values.clone()
    } else {
        DEFAULT_CASE_BATCH_SIZES.to_vec()
    };
    let mixed_batch_sizes = if let Some(value) = mixed_batch_sizes {
        parse_batch_sizes(value, "J2K_ENCODE_COMPARE_MIXED_BATCH_SIZES")?
    } else if let Some(values) = legacy {
        values
    } else {
        DEFAULT_MIXED_BATCH_SIZES.to_vec()
    };
    Ok(BatchSizeConfig {
        case_batch_sizes,
        mixed_batch_sizes,
    })
}

fn legacy_batch_sizes_from_env() -> Result<Option<Vec<usize>>, String> {
    if let Ok(value) = std::env::var("J2K_ENCODE_COMPARE_BATCH_SIZES") {
        return parse_batch_sizes(&value, "J2K_ENCODE_COMPARE_BATCH_SIZES").map(Some);
    }
    Ok(None)
}

fn combined_batch_sizes(case_batch_sizes: &[usize], mixed_batch_sizes: &[usize]) -> Vec<usize> {
    let mut values = case_batch_sizes
        .iter()
        .chain(mixed_batch_sizes.iter())
        .copied()
        .collect::<Vec<_>>();
    values.sort_unstable();
    values.dedup();
    values
}

fn parse_batch_sizes(value: &str, label: &str) -> Result<Vec<usize>, String> {
    let values = value
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(|part| parse_positive_usize(part, label))
        .collect::<Result<Vec<_>, _>>()?;
    if values.is_empty() {
        return Err(format!("{label} must include at least one batch size"));
    }
    Ok(values)
}

fn encode_work_dir() -> Result<PathBuf, String> {
    let dir = std::env::current_dir()
        .map_err(|error| format!("current_dir: {error}"))?
        .join("target")
        .join("j2k-encode-compare")
        .join(std::process::id().to_string());
    fs::create_dir_all(&dir).map_err(|error| format!("create {}: {error}", dir.display()))?;
    Ok(dir)
}

fn all_image_cases(work_dir: &Path) -> Result<Vec<ImageCase>, String> {
    let manifest = encode_manifest_from_env()?;
    let mut cases = if include_generated_images() {
        generated_image_cases(work_dir)?
    } else {
        Vec::new()
    };
    for dir in external_input_dirs() {
        cases.extend(load_external_image_cases(
            &dir,
            work_dir,
            manifest.as_ref(),
        )?);
    }
    if cases.is_empty() {
        return Err(
            "no encode image cases available; enable generated images or set J2K_ENCODE_COMPARE_INPUT_DIRS"
                .to_string(),
        );
    }
    Ok(cases)
}

fn generated_image_cases(work_dir: &Path) -> Result<Vec<ImageCase>, String> {
    let mut cases = Vec::new();
    for (name, width, height, components, pixels) in [
        (
            "generated_gray8_128",
            128,
            128,
            1,
            patterned_gray8(128, 128),
        ),
        ("generated_rgb8_128", 128, 128, 3, patterned_rgb8(128, 128)),
        ("generated_rgb8_512", 512, 512, 3, patterned_rgb8(512, 512)),
    ] {
        let pnm_path = work_dir.join(format!("{name}.{}", pnm_extension(components)?));
        write_pnm(&pnm_path, &pixels, width, height, components)?;
        cases.push(ImageCase {
            name: name.to_string(),
            input_source: "j2k-generated-image".to_string(),
            corpus_category: "generated-dev".to_string(),
            corpus_name: "j2k-generated-encode-matrix".to_string(),
            license_status: "repo-generated".to_string(),
            source_command: "j2k-test-support-pattern".to_string(),
            manifest_status: "generated".to_string(),
            source_format: "generated-pnm".to_string(),
            width,
            height,
            components,
            pixels,
            pnm_path,
        });
    }
    Ok(cases)
}

fn external_input_dirs() -> Vec<PathBuf> {
    if let Some(paths) = std::env::var_os("J2K_ENCODE_COMPARE_INPUT_DIRS") {
        return std::env::split_paths(&paths).collect();
    }
    Vec::new()
}

fn load_external_image_cases(
    dir: &Path,
    work_dir: &Path,
    manifest: Option<&EncodeManifest>,
) -> Result<Vec<ImageCase>, String> {
    if !dir.is_dir() {
        return Err(format!(
            "J2K_ENCODE_COMPARE_INPUT_DIRS entry is not a directory: {}",
            dir.display()
        ));
    }
    let mut paths = Vec::new();
    collect_source_image_paths(dir, &mut paths)?;
    paths.sort();
    if paths.is_empty() {
        return Err(format!(
            "external encode input dir {} contains no supported source images (.pgm/.ppm/.pnm/.png/.jpg/.jpeg/.tif/.tiff/.bmp)",
            dir.display()
        ));
    }
    let mut cases = Vec::new();
    for (index, path) in paths.into_iter().enumerate() {
        let parsed = read_source_image(&path)?;
        let metadata = external_image_metadata(&path, &parsed, manifest)?;
        let name = format!("external_{index:04}_{}", sanitized_stem(&path));
        let pnm_path = work_dir.join(format!("{}.{}", name, pnm_extension(parsed.components)?));
        write_pnm(
            &pnm_path,
            &parsed.pixels,
            parsed.width,
            parsed.height,
            parsed.components,
        )?;
        cases.push(ImageCase {
            name,
            input_source: metadata.input_source,
            corpus_category: metadata.corpus_category,
            corpus_name: metadata.corpus_name,
            license_status: metadata.license_status,
            source_command: metadata.source_command,
            manifest_status: metadata.manifest_status,
            source_format: source_format_label(&path),
            width: parsed.width,
            height: parsed.height,
            components: parsed.components,
            pixels: parsed.pixels,
            pnm_path,
        });
    }
    Ok(cases)
}

fn encode_manifest_from_env() -> Result<Option<EncodeManifest>, String> {
    let Some(path) = std::env::var_os("J2K_ENCODE_COMPARE_MANIFEST").map(PathBuf::from) else {
        return Ok(None);
    };
    let text = fs::read_to_string(&path).map_err(|error| {
        format!(
            "read J2K_ENCODE_COMPARE_MANIFEST {}: {error}",
            path.display()
        )
    })?;
    let base = path.parent().unwrap_or_else(|| Path::new("."));
    let mut lines = text.lines().filter(|line| !line.trim().is_empty());
    let header = lines
        .next()
        .ok_or_else(|| format!("encode manifest {} is empty", path.display()))?;
    let headers = header.split('\t').collect::<Vec<_>>();
    let path_index = manifest_column(&headers, "path")?;
    let category_index = manifest_column(&headers, "corpus_category")?;
    let corpus_name_index = optional_manifest_column(&headers, "corpus_name");
    let license_status_index = optional_manifest_column(&headers, "license_status");
    let source_command_index = optional_manifest_column(&headers, "source_command");
    let hash_index = optional_manifest_column(&headers, "input_fnv1a64");

    let mut entries = HashMap::new();
    for (line_index, line) in lines.enumerate() {
        if line.trim_start().starts_with('#') {
            continue;
        }
        let fields = line.split('\t').collect::<Vec<_>>();
        let row_number = line_index + 2;
        let raw_path = manifest_field(&fields, path_index, "path", row_number)?;
        let resolved_path = if Path::new(raw_path).is_absolute() {
            PathBuf::from(raw_path)
        } else {
            base.join(raw_path)
        };
        let canonical_path = resolved_path.canonicalize().map_err(|error| {
            format!(
                "encode manifest {} row {row_number} path {} cannot be canonicalized: {error}",
                path.display(),
                resolved_path.display()
            )
        })?;
        let entry = EncodeManifestEntry {
            corpus_category: manifest_required_value(
                &fields,
                category_index,
                "corpus_category",
                row_number,
            )?,
            corpus_name: manifest_optional_value(
                &fields,
                corpus_name_index,
                "corpus_name",
                row_number,
            )?
            .unwrap_or_else(|| "not-recorded".to_string()),
            license_status: manifest_optional_value(
                &fields,
                license_status_index,
                "license_status",
                row_number,
            )?
            .unwrap_or_else(|| "not-recorded".to_string()),
            source_command: manifest_optional_value(
                &fields,
                source_command_index,
                "source_command",
                row_number,
            )?
            .unwrap_or_else(|| "not-recorded".to_string()),
            input_fnv1a64: manifest_optional_value(
                &fields,
                hash_index,
                "input_fnv1a64",
                row_number,
            )?,
        };
        if entries.insert(canonical_path, entry).is_some() {
            return Err(format!(
                "encode manifest {} row {row_number} duplicates path {raw_path}",
                path.display()
            ));
        }
    }

    Ok(Some(EncodeManifest { entries }))
}

fn external_image_metadata(
    path: &Path,
    image: &PnmImage,
    manifest: Option<&EncodeManifest>,
) -> Result<ExternalImageMetadata, String> {
    let input_source = external_source_label(path)?;
    let Some(manifest) = manifest else {
        return Ok(ExternalImageMetadata {
            input_source,
            corpus_category: external_corpus_category(path),
            corpus_name: "path-inferred".to_string(),
            license_status: "not-recorded".to_string(),
            source_command: image.source_command.clone(),
            manifest_status: "not-covered".to_string(),
        });
    };
    let canonical_path = path
        .canonicalize()
        .map_err(|error| format!("canonicalize external image {}: {error}", path.display()))?;
    let Some(entry) = manifest.entries.get(&canonical_path) else {
        return Ok(ExternalImageMetadata {
            input_source,
            corpus_category: external_corpus_category(path),
            corpus_name: "path-inferred".to_string(),
            license_status: "not-recorded".to_string(),
            source_command: image.source_command.clone(),
            manifest_status: "not-covered".to_string(),
        });
    };
    if let Some(expected_hash) = &entry.input_fnv1a64 {
        let actual_hash = fnv1a64_hex(&image.pixels);
        if actual_hash != *expected_hash {
            return Err(format!(
                "external encode image {} hash mismatch: manifest {expected_hash} != actual {actual_hash}",
                path.display()
            ));
        }
    }
    let manifest_status = if entry.input_fnv1a64.is_some() {
        "covered"
    } else {
        "covered-unpinned"
    };

    Ok(ExternalImageMetadata {
        input_source,
        corpus_category: entry.corpus_category.clone(),
        corpus_name: entry.corpus_name.clone(),
        license_status: entry.license_status.clone(),
        source_command: entry.source_command.clone(),
        manifest_status: manifest_status.to_string(),
    })
}

fn manifest_column(headers: &[&str], name: &str) -> Result<usize, String> {
    optional_manifest_column(headers, name)
        .ok_or_else(|| format!("encode manifest is missing required {name:?} column"))
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
        .ok_or_else(|| format!("encode manifest row {row_number} is missing {name:?} field"))
}

fn manifest_required_value(
    fields: &[&str],
    index: usize,
    name: &str,
    row_number: usize,
) -> Result<String, String> {
    let value = manifest_field(fields, index, name, row_number)?.trim();
    if value.is_empty() {
        return Err(format!(
            "encode manifest row {row_number} has empty required {name:?} field"
        ));
    }
    validate_manifest_value(value, name, row_number)?;
    Ok(value.to_string())
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
    validate_manifest_value(value, name, row_number)?;
    Ok(Some(value.to_string()))
}

fn validate_manifest_value(value: &str, name: &str, row_number: usize) -> Result<(), String> {
    if value.chars().any(char::is_control) {
        return Err(format!(
            "encode manifest row {row_number} field {name:?} contains a control character"
        ));
    }
    Ok(())
}

fn collect_source_image_paths(dir: &Path, paths: &mut Vec<PathBuf>) -> Result<(), String> {
    for entry in
        fs::read_dir(dir).map_err(|error| format!("read input dir {}: {error}", dir.display()))?
    {
        let path = entry
            .map_err(|error| format!("read input dir entry: {error}"))?
            .path();
        if path.is_dir() {
            collect_source_image_paths(&path, paths)?;
        } else if is_supported_source_image_path(&path) {
            paths.push(path);
        }
    }
    Ok(())
}

fn is_supported_source_image_path(path: &Path) -> bool {
    path.extension()
        .and_then(|value| value.to_str())
        .is_some_and(|extension| {
            matches!(
                extension.to_ascii_lowercase().as_str(),
                "pgm" | "ppm" | "pnm" | "png" | "jpg" | "jpeg" | "tif" | "tiff" | "bmp"
            )
        })
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

fn external_source_label(path: &Path) -> Result<String, String> {
    let source_path = path.display().to_string();
    if source_path.chars().any(char::is_control) {
        return Err(format!(
            "external image path contains a control character: {}",
            source_path.escape_debug()
        ));
    }
    Ok(format!("external:{source_path}"))
}

fn external_corpus_category(path: &Path) -> String {
    let lower = path.to_string_lossy().to_ascii_lowercase();
    for (needle, category) in [
        ("kodak", "natural-image"),
        ("tecnick", "natural-image"),
        ("clic", "natural-image"),
        ("div2k", "natural-image"),
        ("flickr2k", "natural-image"),
        ("sipi", "natural-image"),
        ("wsi", "medical-domain"),
        ("dicom", "medical-domain"),
        ("tcia", "medical-domain"),
        ("idc", "medical-domain"),
        ("sentinel", "remote-sensing"),
        ("landsat", "remote-sensing"),
        ("nitf", "remote-sensing"),
    ] {
        if lower.contains(needle) {
            return category.to_string();
        }
    }
    "external-unspecified".to_string()
}

fn source_format_label(path: &Path) -> String {
    path.extension()
        .and_then(|value| value.to_str())
        .map_or_else(|| "unknown".to_string(), str::to_ascii_lowercase)
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

fn select_cases(cases: Vec<ImageCase>, filters: &[&str]) -> Result<Vec<ImageCase>, String> {
    if filters.is_empty() {
        return Ok(cases);
    }
    let selected = cases
        .into_iter()
        .filter(|case| filters.iter().any(|filter| case.name.contains(filter)))
        .collect::<Vec<_>>();
    if selected.is_empty() {
        return Err(format!(
            "no encode cases matched filters: {}",
            filters.join(",")
        ));
    }
    Ok(selected)
}

fn mixed_external_batches(cases: &[ImageCase]) -> Vec<MixedImageBatch> {
    let mut batches = Vec::new();
    for components in [1, 3] {
        let group = cases
            .iter()
            .filter(|case| {
                case.input_source.starts_with("external:") && case.components == components
            })
            .cloned()
            .collect::<Vec<_>>();
        if unique_image_count(&group) > 1 {
            let label = if components == 1 { "gray8" } else { "rgb8" };
            batches.push(MixedImageBatch {
                name: format!("external_mixed_{label}_encode"),
                cases: group,
                components,
            });
        }
    }
    batches
}

fn all_encoder_tools() -> Result<Vec<EncoderTool>, String> {
    let current = std::env::current_exe().map_err(|error| format!("current_exe: {error}"))?;
    let openjpeg_program = discover_command(
        "J2K_OPENJPEG_COMPRESS_BIN",
        "opj_compress",
        &[
            "/opt/homebrew/bin/opj_compress",
            "/usr/local/bin/opj_compress",
        ],
    );
    let grok_program = discover_command(
        "J2K_GROK_COMPRESS_BIN",
        "grk_compress",
        &[
            "/opt/homebrew/bin/grk_compress",
            "/usr/local/bin/grk_compress",
        ],
    );
    let kakadu_program = discover_command(
        "J2K_KDU_COMPRESS_BIN",
        "kdu_compress",
        &[
            "/opt/homebrew/bin/kdu_compress",
            "/usr/local/bin/kdu_compress",
        ],
    );
    let mut tools = vec![
        EncoderTool {
            kind: EncoderKind::J2k,
            program: current,
            available: true,
        },
        EncoderTool {
            kind: EncoderKind::OpenJpeg,
            program: openjpeg_program
                .clone()
                .unwrap_or_else(|| PathBuf::from("opj_compress")),
            available: openjpeg_program.is_some(),
        },
        EncoderTool {
            kind: EncoderKind::Grok,
            program: grok_program
                .clone()
                .unwrap_or_else(|| PathBuf::from("grk_compress")),
            available: grok_program.is_some(),
        },
    ];
    if include_kakadu_encoder() {
        tools.push(EncoderTool {
            kind: EncoderKind::Kakadu,
            program: kakadu_program
                .clone()
                .unwrap_or_else(|| PathBuf::from("kdu_compress")),
            available: kakadu_program.is_some(),
        });
    }
    Ok(tools)
}

fn selected_encoder_tools(all_tools: &[EncoderTool]) -> Result<Vec<EncoderTool>, String> {
    let Some(selected) = selected_encoder_kinds()? else {
        return Ok(all_tools.to_vec());
    };
    Ok(selected
        .into_iter()
        .filter_map(|kind| all_tools.iter().find(|tool| tool.kind == kind).cloned())
        .collect())
}

fn selected_encoder_kinds() -> Result<Option<Vec<EncoderKind>>, String> {
    let Some(value) = std::env::var("J2K_ENCODE_COMPARE_ENCODERS").ok() else {
        return Ok(None);
    };
    let mut kinds = Vec::new();
    for raw in value
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
    {
        let kind = match raw.to_ascii_lowercase().as_str() {
            "j2k" => EncoderKind::J2k,
            "openjpeg" | "opj" => EncoderKind::OpenJpeg,
            "grok" | "grk" => EncoderKind::Grok,
            "kakadu" | "kdu" => EncoderKind::Kakadu,
            other => {
                return Err(format!(
                    "J2K_ENCODE_COMPARE_ENCODERS has unknown encoder {other:?}; expected j2k, openjpeg, grok, or kakadu"
                ));
            }
        };
        if !kinds.contains(&kind) {
            kinds.push(kind);
        }
    }
    if kinds.is_empty() {
        return Err("J2K_ENCODE_COMPARE_ENCODERS must include at least one encoder".to_string());
    }
    Ok(Some(kinds))
}

fn discover_command(env_name: &str, program: &str, fallbacks: &[&str]) -> Option<PathBuf> {
    if let Some(path) = std::env::var_os(env_name)
        .map(PathBuf::from)
        .filter(|path| path.exists())
    {
        return Some(path);
    }
    if let Some(path) = path_lookup(program) {
        return Some(path);
    }
    fallbacks
        .iter()
        .map(PathBuf::from)
        .find(|path| path.exists())
}

fn path_lookup(program: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        let candidate = dir.join(program);
        if candidate.exists() {
            return Some(candidate);
        }
    }
    None
}

fn measure_case_rows(
    case: &ImageCase,
    tools: &[EncoderTool],
    repeats: usize,
    batch_size: usize,
    work_dir: &Path,
) -> Result<Vec<String>, String> {
    let mut rows = Vec::new();
    let mut states = Vec::new();
    for tool in tools.iter().filter(|tool| tool.available) {
        validate_case_encoder(case, tool, work_dir)?;
        states.push(EncodeMeasurementState {
            tool,
            encoded_bytes_per_repeat: None,
            samples_us: Vec::with_capacity(repeats),
        });
    }
    measure_case_states(case, &mut states, repeats, batch_size, work_dir)?;
    for tool in tools {
        if !tool.available {
            rows.push(skip_row(
                tool.kind,
                case,
                repeats,
                batch_size,
                "encoder-tool-unavailable",
                command_template(tool.kind),
            ));
            continue;
        }
        let state = states
            .iter()
            .find(|state| state.tool.kind == tool.kind)
            .ok_or_else(|| format!("missing measurement for {}", tool.kind.label()))?;
        let measurement = measurement(
            repeats,
            batch_size,
            state.samples_us.clone(),
            state.encoded_bytes_per_repeat,
        )?;
        rows.push(measurement_row(
            tool.kind,
            case,
            &measurement,
            command_template(tool.kind),
        ));
    }
    Ok(rows)
}

fn measure_mixed_rows(
    mixed_batch: &MixedImageBatch,
    tools: &[EncoderTool],
    repeats: usize,
    batch_size: usize,
    work_dir: &Path,
) -> Result<Vec<String>, String> {
    let mut rows = Vec::new();
    let mut states = tools
        .iter()
        .filter(|tool| tool.available)
        .map(|tool| EncodeMeasurementState {
            tool,
            encoded_bytes_per_repeat: None,
            samples_us: Vec::with_capacity(repeats),
        })
        .collect::<Vec<_>>();
    measure_mixed_states(mixed_batch, &mut states, repeats, batch_size, work_dir)?;
    for tool in tools {
        if !tool.available {
            rows.push(mixed_skip_row(
                tool.kind,
                mixed_batch,
                repeats,
                batch_size,
                "encoder-tool-unavailable",
                command_template(tool.kind),
            ));
            continue;
        }
        let state = states
            .iter()
            .find(|state| state.tool.kind == tool.kind)
            .ok_or_else(|| format!("missing measurement for {}", tool.kind.label()))?;
        let measurement = measurement(
            repeats,
            batch_size,
            state.samples_us.clone(),
            state.encoded_bytes_per_repeat,
        )?;
        rows.push(mixed_measurement_row(
            tool.kind,
            mixed_batch,
            &measurement,
            command_template(tool.kind),
        ));
    }
    Ok(rows)
}

fn validate_case_encoder(
    case: &ImageCase,
    tool: &EncoderTool,
    work_dir: &Path,
) -> Result<(), String> {
    let output = run_encoder_once(case, tool, work_dir, "validate")?;
    validate_encoded_profile(&output, case, tool.kind)?;
    let decoded = decode_encoded_output(&output, case)?;
    if decoded != case.pixels {
        return Err(format!(
            "{} {} output did not round-trip losslessly",
            tool.kind.label(),
            case.name
        ));
    }
    Ok(())
}

fn validate_encoded_profile(
    path: &Path,
    case: &ImageCase,
    encoder: EncoderKind,
) -> Result<(), String> {
    let bytes = fs::read(path).map_err(|error| format!("read {}: {error}", path.display()))?;
    let codestream = jp2_codestream_payload(&bytes)?;
    let header = j2k_native::inspect_j2k_codestream_header(codestream)
        .map_err(|error| format!("inspect {} profile: {error}", path.display()))?;
    if header.dimensions != (case.width, case.height) {
        return Err(format!(
            "{} {} profile dimensions {:?} != expected {:?}",
            encoder.label(),
            case.name,
            header.dimensions,
            (case.width, case.height)
        ));
    }
    if header.components != u16::from(case.components) {
        return Err(format!(
            "{} {} profile components {} != expected {}",
            encoder.label(),
            case.name,
            header.components,
            case.components
        ));
    }
    if header.tile_count != (1, 1) {
        return Err(format!(
            "{} {} profile tile count {:?} != expected single tile",
            encoder.label(),
            case.name,
            header.tile_count
        ));
    }
    if header.resolution_levels != 3 {
        return Err(format!(
            "{} {} profile resolution levels {} != expected 3",
            encoder.label(),
            case.name,
            header.resolution_levels
        ));
    }
    if !header.reversible {
        return Err(format!(
            "{} {} profile is not reversible 5/3",
            encoder.label(),
            case.name
        ));
    }
    if header.high_throughput {
        return Err(format!(
            "{} {} profile used HT block coding, expected classic",
            encoder.label(),
            case.name
        ));
    }
    if case.components == 3 && !header.has_mct {
        return Err(format!(
            "{} {} profile missing RGB reversible color transform",
            encoder.label(),
            case.name
        ));
    }
    if case.components == 1 && header.has_mct {
        return Err(format!(
            "{} {} grayscale profile unexpectedly enables MCT",
            encoder.label(),
            case.name
        ));
    }

    let cod = cod_profile(codestream)?;
    if cod.progression_order != 0 {
        return Err(format!(
            "{} {} profile progression order {} != LRCP",
            encoder.label(),
            case.name,
            cod.progression_order
        ));
    }
    if cod.decomposition_levels != 2 {
        return Err(format!(
            "{} {} profile decomposition levels {} != expected 2",
            encoder.label(),
            case.name,
            cod.decomposition_levels
        ));
    }
    if cod.code_block_width_exp != 4 || cod.code_block_height_exp != 4 {
        return Err(format!(
            "{} {} profile code-block exponents {},{} != expected 4,4",
            encoder.label(),
            case.name,
            cod.code_block_width_exp,
            cod.code_block_height_exp
        ));
    }
    if cod.code_block_style & 0x40 != 0 {
        return Err(format!(
            "{} {} profile used HT code-block style",
            encoder.label(),
            case.name
        ));
    }
    if cod.transform != 1 {
        return Err(format!(
            "{} {} profile transform {} != reversible 5/3",
            encoder.label(),
            case.name,
            cod.transform
        ));
    }
    if cod.scod & 0x01 != 0 {
        return Err(format!(
            "{} {} profile overrides precincts",
            encoder.label(),
            case.name
        ));
    }
    if cod.scod & 0x02 != 0 {
        return Err(format!(
            "{} {} profile enables SOP markers",
            encoder.label(),
            case.name
        ));
    }
    if cod.scod & 0x04 != 0 {
        return Err(format!(
            "{} {} profile enables EPH markers",
            encoder.label(),
            case.name
        ));
    }
    Ok(())
}

fn measure_case_states(
    case: &ImageCase,
    states: &mut [EncodeMeasurementState<'_>],
    repeats: usize,
    batch_size: usize,
    work_dir: &Path,
) -> Result<(), String> {
    if states.is_empty() {
        return Ok(());
    }
    for repeat in 0..repeats {
        let offset = repeat % states.len();
        for step in 0..states.len() {
            let index = (offset + step) % states.len();
            let state = &mut states[index];
            let (sample_us, encoded_bytes) = measure_case_encoder_once(
                case,
                state.tool,
                batch_size,
                work_dir,
                &format!("r{repeat}_e{step}"),
            )?;
            state.samples_us.push(sample_us);
            record_encoded_bytes(
                &mut state.encoded_bytes_per_repeat,
                encoded_bytes,
                state.tool.kind,
                &case.name,
            )?;
        }
    }
    Ok(())
}

fn measure_mixed_states(
    mixed_batch: &MixedImageBatch,
    states: &mut [EncodeMeasurementState<'_>],
    repeats: usize,
    batch_size: usize,
    work_dir: &Path,
) -> Result<(), String> {
    if states.is_empty() {
        return Ok(());
    }
    for repeat in 0..repeats {
        let offset = repeat % states.len();
        for step in 0..states.len() {
            let index = (offset + step) % states.len();
            let state = &mut states[index];
            let (sample_us, encoded_bytes) = measure_mixed_encoder_once(
                mixed_batch,
                state.tool,
                batch_size,
                work_dir,
                &format!("mixed_r{repeat}_e{step}"),
            )?;
            state.samples_us.push(sample_us);
            record_encoded_bytes(
                &mut state.encoded_bytes_per_repeat,
                encoded_bytes,
                state.tool.kind,
                &mixed_batch.name,
            )?;
        }
    }
    Ok(())
}

fn measure_case_encoder_once(
    case: &ImageCase,
    tool: &EncoderTool,
    batch_size: usize,
    work_dir: &Path,
    suffix: &str,
) -> Result<(f64, usize), String> {
    let started = Instant::now();
    let mut encoded_bytes = 0_usize;
    for index in 0..batch_size {
        let output = run_encoder_once(case, tool, work_dir, &format!("{suffix}_b{index}"))?;
        encoded_bytes += fs::metadata(&output)
            .map_err(|error| format!("metadata {}: {error}", output.display()))?
            .len() as usize;
        std::hint::black_box(&output);
    }
    Ok((started.elapsed().as_secs_f64() * 1_000_000.0, encoded_bytes))
}

fn measure_mixed_encoder_once(
    mixed_batch: &MixedImageBatch,
    tool: &EncoderTool,
    batch_size: usize,
    work_dir: &Path,
    suffix: &str,
) -> Result<(f64, usize), String> {
    let started = Instant::now();
    let mut encoded_bytes = 0_usize;
    for index in 0..batch_size {
        let case = mixed_case_at(mixed_batch, index);
        let output = run_encoder_once(case, tool, work_dir, &format!("{suffix}_b{index}"))?;
        encoded_bytes += fs::metadata(&output)
            .map_err(|error| format!("metadata {}: {error}", output.display()))?
            .len() as usize;
        std::hint::black_box(&output);
    }
    Ok((started.elapsed().as_secs_f64() * 1_000_000.0, encoded_bytes))
}

fn record_encoded_bytes(
    expected: &mut Option<usize>,
    actual: usize,
    encoder: EncoderKind,
    case_name: &str,
) -> Result<(), String> {
    if let Some(expected) = *expected {
        if actual != expected {
            return Err(format!(
                "{} {} encoded byte count changed: {} vs {expected}",
                encoder.label(),
                case_name,
                actual
            ));
        }
    } else {
        *expected = Some(actual);
    }
    Ok(())
}

fn run_encoder_once(
    case: &ImageCase,
    tool: &EncoderTool,
    work_dir: &Path,
    suffix: &str,
) -> Result<PathBuf, String> {
    let output = work_dir.join(format!(
        "{}_{}_{}.jp2",
        tool.kind.label(),
        case.name,
        suffix
    ));
    let mut command = Command::new(&tool.program);
    match tool.kind {
        EncoderKind::J2k => {
            command
                .arg("--encode-one")
                .arg("--input")
                .arg(&case.pnm_path)
                .arg("--output")
                .arg(&output);
        }
        EncoderKind::OpenJpeg => {
            command
                .arg("-i")
                .arg(&case.pnm_path)
                .arg("-o")
                .arg(&output)
                .arg("-n")
                .arg("3")
                .arg("-b")
                .arg("64,64")
                .arg("-p")
                .arg("LRCP")
                .arg("-threads")
                .arg("1")
                .env("OPJ_NUM_THREADS", "1");
        }
        EncoderKind::Grok => {
            command
                .arg("-i")
                .arg(&case.pnm_path)
                .arg("-o")
                .arg(&output)
                .arg("-n")
                .arg("3")
                .arg("-b")
                .arg("64,64")
                .arg("-p")
                .arg("LRCP")
                .arg("-H")
                .arg("1");
        }
        EncoderKind::Kakadu => {
            command
                .arg("-i")
                .arg(&case.pnm_path)
                .arg("-o")
                .arg(&output)
                .arg("Creversible=yes")
                .arg("Clevels=2")
                .arg("Cblk={64,64}")
                .arg("Corder=LRCP")
                .arg("-rate")
                .arg("-");
        }
    }
    let status = command
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map_err(|error| format!("start {}: {error}", tool.kind.label()))?;
    if !status.success() {
        return Err(format!(
            "{} encoder exited with {status} for {}",
            tool.kind.label(),
            case.name
        ));
    }
    Ok(output)
}

struct CodProfile {
    scod: u8,
    progression_order: u8,
    decomposition_levels: u8,
    code_block_width_exp: u8,
    code_block_height_exp: u8,
    code_block_style: u8,
    transform: u8,
}

fn jp2_codestream_payload(bytes: &[u8]) -> Result<&[u8], String> {
    if !bytes.starts_with(&[0, 0, 0, 12, b'j', b'P', b' ', b' ']) {
        return Err("encoded output is not a JP2 container".to_string());
    }
    let mut offset = 0_usize;
    while offset.checked_add(8).is_some_and(|end| end <= bytes.len()) {
        let lbox = u32::from_be_bytes(
            bytes[offset..offset + 4]
                .try_into()
                .map_err(|_| "invalid JP2 lbox".to_string())?,
        ) as usize;
        let box_type = &bytes[offset + 4..offset + 8];
        let (header_len, box_len) = match lbox {
            0 => (8, bytes.len() - offset),
            1 => {
                let xlbox_offset = offset
                    .checked_add(16)
                    .ok_or_else(|| "JP2 extended box offset overflow".to_string())?;
                if xlbox_offset > bytes.len() {
                    return Err("truncated JP2 extended box length".to_string());
                }
                let xlbox = u64::from_be_bytes(
                    bytes[offset + 8..offset + 16]
                        .try_into()
                        .map_err(|_| "invalid JP2 xlbox".to_string())?,
                );
                let box_len = usize::try_from(xlbox)
                    .map_err(|_| "JP2 extended box length overflows usize".to_string())?;
                (16, box_len)
            }
            value => (8, value),
        };
        if box_len < header_len {
            return Err("invalid JP2 box length".to_string());
        }
        let box_end = offset
            .checked_add(box_len)
            .ok_or_else(|| "JP2 box length overflow".to_string())?;
        if box_end > bytes.len() {
            return Err("truncated JP2 box".to_string());
        }
        if box_type == b"jp2c" {
            return Ok(&bytes[offset + header_len..box_end]);
        }
        offset = box_end;
    }
    Err("JP2 container is missing jp2c codestream box".to_string())
}

fn cod_profile(codestream: &[u8]) -> Result<CodProfile, String> {
    if !j2k_native::looks_like_j2k_codestream(codestream) {
        return Err("codestream is missing SOC marker".to_string());
    }
    let mut offset = 2_usize;
    while offset
        .checked_add(2)
        .is_some_and(|end| end <= codestream.len())
    {
        if codestream[offset] != 0xFF {
            return Err(format!("invalid codestream marker at offset {offset}"));
        }
        let marker = codestream[offset + 1];
        offset += 2;
        match marker {
            0x52 => {
                let payload = codestream_segment_payload(codestream, &mut offset, "COD")?;
                return parse_cod_profile(payload);
            }
            0x90 | 0x93 | 0xD9 => break,
            _ => {
                let _ = codestream_segment_payload(codestream, &mut offset, "marker segment")?;
            }
        }
    }
    Err("codestream is missing COD marker".to_string())
}

fn codestream_segment_payload<'a>(
    codestream: &'a [u8],
    offset: &mut usize,
    label: &str,
) -> Result<&'a [u8], String> {
    let length_end = offset
        .checked_add(2)
        .ok_or_else(|| format!("{label} length offset overflow"))?;
    if length_end > codestream.len() {
        return Err(format!("truncated {label} segment length"));
    }
    let length = u16::from_be_bytes([codestream[*offset], codestream[*offset + 1]]) as usize;
    if length < 2 {
        return Err(format!("invalid {label} segment length"));
    }
    let payload_start = *offset + 2;
    let segment_end = offset
        .checked_add(length)
        .ok_or_else(|| format!("{label} segment length overflow"))?;
    if segment_end > codestream.len() {
        return Err(format!("truncated {label} segment"));
    }
    *offset = segment_end;
    Ok(&codestream[payload_start..segment_end])
}

fn parse_cod_profile(payload: &[u8]) -> Result<CodProfile, String> {
    if payload.len() < 10 {
        return Err("COD payload is shorter than the fixed profile fields".to_string());
    }
    Ok(CodProfile {
        scod: payload[0],
        progression_order: payload[1],
        decomposition_levels: payload[5],
        code_block_width_exp: payload[6],
        code_block_height_exp: payload[7],
        code_block_style: payload[8],
        transform: payload[9],
    })
}

fn decode_encoded_output(path: &Path, case: &ImageCase) -> Result<Vec<u8>, String> {
    let bytes = fs::read(path).map_err(|error| format!("read {}: {error}", path.display()))?;
    let mut decoder = J2kDecoder::new(&bytes).map_err(|error| error.to_string())?;
    let format = case.pixel_format()?;
    let stride = case.width as usize * format.bytes_per_pixel();
    let mut out = vec![0_u8; stride * case.height as usize];
    decoder
        .decode_into(&mut out, stride, format)
        .map_err(|error| error.to_string())?;
    Ok(out)
}

fn measurement(
    repeats: usize,
    batch_size: usize,
    samples_us: Vec<f64>,
    encoded_bytes_per_repeat: Option<usize>,
) -> Result<Measurement, String> {
    let stats = sample_stats(&samples_us)?;
    Ok(Measurement {
        repeats,
        batch_size,
        median_us: stats.median,
        mean_us: stats.mean,
        images_per_second_median: usize_to_f64(batch_size) / (stats.median / 1_000_000.0),
        encoded_bytes_per_repeat: encoded_bytes_per_repeat
            .ok_or_else(|| "missing encoded byte count".to_string())?,
        samples_us,
    })
}

fn emit_metadata(input: MetadataInput<'_>) {
    let blockers = publication_blockers(&input);
    let MetadataInput {
        args,
        repeats,
        batch_sizes,
        case_batch_sizes,
        mixed_batch_sizes,
        cases,
        mixed_batches,
        selected_tools,
        all_tools,
        filters_empty: _,
    } = input;
    println!("command\t{}", args.join(" "));
    println!("benchmark_mode\tclassic-lossless-cli");
    println!("encode_method\tpnm-input-cli-process-output-jp2");
    println!(
        "encode_profile\tclassic-lossless-jp2-single-tile-lrcp-rct53-3resolutions-64x64-codeblocks-no-precinct-overrides-no-sop-eph"
    );
    println!("codec\tj2k");
    println!("container\tjp2");
    println!("repeats\t{repeats}");
    println!("batch_sizes\t{}", join_usizes(batch_sizes));
    println!("case_batch_sizes\t{}", join_usizes(case_batch_sizes));
    println!("mixed_batch_sizes\t{}", join_usizes(mixed_batch_sizes));
    println!("sample_order_policy\tinterleaved-rotating-encoder-order");
    println!("thread_policy\texternal-encoders-forced-single-thread-where-supported");
    println!(
        "selected_encoders\t{}",
        selected_encoders_label(selected_tools)
    );
    println!("j2k_compare_version\t{}", env!("CARGO_PKG_VERSION"));
    println!("host_os\t{}", std::env::consts::OS);
    println!("host_arch\t{}", std::env::consts::ARCH);
    println!("host_hardware\t{}", host_hardware_label());
    println!("build_profile\t{}", build_profile_label());
    println!("debug_assertions\t{}", cfg!(debug_assertions));
    println!("git_revision\t{}", git_revision_label());
    println!("git_dirty\t{}", git_dirty_label());
    println!("selected_cases\t{}", cases.len());
    println!("encode_manifest\t{}", encode_manifest_label());
    println!("generated_case_count\t{}", generated_case_count(cases));
    println!("external_case_count\t{}", external_case_count(cases));
    println!(
        "external_manifest_covered_case_count\t{}",
        external_manifest_covered_case_count(cases)
    );
    println!(
        "external_manifest_missing_case_count\t{}",
        external_manifest_missing_case_count(cases)
    );
    println!(
        "external_unique_input_count\t{}",
        external_unique_input_count(cases)
    );
    println!(
        "external_component_group_count\t{}",
        external_component_group_count(cases)
    );
    println!(
        "external_dimension_count\t{}",
        external_dimension_count(cases)
    );
    println!(
        "external_source_format_count\t{}",
        external_source_format_count(cases)
    );
    println!("mixed_external_batch_group_count\t{}", mixed_batches.len());
    println!(
        "mixed_external_max_distinct_inputs\t{}",
        mixed_external_max_distinct_inputs(mixed_batches)
    );
    println!(
        "mixed_external_min_distinct_inputs\t{}",
        mixed_external_min_distinct_inputs(mixed_batches)
    );
    println!(
        "mixed_external_group_distinct_inputs\t{}",
        mixed_external_group_distinct_inputs_label(mixed_batches)
    );
    println!("min_publication_external_input_count\t{MIN_PUBLICATION_EXTERNAL_IMAGES}");
    println!(
        "openjpeg_compress_available\t{}",
        tool_available(all_tools, EncoderKind::OpenJpeg)
    );
    println!(
        "openjpeg_compress_command\t{}",
        tool_command(all_tools, EncoderKind::OpenJpeg)
    );
    println!(
        "openjpeg_version\t{}",
        tool_version(all_tools, EncoderKind::OpenJpeg)
    );
    println!("openjpeg_linked_library_version\t{}", openjpeg::version());
    println!(
        "grok_compress_available\t{}",
        tool_available(all_tools, EncoderKind::Grok)
    );
    println!(
        "grok_compress_command\t{}",
        tool_command(all_tools, EncoderKind::Grok)
    );
    println!(
        "grok_version\t{}",
        tool_version(all_tools, EncoderKind::Grok)
    );
    println!("grok_linked_library_version\t{}", grok::version());
    println!("kakadu_included\t{}", include_kakadu_encoder());
    println!(
        "kakadu_compress_available\t{}",
        tool_available(all_tools, EncoderKind::Kakadu)
    );
    println!(
        "kakadu_compress_command\t{}",
        tool_command(all_tools, EncoderKind::Kakadu)
    );
    println!(
        "kakadu_version\t{}",
        tool_version(all_tools, EncoderKind::Kakadu)
    );
    println!("publication_eligible\t{}", blockers.is_empty());
    println!("publication_blockers\t{}", join_string_labels(&blockers));
}

fn publication_blockers(input: &MetadataInput<'_>) -> Vec<String> {
    let mut blockers = Vec::new();
    if cfg!(debug_assertions) {
        blockers.push("debug-build".to_string());
    }
    if git_revision().is_err() {
        blockers.push("git-revision-unavailable".to_string());
    }
    match git_dirty_status() {
        Ok("clean") => {}
        Ok(_) => blockers.push("git-worktree-dirty".to_string()),
        Err(_) => blockers.push("git-dirty-state-unavailable".to_string()),
    }
    if !input.filters_empty {
        blockers.push("case-filters-present".to_string());
    }
    if std::env::var_os("J2K_ENCODE_COMPARE_ENCODERS").is_some() {
        blockers.push("encoder-filter-present".to_string());
    }
    for required in [EncoderKind::J2k, EncoderKind::OpenJpeg, EncoderKind::Grok] {
        if !input
            .selected_tools
            .iter()
            .any(|tool| tool.kind == required)
        {
            blockers.push(format!("{}-not-selected", required.label()));
        }
    }
    if input.repeats < DEFAULT_REPEATS {
        blockers.push(format!("repeats-below-{DEFAULT_REPEATS}"));
    }
    if !default_case_batch_sizes_present(input.case_batch_sizes) {
        blockers.push(format!(
            "default-case-batch-sizes-missing:{}",
            join_usizes(DEFAULT_CASE_BATCH_SIZES)
        ));
    }
    if !default_mixed_batch_sizes_present(input.mixed_batch_sizes) {
        blockers.push(format!(
            "default-mixed-batch-sizes-missing:{}",
            join_usizes(DEFAULT_MIXED_BATCH_SIZES)
        ));
    }
    if !env_truthy("J2K_REQUIRE_OPENJPEG") {
        blockers.push("openjpeg-gate-not-required".to_string());
    }
    if !env_truthy("J2K_REQUIRE_GROK") {
        blockers.push("grok-gate-not-required".to_string());
    }
    if !tool_available(input.all_tools, EncoderKind::OpenJpeg) {
        blockers.push("openjpeg-compress-unavailable".to_string());
    }
    if !tool_available(input.all_tools, EncoderKind::Grok) {
        blockers.push("grok-compress-unavailable".to_string());
    }
    if !tool_version_available(input.all_tools, EncoderKind::OpenJpeg) {
        blockers.push("openjpeg-compress-version-unavailable".to_string());
    }
    if !tool_version_available(input.all_tools, EncoderKind::Grok) {
        blockers.push("grok-compress-version-unavailable".to_string());
    }
    if env_truthy("J2K_REQUIRE_KAKADU") && !tool_available(input.all_tools, EncoderKind::Kakadu) {
        blockers.push("kakadu-compress-unavailable".to_string());
    }
    let external_unique = external_unique_input_count(input.cases);
    if generated_case_count(input.cases) > 0 {
        blockers.push("generated-fixtures-included".to_string());
    }
    if external_unique < MIN_PUBLICATION_EXTERNAL_IMAGES {
        blockers.push(format!(
            "external-unique-input-count-below-{MIN_PUBLICATION_EXTERNAL_IMAGES}"
        ));
    }
    if input.mixed_batches.is_empty() {
        blockers.push("mixed-external-batches-missing".to_string());
    }
    if mixed_external_max_distinct_inputs(input.mixed_batches) < MIN_PUBLICATION_EXTERNAL_IMAGES {
        blockers.push(format!(
            "mixed-external-distinct-inputs-below-{MIN_PUBLICATION_EXTERNAL_IMAGES}"
        ));
    }
    for components in [1, 3] {
        require_mixed_encode_group(&mut blockers, input.cases, input.mixed_batches, components);
    }
    let component_groups = external_component_groups(input.cases);
    if !component_groups.contains(&1) {
        blockers.push("external-gray8-source-missing".to_string());
    }
    if !component_groups.contains(&3) {
        blockers.push("external-rgb8-source-missing".to_string());
    }
    if external_dimension_count(input.cases) < MIN_PUBLICATION_EXTERNAL_DIMENSIONS {
        blockers.push(format!(
            "external-dimension-diversity-below-{MIN_PUBLICATION_EXTERNAL_DIMENSIONS}"
        ));
    }
    if external_source_format_count(input.cases) < MIN_PUBLICATION_EXTERNAL_SOURCE_FORMATS {
        blockers.push(format!(
            "external-source-format-diversity-below-{MIN_PUBLICATION_EXTERNAL_SOURCE_FORMATS}"
        ));
    }
    if input
        .cases
        .iter()
        .filter(|case| case.input_source.starts_with("external:"))
        .any(|case| case.manifest_status != "covered")
    {
        blockers.push("external-manifest-coverage-missing".to_string());
    }
    if input
        .cases
        .iter()
        .filter(|case| case.input_source.starts_with("external:"))
        .any(|case| case.corpus_name == "path-inferred" || case.corpus_name == "not-recorded")
    {
        blockers.push("external-corpus-name-missing".to_string());
    }
    if input
        .cases
        .iter()
        .filter(|case| case.input_source.starts_with("external:"))
        .any(|case| case.license_status == "not-recorded")
    {
        blockers.push("external-license-status-missing".to_string());
    }
    if input
        .cases
        .iter()
        .filter(|case| case.input_source.starts_with("external:"))
        .any(|case| !is_publishable_license_status(&case.license_status))
    {
        blockers.push("external-license-status-not-publishable".to_string());
    }
    if input
        .cases
        .iter()
        .filter(|case| case.input_source.starts_with("external:"))
        .any(|case| case.source_command == "not-recorded")
    {
        blockers.push("external-source-command-missing".to_string());
    }
    if !input
        .cases
        .iter()
        .filter(|case| case.input_source.starts_with("external:"))
        .any(|case| {
            matches!(
                case.corpus_category.as_str(),
                "natural-image" | "medical-domain" | "remote-sensing"
            )
        })
    {
        blockers.push("external-workload-corpus-missing".to_string());
    }
    blockers
}

fn require_mixed_encode_group(
    blockers: &mut Vec<String>,
    cases: &[ImageCase],
    mixed_batches: &[MixedImageBatch],
    components: u8,
) {
    let external_count = external_unique_image_count_for_components(cases, components);
    let label = component_label(components);
    if external_count < MIN_PUBLICATION_MIXED_DISTINCT_INPUTS {
        blockers.push(format!(
            "external-{label}-mixed-input-count-below-{MIN_PUBLICATION_MIXED_DISTINCT_INPUTS}"
        ));
        return;
    }
    let mixed_count = mixed_unique_image_count_for_components(mixed_batches, components);
    if mixed_count < external_count {
        blockers.push(format!(
            "mixed-external-{label}-distinct-inputs-below-{external_count}"
        ));
    }
}

fn external_unique_image_count_for_components(cases: &[ImageCase], components: u8) -> usize {
    cases
        .iter()
        .filter(|case| case.input_source.starts_with("external:") && case.components == components)
        .map(ImageCase::input_digest)
        .collect::<HashSet<_>>()
        .len()
}

fn mixed_unique_image_count_for_components(
    mixed_batches: &[MixedImageBatch],
    components: u8,
) -> usize {
    mixed_batches
        .iter()
        .find(|mixed_batch| mixed_batch.components == components)
        .map_or(0, |mixed_batch| unique_image_count(&mixed_batch.cases))
}

fn component_label(components: u8) -> &'static str {
    match components {
        1 => "gray8",
        3 => "rgb8",
        _ => "unsupported",
    }
}

fn is_publishable_license_status(status: &str) -> bool {
    matches!(
        status.trim().to_ascii_lowercase().as_str(),
        "apache-2.0"
            | "bsd"
            | "bsd-2-clause"
            | "bsd-3-clause"
            | "cc-by"
            | "cc-by-4.0"
            | "cc0"
            | "mit"
            | "open-data"
            | "permissive"
            | "permissive-test-fixture"
            | "public-domain"
            | "redistributable"
            | "redistributable-with-attribution"
    )
}

fn default_case_batch_sizes_present(batch_sizes: &[usize]) -> bool {
    DEFAULT_CASE_BATCH_SIZES
        .iter()
        .all(|required| batch_sizes.contains(required))
}

fn default_mixed_batch_sizes_present(batch_sizes: &[usize]) -> bool {
    DEFAULT_MIXED_BATCH_SIZES
        .iter()
        .all(|required| batch_sizes.contains(required))
}

fn measurement_row(
    encoder: EncoderKind,
    case: &ImageCase,
    measurement: &Measurement,
    command_template: &'static str,
) -> String {
    [
        encoder.label().to_string(),
        case.name.clone(),
        "classic-lossless-cli".to_string(),
        "pnm-input-cli-process-output-jp2".to_string(),
        case.input_source.clone(),
        case.corpus_category.clone(),
        case.corpus_name.clone(),
        case.license_status.clone(),
        case.source_command.clone(),
        case.manifest_status.clone(),
        "j2k".to_string(),
        "jp2".to_string(),
        case.format_label().to_string(),
        dimensions_label(case.width, case.height),
        measurement.batch_size.to_string(),
        measurement.repeats.to_string(),
        case_input_bytes_per_repeat(case, measurement.batch_size).to_string(),
        case.input_digest(),
        format!("{:.3}", measurement.median_us),
        format!("{:.3}", measurement.mean_us),
        format!("{:.3}", measurement.images_per_second_median),
        format!(
            "{:.3}",
            mib_per_second(
                case_input_bytes_per_repeat(case, measurement.batch_size),
                measurement.median_us
            )
        ),
        measurement.encoded_bytes_per_repeat.to_string(),
        samples_label(&measurement.samples_us),
        String::new(),
        command_template.to_string(),
    ]
    .join("\t")
}

fn mixed_measurement_row(
    encoder: EncoderKind,
    mixed_batch: &MixedImageBatch,
    measurement: &Measurement,
    command_template: &'static str,
) -> String {
    [
        encoder.label().to_string(),
        mixed_batch.name.clone(),
        "classic-lossless-cli".to_string(),
        "pnm-input-cli-process-output-jp2".to_string(),
        "external:mixed".to_string(),
        mixed_case_value_label(mixed_batch, |case| case.corpus_category.as_str()),
        mixed_case_value_label(mixed_batch, |case| case.corpus_name.as_str()),
        mixed_case_value_label(mixed_batch, |case| case.license_status.as_str()),
        mixed_case_value_label(mixed_batch, |case| case.source_command.as_str()),
        mixed_case_value_label(mixed_batch, |case| case.manifest_status.as_str()),
        "j2k".to_string(),
        "jp2".to_string(),
        if mixed_batch.components == 1 {
            "gray8"
        } else {
            "rgb8"
        }
        .to_string(),
        "mixed".to_string(),
        measurement.batch_size.to_string(),
        measurement.repeats.to_string(),
        mixed_input_bytes_per_repeat(mixed_batch, measurement.batch_size).to_string(),
        mixed_input_digest(mixed_batch, measurement.batch_size),
        format!("{:.3}", measurement.median_us),
        format!("{:.3}", measurement.mean_us),
        format!("{:.3}", measurement.images_per_second_median),
        format!(
            "{:.3}",
            mib_per_second(
                mixed_input_bytes_per_repeat(mixed_batch, measurement.batch_size),
                measurement.median_us
            )
        ),
        measurement.encoded_bytes_per_repeat.to_string(),
        samples_label(&measurement.samples_us),
        String::new(),
        command_template.to_string(),
    ]
    .join("\t")
}

fn skip_row(
    encoder: EncoderKind,
    case: &ImageCase,
    repeats: usize,
    batch_size: usize,
    reason: &'static str,
    command_template: &'static str,
) -> String {
    [
        encoder.label().to_string(),
        case.name.clone(),
        "classic-lossless-cli".to_string(),
        "skipped".to_string(),
        case.input_source.clone(),
        case.corpus_category.clone(),
        case.corpus_name.clone(),
        case.license_status.clone(),
        case.source_command.clone(),
        case.manifest_status.clone(),
        "j2k".to_string(),
        "jp2".to_string(),
        case.format_label().to_string(),
        dimensions_label(case.width, case.height),
        batch_size.to_string(),
        repeats.to_string(),
        case.pixels.len().to_string(),
        case.input_digest(),
        "NA".to_string(),
        "NA".to_string(),
        "NA".to_string(),
        "NA".to_string(),
        "NA".to_string(),
        "NA".to_string(),
        reason.to_string(),
        command_template.to_string(),
    ]
    .join("\t")
}

fn mixed_skip_row(
    encoder: EncoderKind,
    mixed_batch: &MixedImageBatch,
    repeats: usize,
    batch_size: usize,
    reason: &'static str,
    command_template: &'static str,
) -> String {
    [
        encoder.label().to_string(),
        mixed_batch.name.clone(),
        "classic-lossless-cli".to_string(),
        "skipped".to_string(),
        "external:mixed".to_string(),
        mixed_case_value_label(mixed_batch, |case| case.corpus_category.as_str()),
        mixed_case_value_label(mixed_batch, |case| case.corpus_name.as_str()),
        mixed_case_value_label(mixed_batch, |case| case.license_status.as_str()),
        mixed_case_value_label(mixed_batch, |case| case.source_command.as_str()),
        mixed_case_value_label(mixed_batch, |case| case.manifest_status.as_str()),
        "j2k".to_string(),
        "jp2".to_string(),
        if mixed_batch.components == 1 {
            "gray8"
        } else {
            "rgb8"
        }
        .to_string(),
        "mixed".to_string(),
        batch_size.to_string(),
        repeats.to_string(),
        mixed_input_bytes_per_repeat(mixed_batch, batch_size).to_string(),
        mixed_input_digest(mixed_batch, batch_size),
        "NA".to_string(),
        "NA".to_string(),
        "NA".to_string(),
        "NA".to_string(),
        "NA".to_string(),
        "NA".to_string(),
        reason.to_string(),
        command_template.to_string(),
    ]
    .join("\t")
}

fn command_template(encoder: EncoderKind) -> &'static str {
    match encoder {
        EncoderKind::J2k => {
            "jp2k_encode_compare --encode-one --input INPUT.pnm --output OUTPUT.jp2"
        }
        EncoderKind::OpenJpeg => {
            "OPJ_NUM_THREADS=1 opj_compress -i INPUT.pnm -o OUTPUT.jp2 -n 3 -b 64,64 -p LRCP -threads 1"
        }
        EncoderKind::Grok => {
            "grk_compress -i INPUT.pnm -o OUTPUT.jp2 -n 3 -b 64,64 -p LRCP -H 1"
        }
        EncoderKind::Kakadu => {
            "kdu_compress -i INPUT.pnm -o OUTPUT.jp2 Creversible=yes Clevels=2 Cblk={64,64} Corder=LRCP -rate -"
        }
    }
}

fn samples_label(samples: &[f64]) -> String {
    samples
        .iter()
        .map(|value| format!("{value:.3}"))
        .collect::<Vec<_>>()
        .join(",")
}

fn dimensions_label(width: u32, height: u32) -> String {
    format!("{width}x{height}")
}

fn join_usizes(values: &[usize]) -> String {
    values
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

fn join_string_labels(values: &[String]) -> String {
    if values.is_empty() {
        "none".to_string()
    } else {
        values.join(",")
    }
}

fn tool_available(tools: &[EncoderTool], kind: EncoderKind) -> bool {
    tools.iter().any(|tool| tool.kind == kind && tool.available)
}

fn tool_command(tools: &[EncoderTool], kind: EncoderKind) -> String {
    tools.iter().find(|tool| tool.kind == kind).map_or_else(
        || "not found".to_string(),
        |tool| tool.program.display().to_string(),
    )
}

fn tool_version(tools: &[EncoderTool], kind: EncoderKind) -> String {
    let Some(tool) = tools.iter().find(|tool| tool.kind == kind) else {
        return "not found".to_string();
    };
    if !tool.available {
        return "unavailable".to_string();
    }
    command_version_label(tool).unwrap_or_else(|error| format!("unavailable:{error}"))
}

fn tool_version_available(tools: &[EncoderTool], kind: EncoderKind) -> bool {
    let Some(tool) = tools.iter().find(|tool| tool.kind == kind) else {
        return false;
    };
    tool.available && command_version_label(tool).is_ok()
}

fn command_version_label(tool: &EncoderTool) -> Result<String, String> {
    let arg_sets: &[&[&str]] = match tool.kind {
        EncoderKind::J2k => return Ok(env!("CARGO_PKG_VERSION").to_string()),
        EncoderKind::OpenJpeg => &[&["-h"]],
        EncoderKind::Grok => &[&["--help"], &["-h"]],
        EncoderKind::Kakadu => &[&["-usage"], &["-h"]],
    };
    for args in arg_sets {
        let output = Command::new(&tool.program)
            .args(*args)
            .output()
            .map_err(|error| format!("{}:{error}", tool.kind.label()))?;
        let mut text = String::new();
        text.push_str(&String::from_utf8_lossy(&output.stdout));
        text.push_str(&String::from_utf8_lossy(&output.stderr));
        if let Some(line) = extract_version_line(tool.kind, &text) {
            return Ok(line);
        }
    }
    if tool.kind == EncoderKind::Kakadu {
        Ok("available-version-not-reported-by-kdu_compress".to_string())
    } else {
        Err("version-line-not-found".to_string())
    }
}

fn extract_version_line(kind: EncoderKind, text: &str) -> Option<String> {
    version_line_by_priority(kind, text, true)
        .or_else(|| version_line_by_priority(kind, text, false))
}

fn version_line_by_priority(
    kind: EncoderKind,
    text: &str,
    prefer_compiled: bool,
) -> Option<String> {
    text.lines().find_map(|line| {
        let lower = line.to_ascii_lowercase();
        let compiled_match = match kind {
            EncoderKind::J2k => false,
            EncoderKind::OpenJpeg => lower.contains("compiled against openjp2"),
            EncoderKind::Grok => lower.contains("compiled against libgrok"),
            EncoderKind::Kakadu => lower.contains("kakadu"),
        };
        let fallback_match = match kind {
            EncoderKind::J2k => false,
            EncoderKind::OpenJpeg => lower.contains("openjpeg"),
            EncoderKind::Grok => lower.contains("grok"),
            EncoderKind::Kakadu => lower.contains("kdu_compress") || lower.contains("kakadu"),
        };
        let matches_priority = if prefer_compiled {
            compiled_match
        } else {
            fallback_match
        };
        matches_priority.then(|| line.split_whitespace().collect::<Vec<_>>().join(" "))
    })
}

fn selected_encoders_label(tools: &[EncoderTool]) -> String {
    tools
        .iter()
        .map(|tool| tool.kind.label())
        .collect::<Vec<_>>()
        .join(",")
}

fn generated_case_count(cases: &[ImageCase]) -> usize {
    cases
        .iter()
        .filter(|case| case.input_source.starts_with("j2k-generated"))
        .count()
}

fn external_case_count(cases: &[ImageCase]) -> usize {
    cases
        .iter()
        .filter(|case| case.input_source.starts_with("external:"))
        .count()
}

fn external_manifest_covered_case_count(cases: &[ImageCase]) -> usize {
    cases
        .iter()
        .filter(|case| {
            case.input_source.starts_with("external:") && case.manifest_status == "covered"
        })
        .count()
}

fn external_manifest_missing_case_count(cases: &[ImageCase]) -> usize {
    cases
        .iter()
        .filter(|case| {
            case.input_source.starts_with("external:") && case.manifest_status != "covered"
        })
        .count()
}

fn encode_manifest_label() -> String {
    std::env::var("J2K_ENCODE_COMPARE_MANIFEST").unwrap_or_else(|_| "not set".to_string())
}

fn build_profile_label() -> &'static str {
    if cfg!(debug_assertions) {
        "debug"
    } else {
        "release-like"
    }
}

fn git_revision_label() -> String {
    git_revision().unwrap_or_else(|error| format!("unavailable:{error}"))
}

fn git_revision() -> Result<String, String> {
    command_stdout("git", &["rev-parse", "--short=12", "HEAD"])
}

fn git_dirty_label() -> String {
    git_dirty_status().map_or_else(|error| format!("unavailable:{error}"), str::to_string)
}

fn git_dirty_status() -> Result<&'static str, String> {
    let output = Command::new("git")
        .args(["status", "--porcelain"])
        .output()
        .map_err(|error| format!("git:{error}"))?;
    if !output.status.success() {
        return Err(format!("git:status:{}", output.status));
    }
    if output.stdout.is_empty() {
        Ok("clean")
    } else {
        Ok("dirty")
    }
}

fn host_hardware_label() -> String {
    host_hardware_from_platform().unwrap_or_else(|error| format!("unavailable:{error}"))
}

#[cfg(target_os = "macos")]
fn host_hardware_from_platform() -> Result<String, String> {
    command_stdout("sysctl", &["-n", "machdep.cpu.brand_string"])
}

#[cfg(target_os = "linux")]
fn host_hardware_from_platform() -> Result<String, String> {
    let cpuinfo =
        fs::read_to_string("/proc/cpuinfo").map_err(|error| format!("cpuinfo:{error}"))?;
    cpuinfo
        .lines()
        .find_map(|line| line.strip_prefix("model name\t: "))
        .map(str::to_string)
        .ok_or_else(|| "cpuinfo:model-name-missing".to_string())
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn host_hardware_from_platform() -> Result<String, String> {
    Err("unsupported-platform".to_string())
}

fn command_stdout(program: &str, args: &[&str]) -> Result<String, String> {
    let output = Command::new(program)
        .args(args)
        .output()
        .map_err(|error| format!("{program}:{error}"))?;
    if !output.status.success() {
        return Err(format!("{program}:status:{}", output.status));
    }
    let stdout = String::from_utf8(output.stdout).map_err(|error| format!("{program}:{error}"))?;
    let trimmed = stdout.trim();
    if trimmed.is_empty() {
        Err(format!("{program}:empty-output"))
    } else {
        Ok(trimmed.to_string())
    }
}

fn external_unique_input_count(cases: &[ImageCase]) -> usize {
    unique_image_count(
        &cases
            .iter()
            .filter(|case| case.input_source.starts_with("external:"))
            .cloned()
            .collect::<Vec<_>>(),
    )
}

fn external_component_groups(cases: &[ImageCase]) -> HashSet<u8> {
    cases
        .iter()
        .filter(|case| case.input_source.starts_with("external:"))
        .map(|case| case.components)
        .collect()
}

fn external_component_group_count(cases: &[ImageCase]) -> usize {
    external_component_groups(cases).len()
}

fn external_dimension_count(cases: &[ImageCase]) -> usize {
    cases
        .iter()
        .filter(|case| case.input_source.starts_with("external:"))
        .map(|case| (case.width, case.height))
        .collect::<HashSet<_>>()
        .len()
}

fn external_source_format_count(cases: &[ImageCase]) -> usize {
    cases
        .iter()
        .filter(|case| case.input_source.starts_with("external:"))
        .map(|case| case.source_format.as_str())
        .collect::<HashSet<_>>()
        .len()
}

fn unique_image_count(cases: &[ImageCase]) -> usize {
    cases
        .iter()
        .map(ImageCase::input_digest)
        .collect::<HashSet<_>>()
        .len()
}

fn mixed_external_max_distinct_inputs(mixed_batches: &[MixedImageBatch]) -> usize {
    mixed_batches
        .iter()
        .map(|batch| unique_image_count(&batch.cases))
        .max()
        .unwrap_or(0)
}

fn mixed_external_min_distinct_inputs(mixed_batches: &[MixedImageBatch]) -> usize {
    mixed_batches
        .iter()
        .map(|batch| unique_image_count(&batch.cases))
        .min()
        .unwrap_or(0)
}

fn mixed_external_group_distinct_inputs_label(mixed_batches: &[MixedImageBatch]) -> String {
    if mixed_batches.is_empty() {
        return "none".to_string();
    }
    mixed_batches
        .iter()
        .map(|batch| format!("{}:{}", batch.name, unique_image_count(&batch.cases)))
        .collect::<Vec<_>>()
        .join(",")
}

fn case_input_bytes_per_repeat(case: &ImageCase, batch_size: usize) -> usize {
    case.pixels.len() * batch_size
}

fn mib_per_second(bytes: usize, elapsed_us: f64) -> f64 {
    if elapsed_us <= 0.0 {
        return 0.0;
    }
    (usize_to_f64(bytes) / (1024.0 * 1024.0)) / (elapsed_us / 1_000_000.0)
}

fn mixed_case_value_label(
    mixed_batch: &MixedImageBatch,
    value: impl Fn(&ImageCase) -> &str,
) -> String {
    let mut labels = Vec::new();
    for case in &mixed_batch.cases {
        let label = value(case);
        if !labels.contains(&label) {
            labels.push(label);
        }
    }
    if labels.len() == 1 {
        labels[0].to_string()
    } else {
        format!("mixed:{}", labels.join(","))
    }
}

fn mixed_input_bytes_per_repeat(mixed_batch: &MixedImageBatch, batch_size: usize) -> usize {
    (0..batch_size)
        .map(|index| mixed_case_at(mixed_batch, index).pixels.len())
        .sum()
}

fn mixed_input_digest(mixed_batch: &MixedImageBatch, batch_size: usize) -> String {
    let mut slices = Vec::with_capacity(batch_size);
    for index in 0..batch_size {
        slices.push(mixed_case_at(mixed_batch, index).pixels.as_slice());
    }
    fnv1a64_hex_slices(&slices)
}

fn mixed_case_at(mixed_batch: &MixedImageBatch, index: usize) -> &ImageCase {
    &mixed_batch.cases[index % mixed_batch.cases.len()]
}

struct PnmImage {
    width: u32,
    height: u32,
    components: u8,
    pixels: Vec<u8>,
    source_command: String,
}

fn write_pnm(
    path: &Path,
    pixels: &[u8],
    width: u32,
    height: u32,
    components: u8,
) -> Result<(), String> {
    let expected = width as usize * height as usize * usize::from(components);
    if pixels.len() != expected {
        return Err(format!(
            "PNM {} pixel length {} != expected {expected}",
            path.display(),
            pixels.len()
        ));
    }
    let mut file =
        fs::File::create(path).map_err(|error| format!("create {}: {error}", path.display()))?;
    let magic = if components == 1 { "P5" } else { "P6" };
    write!(file, "{magic}\n{width} {height}\n255\n")
        .map_err(|error| format!("write {}: {error}", path.display()))?;
    file.write_all(pixels)
        .map_err(|error| format!("write {}: {error}", path.display()))
}

fn read_source_image(path: &Path) -> Result<PnmImage, String> {
    if is_pnm_path(path) {
        return read_pnm(path);
    }
    read_raster_image(path)
}

fn read_raster_image(path: &Path) -> Result<PnmImage, String> {
    let reader = image::ImageReader::open(path)
        .map_err(|error| format!("open source image {}: {error}", path.display()))?
        .with_guessed_format()
        .map_err(|error| format!("guess source image format {}: {error}", path.display()))?;
    let decoded = reader
        .decode()
        .map_err(|error| format!("decode source image {}: {error}", path.display()))?;
    let width = decoded.width();
    let height = decoded.height();
    match decoded.color() {
        image::ColorType::L8 => Ok(PnmImage {
            width,
            height,
            components: 1,
            pixels: decoded.into_luma8().into_raw(),
            source_command: "image-crate-decode-to-pnm".to_string(),
        }),
        image::ColorType::Rgb8 => Ok(PnmImage {
            width,
            height,
            components: 3,
            pixels: decoded.into_rgb8().into_raw(),
            source_command: "image-crate-decode-to-pnm".to_string(),
        }),
        color => Err(format!(
            "{} has unsupported source color type {color:?}; expected 8-bit grayscale or RGB without alpha",
            path.display()
        )),
    }
}

fn read_pnm(path: &Path) -> Result<PnmImage, String> {
    let bytes = fs::read(path).map_err(|error| format!("read {}: {error}", path.display()))?;
    let mut parser = PnmParser::new(&bytes);
    let magic = parser.token(path)?;
    let components = match magic.as_slice() {
        b"P5" => 1,
        b"P6" => 3,
        _ => return Err(format!("{} is not binary PGM/PPM", path.display())),
    };
    let width = parse_pnm_u32(&parser.token(path)?, "width", path)?;
    let height = parse_pnm_u32(&parser.token(path)?, "height", path)?;
    let maxval = parse_pnm_u32(&parser.token(path)?, "maxval", path)?;
    if maxval != 255 {
        return Err(format!(
            "{} has unsupported maxval {maxval}; expected 255",
            path.display()
        ));
    }
    parser.skip_single_whitespace();
    let pixel_offset = parser.offset;
    let expected = width as usize * height as usize * usize::from(components);
    let pixels = bytes
        .get(pixel_offset..pixel_offset + expected)
        .ok_or_else(|| format!("{} is truncated", path.display()))?
        .to_vec();
    Ok(PnmImage {
        width,
        height,
        components,
        pixels,
        source_command: "source-pnm".to_string(),
    })
}

fn parse_pnm_u32(bytes: &[u8], label: &str, path: &Path) -> Result<u32, String> {
    std::str::from_utf8(bytes)
        .map_err(|error| format!("{} invalid {label}: {error}", path.display()))?
        .parse()
        .map_err(|error| format!("{} invalid {label}: {error}", path.display()))
}

struct PnmParser<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> PnmParser<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn token(&mut self, path: &Path) -> Result<Vec<u8>, String> {
        self.skip_ws_and_comments();
        let start = self.offset;
        while self
            .bytes
            .get(self.offset)
            .is_some_and(|byte| !byte.is_ascii_whitespace() && *byte != b'#')
        {
            self.offset += 1;
        }
        if self.offset == start {
            return Err(format!("{} has malformed PNM header", path.display()));
        }
        Ok(self.bytes[start..self.offset].to_vec())
    }

    fn skip_ws_and_comments(&mut self) {
        loop {
            while self
                .bytes
                .get(self.offset)
                .is_some_and(u8::is_ascii_whitespace)
            {
                self.offset += 1;
            }
            if self.bytes.get(self.offset) != Some(&b'#') {
                break;
            }
            while self
                .bytes
                .get(self.offset)
                .is_some_and(|byte| *byte != b'\n')
            {
                self.offset += 1;
            }
        }
    }

    fn skip_single_whitespace(&mut self) {
        if self
            .bytes
            .get(self.offset)
            .is_some_and(u8::is_ascii_whitespace)
        {
            self.offset += 1;
        }
    }
}

fn pnm_extension(components: u8) -> Result<&'static str, String> {
    match components {
        1 => Ok("pgm"),
        3 => Ok("ppm"),
        other => Err(format!("unsupported component count {other}")),
    }
}

fn fnv1a64_hex(bytes: &[u8]) -> String {
    fnv1a64_hex_slices(&[bytes])
}

fn fnv1a64_hex_slices(slices: &[&[u8]]) -> String {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for bytes in slices {
        for byte in *bytes {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    format!("{hash:016x}")
}

#[cfg(test)]
mod tests {
    use super::{
        batch_size_config_from_values, DEFAULT_CASE_BATCH_SIZES, DEFAULT_MIXED_BATCH_SIZES,
    };

    #[test]
    fn encode_batch_config_defaults_keep_large_batches_mixed_only() {
        let config =
            batch_size_config_from_values(None, None, None).expect("default batch config parses");

        assert_eq!(config.case_batch_sizes, DEFAULT_CASE_BATCH_SIZES);
        assert_eq!(config.mixed_batch_sizes, DEFAULT_MIXED_BATCH_SIZES);
    }

    #[test]
    fn encode_batch_config_split_env_overrides_legacy_independently() {
        let config = batch_size_config_from_values(Some("3"), None, Some(vec![2, 4]))
            .expect("case override with legacy config parses");

        assert_eq!(config.case_batch_sizes, vec![3]);
        assert_eq!(config.mixed_batch_sizes, vec![2, 4]);

        let config = batch_size_config_from_values(None, Some("8,16"), Some(vec![2, 4]))
            .expect("mixed override with legacy config parses");

        assert_eq!(config.case_batch_sizes, vec![2, 4]);
        assert_eq!(config.mixed_batch_sizes, vec![8, 16]);
    }
}
