// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{
    collections::{HashMap, HashSet},
    fs,
    num::NonZeroUsize,
    path::{Path, PathBuf},
    process::Command,
    sync::{
        atomic::{AtomicU64, Ordering},
        OnceLock,
    },
    time::Instant,
};

use j2k::{
    decode_tile_into_in_context, decode_tile_region_into_in_context,
    decode_tile_region_scaled_into_in_context, decode_tile_scaled_into_in_context,
    decode_tiles_into, decode_tiles_region_into, decode_tiles_region_scaled_into,
    decode_tiles_scaled_into, encode_j2k_lossless, wrap_j2k_codestream, CpuDecodeParallelism,
    DecoderContext, EncodeBackendPreference, J2kBlockCodingMode, J2kContext, J2kDecoder,
    J2kEncodeValidation, J2kFileWrapOptions, J2kLosslessEncodeOptions, J2kLosslessSamples,
    J2kScratchPool, TileBatchOptions, TileDecodeJob, TileRegionDecodeJob,
    TileRegionScaledDecodeJob, TileScaledDecodeJob,
};
use j2k_compare::{grok, openjpeg, parse_positive_usize, sample_stats, usize_to_f64};
use j2k_core::{tile_batch_worker_count, Downscale, PixelFormat, Rect};
use j2k_test_support::{
    fnv1a64_hex, fnv1a64_hex_slices, patterned_gray8, patterned_rgb8, wrap_jp2_codestream,
};

mod common;

use common::{
    build_profile_label, canonicalize_manifest_row_path, combined_batch_sizes,
    default_batch_sizes_present, env_falsey, env_truthy, git_dirty_label, git_dirty_status,
    git_revision, git_revision_label, host_hardware_label, is_publishable_license_status,
    join_string_labels, join_usizes, mib_per_second, optional_manifest_column, parse_batch_sizes,
    sanitized_stem,
};

const DEFAULT_REPEATS: usize = 5;
const DEFAULT_CASE_BATCH_SIZES: &[usize] = &[1];
const DEFAULT_MIXED_BATCH_SIZES: &[usize] = &[1, 16, 256, 1024];
const BATCH_INPUT_COPY_LIMIT: usize = 32;
const MIN_PUBLICATION_EXTERNAL_CASES: usize = 24;
const MIN_PUBLICATION_EXTERNAL_INPUTS: usize = 24;
const MIN_PUBLICATION_MIXED_DISTINCT_INPUTS: usize = 2;
const SMALL_SIDE: u32 = 128;
const LARGE_SIDE: u32 = 512;
const DEFAULT_BENCHMARK_MODE: BenchmarkMode = BenchmarkMode::PortableNative;
static OPENJPH_TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);
static KAKADU_TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);
static OPENJPH_EXPAND_PROGRAM: OnceLock<Option<PathBuf>> = OnceLock::new();
static KAKADU_EXPAND_PROGRAM: OnceLock<Option<PathBuf>> = OnceLock::new();

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BenchmarkMode {
    PortableNative,
    PortableEmulated,
    Capability,
}

impl BenchmarkMode {
    const fn label(self) -> &'static str {
        match self {
            Self::PortableNative => "portable-native",
            Self::PortableEmulated => "portable-emulated",
            Self::Capability => "capability",
        }
    }

    const fn comparable_scope(self) -> &'static str {
        match self {
            Self::PortableNative => "native-operations-only",
            Self::PortableEmulated => "task-equivalent-with-method-labels",
            Self::Capability => "feature-coverage-with-explicit-noncomparable-skips",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Codec {
    Classic,
    Htj2k,
    Unknown,
}

impl Codec {
    const fn label(self) -> &'static str {
        match self {
            Self::Classic => "j2k",
            Self::Htj2k => "htj2k",
            Self::Unknown => "unknown",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Container {
    RawCodestream,
    Jp2,
    Jph,
    Jhc,
}

impl Container {
    const fn label(self) -> &'static str {
        match self {
            Self::RawCodestream => "raw-codestream",
            Self::Jp2 => "jp2",
            Self::Jph => "jph",
            Self::Jhc => "jhc",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Operation {
    Full,
    Region(Rect),
    Scaled(Downscale),
    RegionScaled { roi: Rect, scale: Downscale },
}

impl Operation {
    const fn label(self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::Region(_) => "roi",
            Self::Scaled(_) => "scaled",
            Self::RegionScaled { .. } => "roi-scaled",
        }
    }

    const fn roi(self) -> Option<Rect> {
        match self {
            Self::Full | Self::Scaled(_) => None,
            Self::Region(roi) | Self::RegionScaled { roi, .. } => Some(roi),
        }
    }

    const fn scale(self) -> Downscale {
        match self {
            Self::Full | Self::Region(_) => Downscale::None,
            Self::Scaled(scale) | Self::RegionScaled { scale, .. } => scale,
        }
    }

    fn output_rect(self, dimensions: (u32, u32)) -> Rect {
        let source = self.roi().unwrap_or_else(|| Rect::full(dimensions));
        source.scaled_covering(self.scale())
    }

    const fn class(self) -> OperationClass {
        match self {
            Self::Full => OperationClass::Full,
            Self::Region(_) => OperationClass::Region,
            Self::Scaled(_) => OperationClass::Scaled,
            Self::RegionScaled { .. } => OperationClass::RegionScaled,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum OperationClass {
    Full,
    Region,
    Scaled,
    RegionScaled,
}

impl OperationClass {
    const fn label(self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::Region => "roi",
            Self::Scaled => "scaled",
            Self::RegionScaled => "roi-scaled",
        }
    }
}

#[derive(Clone)]
struct FixtureCase {
    name: String,
    input_source: String,
    corpus_category: String,
    corpus_name: String,
    license_status: String,
    encode_command: String,
    manifest_status: String,
    source_fnv1a64: Option<String>,
    codec: Codec,
    container: Container,
    bytes: Vec<u8>,
    dimensions: (u32, u32),
    format: PixelFormat,
    operation: Operation,
}

impl FixtureCase {
    fn input_len(&self) -> usize {
        self.bytes.len()
    }

    fn input_digest(&self) -> String {
        fnv1a64_hex(&self.bytes)
    }

    fn source_digest(&self) -> String {
        self.source_fnv1a64
            .clone()
            .unwrap_or_else(|| self.input_digest())
    }

    fn output_rect(&self) -> Rect {
        self.operation.output_rect(self.dimensions)
    }

    fn output_stride(&self) -> usize {
        self.output_rect().w as usize * self.format.bytes_per_pixel()
    }

    fn output_len(&self) -> usize {
        self.output_stride() * self.output_rect().h as usize
    }
}

#[derive(Clone)]
struct FixtureMetadata {
    input_source: String,
    corpus_category: String,
    corpus_name: String,
    license_status: String,
    encode_command: String,
    manifest_status: String,
    source_fnv1a64: Option<String>,
}

struct FixtureManifest {
    entries: HashMap<PathBuf, ManifestEntry>,
}

struct ManifestEntry {
    corpus_category: String,
    corpus_name: String,
    license_status: String,
    encode_command: String,
    input_fnv1a64: Option<String>,
    source_fnv1a64: Option<String>,
    codec: Option<Codec>,
    container: Option<Container>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DecoderKind {
    J2k,
    OpenJpeg,
    Grok,
    OpenJph,
    Kakadu,
}

impl DecoderKind {
    const fn label(self) -> &'static str {
        match self {
            Self::J2k => "j2k",
            Self::OpenJpeg => "openjpeg",
            Self::Grok => "grok",
            Self::OpenJph => "openjph",
            Self::Kakadu => "kakadu",
        }
    }
}

struct Measurement {
    decoder: DecoderKind,
    repeats: usize,
    batch_size: usize,
    median_us: f64,
    mean_us: f64,
    tiles_per_second_median: f64,
    decoded_bytes_per_repeat: usize,
    samples_us: Vec<f64>,
}

struct ActiveMeasurement {
    decoder: DecoderKind,
    batch_inputs: BatchInputs,
    samples_us: Vec<f64>,
    decoded_bytes_per_repeat: Option<usize>,
}

struct MixedFixtureBatch {
    name: String,
    cases: Vec<FixtureCase>,
    format: PixelFormat,
    operation_class: OperationClass,
}

struct ActiveMixedMeasurement {
    decoder: DecoderKind,
    samples_us: Vec<f64>,
    decoded_bytes_per_repeat: Option<usize>,
}

#[derive(Clone, Copy)]
struct MetadataContext<'a> {
    args: &'a [String],
    benchmark_mode: BenchmarkMode,
    repeats: usize,
    batch_sizes: &'a [usize],
    case_batch_sizes: &'a [usize],
    mixed_batch_sizes: &'a [usize],
    workers: Option<NonZeroUsize>,
    cases: &'a [FixtureCase],
    mixed_batches: &'a [MixedFixtureBatch],
    mode_excluded_cases: &'a [String],
    filters_empty: bool,
}

struct BatchInputs {
    buffers: Vec<Vec<u8>>,
    batch_size: usize,
}

impl BatchInputs {
    fn new(case: &FixtureCase, batch_size: usize) -> Self {
        let copy_count = batch_input_copy_count(batch_size);
        let buffers = (0..copy_count)
            .map(|_| case.bytes.clone())
            .collect::<Vec<_>>();
        Self {
            buffers,
            batch_size,
        }
    }

    const fn len(&self) -> usize {
        self.batch_size
    }

    fn input(&self, index: usize) -> &[u8] {
        self.buffers[index % self.buffers.len()].as_slice()
    }
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

    validate_comparator_gates()?;
    let benchmark_mode = benchmark_mode_from_env()?;
    let repeats = std::env::var("J2K_FIXTURE_COMPARE_REPEATS")
        .ok()
        .map(|value| parse_positive_usize(&value, "J2K_FIXTURE_COMPARE_REPEATS"))
        .transpose()?
        .unwrap_or(DEFAULT_REPEATS);
    let batch_sizes = batch_size_config_from_env()?;
    let combined_batch_sizes = combined_batch_sizes(
        &batch_sizes.case_batch_sizes,
        &batch_sizes.mixed_batch_sizes,
    );
    let workers = std::env::var("J2K_FIXTURE_COMPARE_THREADS")
        .ok()
        .map(|value| parse_positive_usize(&value, "J2K_FIXTURE_COMPARE_THREADS"))
        .transpose()?
        .map(|value| NonZeroUsize::new(value).expect("positive value was validated"));
    let filters = args.iter().skip(1).map(String::as_str).collect::<Vec<_>>();
    let selected_cases = select_cases(all_fixture_cases()?, &filters)?;
    let mode_excluded_cases = selected_cases
        .iter()
        .filter(|case| !include_case_in_mode(case, benchmark_mode))
        .map(|case| case.name.clone())
        .collect::<Vec<_>>();
    let cases = filter_cases_for_mode(selected_cases, benchmark_mode)?;
    let mixed_batches = mixed_external_batches(&cases);
    validate_cases(
        &cases,
        benchmark_mode,
        &batch_sizes.case_batch_sizes,
        workers,
    )?;
    validate_mixed_batches(
        &mixed_batches,
        benchmark_mode,
        &batch_sizes.mixed_batch_sizes,
        workers,
    )?;

    let mut output_rows = Vec::new();
    for (case_index, case) in cases.iter().enumerate() {
        for (batch_index, batch_size) in batch_sizes.case_batch_sizes.iter().enumerate() {
            output_rows.extend(measure_case_batch_rows(
                benchmark_mode,
                case,
                repeats,
                *batch_size,
                workers,
                case_index,
                batch_index,
            )?);
        }
    }
    for (mixed_index, mixed_batch) in mixed_batches.iter().enumerate() {
        for (batch_index, batch_size) in batch_sizes.mixed_batch_sizes.iter().enumerate() {
            output_rows.extend(measure_mixed_batch_rows(
                benchmark_mode,
                mixed_batch,
                repeats,
                *batch_size,
                workers,
                mixed_index,
                batch_index,
            )?);
        }
    }

    emit_metadata(MetadataContext {
        args: &args,
        benchmark_mode,
        repeats,
        batch_sizes: &combined_batch_sizes,
        case_batch_sizes: &batch_sizes.case_batch_sizes,
        mixed_batch_sizes: &batch_sizes.mixed_batch_sizes,
        workers,
        cases: &cases,
        mixed_batches: &mixed_batches,
        mode_excluded_cases: &mode_excluded_cases,
        filters_empty: filters.is_empty(),
    });
    println!(
        "decoder\tcase\tbenchmark_mode\tdecode_method\tinput_source\tcorpus_category\tcorpus_name\tlicense_status\tencode_command\tmanifest_status\tcodec\tcontainer\toperation\tformat\tdimensions\troi\tscale\tbatch_size\trepeats\tinput_bytes\tinput_fnv1a64\tsource_fnv1a64\tmedian_us\tmean_us\ttiles_per_second_median\tdecoded_mib_per_second_median\tdecoded_bytes_per_repeat\tsamples_us\tskip_reason"
    );
    for row in output_rows {
        println!("{row}");
    }
    println!("benchmark_complete\ttrue");

    Ok(())
}

fn print_usage(program: &str) {
    eprintln!("usage: {program} [case-name-filter ...]");
    eprintln!(
        "Runs J2K/OpenJPEG/Grok decode benchmarks over the same named fixture bytes; set J2K_INCLUDE_OPENJPH=1 for optional OpenJPH HTJ2K CLI rows or J2K_INCLUDE_KAKADU=1 for optional Kakadu CLI rows."
    );
}

fn benchmark_mode_from_env() -> Result<BenchmarkMode, String> {
    let Some(value) = std::env::var("J2K_FIXTURE_COMPARE_MODE").ok() else {
        return Ok(DEFAULT_BENCHMARK_MODE);
    };
    match value.as_str() {
        "portable-native" => Ok(BenchmarkMode::PortableNative),
        "portable-emulated" => Ok(BenchmarkMode::PortableEmulated),
        "capability" => Ok(BenchmarkMode::Capability),
        other => Err(format!(
            "J2K_FIXTURE_COMPARE_MODE must be portable-native, portable-emulated, or capability; got {other:?}"
        )),
    }
}

struct BatchSizeConfig {
    case_batch_sizes: Vec<usize>,
    mixed_batch_sizes: Vec<usize>,
}

fn batch_size_config_from_env() -> Result<BatchSizeConfig, String> {
    let case_batch_sizes = std::env::var("J2K_FIXTURE_COMPARE_CASE_BATCH_SIZES").ok();
    let mixed_batch_sizes = std::env::var("J2K_FIXTURE_COMPARE_MIXED_BATCH_SIZES").ok();
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
        parse_batch_sizes(value, "J2K_FIXTURE_COMPARE_CASE_BATCH_SIZES")?
    } else if let Some(values) = &legacy {
        values.clone()
    } else {
        DEFAULT_CASE_BATCH_SIZES.to_vec()
    };
    let mixed_batch_sizes = if let Some(value) = mixed_batch_sizes {
        parse_batch_sizes(value, "J2K_FIXTURE_COMPARE_MIXED_BATCH_SIZES")?
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
    if let Ok(value) = std::env::var("J2K_FIXTURE_COMPARE_BATCH_SIZES") {
        return parse_batch_sizes(&value, "J2K_FIXTURE_COMPARE_BATCH_SIZES").map(Some);
    }
    if let Ok(value) = std::env::var("J2K_FIXTURE_COMPARE_BATCH_SIZE") {
        return parse_batch_sizes(&value, "J2K_FIXTURE_COMPARE_BATCH_SIZE").map(Some);
    }
    Ok(None)
}

fn batch_input_copy_count(batch_size: usize) -> usize {
    if batch_size <= 1 {
        1
    } else {
        batch_size.clamp(2, BATCH_INPUT_COPY_LIMIT)
    }
}

fn validate_comparator_gates() -> Result<(), String> {
    if env_truthy("J2K_REQUIRE_GROK") && !grok::is_available() {
        return Err(
            "J2K_REQUIRE_GROK is set but in-process Grok is unavailable; install libgrokj2k or set J2K_GROK_SOURCE/J2K_GROK_ROOT"
                .to_string(),
        );
    }
    if env_truthy("J2K_REQUIRE_OPENJPEG") && !openjpeg::is_available() {
        return Err("J2K_REQUIRE_OPENJPEG is set but OpenJPEG is unavailable".to_string());
    }
    if env_truthy("J2K_REQUIRE_OPENJPH") && !openjph_is_available() {
        return Err(
            "J2K_REQUIRE_OPENJPH is set but ojph_expand is unavailable; set J2K_OPENJPH_EXPAND_BIN"
                .to_string(),
        );
    }
    if env_truthy("J2K_REQUIRE_KAKADU") && !kakadu_is_available() {
        return Err(
            "J2K_REQUIRE_KAKADU is set but kdu_expand is unavailable; set J2K_KDU_EXPAND_BIN"
                .to_string(),
        );
    }
    Ok(())
}

fn include_generated_fixtures() -> bool {
    !env_falsey("J2K_FIXTURE_COMPARE_INCLUDE_GENERATED")
}

fn include_openjph_comparator() -> bool {
    env_truthy("J2K_INCLUDE_OPENJPH") || env_truthy("J2K_REQUIRE_OPENJPH")
}

fn include_kakadu_comparator() -> bool {
    env_truthy("J2K_INCLUDE_KAKADU") || env_truthy("J2K_REQUIRE_KAKADU")
}

fn active_decoders() -> Vec<DecoderKind> {
    let mut decoders = vec![DecoderKind::J2k, DecoderKind::OpenJpeg, DecoderKind::Grok];
    if include_openjph_comparator() {
        decoders.push(DecoderKind::OpenJph);
    }
    if include_kakadu_comparator() {
        decoders.push(DecoderKind::Kakadu);
    }
    decoders
}

fn select_cases(cases: Vec<FixtureCase>, filters: &[&str]) -> Result<Vec<FixtureCase>, String> {
    if filters.is_empty() {
        return Ok(cases);
    }
    let selected = cases
        .into_iter()
        .filter(|case| filters.iter().any(|filter| case.name.contains(filter)))
        .collect::<Vec<_>>();
    if selected.is_empty() {
        return Err(format!(
            "no fixture cases matched filters: {}",
            filters.join(",")
        ));
    }
    Ok(selected)
}

fn filter_cases_for_mode(
    cases: Vec<FixtureCase>,
    benchmark_mode: BenchmarkMode,
) -> Result<Vec<FixtureCase>, String> {
    let filtered = cases
        .into_iter()
        .filter(|case| include_case_in_mode(case, benchmark_mode))
        .collect::<Vec<_>>();
    if filtered.is_empty() {
        return Err(format!(
            "no fixture cases remain for J2K_FIXTURE_COMPARE_MODE={}; use capability mode for feature-specific rows",
            benchmark_mode.label()
        ));
    }
    Ok(filtered)
}

fn include_case_in_mode(case: &FixtureCase, benchmark_mode: BenchmarkMode) -> bool {
    match benchmark_mode {
        BenchmarkMode::PortableNative => !is_openjpeg_region_scaled_noncomparable(case),
        BenchmarkMode::PortableEmulated | BenchmarkMode::Capability => true,
    }
}

fn all_fixture_cases() -> Result<Vec<FixtureCase>, String> {
    let manifest = fixture_manifest_from_env()?;
    let mut cases = if include_generated_fixtures() {
        fixture_cases()?
    } else {
        Vec::new()
    };
    for dir in external_input_dirs() {
        cases.extend(load_external_fixture_cases(&dir, manifest.as_ref())?);
    }
    if cases.is_empty() {
        return Err(
            "no fixture cases available; enable generated fixtures or set J2K_FIXTURE_COMPARE_INPUT_DIRS"
                .to_string(),
        );
    }
    Ok(cases)
}

fn mixed_external_batches(cases: &[FixtureCase]) -> Vec<MixedFixtureBatch> {
    let mut groups: Vec<MixedFixtureBatch> = Vec::new();
    for case in cases
        .iter()
        .filter(|case| case.input_source.starts_with("external:"))
    {
        let Some(group) = groups.iter_mut().find(|group| {
            group.format == case.format && group.operation_class == case.operation.class()
        }) else {
            groups.push(MixedFixtureBatch {
                name: format!(
                    "external_mixed_{}_{}",
                    pixel_format_label(case.format),
                    case.operation.class().label().replace('-', "_")
                ),
                cases: vec![case.clone()],
                format: case.format,
                operation_class: case.operation.class(),
            });
            continue;
        };
        group.cases.push(case.clone());
    }
    groups
        .into_iter()
        .filter(|group| unique_input_count(&group.cases) > 1)
        .collect()
}

fn fixture_cases() -> Result<Vec<FixtureCase>, String> {
    let roi64 = Rect {
        x: 32,
        y: 32,
        w: 64,
        h: 64,
    };
    let roi256 = Rect {
        x: 128,
        y: 128,
        w: 256,
        h: 256,
    };
    let classic_gray_128 = encode_gray(SMALL_SIDE, SMALL_SIDE, Codec::Classic)?;
    let classic_rgb_128 = encode_rgb(SMALL_SIDE, SMALL_SIDE, Codec::Classic)?;
    let classic_rgb_512 = encode_rgb(LARGE_SIDE, LARGE_SIDE, Codec::Classic)?;
    let htj2k_gray_128 = encode_gray(SMALL_SIDE, SMALL_SIDE, Codec::Htj2k)?;
    let htj2k_rgb_128 = encode_rgb(SMALL_SIDE, SMALL_SIDE, Codec::Htj2k)?;
    let htj2k_rgb_512 = encode_rgb(LARGE_SIDE, LARGE_SIDE, Codec::Htj2k)?;
    let classic_rgb_128_jp2 =
        wrap_jp2_codestream(&classic_rgb_128, SMALL_SIDE, SMALL_SIDE, 3, 8, 16);
    let classic_rgb_512_jp2 =
        wrap_jp2_codestream(&classic_rgb_512, LARGE_SIDE, LARGE_SIDE, 3, 8, 16);
    let htj2k_rgb_128_jph = wrap_j2k_codestream(&htj2k_rgb_128, J2kFileWrapOptions::jph())
        .map_err(|error| format!("wrap generated HTJ2K 128 fixture as JPH: {error}"))?;
    let htj2k_rgb_512_jph = wrap_j2k_codestream(&htj2k_rgb_512, J2kFileWrapOptions::jph())
        .map_err(|error| format!("wrap generated HTJ2K 512 fixture as JPH: {error}"))?;

    Ok(vec![
        case_from_bytes(
            "classic_raw_gray8_128_full",
            generated_metadata("j2k-generated"),
            Codec::Classic,
            Container::RawCodestream,
            classic_gray_128,
            Operation::Full,
        )?,
        case_from_bytes(
            "classic_raw_rgb8_128_full",
            generated_metadata("j2k-generated"),
            Codec::Classic,
            Container::RawCodestream,
            classic_rgb_128,
            Operation::Full,
        )?,
        case_from_bytes(
            "classic_jp2_rgb8_128_full",
            generated_metadata("j2k-generated-jp2-wrapper"),
            Codec::Classic,
            Container::Jp2,
            classic_rgb_128_jp2.clone(),
            Operation::Full,
        )?,
        case_from_bytes(
            "classic_jp2_rgb8_128_roi64",
            generated_metadata("j2k-generated-jp2-wrapper"),
            Codec::Classic,
            Container::Jp2,
            classic_rgb_128_jp2.clone(),
            Operation::Region(roi64),
        )?,
        case_from_bytes(
            "classic_jp2_rgb8_128_q4",
            generated_metadata("j2k-generated-jp2-wrapper"),
            Codec::Classic,
            Container::Jp2,
            classic_rgb_128_jp2.clone(),
            Operation::Scaled(Downscale::Quarter),
        )?,
        case_from_bytes(
            "classic_jp2_rgb8_128_roi64_q4",
            generated_metadata("j2k-generated-jp2-wrapper"),
            Codec::Classic,
            Container::Jp2,
            classic_rgb_128_jp2,
            Operation::RegionScaled {
                roi: roi64,
                scale: Downscale::Quarter,
            },
        )?,
        case_from_bytes(
            "classic_jp2_rgb8_512_roi256_q4",
            generated_metadata("j2k-generated-jp2-wrapper"),
            Codec::Classic,
            Container::Jp2,
            classic_rgb_512_jp2,
            Operation::RegionScaled {
                roi: roi256,
                scale: Downscale::Quarter,
            },
        )?,
        case_from_bytes(
            "htj2k_raw_gray8_128_full",
            generated_metadata("j2k-generated"),
            Codec::Htj2k,
            Container::RawCodestream,
            htj2k_gray_128,
            Operation::Full,
        )?,
        case_from_bytes(
            "htj2k_raw_rgb8_128_full",
            generated_metadata("j2k-generated"),
            Codec::Htj2k,
            Container::RawCodestream,
            htj2k_rgb_128,
            Operation::Full,
        )?,
        case_from_bytes(
            "htj2k_jph_rgb8_128_full",
            generated_metadata("j2k-generated-jph-wrapper"),
            Codec::Htj2k,
            Container::Jph,
            htj2k_rgb_128_jph.clone(),
            Operation::Full,
        )?,
        case_from_bytes(
            "htj2k_jph_rgb8_128_roi64_q4",
            generated_metadata("j2k-generated-jph-wrapper"),
            Codec::Htj2k,
            Container::Jph,
            htj2k_rgb_128_jph,
            Operation::RegionScaled {
                roi: roi64,
                scale: Downscale::Quarter,
            },
        )?,
        case_from_bytes(
            "htj2k_jph_rgb8_512_roi256_q4",
            generated_metadata("j2k-generated-jph-wrapper"),
            Codec::Htj2k,
            Container::Jph,
            htj2k_rgb_512_jph,
            Operation::RegionScaled {
                roi: roi256,
                scale: Downscale::Quarter,
            },
        )?,
    ])
}

fn case_from_bytes(
    name: impl Into<String>,
    metadata: FixtureMetadata,
    codec: Codec,
    container: Container,
    bytes: Vec<u8>,
    operation: Operation,
) -> Result<FixtureCase, String> {
    let name = name.into();
    let info = J2kDecoder::inspect(&bytes).map_err(|error| format!("{name}: inspect: {error}"))?;
    let format = pixel_format(info.components, info.bit_depth)
        .ok_or_else(|| format!("{name}: unsupported output shape for benchmark"))?;
    if let Some(roi) = operation.roi() {
        if !roi.is_within(info.dimensions) {
            return Err(format!("{name}: ROI {roi:?} exceeds {:?}", info.dimensions));
        }
    }
    Ok(FixtureCase {
        name,
        input_source: metadata.input_source,
        corpus_category: metadata.corpus_category,
        corpus_name: metadata.corpus_name,
        license_status: metadata.license_status,
        encode_command: metadata.encode_command,
        manifest_status: metadata.manifest_status,
        codec,
        container,
        bytes,
        dimensions: info.dimensions,
        format,
        operation,
        source_fnv1a64: metadata.source_fnv1a64,
    })
}

fn generated_metadata(input_source: &str) -> FixtureMetadata {
    FixtureMetadata {
        input_source: input_source.to_string(),
        corpus_category: "generated-dev".to_string(),
        corpus_name: "j2k-generated-fixture-matrix".to_string(),
        license_status: "repo-generated".to_string(),
        encode_command: "j2k-lossless-cpu-roundtrip".to_string(),
        manifest_status: "generated".to_string(),
        source_fnv1a64: None,
    }
}

fn external_input_dirs() -> Vec<PathBuf> {
    if let Some(paths) = std::env::var_os("J2K_FIXTURE_COMPARE_INPUT_DIRS") {
        return std::env::split_paths(&paths).collect();
    }
    std::env::var_os("J2K_FIXTURE_COMPARE_INPUT_DIR")
        .map(PathBuf::from)
        .into_iter()
        .collect()
}

fn load_external_fixture_cases(
    dir: &Path,
    manifest: Option<&FixtureManifest>,
) -> Result<Vec<FixtureCase>, String> {
    if !dir.is_dir() {
        return Err(format!(
            "J2K_FIXTURE_COMPARE_INPUT_DIR is not a directory: {}",
            dir.display()
        ));
    }
    let mut paths = Vec::new();
    collect_j2k_paths(dir, &mut paths)?;
    paths.sort();
    if paths.is_empty() {
        return Err(format!(
            "external input dir {} contains no .j2k/.j2c/.jp2/.jph/.jhc fixtures",
            dir.display()
        ));
    }

    let mut cases = Vec::new();
    for path in paths {
        let bytes =
            std::fs::read(&path).map_err(|error| format!("read {}: {error}", path.display()))?;
        let info = J2kDecoder::inspect(&bytes)
            .map_err(|error| format!("inspect external fixture {}: {error}", path.display()))?;
        if pixel_format(info.components, info.bit_depth).is_none() {
            return Err(format!(
                "external fixture {} has unsupported benchmark shape: components={} bit_depth={}",
                path.display(),
                info.components,
                info.bit_depth
            ));
        }
        let stem = sanitized_stem(&path);
        let codec = codec_from_bytes(&bytes);
        let container = container_from_path_and_bytes(&path, &bytes);
        let metadata = external_fixture_metadata(&path, &bytes, codec, container, manifest)?;
        cases.push(case_from_bytes(
            format!("external_{stem}_full"),
            metadata.clone(),
            codec,
            container,
            bytes.clone(),
            Operation::Full,
        )?);

        let min_side = info.dimensions.0.min(info.dimensions.1);
        if min_side >= 128 && should_emit_external_region_scaled(&metadata) {
            let roi = external_scaled_roi(info.dimensions);
            cases.push(case_from_bytes(
                format!("external_{stem}_roi{}_q4", roi.w),
                metadata,
                codec,
                container,
                bytes,
                Operation::RegionScaled {
                    roi,
                    scale: Downscale::Quarter,
                },
            )?);
        }
    }
    Ok(cases)
}

fn should_emit_external_region_scaled(metadata: &FixtureMetadata) -> bool {
    matches!(
        metadata.corpus_category.as_str(),
        "natural-image" | "medical-domain" | "remote-sensing"
    )
}

fn external_scaled_roi(dimensions: (u32, u32)) -> Rect {
    let min_side = dimensions.0.min(dimensions.1);
    let denominator = Downscale::Quarter.denominator();
    let roi_side = round_down_to_multiple((min_side / 2).max(64), denominator);
    let x = round_down_to_multiple((dimensions.0 - roi_side) / 2, denominator);
    let y = round_down_to_multiple((dimensions.1 - roi_side) / 2, denominator);
    Rect {
        x,
        y,
        w: roi_side,
        h: roi_side,
    }
}

fn round_down_to_multiple(value: u32, multiple: u32) -> u32 {
    debug_assert!(multiple > 0);
    value - (value % multiple)
}

fn fixture_manifest_from_env() -> Result<Option<FixtureManifest>, String> {
    let Some(path) = std::env::var_os("J2K_FIXTURE_COMPARE_MANIFEST").map(PathBuf::from) else {
        return Ok(None);
    };
    let text = std::fs::read_to_string(&path).map_err(|error| {
        format!(
            "read J2K_FIXTURE_COMPARE_MANIFEST {}: {error}",
            path.display()
        )
    })?;
    let base = path.parent().unwrap_or_else(|| Path::new("."));
    let relocation_roots = external_input_dirs();
    let mut lines = text.lines().filter(|line| !line.trim().is_empty());
    let header = lines
        .next()
        .ok_or_else(|| format!("fixture manifest {} is empty", path.display()))?;
    let headers = header.split('\t').collect::<Vec<_>>();
    let path_index = manifest_column(&headers, "path")?;
    let category_index = manifest_column(&headers, "corpus_category")?;
    let corpus_name_index = optional_manifest_column(&headers, "corpus_name");
    let license_status_index = optional_manifest_column(&headers, "license_status");
    let encode_command_index = optional_manifest_column(&headers, "encode_command");
    let hash_index = optional_manifest_column(&headers, "input_fnv1a64");
    let source_hash_index = optional_manifest_column(&headers, "source_fnv1a64");
    let codec_index = optional_manifest_column(&headers, "codec");
    let container_index = optional_manifest_column(&headers, "container");

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
            "fixture manifest",
            &path,
            row_number,
        )?;
        let corpus_category =
            manifest_required_value(&fields, category_index, "corpus_category", row_number)?;
        let entry = ManifestEntry {
            corpus_category,
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
            encode_command: manifest_optional_value(
                &fields,
                encode_command_index,
                "encode_command",
                row_number,
            )?
            .unwrap_or_else(|| "not-recorded".to_string()),
            input_fnv1a64: manifest_optional_value(
                &fields,
                hash_index,
                "input_fnv1a64",
                row_number,
            )?,
            source_fnv1a64: manifest_optional_value(
                &fields,
                source_hash_index,
                "source_fnv1a64",
                row_number,
            )?,
            codec: parse_manifest_codec(
                manifest_optional_value(&fields, codec_index, "codec", row_number)?.as_deref(),
                row_number,
            )?,
            container: parse_manifest_container(
                manifest_optional_value(&fields, container_index, "container", row_number)?
                    .as_deref(),
                row_number,
            )?,
        };
        if entries.insert(canonical_path, entry).is_some() {
            return Err(format!(
                "fixture manifest {} row {row_number} duplicates path {raw_path}",
                path.display()
            ));
        }
    }

    Ok(Some(FixtureManifest { entries }))
}

fn external_fixture_metadata(
    path: &Path,
    bytes: &[u8],
    codec: Codec,
    container: Container,
    manifest: Option<&FixtureManifest>,
) -> Result<FixtureMetadata, String> {
    let input_source = external_source_label(path)?;
    let Some(manifest) = manifest else {
        return Ok(FixtureMetadata {
            input_source,
            corpus_category: external_corpus_category(path),
            corpus_name: "path-inferred".to_string(),
            license_status: "not-recorded".to_string(),
            encode_command: "not-recorded".to_string(),
            manifest_status: "not-covered".to_string(),
            source_fnv1a64: None,
        });
    };
    let canonical_path = path
        .canonicalize()
        .map_err(|error| format!("canonicalize external fixture {}: {error}", path.display()))?;
    let Some(entry) = manifest.entries.get(&canonical_path) else {
        return Ok(FixtureMetadata {
            input_source,
            corpus_category: external_corpus_category(path),
            corpus_name: "path-inferred".to_string(),
            license_status: "not-recorded".to_string(),
            encode_command: "not-recorded".to_string(),
            manifest_status: "not-covered".to_string(),
            source_fnv1a64: None,
        });
    };

    if let Some(expected_hash) = &entry.input_fnv1a64 {
        let actual_hash = fnv1a64_hex(bytes);
        if actual_hash != *expected_hash {
            return Err(format!(
                "external fixture {} hash mismatch: manifest {expected_hash} != actual {actual_hash}",
                path.display()
            ));
        }
    }
    if let Some(expected_codec) = entry.codec {
        if codec != expected_codec {
            return Err(format!(
                "external fixture {} codec mismatch: manifest {} != detected {}",
                path.display(),
                expected_codec.label(),
                codec.label()
            ));
        }
    }
    if let Some(expected_container) = entry.container {
        if container != expected_container {
            return Err(format!(
                "external fixture {} container mismatch: manifest {} != detected {}",
                path.display(),
                expected_container.label(),
                container.label()
            ));
        }
    }

    let manifest_status =
        if entry.input_fnv1a64.is_some() && entry.codec.is_some() && entry.container.is_some() {
            "covered"
        } else {
            "covered-unpinned"
        };

    Ok(FixtureMetadata {
        input_source,
        corpus_category: entry.corpus_category.clone(),
        corpus_name: entry.corpus_name.clone(),
        license_status: entry.license_status.clone(),
        encode_command: entry.encode_command.clone(),
        manifest_status: manifest_status.to_string(),
        source_fnv1a64: entry.source_fnv1a64.clone(),
    })
}

fn manifest_column(headers: &[&str], name: &str) -> Result<usize, String> {
    common::manifest_column(headers, name, "fixture")
}

fn manifest_field<'a>(
    fields: &'a [&str],
    index: usize,
    name: &str,
    row_number: usize,
) -> Result<&'a str, String> {
    common::manifest_field(fields, index, name, row_number, "fixture")
}

fn manifest_required_value(
    fields: &[&str],
    index: usize,
    name: &str,
    row_number: usize,
) -> Result<String, String> {
    common::manifest_required_value(fields, index, name, row_number, "fixture")
}

fn manifest_optional_value(
    fields: &[&str],
    index: Option<usize>,
    name: &str,
    row_number: usize,
) -> Result<Option<String>, String> {
    common::manifest_optional_value(fields, index, name, row_number, "fixture")
}

fn parse_manifest_codec(value: Option<&str>, row_number: usize) -> Result<Option<Codec>, String> {
    match value {
        None => Ok(None),
        Some("j2k" | "classic") => Ok(Some(Codec::Classic)),
        Some("htj2k") => Ok(Some(Codec::Htj2k)),
        Some("unknown") => Ok(Some(Codec::Unknown)),
        Some(other) => Err(format!(
            "fixture manifest row {row_number} has invalid codec {other:?}; expected j2k, classic, htj2k, or unknown"
        )),
    }
}

fn parse_manifest_container(
    value: Option<&str>,
    row_number: usize,
) -> Result<Option<Container>, String> {
    match value {
        None => Ok(None),
        Some("raw-codestream" | "j2k" | "j2c") => Ok(Some(Container::RawCodestream)),
        Some("jp2") => Ok(Some(Container::Jp2)),
        Some("jph") => Ok(Some(Container::Jph)),
        Some("jhc") => Ok(Some(Container::Jhc)),
        Some(other) => Err(format!(
            "fixture manifest row {row_number} has invalid container {other:?}; expected raw-codestream, j2k, j2c, jp2, jph, or jhc"
        )),
    }
}

fn collect_j2k_paths(dir: &Path, paths: &mut Vec<PathBuf>) -> Result<(), String> {
    for entry in std::fs::read_dir(dir)
        .map_err(|error| format!("read external input dir {}: {error}", dir.display()))?
    {
        let path = entry
            .map_err(|error| format!("read external input dir entry: {error}"))?
            .path();
        if path.is_dir() {
            collect_j2k_paths(&path, paths)?;
        } else if is_j2k_path(&path) {
            paths.push(path);
        }
    }
    Ok(())
}

fn external_source_label(path: &Path) -> Result<String, String> {
    let source_path = path.display().to_string();
    if source_path.chars().any(char::is_control) {
        return Err(format!(
            "external fixture path contains a control character and cannot be represented safely: {}",
            source_path.escape_debug()
        ));
    }
    Ok(format!("external:{source_path}"))
}

fn is_j2k_path(path: &Path) -> bool {
    path.extension()
        .and_then(|value| value.to_str())
        .is_some_and(|extension| {
            matches!(
                extension.to_ascii_lowercase().as_str(),
                "j2k" | "j2c" | "jp2" | "jph" | "jhc"
            )
        })
}

fn external_corpus_category(path: &Path) -> String {
    let lower = path.to_string_lossy().to_ascii_lowercase();
    for (needle, category) in [
        ("iso", "conformance"),
        ("conformance", "conformance"),
        ("openjpeg", "interop"),
        ("openjph", "interop"),
        ("jpylyzer", "parser-robustness"),
        ("kodak", "natural-image"),
        ("tecnick", "natural-image"),
        ("testimages", "natural-image"),
        ("clic", "natural-image"),
        ("div2k", "natural-image"),
        ("flickr2k", "natural-image"),
        ("sipi", "natural-image"),
        ("dicom", "medical-domain"),
        ("wsi", "medical-domain"),
        ("tcia", "medical-domain"),
        ("idc", "medical-domain"),
        ("camelyon", "medical-domain"),
        ("panda", "medical-domain"),
        ("tcga", "medical-domain"),
        ("openslide", "medical-domain"),
        ("space", "remote-sensing"),
        ("sentinel", "remote-sensing"),
        ("landsat", "remote-sensing"),
        ("nitf", "remote-sensing"),
        ("gdal", "remote-sensing"),
    ] {
        if lower.contains(needle) {
            return category.to_string();
        }
    }
    "external-unspecified".to_string()
}

fn container_from_path_and_bytes(path: &Path, bytes: &[u8]) -> Container {
    if let Some(extension) = path.extension().and_then(|value| value.to_str()) {
        match extension.to_ascii_lowercase().as_str() {
            "jph" => return Container::Jph,
            "jhc" => return Container::Jhc,
            _ => {}
        }
    }
    container_from_bytes(bytes)
}

fn container_from_bytes(bytes: &[u8]) -> Container {
    if bytes.starts_with(&[0, 0, 0, 12, b'j', b'P', b' ', b' ']) {
        Container::Jp2
    } else {
        Container::RawCodestream
    }
}

fn codec_from_bytes(bytes: &[u8]) -> Codec {
    let Ok(payload) = j2k::extract_j2k_codestream_payload(bytes) else {
        return Codec::Unknown;
    };
    match j2k_native::inspect_j2k_codestream_header(payload.codestream()) {
        Ok(header) if header.high_throughput => Codec::Htj2k,
        Ok(_) => Codec::Classic,
        Err(_) => Codec::Unknown,
    }
}

fn pixel_format(components: u16, bit_depth: u8) -> Option<PixelFormat> {
    match (components, bit_depth) {
        (1, 8) => Some(PixelFormat::Gray8),
        (3, 8) => Some(PixelFormat::Rgb8),
        _ => None,
    }
}

fn encode_gray(width: u32, height: u32, codec: Codec) -> Result<Vec<u8>, String> {
    let pixels = patterned_gray8(width, height);
    encode_lossless(&pixels, width, height, 1, codec)
}

fn encode_rgb(width: u32, height: u32, codec: Codec) -> Result<Vec<u8>, String> {
    let pixels = patterned_rgb8(width, height);
    encode_lossless(&pixels, width, height, 3, codec)
}

fn encode_lossless(
    pixels: &[u8],
    width: u32,
    height: u32,
    components: u8,
    codec: Codec,
) -> Result<Vec<u8>, String> {
    let samples = J2kLosslessSamples::new(pixels, width, height, u16::from(components), 8, false)
        .map_err(|error| error.to_string())?;
    let block_coding_mode = match codec {
        Codec::Classic => J2kBlockCodingMode::Classic,
        Codec::Htj2k => J2kBlockCodingMode::HighThroughput,
        Codec::Unknown => {
            return Err("cannot encode generated fixture for unknown codec".to_string())
        }
    };
    let options = J2kLosslessEncodeOptions::default()
        .with_backend(EncodeBackendPreference::CpuOnly)
        .with_block_coding_mode(block_coding_mode)
        .with_max_decomposition_levels(Some(2))
        .with_validation(J2kEncodeValidation::CpuRoundTrip);
    Ok(encode_j2k_lossless(samples, &options)
        .map_err(|error| error.to_string())?
        .codestream)
}

fn emit_metadata(context: MetadataContext<'_>) {
    let MetadataContext {
        args,
        benchmark_mode,
        repeats,
        batch_sizes,
        case_batch_sizes,
        mixed_batch_sizes,
        workers,
        cases,
        mixed_batches,
        mode_excluded_cases,
        filters_empty,
    } = context;
    let publication_blockers = publication_blockers(
        benchmark_mode,
        repeats,
        case_batch_sizes,
        mixed_batch_sizes,
        filters_empty,
        cases,
        mixed_batches,
    );
    println!("command\t{}", args.join(" "));
    println!("host_os\t{}", std::env::consts::OS);
    println!("host_arch\t{}", std::env::consts::ARCH);
    println!("host_hardware\t{}", host_hardware_label());
    println!("j2k_compare_version\t{}", env!("CARGO_PKG_VERSION"));
    println!("build_profile\t{}", build_profile_label());
    println!("debug_assertions\t{}", cfg!(debug_assertions));
    println!("git_revision\t{}", git_revision_label());
    println!("git_dirty\t{}", git_dirty_label());
    println!("benchmark_mode\t{}", benchmark_mode.label());
    println!("comparable_scope\t{}", benchmark_mode.comparable_scope());
    println!("repeats\t{repeats}");
    println!("batch_sizes\t{}", join_usizes(batch_sizes));
    println!("case_batch_sizes\t{}", join_usizes(case_batch_sizes));
    println!("mixed_batch_sizes\t{}", join_usizes(mixed_batch_sizes));
    println!("workers\t{}", worker_policy_label(workers));
    println!("available_parallelism\t{}", available_parallelism_count());
    println!(
        "resolved_workers_by_batch\t{}",
        resolved_workers_label(batch_sizes, workers)
    );
    println!(
        "j2k_inner_parallelism_by_batch\t{}",
        j2k_inner_parallelism_label(batch_sizes)
    );
    println!("external_decoder_internal_threads\t1");
    println!("sample_order_policy\tinterleaved-rotating-decoder-order");
    println!("batch_input_policy\trotating-owned-copies-built-outside-timed-loop");
    println!(
        "mixed_external_batch_policy\tgroup-external-cases-by-format-operation-cycle-distinct-inputs"
    );
    println!("batch_input_copy_limit\t{BATCH_INPUT_COPY_LIMIT}");
    println!(
        "batch_input_copy_counts_by_batch\t{}",
        batch_input_copy_counts_label(batch_sizes)
    );
    println!(
        "execution_policy\tj2k-batch-api;external-comparators-single-image-decodes-parallelized-across-batch-workers"
    );
    println!("thread_env\tJ2K_FIXTURE_COMPARE_THREADS");
    println!("input_policy\teach decoder receives identical fixture bytes for a case");
    println!("correctness_preflight\tnon-skipped-comparators-match-j2k-baseline-all-batches");
    println!(
        "external_input_dirs\t{}",
        external_input_dirs_label(&external_input_dirs())
    );
    println!("fixture_manifest\t{}", fixture_manifest_label());
    println!(
        "generated_fixtures_included\t{}",
        include_generated_fixtures()
    );
    println!("selected_cases\t{}", cases.len());
    println!("min_publication_external_case_count\t{MIN_PUBLICATION_EXTERNAL_CASES}");
    println!("min_publication_external_input_count\t{MIN_PUBLICATION_EXTERNAL_INPUTS}");
    println!("mode_excluded_case_count\t{}", mode_excluded_cases.len());
    println!(
        "mode_excluded_cases\t{}",
        join_string_labels(mode_excluded_cases)
    );
    println!(
        "external_manifest_covered_case_count\t{}",
        external_manifest_covered_case_count(cases)
    );
    println!(
        "external_manifest_missing_case_count\t{}",
        external_manifest_missing_case_count(cases)
    );
    println!("generated_case_count\t{}", generated_case_count(cases));
    println!("external_case_count\t{}", external_case_count(cases));
    println!(
        "external_native_case_count\t{}",
        external_native_case_count(cases)
    );
    println!(
        "external_materialized_case_count\t{}",
        external_materialized_case_count(cases)
    );
    println!(
        "external_unique_input_count\t{}",
        external_unique_input_count(cases)
    );
    println!(
        "external_native_unique_input_count\t{}",
        external_native_unique_input_count(cases)
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
    println!("required_comparators\t{}", required_comparators_label());
    println!("matched_comparators\t{}", matched_comparators_label());
    println!(
        "skipped_comparators\t{}",
        skipped_comparators_label(benchmark_mode, cases)
    );
    println!(
        "publication_gate_skipped_comparators\t{}",
        publication_gate_skipped_comparators_label(benchmark_mode, cases)
    );
    println!("publication_eligible\t{}", publication_blockers.is_empty());
    println!(
        "publication_blockers\t{}",
        join_string_labels(&publication_blockers)
    );
    println!("openjpeg_available\t{}", openjpeg::is_available());
    println!("openjpeg_version\t{}", openjpeg::version());
    println!("openjpeg_library\t{}", openjpeg::library_path());
    println!("grok_available\t{}", grok::is_available());
    println!("grok_version\t{}", grok::version());
    println!("grok_library\t{}", grok::library_path());
    println!("openjph_included\t{}", include_openjph_comparator());
    println!("openjph_available\t{}", openjph_is_available());
    println!("openjph_expand_command\t{}", openjph_command_label());
    println!("openjph_version\t{}", openjph_version_label());
    println!("kakadu_included\t{}", include_kakadu_comparator());
    println!("kakadu_available\t{}", kakadu_is_available());
    println!("kakadu_expand_command\t{}", kakadu_command_label());
    println!("kakadu_version\t{}", kakadu_version_label());
}

fn batch_input_copy_counts_label(batch_sizes: &[usize]) -> String {
    batch_sizes
        .iter()
        .map(|batch_size| format!("{batch_size}:{}", batch_input_copy_count(*batch_size)))
        .collect::<Vec<_>>()
        .join(",")
}

fn generated_case_count(cases: &[FixtureCase]) -> usize {
    cases
        .iter()
        .filter(|case| case.input_source.starts_with("j2k-generated"))
        .count()
}

fn external_case_count(cases: &[FixtureCase]) -> usize {
    cases
        .iter()
        .filter(|case| case.input_source.starts_with("external:"))
        .count()
}

fn external_native_case_count(cases: &[FixtureCase]) -> usize {
    external_native_cases(cases).len()
}

fn external_materialized_case_count(cases: &[FixtureCase]) -> usize {
    cases
        .iter()
        .filter(|case| is_materialized_external_case(case))
        .count()
}

fn external_unique_input_count(cases: &[FixtureCase]) -> usize {
    unique_input_count(
        &cases
            .iter()
            .filter(|case| case.input_source.starts_with("external:"))
            .cloned()
            .collect::<Vec<_>>(),
    )
}

fn external_native_unique_input_count(cases: &[FixtureCase]) -> usize {
    unique_input_count(&external_native_cases(cases))
}

fn external_native_cases(cases: &[FixtureCase]) -> Vec<FixtureCase> {
    cases
        .iter()
        .filter(|case| {
            case.input_source.starts_with("external:") && !is_materialized_external_case(case)
        })
        .cloned()
        .collect()
}

fn is_materialized_external_case(case: &FixtureCase) -> bool {
    case.encode_command
        .starts_with("cargo-xtask-adoption-materialize")
        || case.encode_command.starts_with("j2k-adoption-materialize:")
}

fn unique_input_count(cases: &[FixtureCase]) -> usize {
    cases
        .iter()
        .map(FixtureCase::source_digest)
        .collect::<HashSet<_>>()
        .len()
}

fn mixed_external_max_distinct_inputs(mixed_batches: &[MixedFixtureBatch]) -> usize {
    mixed_batches
        .iter()
        .map(|mixed_batch| unique_input_count(&mixed_batch.cases))
        .max()
        .unwrap_or(0)
}

fn mixed_external_min_distinct_inputs(mixed_batches: &[MixedFixtureBatch]) -> usize {
    mixed_batches
        .iter()
        .map(|mixed_batch| unique_input_count(&mixed_batch.cases))
        .min()
        .unwrap_or(0)
}

fn mixed_external_group_distinct_inputs_label(mixed_batches: &[MixedFixtureBatch]) -> String {
    if mixed_batches.is_empty() {
        return "none".to_string();
    }
    mixed_batches
        .iter()
        .map(|mixed_batch| {
            format!(
                "{}:{}",
                mixed_batch.name,
                unique_input_count(&mixed_batch.cases)
            )
        })
        .collect::<Vec<_>>()
        .join(",")
}

fn external_manifest_covered_case_count(cases: &[FixtureCase]) -> usize {
    cases
        .iter()
        .filter(|case| {
            case.input_source.starts_with("external:") && case.manifest_status == "covered"
        })
        .count()
}

fn external_manifest_missing_case_count(cases: &[FixtureCase]) -> usize {
    cases
        .iter()
        .filter(|case| {
            case.input_source.starts_with("external:") && case.manifest_status != "covered"
        })
        .count()
}

fn fixture_manifest_label() -> String {
    std::env::var("J2K_FIXTURE_COMPARE_MANIFEST").unwrap_or_else(|_| "not set".to_string())
}

fn required_comparators_label() -> String {
    let mut required = Vec::new();
    if env_truthy("J2K_REQUIRE_OPENJPEG") {
        required.push("openjpeg");
    }
    if env_truthy("J2K_REQUIRE_GROK") {
        required.push("grok");
    }
    if env_truthy("J2K_REQUIRE_OPENJPH") {
        required.push("openjph");
    }
    if env_truthy("J2K_REQUIRE_KAKADU") {
        required.push("kakadu");
    }
    join_labels(&required)
}

fn matched_comparators_label() -> String {
    let mut matched = Vec::new();
    if openjpeg::is_available() {
        matched.push("openjpeg");
    }
    if grok::is_available() {
        matched.push("grok");
    }
    if include_openjph_comparator() && openjph_is_available() {
        matched.push("openjph");
    }
    if include_kakadu_comparator() && kakadu_is_available() {
        matched.push("kakadu");
    }
    join_labels(&matched)
}

fn skipped_comparators_label(benchmark_mode: BenchmarkMode, cases: &[FixtureCase]) -> String {
    let mut skipped = Vec::new();
    if !openjpeg::is_available() {
        skipped.push("openjpeg:openjpeg-unavailable");
    }
    if !grok::is_available() {
        skipped.push("grok:grok-unavailable");
    }
    if include_openjph_comparator() && !openjph_is_available() {
        skipped.push("openjph:openjph-unavailable");
    }
    if include_kakadu_comparator() && !kakadu_is_available() {
        skipped.push("kakadu:kakadu-unavailable");
    }
    if benchmark_mode == BenchmarkMode::Capability
        && cases
            .iter()
            .any(is_openjpeg_htj2k_region_scaled_noncomparable)
    {
        skipped.push("openjpeg:openjpeg-htj2k-roi-scaled-noncomparable");
    }
    if benchmark_mode == BenchmarkMode::Capability
        && cases
            .iter()
            .any(is_openjpeg_external_gray_region_scaled_noncomparable)
    {
        skipped.push("openjpeg:openjpeg-external-gray-roi-scaled-noncomparable");
    }
    if include_openjph_comparator()
        && cases.iter().any(|case| {
            case.input_source.starts_with("j2k-generated")
                || !matches!(case.codec, Codec::Htj2k)
                || matches!(
                    case.operation,
                    Operation::Region(_) | Operation::RegionScaled { .. }
                )
        })
    {
        skipped.push("openjph:openjph-htj2k-full-scaled-only");
    }
    if include_kakadu_comparator()
        && cases.iter().any(|case| {
            matches!(
                case.operation,
                Operation::Region(_) | Operation::RegionScaled { .. }
            )
        })
    {
        skipped.push("kakadu:kakadu-full-scaled-only");
    }
    join_labels(&skipped)
}

fn publication_gate_skipped_comparators_label(
    benchmark_mode: BenchmarkMode,
    cases: &[FixtureCase],
) -> String {
    let mut skipped = Vec::new();
    if !openjpeg::is_available() {
        skipped.push("openjpeg:openjpeg-unavailable");
    }
    if !grok::is_available() {
        skipped.push("grok:grok-unavailable");
    }
    if benchmark_mode == BenchmarkMode::Capability
        && cases
            .iter()
            .any(is_openjpeg_htj2k_region_scaled_noncomparable)
    {
        skipped.push("openjpeg:openjpeg-htj2k-roi-scaled-noncomparable");
    }
    if benchmark_mode == BenchmarkMode::Capability
        && cases
            .iter()
            .any(is_openjpeg_external_gray_region_scaled_noncomparable)
    {
        skipped.push("openjpeg:openjpeg-external-gray-roi-scaled-noncomparable");
    }
    join_labels(&skipped)
}

fn publication_blockers(
    benchmark_mode: BenchmarkMode,
    repeats: usize,
    case_batch_sizes: &[usize],
    mixed_batch_sizes: &[usize],
    filters_empty: bool,
    cases: &[FixtureCase],
    mixed_batches: &[MixedFixtureBatch],
) -> Vec<String> {
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
    if benchmark_mode != BenchmarkMode::PortableNative {
        blockers.push("benchmark-mode-not-portable-native".to_string());
    }
    if !filters_empty {
        blockers.push("case-filters-present".to_string());
    }
    if repeats < DEFAULT_REPEATS {
        blockers.push(format!("repeats-below-{DEFAULT_REPEATS}"));
    }
    if !default_batch_sizes_present(case_batch_sizes, DEFAULT_CASE_BATCH_SIZES) {
        blockers.push(format!(
            "default-case-batch-sizes-missing:{}",
            join_usizes(DEFAULT_CASE_BATCH_SIZES)
        ));
    }
    if !default_batch_sizes_present(mixed_batch_sizes, DEFAULT_MIXED_BATCH_SIZES) {
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
    if !openjpeg::is_available() {
        blockers.push("openjpeg-unavailable".to_string());
    }
    if !grok::is_available() {
        blockers.push("grok-unavailable".to_string());
    }
    if env_truthy("J2K_REQUIRE_OPENJPH") && !openjph_is_available() {
        blockers.push("openjph-unavailable".to_string());
    }
    if env_truthy("J2K_REQUIRE_KAKADU") && !kakadu_is_available() {
        blockers.push("kakadu-unavailable".to_string());
    }
    if publication_gate_skipped_comparators_label(benchmark_mode, cases) != "none" {
        blockers.push("skipped-comparators-present".to_string());
    }

    let external_cases = cases
        .iter()
        .filter(|case| case.input_source.starts_with("external:"))
        .collect::<Vec<_>>();
    if generated_case_count(cases) > 0 {
        blockers.push("generated-fixtures-included".to_string());
    }
    if external_cases.len() < MIN_PUBLICATION_EXTERNAL_CASES {
        blockers.push(format!(
            "external-case-count-below-{MIN_PUBLICATION_EXTERNAL_CASES}"
        ));
    }
    let external_unique_inputs = external_unique_input_count(cases);
    if external_unique_inputs < MIN_PUBLICATION_EXTERNAL_INPUTS {
        blockers.push(format!(
            "external-unique-input-count-below-{MIN_PUBLICATION_EXTERNAL_INPUTS}"
        ));
    }
    if mixed_batches.is_empty() {
        blockers.push("mixed-external-batches-missing".to_string());
    }
    if mixed_external_max_distinct_inputs(mixed_batches) < MIN_PUBLICATION_EXTERNAL_INPUTS {
        blockers.push(format!(
            "mixed-external-distinct-inputs-below-{MIN_PUBLICATION_EXTERNAL_INPUTS}"
        ));
    }
    for format in [PixelFormat::Gray8, PixelFormat::Rgb8] {
        for operation_class in [OperationClass::Full, OperationClass::RegionScaled] {
            require_mixed_fixture_group(
                &mut blockers,
                cases,
                mixed_batches,
                format,
                operation_class,
            );
        }
    }
    if external_cases
        .iter()
        .any(|case| case.manifest_status != "covered")
    {
        blockers.push("external-manifest-coverage-missing".to_string());
    }
    if external_cases
        .iter()
        .any(|case| case.corpus_name == "path-inferred" || case.corpus_name == "not-recorded")
    {
        blockers.push("external-corpus-name-missing".to_string());
    }
    if external_cases
        .iter()
        .any(|case| case.license_status == "not-recorded")
    {
        blockers.push("external-license-status-missing".to_string());
    }
    if external_cases
        .iter()
        .any(|case| !is_publishable_license_status(&case.license_status))
    {
        blockers.push("external-license-status-not-publishable".to_string());
    }
    if external_cases
        .iter()
        .any(|case| case.encode_command == "not-recorded")
    {
        blockers.push("external-encode-command-missing".to_string());
    }
    if external_cases
        .iter()
        .any(|case| case.codec == Codec::Unknown)
    {
        blockers.push("external-unknown-codec-present".to_string());
    }
    let native_external_cases = external_native_cases(cases);
    if native_external_cases.len() < MIN_PUBLICATION_EXTERNAL_INPUTS {
        blockers.push(format!(
            "external-native-case-count-below-{MIN_PUBLICATION_EXTERNAL_INPUTS}"
        ));
    }
    let native_unique_inputs = unique_input_count(&native_external_cases);
    if native_unique_inputs < MIN_PUBLICATION_EXTERNAL_INPUTS {
        blockers.push(format!(
            "external-native-unique-input-count-below-{MIN_PUBLICATION_EXTERNAL_INPUTS}"
        ));
    }
    if !external_cases
        .iter()
        .any(|case| case.codec == Codec::Classic)
    {
        blockers.push("external-classic-j2k-missing".to_string());
    }
    if !external_cases.iter().any(|case| case.codec == Codec::Htj2k) {
        blockers.push("external-htj2k-missing".to_string());
    }
    if !native_external_cases
        .iter()
        .any(|case| case.codec == Codec::Classic)
    {
        blockers.push("external-native-classic-j2k-missing".to_string());
    }
    if !native_external_cases
        .iter()
        .any(|case| case.codec == Codec::Htj2k)
    {
        blockers.push("external-native-htj2k-missing".to_string());
    }
    if !external_cases
        .iter()
        .any(|case| matches!(case.container, Container::RawCodestream | Container::Jhc))
    {
        blockers.push("external-raw-codestream-missing".to_string());
    }
    if !external_cases
        .iter()
        .any(|case| matches!(case.container, Container::Jp2 | Container::Jph))
    {
        blockers.push("external-jp2-missing".to_string());
    }
    if !external_cases
        .iter()
        .any(|case| matches!(case.operation, Operation::Full))
    {
        blockers.push("external-full-operation-missing".to_string());
    }
    if !external_cases
        .iter()
        .any(|case| matches!(case.operation, Operation::RegionScaled { .. }))
    {
        blockers.push("external-roi-scaled-operation-missing".to_string());
    }
    if !external_cases
        .iter()
        .any(|case| case.corpus_category == "conformance")
    {
        blockers.push("external-conformance-corpus-missing".to_string());
    }
    if !external_cases
        .iter()
        .any(|case| case.corpus_category == "interop")
    {
        blockers.push("external-interop-corpus-missing".to_string());
    }
    if !external_cases.iter().any(|case| {
        matches!(
            case.corpus_category.as_str(),
            "natural-image" | "medical-domain" | "remote-sensing"
        )
    }) {
        blockers.push("external-workload-corpus-missing".to_string());
    }
    blockers
}

fn require_mixed_fixture_group(
    blockers: &mut Vec<String>,
    cases: &[FixtureCase],
    mixed_batches: &[MixedFixtureBatch],
    format: PixelFormat,
    operation_class: OperationClass,
) {
    let external_count =
        external_unique_input_count_for_format_operation(cases, format, operation_class);
    let label = format!("{}-{}", pixel_format_label(format), operation_class.label());
    if external_count < MIN_PUBLICATION_MIXED_DISTINCT_INPUTS {
        if operation_class == OperationClass::Full {
            blockers.push(format!(
                "external-{label}-mixed-input-count-below-{MIN_PUBLICATION_MIXED_DISTINCT_INPUTS}"
            ));
        }
        return;
    }
    let mixed_count =
        mixed_unique_input_count_for_format_operation(mixed_batches, format, operation_class);
    if mixed_count < external_count {
        blockers.push(format!(
            "mixed-external-{label}-distinct-inputs-below-{external_count}"
        ));
    }
}

fn external_unique_input_count_for_format_operation(
    cases: &[FixtureCase],
    format: PixelFormat,
    operation_class: OperationClass,
) -> usize {
    cases
        .iter()
        .filter(|case| {
            case.input_source.starts_with("external:")
                && case.format == format
                && case.operation.class() == operation_class
        })
        .map(FixtureCase::source_digest)
        .collect::<HashSet<_>>()
        .len()
}

fn mixed_unique_input_count_for_format_operation(
    mixed_batches: &[MixedFixtureBatch],
    format: PixelFormat,
    operation_class: OperationClass,
) -> usize {
    mixed_batches
        .iter()
        .find(|mixed_batch| {
            mixed_batch.format == format && mixed_batch.operation_class == operation_class
        })
        .map_or(0, |mixed_batch| unique_input_count(&mixed_batch.cases))
}

fn join_labels(values: &[&str]) -> String {
    if values.is_empty() {
        "none".to_string()
    } else {
        values.join(",")
    }
}

fn external_input_dirs_label(paths: &[PathBuf]) -> String {
    if paths.is_empty() {
        return "not set".to_string();
    }
    paths
        .iter()
        .map(|path| path.display().to_string())
        .collect::<Vec<_>>()
        .join(":")
}

fn worker_policy_label(workers: Option<NonZeroUsize>) -> String {
    workers.map_or_else(|| "auto".to_string(), |value| value.get().to_string())
}

fn available_parallelism_count() -> usize {
    std::thread::available_parallelism().map_or(1, NonZeroUsize::get)
}

fn resolved_workers_label(batch_sizes: &[usize], workers: Option<NonZeroUsize>) -> String {
    let available = available_parallelism_count();
    batch_sizes
        .iter()
        .map(|batch_size| {
            let resolved =
                tile_batch_worker_count(*batch_size, TileBatchOptions { workers }, available);
            format!("{batch_size}:{resolved}")
        })
        .collect::<Vec<_>>()
        .join(",")
}

fn j2k_inner_parallelism_label(batch_sizes: &[usize]) -> String {
    batch_sizes
        .iter()
        .map(|batch_size| format!("{batch_size}:serial"))
        .collect::<Vec<_>>()
        .join(",")
}

fn validate_cases(
    cases: &[FixtureCase],
    benchmark_mode: BenchmarkMode,
    batch_sizes: &[usize],
    workers: Option<NonZeroUsize>,
) -> Result<(), String> {
    for case in cases {
        for batch_size in batch_sizes {
            validate_case(case, benchmark_mode, *batch_size, workers)?;
        }
    }
    Ok(())
}

fn validate_mixed_batches(
    mixed_batches: &[MixedFixtureBatch],
    benchmark_mode: BenchmarkMode,
    batch_sizes: &[usize],
    workers: Option<NonZeroUsize>,
) -> Result<(), String> {
    for mixed_batch in mixed_batches {
        for batch_size in batch_sizes {
            validate_mixed_batch(mixed_batch, benchmark_mode, *batch_size, workers)?;
        }
    }
    Ok(())
}

fn validate_case(
    case: &FixtureCase,
    benchmark_mode: BenchmarkMode,
    batch_size: usize,
    workers: Option<NonZeroUsize>,
) -> Result<(), String> {
    let batch_inputs = BatchInputs::new(case, batch_size);
    let expected = decode_batch(
        benchmark_mode,
        case,
        DecoderKind::J2k,
        &batch_inputs,
        workers,
    )?;
    for decoder in active_decoders()
        .into_iter()
        .filter(|decoder| *decoder != DecoderKind::J2k)
    {
        if skip_reason(benchmark_mode, decoder, case).is_some() {
            continue;
        }
        let actual = decode_batch(benchmark_mode, case, decoder, &batch_inputs, workers)?;
        if actual != expected {
            return Err(format!(
                "{}: {} output mismatch against j2k: {} vs {} bytes",
                case.name,
                decoder.label(),
                actual.len(),
                expected.len()
            ));
        }
    }
    Ok(())
}

fn validate_mixed_batch(
    mixed_batch: &MixedFixtureBatch,
    benchmark_mode: BenchmarkMode,
    batch_size: usize,
    workers: Option<NonZeroUsize>,
) -> Result<(), String> {
    let expected = decode_mixed_batch(
        benchmark_mode,
        mixed_batch,
        DecoderKind::J2k,
        batch_size,
        workers,
    )?;
    for decoder in active_decoders()
        .into_iter()
        .filter(|decoder| *decoder != DecoderKind::J2k)
    {
        if mixed_batch
            .cases
            .iter()
            .any(|case| skip_reason(benchmark_mode, decoder, case).is_some())
        {
            continue;
        }
        let actual = decode_mixed_batch(benchmark_mode, mixed_batch, decoder, batch_size, workers)?;
        if actual != expected {
            return Err(format!(
                "{}: {} mixed-batch output mismatch against j2k: {} vs {} bytes",
                mixed_batch.name,
                decoder.label(),
                actual.len(),
                expected.len()
            ));
        }
    }
    Ok(())
}

fn skip_reason(
    benchmark_mode: BenchmarkMode,
    decoder: DecoderKind,
    case: &FixtureCase,
) -> Option<&'static str> {
    match decoder {
        DecoderKind::OpenJpeg if !openjpeg::is_available() => Some("openjpeg-unavailable"),
        DecoderKind::Grok if !grok::is_available() => Some("grok-unavailable"),
        DecoderKind::OpenJph if !openjph_is_available() => Some("openjph-unavailable"),
        DecoderKind::Kakadu if !kakadu_is_available() => Some("kakadu-unavailable"),
        DecoderKind::OpenJph if case.codec != Codec::Htj2k => Some("openjph-htj2k-only"),
        DecoderKind::OpenJph if !matches!(case.container, Container::Jph | Container::Jhc) => {
            Some("openjph-jph-compatible-stream-required")
        }
        DecoderKind::OpenJph if case.input_source.starts_with("j2k-generated") => {
            Some("openjph-jph-compatible-stream-required")
        }
        DecoderKind::OpenJph
            if matches!(
                case.operation,
                Operation::Region(_) | Operation::RegionScaled { .. }
            ) =>
        {
            Some("openjph-roi-unsupported")
        }
        DecoderKind::Kakadu
            if matches!(
                case.operation,
                Operation::Region(_) | Operation::RegionScaled { .. }
            ) =>
        {
            Some("kakadu-roi-unsupported")
        }
        DecoderKind::OpenJpeg
            if benchmark_mode == BenchmarkMode::Capability
                && is_openjpeg_htj2k_region_scaled_noncomparable(case) =>
        {
            Some("openjpeg-htj2k-roi-scaled-noncomparable")
        }
        DecoderKind::OpenJpeg
            if benchmark_mode == BenchmarkMode::Capability
                && is_openjpeg_external_gray_region_scaled_noncomparable(case) =>
        {
            Some("openjpeg-external-gray-roi-scaled-noncomparable")
        }
        DecoderKind::J2k
        | DecoderKind::OpenJpeg
        | DecoderKind::Grok
        | DecoderKind::OpenJph
        | DecoderKind::Kakadu => None,
    }
}

fn is_openjpeg_htj2k_region_scaled_noncomparable(case: &FixtureCase) -> bool {
    matches!(case.codec, Codec::Htj2k) && matches!(case.operation, Operation::RegionScaled { .. })
}

fn is_openjpeg_external_gray_region_scaled_noncomparable(case: &FixtureCase) -> bool {
    case.input_source.starts_with("external:")
        && matches!(case.format, PixelFormat::Gray8)
        && matches!(case.operation, Operation::RegionScaled { .. })
}

fn is_openjpeg_region_scaled_noncomparable(case: &FixtureCase) -> bool {
    is_openjpeg_htj2k_region_scaled_noncomparable(case)
        || is_openjpeg_external_gray_region_scaled_noncomparable(case)
}

fn measure_case_batch_rows(
    benchmark_mode: BenchmarkMode,
    case: &FixtureCase,
    repeats: usize,
    batch_size: usize,
    workers: Option<NonZeroUsize>,
    case_index: usize,
    batch_index: usize,
) -> Result<Vec<String>, String> {
    let mut rows = Vec::new();
    let mut active = Vec::new();
    for decoder in active_decoders() {
        if let Some(reason) = skip_reason(benchmark_mode, decoder, case) {
            rows.push(skip_row(
                benchmark_mode,
                decoder,
                case,
                repeats,
                batch_size,
                reason,
            ));
        } else {
            active.push(ActiveMeasurement {
                decoder,
                batch_inputs: BatchInputs::new(case, batch_size),
                samples_us: Vec::with_capacity(repeats),
                decoded_bytes_per_repeat: None,
            });
        }
    }

    if active.is_empty() {
        return Ok(rows);
    }

    for repeat in 0..repeats {
        let offset = (case_index + batch_index + repeat) % active.len();
        for step in 0..active.len() {
            let active_index = (offset + step) % active.len();
            let active_measurement = &mut active[active_index];
            let started = Instant::now();
            let output = decode_batch(
                benchmark_mode,
                case,
                active_measurement.decoder,
                &active_measurement.batch_inputs,
                workers,
            )?;
            let elapsed_us = started.elapsed().as_secs_f64() * 1_000_000.0;
            std::hint::black_box(&output);
            let decoded_len = output.len();
            if let Some(expected_len) = active_measurement.decoded_bytes_per_repeat {
                if decoded_len != expected_len {
                    return Err(format!(
                        "{} {} decoded length changed between repeats: {} vs {} bytes",
                        case.name,
                        active_measurement.decoder.label(),
                        decoded_len,
                        expected_len
                    ));
                }
            } else {
                active_measurement.decoded_bytes_per_repeat = Some(decoded_len);
            }
            active_measurement.samples_us.push(elapsed_us);
        }
    }

    for active_measurement in active {
        let stats = sample_stats(&active_measurement.samples_us)?;
        rows.push(measurement_row(
            benchmark_mode,
            case,
            &Measurement {
                decoder: active_measurement.decoder,
                repeats,
                batch_size,
                median_us: stats.median,
                mean_us: stats.mean,
                tiles_per_second_median: usize_to_f64(batch_size) / (stats.median / 1_000_000.0),
                decoded_bytes_per_repeat: active_measurement
                    .decoded_bytes_per_repeat
                    .ok_or_else(|| "missing decoded length for measured decoder".to_string())?,
                samples_us: active_measurement.samples_us,
            },
        ));
    }

    Ok(rows)
}

fn measure_mixed_batch_rows(
    benchmark_mode: BenchmarkMode,
    mixed_batch: &MixedFixtureBatch,
    repeats: usize,
    batch_size: usize,
    workers: Option<NonZeroUsize>,
    mixed_index: usize,
    batch_index: usize,
) -> Result<Vec<String>, String> {
    let mut rows = Vec::new();
    let mut active = Vec::new();
    for decoder in active_decoders() {
        if let Some(reason) = mixed_skip_reason(benchmark_mode, decoder, mixed_batch) {
            rows.push(mixed_skip_row(
                benchmark_mode,
                decoder,
                mixed_batch,
                repeats,
                batch_size,
                reason,
            ));
        } else {
            active.push(ActiveMixedMeasurement {
                decoder,
                samples_us: Vec::with_capacity(repeats),
                decoded_bytes_per_repeat: None,
            });
        }
    }

    if active.is_empty() {
        return Ok(rows);
    }

    for repeat in 0..repeats {
        let offset = (mixed_index + batch_index + repeat) % active.len();
        for step in 0..active.len() {
            let active_index = (offset + step) % active.len();
            let active_measurement = &mut active[active_index];
            let started = Instant::now();
            let output = decode_mixed_batch(
                benchmark_mode,
                mixed_batch,
                active_measurement.decoder,
                batch_size,
                workers,
            )?;
            let elapsed_us = started.elapsed().as_secs_f64() * 1_000_000.0;
            std::hint::black_box(&output);
            let decoded_len = output.len();
            if let Some(expected_len) = active_measurement.decoded_bytes_per_repeat {
                if decoded_len != expected_len {
                    return Err(format!(
                        "{} {} decoded length changed between repeats: {} vs {} bytes",
                        mixed_batch.name,
                        active_measurement.decoder.label(),
                        decoded_len,
                        expected_len
                    ));
                }
            } else {
                active_measurement.decoded_bytes_per_repeat = Some(decoded_len);
            }
            active_measurement.samples_us.push(elapsed_us);
        }
    }

    for active_measurement in active {
        let stats = sample_stats(&active_measurement.samples_us)?;
        rows.push(mixed_measurement_row(
            benchmark_mode,
            mixed_batch,
            &Measurement {
                decoder: active_measurement.decoder,
                repeats,
                batch_size,
                median_us: stats.median,
                mean_us: stats.mean,
                tiles_per_second_median: usize_to_f64(batch_size) / (stats.median / 1_000_000.0),
                decoded_bytes_per_repeat: active_measurement
                    .decoded_bytes_per_repeat
                    .ok_or_else(|| "missing decoded length for mixed decoder".to_string())?,
                samples_us: active_measurement.samples_us,
            },
        ));
    }

    Ok(rows)
}

fn mixed_skip_reason(
    benchmark_mode: BenchmarkMode,
    decoder: DecoderKind,
    mixed_batch: &MixedFixtureBatch,
) -> Option<&'static str> {
    mixed_batch
        .cases
        .iter()
        .find_map(|case| skip_reason(benchmark_mode, decoder, case))
}

fn decode_batch(
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

fn decode_mixed_batch(
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

fn decode_j2k_batch(
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

fn decode_j2k_mixed_batch(
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

fn decode_j2k_single_case(case: &FixtureCase, input: &[u8]) -> Result<Vec<u8>, String> {
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
            &mut output,
            case.output_stride(),
            case.format,
        )
        .map_err(|error| format!("j2k serial full decode failed: {error}"))?,
        Operation::Region(roi) => decode_tile_region_into_in_context(
            input,
            &mut ctx,
            &mut pool,
            &mut output,
            case.output_stride(),
            case.format,
            roi,
        )
        .map_err(|error| format!("j2k serial ROI decode failed: {error}"))?,
        Operation::Scaled(scale) => decode_tile_scaled_into_in_context(
            input,
            &mut ctx,
            &mut pool,
            &mut output,
            case.output_stride(),
            case.format,
            scale,
        )
        .map_err(|error| format!("j2k serial scaled decode failed: {error}"))?,
        Operation::RegionScaled { roi, scale } => decode_tile_region_scaled_into_in_context(
            input,
            &mut ctx,
            &mut pool,
            &mut output,
            case.output_stride(),
            case.format,
            roi,
            scale,
        )
        .map_err(|error| format!("j2k serial ROI+scaled decode failed: {error}"))?,
    };
    Ok(output)
}

fn decode_external_batch(
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

fn decode_external_mixed_batch(
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

fn mixed_case_at(mixed_batch: &MixedFixtureBatch, index: usize) -> &FixtureCase {
    &mixed_batch.cases[index % mixed_batch.cases.len()]
}

fn decode_external_once(
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

fn should_emulate_region_scaled(
    benchmark_mode: BenchmarkMode,
    decoder: DecoderKind,
    case: &FixtureCase,
) -> bool {
    benchmark_mode == BenchmarkMode::PortableEmulated
        && decoder == DecoderKind::OpenJpeg
        && is_openjpeg_region_scaled_noncomparable(case)
}

fn decode_method_label(
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

fn decode_external_region_scaled_emulated_once(
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

fn crop_interleaved(
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

fn reduce_factor(scale: Downscale) -> Result<u32, String> {
    match scale {
        Downscale::None => Ok(0),
        Downscale::Half => Ok(1),
        Downscale::Quarter => Ok(2),
        Downscale::Eighth => Ok(3),
        _ => Err(format!(
            "unsupported downscale for external comparator: {scale:?}"
        )),
    }
}

fn decode_openjph_once(case: &FixtureCase, input: &[u8]) -> Result<Vec<u8>, String> {
    let Some(program) = openjph_expand_program() else {
        return Err("ojph_expand is unavailable".to_string());
    };
    let temp_dir = openjph_temp_dir()?;
    let token = OPENJPH_TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let input_path = temp_dir.join(format!(
        "{}_{}_input.{}",
        std::process::id(),
        token,
        openjph_input_extension(case.container)
    ));
    let output_path = temp_dir.join(format!(
        "{}_{}_output.{}",
        std::process::id(),
        token,
        openjph_output_extension(case.format)
    ));
    let result = (|| {
        fs::write(&input_path, input).map_err(|error| {
            format!(
                "write OpenJPH staged input {}: {error}",
                input_path.display()
            )
        })?;
        let mut command = Command::new(program);
        command
            .arg("-i")
            .arg(&input_path)
            .arg("-o")
            .arg(&output_path);
        if case.operation.scale() != Downscale::None {
            let reduce = reduce_factor(case.operation.scale())?;
            command.arg("-skip_res").arg(format!("{reduce},{reduce}"));
        }
        let output = command
            .output()
            .map_err(|error| format!("start ojph_expand: {error}"))?;
        if !output.status.success() {
            return Err(format!(
                "ojph_expand exited with {}: {}",
                output.status,
                String::from_utf8_lossy(&output.stderr).trim()
            ));
        }
        read_cli_pnm_output("OpenJPH", &output_path, case.format)
    })();
    cleanup_openjph_temp(&input_path, result.is_ok())?;
    cleanup_openjph_temp(&output_path, result.is_ok())?;
    result
}

fn decode_kakadu_once(case: &FixtureCase, input: &[u8]) -> Result<Vec<u8>, String> {
    let Some(program) = kakadu_expand_program() else {
        return Err("kdu_expand is unavailable".to_string());
    };
    let temp_dir = kakadu_temp_dir()?;
    let token = KAKADU_TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let input_path = temp_dir.join(format!(
        "{}_{}_input.{}",
        std::process::id(),
        token,
        openjph_input_extension(case.container)
    ));
    let output_path = temp_dir.join(format!(
        "{}_{}_output.{}",
        std::process::id(),
        token,
        openjph_output_extension(case.format)
    ));
    let result = (|| {
        fs::write(&input_path, input).map_err(|error| {
            format!(
                "write Kakadu staged input {}: {error}",
                input_path.display()
            )
        })?;
        let mut command = Command::new(program);
        command
            .arg("-i")
            .arg(&input_path)
            .arg("-o")
            .arg(&output_path);
        if case.operation.scale() != Downscale::None {
            command
                .arg("-reduce")
                .arg(reduce_factor(case.operation.scale())?.to_string());
        }
        let output = command
            .output()
            .map_err(|error| format!("start kdu_expand: {error}"))?;
        if !output.status.success() {
            return Err(format!(
                "kdu_expand exited with {}: {}",
                output.status,
                String::from_utf8_lossy(&output.stderr).trim()
            ));
        }
        read_cli_pnm_output("Kakadu", &output_path, case.format)
    })();
    cleanup_openjph_temp(&input_path, result.is_ok())?;
    cleanup_openjph_temp(&output_path, result.is_ok())?;
    result
}

fn openjph_temp_dir() -> Result<PathBuf, String> {
    let dir = std::env::current_dir()
        .map_err(|error| format!("current_dir: {error}"))?
        .join("target")
        .join("j2k-openjph-expand");
    fs::create_dir_all(&dir).map_err(|error| format!("create {}: {error}", dir.display()))?;
    Ok(dir)
}

fn kakadu_temp_dir() -> Result<PathBuf, String> {
    let dir = std::env::current_dir()
        .map_err(|error| format!("current_dir: {error}"))?
        .join("target")
        .join("j2k-kakadu-expand");
    fs::create_dir_all(&dir).map_err(|error| format!("create {}: {error}", dir.display()))?;
    Ok(dir)
}

fn cleanup_openjph_temp(path: &Path, fail_on_cleanup_error: bool) -> Result<(), String> {
    if !path.exists() {
        return Ok(());
    }
    match fs::remove_file(path) {
        Err(error) if fail_on_cleanup_error => {
            Err(format!("remove temp file {}: {error}", path.display()))
        }
        Ok(()) | Err(_) => Ok(()),
    }
}

fn openjph_input_extension(container: Container) -> &'static str {
    match container {
        Container::RawCodestream => "j2c",
        Container::Jp2 => "jp2",
        Container::Jph => "jph",
        Container::Jhc => "jhc",
    }
}

fn openjph_output_extension(format: PixelFormat) -> &'static str {
    match format {
        PixelFormat::Gray8 => "pgm",
        PixelFormat::Rgb8 => "ppm",
        _ => "pnm",
    }
}

fn read_cli_pnm_output(
    tool_label: &str,
    path: &Path,
    format: PixelFormat,
) -> Result<Vec<u8>, String> {
    let image = image::ImageReader::open(path)
        .map_err(|error| format!("open {tool_label} output {}: {error}", path.display()))?
        .with_guessed_format()
        .map_err(|error| format!("guess {tool_label} output {}: {error}", path.display()))?
        .decode()
        .map_err(|error| format!("decode {tool_label} output {}: {error}", path.display()))?;
    match format {
        PixelFormat::Gray8 => Ok(image.into_luma8().into_raw()),
        PixelFormat::Rgb8 => Ok(image.into_rgb8().into_raw()),
        other => Err(format!(
            "{tool_label} output format {other:?} is unsupported"
        )),
    }
}

fn openjph_is_available() -> bool {
    openjph_expand_program().is_some()
}

fn openjph_expand_program() -> Option<&'static PathBuf> {
    OPENJPH_EXPAND_PROGRAM
        .get_or_init(discover_openjph_expand_program)
        .as_ref()
}

fn discover_openjph_expand_program() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("J2K_OPENJPH_EXPAND_BIN").map(PathBuf::from) {
        return command_is_runnable(&path).then_some(path);
    }
    [
        PathBuf::from("/opt/homebrew/bin/ojph_expand"),
        PathBuf::from("/usr/local/bin/ojph_expand"),
        PathBuf::from("ojph_expand"),
    ]
    .into_iter()
    .find(|candidate| command_is_runnable(candidate))
}

fn command_is_runnable(program: &Path) -> bool {
    Command::new(program).output().is_ok()
}

fn openjph_command_label() -> String {
    openjph_expand_program().map_or_else(
        || "not found".to_string(),
        |program| program.display().to_string(),
    )
}

fn openjph_version_label() -> &'static str {
    if openjph_is_available() {
        "available-version-not-reported-by-ojph_expand"
    } else {
        "unavailable"
    }
}

fn kakadu_is_available() -> bool {
    kakadu_expand_program().is_some()
}

fn kakadu_expand_program() -> Option<&'static PathBuf> {
    KAKADU_EXPAND_PROGRAM
        .get_or_init(discover_kakadu_expand_program)
        .as_ref()
}

fn discover_kakadu_expand_program() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("J2K_KDU_EXPAND_BIN").map(PathBuf::from) {
        return command_is_runnable(&path).then_some(path);
    }
    [
        PathBuf::from("/opt/homebrew/bin/kdu_expand"),
        PathBuf::from("/usr/local/bin/kdu_expand"),
        PathBuf::from("kdu_expand"),
    ]
    .into_iter()
    .find(|candidate| command_is_runnable(candidate))
}

fn kakadu_command_label() -> String {
    kakadu_expand_program().map_or_else(
        || "not found".to_string(),
        |program| program.display().to_string(),
    )
}

fn kakadu_version_label() -> &'static str {
    if kakadu_is_available() {
        "available-version-not-reported-by-kdu_expand"
    } else {
        "unavailable"
    }
}

fn flatten_outputs(outputs: Vec<Vec<u8>>) -> Vec<u8> {
    let total_len = outputs.iter().map(Vec::len).sum();
    let mut flattened = Vec::with_capacity(total_len);
    for output in outputs {
        flattened.extend(output);
    }
    flattened
}

fn mixed_measurement_row(
    benchmark_mode: BenchmarkMode,
    mixed_batch: &MixedFixtureBatch,
    row: &Measurement,
) -> String {
    let samples = row
        .samples_us
        .iter()
        .map(|value| format!("{value:.3}"))
        .collect::<Vec<_>>()
        .join(",");
    [
        row.decoder.label().to_string(),
        mixed_batch.name.clone(),
        benchmark_mode.label().to_string(),
        mixed_decode_method_label(benchmark_mode, row.decoder, mixed_batch),
        "external:mixed".to_string(),
        mixed_case_value_label(mixed_batch, |case| case.corpus_category.as_str()),
        mixed_case_value_label(mixed_batch, |case| case.corpus_name.as_str()),
        mixed_case_value_label(mixed_batch, |case| case.license_status.as_str()),
        mixed_case_value_label(mixed_batch, |case| case.encode_command.as_str()),
        mixed_case_value_label(mixed_batch, |case| case.manifest_status.as_str()),
        mixed_case_value_label(mixed_batch, |case| case.codec.label()),
        mixed_case_value_label(mixed_batch, |case| case.container.label()),
        mixed_batch.operation_class.label().to_string(),
        pixel_format_label(mixed_batch.format).to_string(),
        "mixed".to_string(),
        "mixed".to_string(),
        mixed_case_value_label(mixed_batch, |case| {
            if case.operation.scale() == Downscale::None {
                "1"
            } else {
                match case.operation.scale() {
                    Downscale::Half => "2",
                    Downscale::Quarter => "4",
                    Downscale::Eighth => "8",
                    _ => "other",
                }
            }
        }),
        row.batch_size.to_string(),
        row.repeats.to_string(),
        mixed_input_bytes_per_repeat(mixed_batch, row.batch_size).to_string(),
        mixed_input_digest(mixed_batch, row.batch_size),
        mixed_source_digest(mixed_batch, row.batch_size),
        format!("{:.3}", row.median_us),
        format!("{:.3}", row.mean_us),
        format!("{:.3}", row.tiles_per_second_median),
        format!(
            "{:.3}",
            mib_per_second(row.decoded_bytes_per_repeat, row.median_us)
        ),
        row.decoded_bytes_per_repeat.to_string(),
        samples,
        String::new(),
    ]
    .join("\t")
}

fn measurement_row(benchmark_mode: BenchmarkMode, case: &FixtureCase, row: &Measurement) -> String {
    let samples = row
        .samples_us
        .iter()
        .map(|value| format!("{value:.3}"))
        .collect::<Vec<_>>()
        .join(",");
    [
        row.decoder.label().to_string(),
        case.name.clone(),
        benchmark_mode.label().to_string(),
        decode_method_label(benchmark_mode, row.decoder, case).to_string(),
        case.input_source.clone(),
        case.corpus_category.clone(),
        case.corpus_name.clone(),
        case.license_status.clone(),
        case.encode_command.clone(),
        case.manifest_status.clone(),
        case.codec.label().to_string(),
        case.container.label().to_string(),
        case.operation.label().to_string(),
        pixel_format_label(case.format).to_string(),
        dimensions_label(case.dimensions),
        roi_label(case.operation.roi()),
        scale_label(case.operation.scale()),
        row.batch_size.to_string(),
        row.repeats.to_string(),
        (case.input_len() * row.batch_size).to_string(),
        case.input_digest(),
        case.source_digest(),
        format!("{:.3}", row.median_us),
        format!("{:.3}", row.mean_us),
        format!("{:.3}", row.tiles_per_second_median),
        format!(
            "{:.3}",
            mib_per_second(row.decoded_bytes_per_repeat, row.median_us)
        ),
        row.decoded_bytes_per_repeat.to_string(),
        samples,
        String::new(),
    ]
    .join("\t")
}

fn mixed_decode_method_label(
    benchmark_mode: BenchmarkMode,
    decoder: DecoderKind,
    mixed_batch: &MixedFixtureBatch,
) -> String {
    let mut labels = Vec::new();
    for case in &mixed_batch.cases {
        let label = decode_method_label(benchmark_mode, decoder, case);
        if !labels.contains(&label) {
            labels.push(label);
        }
    }
    if labels == ["native"] {
        "native-mixed-external-batch".to_string()
    } else if labels.len() == 1 {
        format!("{}-mixed-external-batch", labels[0])
    } else {
        format!("mixed-methods:{}", labels.join(","))
    }
}

fn mixed_skip_row(
    benchmark_mode: BenchmarkMode,
    decoder: DecoderKind,
    mixed_batch: &MixedFixtureBatch,
    repeats: usize,
    batch_size: usize,
    reason: &'static str,
) -> String {
    let mut row = common::skipped_external_mixed_prefix(
        decoder.label(),
        &mixed_batch.name,
        benchmark_mode.label(),
    );
    row.extend(mixed_fixture_corpus_columns(mixed_batch));
    row.extend([
        mixed_case_value_label(mixed_batch, |case| case.codec.label()),
        mixed_case_value_label(mixed_batch, |case| case.container.label()),
        mixed_batch.operation_class.label().to_string(),
        pixel_format_label(mixed_batch.format).to_string(),
        "mixed".to_string(),
        "mixed".to_string(),
        "mixed".to_string(),
    ]);
    common::append_batch_input_columns(
        &mut row,
        batch_size,
        repeats,
        mixed_input_bytes_per_repeat(mixed_batch, batch_size),
        mixed_input_digest(mixed_batch, batch_size),
    );
    row.push(mixed_source_digest(mixed_batch, batch_size));
    common::append_na_columns(&mut row, 6);
    row.push(reason.to_string());
    common::join_tsv_row(row)
}

fn mixed_fixture_corpus_columns(mixed_batch: &MixedFixtureBatch) -> [String; 5] {
    [
        mixed_case_value_label(mixed_batch, |case| case.corpus_category.as_str()),
        mixed_case_value_label(mixed_batch, |case| case.corpus_name.as_str()),
        mixed_case_value_label(mixed_batch, |case| case.license_status.as_str()),
        mixed_case_value_label(mixed_batch, |case| case.encode_command.as_str()),
        mixed_case_value_label(mixed_batch, |case| case.manifest_status.as_str()),
    ]
}

fn skip_row(
    benchmark_mode: BenchmarkMode,
    decoder: DecoderKind,
    case: &FixtureCase,
    repeats: usize,
    batch_size: usize,
    reason: &'static str,
) -> String {
    [
        decoder.label().to_string(),
        case.name.clone(),
        benchmark_mode.label().to_string(),
        "skipped".to_string(),
        case.input_source.clone(),
        case.corpus_category.clone(),
        case.corpus_name.clone(),
        case.license_status.clone(),
        case.encode_command.clone(),
        case.manifest_status.clone(),
        case.codec.label().to_string(),
        case.container.label().to_string(),
        case.operation.label().to_string(),
        pixel_format_label(case.format).to_string(),
        dimensions_label(case.dimensions),
        roi_label(case.operation.roi()),
        scale_label(case.operation.scale()),
        batch_size.to_string(),
        repeats.to_string(),
        case.input_len().to_string(),
        case.input_digest(),
        case.source_digest(),
        "NA".to_string(),
        "NA".to_string(),
        "NA".to_string(),
        "NA".to_string(),
        "NA".to_string(),
        "NA".to_string(),
        reason.to_string(),
    ]
    .join("\t")
}

fn mixed_case_value_label(
    mixed_batch: &MixedFixtureBatch,
    value: impl Fn(&FixtureCase) -> &str,
) -> String {
    let mut labels: Vec<&str> = Vec::new();
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

fn mixed_input_bytes_per_repeat(mixed_batch: &MixedFixtureBatch, batch_size: usize) -> usize {
    (0..batch_size)
        .map(|index| mixed_case_at(mixed_batch, index).input_len())
        .sum()
}

fn mixed_input_digest(mixed_batch: &MixedFixtureBatch, batch_size: usize) -> String {
    let mut slices = Vec::with_capacity(batch_size);
    for index in 0..batch_size {
        slices.push(mixed_case_at(mixed_batch, index).bytes.as_slice());
    }
    fnv1a64_hex_slices(&slices)
}

fn mixed_source_digest(mixed_batch: &MixedFixtureBatch, batch_size: usize) -> String {
    let labels = (0..batch_size)
        .map(|index| mixed_case_at(mixed_batch, index).source_digest())
        .collect::<Vec<_>>();
    fnv1a64_hex_slices(&labels.iter().map(String::as_bytes).collect::<Vec<_>>())
}

fn pixel_format_label(format: PixelFormat) -> &'static str {
    match format {
        PixelFormat::Gray8 => "gray8",
        PixelFormat::Rgb8 => "rgb8",
        _ => "unsupported",
    }
}

fn dimensions_label(dimensions: (u32, u32)) -> String {
    format!("{}x{}", dimensions.0, dimensions.1)
}

fn roi_label(roi: Option<Rect>) -> String {
    roi.map_or_else(
        || "full".to_string(),
        |rect| format!("{},{},{},{}", rect.x, rect.y, rect.w, rect.h),
    )
}

fn scale_label(scale: Downscale) -> String {
    match scale {
        Downscale::None => "1".to_string(),
        Downscale::Half => "2".to_string(),
        Downscale::Quarter => "4".to_string(),
        Downscale::Eighth => "8".to_string(),
        _ => format!("{scale:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        batch_size_config_from_values, canonicalize_manifest_row_path, DEFAULT_CASE_BATCH_SIZES,
        DEFAULT_MIXED_BATCH_SIZES,
    };
    use std::path::Path;

    #[test]
    fn decode_batch_config_defaults_keep_large_batches_mixed_only() {
        let config =
            batch_size_config_from_values(None, None, None).expect("default batch config parses");

        assert_eq!(config.case_batch_sizes, DEFAULT_CASE_BATCH_SIZES);
        assert_eq!(config.mixed_batch_sizes, DEFAULT_MIXED_BATCH_SIZES);
    }

    #[test]
    fn decode_batch_config_split_env_overrides_legacy_independently() {
        let config = batch_size_config_from_values(Some("3"), None, Some(vec![2, 4]))
            .expect("case override with legacy config parses");

        assert_eq!(config.case_batch_sizes, vec![3]);
        assert_eq!(config.mixed_batch_sizes, vec![2, 4]);

        let config = batch_size_config_from_values(None, Some("8,16"), Some(vec![2, 4]))
            .expect("mixed override with legacy config parses");

        assert_eq!(config.case_batch_sizes, vec![2, 4]);
        assert_eq!(config.mixed_batch_sizes, vec![8, 16]);
    }

    #[test]
    fn decode_manifest_path_remaps_to_supplied_fixture_root_by_suffix() {
        let root = std::env::current_dir()
            .expect("current dir")
            .join("target")
            .join("j2k-fixture-manifest-remap-test")
            .join(std::process::id().to_string());
        let fixture_root = root.join("decode-fixtures");
        let fixture = fixture_root.join("classic").join("sample.jp2");
        std::fs::create_dir_all(fixture.parent().expect("fixture parent")).expect("create dirs");
        std::fs::write(&fixture, b"jp2").expect("fixture");

        let resolved = canonicalize_manifest_row_path(
            "/old/worktree/target/j2k-public-corpora/materialized-kodak/decode-fixtures/classic/sample.jp2",
            Path::new("/unused"),
            &[fixture_root],
            "fixture manifest",
            Path::new("fixtures.tsv"),
            2,
        )
        .expect("remap stale absolute path");

        assert_eq!(resolved, fixture.canonicalize().expect("canonical fixture"));
    }
}
