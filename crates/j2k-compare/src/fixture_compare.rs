// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{
    collections::HashSet,
    num::NonZeroUsize,
    path::{Path, PathBuf},
    time::Instant,
};

use crate::{common, grok, openjpeg, parse_positive_usize, sample_stats, usize_to_f64};
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
use j2k_core::{tile_batch_worker_count, Downscale, PixelFormat, Rect};
use j2k_test_support::{fnv1a64_hex, patterned_gray8, patterned_rgb8, wrap_jp2_codestream};

use crate::common::{
    build_profile_label, canonicalize_manifest_row_path, combined_batch_sizes, env_falsey,
    env_truthy, git_dirty_label, git_revision_label, host_hardware_label, join_string_labels,
    join_usizes, optional_manifest_column, sanitized_stem,
};

mod comparators;
mod gates;
mod manifest;
mod rows;
mod types;
use self::comparators::{
    decode_kakadu_once, decode_openjph_once, kakadu_command_label, kakadu_is_available,
    kakadu_version_label, openjph_command_label, openjph_is_available, openjph_version_label,
    reduce_factor,
};
use self::gates::{publication_blockers, publication_gate_skipped_comparators_label};
use self::manifest::{external_fixture_metadata, fixture_manifest_from_env};
use self::rows::{measurement_row, mixed_measurement_row, mixed_skip_row, skip_row};
use self::types::{
    ActiveMeasurement, ActiveMixedMeasurement, BatchInputs, BenchmarkMode, Codec, Container,
    DecoderKind, FixtureCase, FixtureManifest, FixtureMetadata, ManifestEntry, Measurement,
    MetadataContext, MixedFixtureBatch, Operation, OperationClass, BATCH_INPUT_COPY_LIMIT,
    DEFAULT_BENCHMARK_MODE, DEFAULT_CASE_BATCH_SIZES, DEFAULT_MIXED_BATCH_SIZES, DEFAULT_REPEATS,
    LARGE_SIDE, MIN_PUBLICATION_EXTERNAL_CASES, MIN_PUBLICATION_EXTERNAL_INPUTS,
    MIN_PUBLICATION_MIXED_DISTINCT_INPUTS, SMALL_SIDE,
};

mod cli;
use self::cli::{
    active_decoders, batch_input_copy_count, batch_size_config_from_env, benchmark_mode_from_env,
    filter_cases_for_mode, include_case_in_mode, include_generated_fixtures,
    include_kakadu_comparator, include_openjph_comparator, print_usage, select_cases,
    validate_comparator_gates,
};
mod fixtures;
use self::fixtures::{
    all_fixture_cases, external_input_dirs, external_source_label, mixed_external_batches,
};
mod metadata;
use self::metadata::{
    emit_metadata, external_native_cases, external_unique_input_count, generated_case_count,
    join_labels, mixed_external_max_distinct_inputs, unique_input_count,
};
mod validation;
use self::validation::{
    is_openjpeg_external_gray_region_scaled_noncomparable,
    is_openjpeg_htj2k_region_scaled_noncomparable, is_openjpeg_region_scaled_noncomparable,
    mixed_skip_reason, skip_reason, validate_cases, validate_mixed_batches,
};
mod measurement;
use self::measurement::{measure_case_batch_rows, measure_mixed_batch_rows};
mod decode;
use self::decode::{
    decode_batch, decode_method_label, decode_mixed_batch, mixed_case_at, pixel_format_label,
};

pub fn main() {
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

#[cfg(test)]
mod tests;
