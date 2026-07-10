// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    batch_input_copy_count, build_profile_label, env_truthy, external_input_dirs, git_dirty_label,
    git_revision_label, grok, host_hardware_label, include_generated_fixtures,
    include_kakadu_comparator, include_openjph_comparator,
    is_openjpeg_external_gray_region_scaled_noncomparable,
    is_openjpeg_htj2k_region_scaled_noncomparable, join_string_labels, join_usizes,
    kakadu_command_label, kakadu_is_available, kakadu_version_label, openjpeg,
    openjph_command_label, openjph_is_available, openjph_version_label, publication_blockers,
    publication_gate_skipped_comparators_label, tile_batch_worker_count, BenchmarkMode, Codec,
    FixtureCase, HashSet, MetadataContext, MixedFixtureBatch, NonZeroUsize, Operation, PathBuf,
    TileBatchOptions, BATCH_INPUT_COPY_LIMIT, MIN_PUBLICATION_EXTERNAL_CASES,
    MIN_PUBLICATION_EXTERNAL_INPUTS,
};

pub(super) fn emit_metadata(context: MetadataContext<'_>) {
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

pub(super) fn batch_input_copy_counts_label(batch_sizes: &[usize]) -> String {
    batch_sizes
        .iter()
        .map(|batch_size| format!("{batch_size}:{}", batch_input_copy_count(*batch_size)))
        .collect::<Vec<_>>()
        .join(",")
}

pub(super) fn generated_case_count(cases: &[FixtureCase]) -> usize {
    cases
        .iter()
        .filter(|case| case.input_source.starts_with("j2k-generated"))
        .count()
}

pub(super) fn external_case_count(cases: &[FixtureCase]) -> usize {
    cases
        .iter()
        .filter(|case| case.input_source.starts_with("external:"))
        .count()
}

pub(super) fn external_native_case_count(cases: &[FixtureCase]) -> usize {
    external_native_cases(cases).len()
}

pub(super) fn external_materialized_case_count(cases: &[FixtureCase]) -> usize {
    cases
        .iter()
        .filter(|case| is_materialized_external_case(case))
        .count()
}

pub(super) fn external_unique_input_count(cases: &[FixtureCase]) -> usize {
    unique_input_count(
        &cases
            .iter()
            .filter(|case| case.input_source.starts_with("external:"))
            .cloned()
            .collect::<Vec<_>>(),
    )
}

pub(super) fn external_native_unique_input_count(cases: &[FixtureCase]) -> usize {
    unique_input_count(&external_native_cases(cases))
}

pub(super) fn external_native_cases(cases: &[FixtureCase]) -> Vec<FixtureCase> {
    cases
        .iter()
        .filter(|case| {
            case.input_source.starts_with("external:") && !is_materialized_external_case(case)
        })
        .cloned()
        .collect()
}

pub(super) fn is_materialized_external_case(case: &FixtureCase) -> bool {
    case.encode_command
        .starts_with("cargo-xtask-adoption-materialize")
        || case.encode_command.starts_with("j2k-adoption-materialize:")
}

pub(super) fn unique_input_count(cases: &[FixtureCase]) -> usize {
    cases
        .iter()
        .map(FixtureCase::source_digest)
        .collect::<HashSet<_>>()
        .len()
}

pub(super) fn mixed_external_max_distinct_inputs(mixed_batches: &[MixedFixtureBatch]) -> usize {
    mixed_batches
        .iter()
        .map(|mixed_batch| unique_input_count(&mixed_batch.cases))
        .max()
        .unwrap_or(0)
}

pub(super) fn mixed_external_min_distinct_inputs(mixed_batches: &[MixedFixtureBatch]) -> usize {
    mixed_batches
        .iter()
        .map(|mixed_batch| unique_input_count(&mixed_batch.cases))
        .min()
        .unwrap_or(0)
}

pub(super) fn mixed_external_group_distinct_inputs_label(
    mixed_batches: &[MixedFixtureBatch],
) -> String {
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

pub(super) fn external_manifest_covered_case_count(cases: &[FixtureCase]) -> usize {
    cases
        .iter()
        .filter(|case| {
            case.input_source.starts_with("external:") && case.manifest_status == "covered"
        })
        .count()
}

pub(super) fn external_manifest_missing_case_count(cases: &[FixtureCase]) -> usize {
    cases
        .iter()
        .filter(|case| {
            case.input_source.starts_with("external:") && case.manifest_status != "covered"
        })
        .count()
}

pub(super) fn fixture_manifest_label() -> String {
    std::env::var("J2K_FIXTURE_COMPARE_MANIFEST").unwrap_or_else(|_| "not set".to_string())
}

pub(super) fn required_comparators_label() -> String {
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

pub(super) fn matched_comparators_label() -> String {
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

pub(super) fn skipped_comparators_label(
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

pub(super) fn join_labels(values: &[&str]) -> String {
    if values.is_empty() {
        "none".to_string()
    } else {
        values.join(",")
    }
}

pub(super) fn external_input_dirs_label(paths: &[PathBuf]) -> String {
    if paths.is_empty() {
        return "not set".to_string();
    }
    paths
        .iter()
        .map(|path| path.display().to_string())
        .collect::<Vec<_>>()
        .join(":")
}

pub(super) fn worker_policy_label(workers: Option<NonZeroUsize>) -> String {
    workers.map_or_else(|| "auto".to_string(), |value| value.get().to_string())
}

pub(super) fn available_parallelism_count() -> usize {
    std::thread::available_parallelism().map_or(1, NonZeroUsize::get)
}

pub(super) fn resolved_workers_label(
    batch_sizes: &[usize],
    workers: Option<NonZeroUsize>,
) -> String {
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

pub(super) fn j2k_inner_parallelism_label(batch_sizes: &[usize]) -> String {
    batch_sizes
        .iter()
        .map(|batch_size| format!("{batch_size}:serial"))
        .collect::<Vec<_>>()
        .join(",")
}
