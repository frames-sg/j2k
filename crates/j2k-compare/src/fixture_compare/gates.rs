// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashSet;

use j2k_core::PixelFormat;

use crate::{
    common::{
        default_batch_sizes_present, env_truthy, git_dirty_status, git_revision,
        is_publishable_license_status, join_usizes,
    },
    grok, openjpeg,
};

use super::{
    external_native_cases, external_unique_input_count, generated_case_count,
    is_openjpeg_external_gray_region_scaled_noncomparable,
    is_openjpeg_htj2k_region_scaled_noncomparable, join_labels, kakadu_is_available,
    mixed_external_max_distinct_inputs, openjph_is_available, pixel_format_label,
    unique_input_count, BenchmarkMode, Codec, Container, FixtureCase, MixedFixtureBatch, Operation,
    OperationClass, DEFAULT_CASE_BATCH_SIZES, DEFAULT_MIXED_BATCH_SIZES, DEFAULT_REPEATS,
    MIN_PUBLICATION_EXTERNAL_CASES, MIN_PUBLICATION_EXTERNAL_INPUTS,
    MIN_PUBLICATION_MIXED_DISTINCT_INPUTS,
};

pub(super) fn publication_gate_skipped_comparators_label(
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

pub(super) fn publication_blockers(
    benchmark_mode: BenchmarkMode,
    repeats: usize,
    case_batch_sizes: &[usize],
    mixed_batch_sizes: &[usize],
    filters_empty: bool,
    cases: &[FixtureCase],
    mixed_batches: &[MixedFixtureBatch],
) -> Vec<String> {
    let mut blockers = Vec::new();
    append_fixture_run_blockers(
        &mut blockers,
        benchmark_mode,
        repeats,
        case_batch_sizes,
        mixed_batch_sizes,
        filters_empty,
        cases,
    );
    let external_cases = cases
        .iter()
        .filter(|case| case.input_source.starts_with("external:"))
        .collect::<Vec<_>>();
    append_fixture_inventory_blockers(&mut blockers, cases, mixed_batches, &external_cases);
    append_fixture_metadata_blockers(&mut blockers, &external_cases);
    let native_external_cases = external_native_cases(cases);
    append_fixture_coverage_blockers(&mut blockers, &external_cases, &native_external_cases);
    blockers
}

fn append_fixture_run_blockers(
    blockers: &mut Vec<String>,
    benchmark_mode: BenchmarkMode,
    repeats: usize,
    case_batch_sizes: &[usize],
    mixed_batch_sizes: &[usize],
    filters_empty: bool,
    cases: &[FixtureCase],
) {
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
}

fn append_fixture_inventory_blockers(
    blockers: &mut Vec<String>,
    cases: &[FixtureCase],
    mixed_batches: &[MixedFixtureBatch],
    external_cases: &[&FixtureCase],
) {
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
            require_mixed_fixture_group(blockers, cases, mixed_batches, format, operation_class);
        }
    }
}

fn append_fixture_metadata_blockers(blockers: &mut Vec<String>, external_cases: &[&FixtureCase]) {
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
}

fn append_fixture_coverage_blockers(
    blockers: &mut Vec<String>,
    external_cases: &[&FixtureCase],
    native_external_cases: &[FixtureCase],
) {
    if native_external_cases.len() < MIN_PUBLICATION_EXTERNAL_INPUTS {
        blockers.push(format!(
            "external-native-case-count-below-{MIN_PUBLICATION_EXTERNAL_INPUTS}"
        ));
    }
    let native_unique_inputs = unique_input_count(native_external_cases);
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
