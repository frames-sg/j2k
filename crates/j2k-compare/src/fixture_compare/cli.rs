// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    common, env_falsey, env_truthy, grok, is_openjpeg_region_scaled_noncomparable,
    kakadu_is_available, openjpeg, openjph_is_available, BenchmarkMode, DecoderKind, FixtureCase,
    BATCH_INPUT_COPY_LIMIT, DEFAULT_BENCHMARK_MODE, DEFAULT_CASE_BATCH_SIZES,
    DEFAULT_MIXED_BATCH_SIZES,
};

pub(super) fn print_usage(program: &str) {
    eprintln!("usage: {program} [case-name-filter ...]");
    eprintln!(
        "Runs J2K/OpenJPEG/Grok decode benchmarks over the same named fixture bytes; set J2K_INCLUDE_OPENJPH=1 for optional OpenJPH HTJ2K CLI rows or J2K_INCLUDE_KAKADU=1 for optional Kakadu CLI rows."
    );
}

pub(super) fn benchmark_mode_from_env() -> Result<BenchmarkMode, String> {
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

pub(super) fn batch_size_config_from_env() -> Result<common::BatchSizeConfig, String> {
    common::batch_size_config_from_env(
        common::BatchSizeEnv {
            case_batch_sizes: "J2K_FIXTURE_COMPARE_CASE_BATCH_SIZES",
            mixed_batch_sizes: "J2K_FIXTURE_COMPARE_MIXED_BATCH_SIZES",
            legacy_batch_sizes: "J2K_FIXTURE_COMPARE_BATCH_SIZES",
            legacy_batch_size: Some("J2K_FIXTURE_COMPARE_BATCH_SIZE"),
        },
        DEFAULT_CASE_BATCH_SIZES,
        DEFAULT_MIXED_BATCH_SIZES,
    )
}

pub(super) fn batch_input_copy_count(batch_size: usize) -> usize {
    if batch_size <= 1 {
        1
    } else {
        batch_size.clamp(2, BATCH_INPUT_COPY_LIMIT)
    }
}

pub(super) fn validate_comparator_gates() -> Result<(), String> {
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

pub(super) fn include_generated_fixtures() -> bool {
    !env_falsey("J2K_FIXTURE_COMPARE_INCLUDE_GENERATED")
}

pub(super) fn include_openjph_comparator() -> bool {
    env_truthy("J2K_INCLUDE_OPENJPH") || env_truthy("J2K_REQUIRE_OPENJPH")
}

pub(super) fn include_kakadu_comparator() -> bool {
    env_truthy("J2K_INCLUDE_KAKADU") || env_truthy("J2K_REQUIRE_KAKADU")
}

pub(super) fn active_decoders() -> Vec<DecoderKind> {
    let mut decoders = vec![DecoderKind::J2k, DecoderKind::OpenJpeg, DecoderKind::Grok];
    if include_openjph_comparator() {
        decoders.push(DecoderKind::OpenJph);
    }
    if include_kakadu_comparator() {
        decoders.push(DecoderKind::Kakadu);
    }
    decoders
}

pub(super) fn select_cases(
    cases: Vec<FixtureCase>,
    filters: &[&str],
) -> Result<Vec<FixtureCase>, String> {
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

pub(super) fn filter_cases_for_mode(
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

pub(super) fn include_case_in_mode(case: &FixtureCase, benchmark_mode: BenchmarkMode) -> bool {
    match benchmark_mode {
        BenchmarkMode::PortableNative => !is_openjpeg_region_scaled_noncomparable(case),
        BenchmarkMode::PortableEmulated | BenchmarkMode::Capability => true,
    }
}
