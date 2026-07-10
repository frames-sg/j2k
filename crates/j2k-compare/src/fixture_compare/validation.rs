// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    active_decoders, batch_input_copy_count, decode_batch, decode_mixed_batch, grok,
    kakadu_is_available, openjpeg, openjph_is_available, BatchInputs, BenchmarkMode, Codec,
    Container, DecoderKind, FixtureCase, MixedFixtureBatch, NonZeroUsize, Operation, PixelFormat,
};

pub(super) fn validate_cases(
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

pub(super) fn validate_mixed_batches(
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

pub(super) fn validate_case(
    case: &FixtureCase,
    benchmark_mode: BenchmarkMode,
    batch_size: usize,
    workers: Option<NonZeroUsize>,
) -> Result<(), String> {
    let batch_inputs =
        BatchInputs::new(&case.bytes, batch_size, batch_input_copy_count(batch_size));
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

pub(super) fn validate_mixed_batch(
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

pub(super) fn skip_reason(
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

pub(super) fn is_openjpeg_htj2k_region_scaled_noncomparable(case: &FixtureCase) -> bool {
    matches!(case.codec, Codec::Htj2k) && matches!(case.operation, Operation::RegionScaled { .. })
}

pub(super) fn is_openjpeg_external_gray_region_scaled_noncomparable(case: &FixtureCase) -> bool {
    case.input_source.starts_with("external:")
        && matches!(case.format, PixelFormat::Gray8)
        && matches!(case.operation, Operation::RegionScaled { .. })
}

pub(super) fn is_openjpeg_region_scaled_noncomparable(case: &FixtureCase) -> bool {
    is_openjpeg_htj2k_region_scaled_noncomparable(case)
        || is_openjpeg_external_gray_region_scaled_noncomparable(case)
}

pub(super) fn mixed_skip_reason(
    benchmark_mode: BenchmarkMode,
    decoder: DecoderKind,
    mixed_batch: &MixedFixtureBatch,
) -> Option<&'static str> {
    mixed_batch
        .cases
        .iter()
        .find_map(|case| skip_reason(benchmark_mode, decoder, case))
}
