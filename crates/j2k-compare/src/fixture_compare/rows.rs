// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::common::{self, mib_per_second};
use j2k_core::{Downscale, Rect};
use j2k_test_support::fnv1a64_hex_slices;

use super::{
    decode_method_label, mixed_case_at, pixel_format_label, BenchmarkMode, DecoderKind,
    FixtureCase, Measurement, MixedFixtureBatch,
};

pub(super) fn mixed_measurement_row(
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

pub(super) fn measurement_row(
    benchmark_mode: BenchmarkMode,
    case: &FixtureCase,
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

pub(super) fn mixed_skip_row(
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
    common::join_tsv_row(&row)
}

pub(super) fn skip_row(
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

fn mixed_fixture_corpus_columns(mixed_batch: &MixedFixtureBatch) -> [String; 5] {
    [
        mixed_case_value_label(mixed_batch, |case| case.corpus_category.as_str()),
        mixed_case_value_label(mixed_batch, |case| case.corpus_name.as_str()),
        mixed_case_value_label(mixed_batch, |case| case.license_status.as_str()),
        mixed_case_value_label(mixed_batch, |case| case.encode_command.as_str()),
        mixed_case_value_label(mixed_batch, |case| case.manifest_status.as_str()),
    ]
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

fn dimensions_label(dimensions: (u32, u32)) -> String {
    common::dimensions_label(dimensions.0, dimensions.1)
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
