// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    build_profile_label, common, default_batch_sizes_present, dimensions_label, env_truthy,
    fnv1a64_hex_slices, git_dirty_label, git_dirty_status, git_revision, git_revision_label, grok,
    host_hardware_label, include_kakadu_encoder, is_publishable_license_status, join_string_labels,
    join_usizes, mib_per_second, openjpeg, samples_label, selected_encoders_label, tool_available,
    tool_command, tool_version, tool_version_available, EncoderKind, HashSet, ImageCase,
    Measurement, MetadataInput, MixedImageBatch, DEFAULT_CASE_BATCH_SIZES,
    DEFAULT_MIXED_BATCH_SIZES, DEFAULT_REPEATS, MIN_PUBLICATION_EXTERNAL_DIMENSIONS,
    MIN_PUBLICATION_EXTERNAL_IMAGES, MIN_PUBLICATION_EXTERNAL_SOURCE_FORMATS,
    MIN_PUBLICATION_MIXED_DISTINCT_INPUTS,
};

pub(super) fn emit_metadata(input: MetadataInput<'_>) {
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

pub(super) fn publication_blockers(input: &MetadataInput<'_>) -> Vec<String> {
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
    if !default_batch_sizes_present(input.case_batch_sizes, DEFAULT_CASE_BATCH_SIZES) {
        blockers.push(format!(
            "default-case-batch-sizes-missing:{}",
            join_usizes(DEFAULT_CASE_BATCH_SIZES)
        ));
    }
    if !default_batch_sizes_present(input.mixed_batch_sizes, DEFAULT_MIXED_BATCH_SIZES) {
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

pub(super) fn require_mixed_encode_group(
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

pub(super) fn external_unique_image_count_for_components(
    cases: &[ImageCase],
    components: u8,
) -> usize {
    cases
        .iter()
        .filter(|case| case.input_source.starts_with("external:") && case.components == components)
        .map(ImageCase::input_digest)
        .collect::<HashSet<_>>()
        .len()
}

pub(super) fn mixed_unique_image_count_for_components(
    mixed_batches: &[MixedImageBatch],
    components: u8,
) -> usize {
    mixed_batches
        .iter()
        .find(|mixed_batch| mixed_batch.components == components)
        .map_or(0, |mixed_batch| unique_image_count(&mixed_batch.cases))
}

pub(super) fn component_label(components: u8) -> &'static str {
    match components {
        1 => "gray8",
        3 => "rgb8",
        _ => "unsupported",
    }
}

pub(super) fn measurement_row(
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

pub(super) fn mixed_measurement_row(
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

pub(super) fn skip_row(
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

pub(super) fn mixed_skip_row(
    encoder: EncoderKind,
    mixed_batch: &MixedImageBatch,
    repeats: usize,
    batch_size: usize,
    reason: &'static str,
    command_template: &'static str,
) -> String {
    let mut row = common::skipped_external_mixed_prefix(
        encoder.label(),
        &mixed_batch.name,
        "classic-lossless-cli",
    );
    row.extend(mixed_encode_corpus_columns(mixed_batch));
    row.extend([
        "j2k".to_string(),
        "jp2".to_string(),
        if mixed_batch.components == 1 {
            "gray8"
        } else {
            "rgb8"
        }
        .to_string(),
        "mixed".to_string(),
    ]);
    common::append_batch_input_columns(
        &mut row,
        batch_size,
        repeats,
        mixed_input_bytes_per_repeat(mixed_batch, batch_size),
        mixed_input_digest(mixed_batch, batch_size),
    );
    common::append_na_columns(&mut row, 6);
    row.push(reason.to_string());
    row.push(command_template.to_string());
    common::join_tsv_row(&row)
}

pub(super) fn mixed_encode_corpus_columns(mixed_batch: &MixedImageBatch) -> [String; 5] {
    [
        mixed_case_value_label(mixed_batch, |case| case.corpus_category.as_str()),
        mixed_case_value_label(mixed_batch, |case| case.corpus_name.as_str()),
        mixed_case_value_label(mixed_batch, |case| case.license_status.as_str()),
        mixed_case_value_label(mixed_batch, |case| case.source_command.as_str()),
        mixed_case_value_label(mixed_batch, |case| case.manifest_status.as_str()),
    ]
}

pub(super) fn generated_case_count(cases: &[ImageCase]) -> usize {
    cases
        .iter()
        .filter(|case| case.input_source.starts_with("j2k-generated"))
        .count()
}

pub(super) fn external_case_count(cases: &[ImageCase]) -> usize {
    cases
        .iter()
        .filter(|case| case.input_source.starts_with("external:"))
        .count()
}

pub(super) fn external_manifest_covered_case_count(cases: &[ImageCase]) -> usize {
    cases
        .iter()
        .filter(|case| {
            case.input_source.starts_with("external:") && case.manifest_status == "covered"
        })
        .count()
}

pub(super) fn external_manifest_missing_case_count(cases: &[ImageCase]) -> usize {
    cases
        .iter()
        .filter(|case| {
            case.input_source.starts_with("external:") && case.manifest_status != "covered"
        })
        .count()
}

pub(super) fn encode_manifest_label() -> String {
    std::env::var("J2K_ENCODE_COMPARE_MANIFEST").unwrap_or_else(|_| "not set".to_string())
}

pub(super) fn external_unique_input_count(cases: &[ImageCase]) -> usize {
    unique_image_count(
        &cases
            .iter()
            .filter(|case| case.input_source.starts_with("external:"))
            .cloned()
            .collect::<Vec<_>>(),
    )
}

pub(super) fn external_component_groups(cases: &[ImageCase]) -> HashSet<u8> {
    cases
        .iter()
        .filter(|case| case.input_source.starts_with("external:"))
        .map(|case| case.components)
        .collect()
}

pub(super) fn external_component_group_count(cases: &[ImageCase]) -> usize {
    external_component_groups(cases).len()
}

pub(super) fn external_dimension_count(cases: &[ImageCase]) -> usize {
    cases
        .iter()
        .filter(|case| case.input_source.starts_with("external:"))
        .map(|case| (case.width, case.height))
        .collect::<HashSet<_>>()
        .len()
}

pub(super) fn external_source_format_count(cases: &[ImageCase]) -> usize {
    cases
        .iter()
        .filter(|case| case.input_source.starts_with("external:"))
        .map(|case| case.source_format.as_str())
        .collect::<HashSet<_>>()
        .len()
}

pub(super) fn unique_image_count(cases: &[ImageCase]) -> usize {
    cases
        .iter()
        .map(ImageCase::input_digest)
        .collect::<HashSet<_>>()
        .len()
}

pub(super) fn mixed_external_max_distinct_inputs(mixed_batches: &[MixedImageBatch]) -> usize {
    mixed_batches
        .iter()
        .map(|batch| unique_image_count(&batch.cases))
        .max()
        .unwrap_or(0)
}

pub(super) fn mixed_external_min_distinct_inputs(mixed_batches: &[MixedImageBatch]) -> usize {
    mixed_batches
        .iter()
        .map(|batch| unique_image_count(&batch.cases))
        .min()
        .unwrap_or(0)
}

pub(super) fn mixed_external_group_distinct_inputs_label(
    mixed_batches: &[MixedImageBatch],
) -> String {
    if mixed_batches.is_empty() {
        return "none".to_string();
    }
    mixed_batches
        .iter()
        .map(|batch| format!("{}:{}", batch.name, unique_image_count(&batch.cases)))
        .collect::<Vec<_>>()
        .join(",")
}

pub(super) fn case_input_bytes_per_repeat(case: &ImageCase, batch_size: usize) -> usize {
    case.pixels.len() * batch_size
}

pub(super) fn mixed_case_value_label(
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

pub(super) fn mixed_input_bytes_per_repeat(
    mixed_batch: &MixedImageBatch,
    batch_size: usize,
) -> usize {
    (0..batch_size)
        .map(|index| mixed_case_at(mixed_batch, index).pixels.len())
        .sum()
}

pub(super) fn mixed_input_digest(mixed_batch: &MixedImageBatch, batch_size: usize) -> String {
    let mut slices = Vec::with_capacity(batch_size);
    for index in 0..batch_size {
        slices.push(mixed_case_at(mixed_batch, index).pixels.as_slice());
    }
    fnv1a64_hex_slices(&slices)
}

pub(super) fn mixed_case_at(mixed_batch: &MixedImageBatch, index: usize) -> &ImageCase {
    &mixed_batch.cases[index % mixed_batch.cases.len()]
}
