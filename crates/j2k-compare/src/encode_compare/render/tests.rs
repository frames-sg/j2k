// SPDX-License-Identifier: MIT OR Apache-2.0

use std::path::PathBuf;

use super::{
    append_encode_corpus_blockers, append_encode_tool_blockers, case_input_bytes_per_repeat,
    component_label, external_case_count, external_component_group_count, external_dimension_count,
    external_manifest_covered_case_count, external_manifest_missing_case_count,
    external_source_format_count, external_unique_image_count_for_components,
    external_unique_input_count, generated_case_count, measurement_row, mixed_case_at,
    mixed_case_value_label, mixed_encode_corpus_columns,
    mixed_external_group_distinct_inputs_label, mixed_external_max_distinct_inputs,
    mixed_external_min_distinct_inputs, mixed_input_bytes_per_repeat, mixed_measurement_row,
    mixed_skip_row, mixed_unique_image_count_for_components, require_mixed_encode_group, skip_row,
    unique_image_count,
};
use crate::encode_compare::{
    EncoderKind, EncoderTool, ImageCase, Measurement, MetadataInput, MixedImageBatch,
};

fn image_case(name: &str, source: &str, components: u8, pixels: &[u8]) -> ImageCase {
    ImageCase {
        name: name.to_string(),
        input_source: source.to_string(),
        corpus_category: "natural-image".to_string(),
        corpus_name: "render-fixture".to_string(),
        license_status: "cc0".to_string(),
        source_command: "fixture-command".to_string(),
        manifest_status: "covered".to_string(),
        source_format: if components == 1 { "pgm" } else { "ppm" }.to_string(),
        width: u32::try_from(pixels.len()).expect("fixture pixel length fits u32")
            / u32::from(components),
        height: 1,
        components,
        pixels: pixels.to_vec(),
        pnm_path: PathBuf::from(format!("{name}.pnm")),
    }
}

fn measurement(batch_size: usize) -> Measurement {
    Measurement {
        batch_size,
        repeats: 3,
        median_us: 8.0,
        mean_us: 9.5,
        images_per_second_median: 250.0,
        encoded_bytes_per_repeat: 77,
        samples_us: vec![7.0, 8.0, 9.5],
    }
}

#[test]
fn render_rows_keep_schema_values_for_measured_and_skipped_cases() {
    let case = image_case("rgb", "external:rgb", 3, &[1, 2, 3, 4, 5, 6]);
    let measured = measurement_row(EncoderKind::OpenJpeg, &case, &measurement(2), "encoder cmd");
    let columns = measured.split('\t').collect::<Vec<_>>();
    assert_eq!(columns.len(), 26);
    assert_eq!(columns[0], "openjpeg");
    assert_eq!(columns[3], "pnm-input-cli-process-output-jp2");
    assert_eq!(columns[12], "rgb8");
    assert_eq!(columns[14], "2");
    assert_eq!(columns[16], "12");
    assert_eq!(columns[18], "8.000");
    assert_eq!(columns[23], "7.000,8.000,9.500");
    assert_eq!(columns[25], "encoder cmd");

    let skipped = skip_row(EncoderKind::Grok, &case, 4, 3, "missing", "grok cmd");
    let columns = skipped.split('\t').collect::<Vec<_>>();
    assert_eq!(columns.len(), 26);
    assert_eq!(columns[3], "skipped");
    assert_eq!(columns[16], case.pixels.len().to_string());
    assert!(columns[18..24].iter().all(|value| *value == "NA"));
    assert_eq!(columns[24], "missing");
}

#[test]
fn mixed_rows_cycle_inputs_and_render_homogeneous_or_mixed_metadata() {
    let mut first = image_case("first", "external:first", 3, &[1, 2, 3]);
    let mut second = image_case("second", "external:second", 3, &[4, 5, 6, 7, 8, 9]);
    second.corpus_name = "other-corpus".to_string();
    second.source_command = "other-command".to_string();
    first.license_status = "cc-by".to_string();
    let batch = MixedImageBatch {
        name: "mixed-rgb".to_string(),
        cases: vec![first, second],
        components: 3,
    };

    assert_eq!(mixed_case_at(&batch, 0).name, "first");
    assert_eq!(mixed_case_at(&batch, 2).name, "first");
    assert_eq!(mixed_input_bytes_per_repeat(&batch, 3), 12);
    assert_eq!(
        mixed_case_value_label(&batch, |case| &case.corpus_category),
        "natural-image"
    );
    assert_eq!(
        mixed_case_value_label(&batch, |case| &case.corpus_name),
        "mixed:render-fixture,other-corpus"
    );
    assert_eq!(
        mixed_encode_corpus_columns(&batch)[3],
        "mixed:fixture-command,other-command"
    );

    let measured = mixed_measurement_row(EncoderKind::J2k, &batch, &measurement(3), "j2k cmd");
    let columns = measured.split('\t').collect::<Vec<_>>();
    assert_eq!(columns.len(), 26);
    assert_eq!(columns[4], "external:mixed");
    assert_eq!(columns[12], "rgb8");
    assert_eq!(columns[13], "mixed");
    assert_eq!(columns[16], "12");

    let skipped = mixed_skip_row(EncoderKind::Kakadu, &batch, 2, 3, "opt-in", "kdu cmd");
    let columns = skipped.split('\t').collect::<Vec<_>>();
    assert_eq!(columns.len(), 26);
    assert_eq!(columns[3], "skipped");
    assert_eq!(columns[12], "rgb8");
    assert_eq!(columns[24], "opt-in");

    let gray_batch = MixedImageBatch {
        name: "mixed-gray".to_string(),
        cases: vec![image_case("gray", "external:gray", 1, &[9, 8])],
        components: 1,
    };
    assert_eq!(
        mixed_measurement_row(EncoderKind::J2k, &gray_batch, &measurement(1), "cmd")
            .split('\t')
            .nth(12),
        Some("gray8")
    );
    assert_eq!(
        mixed_skip_row(EncoderKind::J2k, &gray_batch, 1, 1, "why", "cmd")
            .split('\t')
            .nth(12),
        Some("gray8")
    );
}

#[test]
fn corpus_counts_and_mixed_requirements_distinguish_sources_and_content() {
    let gray_a = image_case("gray-a", "external:a", 1, &[1, 2]);
    let gray_duplicate = image_case("gray-copy", "external:b", 1, &[1, 2]);
    let gray_b = image_case("gray-b", "external:c", 1, &[3, 4]);
    let mut rgb = image_case("rgb", "external:d", 3, &[1, 2, 3]);
    rgb.width = 1;
    rgb.source_format = "png".to_string();
    let generated = image_case("generated", "j2k-generated-image", 1, &[7]);
    let cases = vec![
        gray_a.clone(),
        gray_duplicate,
        gray_b.clone(),
        rgb,
        generated,
    ];
    let gray_batch = MixedImageBatch {
        name: "gray-group".to_string(),
        cases: vec![gray_a, gray_b],
        components: 1,
    };

    assert_eq!(generated_case_count(&cases), 1);
    assert_eq!(external_case_count(&cases), 4);
    assert_eq!(external_manifest_covered_case_count(&cases), 4);
    assert_eq!(external_manifest_missing_case_count(&cases), 0);
    assert_eq!(external_unique_input_count(&cases), 3);
    assert_eq!(external_unique_image_count_for_components(&cases, 1), 2);
    assert_eq!(external_component_group_count(&cases), 2);
    assert_eq!(external_dimension_count(&cases), 2);
    assert_eq!(external_source_format_count(&cases), 2);
    assert_eq!(unique_image_count(&cases), 4);
    assert_eq!(mixed_unique_image_count_for_components(&[gray_batch], 1), 2);
    assert_eq!(component_label(1), "gray8");
    assert_eq!(component_label(3), "rgb8");
    assert_eq!(component_label(4), "unsupported");
    assert_eq!(case_input_bytes_per_repeat(&cases[0], 4), 8);
}

#[test]
fn corpus_blockers_report_each_missing_publication_invariant() {
    let mut external = image_case("external", "external:one", 3, &[1, 2, 3]);
    external.corpus_category = "synthetic".to_string();
    external.corpus_name = "path-inferred".to_string();
    external.license_status = "not-recorded".to_string();
    external.source_command = "not-recorded".to_string();
    external.manifest_status = "not-covered".to_string();
    let generated = image_case("generated", "j2k-generated-image", 1, &[5]);
    let cases = vec![external, generated];
    let tools = vec![EncoderTool {
        kind: EncoderKind::OpenJpeg,
        program: PathBuf::from("unavailable"),
        available: false,
    }];
    let input = MetadataInput {
        args: &[],
        repeats: 1,
        batch_sizes: &[1],
        case_batch_sizes: &[1],
        mixed_batch_sizes: &[1],
        cases: &cases,
        mixed_batches: &[],
        selected_tools: &tools,
        all_tools: &tools,
        filters_empty: true,
    };
    let mut blockers = Vec::new();
    append_encode_corpus_blockers(&mut blockers, &input);
    for expected in [
        "generated-fixtures-included",
        "external-unique-input-count-below-24",
        "mixed-external-batches-missing",
        "external-gray8-source-missing",
        "external-dimension-diversity-below-3",
        "external-source-format-diversity-below-2",
        "external-manifest-coverage-missing",
        "external-corpus-name-missing",
        "external-license-status-missing",
        "external-license-status-not-publishable",
        "external-source-command-missing",
        "external-workload-corpus-missing",
    ] {
        assert!(
            blockers.iter().any(|blocker| blocker == expected),
            "missing {expected}: {blockers:?}"
        );
    }

    let mut tool_blockers = Vec::new();
    append_encode_tool_blockers(&mut tool_blockers, &input);
    assert!(tool_blockers.contains(&"openjpeg-compress-unavailable".to_string()));
    assert!(tool_blockers.contains(&"grok-compress-unavailable".to_string()));
    assert!(tool_blockers.contains(&"openjpeg-compress-version-unavailable".to_string()));
    assert!(tool_blockers.contains(&"grok-compress-version-unavailable".to_string()));
}

#[test]
fn mixed_group_labels_and_blockers_cover_empty_partial_and_complete_groups() {
    assert_eq!(mixed_external_max_distinct_inputs(&[]), 0);
    assert_eq!(mixed_external_min_distinct_inputs(&[]), 0);
    assert_eq!(mixed_external_group_distinct_inputs_label(&[]), "none");

    let first = image_case("first", "external:first", 1, &[1]);
    let second = image_case("second", "external:second", 1, &[2]);
    let cases = vec![first.clone(), second.clone()];
    let partial = MixedImageBatch {
        name: "partial".to_string(),
        cases: vec![first],
        components: 1,
    };
    let complete = MixedImageBatch {
        name: "complete".to_string(),
        cases: vec![second, cases[0].clone()],
        components: 1,
    };
    assert_eq!(mixed_external_max_distinct_inputs(&[partial, complete]), 2);
    assert_eq!(
        mixed_external_min_distinct_inputs(&[MixedImageBatch {
            name: "one".to_string(),
            cases: vec![cases[0].clone()],
            components: 1,
        }]),
        1
    );

    let mut blockers = Vec::new();
    require_mixed_encode_group(&mut blockers, &cases, &[], 1);
    assert_eq!(blockers, ["mixed-external-gray8-distinct-inputs-below-2"]);
    blockers.clear();
    require_mixed_encode_group(&mut blockers, &cases[..1], &[], 1);
    assert_eq!(blockers, ["external-gray8-mixed-input-count-below-2"]);
}
