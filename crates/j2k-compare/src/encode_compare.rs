// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{
    collections::{HashMap, HashSet},
    fs,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    time::Instant,
};

use crate::{common, grok, openjpeg, parse_positive_usize, sample_stats, usize_to_f64};
use j2k::{
    encode_j2k_lossless, EncodeBackendPreference, J2kBlockCodingMode, J2kDecoder,
    J2kEncodeValidation, J2kLosslessEncodeOptions, J2kLosslessSamples,
};
use j2k_core::PixelFormat;
use j2k_test_support::{
    fnv1a64_hex, fnv1a64_hex_slices, patterned_gray8, patterned_rgb8, wrap_jp2_codestream,
};

use crate::common::{
    build_profile_label, canonicalize_manifest_row_path, combined_batch_sizes,
    default_batch_sizes_present, env_falsey, env_truthy, git_dirty_label, git_dirty_status,
    git_revision, git_revision_label, host_hardware_label, is_publishable_license_status,
    join_string_labels, join_usizes, mib_per_second, optional_manifest_column, sanitized_stem,
};

mod types;
use self::types::{
    EncodeManifest, EncodeManifestEntry, EncodeMeasurementState, EncoderKind, EncoderTool,
    ExternalImageMetadata, ImageCase, Measurement, MetadataInput, MixedImageBatch, PnmImage,
    DEFAULT_CASE_BATCH_SIZES, DEFAULT_MIXED_BATCH_SIZES, DEFAULT_REPEATS,
    MIN_PUBLICATION_EXTERNAL_DIMENSIONS, MIN_PUBLICATION_EXTERNAL_IMAGES,
    MIN_PUBLICATION_EXTERNAL_SOURCE_FORMATS, MIN_PUBLICATION_MIXED_DISTINCT_INPUTS,
};
mod cli;
use self::cli::{
    batch_size_config_from_env, encode_one, encode_work_dir, include_generated_images,
    include_kakadu_encoder, print_usage, validate_tool_gates,
};
mod images;
use self::images::{all_image_cases, mixed_external_batches, read_pnm, select_cases};
mod tools;
use self::tools::{
    all_encoder_tools, command_template, dimensions_label, run_encoder_once, samples_label,
    selected_encoder_tools, selected_encoders_label, tool_available, tool_command, tool_version,
    tool_version_available,
};
mod validation;
use self::validation::validate_case_encoder;
mod measurement;
use self::measurement::{measure_case_rows, measure_mixed_rows};
mod render;
use self::render::{
    emit_metadata, measurement_row, mixed_case_at, mixed_measurement_row, mixed_skip_row, skip_row,
    unique_image_count,
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

#[cfg(test)]
mod tests {
    use super::render::{
        external_manifest_covered_case_count, external_manifest_missing_case_count,
        mixed_external_group_distinct_inputs_label, publication_blockers,
    };
    use super::{
        canonicalize_manifest_row_path, measurement_row, EncoderKind, EncoderTool, ImageCase,
        Measurement, MetadataInput, MixedImageBatch, DEFAULT_CASE_BATCH_SIZES,
        DEFAULT_MIXED_BATCH_SIZES,
    };
    use crate::common;
    use std::path::Path;

    fn test_batch_size_config_from_values(
        case_batch_sizes: Option<&str>,
        mixed_batch_sizes: Option<&str>,
        legacy: Option<Vec<usize>>,
    ) -> Result<common::BatchSizeConfig, String> {
        common::batch_size_config_from_values(
            case_batch_sizes,
            mixed_batch_sizes,
            legacy,
            "J2K_ENCODE_COMPARE_CASE_BATCH_SIZES",
            "J2K_ENCODE_COMPARE_MIXED_BATCH_SIZES",
            DEFAULT_CASE_BATCH_SIZES,
            DEFAULT_MIXED_BATCH_SIZES,
        )
    }

    #[test]
    fn encode_batch_config_defaults_keep_large_batches_mixed_only() {
        let config = test_batch_size_config_from_values(None, None, None)
            .expect("default batch config parses");

        assert_eq!(config.case_batch_sizes, DEFAULT_CASE_BATCH_SIZES);
        assert_eq!(config.mixed_batch_sizes, DEFAULT_MIXED_BATCH_SIZES);
    }

    #[test]
    fn encode_batch_config_split_env_overrides_legacy_independently() {
        let config = test_batch_size_config_from_values(Some("3"), None, Some(vec![2, 4]))
            .expect("case override with legacy config parses");

        assert_eq!(config.case_batch_sizes, vec![3]);
        assert_eq!(config.mixed_batch_sizes, vec![2, 4]);

        let config = test_batch_size_config_from_values(None, Some("8,16"), Some(vec![2, 4]))
            .expect("mixed override with legacy config parses");

        assert_eq!(config.case_batch_sizes, vec![2, 4]);
        assert_eq!(config.mixed_batch_sizes, vec![8, 16]);
    }

    #[test]
    fn encode_manifest_path_remaps_to_supplied_fixture_root_by_suffix() {
        let root = std::env::current_dir()
            .expect("current dir")
            .join("target")
            .join("j2k-encode-manifest-remap-test")
            .join(std::process::id().to_string());
        let fixture_root = root.join("staged-pnm");
        let fixture = fixture_root.join("sample.ppm");
        std::fs::create_dir_all(&fixture_root).expect("create dirs");
        std::fs::write(&fixture, b"P6\n1 1\n255\nabc").expect("fixture");

        let resolved = canonicalize_manifest_row_path(
            "/old/worktree/target/j2k-public-corpora/materialized-kodak/staged-pnm/sample.ppm",
            Path::new("/unused"),
            &[fixture_root],
            "encode manifest",
            Path::new("encode-fixtures.tsv"),
            2,
        )
        .expect("remap stale absolute path");

        assert_eq!(resolved, fixture.canonicalize().expect("canonical fixture"));
    }

    #[test]
    fn encode_manifest_mixed_publication_and_row_width_have_direct_owners() {
        let gray = image_case("gray", "external:gray", 1, "covered", 64, 64);
        let mut rgb = image_case("rgb", "external:rgb", 3, "missing", 128, 64);
        rgb.source_format = "ppm".to_string();
        let cases = vec![gray.clone(), rgb.clone()];
        assert_eq!(external_manifest_covered_case_count(&cases), 1);
        assert_eq!(external_manifest_missing_case_count(&cases), 1);

        let mixed = MixedImageBatch {
            name: "external_mixed_rgb8_encode".to_string(),
            cases: vec![gray.clone(), rgb.clone()],
            components: 3,
        };
        assert_eq!(
            mixed_external_group_distinct_inputs_label(&[mixed]),
            "external_mixed_rgb8_encode:2"
        );

        let selected_tools = vec![tool(EncoderKind::J2k, true)];
        let all_tools = vec![
            tool(EncoderKind::J2k, true),
            tool(EncoderKind::OpenJpeg, false),
            tool(EncoderKind::Grok, false),
        ];
        let input = MetadataInput {
            args: &["jp2k_encode_compare".to_string()],
            repeats: 1,
            batch_sizes: &[1],
            case_batch_sizes: &[1],
            mixed_batch_sizes: &[1],
            cases: &cases,
            mixed_batches: &[],
            selected_tools: &selected_tools,
            all_tools: &all_tools,
            filters_empty: false,
        };
        let blockers = publication_blockers(&input);
        assert!(blockers.contains(&"case-filters-present".to_string()));
        assert!(blockers.contains(&"openjpeg-not-selected".to_string()));
        assert!(blockers.contains(&"external-manifest-coverage-missing".to_string()));
        assert!(blockers.contains(&"mixed-external-batches-missing".to_string()));

        let measurement = Measurement {
            batch_size: 1,
            repeats: 1,
            median_us: 10.0,
            mean_us: 11.0,
            images_per_second_median: 100.0,
            encoded_bytes_per_repeat: 32,
            samples_us: vec![10.0],
        };
        let row = measurement_row(EncoderKind::J2k, &gray, &measurement, "jp2k_encode_compare");
        assert_eq!(row.split('\t').count(), 26);
    }

    fn image_case(
        name: &str,
        input_source: &str,
        components: u8,
        manifest_status: &str,
        width: u32,
        height: u32,
    ) -> ImageCase {
        ImageCase {
            name: name.to_string(),
            input_source: input_source.to_string(),
            corpus_category: "natural-image".to_string(),
            corpus_name: "unit-corpus".to_string(),
            license_status: "cc0".to_string(),
            source_command: "unit-source".to_string(),
            manifest_status: manifest_status.to_string(),
            source_format: if components == 1 { "pgm" } else { "ppm" }.to_string(),
            width,
            height,
            components,
            pixels: vec![components; width as usize * height as usize * components as usize],
            pnm_path: Path::new("unit.pnm").to_path_buf(),
        }
    }

    fn tool(kind: EncoderKind, available: bool) -> EncoderTool {
        EncoderTool {
            kind,
            program: Path::new(kind.label()).to_path_buf(),
            available,
        }
    }
}
