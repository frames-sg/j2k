// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    all_encoder_tools, common, encode_j2k_lossless, env_falsey, env_truthy, fs, read_pnm,
    selected_encoder_tools, tool_available, wrap_jp2_codestream, EncodeBackendPreference,
    EncoderKind, J2kBlockCodingMode, J2kEncodeValidation, J2kLosslessEncodeOptions,
    J2kLosslessSamples, PathBuf, DEFAULT_CASE_BATCH_SIZES, DEFAULT_MIXED_BATCH_SIZES,
};

pub(super) fn print_usage(program: &str) {
    eprintln!("usage: {program} [case-name-filter ...]");
    eprintln!("       {program} --encode-one --input FILE.pnm --output FILE.jp2");
    eprintln!("Runs CLI-style lossless classic JPEG 2000 encoder benchmarks.");
}

pub(super) fn encode_one(args: &[String]) -> Result<(), String> {
    let mut input = None;
    let mut output = None;
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--input" => {
                index += 1;
                input = args.get(index).map(PathBuf::from);
            }
            "--output" => {
                index += 1;
                output = args.get(index).map(PathBuf::from);
            }
            other => return Err(format!("unknown --encode-one argument `{other}`")),
        }
        index += 1;
    }
    let input = input.ok_or_else(|| "--encode-one requires --input".to_string())?;
    let output = output.ok_or_else(|| "--encode-one requires --output".to_string())?;
    let image = read_pnm(&input)?;
    let samples = J2kLosslessSamples::new(
        &image.pixels,
        image.width,
        image.height,
        u16::from(image.components),
        8,
        false,
    )
    .map_err(|error| error.to_string())?;
    let options = J2kLosslessEncodeOptions::default()
        .with_backend(EncodeBackendPreference::CpuOnly)
        .with_block_coding_mode(J2kBlockCodingMode::Classic)
        .with_max_decomposition_levels(Some(2))
        .with_validation(J2kEncodeValidation::External);
    let encoded = encode_j2k_lossless(samples, &options).map_err(|error| error.to_string())?;
    let jp2 = wrap_jp2_codestream(
        &encoded.codestream,
        image.width,
        image.height,
        u16::from(image.components),
        8,
        16,
    );
    fs::write(&output, jp2).map_err(|error| format!("write {}: {error}", output.display()))
}

pub(super) fn validate_tool_gates() -> Result<(), String> {
    let all_tools = all_encoder_tools()?;
    let selected_tools = selected_encoder_tools(&all_tools)?;
    if env_truthy("J2K_REQUIRE_OPENJPEG") && !tool_available(&all_tools, EncoderKind::OpenJpeg) {
        return Err("J2K_REQUIRE_OPENJPEG is set but opj_compress is unavailable".to_string());
    }
    if env_truthy("J2K_REQUIRE_GROK") && !tool_available(&all_tools, EncoderKind::Grok) {
        return Err("J2K_REQUIRE_GROK is set but grk_compress is unavailable".to_string());
    }
    if env_truthy("J2K_REQUIRE_KAKADU") && !tool_available(&all_tools, EncoderKind::Kakadu) {
        return Err(
            "J2K_REQUIRE_KAKADU is set but kdu_compress is unavailable; set J2K_KDU_COMPRESS_BIN"
                .to_string(),
        );
    }
    if env_truthy("J2K_REQUIRE_OPENJPEG")
        && !selected_tools
            .iter()
            .any(|tool| tool.kind == EncoderKind::OpenJpeg)
    {
        return Err(
            "J2K_REQUIRE_OPENJPEG is set but J2K_ENCODE_COMPARE_ENCODERS excludes openjpeg"
                .to_string(),
        );
    }
    if env_truthy("J2K_REQUIRE_GROK")
        && !selected_tools
            .iter()
            .any(|tool| tool.kind == EncoderKind::Grok)
    {
        return Err(
            "J2K_REQUIRE_GROK is set but J2K_ENCODE_COMPARE_ENCODERS excludes grok".to_string(),
        );
    }
    if env_truthy("J2K_REQUIRE_KAKADU")
        && !selected_tools
            .iter()
            .any(|tool| tool.kind == EncoderKind::Kakadu)
    {
        return Err(
            "J2K_REQUIRE_KAKADU is set but J2K_ENCODE_COMPARE_ENCODERS excludes kakadu".to_string(),
        );
    }
    Ok(())
}

pub(super) fn include_generated_images() -> bool {
    !env_falsey("J2K_ENCODE_COMPARE_INCLUDE_GENERATED")
}

pub(super) fn include_kakadu_encoder() -> bool {
    env_truthy("J2K_INCLUDE_KAKADU")
        || env_truthy("J2K_REQUIRE_KAKADU")
        || std::env::var("J2K_ENCODE_COMPARE_ENCODERS")
            .ok()
            .is_some_and(|value| {
                value
                    .split(',')
                    .map(str::trim)
                    .map(str::to_ascii_lowercase)
                    .any(|part| matches!(part.as_str(), "kakadu" | "kdu"))
            })
}

pub(super) fn batch_size_config_from_env() -> Result<common::BatchSizeConfig, String> {
    common::batch_size_config_from_env(
        common::BatchSizeEnv {
            case_batch_sizes: "J2K_ENCODE_COMPARE_CASE_BATCH_SIZES",
            mixed_batch_sizes: "J2K_ENCODE_COMPARE_MIXED_BATCH_SIZES",
            legacy_batch_sizes: "J2K_ENCODE_COMPARE_BATCH_SIZES",
            legacy_batch_size: None,
        },
        DEFAULT_CASE_BATCH_SIZES,
        DEFAULT_MIXED_BATCH_SIZES,
    )
}

pub(super) fn encode_work_dir() -> Result<PathBuf, String> {
    let dir = std::env::current_dir()
        .map_err(|error| format!("current_dir: {error}"))?
        .join("target")
        .join("j2k-encode-compare")
        .join(std::process::id().to_string());
    fs::create_dir_all(&dir).map_err(|error| format!("create {}: {error}", dir.display()))?;
    Ok(dir)
}
