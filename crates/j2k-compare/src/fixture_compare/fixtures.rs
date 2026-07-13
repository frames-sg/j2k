// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    common, encode_j2k_lossless, external_fixture_metadata, fixture_manifest_from_env,
    include_generated_fixtures, patterned_gray8, patterned_rgb8, pixel_format_label,
    sanitized_stem, unique_input_count, wrap_j2k_codestream, wrap_jp2_codestream, Codec, Container,
    Downscale, EncodeBackendPreference, FixtureCase, FixtureManifest, FixtureMetadata,
    J2kBlockCodingMode, J2kDecoder, J2kEncodeValidation, J2kFileWrapOptions,
    J2kLosslessEncodeOptions, J2kLosslessSamples, MixedFixtureBatch, Operation, Path, PathBuf,
    PixelFormat, Rect, LARGE_SIDE, SMALL_SIDE,
};

pub(super) fn all_fixture_cases() -> Result<Vec<FixtureCase>, String> {
    let manifest = fixture_manifest_from_env()?;
    let mut cases = if include_generated_fixtures() {
        fixture_cases()?
    } else {
        Vec::new()
    };
    for dir in external_input_dirs() {
        cases.extend(load_external_fixture_cases(&dir, manifest.as_ref())?);
    }
    if cases.is_empty() {
        return Err(
            "no fixture cases available; enable generated fixtures or set J2K_FIXTURE_COMPARE_INPUT_DIRS"
                .to_string(),
        );
    }
    Ok(cases)
}

pub(super) fn mixed_external_batches(cases: &[FixtureCase]) -> Vec<MixedFixtureBatch> {
    let mut groups: Vec<MixedFixtureBatch> = Vec::new();
    for case in cases
        .iter()
        .filter(|case| case.input_source.starts_with("external:"))
    {
        let Some(group) = groups.iter_mut().find(|group| {
            group.format == case.format && group.operation_class == case.operation.class()
        }) else {
            groups.push(MixedFixtureBatch {
                name: format!(
                    "external_mixed_{}_{}",
                    pixel_format_label(case.format),
                    case.operation.class().label().replace('-', "_")
                ),
                cases: vec![case.clone()],
                format: case.format,
                operation_class: case.operation.class(),
            });
            continue;
        };
        group.cases.push(case.clone());
    }
    groups
        .into_iter()
        .filter(|group| unique_input_count(&group.cases) > 1)
        .collect()
}

#[expect(
    clippy::too_many_lines,
    reason = "generated fixture catalog keeps names, wrappers, and operations in one reviewable inventory"
)]
pub(super) fn fixture_cases() -> Result<Vec<FixtureCase>, String> {
    let roi64 = Rect {
        x: 32,
        y: 32,
        w: 64,
        h: 64,
    };
    let roi256 = Rect {
        x: 128,
        y: 128,
        w: 256,
        h: 256,
    };
    let classic_gray_128 = encode_gray(SMALL_SIDE, SMALL_SIDE, Codec::Classic)?;
    let classic_rgb_128 = encode_rgb(SMALL_SIDE, SMALL_SIDE, Codec::Classic)?;
    let classic_rgb_512 = encode_rgb(LARGE_SIDE, LARGE_SIDE, Codec::Classic)?;
    let htj2k_gray_128 = encode_gray(SMALL_SIDE, SMALL_SIDE, Codec::Htj2k)?;
    let htj2k_rgb_128 = encode_rgb(SMALL_SIDE, SMALL_SIDE, Codec::Htj2k)?;
    let htj2k_rgb_512 = encode_rgb(LARGE_SIDE, LARGE_SIDE, Codec::Htj2k)?;
    let classic_rgb_128_jp2 =
        wrap_jp2_codestream(&classic_rgb_128, SMALL_SIDE, SMALL_SIDE, 3, 8, 16);
    let classic_rgb_512_jp2 =
        wrap_jp2_codestream(&classic_rgb_512, LARGE_SIDE, LARGE_SIDE, 3, 8, 16);
    let htj2k_rgb_128_jph = wrap_j2k_codestream(&htj2k_rgb_128, J2kFileWrapOptions::jph())
        .map_err(|error| format!("wrap generated HTJ2K 128 fixture as JPH: {error}"))?;
    let htj2k_rgb_512_jph = wrap_j2k_codestream(&htj2k_rgb_512, J2kFileWrapOptions::jph())
        .map_err(|error| format!("wrap generated HTJ2K 512 fixture as JPH: {error}"))?;

    Ok(vec![
        case_from_bytes(
            "classic_raw_gray8_128_full",
            generated_metadata("j2k-generated"),
            Codec::Classic,
            Container::RawCodestream,
            classic_gray_128,
            Operation::Full,
        )?,
        case_from_bytes(
            "classic_raw_rgb8_128_full",
            generated_metadata("j2k-generated"),
            Codec::Classic,
            Container::RawCodestream,
            classic_rgb_128,
            Operation::Full,
        )?,
        case_from_bytes(
            "classic_jp2_rgb8_128_full",
            generated_metadata("j2k-generated-jp2-wrapper"),
            Codec::Classic,
            Container::Jp2,
            classic_rgb_128_jp2.clone(),
            Operation::Full,
        )?,
        case_from_bytes(
            "classic_jp2_rgb8_128_roi64",
            generated_metadata("j2k-generated-jp2-wrapper"),
            Codec::Classic,
            Container::Jp2,
            classic_rgb_128_jp2.clone(),
            Operation::Region(roi64),
        )?,
        case_from_bytes(
            "classic_jp2_rgb8_128_q4",
            generated_metadata("j2k-generated-jp2-wrapper"),
            Codec::Classic,
            Container::Jp2,
            classic_rgb_128_jp2.clone(),
            Operation::Scaled(Downscale::Quarter),
        )?,
        case_from_bytes(
            "classic_jp2_rgb8_128_roi64_q4",
            generated_metadata("j2k-generated-jp2-wrapper"),
            Codec::Classic,
            Container::Jp2,
            classic_rgb_128_jp2,
            Operation::RegionScaled {
                roi: roi64,
                scale: Downscale::Quarter,
            },
        )?,
        case_from_bytes(
            "classic_jp2_rgb8_512_roi256_q4",
            generated_metadata("j2k-generated-jp2-wrapper"),
            Codec::Classic,
            Container::Jp2,
            classic_rgb_512_jp2,
            Operation::RegionScaled {
                roi: roi256,
                scale: Downscale::Quarter,
            },
        )?,
        case_from_bytes(
            "htj2k_raw_gray8_128_full",
            generated_metadata("j2k-generated"),
            Codec::Htj2k,
            Container::RawCodestream,
            htj2k_gray_128,
            Operation::Full,
        )?,
        case_from_bytes(
            "htj2k_raw_rgb8_128_full",
            generated_metadata("j2k-generated"),
            Codec::Htj2k,
            Container::RawCodestream,
            htj2k_rgb_128,
            Operation::Full,
        )?,
        case_from_bytes(
            "htj2k_jph_rgb8_128_full",
            generated_metadata("j2k-generated-jph-wrapper"),
            Codec::Htj2k,
            Container::Jph,
            htj2k_rgb_128_jph.clone(),
            Operation::Full,
        )?,
        case_from_bytes(
            "htj2k_jph_rgb8_128_roi64_q4",
            generated_metadata("j2k-generated-jph-wrapper"),
            Codec::Htj2k,
            Container::Jph,
            htj2k_rgb_128_jph,
            Operation::RegionScaled {
                roi: roi64,
                scale: Downscale::Quarter,
            },
        )?,
        case_from_bytes(
            "htj2k_jph_rgb8_512_roi256_q4",
            generated_metadata("j2k-generated-jph-wrapper"),
            Codec::Htj2k,
            Container::Jph,
            htj2k_rgb_512_jph,
            Operation::RegionScaled {
                roi: roi256,
                scale: Downscale::Quarter,
            },
        )?,
    ])
}

pub(super) fn case_from_bytes(
    name: impl Into<String>,
    metadata: FixtureMetadata,
    codec: Codec,
    container: Container,
    bytes: Vec<u8>,
    operation: Operation,
) -> Result<FixtureCase, String> {
    let name = name.into();
    let info = J2kDecoder::inspect(&bytes).map_err(|error| format!("{name}: inspect: {error}"))?;
    let format = pixel_format(info.components, info.bit_depth)
        .ok_or_else(|| format!("{name}: unsupported output shape for benchmark"))?;
    if let Some(roi) = operation.roi() {
        if !roi.is_within(info.dimensions) {
            return Err(format!("{name}: ROI {roi:?} exceeds {:?}", info.dimensions));
        }
    }
    Ok(FixtureCase {
        name,
        input_source: metadata.input_source,
        corpus_category: metadata.corpus_category,
        corpus_name: metadata.corpus_name,
        license_status: metadata.license_status,
        encode_command: metadata.encode_command,
        manifest_status: metadata.manifest_status,
        codec,
        container,
        bytes,
        dimensions: info.dimensions,
        format,
        operation,
        source_fnv1a64: metadata.source_fnv1a64,
    })
}

pub(super) fn generated_metadata(input_source: &str) -> FixtureMetadata {
    FixtureMetadata {
        input_source: input_source.to_string(),
        corpus_category: "generated-dev".to_string(),
        corpus_name: "j2k-generated-fixture-matrix".to_string(),
        license_status: "repo-generated".to_string(),
        encode_command: "j2k-lossless-cpu-roundtrip".to_string(),
        manifest_status: "generated".to_string(),
        source_fnv1a64: None,
    }
}

pub(super) fn external_input_dirs() -> Vec<PathBuf> {
    if let Some(paths) = std::env::var_os("J2K_FIXTURE_COMPARE_INPUT_DIRS") {
        return std::env::split_paths(&paths).collect();
    }
    std::env::var_os("J2K_FIXTURE_COMPARE_INPUT_DIR")
        .map(PathBuf::from)
        .into_iter()
        .collect()
}

pub(super) fn load_external_fixture_cases(
    dir: &Path,
    manifest: Option<&FixtureManifest>,
) -> Result<Vec<FixtureCase>, String> {
    if !dir.is_dir() {
        return Err(format!(
            "J2K_FIXTURE_COMPARE_INPUT_DIR is not a directory: {}",
            dir.display()
        ));
    }
    let mut paths = Vec::new();
    collect_j2k_paths(dir, &mut paths)?;
    paths.sort();
    if paths.is_empty() {
        return Err(format!(
            "external input dir {} contains no .j2k/.j2c/.jp2/.jph/.jhc fixtures",
            dir.display()
        ));
    }

    let mut cases = Vec::new();
    for path in paths {
        let bytes =
            std::fs::read(&path).map_err(|error| format!("read {}: {error}", path.display()))?;
        let info = J2kDecoder::inspect(&bytes)
            .map_err(|error| format!("inspect external fixture {}: {error}", path.display()))?;
        if pixel_format(info.components, info.bit_depth).is_none() {
            return Err(format!(
                "external fixture {} has unsupported benchmark shape: components={} bit_depth={}",
                path.display(),
                info.components,
                info.bit_depth
            ));
        }
        let stem = sanitized_stem(&path);
        let codec = codec_from_bytes(&bytes);
        let container = container_from_path_and_bytes(&path, &bytes);
        let metadata = external_fixture_metadata(&path, &bytes, codec, container, manifest)?;
        cases.push(case_from_bytes(
            format!("external_{stem}_full"),
            metadata.clone(),
            codec,
            container,
            bytes.clone(),
            Operation::Full,
        )?);

        let min_side = info.dimensions.0.min(info.dimensions.1);
        if min_side >= 128 && should_emit_external_region_scaled(&metadata) {
            let roi = external_scaled_roi(info.dimensions);
            cases.push(case_from_bytes(
                format!("external_{stem}_roi{}_q4", roi.w),
                metadata,
                codec,
                container,
                bytes,
                Operation::RegionScaled {
                    roi,
                    scale: Downscale::Quarter,
                },
            )?);
        }
    }
    Ok(cases)
}

pub(super) fn should_emit_external_region_scaled(metadata: &FixtureMetadata) -> bool {
    matches!(
        metadata.corpus_category.as_str(),
        "natural-image" | "medical-domain" | "remote-sensing"
    )
}

pub(super) fn external_scaled_roi(dimensions: (u32, u32)) -> Rect {
    let min_side = dimensions.0.min(dimensions.1);
    let denominator = Downscale::Quarter.denominator();
    let roi_side = round_down_to_multiple((min_side / 2).max(64), denominator);
    let x = round_down_to_multiple((dimensions.0 - roi_side) / 2, denominator);
    let y = round_down_to_multiple((dimensions.1 - roi_side) / 2, denominator);
    Rect {
        x,
        y,
        w: roi_side,
        h: roi_side,
    }
}

pub(super) fn round_down_to_multiple(value: u32, multiple: u32) -> u32 {
    debug_assert!(multiple > 0);
    value - (value % multiple)
}

pub(super) fn collect_j2k_paths(dir: &Path, paths: &mut Vec<PathBuf>) -> Result<(), String> {
    for entry in std::fs::read_dir(dir)
        .map_err(|error| format!("read external input dir {}: {error}", dir.display()))?
    {
        let path = entry
            .map_err(|error| format!("read external input dir entry: {error}"))?
            .path();
        if path.is_dir() {
            collect_j2k_paths(&path, paths)?;
        } else if is_j2k_path(&path) {
            paths.push(path);
        }
    }
    Ok(())
}

pub(super) fn external_source_label(path: &Path) -> Result<String, String> {
    common::external_source_label(
        path,
        "external fixture path contains a control character and cannot be represented safely",
    )
}

pub(super) fn is_j2k_path(path: &Path) -> bool {
    path.extension()
        .and_then(|value| value.to_str())
        .is_some_and(|extension| {
            matches!(
                extension.to_ascii_lowercase().as_str(),
                "j2k" | "j2c" | "jp2" | "jph" | "jhc"
            )
        })
}

pub(super) fn container_from_path_and_bytes(path: &Path, bytes: &[u8]) -> Container {
    if let Some(extension) = path.extension().and_then(|value| value.to_str()) {
        match extension.to_ascii_lowercase().as_str() {
            "jph" => return Container::Jph,
            "jhc" => return Container::Jhc,
            _ => {}
        }
    }
    container_from_bytes(bytes)
}

pub(super) fn container_from_bytes(bytes: &[u8]) -> Container {
    if bytes.starts_with(&[0, 0, 0, 12, b'j', b'P', b' ', b' ']) {
        Container::Jp2
    } else {
        Container::RawCodestream
    }
}

pub(super) fn codec_from_bytes(bytes: &[u8]) -> Codec {
    let Ok(payload) = j2k::extract_j2k_codestream_payload(bytes) else {
        return Codec::Unknown;
    };
    match j2k_native::inspect_j2k_codestream_header(payload.codestream()) {
        Ok(header) if header.high_throughput => Codec::Htj2k,
        Ok(_) => Codec::Classic,
        Err(_) => Codec::Unknown,
    }
}

pub(super) fn pixel_format(components: u16, bit_depth: u8) -> Option<PixelFormat> {
    match (components, bit_depth) {
        (1, 8) => Some(PixelFormat::Gray8),
        (3, 8) => Some(PixelFormat::Rgb8),
        _ => None,
    }
}

pub(super) fn encode_gray(width: u32, height: u32, codec: Codec) -> Result<Vec<u8>, String> {
    let pixels = patterned_gray8(width, height);
    encode_lossless(&pixels, width, height, 1, codec)
}

pub(super) fn encode_rgb(width: u32, height: u32, codec: Codec) -> Result<Vec<u8>, String> {
    let pixels = patterned_rgb8(width, height);
    encode_lossless(&pixels, width, height, 3, codec)
}

pub(super) fn encode_lossless(
    pixels: &[u8],
    width: u32,
    height: u32,
    components: u8,
    codec: Codec,
) -> Result<Vec<u8>, String> {
    let samples = J2kLosslessSamples::new(pixels, width, height, u16::from(components), 8, false)
        .map_err(|error| error.to_string())?;
    let block_coding_mode = match codec {
        Codec::Classic => J2kBlockCodingMode::Classic,
        Codec::Htj2k => J2kBlockCodingMode::HighThroughput,
        Codec::Unknown => {
            return Err("cannot encode generated fixture for unknown codec".to_string())
        }
    };
    let options = J2kLosslessEncodeOptions::default()
        .with_backend(EncodeBackendPreference::CpuOnly)
        .with_block_coding_mode(block_coding_mode)
        .with_max_decomposition_levels(Some(2))
        .with_validation(J2kEncodeValidation::CpuRoundTrip);
    Ok(encode_j2k_lossless(samples, &options)
        .map_err(|error| error.to_string())?
        .codestream)
}

#[cfg(test)]
mod tests;
