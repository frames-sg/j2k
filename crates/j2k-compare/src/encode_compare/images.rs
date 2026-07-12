// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    canonicalize_manifest_row_path, common, fnv1a64_hex, fs, include_generated_images,
    optional_manifest_column, patterned_gray8, patterned_rgb8, sanitized_stem, unique_image_count,
    EncodeManifest, EncodeManifestEntry, ExternalImageMetadata, HashMap, ImageCase,
    MixedImageBatch, Path, PathBuf, PnmImage,
};

pub(super) fn all_image_cases(work_dir: &Path) -> Result<Vec<ImageCase>, String> {
    let manifest = encode_manifest_from_env()?;
    let mut cases = if include_generated_images() {
        generated_image_cases(work_dir)?
    } else {
        Vec::new()
    };
    for dir in external_input_dirs() {
        cases.extend(load_external_image_cases(
            &dir,
            work_dir,
            manifest.as_ref(),
        )?);
    }
    if cases.is_empty() {
        return Err(
            "no encode image cases available; enable generated images or set J2K_ENCODE_COMPARE_INPUT_DIRS"
                .to_string(),
        );
    }
    Ok(cases)
}

pub(super) fn generated_image_cases(work_dir: &Path) -> Result<Vec<ImageCase>, String> {
    let mut cases = Vec::new();
    for (name, width, height, components, pixels) in [
        (
            "generated_gray8_128",
            128,
            128,
            1,
            patterned_gray8(128, 128),
        ),
        ("generated_rgb8_128", 128, 128, 3, patterned_rgb8(128, 128)),
        ("generated_rgb8_512", 512, 512, 3, patterned_rgb8(512, 512)),
    ] {
        let pnm_path = work_dir.join(format!("{name}.{}", pnm_extension(components)?));
        write_pnm(&pnm_path, &pixels, width, height, components)?;
        cases.push(ImageCase {
            name: name.to_string(),
            input_source: "j2k-generated-image".to_string(),
            corpus_category: "generated-dev".to_string(),
            corpus_name: "j2k-generated-encode-matrix".to_string(),
            license_status: "repo-generated".to_string(),
            source_command: "j2k-test-support-pattern".to_string(),
            manifest_status: "generated".to_string(),
            source_format: "generated-pnm".to_string(),
            width,
            height,
            components,
            pixels,
            pnm_path,
        });
    }
    Ok(cases)
}

pub(super) fn external_input_dirs() -> Vec<PathBuf> {
    if let Some(paths) = std::env::var_os("J2K_ENCODE_COMPARE_INPUT_DIRS") {
        return std::env::split_paths(&paths).collect();
    }
    Vec::new()
}

pub(super) fn load_external_image_cases(
    dir: &Path,
    work_dir: &Path,
    manifest: Option<&EncodeManifest>,
) -> Result<Vec<ImageCase>, String> {
    if !dir.is_dir() {
        return Err(format!(
            "J2K_ENCODE_COMPARE_INPUT_DIRS entry is not a directory: {}",
            dir.display()
        ));
    }
    let mut paths = Vec::new();
    collect_source_image_paths(dir, &mut paths)?;
    paths.sort();
    if paths.is_empty() {
        return Err(format!(
            "external encode input dir {} contains no supported source images (.pgm/.ppm/.pnm/.png/.jpg/.jpeg/.tif/.tiff/.bmp)",
            dir.display()
        ));
    }
    let mut cases = Vec::new();
    for (index, path) in paths.into_iter().enumerate() {
        let parsed = read_source_image(&path)?;
        let metadata = external_image_metadata(&path, &parsed, manifest)?;
        let name = format!("external_{index:04}_{}", sanitized_stem(&path));
        let pnm_path = work_dir.join(format!("{}.{}", name, pnm_extension(parsed.components)?));
        write_pnm(
            &pnm_path,
            &parsed.pixels,
            parsed.width,
            parsed.height,
            parsed.components,
        )?;
        cases.push(ImageCase {
            name,
            input_source: metadata.input_source,
            corpus_category: metadata.corpus_category,
            corpus_name: metadata.corpus_name,
            license_status: metadata.license_status,
            source_command: metadata.source_command,
            manifest_status: metadata.manifest_status,
            source_format: source_format_label(&path),
            width: parsed.width,
            height: parsed.height,
            components: parsed.components,
            pixels: parsed.pixels,
            pnm_path,
        });
    }
    Ok(cases)
}

pub(super) fn encode_manifest_from_env() -> Result<Option<EncodeManifest>, String> {
    let Some(path) = std::env::var_os("J2K_ENCODE_COMPARE_MANIFEST").map(PathBuf::from) else {
        return Ok(None);
    };
    let text = fs::read_to_string(&path).map_err(|error| {
        format!(
            "read J2K_ENCODE_COMPARE_MANIFEST {}: {error}",
            path.display()
        )
    })?;
    let base = path.parent().unwrap_or_else(|| Path::new("."));
    let relocation_roots = external_input_dirs();
    let mut lines = text.lines().filter(|line| !line.trim().is_empty());
    let header = lines
        .next()
        .ok_or_else(|| format!("encode manifest {} is empty", path.display()))?;
    let headers = header.split('\t').collect::<Vec<_>>();
    let path_index = manifest_column(&headers, "path")?;
    let category_index = manifest_column(&headers, "corpus_category")?;
    let corpus_name_index = optional_manifest_column(&headers, "corpus_name");
    let license_status_index = optional_manifest_column(&headers, "license_status");
    let source_command_index = optional_manifest_column(&headers, "source_command");
    let hash_index = optional_manifest_column(&headers, "input_fnv1a64");

    let mut entries = HashMap::new();
    for (line_index, line) in lines.enumerate() {
        if line.trim_start().starts_with('#') {
            continue;
        }
        let fields = line.split('\t').collect::<Vec<_>>();
        let row_number = line_index + 2;
        let raw_path = manifest_field(&fields, path_index, "path", row_number)?;
        let canonical_path = canonicalize_manifest_row_path(
            raw_path,
            base,
            &relocation_roots,
            "encode manifest",
            &path,
            row_number,
        )?;
        let entry = EncodeManifestEntry {
            corpus_category: manifest_required_value(
                &fields,
                category_index,
                "corpus_category",
                row_number,
            )?,
            corpus_name: manifest_optional_value(
                &fields,
                corpus_name_index,
                "corpus_name",
                row_number,
            )?
            .unwrap_or_else(|| "not-recorded".to_string()),
            license_status: manifest_optional_value(
                &fields,
                license_status_index,
                "license_status",
                row_number,
            )?
            .unwrap_or_else(|| "not-recorded".to_string()),
            source_command: manifest_optional_value(
                &fields,
                source_command_index,
                "source_command",
                row_number,
            )?
            .unwrap_or_else(|| "not-recorded".to_string()),
            input_fnv1a64: manifest_optional_value(
                &fields,
                hash_index,
                "input_fnv1a64",
                row_number,
            )?,
        };
        if entries.insert(canonical_path, entry).is_some() {
            return Err(format!(
                "encode manifest {} row {row_number} duplicates path {raw_path}",
                path.display()
            ));
        }
    }

    Ok(Some(EncodeManifest { entries }))
}

pub(super) fn external_image_metadata(
    path: &Path,
    image: &PnmImage,
    manifest: Option<&EncodeManifest>,
) -> Result<ExternalImageMetadata, String> {
    let input_source = external_source_label(path)?;
    let Some(manifest) = manifest else {
        return Ok(ExternalImageMetadata {
            input_source,
            corpus_category: external_corpus_category(path),
            corpus_name: "path-inferred".to_string(),
            license_status: "not-recorded".to_string(),
            source_command: image.source_command.clone(),
            manifest_status: "not-covered".to_string(),
        });
    };
    let canonical_path = path
        .canonicalize()
        .map_err(|error| format!("canonicalize external image {}: {error}", path.display()))?;
    let Some(entry) = manifest.entries.get(&canonical_path) else {
        return Ok(ExternalImageMetadata {
            input_source,
            corpus_category: external_corpus_category(path),
            corpus_name: "path-inferred".to_string(),
            license_status: "not-recorded".to_string(),
            source_command: image.source_command.clone(),
            manifest_status: "not-covered".to_string(),
        });
    };
    if let Some(expected_hash) = &entry.input_fnv1a64 {
        let actual_hash = fnv1a64_hex(&image.pixels);
        if actual_hash != *expected_hash {
            return Err(format!(
                "external encode image {} hash mismatch: manifest {expected_hash} != actual {actual_hash}",
                path.display()
            ));
        }
    }
    let manifest_status = if entry.input_fnv1a64.is_some() {
        "covered"
    } else {
        "covered-unpinned"
    };

    Ok(ExternalImageMetadata {
        input_source,
        corpus_category: entry.corpus_category.clone(),
        corpus_name: entry.corpus_name.clone(),
        license_status: entry.license_status.clone(),
        source_command: entry.source_command.clone(),
        manifest_status: manifest_status.to_string(),
    })
}

pub(super) fn manifest_column(headers: &[&str], name: &str) -> Result<usize, String> {
    common::manifest_column(headers, name, "encode")
}

pub(super) fn manifest_field<'a>(
    fields: &'a [&str],
    index: usize,
    name: &str,
    row_number: usize,
) -> Result<&'a str, String> {
    common::manifest_field(fields, index, name, row_number, "encode")
}

pub(super) fn manifest_required_value(
    fields: &[&str],
    index: usize,
    name: &str,
    row_number: usize,
) -> Result<String, String> {
    common::manifest_required_value(fields, index, name, row_number, "encode")
}

pub(super) fn manifest_optional_value(
    fields: &[&str],
    index: Option<usize>,
    name: &str,
    row_number: usize,
) -> Result<Option<String>, String> {
    common::manifest_optional_value(fields, index, name, row_number, "encode")
}

pub(super) fn collect_source_image_paths(
    dir: &Path,
    paths: &mut Vec<PathBuf>,
) -> Result<(), String> {
    for entry in
        fs::read_dir(dir).map_err(|error| format!("read input dir {}: {error}", dir.display()))?
    {
        let path = entry
            .map_err(|error| format!("read input dir entry: {error}"))?
            .path();
        if path.is_dir() {
            collect_source_image_paths(&path, paths)?;
        } else if is_supported_source_image_path(&path) {
            paths.push(path);
        }
    }
    Ok(())
}

pub(super) fn is_supported_source_image_path(path: &Path) -> bool {
    path.extension()
        .and_then(|value| value.to_str())
        .is_some_and(|extension| {
            matches!(
                extension.to_ascii_lowercase().as_str(),
                "pgm" | "ppm" | "pnm" | "png" | "jpg" | "jpeg" | "tif" | "tiff" | "bmp"
            )
        })
}

pub(super) fn is_pnm_path(path: &Path) -> bool {
    path.extension()
        .and_then(|value| value.to_str())
        .is_some_and(|extension| {
            matches!(
                extension.to_ascii_lowercase().as_str(),
                "pgm" | "ppm" | "pnm"
            )
        })
}

pub(super) fn external_source_label(path: &Path) -> Result<String, String> {
    common::external_source_label(path, "external image path contains a control character")
}

pub(super) fn external_corpus_category(path: &Path) -> String {
    common::infer_corpus_category(path).to_string()
}

pub(super) fn source_format_label(path: &Path) -> String {
    path.extension()
        .and_then(|value| value.to_str())
        .map_or_else(|| "unknown".to_string(), str::to_ascii_lowercase)
}

pub(super) fn select_cases(
    cases: Vec<ImageCase>,
    filters: &[&str],
) -> Result<Vec<ImageCase>, String> {
    if filters.is_empty() {
        return Ok(cases);
    }
    let selected = cases
        .into_iter()
        .filter(|case| filters.iter().any(|filter| case.name.contains(filter)))
        .collect::<Vec<_>>();
    if selected.is_empty() {
        return Err(format!(
            "no encode cases matched filters: {}",
            filters.join(",")
        ));
    }
    Ok(selected)
}

pub(super) fn mixed_external_batches(cases: &[ImageCase]) -> Vec<MixedImageBatch> {
    let mut batches = Vec::new();
    for components in [1, 3] {
        let group = cases
            .iter()
            .filter(|case| {
                case.input_source.starts_with("external:") && case.components == components
            })
            .cloned()
            .collect::<Vec<_>>();
        if unique_image_count(&group) > 1 {
            let label = if components == 1 { "gray8" } else { "rgb8" };
            batches.push(MixedImageBatch {
                name: format!("external_mixed_{label}_encode"),
                cases: group,
                components,
            });
        }
    }
    batches
}

pub(super) fn write_pnm(
    path: &Path,
    pixels: &[u8],
    width: u32,
    height: u32,
    components: u8,
) -> Result<(), String> {
    j2k_test_support::write_pnm(path, pixels, width, height, usize::from(components))
        .map_err(|error| format!("write {}: {error}", path.display()))
}

pub(super) fn read_source_image(path: &Path) -> Result<PnmImage, String> {
    if is_pnm_path(path) {
        return read_pnm(path);
    }
    read_raster_image(path)
}

pub(super) fn read_raster_image(path: &Path) -> Result<PnmImage, String> {
    let reader = image::ImageReader::open(path)
        .map_err(|error| format!("open source image {}: {error}", path.display()))?
        .with_guessed_format()
        .map_err(|error| format!("guess source image format {}: {error}", path.display()))?;
    let decoded = reader
        .decode()
        .map_err(|error| format!("decode source image {}: {error}", path.display()))?;
    let width = decoded.width();
    let height = decoded.height();
    match decoded.color() {
        image::ColorType::L8 => Ok(PnmImage {
            width,
            height,
            components: 1,
            pixels: decoded.into_luma8().into_raw(),
            source_command: "image-crate-decode-to-pnm".to_string(),
        }),
        image::ColorType::Rgb8 => Ok(PnmImage {
            width,
            height,
            components: 3,
            pixels: decoded.into_rgb8().into_raw(),
            source_command: "image-crate-decode-to-pnm".to_string(),
        }),
        color => Err(format!(
            "{} has unsupported source color type {color:?}; expected 8-bit grayscale or RGB without alpha",
            path.display()
        )),
    }
}

pub(super) fn read_pnm(path: &Path) -> Result<PnmImage, String> {
    let image = j2k_test_support::read_pnm_image(path)
        .map_err(|error| format!("read {}: {error}", path.display()))?;
    let components = u8::try_from(image.channels).map_err(|_| {
        format!(
            "{} has unsupported component count {}",
            path.display(),
            image.channels
        )
    })?;
    Ok(PnmImage {
        width: image.width,
        height: image.height,
        components,
        pixels: image.pixels,
        source_command: "source-pnm".to_string(),
    })
}

pub(super) fn pnm_extension(components: u8) -> Result<&'static str, String> {
    match components {
        1 => Ok("pgm"),
        3 => Ok("ppm"),
        other => Err(format!("unsupported component count {other}")),
    }
}

#[cfg(test)]
mod tests;
