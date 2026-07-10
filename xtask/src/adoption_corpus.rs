use std::{
    fs,
    path::{Path, PathBuf},
};

use j2k_test_support::{fnv1a64_hex, read_pnm_image};

pub(crate) struct SourceImage {
    pub(crate) pixels: Vec<u8>,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) channels: u8,
}

pub(crate) fn collect_decode_fixture_paths(
    path_list: &str,
) -> Result<Vec<(PathBuf, PathBuf)>, String> {
    collect_paths(path_list, is_decode_fixture_path)
}

pub(crate) fn collect_encode_source_paths(
    path_list: &str,
) -> Result<Vec<(PathBuf, PathBuf)>, String> {
    collect_paths(path_list, is_encode_source_path)
}

pub(crate) fn manifest_row(fields: &[String]) -> Result<String, String> {
    for field in fields {
        validate_tsv_field(field)?;
    }
    Ok(format!("{}\n", fields.join("\t")))
}

pub(crate) fn canonical_label(path: &Path) -> Result<String, String> {
    path.canonicalize()
        .map_err(|err| format!("canonicalize {}: {err}", path.display()))
        .map(|path| path.display().to_string())
}

pub(crate) fn validate_tsv_field(value: &str) -> Result<(), String> {
    if value.is_empty() {
        return Err("manifest field cannot be empty".to_string());
    }
    if value.chars().any(|ch| ch == '\t' || ch.is_control()) {
        return Err(format!(
            "manifest field contains a control character: {}",
            value.escape_debug()
        ));
    }
    Ok(())
}

pub(crate) fn corpus_category(path: &Path, override_value: Option<&str>) -> String {
    override_value.map_or_else(
        || j2k_compare::common::infer_corpus_category(path).to_string(),
        ToString::to_string,
    )
}

pub(crate) fn corpus_name(root: &Path, override_value: Option<&str>) -> String {
    override_value.map_or_else(
        || {
            root.file_name()
                .and_then(|value| value.to_str())
                .filter(|value| !value.is_empty())
                .unwrap_or("external-corpus")
                .to_string()
        },
        ToString::to_string,
    )
}

pub(crate) fn codec_from_bytes(bytes: &[u8]) -> &'static str {
    let Ok(payload) = j2k::extract_j2k_codestream_payload(bytes) else {
        return "unknown";
    };
    match j2k_native::inspect_j2k_codestream_header(payload.codestream()) {
        Ok(header) if header.high_throughput => "htj2k",
        Ok(_) => "j2k",
        Err(_) => "unknown",
    }
}

pub(crate) fn container_from_path_and_bytes(path: &Path, bytes: &[u8]) -> &'static str {
    if let Some(extension) = path.extension().and_then(|value| value.to_str()) {
        match extension.to_ascii_lowercase().as_str() {
            "jph" => return "jph",
            "jhc" => return "jhc",
            _ => {}
        }
    }
    if bytes.starts_with(&[0, 0, 0, 12, b'j', b'P', b' ', b' ']) {
        "jp2"
    } else {
        "raw-codestream"
    }
}

pub(crate) fn source_image_pixel_hash(path: &Path) -> Result<String, String> {
    load_source_image(path).map(|image| fnv1a64_hex(&image.pixels))
}

pub(crate) fn load_source_image(path: &Path) -> Result<SourceImage, String> {
    if is_pnm_path(path) {
        let image = read_pnm_image(path)
            .map_err(|err| format!("read source PNM {}: {err}", path.display()))?;
        let channels = u8::try_from(image.channels).map_err(|_| {
            format!(
                "{} has unsupported channel count {}",
                path.display(),
                image.channels
            )
        })?;
        if !matches!(channels, 1 | 3) {
            return Err(format!(
                "{} has unsupported source channel count {channels}; expected 1 or 3",
                path.display()
            ));
        }
        return Ok(SourceImage {
            pixels: image.pixels,
            width: image.width,
            height: image.height,
            channels,
        });
    }
    let reader = image::ImageReader::open(path)
        .map_err(|err| format!("open source image {}: {err}", path.display()))?
        .with_guessed_format()
        .map_err(|err| format!("guess source image format {}: {err}", path.display()))?;
    let decoded = reader
        .decode()
        .map_err(|err| format!("decode source image {}: {err}", path.display()))?;
    match decoded.color() {
        image::ColorType::L8 => {
            let image = decoded.into_luma8();
            Ok(SourceImage {
                width: image.width(),
                height: image.height(),
                pixels: image.into_raw(),
                channels: 1,
            })
        }
        image::ColorType::Rgb8 => {
            let image = decoded.into_rgb8();
            Ok(SourceImage {
                width: image.width(),
                height: image.height(),
                pixels: image.into_raw(),
                channels: 3,
            })
        }
        color => Err(format!(
            "{} has unsupported source color type {color:?}; expected 8-bit grayscale or RGB without alpha",
            path.display()
        )),
    }
}

fn collect_paths(
    path_list: &str,
    predicate: fn(&Path) -> bool,
) -> Result<Vec<(PathBuf, PathBuf)>, String> {
    let mut paths = Vec::new();
    for root in std::env::split_paths(path_list) {
        if !root.is_dir() {
            return Err(format!(
                "manifest input is not a directory: {}",
                root.display()
            ));
        }
        collect_paths_from_root(&root, &root, predicate, &mut paths)?;
    }
    paths.sort();
    Ok(paths)
}

fn collect_paths_from_root(
    root: &Path,
    dir: &Path,
    predicate: fn(&Path) -> bool,
    paths: &mut Vec<(PathBuf, PathBuf)>,
) -> Result<(), String> {
    for entry in fs::read_dir(dir).map_err(|err| format!("read {}: {err}", dir.display()))? {
        let path = entry
            .map_err(|err| format!("read {} entry: {err}", dir.display()))?
            .path();
        if path.is_dir() {
            collect_paths_from_root(root, &path, predicate, paths)?;
        } else if predicate(&path) {
            paths.push((root.to_path_buf(), path));
        }
    }
    Ok(())
}

fn is_decode_fixture_path(path: &Path) -> bool {
    extension_matches(path, &["j2k", "j2c", "jp2", "jph", "jhc"])
}

fn is_encode_source_path(path: &Path) -> bool {
    extension_matches(
        path,
        &[
            "pgm", "ppm", "pnm", "png", "jpg", "jpeg", "tif", "tiff", "bmp",
        ],
    )
}

fn is_pnm_path(path: &Path) -> bool {
    extension_matches(path, &["pgm", "ppm", "pnm"])
}

fn extension_matches(path: &Path, extensions: &[&str]) -> bool {
    path.extension()
        .and_then(|value| value.to_str())
        .is_some_and(|extension| {
            let extension = extension.to_ascii_lowercase();
            extensions.contains(&extension.as_str())
        })
}

/// Lowercases `value` into `[a-z0-9_]`, falling back when nothing survives.
pub(crate) fn sanitize_id(value: &str, fallback: &str) -> String {
    let mut out = String::new();
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
        } else if !out.ends_with('_') {
            out.push('_');
        }
    }
    let trimmed = out.trim_matches('_');
    if trimmed.is_empty() {
        fallback.to_string()
    } else {
        trimmed.to_string()
    }
}
