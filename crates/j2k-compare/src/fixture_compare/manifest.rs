// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

use super::{
    canonicalize_manifest_row_path, common, external_input_dirs, external_source_label,
    fnv1a64_hex, optional_manifest_column, Codec, Container, FixtureManifest, FixtureMetadata,
    ManifestEntry,
};

pub(super) fn fixture_manifest_from_env() -> Result<Option<FixtureManifest>, String> {
    let Some(path) = std::env::var_os("J2K_FIXTURE_COMPARE_MANIFEST").map(PathBuf::from) else {
        return Ok(None);
    };
    let text = std::fs::read_to_string(&path).map_err(|error| {
        format!(
            "read J2K_FIXTURE_COMPARE_MANIFEST {}: {error}",
            path.display()
        )
    })?;
    let base = path.parent().unwrap_or_else(|| Path::new("."));
    let relocation_roots = external_input_dirs();
    let mut lines = text.lines().filter(|line| !line.trim().is_empty());
    let header = lines
        .next()
        .ok_or_else(|| format!("fixture manifest {} is empty", path.display()))?;
    let headers = header.split('\t').collect::<Vec<_>>();
    let path_index = manifest_column(&headers, "path")?;
    let category_index = manifest_column(&headers, "corpus_category")?;
    let corpus_name_index = optional_manifest_column(&headers, "corpus_name");
    let license_status_index = optional_manifest_column(&headers, "license_status");
    let encode_command_index = optional_manifest_column(&headers, "encode_command");
    let hash_index = optional_manifest_column(&headers, "input_fnv1a64");
    let source_hash_index = optional_manifest_column(&headers, "source_fnv1a64");
    let codec_index = optional_manifest_column(&headers, "codec");
    let container_index = optional_manifest_column(&headers, "container");

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
            "fixture manifest",
            &path,
            row_number,
        )?;
        let corpus_category =
            manifest_required_value(&fields, category_index, "corpus_category", row_number)?;
        let entry = ManifestEntry {
            corpus_category,
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
            encode_command: manifest_optional_value(
                &fields,
                encode_command_index,
                "encode_command",
                row_number,
            )?
            .unwrap_or_else(|| "not-recorded".to_string()),
            input_fnv1a64: manifest_optional_value(
                &fields,
                hash_index,
                "input_fnv1a64",
                row_number,
            )?,
            source_fnv1a64: manifest_optional_value(
                &fields,
                source_hash_index,
                "source_fnv1a64",
                row_number,
            )?,
            codec: parse_manifest_codec(
                manifest_optional_value(&fields, codec_index, "codec", row_number)?.as_deref(),
                row_number,
            )?,
            container: parse_manifest_container(
                manifest_optional_value(&fields, container_index, "container", row_number)?
                    .as_deref(),
                row_number,
            )?,
        };
        if entries.insert(canonical_path, entry).is_some() {
            return Err(format!(
                "fixture manifest {} row {row_number} duplicates path {raw_path}",
                path.display()
            ));
        }
    }

    Ok(Some(FixtureManifest { entries }))
}

pub(super) fn external_fixture_metadata(
    path: &Path,
    bytes: &[u8],
    codec: Codec,
    container: Container,
    manifest: Option<&FixtureManifest>,
) -> Result<FixtureMetadata, String> {
    let input_source = external_source_label(path)?;
    let Some(manifest) = manifest else {
        return Ok(FixtureMetadata {
            input_source,
            corpus_category: external_corpus_category(path),
            corpus_name: "path-inferred".to_string(),
            license_status: "not-recorded".to_string(),
            encode_command: "not-recorded".to_string(),
            manifest_status: "not-covered".to_string(),
            source_fnv1a64: None,
        });
    };
    let canonical_path = path
        .canonicalize()
        .map_err(|error| format!("canonicalize external fixture {}: {error}", path.display()))?;
    let Some(entry) = manifest.entries.get(&canonical_path) else {
        return Ok(FixtureMetadata {
            input_source,
            corpus_category: external_corpus_category(path),
            corpus_name: "path-inferred".to_string(),
            license_status: "not-recorded".to_string(),
            encode_command: "not-recorded".to_string(),
            manifest_status: "not-covered".to_string(),
            source_fnv1a64: None,
        });
    };

    if let Some(expected_hash) = &entry.input_fnv1a64 {
        let actual_hash = fnv1a64_hex(bytes);
        if actual_hash != *expected_hash {
            return Err(format!(
                "external fixture {} hash mismatch: manifest {expected_hash} != actual {actual_hash}",
                path.display()
            ));
        }
    }
    if let Some(expected_codec) = entry.codec {
        if codec != expected_codec {
            return Err(format!(
                "external fixture {} codec mismatch: manifest {} != detected {}",
                path.display(),
                expected_codec.label(),
                codec.label()
            ));
        }
    }
    if let Some(expected_container) = entry.container {
        if container != expected_container {
            return Err(format!(
                "external fixture {} container mismatch: manifest {} != detected {}",
                path.display(),
                expected_container.label(),
                container.label()
            ));
        }
    }

    let manifest_status =
        if entry.input_fnv1a64.is_some() && entry.codec.is_some() && entry.container.is_some() {
            "covered"
        } else {
            "covered-unpinned"
        };

    Ok(FixtureMetadata {
        input_source,
        corpus_category: entry.corpus_category.clone(),
        corpus_name: entry.corpus_name.clone(),
        license_status: entry.license_status.clone(),
        encode_command: entry.encode_command.clone(),
        manifest_status: manifest_status.to_string(),
        source_fnv1a64: entry.source_fnv1a64.clone(),
    })
}

pub(super) fn manifest_column(headers: &[&str], name: &str) -> Result<usize, String> {
    common::manifest_column(headers, name, "fixture")
}

pub(super) fn manifest_field<'a>(
    fields: &'a [&str],
    index: usize,
    name: &str,
    row_number: usize,
) -> Result<&'a str, String> {
    common::manifest_field(fields, index, name, row_number, "fixture")
}

pub(super) fn manifest_required_value(
    fields: &[&str],
    index: usize,
    name: &str,
    row_number: usize,
) -> Result<String, String> {
    common::manifest_required_value(fields, index, name, row_number, "fixture")
}

pub(super) fn manifest_optional_value(
    fields: &[&str],
    index: Option<usize>,
    name: &str,
    row_number: usize,
) -> Result<Option<String>, String> {
    common::manifest_optional_value(fields, index, name, row_number, "fixture")
}

pub(super) fn parse_manifest_codec(
    value: Option<&str>,
    row_number: usize,
) -> Result<Option<Codec>, String> {
    match value {
        None => Ok(None),
        Some("j2k" | "classic") => Ok(Some(Codec::Classic)),
        Some("htj2k") => Ok(Some(Codec::Htj2k)),
        Some("unknown") => Ok(Some(Codec::Unknown)),
        Some(other) => Err(format!(
            "fixture manifest row {row_number} has invalid codec {other:?}; expected j2k, classic, htj2k, or unknown"
        )),
    }
}

pub(super) fn parse_manifest_container(
    value: Option<&str>,
    row_number: usize,
) -> Result<Option<Container>, String> {
    match value {
        None => Ok(None),
        Some("raw-codestream" | "j2k" | "j2c") => Ok(Some(Container::RawCodestream)),
        Some("jp2") => Ok(Some(Container::Jp2)),
        Some("jph") => Ok(Some(Container::Jph)),
        Some("jhc") => Ok(Some(Container::Jhc)),
        Some(other) => Err(format!(
            "fixture manifest row {row_number} has invalid container {other:?}; expected raw-codestream, j2k, j2c, jp2, jph, or jhc"
        )),
    }
}
pub(super) fn external_corpus_category(path: &Path) -> String {
    crate::common::infer_corpus_category(path).to_string()
}
