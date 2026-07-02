// SPDX-License-Identifier: MIT OR Apache-2.0

use std::path::{Path, PathBuf};

/// Resolves a manifest row path, with suffix-based relocation fallback.
///
/// # Errors
///
/// Returns an error when the path cannot be canonicalized, when suffix relocation
/// is ambiguous, or when no matching relocated path exists under the supplied roots.
pub fn canonicalize_manifest_row_path(
    raw_path: &str,
    base: &Path,
    relocation_roots: &[PathBuf],
    manifest_label: &str,
    manifest_path: &Path,
    row_number: usize,
) -> Result<PathBuf, String> {
    let raw = Path::new(raw_path);
    let resolved_path = if raw.is_absolute() {
        raw.to_path_buf()
    } else {
        base.join(raw)
    };
    match resolved_path.canonicalize() {
        Ok(path) => Ok(path),
        Err(primary_error) => {
            let candidates = manifest_relocation_candidates(raw, relocation_roots);
            if candidates.len() == 1 {
                Ok(candidates[0].clone())
            } else if !candidates.is_empty() {
                Err(format!(
                    "{manifest_label} {} row {row_number} path {} is ambiguous after suffix remap: {}",
                    manifest_path.display(),
                    raw_path,
                    join_path_labels(&candidates)
                ))
            } else {
                Err(format!(
                    "{manifest_label} {} row {row_number} path {} cannot be canonicalized: {primary_error}; no suffix remap found under {}",
                    manifest_path.display(),
                    resolved_path.display(),
                    join_path_labels(relocation_roots)
                ))
            }
        }
    }
}

/// Returns the index for a required manifest column.
///
/// # Errors
///
/// Returns an error when the named column is absent.
pub fn manifest_column(
    headers: &[&str],
    manifest_label: &str,
    name: &str,
) -> Result<usize, String> {
    optional_manifest_column(headers, name)
        .ok_or_else(|| format!("{manifest_label} is missing required {name:?} column"))
}

pub fn optional_manifest_column(headers: &[&str], name: &str) -> Option<usize> {
    headers.iter().position(|header| *header == name)
}

/// Returns a manifest field by index.
///
/// # Errors
///
/// Returns an error when the field index is outside the row.
pub fn manifest_field<'a>(
    fields: &'a [&str],
    manifest_label: &str,
    index: usize,
    name: &str,
    row_number: usize,
) -> Result<&'a str, String> {
    fields
        .get(index)
        .copied()
        .ok_or_else(|| format!("{manifest_label} row {row_number} is missing {name:?} field"))
}

/// Returns a trimmed optional manifest field value.
///
/// # Errors
///
/// Returns an error when the field index is outside the row or the value contains a control
/// character.
pub fn manifest_optional_value(
    fields: &[&str],
    manifest_label: &str,
    index: Option<usize>,
    name: &str,
    row_number: usize,
) -> Result<Option<String>, String> {
    let Some(index) = index else {
        return Ok(None);
    };
    let value = manifest_field(fields, manifest_label, index, name, row_number)?.trim();
    if value.is_empty() {
        return Ok(None);
    }
    if value.chars().any(char::is_control) {
        return Err(format!(
            "{manifest_label} row {row_number} field {name:?} contains a control character"
        ));
    }
    Ok(Some(value.to_string()))
}

fn manifest_relocation_candidates(raw_path: &Path, relocation_roots: &[PathBuf]) -> Vec<PathBuf> {
    let suffixes = normal_path_suffixes(raw_path);
    let mut candidates = Vec::new();
    for root in relocation_roots {
        for suffix in &suffixes {
            let candidate = root.join(suffix);
            let Ok(canonical) = candidate.canonicalize() else {
                continue;
            };
            if !candidates.contains(&canonical) {
                candidates.push(canonical);
            }
        }
    }
    candidates
}

fn normal_path_suffixes(path: &Path) -> Vec<PathBuf> {
    let parts = path
        .components()
        .filter_map(|component| match component {
            std::path::Component::Normal(part) => Some(part.to_owned()),
            _ => None,
        })
        .collect::<Vec<_>>();
    let mut suffixes = Vec::new();
    for start in 0..parts.len() {
        let mut suffix = PathBuf::new();
        for part in &parts[start..] {
            suffix.push(part);
        }
        suffixes.push(suffix);
    }
    suffixes
}

fn join_path_labels(paths: &[PathBuf]) -> String {
    if paths.is_empty() {
        return "none".to_string();
    }
    paths
        .iter()
        .map(|path| path.display().to_string())
        .collect::<Vec<_>>()
        .join(",")
}

#[cfg(test)]
mod tests {
    use super::{manifest_column, manifest_field, manifest_optional_value};

    #[test]
    fn manifest_helpers_preserve_labelled_diagnostics() {
        let headers = ["path", "input_fnv1a64"];
        assert_eq!(
            manifest_column(&headers, "CUDA encode manifest", "path"),
            Ok(0)
        );
        assert_eq!(
            manifest_column(&headers, "CUDA encode manifest", "codec"),
            Err("CUDA encode manifest is missing required \"codec\" column".to_string())
        );

        let fields = ["fixture.j2k"];
        assert_eq!(
            manifest_field(&fields, "CUDA encode manifest", 1, "input_fnv1a64", 4),
            Err("CUDA encode manifest row 4 is missing \"input_fnv1a64\" field".to_string())
        );
    }

    #[test]
    fn manifest_optional_value_trims_empty_and_rejects_control_characters() {
        let fields = ["fixture.j2k", "  htj2k  ", "   ", "bad\nvalue"];
        assert_eq!(
            manifest_optional_value(&fields, "Metal decode manifest", Some(1), "codec", 7),
            Ok(Some("htj2k".to_string()))
        );
        assert_eq!(
            manifest_optional_value(&fields, "Metal decode manifest", Some(2), "container", 7),
            Ok(None)
        );
        assert_eq!(
            manifest_optional_value(&fields, "Metal decode manifest", Some(3), "container", 7),
            Err(
                "Metal decode manifest row 7 field \"container\" contains a control character"
                    .to_string()
            )
        );
        assert_eq!(
            manifest_optional_value(&fields, "Metal decode manifest", None, "codec", 7),
            Ok(None)
        );
    }
}
