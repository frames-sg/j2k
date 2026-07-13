// SPDX-License-Identifier: MIT OR Apache-2.0

use std::path::{Component, Path};

use serde_json::Value;

use super::{CompilerRegion, CompilerRegionReport, SourceSpan};
use crate::coverage::parsing::normalize_coverage_path;

const LLVM_COVERAGE_EXPORT_TYPE: &str = "llvm.coverage.json.export";
const CODE_REGION_KIND: u64 = 0;

pub(in crate::coverage) fn parse_compiler_regions(
    input: &str,
    root: &Path,
) -> Result<CompilerRegionReport, String> {
    let document: Value = serde_json::from_str(input)
        .map_err(|error| format!("failed to parse compiler coverage JSON: {error}"))?;
    let object = document
        .as_object()
        .ok_or_else(|| "compiler coverage JSON root must be an object".to_string())?;
    let report_type = required_string(object.get("type"), "compiler coverage JSON type")?;
    if report_type != LLVM_COVERAGE_EXPORT_TYPE {
        return Err(format!(
            "compiler coverage JSON type must be `{LLVM_COVERAGE_EXPORT_TYPE}`, found `{report_type}`"
        ));
    }
    required_string(object.get("version"), "compiler coverage JSON version")?;
    let data = required_array(object.get("data"), "compiler coverage JSON data")?;
    if data.is_empty() {
        return Err("compiler coverage JSON data must not be empty".to_string());
    }

    let mut report = CompilerRegionReport::default();
    for (unit_index, unit) in data.iter().enumerate() {
        let unit = unit.as_object().ok_or_else(|| {
            format!("compiler coverage JSON data[{unit_index}] must be an object")
        })?;
        let files = required_array(
            unit.get("files"),
            &format!("compiler coverage JSON data[{unit_index}].files"),
        )?;
        for (file_index, file) in files.iter().enumerate() {
            let file = file.as_object().ok_or_else(|| {
                format!(
                    "compiler coverage JSON data[{unit_index}].files[{file_index}] must be an object"
                )
            })?;
            let filename = required_string(
                file.get("filename"),
                &format!("compiler coverage JSON data[{unit_index}].files[{file_index}].filename"),
            )?;
            if let Some(path) = normalize_compiler_path(filename, root)? {
                report.files.insert(path);
            }
        }
        let functions = required_array(
            unit.get("functions"),
            &format!("compiler coverage JSON data[{unit_index}].functions"),
        )?;
        for (function_index, function) in functions.iter().enumerate() {
            parse_function_regions(function, unit_index, function_index, root, &mut report)?;
        }
    }
    Ok(report)
}

fn parse_function_regions(
    function: &Value,
    unit_index: usize,
    function_index: usize,
    root: &Path,
    report: &mut CompilerRegionReport,
) -> Result<(), String> {
    let context = format!("compiler coverage JSON data[{unit_index}].functions[{function_index}]");
    let function = function
        .as_object()
        .ok_or_else(|| format!("{context} must be an object"))?;
    let filenames = required_array(function.get("filenames"), &format!("{context}.filenames"))?
        .iter()
        .enumerate()
        .map(|(index, filename)| {
            let filename = filename
                .as_str()
                .filter(|filename| !filename.is_empty())
                .ok_or_else(|| {
                    format!("{context}.filenames[{index}] must be a non-empty string")
                })?;
            normalize_compiler_path(filename, root)
        })
        .collect::<Result<Vec<_>, _>>()?;
    if filenames.is_empty() {
        return Err(format!("{context}.filenames must not be empty"));
    }
    report.files.extend(filenames.iter().flatten().cloned());

    for (region_index, region) in
        required_array(function.get("regions"), &format!("{context}.regions"))?
            .iter()
            .enumerate()
    {
        let context = format!("{context}.regions[{region_index}]");
        let fields = region
            .as_array()
            .ok_or_else(|| format!("{context} must be an array"))?;
        if fields.len() != 8 {
            return Err(format!(
                "{context} must contain exactly 8 integer fields, found {}",
                fields.len()
            ));
        }
        let values = fields
            .iter()
            .enumerate()
            .map(|(field_index, field)| {
                field
                    .as_u64()
                    .ok_or_else(|| format!("{context}[{field_index}] must be an unsigned integer"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let file_index = usize::try_from(values[5])
            .map_err(|_| format!("{context} file index does not fit usize"))?;
        let path = filenames
            .get(file_index)
            .ok_or_else(|| format!("{context} references missing filename index {file_index}"))?;
        if values[7] != CODE_REGION_KIND {
            continue;
        }
        let span = SourceSpan::new(
            usize_field(values[0], &context, "start line")?,
            usize_field(values[1], &context, "start column")?,
            usize_field(values[2], &context, "end line")?,
            usize_field(values[3], &context, "end column")?,
        )?;
        let Some(path) = path else {
            continue;
        };
        report
            .regions
            .entry(path.clone())
            .or_default()
            .push(CompilerRegion {
                span,
                count: values[4],
            });
    }
    Ok(())
}

fn normalize_compiler_path(path: &str, root: &Path) -> Result<Option<String>, String> {
    let path = Path::new(path);
    if path.is_absolute() && !path.starts_with(root) {
        return Ok(None);
    }
    if !path.is_absolute()
        && path.components().any(|component| {
            matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        })
    {
        return Err(format!(
            "compiler coverage source `{}` escapes the repository root",
            path.display()
        ));
    }
    normalize_coverage_path(
        path.to_str()
            .ok_or_else(|| "compiler coverage source path is not valid UTF-8".to_string())?,
        root,
    )
    .map(Some)
}

fn usize_field(value: u64, context: &str, field: &str) -> Result<usize, String> {
    usize::try_from(value).map_err(|_| format!("{context} {field} does not fit usize"))
}

fn required_array<'a>(value: Option<&'a Value>, context: &str) -> Result<&'a Vec<Value>, String> {
    value
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{context} must be an array"))
}

fn required_string<'a>(value: Option<&'a Value>, context: &str) -> Result<&'a str, String> {
    value
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| format!("{context} must be a non-empty string"))
}
