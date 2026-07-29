// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;
use std::fs;
use std::sync::OnceLock;

use serde_yaml_ng::{Mapping, Value};

use super::super::repo_root;

#[derive(Debug)]
pub(super) struct Workflow {
    pub(super) file_name: String,
    pub(super) document: Value,
}

pub(super) fn workflows() -> &'static [Workflow] {
    static WORKFLOWS: OnceLock<Vec<Workflow>> = OnceLock::new();
    WORKFLOWS.get_or_init(|| {
        let directory = repo_root().join(".github/workflows");
        let mut paths = fs::read_dir(&directory)
            .expect("read workflow directory")
            .map(|entry| entry.expect("read workflow entry").path())
            .filter(|path| {
                matches!(
                    path.extension().and_then(|extension| extension.to_str()),
                    Some("yml" | "yaml")
                )
            })
            .collect::<Vec<_>>();
        paths.sort();
        assert!(!paths.is_empty(), "no GitHub Actions workflows found");

        paths
            .into_iter()
            .map(|path| {
                let source = fs::read_to_string(&path)
                    .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
                let document = serde_yaml_ng::from_str(&source)
                    .unwrap_or_else(|error| panic!("parse {}: {error}", path.display()));
                let file_name = path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or_else(|| {
                        panic!("workflow path has no UTF-8 name: {}", path.display())
                    })
                    .to_owned();
                Workflow {
                    file_name,
                    document,
                }
            })
            .collect()
    })
}

pub(super) fn workflow(file_name: &str) -> &'static Workflow {
    workflows()
        .iter()
        .find(|workflow| workflow.file_name == file_name)
        .unwrap_or_else(|| panic!("missing workflow {file_name}"))
}

pub(super) fn value_mapping<'a>(value: &'a Value, label: &str) -> &'a Mapping {
    value
        .as_mapping()
        .unwrap_or_else(|| panic!("{label} must contain a YAML mapping"))
}

pub(super) fn jobs<'a>(root: &'a Mapping, label: &str) -> &'a Mapping {
    mapping_get(root, "jobs")
        .and_then(Value::as_mapping)
        .unwrap_or_else(|| panic!("{label} must contain a jobs mapping"))
}

pub(super) fn mapping_get<'a>(mapping: &'a Mapping, key: &str) -> Option<&'a Value> {
    mapping.get(Value::String(key.to_owned())).or_else(|| {
        (key == "on")
            .then(|| mapping.get(Value::Bool(true)))
            .flatten()
    })
}

pub(super) fn visit_mappings(value: &Value, visitor: &mut impl FnMut(&Mapping)) {
    match value {
        Value::Mapping(mapping) => {
            visitor(mapping);
            for (key, value) in mapping {
                visit_mappings(key, visitor);
                visit_mappings(value, visitor);
            }
        }
        Value::Sequence(values) => {
            for value in values {
                visit_mappings(value, visitor);
            }
        }
        _ => {}
    }
}

pub(super) fn string_set(value: &Value, label: &str) -> BTreeSet<String> {
    match value {
        Value::String(value) => [value.clone()].into_iter().collect(),
        Value::Sequence(values) => values
            .iter()
            .map(|value| {
                value
                    .as_str()
                    .unwrap_or_else(|| panic!("{label} entries must be strings"))
                    .to_owned()
            })
            .collect(),
        _ => panic!("{label} must be a string or sequence"),
    }
}

pub(super) fn is_numeric_zero(value: &Value) -> bool {
    value.as_u64() == Some(0) || value.as_i64() == Some(0) || value.as_str() == Some("0")
}

pub(super) fn display_key(value: &Value) -> String {
    value
        .as_str()
        .map_or_else(|| format!("{value:?}"), str::to_owned)
}
