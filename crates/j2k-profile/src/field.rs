use alloc::borrow::ToOwned;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum ProfileFieldKind {
    /// Field is intended to identify/group rows.
    Label,
    /// Field is a metric and may be aggregated by summaries.
    Metric {
        /// Whether summaries should aggregate this metric.
        summarize: bool,
    },
}

/// A typed profiling field that still formats as `key=value` for compatibility.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProfileField {
    key: String,
    value: String,
    kind: ProfileFieldKind,
}

impl ProfileField {
    /// Creates a label field.
    pub fn label(key: impl AsRef<str>, value: impl fmt::Display) -> Self {
        Self::new(key, value, ProfileFieldKind::Label)
    }

    /// Creates a summarizable metric field.
    pub fn metric(key: impl AsRef<str>, value: impl fmt::Display) -> Self {
        Self::new(key, value, ProfileFieldKind::Metric { summarize: true })
    }

    /// Creates a metric field with explicit summary behavior.
    pub fn metric_with_summary(
        key: impl AsRef<str>,
        value: impl fmt::Display,
        summarize: bool,
    ) -> Self {
        Self::new(key, value, ProfileFieldKind::Metric { summarize })
    }

    fn new(key: impl AsRef<str>, value: impl fmt::Display, kind: ProfileFieldKind) -> Self {
        Self {
            key: key.as_ref().to_owned(),
            value: value.to_string(),
            kind,
        }
    }

    /// Field key.
    pub fn key(&self) -> &str {
        &self.key
    }

    /// Field value.
    pub fn value(&self) -> &str {
        &self.value
    }

    pub(crate) fn summarize_metric(&self) -> bool {
        matches!(self.kind, ProfileFieldKind::Metric { summarize: true })
    }
}

pub(crate) fn field_pairs(fields: &[ProfileField]) -> Vec<(&str, &str)> {
    fields
        .iter()
        .map(|field| (field.key(), field.value()))
        .collect()
}
