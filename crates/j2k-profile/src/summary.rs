use alloc::borrow::ToOwned;
use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt::Write as _;

use crate::field::{self, ProfileField};
use crate::format::format_profile_summary_prefix;

/// Maps an input profiling field to the label name used in summaries.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SummaryLabel {
    input_key: String,
    summary_key: String,
}

impl SummaryLabel {
    /// Creates a label remapping from an input field key to a summary key.
    pub fn new(input_key: impl AsRef<str>, summary_key: impl AsRef<str>) -> Self {
        Self {
            input_key: input_key.as_ref().to_owned(),
            summary_key: summary_key.as_ref().to_owned(),
        }
    }

    /// Creates a label whose input and summary keys are the same.
    pub fn same(key: impl AsRef<str>) -> Self {
        Self::new(key.as_ref(), key.as_ref())
    }
}

/// Creates same-key summary labels from field keys.
pub fn same_summary_labels(keys: &[&str]) -> Vec<SummaryLabel> {
    keys.iter().map(SummaryLabel::same).collect()
}

/// Aggregates profiling rows by codec, operation, path, and configured labels.
#[derive(Debug)]
pub struct ProfileSummary {
    labels: Vec<SummaryLabel>,
    numeric_mode: SummaryNumericMode,
    rows: BTreeMap<SummaryKey, SummaryRow>,
    #[cfg(feature = "std")]
    emit_on_drop: bool,
}

impl Clone for ProfileSummary {
    fn clone(&self) -> Self {
        Self {
            labels: self.labels.clone(),
            numeric_mode: self.numeric_mode,
            rows: self.rows.clone(),
            #[cfg(feature = "std")]
            emit_on_drop: false,
        }
    }
}

impl ProfileSummary {
    /// Creates an empty profile summary with the given summary labels.
    pub fn new(labels: impl IntoIterator<Item = SummaryLabel>) -> Self {
        Self::with_numeric_mode(labels, SummaryNumericMode::Aggregate)
    }

    /// Creates an empty profile summary that counts rows without aggregating numeric fields.
    pub fn counts_only(labels: impl IntoIterator<Item = SummaryLabel>) -> Self {
        Self::with_numeric_mode(labels, SummaryNumericMode::CountOnly)
    }

    fn with_numeric_mode(
        labels: impl IntoIterator<Item = SummaryLabel>,
        numeric_mode: SummaryNumericMode,
    ) -> Self {
        Self {
            labels: labels.into_iter().collect(),
            numeric_mode,
            rows: BTreeMap::new(),
            #[cfg(feature = "std")]
            emit_on_drop: false,
        }
    }

    /// Emits accumulated rows to stderr when the summary is dropped.
    #[cfg(feature = "std")]
    #[must_use]
    pub fn emit_on_drop(mut self) -> Self {
        self.emit_on_drop = true;
        self
    }

    /// Flushes accumulated rows to stderr and clears them.
    #[cfg(feature = "std")]
    pub fn flush_to_stderr(&mut self) {
        for row in self.take_formatted_rows() {
            std::eprintln!("{row}");
        }
    }

    #[cfg(all(test, feature = "std"))]
    pub(crate) const fn emit_on_drop_enabled(&self) -> bool {
        self.emit_on_drop
    }

    /// Records a profiling row with string field values.
    pub fn record_str<K, V>(
        &mut self,
        codec: impl AsRef<str>,
        op: impl AsRef<str>,
        path: impl AsRef<str>,
        fields: &[(K, V)],
    ) where
        K: AsRef<str>,
        V: AsRef<str>,
    {
        let key = self.summary_key_from_str(codec.as_ref(), op.as_ref(), path.as_ref(), fields);
        let mut numeric_fields = Vec::new();
        if self.numeric_mode == SummaryNumericMode::Aggregate {
            for (field_key, field_value) in fields {
                let field_key = field_key.as_ref();
                if self.is_label_key(field_key) {
                    continue;
                }
                if let Ok(value) = field_value.as_ref().parse::<u128>() {
                    numeric_fields.push((field_key.to_owned(), value));
                }
            }
        }

        let row = self.rows.entry(key).or_default();
        row.count = row.count.saturating_add(1);
        for (field_key, value) in numeric_fields {
            row.record_numeric(&field_key, value);
        }
    }

    /// Records a profiling row with unsigned integer field values.
    pub fn record_u128<K>(
        &mut self,
        codec: impl AsRef<str>,
        op: impl AsRef<str>,
        path: impl AsRef<str>,
        fields: &[(K, u128)],
    ) where
        K: AsRef<str>,
    {
        let key = self.summary_key_from_u128(codec.as_ref(), op.as_ref(), path.as_ref(), fields);
        let mut numeric_fields = Vec::new();
        if self.numeric_mode == SummaryNumericMode::Aggregate {
            for (field_key, value) in fields {
                let field_key = field_key.as_ref();
                if self.is_label_key(field_key) {
                    continue;
                }
                numeric_fields.push((field_key.to_owned(), *value));
            }
        }

        let row = self.rows.entry(key).or_default();
        row.count = row.count.saturating_add(1);
        for (field_key, value) in numeric_fields {
            row.record_numeric(&field_key, value);
        }
    }

    /// Records a profiling row with typed fields.
    pub fn record_fields(
        &mut self,
        codec: impl AsRef<str>,
        op: impl AsRef<str>,
        path: impl AsRef<str>,
        fields: &[ProfileField],
    ) {
        let pairs = field::field_pairs(fields);
        let key = self.summary_key_from_str(codec.as_ref(), op.as_ref(), path.as_ref(), &pairs);
        let mut numeric_fields = Vec::new();
        if self.numeric_mode == SummaryNumericMode::Aggregate {
            for field in fields {
                if self.is_label_key(field.key()) || !field.summarize_metric() {
                    continue;
                }
                if let Ok(value) = field.value().parse::<u128>() {
                    numeric_fields.push((field.key().to_owned(), value));
                }
            }
        }

        let row = self.rows.entry(key).or_default();
        row.count = row.count.saturating_add(1);
        for (field_key, value) in numeric_fields {
            row.record_numeric(&field_key, value);
        }
    }

    /// Formats deterministic summary rows.
    pub fn format_rows(&self) -> Vec<String> {
        self.rows
            .iter()
            .map(|(key, row)| row.format_with_key(key))
            .collect()
    }

    /// Formats deterministic summary rows and clears accumulated state.
    pub fn take_formatted_rows(&mut self) -> Vec<String> {
        let rows = self.format_rows();
        self.rows.clear();
        rows
    }

    fn summary_key_from_str<K, V>(
        &self,
        codec: &str,
        op: &str,
        path: &str,
        fields: &[(K, V)],
    ) -> SummaryKey
    where
        K: AsRef<str>,
        V: AsRef<str>,
    {
        let labels = self
            .labels
            .iter()
            .filter_map(|label| {
                find_str_field(fields, &label.input_key)
                    .map(|field_value| (label.summary_key.clone(), field_value))
            })
            .collect();

        SummaryKey::new(codec, op, path, labels)
    }

    fn summary_key_from_u128<K>(
        &self,
        codec: &str,
        op: &str,
        path: &str,
        fields: &[(K, u128)],
    ) -> SummaryKey
    where
        K: AsRef<str>,
    {
        let labels = self
            .labels
            .iter()
            .filter_map(|label| {
                find_u128_field(fields, &label.input_key)
                    .map(|field_value| (label.summary_key.clone(), field_value))
            })
            .collect();

        SummaryKey::new(codec, op, path, labels)
    }

    fn is_label_key(&self, key: &str) -> bool {
        self.labels.iter().any(|label| label.input_key == key)
    }
}

/// Records a string-valued summary row using only configured labels and timing fields.
#[cfg(any(feature = "std", test))]
pub(crate) fn record_timing_summary_str(
    summary: &mut ProfileSummary,
    codec: &str,
    op: &str,
    path: &str,
    fields: &[(&str, &str)],
    summary_label_keys: &[&str],
) {
    let summary_fields = fields
        .iter()
        .copied()
        .filter(|(field, _)| summary_label_keys.contains(field) || is_timing_field(field))
        .collect::<Vec<_>>();
    summary.record_str(codec, op, path, &summary_fields);
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SummaryNumericMode {
    Aggregate,
    CountOnly,
}

impl Default for ProfileSummary {
    fn default() -> Self {
        Self::new([])
    }
}

#[cfg(feature = "std")]
impl Drop for ProfileSummary {
    fn drop(&mut self) {
        if self.emit_on_drop {
            self.flush_to_stderr();
        }
    }
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct SummaryKey {
    codec: String,
    op: String,
    path: String,
    labels: Vec<(String, String)>,
}

impl SummaryKey {
    fn new(codec: &str, op: &str, path: &str, labels: Vec<(String, String)>) -> Self {
        Self {
            codec: codec.to_owned(),
            op: op.to_owned(),
            path: path.to_owned(),
            labels,
        }
    }
}

#[derive(Clone, Debug, Default)]
struct SummaryRow {
    count: u128,
    numeric_sums: BTreeMap<String, u128>,
}

impl SummaryRow {
    fn record_numeric(&mut self, key: &str, value: u128) {
        self.numeric_sums
            .entry(key.to_owned())
            .and_modify(|sum| *sum = sum.saturating_add(value))
            .or_insert(value);
    }

    fn format_with_key(&self, key: &SummaryKey) -> String {
        let mut row = format_profile_summary_prefix(&key.codec, &key.op, &key.path);

        for (label_key, label_value) in &key.labels {
            write!(row, " {label_key}={label_value}").expect("writing to String failed");
        }
        write!(row, " count={}", self.count).expect("writing to String failed");

        for (field_key, sum) in &self.numeric_sums {
            write!(row, " {field_key}_sum={sum}").expect("writing to String failed");
            if is_timing_field(field_key) {
                let average = sum / self.count;
                write!(row, " {field_key}_avg={average}").expect("writing to String failed");
            }
        }

        row
    }
}

fn find_str_field<K, V>(fields: &[(K, V)], key: &str) -> Option<String>
where
    K: AsRef<str>,
    V: AsRef<str>,
{
    fields
        .iter()
        .find(|(field_key, _)| field_key.as_ref() == key)
        .map(|(_, field_value)| field_value.as_ref().to_owned())
}

fn find_u128_field<K>(fields: &[(K, u128)], key: &str) -> Option<String>
where
    K: AsRef<str>,
{
    fields
        .iter()
        .find(|(field_key, _)| field_key.as_ref() == key)
        .map(|(_, field_value)| {
            let mut value = String::new();
            write!(value, "{field_value}").expect("writing to String failed");
            value
        })
}

fn is_timing_field(field_key: &str) -> bool {
    field_key.ends_with("_us") || field_key.ends_with("_ms") || field_key.ends_with("_ns")
}
