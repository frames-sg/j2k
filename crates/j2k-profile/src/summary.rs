// SPDX-License-Identifier: MIT OR Apache-2.0

//! Bounded deterministic profile summaries.

mod output;
mod record;

use alloc::string::String;
use alloc::vec::Vec;

use crate::allocation::{element_bytes, try_string, try_vec, HeapBudget};
use crate::text::{include_input_bytes, validate_key};
use crate::{ProfileError, ProfileLimits, ProfileResult};

#[cfg(any(feature = "std", test))]
pub(crate) use record::record_timing_summary_str;

/// Maps an input profiling field to the label name used in summaries.
#[derive(Debug, Eq, PartialEq)]
pub struct SummaryLabel {
    input_key: String,
    summary_key: String,
}

impl SummaryLabel {
    /// Creates a label remapping using default limits.
    pub fn new(input_key: impl AsRef<str>, summary_key: impl AsRef<str>) -> ProfileResult<Self> {
        Self::new_with_limits(input_key, summary_key, ProfileLimits::default())
    }

    /// Creates a label remapping using caller-supplied limits.
    pub fn new_with_limits(
        input_key: impl AsRef<str>,
        summary_key: impl AsRef<str>,
        limits: ProfileLimits,
    ) -> ProfileResult<Self> {
        limits.validate()?;
        let input_key = input_key.as_ref();
        let summary_key = summary_key.as_ref();
        validate_key(input_key, limits)?;
        validate_key(summary_key, limits)?;
        let mut input_bytes = 0;
        include_input_bytes(&mut input_bytes, input_key.len(), limits)?;
        include_input_bytes(&mut input_bytes, summary_key.len(), limits)?;
        let mut budget = HeapBudget::new(0, limits.max_retained_bytes());
        Self::new_in_budget(input_key, summary_key, &mut budget)
    }

    /// Creates a same-key label using default limits.
    pub fn same(key: impl AsRef<str>) -> ProfileResult<Self> {
        Self::same_with_limits(key, ProfileLimits::default())
    }

    /// Creates a same-key label using caller-supplied limits.
    pub fn same_with_limits(key: impl AsRef<str>, limits: ProfileLimits) -> ProfileResult<Self> {
        let key = key.as_ref();
        Self::new_with_limits(key, key, limits)
    }

    fn new_in_budget(
        input_key: &str,
        summary_key: &str,
        budget: &mut HeapBudget,
    ) -> ProfileResult<Self> {
        Ok(Self {
            input_key: try_string(input_key, budget, "summary label input key")?,
            summary_key: try_string(summary_key, budget, "summary label output key")?,
        })
    }

    pub(crate) fn retained_bytes(&self) -> ProfileResult<usize> {
        self.input_key
            .capacity()
            .checked_add(self.summary_key.capacity())
            .ok_or(ProfileError::SizeOverflow {
                what: "summary label retained bytes",
            })
    }

    fn validate_for(&self, limits: ProfileLimits) -> ProfileResult<()> {
        validate_key(&self.input_key, limits)?;
        validate_key(&self.summary_key, limits)
    }
}

/// Creates same-key summary labels using default limits.
pub fn same_summary_labels(keys: &[&str]) -> ProfileResult<Vec<SummaryLabel>> {
    same_summary_labels_with_limits(keys, ProfileLimits::default())
}

/// Creates same-key summary labels using caller-supplied limits.
pub fn same_summary_labels_with_limits(
    keys: &[&str],
    limits: ProfileLimits,
) -> ProfileResult<Vec<SummaryLabel>> {
    limits.validate()?;
    crate::allocation::ensure_limit(keys.len(), limits.max_labels(), "summary label count")?;
    let mut input_bytes = 0;
    for (index, key) in keys.iter().enumerate() {
        validate_key(key, limits)?;
        if keys[..index].contains(key) {
            return Err(ProfileError::InvalidInput {
                what: "duplicate summary label key",
            });
        }
        include_input_bytes(&mut input_bytes, key.len(), limits)?;
        include_input_bytes(&mut input_bytes, key.len(), limits)?;
    }

    let mut budget = HeapBudget::new(0, limits.max_retained_bytes());
    let mut labels = try_vec(keys.len(), &mut budget, "summary label storage")?;
    for key in keys {
        labels.push(SummaryLabel::new_in_budget(key, key, &mut budget)?);
    }
    Ok(labels)
}

/// Aggregates profiling rows by codec, operation, path, and configured labels.
#[derive(Debug)]
pub struct ProfileSummary {
    limits: ProfileLimits,
    labels: Vec<SummaryLabel>,
    numeric_mode: SummaryNumericMode,
    rows: Vec<SummaryEntry>,
    retained_bytes: usize,
    #[cfg(feature = "std")]
    emit_on_drop: bool,
}

impl ProfileSummary {
    /// Creates an empty aggregate summary using default limits.
    pub fn new(labels: impl IntoIterator<Item = SummaryLabel>) -> ProfileResult<Self> {
        Self::new_with_limits(labels, ProfileLimits::default())
    }

    /// Creates an empty aggregate summary using caller-supplied limits.
    pub fn new_with_limits(
        labels: impl IntoIterator<Item = SummaryLabel>,
        limits: ProfileLimits,
    ) -> ProfileResult<Self> {
        Self::with_numeric_mode(labels, SummaryNumericMode::Aggregate, limits)
    }

    /// Creates an empty count-only summary using default limits.
    pub fn counts_only(labels: impl IntoIterator<Item = SummaryLabel>) -> ProfileResult<Self> {
        Self::counts_only_with_limits(labels, ProfileLimits::default())
    }

    /// Creates an empty count-only summary using caller-supplied limits.
    pub fn counts_only_with_limits(
        labels: impl IntoIterator<Item = SummaryLabel>,
        limits: ProfileLimits,
    ) -> ProfileResult<Self> {
        Self::with_numeric_mode(labels, SummaryNumericMode::CountOnly, limits)
    }

    fn with_numeric_mode(
        labels: impl IntoIterator<Item = SummaryLabel>,
        numeric_mode: SummaryNumericMode,
        limits: ProfileLimits,
    ) -> ProfileResult<Self> {
        limits.validate()?;
        let mut budget = HeapBudget::new(0, limits.max_retained_bytes());
        let mut owned_labels = Vec::new();
        let mut input_bytes = 0_usize;

        for label in labels {
            label.validate_for(limits)?;
            let label_count =
                owned_labels
                    .len()
                    .checked_add(1)
                    .ok_or(ProfileError::SizeOverflow {
                        what: "summary label count",
                    })?;
            crate::allocation::ensure_limit(
                label_count,
                limits.max_labels(),
                "summary label count",
            )?;
            include_input_bytes(&mut input_bytes, label.input_key.len(), limits)?;
            include_input_bytes(&mut input_bytes, label.summary_key.len(), limits)?;
            if owned_labels.iter().any(|existing: &SummaryLabel| {
                existing.input_key == label.input_key || existing.summary_key == label.summary_key
            }) {
                return Err(ProfileError::InvalidInput {
                    what: "duplicate summary label key",
                });
            }

            budget.include(label.retained_bytes()?, "summary labels")?;
            if owned_labels.len() == owned_labels.capacity() {
                let old_outer = element_bytes::<SummaryLabel>(
                    owned_labels.capacity(),
                    "summary label storage",
                )?;
                let mut replacement = try_vec(label_count, &mut budget, "summary label storage")?;
                replacement.append(&mut owned_labels);
                budget.release(old_outer, "summary label storage replacement")?;
                owned_labels = replacement;
            }
            owned_labels.push(label);
        }

        Ok(Self {
            limits,
            labels: owned_labels,
            numeric_mode,
            rows: Vec::new(),
            retained_bytes: budget.used(),
            #[cfg(feature = "std")]
            emit_on_drop: false,
        })
    }

    /// Returns this summary's configured limits.
    pub fn limits(&self) -> ProfileLimits {
        self.limits
    }

    /// Returns the number of distinct retained summary rows.
    pub fn row_count(&self) -> usize {
        self.rows.len()
    }

    /// Returns allocator-reported heap capacity retained by this summary.
    pub fn retained_capacity_bytes(&self) -> usize {
        self.retained_bytes
    }

    /// Emits accumulated rows to stderr when the summary is dropped.
    #[cfg(feature = "std")]
    #[must_use]
    pub fn emit_on_drop(mut self) -> Self {
        self.emit_on_drop = true;
        self
    }

    #[cfg(all(test, feature = "std"))]
    pub(crate) const fn emit_on_drop_enabled(&self) -> bool {
        self.emit_on_drop
    }

    fn is_label_key(&self, key: &str) -> bool {
        self.labels.iter().any(|label| label.input_key == key)
    }

    #[cfg(feature = "std")]
    pub(crate) fn empty_counts_only() -> Self {
        Self {
            limits: ProfileLimits::default(),
            labels: Vec::new(),
            numeric_mode: SummaryNumericMode::CountOnly,
            rows: Vec::new(),
            retained_bytes: 0,
            emit_on_drop: false,
        }
    }

    fn empty_retained_bytes(&self) -> ProfileResult<usize> {
        let mut retained =
            element_bytes::<SummaryLabel>(self.labels.capacity(), "summary label storage")?;
        for label in &self.labels {
            retained = retained.checked_add(label.retained_bytes()?).ok_or(
                ProfileError::SizeOverflow {
                    what: "empty summary retained bytes",
                },
            )?;
        }
        retained
            .checked_add(element_bytes::<SummaryEntry>(
                self.rows.capacity(),
                "summary row storage",
            )?)
            .ok_or(ProfileError::SizeOverflow {
                what: "empty summary retained bytes",
            })
    }
}

impl Default for ProfileSummary {
    fn default() -> Self {
        Self {
            limits: ProfileLimits::default(),
            labels: Vec::new(),
            numeric_mode: SummaryNumericMode::Aggregate,
            rows: Vec::new(),
            retained_bytes: 0,
            #[cfg(feature = "std")]
            emit_on_drop: false,
        }
    }
}

#[cfg(feature = "std")]
impl Drop for ProfileSummary {
    fn drop(&mut self) {
        if self.emit_on_drop {
            if let Err(error) = self.flush_to_stderr() {
                crate::emit_profile_error("summary_drop", &error);
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SummaryNumericMode {
    Aggregate,
    CountOnly,
}

#[derive(Debug)]
struct SummaryEntry {
    key: SummaryKey,
    row: SummaryRow,
}

#[derive(Debug)]
struct SummaryKey {
    codec: String,
    op: String,
    path: String,
    labels: Vec<SummaryLabelValue>,
}

#[derive(Debug)]
struct SummaryLabelValue {
    label_index: usize,
    value: String,
}

#[derive(Debug)]
struct SummaryRow {
    count: u128,
    numeric_sums: Vec<NumericSum>,
}

#[derive(Debug)]
struct NumericSum {
    key: String,
    sum: u128,
}

fn is_timing_field(field_key: &str) -> bool {
    field_key.ends_with("_us") || field_key.ends_with("_ms") || field_key.ends_with("_ns")
}
