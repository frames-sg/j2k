// SPDX-License-Identifier: MIT OR Apache-2.0

//! Transactional profile-summary recording.

use alloc::string::String;
use core::cmp::Ordering;

use super::{
    is_timing_field, NumericSum, ProfileSummary, SummaryEntry, SummaryKey, SummaryLabelValue,
    SummaryNumericMode, SummaryRow,
};
use crate::allocation::{
    element_bytes, ensure_limit, try_string, try_string_capacity, try_vec, HeapBudget,
};
use crate::text::{
    compare_text_to_u128, decimal_len, include_input_bytes, push_u128, validate_identity,
    validate_key, validate_value,
};
use crate::{ProfileError, ProfileField, ProfileResult};

impl ProfileSummary {
    /// Records a profiling row with string field values.
    pub fn record_str<K, V>(
        &mut self,
        codec: impl AsRef<str>,
        op: impl AsRef<str>,
        path: impl AsRef<str>,
        fields: &[(K, V)],
    ) -> ProfileResult<()>
    where
        K: AsRef<str>,
        V: AsRef<str>,
    {
        self.record(
            codec.as_ref(),
            op.as_ref(),
            path.as_ref(),
            &StringFields {
                fields,
                timing_only: false,
            },
        )
    }

    /// Records a profiling row with unsigned integer field values.
    pub fn record_u128<K>(
        &mut self,
        codec: impl AsRef<str>,
        op: impl AsRef<str>,
        path: impl AsRef<str>,
        fields: &[(K, u128)],
    ) -> ProfileResult<()>
    where
        K: AsRef<str>,
    {
        self.record(
            codec.as_ref(),
            op.as_ref(),
            path.as_ref(),
            &IntegerFields { fields },
        )
    }

    /// Records a profiling row with typed fields.
    pub fn record_fields(
        &mut self,
        codec: impl AsRef<str>,
        op: impl AsRef<str>,
        path: impl AsRef<str>,
        fields: &[ProfileField],
    ) -> ProfileResult<()> {
        self.record(
            codec.as_ref(),
            op.as_ref(),
            path.as_ref(),
            &TypedFields { fields },
        )
    }

    fn record<F: RecordFields>(
        &mut self,
        codec: &str,
        op: &str,
        path: &str,
        fields: &F,
    ) -> ProfileResult<()> {
        self.validate_record(codec, op, path, fields)?;
        match self.rows.binary_search_by(|entry| {
            self.compare_key_to_fields(&entry.key, codec, op, path, fields)
        }) {
            Ok(index) => self.record_existing(index, fields),
            Err(index) => self.insert_new(index, codec, op, path, fields),
        }
    }

    fn validate_record<F: RecordFields>(
        &self,
        codec: &str,
        op: &str,
        path: &str,
        fields: &F,
    ) -> ProfileResult<()> {
        self.limits.validate()?;
        ensure_limit(fields.len(), self.limits.max_fields(), "field count")?;
        validate_identity(codec, self.limits)?;
        validate_identity(op, self.limits)?;
        validate_identity(path, self.limits)?;
        let mut input_bytes = 0;
        include_input_bytes(&mut input_bytes, codec.len(), self.limits)?;
        include_input_bytes(&mut input_bytes, op.len(), self.limits)?;
        include_input_bytes(&mut input_bytes, path.len(), self.limits)?;

        for index in 0..fields.len() {
            let key = fields.key(index);
            validate_key(key, self.limits)?;
            fields.validate_value(index, self.limits)?;
            include_input_bytes(&mut input_bytes, key.len(), self.limits)?;
            include_input_bytes(&mut input_bytes, fields.value_len(index), self.limits)?;
            if (0..index).any(|prior| fields.key(prior) == key) {
                return Err(ProfileError::InvalidInput {
                    what: "duplicate profile field key",
                });
            }
        }
        Ok(())
    }

    fn compare_key_to_fields<F: RecordFields>(
        &self,
        key: &SummaryKey,
        codec: &str,
        op: &str,
        path: &str,
        fields: &F,
    ) -> Ordering {
        let base = key
            .codec
            .as_str()
            .cmp(codec)
            .then_with(|| key.op.as_str().cmp(op))
            .then_with(|| key.path.as_str().cmp(path));
        if base != Ordering::Equal {
            return base;
        }

        let mut stored = key.labels.iter();
        let mut current = stored.next();
        for (label_index, label) in self.labels.iter().enumerate() {
            let Some(field_index) = fields.find(label.input_key.as_str()) else {
                continue;
            };
            let Some(stored_label) = current else {
                return Ordering::Less;
            };
            let order = stored_label
                .label_index
                .cmp(&label_index)
                .then_with(|| fields.compare_stored_value(&stored_label.value, field_index));
            if order != Ordering::Equal {
                return order;
            }
            current = stored.next();
        }
        if current.is_some() {
            Ordering::Greater
        } else {
            Ordering::Equal
        }
    }

    fn insert_new<F: RecordFields>(
        &mut self,
        index: usize,
        codec: &str,
        op: &str,
        path: &str,
        fields: &F,
    ) -> ProfileResult<()> {
        let row_count = self
            .rows
            .len()
            .checked_add(1)
            .ok_or(ProfileError::SizeOverflow {
                what: "summary row count",
            })?;
        ensure_limit(row_count, self.limits.max_rows(), "summary row count")?;

        let mut budget = HeapBudget::new(self.retained_bytes, self.limits.max_retained_bytes());
        let entry = self.build_entry(codec, op, path, fields, &mut budget)?;

        if self.rows.len() < self.rows.capacity() {
            self.rows.insert(index, entry);
            self.retained_bytes = budget.used();
            return Ok(());
        }

        let old_outer = element_bytes::<SummaryEntry>(self.rows.capacity(), "summary row storage")?;
        let mut replacement = try_vec(row_count, &mut budget, "summary row storage")?;
        let committed_retained =
            budget
                .used()
                .checked_sub(old_outer)
                .ok_or(ProfileError::SizeOverflow {
                    what: "summary row storage replacement",
                })?;

        replacement.extend(self.rows.drain(..index));
        replacement.push(entry);
        replacement.append(&mut self.rows);
        self.rows = replacement;
        self.retained_bytes = committed_retained;
        Ok(())
    }

    fn build_entry<F: RecordFields>(
        &self,
        codec: &str,
        op: &str,
        path: &str,
        fields: &F,
        budget: &mut HeapBudget,
    ) -> ProfileResult<SummaryEntry> {
        let label_count = self
            .labels
            .iter()
            .filter(|label| fields.find(&label.input_key).is_some())
            .count();
        let mut labels = try_vec(label_count, budget, "summary row labels")?;
        for (label_index, label) in self.labels.iter().enumerate() {
            let Some(field_index) = fields.find(&label.input_key) else {
                continue;
            };
            let mut value =
                try_string_capacity(fields.value_len(field_index), budget, "summary label value")?;
            fields.push_value(&mut value, field_index);
            labels.push(SummaryLabelValue { label_index, value });
        }

        let numeric_count = self.numeric_field_count(fields);
        ensure_limit(
            numeric_count,
            self.limits.max_numeric_fields_per_row(),
            "summary numeric field count",
        )?;
        let mut numeric_sums = try_vec(numeric_count, budget, "summary numeric fields")?;
        if self.numeric_mode == SummaryNumericMode::Aggregate {
            for field_index in 0..fields.len() {
                if let Some(value) = self.numeric_value(fields, field_index) {
                    numeric_sums.push(NumericSum {
                        key: try_string(
                            fields.key(field_index),
                            budget,
                            "summary numeric field key",
                        )?,
                        sum: value,
                    });
                }
            }
            numeric_sums.sort_unstable_by(|left, right| left.key.cmp(&right.key));
        }

        Ok(SummaryEntry {
            key: SummaryKey {
                codec: try_string(codec, budget, "summary codec")?,
                op: try_string(op, budget, "summary operation")?,
                path: try_string(path, budget, "summary path")?,
                labels,
            },
            row: SummaryRow {
                count: 1,
                numeric_sums,
            },
        })
    }

    fn record_existing<F: RecordFields>(
        &mut self,
        row_index: usize,
        fields: &F,
    ) -> ProfileResult<()> {
        let next_count =
            self.rows[row_index]
                .row
                .count
                .checked_add(1)
                .ok_or(ProfileError::SizeOverflow {
                    what: "summary row count",
                })?;
        if self.numeric_mode == SummaryNumericMode::CountOnly {
            self.rows[row_index].row.count = next_count;
            return Ok(());
        }

        let mut new_numeric_count = 0_usize;
        for field_index in 0..fields.len() {
            let Some(value) = self.numeric_value(fields, field_index) else {
                continue;
            };
            let key = fields.key(field_index);
            match self.rows[row_index]
                .row
                .numeric_sums
                .binary_search_by(|sum| sum.key.as_str().cmp(key))
            {
                Ok(existing) => {
                    self.rows[row_index].row.numeric_sums[existing]
                        .sum
                        .checked_add(value)
                        .ok_or(ProfileError::SizeOverflow {
                            what: "summary numeric sum",
                        })?;
                }
                Err(_) => {
                    new_numeric_count =
                        new_numeric_count
                            .checked_add(1)
                            .ok_or(ProfileError::SizeOverflow {
                                what: "summary numeric field count",
                            })?;
                }
            }
        }

        if new_numeric_count == 0 {
            for field_index in 0..fields.len() {
                let Some(value) = self.numeric_value(fields, field_index) else {
                    continue;
                };
                let key = fields.key(field_index);
                let existing = self.rows[row_index]
                    .row
                    .numeric_sums
                    .binary_search_by(|sum| sum.key.as_str().cmp(key))
                    .map_err(|_| ProfileError::InvalidInput {
                        what: "summary numeric update changed after preflight",
                    })?;
                self.rows[row_index].row.numeric_sums[existing].sum += value;
            }
            self.rows[row_index].row.count = next_count;
            return Ok(());
        }

        self.replace_numeric_fields(row_index, fields, new_numeric_count, next_count)
    }

    fn replace_numeric_fields<F: RecordFields>(
        &mut self,
        row_index: usize,
        fields: &F,
        new_numeric_count: usize,
        next_count: u128,
    ) -> ProfileResult<()> {
        let target_count = self.rows[row_index]
            .row
            .numeric_sums
            .len()
            .checked_add(new_numeric_count)
            .ok_or(ProfileError::SizeOverflow {
                what: "summary numeric field count",
            })?;
        ensure_limit(
            target_count,
            self.limits.max_numeric_fields_per_row(),
            "summary numeric field count",
        )?;

        let mut budget = HeapBudget::new(self.retained_bytes, self.limits.max_retained_bytes());
        let mut replacement = try_vec(target_count, &mut budget, "summary numeric fields")?;
        for existing in &self.rows[row_index].row.numeric_sums {
            let incoming = fields
                .find(&existing.key)
                .and_then(|field_index| self.numeric_value(fields, field_index))
                .unwrap_or(0);
            replacement.push(NumericSum {
                key: try_string(&existing.key, &mut budget, "summary numeric field key")?,
                sum: existing
                    .sum
                    .checked_add(incoming)
                    .ok_or(ProfileError::SizeOverflow {
                        what: "summary numeric sum",
                    })?,
            });
        }
        for field_index in 0..fields.len() {
            let Some(value) = self.numeric_value(fields, field_index) else {
                continue;
            };
            let key = fields.key(field_index);
            if self.rows[row_index]
                .row
                .numeric_sums
                .binary_search_by(|sum| sum.key.as_str().cmp(key))
                .is_err()
            {
                replacement.push(NumericSum {
                    key: try_string(key, &mut budget, "summary numeric field key")?,
                    sum: value,
                });
            }
        }
        replacement.sort_unstable_by(|left, right| left.key.cmp(&right.key));

        let old_retained = numeric_retained_bytes(
            &self.rows[row_index].row.numeric_sums,
            self.rows[row_index].row.numeric_sums.capacity(),
        )?;
        let committed_retained =
            budget
                .used()
                .checked_sub(old_retained)
                .ok_or(ProfileError::SizeOverflow {
                    what: "summary numeric storage replacement",
                })?;
        self.rows[row_index].row.numeric_sums = replacement;
        self.rows[row_index].row.count = next_count;
        self.retained_bytes = committed_retained;
        Ok(())
    }

    fn numeric_field_count<F: RecordFields>(&self, fields: &F) -> usize {
        if self.numeric_mode == SummaryNumericMode::CountOnly {
            return 0;
        }
        (0..fields.len())
            .filter(|index| self.numeric_value(fields, *index).is_some())
            .count()
    }

    fn numeric_value<F: RecordFields>(&self, fields: &F, index: usize) -> Option<u128> {
        (!self.is_label_key(fields.key(index)) && fields.summarize_metric(index))
            .then(|| fields.numeric_value(index))
            .flatten()
    }
}

/// Records a string row using only configured labels and timing metrics.
#[cfg(any(feature = "std", test))]
pub(crate) fn record_timing_summary_str(
    summary: &mut ProfileSummary,
    codec: &str,
    op: &str,
    path: &str,
    fields: &[(&str, &str)],
    _summary_label_keys: &[&str],
) -> ProfileResult<()> {
    summary.record(
        codec,
        op,
        path,
        &StringFields {
            fields,
            timing_only: true,
        },
    )
}

fn numeric_retained_bytes(values: &[NumericSum], capacity: usize) -> ProfileResult<usize> {
    let mut retained = element_bytes::<NumericSum>(capacity, "summary numeric fields")?;
    for value in values {
        retained =
            retained
                .checked_add(value.key.capacity())
                .ok_or(ProfileError::SizeOverflow {
                    what: "summary numeric retained bytes",
                })?;
    }
    Ok(retained)
}

trait RecordFields {
    fn len(&self) -> usize;
    fn key(&self, index: usize) -> &str;
    fn value_len(&self, index: usize) -> usize;
    fn validate_value(&self, index: usize, limits: crate::ProfileLimits) -> ProfileResult<()>;
    fn compare_stored_value(&self, stored: &str, index: usize) -> Ordering;
    fn push_value(&self, output: &mut String, index: usize);
    fn numeric_value(&self, index: usize) -> Option<u128>;
    fn summarize_metric(&self, index: usize) -> bool;

    fn find(&self, key: &str) -> Option<usize> {
        (0..self.len()).find(|index| self.key(*index) == key)
    }
}

struct StringFields<'a, K, V> {
    fields: &'a [(K, V)],
    timing_only: bool,
}

impl<K: AsRef<str>, V: AsRef<str>> RecordFields for StringFields<'_, K, V> {
    fn len(&self) -> usize {
        self.fields.len()
    }

    fn key(&self, index: usize) -> &str {
        self.fields[index].0.as_ref()
    }

    fn value_len(&self, index: usize) -> usize {
        self.fields[index].1.as_ref().len()
    }

    fn validate_value(&self, index: usize, limits: crate::ProfileLimits) -> ProfileResult<()> {
        validate_value(self.fields[index].1.as_ref(), limits)
    }

    fn compare_stored_value(&self, stored: &str, index: usize) -> Ordering {
        stored.cmp(self.fields[index].1.as_ref())
    }

    fn push_value(&self, output: &mut String, index: usize) {
        output.push_str(self.fields[index].1.as_ref());
    }

    fn numeric_value(&self, index: usize) -> Option<u128> {
        self.fields[index].1.as_ref().parse().ok()
    }

    fn summarize_metric(&self, index: usize) -> bool {
        !self.timing_only || is_timing_field(self.key(index))
    }
}

struct IntegerFields<'a, K> {
    fields: &'a [(K, u128)],
}

impl<K: AsRef<str>> RecordFields for IntegerFields<'_, K> {
    fn len(&self) -> usize {
        self.fields.len()
    }

    fn key(&self, index: usize) -> &str {
        self.fields[index].0.as_ref()
    }

    fn value_len(&self, index: usize) -> usize {
        decimal_len(self.fields[index].1)
    }

    fn validate_value(&self, index: usize, limits: crate::ProfileLimits) -> ProfileResult<()> {
        ensure_limit(
            self.value_len(index),
            limits.max_token_bytes(),
            "field value",
        )
    }

    fn compare_stored_value(&self, stored: &str, index: usize) -> Ordering {
        compare_text_to_u128(stored, self.fields[index].1)
    }

    fn push_value(&self, output: &mut String, index: usize) {
        push_u128(output, self.fields[index].1);
    }

    fn numeric_value(&self, index: usize) -> Option<u128> {
        Some(self.fields[index].1)
    }

    fn summarize_metric(&self, _index: usize) -> bool {
        true
    }
}

struct TypedFields<'a> {
    fields: &'a [ProfileField],
}

impl RecordFields for TypedFields<'_> {
    fn len(&self) -> usize {
        self.fields.len()
    }

    fn key(&self, index: usize) -> &str {
        self.fields[index].key()
    }

    fn value_len(&self, index: usize) -> usize {
        self.fields[index].value().len()
    }

    fn validate_value(&self, index: usize, limits: crate::ProfileLimits) -> ProfileResult<()> {
        validate_value(self.fields[index].value(), limits)
    }

    fn compare_stored_value(&self, stored: &str, index: usize) -> Ordering {
        stored.cmp(self.fields[index].value())
    }

    fn push_value(&self, output: &mut String, index: usize) {
        output.push_str(self.fields[index].value());
    }

    fn numeric_value(&self, index: usize) -> Option<u128> {
        self.fields[index].value().parse().ok()
    }

    fn summarize_metric(&self, index: usize) -> bool {
        self.fields[index].summarize_metric()
    }
}

#[cfg(test)]
mod tests;
