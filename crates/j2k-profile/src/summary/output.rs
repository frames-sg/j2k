// SPDX-License-Identifier: MIT OR Apache-2.0

//! Exact transactional formatting for retained profile summaries.

use alloc::string::String;
use alloc::vec::Vec;

use super::{is_timing_field, ProfileSummary, SummaryEntry};
use crate::allocation::{checked_add, try_string_capacity, try_vec, HeapBudget};
use crate::format::{field_output_len, push_field};
use crate::parse::PROFILE_SUMMARY_PREFIX;
use crate::text::{decimal_len, push_u128};
use crate::{ProfileError, ProfileResult};

impl ProfileSummary {
    /// Formats all retained rows deterministically without changing the summary.
    pub fn format_rows(&self) -> ProfileResult<Vec<String>> {
        let mut budget = HeapBudget::new(0, self.limits.max_output_bytes());
        let mut output = try_vec(self.rows.len(), &mut budget, "summary output rows")?;
        for entry in &self.rows {
            output.push(self.format_entry(entry, &mut budget)?);
        }
        Ok(output)
    }

    /// Formats all retained rows and clears state only after every row succeeds.
    pub fn take_formatted_rows(&mut self) -> ProfileResult<Vec<String>> {
        let empty_retained = self.empty_retained_bytes()?;
        let rows = self.format_rows()?;
        self.rows.clear();
        self.retained_bytes = empty_retained;
        Ok(rows)
    }

    /// Flushes all retained rows to stderr and clears them transactionally.
    #[cfg(feature = "std")]
    pub fn flush_to_stderr(&mut self) -> ProfileResult<()> {
        for row in self.take_formatted_rows()? {
            std::eprintln!("{row}");
        }
        Ok(())
    }

    fn format_entry(&self, entry: &SummaryEntry, budget: &mut HeapBudget) -> ProfileResult<String> {
        let mut length = summary_prefix_len(&entry.key.codec, &entry.key.op, &entry.key.path)?;
        for label in &entry.key.labels {
            let configured =
                self.labels
                    .get(label.label_index)
                    .ok_or(ProfileError::InvalidInput {
                        what: "summary label index is invalid",
                    })?;
            length = field_output_len(length, configured.summary_key.len(), label.value.len())?;
        }
        length = field_output_len(length, "count".len(), decimal_len(entry.row.count))?;
        for numeric in &entry.row.numeric_sums {
            let sum_key_len = checked_add(
                numeric.key.len(),
                "_sum".len(),
                "summary numeric output key",
            )?;
            length = field_output_len(length, sum_key_len, decimal_len(numeric.sum))?;
            if is_timing_field(&numeric.key) {
                let average_key_len = checked_add(
                    numeric.key.len(),
                    "_avg".len(),
                    "summary numeric output key",
                )?;
                length = field_output_len(
                    length,
                    average_key_len,
                    decimal_len(numeric.sum / entry.row.count),
                )?;
            }
        }

        let mut output = try_string_capacity(length, budget, "summary formatted row")?;
        push_summary_prefix(
            &mut output,
            &entry.key.codec,
            &entry.key.op,
            &entry.key.path,
        );
        for label in &entry.key.labels {
            let configured =
                self.labels
                    .get(label.label_index)
                    .ok_or(ProfileError::InvalidInput {
                        what: "summary label index is invalid",
                    })?;
            push_field(&mut output, &configured.summary_key, &label.value);
        }
        push_numeric_field(&mut output, "count", "", entry.row.count);
        for numeric in &entry.row.numeric_sums {
            push_numeric_field(&mut output, &numeric.key, "_sum", numeric.sum);
            if is_timing_field(&numeric.key) {
                push_numeric_field(
                    &mut output,
                    &numeric.key,
                    "_avg",
                    numeric.sum / entry.row.count,
                );
            }
        }
        Ok(output)
    }
}

fn summary_prefix_len(codec: &str, op: &str, path: &str) -> ProfileResult<usize> {
    let length = checked_add(
        PROFILE_SUMMARY_PREFIX.len(),
        " codec=".len(),
        "summary output",
    )?;
    let length = checked_add(length, codec.len(), "summary output")?;
    let length = checked_add(length, " op=".len(), "summary output")?;
    let length = checked_add(length, op.len(), "summary output")?;
    let length = checked_add(length, " path=".len(), "summary output")?;
    checked_add(length, path.len(), "summary output")
}

fn push_summary_prefix(output: &mut String, codec: &str, op: &str, path: &str) {
    output.push_str(PROFILE_SUMMARY_PREFIX);
    push_field(output, "codec", codec);
    push_field(output, "op", op);
    push_field(output, "path", path);
}

fn push_numeric_field(output: &mut String, key: &str, suffix: &str, value: u128) {
    output.push(' ');
    output.push_str(key);
    output.push_str(suffix);
    output.push('=');
    push_u128(output, value);
}
