// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fallible owned profile fields.

use alloc::string::String;
use core::fmt;

use crate::allocation::{checked_add, ensure_limit, try_extend_string, try_string, HeapBudget};
use crate::text::validate_key;
use crate::{ProfileError, ProfileLimits, ProfileResult};

#[derive(Debug, Eq, PartialEq)]
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
#[derive(Debug, Eq, PartialEq)]
pub struct ProfileField {
    key: String,
    value: String,
    kind: ProfileFieldKind,
}

impl ProfileField {
    /// Creates a label field using default limits.
    pub fn label(key: impl AsRef<str>, value: impl fmt::Display) -> ProfileResult<Self> {
        Self::label_with_limits(key, value, ProfileLimits::default())
    }

    /// Creates a label field using caller-supplied limits.
    pub fn label_with_limits(
        key: impl AsRef<str>,
        value: impl fmt::Display,
        limits: ProfileLimits,
    ) -> ProfileResult<Self> {
        Self::new(key, value, ProfileFieldKind::Label, limits)
    }

    /// Creates a summarizable metric field using default limits.
    pub fn metric(key: impl AsRef<str>, value: impl fmt::Display) -> ProfileResult<Self> {
        Self::metric_with_limits(key, value, ProfileLimits::default())
    }

    /// Creates a summarizable metric field using caller-supplied limits.
    pub fn metric_with_limits(
        key: impl AsRef<str>,
        value: impl fmt::Display,
        limits: ProfileLimits,
    ) -> ProfileResult<Self> {
        Self::new(
            key,
            value,
            ProfileFieldKind::Metric { summarize: true },
            limits,
        )
    }

    /// Creates a metric field with explicit summary behavior using default limits.
    pub fn metric_with_summary(
        key: impl AsRef<str>,
        value: impl fmt::Display,
        summarize: bool,
    ) -> ProfileResult<Self> {
        Self::metric_with_summary_and_limits(key, value, summarize, ProfileLimits::default())
    }

    /// Creates a metric field with explicit summary behavior and limits.
    pub fn metric_with_summary_and_limits(
        key: impl AsRef<str>,
        value: impl fmt::Display,
        summarize: bool,
        limits: ProfileLimits,
    ) -> ProfileResult<Self> {
        Self::new(key, value, ProfileFieldKind::Metric { summarize }, limits)
    }

    fn new(
        key: impl AsRef<str>,
        value: impl fmt::Display,
        kind: ProfileFieldKind,
        limits: ProfileLimits,
    ) -> ProfileResult<Self> {
        limits.validate()?;
        let key = key.as_ref();
        validate_key(key, limits)?;
        ensure_limit(key.len(), limits.max_input_bytes(), "input bytes")?;

        let mut budget = HeapBudget::new(0, limits.max_retained_bytes());
        let key = try_string(key, &mut budget, "profile field key")?;
        let mut formatted_value = String::new();
        let mut writer = BoundedValueWriter {
            output: &mut formatted_value,
            budget: &mut budget,
            limits,
            input_bytes: key.len(),
            error: None,
        };
        if fmt::write(&mut writer, format_args!("{value}")).is_err() {
            return Err(writer.error.unwrap_or(ProfileError::InvalidInput {
                what: "profile value formatter failed",
            }));
        }

        Ok(Self {
            key,
            value: formatted_value,
            kind,
        })
    }

    /// Field key.
    pub fn key(&self) -> &str {
        &self.key
    }

    /// Field value.
    pub fn value(&self) -> &str {
        &self.value
    }

    /// Consumes this field and returns its bounded formatted value.
    pub fn into_value(self) -> String {
        self.value
    }

    pub(crate) fn summarize_metric(&self) -> bool {
        matches!(self.kind, ProfileFieldKind::Metric { summarize: true })
    }

    #[cfg(any(feature = "std", test))]
    pub(crate) fn retained_bytes(&self) -> ProfileResult<usize> {
        self.key
            .capacity()
            .checked_add(self.value.capacity())
            .ok_or(ProfileError::SizeOverflow {
                what: "profile field retained bytes",
            })
    }
}

struct BoundedValueWriter<'a> {
    output: &'a mut String,
    budget: &'a mut HeapBudget,
    limits: ProfileLimits,
    input_bytes: usize,
    error: Option<ProfileError>,
}

impl fmt::Write for BoundedValueWriter<'_> {
    fn write_str(&mut self, text: &str) -> fmt::Result {
        if self.error.is_some() {
            return Err(fmt::Error);
        }

        let result = self.try_write_str(text);
        if let Err(error) = result {
            self.error = Some(error);
            return Err(fmt::Error);
        }
        Ok(())
    }
}

impl BoundedValueWriter<'_> {
    fn try_write_str(&mut self, text: &str) -> ProfileResult<()> {
        if text.chars().any(char::is_whitespace) {
            return Err(ProfileError::InvalidInput {
                what: "profile token contains whitespace",
            });
        }
        let value_len = checked_add(self.output.len(), text.len(), "profile field value")?;
        ensure_limit(value_len, self.limits.max_token_bytes(), "field value")?;
        let input_bytes = checked_add(self.input_bytes, text.len(), "profile input bytes")?;
        ensure_limit(input_bytes, self.limits.max_input_bytes(), "input bytes")?;
        try_extend_string(self.output, text, self.budget, "profile field value")?;
        self.input_bytes = input_bytes;
        Ok(())
    }
}
