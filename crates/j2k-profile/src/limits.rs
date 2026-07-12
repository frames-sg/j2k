// SPDX-License-Identifier: MIT OR Apache-2.0

//! Explicit limits for opt-in profile text and retained summaries.

use crate::{ProfileError, ProfileResult};

const DEFAULT_INPUT_BYTES: usize = 64 * 1024;
const DEFAULT_TOKEN_BYTES: usize = 16 * 1024;
const DEFAULT_FIELDS: usize = 256;
const DEFAULT_ROWS: usize = 1_024;
const DEFAULT_LABELS: usize = 32;
const DEFAULT_NUMERIC_FIELDS: usize = 128;
const DEFAULT_RETAINED_BYTES: usize = 8 * 1024 * 1024;
const DEFAULT_OUTPUT_BYTES: usize = 8 * 1024 * 1024;

/// Limits applied to public profile parsing, formatting, and aggregation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProfileLimits {
    input_bytes: usize,
    token_bytes: usize,
    fields: usize,
    rows: usize,
    labels: usize,
    numeric_fields_per_row: usize,
    retained_bytes: usize,
    output_bytes: usize,
}

impl ProfileLimits {
    /// Maximum caller text bytes accepted by one parse, format, or record operation.
    pub const fn max_input_bytes(self) -> usize {
        self.input_bytes
    }

    /// Maximum bytes accepted for one key, value, codec, operation, or path token.
    pub const fn max_token_bytes(self) -> usize {
        self.token_bytes
    }

    /// Maximum fields accepted by one profile row.
    pub const fn max_fields(self) -> usize {
        self.fields
    }

    /// Maximum distinct rows retained by one summary.
    pub const fn max_rows(self) -> usize {
        self.rows
    }

    /// Maximum configured labels retained by one summary.
    pub const fn max_labels(self) -> usize {
        self.labels
    }

    /// Maximum distinct numeric fields retained by one summary row.
    pub const fn max_numeric_fields_per_row(self) -> usize {
        self.numeric_fields_per_row
    }

    /// Maximum allocator-reported heap capacity retained by one summary or parsed field set.
    pub const fn max_retained_bytes(self) -> usize {
        self.retained_bytes
    }

    /// Maximum allocator-reported capacity of one formatted output collection.
    pub const fn max_output_bytes(self) -> usize {
        self.output_bytes
    }

    /// Return these limits with a different per-operation input-byte ceiling.
    #[must_use]
    pub const fn with_max_input_bytes(mut self, value: usize) -> Self {
        self.input_bytes = value;
        self
    }

    /// Return these limits with a different token-byte ceiling.
    #[must_use]
    pub const fn with_max_token_bytes(mut self, value: usize) -> Self {
        self.token_bytes = value;
        self
    }

    /// Return these limits with a different field-count ceiling.
    #[must_use]
    pub const fn with_max_fields(mut self, value: usize) -> Self {
        self.fields = value;
        self
    }

    /// Return these limits with a different distinct-row ceiling.
    #[must_use]
    pub const fn with_max_rows(mut self, value: usize) -> Self {
        self.rows = value;
        self
    }

    /// Return these limits with a different summary-label ceiling.
    #[must_use]
    pub const fn with_max_labels(mut self, value: usize) -> Self {
        self.labels = value;
        self
    }

    /// Return these limits with a different per-row numeric-field ceiling.
    #[must_use]
    pub const fn with_max_numeric_fields_per_row(mut self, value: usize) -> Self {
        self.numeric_fields_per_row = value;
        self
    }

    /// Return these limits with a different retained allocator-capacity ceiling.
    #[must_use]
    pub const fn with_max_retained_bytes(mut self, value: usize) -> Self {
        self.retained_bytes = value;
        self
    }

    /// Return these limits with a different formatted-output capacity ceiling.
    #[must_use]
    pub const fn with_max_output_bytes(mut self, value: usize) -> Self {
        self.output_bytes = value;
        self
    }

    pub(crate) fn validate(self) -> ProfileResult<()> {
        if self.token_bytes > self.input_bytes {
            return Err(ProfileError::InvalidLimits {
                what: "token bytes exceed input bytes",
            });
        }
        Ok(())
    }
}

impl Default for ProfileLimits {
    fn default() -> Self {
        Self {
            input_bytes: DEFAULT_INPUT_BYTES,
            token_bytes: DEFAULT_TOKEN_BYTES,
            fields: DEFAULT_FIELDS,
            rows: DEFAULT_ROWS,
            labels: DEFAULT_LABELS,
            numeric_fields_per_row: DEFAULT_NUMERIC_FIELDS,
            retained_bytes: DEFAULT_RETAINED_BYTES,
            output_bytes: DEFAULT_OUTPUT_BYTES,
        }
    }
}
