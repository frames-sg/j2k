// SPDX-License-Identifier: MIT OR Apache-2.0

//! Bounded parsing for the workspace profile-line grammar.

use alloc::string::String;
use alloc::vec::Vec;

use crate::allocation::{ensure_limit, try_string, try_vec, HeapBudget};
use crate::text::{validate_key, validate_value};
use crate::{ProfileError, ProfileLimits, ProfileResult};

#[cfg(any(feature = "std", test))]
pub(crate) const PROFILE_ROW_PREFIX: &str = "j2k_profile";
pub(crate) const PROFILE_SUMMARY_PREFIX: &str = "j2k_profile_summary";

const PROFILE_ROW_LINE_PREFIX: &str = "j2k_profile ";
const PROFILE_SUMMARY_LINE_PREFIX: &str = "j2k_profile_summary ";

/// Identifies the profile line family parsed from text.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ParsedProfileKind {
    /// A raw profile row emitted as `j2k_profile`.
    Row,
    /// An aggregate profile row emitted as `j2k_profile_summary`.
    Summary,
}

/// Parsed `key=value` profile fields.
#[derive(Debug, Eq, PartialEq)]
pub struct ParsedProfileFields {
    kind: ParsedProfileKind,
    fields: Vec<(String, String)>,
}

impl ParsedProfileFields {
    /// Parsed profile line kind.
    pub fn kind(&self) -> ParsedProfileKind {
        self.kind
    }

    /// Returns the first value for `key`.
    pub fn get(&self, key: &str) -> Option<&str> {
        self.fields
            .iter()
            .find_map(|(field_key, value)| (field_key == key).then_some(value.as_str()))
    }

    /// Returns all parsed fields in original order.
    pub fn fields(&self) -> &[(String, String)] {
        &self.fields
    }
}

/// Parses a `j2k_profile` or `j2k_profile_summary` line with default limits.
pub fn parse_profile_line(line: &str) -> ProfileResult<Option<ParsedProfileFields>> {
    parse_profile_line_with_limits(line, ProfileLimits::default())
}

/// Parses a profile line with caller-supplied limits.
pub fn parse_profile_line_with_limits(
    line: &str,
    limits: ProfileLimits,
) -> ProfileResult<Option<ParsedProfileFields>> {
    limits.validate()?;
    ensure_limit(line.len(), limits.max_input_bytes(), "input bytes")?;

    let (kind, rest) = if let Some(rest) = line.strip_prefix(PROFILE_ROW_LINE_PREFIX) {
        (ParsedProfileKind::Row, rest)
    } else if let Some(rest) = line.strip_prefix(PROFILE_SUMMARY_LINE_PREFIX) {
        (ParsedProfileKind::Summary, rest)
    } else {
        return Ok(None);
    };

    Ok(Some(ParsedProfileFields {
        kind,
        fields: parse_profile_key_value_fields_with_limits_inner(rest, limits, false)?,
    }))
}

/// Parses the whitespace-delimited `key=value` field list used by profile rows.
pub fn parse_profile_key_value_fields(text: &str) -> ProfileResult<Vec<(String, String)>> {
    parse_profile_key_value_fields_with_limits(text, ProfileLimits::default())
}

/// Parses a profile field list with caller-supplied limits.
pub fn parse_profile_key_value_fields_with_limits(
    text: &str,
    limits: ProfileLimits,
) -> ProfileResult<Vec<(String, String)>> {
    limits.validate()?;
    parse_profile_key_value_fields_with_limits_inner(text, limits, true)
}

fn parse_profile_key_value_fields_with_limits_inner(
    text: &str,
    limits: ProfileLimits,
    check_input_limit: bool,
) -> ProfileResult<Vec<(String, String)>> {
    if check_input_limit {
        ensure_limit(text.len(), limits.max_input_bytes(), "input bytes")?;
    }

    let mut count = 0_usize;
    for part in text.split_whitespace() {
        let (key, value) = split_field(part)?;
        validate_key(key, limits)?;
        validate_value(value, limits)?;
        count = count.checked_add(1).ok_or(ProfileError::SizeOverflow {
            what: "profile field count",
        })?;
        ensure_limit(count, limits.max_fields(), "field count")?;
    }

    let mut budget = HeapBudget::new(0, limits.max_retained_bytes());
    let mut fields = try_vec(count, &mut budget, "parsed field storage")?;
    for part in text.split_whitespace() {
        let (key, value) = split_field(part)?;
        let key = try_string(key, &mut budget, "parsed field key")?;
        let value = try_string(value, &mut budget, "parsed field value")?;
        fields.push((key, value));
    }
    Ok(fields)
}

fn split_field(part: &str) -> ProfileResult<(&str, &str)> {
    part.split_once('=').ok_or(ProfileError::InvalidInput {
        what: "profile field is not key=value",
    })
}
