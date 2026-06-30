use alloc::borrow::ToOwned;
use alloc::string::String;
use alloc::vec::Vec;

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
#[derive(Clone, Debug, Eq, PartialEq)]
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

/// Parses a `j2k_profile` or `j2k_profile_summary` line.
pub fn parse_profile_line(line: &str) -> Option<ParsedProfileFields> {
    if let Some(rest) = line.strip_prefix(PROFILE_ROW_LINE_PREFIX) {
        Some(ParsedProfileFields {
            kind: ParsedProfileKind::Row,
            fields: parse_profile_key_value_fields(rest),
        })
    } else {
        line.strip_prefix(PROFILE_SUMMARY_LINE_PREFIX)
            .map(|rest| ParsedProfileFields {
                kind: ParsedProfileKind::Summary,
                fields: parse_profile_key_value_fields(rest),
            })
    }
}

/// Parses the whitespace-delimited `key=value` field list used by profile rows.
pub fn parse_profile_key_value_fields(text: &str) -> Vec<(String, String)> {
    text.split_whitespace()
        .filter_map(|part| {
            part.split_once('=')
                .map(|(key, value)| (key.to_owned(), value.to_owned()))
        })
        .collect()
}
