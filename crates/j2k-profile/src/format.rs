use alloc::string::String;
use core::fmt::Write as _;

#[cfg(any(feature = "std", test))]
use crate::field::{self, ProfileField};
#[cfg(any(feature = "std", test))]
use crate::parse::PROFILE_ROW_PREFIX;
use crate::parse::PROFILE_SUMMARY_PREFIX;

/// Formats a profiling row from string fields.
#[cfg(any(feature = "std", test))]
pub(crate) fn format_profile_row<K, V>(
    codec: impl AsRef<str>,
    op: impl AsRef<str>,
    path: impl AsRef<str>,
    fields: &[(K, V)],
) -> String
where
    K: AsRef<str>,
    V: AsRef<str>,
{
    let mut row = format_profile_prefix(codec.as_ref(), op.as_ref(), path.as_ref());
    row.push_str(&format_profile_key_value_fields(fields));
    row
}

/// Formats profiling key/value fields without adding the standard row prefix.
pub fn format_profile_key_value_fields<K, V>(fields: &[(K, V)]) -> String
where
    K: AsRef<str>,
    V: AsRef<str>,
{
    let mut row = String::new();
    for (key, value) in fields {
        write!(row, " {}={}", key.as_ref(), value.as_ref()).expect("writing to String failed");
    }
    row
}

/// Formats a profiling row from typed fields.
#[cfg(any(feature = "std", test))]
pub(crate) fn format_profile_fields(
    codec: impl AsRef<str>,
    op: impl AsRef<str>,
    path: impl AsRef<str>,
    fields: &[ProfileField],
) -> String {
    let pairs = field::field_pairs(fields);
    format_profile_row(codec, op, path, &pairs)
}

/// Formats a profiling row from integer fields.
#[cfg(any(feature = "std", test))]
pub(crate) fn format_profile_row_u128<K>(
    codec: impl AsRef<str>,
    op: impl AsRef<str>,
    path: impl AsRef<str>,
    fields: &[(K, u128)],
) -> String
where
    K: AsRef<str>,
{
    let mut row = format_profile_prefix(codec.as_ref(), op.as_ref(), path.as_ref());
    for (key, value) in fields {
        write!(row, " {}={value}", key.as_ref()).expect("writing to String failed");
    }
    row
}

pub(crate) fn format_profile_summary_prefix(codec: &str, op: &str, path: &str) -> String {
    let mut row = String::new();
    write!(
        row,
        "{PROFILE_SUMMARY_PREFIX} codec={codec} op={op} path={path}"
    )
    .expect("writing to String failed");
    row
}

#[cfg(any(feature = "std", test))]
fn format_profile_prefix(codec: &str, op: &str, path: &str) -> String {
    let mut row = String::new();
    write!(
        row,
        "{PROFILE_ROW_PREFIX} codec={codec} op={op} path={path}"
    )
    .expect("writing to String failed");
    row
}
