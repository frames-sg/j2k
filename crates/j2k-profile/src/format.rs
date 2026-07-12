// SPDX-License-Identifier: MIT OR Apache-2.0

//! Exact, bounded formatting for profile rows and field lists.

use alloc::string::String;

use crate::allocation::{checked_add, ensure_limit, try_output_string};
#[cfg(any(feature = "std", test))]
use crate::field::ProfileField;
#[cfg(any(feature = "std", test))]
use crate::parse::PROFILE_ROW_PREFIX;
#[cfg(any(feature = "std", test))]
use crate::text::{decimal_len, push_u128, validate_identity};
use crate::text::{include_input_bytes, validate_key, validate_value};
use crate::{ProfileLimits, ProfileResult};

/// Formats profiling key/value fields using default limits.
pub fn format_profile_key_value_fields<K, V>(fields: &[(K, V)]) -> ProfileResult<String>
where
    K: AsRef<str>,
    V: AsRef<str>,
{
    format_profile_key_value_fields_with_limits(fields, ProfileLimits::default())
}

/// Formats profiling key/value fields using caller-supplied limits.
pub fn format_profile_key_value_fields_with_limits<K, V>(
    fields: &[(K, V)],
    limits: ProfileLimits,
) -> ProfileResult<String>
where
    K: AsRef<str>,
    V: AsRef<str>,
{
    limits.validate()?;
    ensure_limit(fields.len(), limits.max_fields(), "field count")?;

    let mut input_bytes = 0_usize;
    let mut output_bytes = 0_usize;
    for (key, value) in fields {
        let key = key.as_ref();
        let value = value.as_ref();
        validate_key(key, limits)?;
        validate_value(value, limits)?;
        include_input_bytes(&mut input_bytes, key.len(), limits)?;
        include_input_bytes(&mut input_bytes, value.len(), limits)?;
        output_bytes = field_output_len(output_bytes, key.len(), value.len())?;
    }

    let mut output = try_output_string(
        output_bytes,
        limits.max_output_bytes(),
        "formatted field output",
    )?;
    for (key, value) in fields {
        push_field(&mut output, key.as_ref(), value.as_ref());
    }
    Ok(output)
}

/// Formats a profiling row from string fields.
#[cfg(any(feature = "std", test))]
pub(crate) fn format_profile_row<K, V>(
    codec: impl AsRef<str>,
    op: impl AsRef<str>,
    path: impl AsRef<str>,
    fields: &[(K, V)],
) -> ProfileResult<String>
where
    K: AsRef<str>,
    V: AsRef<str>,
{
    let limits = ProfileLimits::default();
    let codec = codec.as_ref();
    let op = op.as_ref();
    let path = path.as_ref();
    let mut output_bytes = validate_prefix(codec, op, path, fields.len(), limits)?;
    let mut input_bytes = prefix_input_bytes(codec, op, path, limits)?;

    for (key, value) in fields {
        let key = key.as_ref();
        let value = value.as_ref();
        validate_key(key, limits)?;
        validate_value(value, limits)?;
        include_input_bytes(&mut input_bytes, key.len(), limits)?;
        include_input_bytes(&mut input_bytes, value.len(), limits)?;
        output_bytes = field_output_len(output_bytes, key.len(), value.len())?;
    }

    let mut output = try_output_string(
        output_bytes,
        limits.max_output_bytes(),
        "formatted profile row",
    )?;
    push_prefix(&mut output, codec, op, path);
    for (key, value) in fields {
        push_field(&mut output, key.as_ref(), value.as_ref());
    }
    Ok(output)
}

/// Formats a profiling row from typed fields.
#[cfg(any(feature = "std", test))]
pub(crate) fn format_profile_fields(
    codec: impl AsRef<str>,
    op: impl AsRef<str>,
    path: impl AsRef<str>,
    fields: &[ProfileField],
) -> ProfileResult<String> {
    let limits = ProfileLimits::default();
    let codec = codec.as_ref();
    let op = op.as_ref();
    let path = path.as_ref();
    let mut output_bytes = validate_prefix(codec, op, path, fields.len(), limits)?;
    let mut input_bytes = prefix_input_bytes(codec, op, path, limits)?;

    for field in fields {
        validate_key(field.key(), limits)?;
        validate_value(field.value(), limits)?;
        include_input_bytes(&mut input_bytes, field.key().len(), limits)?;
        include_input_bytes(&mut input_bytes, field.value().len(), limits)?;
        output_bytes = field_output_len(output_bytes, field.key().len(), field.value().len())?;
    }

    let mut output = try_output_string(
        output_bytes,
        limits.max_output_bytes(),
        "formatted typed profile row",
    )?;
    push_prefix(&mut output, codec, op, path);
    for field in fields {
        push_field(&mut output, field.key(), field.value());
    }
    Ok(output)
}

/// Formats a profiling row from integer fields.
#[cfg(any(feature = "std", test))]
pub fn format_profile_row_u128<K>(
    codec: impl AsRef<str>,
    op: impl AsRef<str>,
    path: impl AsRef<str>,
    fields: &[(K, u128)],
) -> ProfileResult<String>
where
    K: AsRef<str>,
{
    let limits = ProfileLimits::default();
    let codec = codec.as_ref();
    let op = op.as_ref();
    let path = path.as_ref();
    let mut output_bytes = validate_prefix(codec, op, path, fields.len(), limits)?;
    let mut input_bytes = prefix_input_bytes(codec, op, path, limits)?;

    for (key, value) in fields {
        let key = key.as_ref();
        let value_len = decimal_len(*value);
        validate_key(key, limits)?;
        ensure_limit(value_len, limits.max_token_bytes(), "field value")?;
        include_input_bytes(&mut input_bytes, key.len(), limits)?;
        include_input_bytes(&mut input_bytes, value_len, limits)?;
        output_bytes = field_output_len(output_bytes, key.len(), value_len)?;
    }

    let mut output = try_output_string(
        output_bytes,
        limits.max_output_bytes(),
        "formatted integer profile row",
    )?;
    push_prefix(&mut output, codec, op, path);
    for (key, value) in fields {
        output.push(' ');
        output.push_str(key.as_ref());
        output.push('=');
        push_u128(&mut output, *value);
    }
    Ok(output)
}

#[cfg(any(feature = "std", test))]
fn validate_prefix(
    codec: &str,
    op: &str,
    path: &str,
    field_count: usize,
    limits: ProfileLimits,
) -> ProfileResult<usize> {
    limits.validate()?;
    ensure_limit(field_count, limits.max_fields(), "field count")?;
    validate_identity(codec, limits)?;
    validate_identity(op, limits)?;
    validate_identity(path, limits)?;
    let length = checked_add(PROFILE_ROW_PREFIX.len(), " codec=".len(), "profile row")?;
    let length = checked_add(length, codec.len(), "profile row")?;
    let length = checked_add(length, " op=".len(), "profile row")?;
    let length = checked_add(length, op.len(), "profile row")?;
    let length = checked_add(length, " path=".len(), "profile row")?;
    checked_add(length, path.len(), "profile row")
}

#[cfg(any(feature = "std", test))]
fn prefix_input_bytes(
    codec: &str,
    op: &str,
    path: &str,
    limits: ProfileLimits,
) -> ProfileResult<usize> {
    let mut input_bytes = 0;
    include_input_bytes(&mut input_bytes, codec.len(), limits)?;
    include_input_bytes(&mut input_bytes, op.len(), limits)?;
    include_input_bytes(&mut input_bytes, path.len(), limits)?;
    Ok(input_bytes)
}

pub(crate) fn field_output_len(
    current: usize,
    key_len: usize,
    value_len: usize,
) -> ProfileResult<usize> {
    let length = checked_add(current, 1, "formatted profile output")?;
    let length = checked_add(length, key_len, "formatted profile output")?;
    let length = checked_add(length, 1, "formatted profile output")?;
    checked_add(length, value_len, "formatted profile output")
}

pub(crate) fn push_field(output: &mut String, key: &str, value: &str) {
    output.push(' ');
    output.push_str(key);
    output.push('=');
    output.push_str(value);
}

#[cfg(any(feature = "std", test))]
fn push_prefix(output: &mut String, codec: &str, op: &str, path: &str) {
    output.push_str(PROFILE_ROW_PREFIX);
    push_field(output, "codec", codec);
    push_field(output, "op", op);
    push_field(output, "path", path);
}
