// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared validation and allocation-free decimal helpers for profile text.

use alloc::string::String;
use core::cmp::Ordering;

use crate::allocation::{checked_add, ensure_limit};
use crate::{ProfileError, ProfileLimits, ProfileResult};

pub(crate) fn validate_key(key: &str, limits: ProfileLimits) -> ProfileResult<()> {
    validate_token(key, limits, "field key")?;
    if key.contains('=') {
        return Err(ProfileError::InvalidInput {
            what: "field key contains '='",
        });
    }
    Ok(())
}

pub(crate) fn validate_value(value: &str, limits: ProfileLimits) -> ProfileResult<()> {
    validate_token_allow_empty(value, limits, "field value")
}

pub(crate) fn validate_identity(value: &str, limits: ProfileLimits) -> ProfileResult<()> {
    validate_token(value, limits, "codec, operation, or path")
}

fn validate_token(value: &str, limits: ProfileLimits, what: &'static str) -> ProfileResult<()> {
    if value.is_empty() {
        return Err(ProfileError::InvalidInput {
            what: "profile token is empty",
        });
    }
    validate_token_allow_empty(value, limits, what)
}

fn validate_token_allow_empty(
    value: &str,
    limits: ProfileLimits,
    what: &'static str,
) -> ProfileResult<()> {
    ensure_limit(value.len(), limits.max_token_bytes(), what)?;
    if value.chars().any(char::is_whitespace) {
        return Err(ProfileError::InvalidInput {
            what: "profile token contains whitespace",
        });
    }
    Ok(())
}

pub(crate) fn include_input_bytes(
    used: &mut usize,
    bytes: usize,
    limits: ProfileLimits,
) -> ProfileResult<()> {
    *used = checked_add(*used, bytes, "profile input bytes")?;
    ensure_limit(*used, limits.max_input_bytes(), "input bytes")
}

pub(crate) const fn decimal_len(mut value: u128) -> usize {
    let mut digits = 1;
    while value >= 10 {
        value /= 10;
        digits += 1;
    }
    digits
}

pub(crate) fn push_u128(output: &mut String, mut value: u128) {
    let mut digits = [0_u8; 39];
    let mut cursor = digits.len();
    loop {
        cursor -= 1;
        digits[cursor] = (value % 10) as u8;
        value /= 10;
        if value == 0 {
            break;
        }
    }
    for digit in &digits[cursor..] {
        output.push(char::from(b'0' + *digit));
    }
}

pub(crate) fn compare_text_to_u128(text: &str, mut value: u128) -> Ordering {
    let mut digits = [0_u8; 39];
    let mut cursor = digits.len();
    loop {
        cursor -= 1;
        digits[cursor] = b'0' + (value % 10) as u8;
        value /= 10;
        if value == 0 {
            break;
        }
    }
    text.as_bytes().cmp(&digits[cursor..])
}
