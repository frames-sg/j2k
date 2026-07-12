use crate::format::{format_profile_fields, format_profile_row, format_profile_row_u128};
use crate::summary::{record_timing_summary_str, ProfileSummary};
use crate::{ProfileField, ProfileResult, ProfileStageMode};

/// Emits a preformatted profile row to stderr.
pub fn emit_profile_line(row: impl AsRef<str>) {
    std::eprintln!("{}", row.as_ref());
}

/// Formats and emits a string-valued profile row to stderr.
pub fn emit_profile_row_now<K, V>(
    codec: impl AsRef<str>,
    op: impl AsRef<str>,
    path: impl AsRef<str>,
    fields: &[(K, V)],
) where
    K: AsRef<str>,
    V: AsRef<str>,
{
    emit_formatted("row_now", format_profile_row(codec, op, path, fields));
}

/// Emits or records a string-valued profiling row according to the stage mode.
pub fn emit_profile_row<K, V>(
    mode: ProfileStageMode,
    summary: &'static std::thread::LocalKey<std::cell::RefCell<ProfileSummary>>,
    codec: impl AsRef<str>,
    op: impl AsRef<str>,
    path: impl AsRef<str>,
    fields: &[(K, V)],
) where
    K: AsRef<str>,
    V: AsRef<str>,
{
    let codec = codec.as_ref();
    let op = op.as_ref();
    let path = path.as_ref();
    match mode {
        ProfileStageMode::Disabled => {}
        ProfileStageMode::Rows => {
            emit_formatted("row", format_profile_row(codec, op, path, fields));
        }
        ProfileStageMode::Summary => {
            summary.with(|summary| {
                if let Err(error) = summary.borrow_mut().record_str(codec, op, path, fields) {
                    emit_profile_error("summary_record", &error);
                }
            });
        }
    }
}

/// Emits or records an integer-valued profiling row according to the stage mode.
pub fn emit_profile_row_u128<K>(
    mode: ProfileStageMode,
    summary: &'static std::thread::LocalKey<std::cell::RefCell<ProfileSummary>>,
    codec: impl AsRef<str>,
    op: impl AsRef<str>,
    path: impl AsRef<str>,
    fields: &[(K, u128)],
) where
    K: AsRef<str>,
{
    let codec = codec.as_ref();
    let op = op.as_ref();
    let path = path.as_ref();
    match mode {
        ProfileStageMode::Disabled => {}
        ProfileStageMode::Rows => {
            emit_formatted(
                "integer_row",
                format_profile_row_u128(codec, op, path, fields),
            );
        }
        ProfileStageMode::Summary => {
            summary.with(|summary| {
                if let Err(error) = summary.borrow_mut().record_u128(codec, op, path, fields) {
                    emit_profile_error("integer_summary_record", &error);
                }
            });
        }
    }
}

/// Emits or records a typed profiling row according to the stage mode.
pub fn emit_profile_fields(
    mode: ProfileStageMode,
    summary: &'static std::thread::LocalKey<std::cell::RefCell<ProfileSummary>>,
    codec: impl AsRef<str>,
    op: impl AsRef<str>,
    path: impl AsRef<str>,
    fields: &[ProfileField],
) {
    let codec = codec.as_ref();
    let op = op.as_ref();
    let path = path.as_ref();
    match mode {
        ProfileStageMode::Disabled => {}
        ProfileStageMode::Rows => {
            emit_formatted("typed_row", format_profile_fields(codec, op, path, fields));
        }
        ProfileStageMode::Summary => {
            summary.with(|summary| {
                if let Err(error) = summary.borrow_mut().record_fields(codec, op, path, fields) {
                    emit_profile_error("typed_summary_record", &error);
                }
            });
        }
    }
}

/// Emits string-valued rows, or records a timing-filtered summary row.
pub fn emit_profile_row_with_timing_summary(
    mode: ProfileStageMode,
    summary: &'static std::thread::LocalKey<std::cell::RefCell<ProfileSummary>>,
    codec: &str,
    op: &str,
    path: &str,
    fields: &[(&str, &str)],
    summary_label_keys: &[&str],
) {
    match mode {
        ProfileStageMode::Disabled => {}
        ProfileStageMode::Rows => {
            emit_formatted("timing_row", format_profile_row(codec, op, path, fields));
        }
        ProfileStageMode::Summary => {
            summary.with(|summary| {
                if let Err(error) = record_timing_summary_str(
                    &mut summary.borrow_mut(),
                    codec,
                    op,
                    path,
                    fields,
                    summary_label_keys,
                ) {
                    emit_profile_error("timing_summary_record", &error);
                }
            });
        }
    }
}

fn emit_formatted(operation: &'static str, result: ProfileResult<String>) {
    match result {
        Ok(row) => emit_profile_line(row),
        Err(error) => emit_profile_error(operation, &error),
    }
}

/// Emits a profile-construction failure without changing codec success.
pub fn emit_profile_error<E: std::fmt::Display + ?Sized>(operation: &str, error: &E) {
    std::eprintln!("j2k_profile_error operation={operation} error={error}");
}
