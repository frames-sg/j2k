use crate::format::{format_profile_fields, format_profile_row, format_profile_row_u128};
use crate::summary::{record_timing_summary_str, ProfileSummary};
use crate::{ProfileField, ProfileStageMode};

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
    emit_profile_line(format_profile_row(codec, op, path, fields));
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
    match mode {
        ProfileStageMode::Disabled => {}
        ProfileStageMode::Rows => {
            std::eprintln!("{}", format_profile_row(codec, op, path, fields));
        }
        ProfileStageMode::Summary => {
            summary.with(|summary| {
                summary
                    .borrow_mut()
                    .record_str(codec.as_ref(), op.as_ref(), path.as_ref(), fields);
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
    match mode {
        ProfileStageMode::Disabled => {}
        ProfileStageMode::Rows => {
            std::eprintln!("{}", format_profile_row_u128(codec, op, path, fields));
        }
        ProfileStageMode::Summary => {
            summary.with(|summary| {
                summary.borrow_mut().record_u128(
                    codec.as_ref(),
                    op.as_ref(),
                    path.as_ref(),
                    fields,
                );
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
    match mode {
        ProfileStageMode::Disabled => {}
        ProfileStageMode::Rows => {
            std::eprintln!("{}", format_profile_fields(codec, op, path, fields));
        }
        ProfileStageMode::Summary => {
            summary.with(|summary| {
                summary.borrow_mut().record_fields(
                    codec.as_ref(),
                    op.as_ref(),
                    path.as_ref(),
                    fields,
                );
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
            std::eprintln!("{}", format_profile_row(codec, op, path, fields));
        }
        ProfileStageMode::Summary => {
            summary.with(|summary| {
                record_timing_summary_str(
                    &mut summary.borrow_mut(),
                    codec,
                    op,
                    path,
                    fields,
                    summary_label_keys,
                );
            });
        }
    }
}
