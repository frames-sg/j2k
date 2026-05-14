#[cfg(feature = "std")]
use core::cell::RefCell;

#[cfg(feature = "std")]
use signinum_profile::{profile_stage_mode_from_value, ProfileStageMode};

#[cfg(feature = "std")]
use std::sync::OnceLock;
#[cfg(feature = "std")]
use std::time::Instant;

#[cfg(all(test, feature = "std"))]
pub(crate) use signinum_profile::ProfileSummary;

#[cfg(feature = "std")]
const PROFILE_ENV_VAR: &str = "SIGNINUM_J2K_PROFILE_STAGES";

#[cfg(feature = "std")]
#[cfg(test)]
pub(crate) fn parse_profile_env_flag(value: Option<&str>) -> bool {
    profile_stage_mode_from_value(value) != ProfileStageMode::Disabled
}

#[cfg(feature = "std")]
pub(crate) fn profile_stages_enabled() -> bool {
    profile_stage_mode() != ProfileStageMode::Disabled
}

#[cfg(feature = "std")]
fn profile_stage_mode() -> ProfileStageMode {
    static MODE: OnceLock<ProfileStageMode> = OnceLock::new();
    *MODE.get_or_init(|| {
        profile_stage_mode_from_value(std::env::var(PROFILE_ENV_VAR).ok().as_deref())
    })
}

#[cfg(not(feature = "std"))]
pub(crate) fn profile_stages_enabled() -> bool {
    false
}

#[cfg(feature = "std")]
pub(crate) type ProfileInstant = Instant;

#[cfg(not(feature = "std"))]
pub(crate) struct ProfileInstant;

#[cfg(feature = "std")]
pub(crate) fn profile_now(enabled: bool) -> Option<ProfileInstant> {
    enabled.then(Instant::now)
}

#[cfg(not(feature = "std"))]
pub(crate) fn profile_now(_enabled: bool) -> Option<ProfileInstant> {
    None
}

#[cfg(feature = "std")]
pub(crate) fn elapsed_us(start: Option<ProfileInstant>) -> u128 {
    start.map_or(0, |start| start.elapsed().as_micros())
}

#[cfg(not(feature = "std"))]
pub(crate) fn elapsed_us(_start: Option<ProfileInstant>) -> u128 {
    0
}

#[cfg(feature = "std")]
pub(crate) fn format_profile_row(op: &str, path: &str, fields: &[(&str, u128)]) -> String {
    signinum_profile::format_profile_row_u128("j2k", op, path, fields)
}

#[cfg(feature = "std")]
pub(crate) fn emit_profile_row(op: &str, path: &str, fields: &[(&str, u128)]) {
    match profile_stage_mode() {
        ProfileStageMode::Disabled => {}
        ProfileStageMode::Rows => eprintln!("{}", format_profile_row(op, path, fields)),
        ProfileStageMode::Summary => {
            signinum_profile::emit_profile_row_u128(
                ProfileStageMode::Summary,
                &PROFILE_SUMMARY,
                "j2k",
                op,
                path,
                fields,
            );
        }
    }
}

#[cfg(not(feature = "std"))]
pub(crate) fn emit_profile_row(_op: &str, _path: &str, _fields: &[(&str, u128)]) {}

#[cfg(feature = "std")]
thread_local! {
    static PROFILE_SUMMARY: RefCell<signinum_profile::ProfileSummary> =
        RefCell::new(signinum_profile::ProfileSummary::default());
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;

    #[test]
    fn parses_profile_env_flag_when_any_profile_mode_is_enabled() {
        assert!(parse_profile_env_flag(Some("1")));
        assert!(parse_profile_env_flag(Some("true")));
        assert!(parse_profile_env_flag(Some("summary")));
        assert!(parse_profile_env_flag(Some("aggregate")));
        assert!(!parse_profile_env_flag(None));
        assert!(!parse_profile_env_flag(Some("")));
        assert!(!parse_profile_env_flag(Some("0")));
    }

    #[test]
    fn formats_profile_row_as_compact_key_value_pairs() {
        let row = format_profile_row(
            "encode",
            "cpu",
            &[("deinterleave_us", 10), ("mct_us", 0), ("total_us", 42)],
        );

        assert_eq!(
            row,
            "signinum_profile codec=j2k op=encode path=cpu deinterleave_us=10 mct_us=0 total_us=42"
        );
    }

    #[test]
    fn parses_profile_env_flag_summary_mode() {
        assert_eq!(
            profile_stage_mode_from_value(Some("summary")),
            ProfileStageMode::Summary
        );
        assert_eq!(
            profile_stage_mode_from_value(Some("aggregate")),
            ProfileStageMode::Summary
        );
        assert_eq!(
            profile_stage_mode_from_value(Some("1")),
            ProfileStageMode::Rows
        );
        assert_eq!(
            profile_stage_mode_from_value(Some("0")),
            ProfileStageMode::Disabled
        );
    }

    #[test]
    fn profile_summary_groups_rows_and_averages_timing_fields() {
        let mut summary = ProfileSummary::default();
        summary.record_u128(
            "j2k",
            "encode",
            "cpu",
            &[
                ("deinterleave_us", 10),
                ("dwt_us", 30),
                ("block_encode_us", 60),
                ("total_us", 100),
            ],
        );
        summary.record_u128(
            "j2k",
            "encode",
            "cpu",
            &[
                ("deinterleave_us", 14),
                ("dwt_us", 42),
                ("block_encode_us", 72),
                ("total_us", 128),
            ],
        );

        assert_eq!(
            summary.format_rows(),
            vec![
                "signinum_profile_summary codec=j2k op=encode path=cpu count=2 block_encode_us_sum=132 block_encode_us_avg=66 deinterleave_us_sum=24 deinterleave_us_avg=12 dwt_us_sum=72 dwt_us_avg=36 total_us_sum=228 total_us_avg=114"
            ]
        );
    }
}
