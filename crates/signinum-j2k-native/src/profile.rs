#[cfg(feature = "std")]
use alloc::collections::BTreeMap;
#[cfg(feature = "std")]
use alloc::string::String;
#[cfg(feature = "std")]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use core::cell::RefCell;
#[cfg(feature = "std")]
use core::fmt::Write as _;

#[cfg(feature = "std")]
use std::sync::OnceLock;
#[cfg(feature = "std")]
use std::time::Instant;

#[cfg(feature = "std")]
const PROFILE_ENV_VAR: &str = "SIGNINUM_J2K_PROFILE_STAGES";

#[cfg(feature = "std")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ProfileStageMode {
    Off,
    Rows,
    Summary,
}

#[cfg(feature = "std")]
#[cfg(test)]
pub(crate) fn parse_profile_env_flag(value: Option<&str>) -> bool {
    profile_stage_mode_from_value(value) != ProfileStageMode::Off
}

#[cfg(feature = "std")]
pub(crate) fn profile_stage_mode_from_value(value: Option<&str>) -> ProfileStageMode {
    match value.map(str::trim).map(str::to_ascii_lowercase).as_deref() {
        Some("1" | "true" | "yes" | "on") => ProfileStageMode::Rows,
        Some("summary" | "aggregate") => ProfileStageMode::Summary,
        _ => ProfileStageMode::Off,
    }
}

#[cfg(feature = "std")]
pub(crate) fn profile_stages_enabled() -> bool {
    profile_stage_mode() != ProfileStageMode::Off
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
    let mut row = String::from("signinum_profile codec=j2k");
    let _ = write!(row, " op={op} path={path}");
    for (key, value) in fields {
        let _ = write!(row, " {key}={value}");
    }
    row
}

#[cfg(feature = "std")]
pub(crate) fn emit_profile_row(op: &str, path: &str, fields: &[(&str, u128)]) {
    match profile_stage_mode() {
        ProfileStageMode::Rows => eprintln!("{}", format_profile_row(op, path, fields)),
        ProfileStageMode::Summary => {
            PROFILE_SUMMARY.with(|summary| {
                summary.borrow_mut().record_u128("j2k", op, path, fields);
            });
        }
        ProfileStageMode::Off => {}
    }
}

#[cfg(not(feature = "std"))]
pub(crate) fn emit_profile_row(_op: &str, _path: &str, _fields: &[(&str, u128)]) {}

#[cfg(feature = "std")]
thread_local! {
    static PROFILE_SUMMARY: RefCell<ProfileSummary> = RefCell::new(ProfileSummary::default());
}

#[cfg(feature = "std")]
#[derive(Default)]
pub(crate) struct ProfileSummary {
    rows: BTreeMap<SummaryKey, SummaryRow>,
}

#[cfg(feature = "std")]
impl ProfileSummary {
    pub(crate) fn record_u128(
        &mut self,
        codec: &str,
        op: &str,
        path: &str,
        fields: &[(&str, u128)],
    ) {
        let key = SummaryKey::new(codec, op, path);
        let row = self.rows.entry(key).or_default();
        row.count += 1;
        for (field, value) in fields {
            if is_timing_field(field) {
                *row.metrics.entry((*field).to_owned()).or_default() += value;
            }
        }
    }

    pub(crate) fn format_rows(&self) -> Vec<String> {
        self.rows
            .iter()
            .map(|(key, row)| key.format_summary_row(row))
            .collect()
    }
}

#[cfg(feature = "std")]
impl Drop for ProfileSummary {
    fn drop(&mut self) {
        for row in self.format_rows() {
            eprintln!("{row}");
        }
    }
}

#[cfg(feature = "std")]
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct SummaryKey {
    codec: String,
    op: String,
    path: String,
}

#[cfg(feature = "std")]
impl SummaryKey {
    fn new(codec: &str, op: &str, path: &str) -> Self {
        Self {
            codec: codec.to_owned(),
            op: op.to_owned(),
            path: path.to_owned(),
        }
    }

    fn format_summary_row(&self, row: &SummaryRow) -> String {
        let mut output = String::from("signinum_profile_summary");
        let _ = write!(
            output,
            " codec={} op={} path={}",
            self.codec, self.op, self.path
        );
        let _ = write!(output, " count={}", row.count);
        for (metric, sum) in &row.metrics {
            let average = if row.count == 0 { 0 } else { sum / row.count };
            let _ = write!(output, " {metric}_sum={sum} {metric}_avg={average}");
        }
        output
    }
}

#[cfg(feature = "std")]
#[derive(Default)]
struct SummaryRow {
    count: u128,
    metrics: BTreeMap<String, u128>,
}

#[cfg(feature = "std")]
fn is_timing_field(field: &str) -> bool {
    field.ends_with("_us") || field.ends_with("_ns")
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
            ProfileStageMode::Off
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
