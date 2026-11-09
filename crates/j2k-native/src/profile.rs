#[cfg(feature = "profile")]
use core::cell::RefCell;

#[cfg(feature = "profile")]
use j2k_profile::{ProfileStageMode, StageModeCache};

pub(crate) use j2k_profile::{elapsed_us, profile_now, ProfileInstant};

#[cfg(all(test, feature = "profile"))]
pub(crate) use j2k_profile::ProfileSummary;

#[cfg(feature = "profile")]
const PROFILE_ENV_VAR: &str = "J2K_PROFILE_STAGES";

#[cfg(feature = "profile")]
#[cfg(test)]
pub(crate) fn parse_profile_env_flag(value: Option<&str>) -> bool {
    j2k_profile::profile_stage_mode_from_value(value) != ProfileStageMode::Disabled
}

#[cfg(feature = "profile")]
pub(crate) fn profile_stages_enabled() -> bool {
    profile_stage_mode() != ProfileStageMode::Disabled
}

#[cfg(feature = "profile")]
fn profile_stage_mode() -> ProfileStageMode {
    static MODE: StageModeCache = StageModeCache::new();
    MODE.mode_from_env(PROFILE_ENV_VAR)
}

#[cfg(not(feature = "profile"))]
pub(crate) fn profile_stages_enabled() -> bool {
    false
}

#[cfg(feature = "profile")]
pub(crate) fn emit_profile_row(op: &str, path: &str, fields: &[(&str, u128)]) {
    j2k_profile::emit_profile_row_u128(
        profile_stage_mode(),
        &PROFILE_SUMMARY,
        "j2k",
        op,
        path,
        fields,
    );
}

#[cfg(not(feature = "profile"))]
pub(crate) fn emit_profile_row(_op: &str, _path: &str, _fields: &[(&str, u128)]) {}

#[cfg(feature = "profile")]
thread_local! {
    static PROFILE_SUMMARY: RefCell<j2k_profile::ProfileSummary> =
        RefCell::new(j2k_profile::ProfileSummary::default().emit_on_drop());
}

/// Run `work` on a worker of the current Rayon pool when the pool has more
/// than one thread and no other call is in flight. Called from outside the
/// pool, every parallel section of the work would wake the workers and put
/// the calling thread to sleep until they finish; on a worker, the sections
/// that follow find the workers awake. Beside other callers, the moved work's
/// waiting worker would take their jobs and hold the moved work until they
/// end, so concurrent callers keep their own threads as before. The calling
/// thread's stage summary moves to the worker for the duration, so the rows
/// the work records aggregate exactly as before.
#[cfg(feature = "parallel")]
pub(crate) fn on_pool_worker<R: Send>(work: impl FnOnce() -> R + Send) -> R {
    use core::sync::atomic::{AtomicUsize, Ordering};
    use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};

    static IN_FLIGHT: AtomicUsize = AtomicUsize::new(0);
    if rayon::current_num_threads() <= 1 || rayon::current_thread_index().is_some() {
        return work();
    }
    let alone = IN_FLIGHT.fetch_add(1, Ordering::AcqRel) == 0;
    let ran = catch_unwind(AssertUnwindSafe(|| {
        if !alone {
            return (work(), None);
        }
        let summary = take_stage_summary();
        rayon::scope(move |_| {
            let worker_summary = replace_stage_summary(summary);
            let result = work();
            (result, Some(replace_stage_summary(worker_summary)))
        })
    }));
    IN_FLIGHT.fetch_sub(1, Ordering::AcqRel);
    let (result, summary) = ran.unwrap_or_else(|panic| resume_unwind(panic));
    if let Some(summary) = summary {
        replace_stage_summary(summary);
    }
    result
}

#[cfg(not(feature = "parallel"))]
pub(crate) fn on_pool_worker<R>(work: impl FnOnce() -> R) -> R {
    work()
}

#[cfg(all(feature = "parallel", feature = "profile"))]
fn take_stage_summary() -> j2k_profile::ProfileSummary {
    PROFILE_SUMMARY.take()
}

#[cfg(all(feature = "parallel", feature = "profile"))]
fn replace_stage_summary(summary: j2k_profile::ProfileSummary) -> j2k_profile::ProfileSummary {
    PROFILE_SUMMARY.replace(summary)
}

#[cfg(all(feature = "parallel", not(feature = "profile")))]
const fn take_stage_summary() {}

#[cfg(all(feature = "parallel", not(feature = "profile")))]
const fn replace_stage_summary((): ()) {}

#[cfg(all(test, feature = "profile"))]
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

    #[cfg(feature = "parallel")]
    #[test]
    fn pooled_work_returns_its_value_and_its_summary_rows_to_the_caller() {
        // Moved to a worker, or run in place on a pool thread or beside other
        // tests' calls, the work's value and rows come back to its caller.
        let run = || {
            let value = on_pool_worker(|| {
                PROFILE_SUMMARY
                    .with_borrow_mut(|summary| {
                        summary.record_u128("j2k", "encode", "cpu", &[("total_us", 7)])
                    })
                    .expect("row should record");
                42
            });
            let rows = PROFILE_SUMMARY
                .with_borrow_mut(ProfileSummary::take_formatted_rows)
                .expect("summary should format");
            (value, rows)
        };
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .expect("test pool should build");
        let expected = (
            42,
            vec![String::from(
                "j2k_profile_summary codec=j2k op=encode path=cpu count=1 total_us_sum=7 total_us_avg=7",
            )],
        );
        for (caller, result) in [("test thread", run()), ("pool thread", pool.install(run))] {
            assert_eq!(result, expected, "{caller}");
        }
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn pooled_work_that_panics_reaches_the_caller_and_later_work_still_runs() {
        let panicked = std::panic::catch_unwind(|| on_pool_worker(|| panic!("pooled work failed")));
        assert!(panicked.is_err());
        assert_eq!(on_pool_worker(|| 7), 7);
    }

    #[test]
    fn profile_summary_groups_rows_and_averages_timing_fields() {
        let mut summary = ProfileSummary::default();
        summary
            .record_u128(
                "j2k",
                "encode",
                "cpu",
                &[
                    ("deinterleave_us", 10),
                    ("dwt_us", 30),
                    ("block_encode_us", 60),
                    ("total_us", 100),
                ],
            )
            .expect("first profile row should record");
        summary
            .record_u128(
                "j2k",
                "encode",
                "cpu",
                &[
                    ("deinterleave_us", 14),
                    ("dwt_us", 42),
                    ("block_encode_us", 72),
                    ("total_us", 128),
                ],
            )
            .expect("second profile row should record");

        assert_eq!(
            summary.format_rows().expect("summary should format"),
            vec![
                "j2k_profile_summary codec=j2k op=encode path=cpu count=2 block_encode_us_sum=132 block_encode_us_avg=66 deinterleave_us_sum=24 deinterleave_us_avg=12 dwt_us_sum=72 dwt_us_avg=36 total_us_sum=228 total_us_avg=114"
            ]
        );
    }
}
