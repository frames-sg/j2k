use crate::ProfileStageMode;

/// Returns whether an optional environment value is a recognized truthy flag.
pub(crate) fn env_flag_from_value(value: Option<&str>) -> bool {
    let Some(value) = value else {
        return false;
    };
    let value = value.trim();

    matches!(value, "1")
        || value.eq_ignore_ascii_case("true")
        || value.eq_ignore_ascii_case("t")
        || value.eq_ignore_ascii_case("yes")
        || value.eq_ignore_ascii_case("y")
        || value.eq_ignore_ascii_case("on")
        || value.eq_ignore_ascii_case("enable")
        || value.eq_ignore_ascii_case("enabled")
}

#[cfg(feature = "std")]
/// Returns whether a named environment variable is a recognized truthy flag.
pub fn env_flag_from_env(key: &str) -> bool {
    env_flag_from_value(std::env::var(key).ok().as_deref())
}

/// Parses a profiling stage mode from an optional environment value.
pub fn profile_stage_mode_from_value(value: Option<&str>) -> ProfileStageMode {
    let Some(value) = value else {
        return ProfileStageMode::Disabled;
    };
    let value = value.trim();

    if value.eq_ignore_ascii_case("summary")
        || value.eq_ignore_ascii_case("summaries")
        || value.eq_ignore_ascii_case("aggregate")
        || value.eq_ignore_ascii_case("aggregates")
    {
        ProfileStageMode::Summary
    } else if env_flag_from_value(Some(value)) {
        ProfileStageMode::Rows
    } else {
        ProfileStageMode::Disabled
    }
}

#[cfg(feature = "std")]
/// Parses a profiling stage mode from a named environment variable.
pub fn profile_stage_mode_from_env(key: &str) -> ProfileStageMode {
    profile_stage_mode_from_value(std::env::var(key).ok().as_deref())
}

#[cfg(feature = "std")]
/// Caches one profiling stage mode parsed from a process environment variable.
#[derive(Debug, Default)]
pub struct StageModeCache {
    mode: std::sync::OnceLock<ProfileStageMode>,
}

#[cfg(feature = "std")]
impl StageModeCache {
    /// Creates an empty stage-mode cache.
    pub const fn new() -> Self {
        Self {
            mode: std::sync::OnceLock::new(),
        }
    }

    /// Returns the cached mode, initializing it from `key` on first use.
    pub fn mode_from_env(&self, key: &str) -> ProfileStageMode {
        *self.mode.get_or_init(|| profile_stage_mode_from_env(key))
    }
}
