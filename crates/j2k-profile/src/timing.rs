#[cfg(feature = "std")]
use alloc::string::String;
#[cfg(feature = "std")]
use core::fmt::Write as _;

#[cfg(feature = "std")]
/// Instant type used by profile timing helpers.
pub type ProfileInstant = std::time::Instant;

#[cfg(not(feature = "std"))]
/// Placeholder instant type used when profiling timers are unavailable.
pub struct ProfileInstant;

#[cfg(feature = "std")]
/// Returns an instant only when profiling is enabled.
pub fn profile_now(enabled: bool) -> Option<ProfileInstant> {
    enabled.then(std::time::Instant::now)
}

#[cfg(not(feature = "std"))]
/// Returns no instant when profiling timers are unavailable.
pub fn profile_now(_enabled: bool) -> Option<ProfileInstant> {
    None
}

#[cfg(feature = "std")]
/// Returns elapsed whole microseconds from an optional profile instant.
pub fn elapsed_us(start: Option<ProfileInstant>) -> u128 {
    start.map_or(0, |start| start.elapsed().as_micros())
}

#[cfg(not(feature = "std"))]
/// Returns zero when profiling timers are unavailable.
pub fn elapsed_us(_start: Option<ProfileInstant>) -> u128 {
    0
}

#[cfg(feature = "std")]
/// Formats a duration as whole microseconds.
pub fn duration_us_string(duration: std::time::Duration) -> String {
    let mut value = String::new();
    write!(value, "{}", duration.as_micros()).expect("writing to String failed");
    value
}
