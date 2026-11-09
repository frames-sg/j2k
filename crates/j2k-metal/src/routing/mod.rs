// SPDX-License-Identifier: MIT OR Apache-2.0

mod availability;
mod decision;
mod eligibility;
#[cfg(any(test, target_os = "macos"))]
pub(crate) mod promotion;
mod rejection;
mod telemetry;

#[cfg(target_os = "macos")]
pub(crate) use decision::precision_rejection;
pub(crate) use decision::{
    decide_route_for_precision, decision_error, exceeds_metal_decode_precision, RouteDecision,
    MAX_METAL_DECODE_PRECISION,
};
#[cfg(any(test, target_os = "macos"))]
pub(crate) use promotion::{
    auto_full_decode_uses_metal, auto_repeated_decode_uses_metal, auto_scaled_decode_uses_metal,
};

pub(crate) const AUTO_DECODE_CPU_FALLBACK_REASON: &str =
    "J2K Metal Auto decode stays on CPU until decode benchmark evidence justifies Metal routing";

#[cfg(test)]
mod tests;
