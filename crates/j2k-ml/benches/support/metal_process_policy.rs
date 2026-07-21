// SPDX-License-Identifier: MIT OR Apache-2.0

use super::process_policy::PROCESS_MODE_ENV;

pub(crate) fn ensure_metal_criterion_instrumentation_disabled() -> Result<(), String> {
    const INSTRUMENTATION_FLAGS: &[&str] = &[
        "J2K_PROFILE_STAGES",
        "J2K_METAL_PROFILE_STAGES",
        "J2K_METAL_PROFILE_SIGNPOSTS",
        "J2K_METAL_PROFILE_DECODE_SPLIT_COMMANDS",
        "J2K_METAL_PROFILE_COEFFICIENT_PREP_SPLIT_COMMANDS",
        "J2K_METAL_PROFILE_CLASSIC_TIER1_DENSITY",
        "J2K_METAL_PROFILE_CLASSIC_TIER1_RAW_PACK",
        "J2K_METAL_PROFILE_CLASSIC_TIER1_ARITHMETIC_PACK",
        "J2K_METAL_PROFILE_CLASSIC_TIER1_SYMBOL_PLAN",
        "J2K_METAL_PROFILE_CLASSIC_TIER1_PASS_PLAN",
        "J2K_METAL_PROFILE_CLASSIC_TIER1_TOKEN_EMIT",
        "J2K_METAL_PROFILE_CLASSIC_TIER1_SPLIT_TOKEN_EMIT",
        "J2K_METAL_PROFILE_CLASSIC_TIER1_TOKEN_PACK",
        "MTL_CAPTURE_ENABLED",
    ];
    let enabled = INSTRUMENTATION_FLAGS
        .iter()
        .copied()
        .filter(|name| {
            std::env::var(name)
                .ok()
                .is_some_and(|value| instrumentation_value_enabled(&value))
        })
        .collect::<Vec<_>>();
    if enabled.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "Criterion acceptance measurements require profiling, signposts, and capture to be disabled; unset {} or run {PROCESS_MODE_ENV}=profile",
            enabled.join(", ")
        ))
    }
}

fn instrumentation_value_enabled(value: &str) -> bool {
    !matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "" | "0" | "false" | "off" | "no"
    )
}
