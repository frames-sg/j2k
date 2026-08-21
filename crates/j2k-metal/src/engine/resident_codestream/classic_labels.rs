// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(target_os = "macos")]
pub(super) const CLASSIC_TIER1_DENSITY_LABEL: &str = "j2k classic resident Tier-1 density profile";
#[cfg(target_os = "macos")]
pub(super) const CLASSIC_TIER1_RAW_PACK_LABEL: &str =
    "j2k classic resident Tier-1 raw-pack profile";
#[cfg(target_os = "macos")]
pub(super) const CLASSIC_TIER1_ARITHMETIC_PACK_LABEL: &str =
    "j2k classic resident Tier-1 arithmetic-pack profile";
#[cfg(target_os = "macos")]
pub(super) const CLASSIC_TIER1_SYMBOL_PLAN_LABEL: &str = "j2k classic resident Tier-1 symbol plan";
#[cfg(target_os = "macos")]
pub(super) const CLASSIC_TIER1_PASS_PLAN_LABEL: &str = "j2k classic resident Tier-1 pass plan";
#[cfg(target_os = "macos")]
pub(super) const CLASSIC_TIER1_TOKEN_EMIT_LABEL: &str = "j2k classic resident Tier-1 token emit";
#[cfg(target_os = "macos")]
pub(super) const CLASSIC_TIER1_SPLIT_TOKEN_EMIT_LABEL: &str =
    "j2k classic resident Tier-1 split token emit";
#[cfg(target_os = "macos")]
pub(super) const CLASSIC_RESIDENT_PACKETIZATION_LABEL: &str = "j2k classic resident packetization";

/// Label of the first enabled downstream classic Tier-1 profiling stage.
#[cfg(target_os = "macos")]
pub(super) fn next_enabled_classic_stage_label(
    candidates: &[(bool, &'static str)],
) -> &'static str {
    candidates
        .iter()
        .find_map(|&(enabled, label)| enabled.then_some(label))
        .unwrap_or(CLASSIC_RESIDENT_PACKETIZATION_LABEL)
}
