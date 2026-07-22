// SPDX-License-Identifier: MIT OR Apache-2.0

use super::abi::J2K_CLASSIC_STYLE_SELECTIVE_ARITHMETIC_CODING_BYPASS;
use super::J2kResidentEncodeStageStats;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct J2kClassicTier1PassClassCounts {
    pub(super) arithmetic: usize,
    pub(super) raw: usize,
    pub(super) cleanup: usize,
    pub(super) sigprop: usize,
    pub(super) magref: usize,
    pub(super) arithmetic_cleanup: usize,
    pub(super) arithmetic_sigprop: usize,
    pub(super) arithmetic_magref: usize,
    pub(super) raw_sigprop: usize,
    pub(super) raw_magref: usize,
}

pub(super) fn classic_tier1_pass_class_counts(
    coding_passes: usize,
    style_flags: u32,
) -> J2kClassicTier1PassClassCounts {
    let selective_bypass =
        (style_flags & J2K_CLASSIC_STYLE_SELECTIVE_ARITHMETIC_CODING_BYPASS) != 0;
    let mut counts = J2kClassicTier1PassClassCounts::default();
    for coding_pass in 0..coding_passes {
        let pass_type = coding_pass % 3;
        let arithmetic = !selective_bypass || coding_pass <= 9 || pass_type == 0;
        match pass_type {
            0 => {
                counts.cleanup = counts.cleanup.saturating_add(1);
                counts.arithmetic_cleanup = counts.arithmetic_cleanup.saturating_add(1);
            }
            1 => {
                counts.sigprop = counts.sigprop.saturating_add(1);
                if arithmetic {
                    counts.arithmetic_sigprop = counts.arithmetic_sigprop.saturating_add(1);
                } else {
                    counts.raw_sigprop = counts.raw_sigprop.saturating_add(1);
                }
            }
            _ => {
                counts.magref = counts.magref.saturating_add(1);
                if arithmetic {
                    counts.arithmetic_magref = counts.arithmetic_magref.saturating_add(1);
                } else {
                    counts.raw_magref = counts.raw_magref.saturating_add(1);
                }
            }
        }
        if arithmetic {
            counts.arithmetic = counts.arithmetic.saturating_add(1);
        } else {
            counts.raw = counts.raw.saturating_add(1);
        }
    }
    counts
}

pub(super) fn accumulate_classic_tier1_scan_estimates(
    stage_stats: &mut J2kResidentEncodeStageStats,
    pass_counts: J2kClassicTier1PassClassCounts,
    coeff_count: usize,
) {
    let full_scan_visits = pass_counts
        .cleanup
        .saturating_add(pass_counts.sigprop)
        .saturating_add(pass_counts.magref)
        .saturating_mul(coeff_count);
    stage_stats.tier1_full_scan_coeff_visit_count_total = stage_stats
        .tier1_full_scan_coeff_visit_count_total
        .saturating_add(full_scan_visits);
    stage_stats.max_tier1_full_scan_coeff_visits_per_block = stage_stats
        .max_tier1_full_scan_coeff_visits_per_block
        .max(full_scan_visits);
    stage_stats.tier1_arithmetic_scan_coeff_visit_count_total = stage_stats
        .tier1_arithmetic_scan_coeff_visit_count_total
        .saturating_add(pass_counts.arithmetic.saturating_mul(coeff_count));
    stage_stats.tier1_raw_scan_coeff_visit_count_total = stage_stats
        .tier1_raw_scan_coeff_visit_count_total
        .saturating_add(pass_counts.raw.saturating_mul(coeff_count));
    stage_stats.tier1_cleanup_scan_coeff_visit_count_total = stage_stats
        .tier1_cleanup_scan_coeff_visit_count_total
        .saturating_add(pass_counts.cleanup.saturating_mul(coeff_count));
    stage_stats.tier1_sigprop_scan_coeff_visit_count_total = stage_stats
        .tier1_sigprop_scan_coeff_visit_count_total
        .saturating_add(pass_counts.sigprop.saturating_mul(coeff_count));
    stage_stats.tier1_magref_scan_coeff_visit_count_total = stage_stats
        .tier1_magref_scan_coeff_visit_count_total
        .saturating_add(pass_counts.magref.saturating_mul(coeff_count));
}
