// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::{
    checked_buffer_slice, encode_status_error, Error, J2kClassicTier1DensityCounters,
    J2kClassicTier1PassPlanCounters, J2kClassicTier1SymbolPlanCounters,
    J2kResidentClassicTier1DensityReadback, J2kResidentClassicTier1PassPlanReadback,
    J2kResidentClassicTier1SymbolPlanReadback, J2kResidentEncodeStageStats, J2K_ENCODE_STATUS_OK,
};

#[cfg(target_os = "macos")]
pub(in crate::compute) fn record_classic_tier1_density_counters(
    stage_stats: &mut J2kResidentEncodeStageStats,
    readback: &J2kResidentClassicTier1DensityReadback,
) -> Result<(), Error> {
    let counters = checked_buffer_slice::<J2kClassicTier1DensityCounters>(
        &readback.buffer,
        readback.count,
        "classic Tier-1 density counters",
    )?;
    for counter in counters {
        stage_stats.tier1_sigprop_active_candidate_count_total = stage_stats
            .tier1_sigprop_active_candidate_count_total
            .saturating_add(
                usize::try_from(counter.sigprop_active_candidates).map_err(|_| {
                    Error::MetalKernel {
                        message: "J2K Metal classic Tier-1 sigprop candidate count exceeds usize"
                            .to_string(),
                    }
                })?,
            );
        stage_stats.tier1_sigprop_new_significant_count_total = stage_stats
            .tier1_sigprop_new_significant_count_total
            .saturating_add(
                usize::try_from(counter.sigprop_new_significant).map_err(|_| {
                    Error::MetalKernel {
                        message:
                            "J2K Metal classic Tier-1 sigprop significance count exceeds usize"
                                .to_string(),
                    }
                })?,
            );
        stage_stats.tier1_magref_active_candidate_count_total = stage_stats
            .tier1_magref_active_candidate_count_total
            .saturating_add(
                usize::try_from(counter.magref_active_candidates).map_err(|_| {
                    Error::MetalKernel {
                        message: "J2K Metal classic Tier-1 magref candidate count exceeds usize"
                            .to_string(),
                    }
                })?,
            );
        stage_stats.tier1_arithmetic_sigprop_active_candidate_count_total = stage_stats
            .tier1_arithmetic_sigprop_active_candidate_count_total
            .saturating_add(
                usize::try_from(counter.arithmetic_sigprop_active_candidates).map_err(|_| {
                    Error::MetalKernel {
                        message:
                            "J2K Metal classic Tier-1 arithmetic sigprop candidate count exceeds usize"
                                .to_string(),
                    }
                })?,
            );
        stage_stats.tier1_arithmetic_sigprop_new_significant_count_total = stage_stats
            .tier1_arithmetic_sigprop_new_significant_count_total
            .saturating_add(
                usize::try_from(counter.arithmetic_sigprop_new_significant).map_err(|_| {
                    Error::MetalKernel {
                        message:
                            "J2K Metal classic Tier-1 arithmetic sigprop significance count exceeds usize"
                                .to_string(),
                    }
                })?,
            );
        stage_stats.tier1_raw_sigprop_active_candidate_count_total = stage_stats
            .tier1_raw_sigprop_active_candidate_count_total
            .saturating_add(
                usize::try_from(counter.raw_sigprop_active_candidates).map_err(|_| {
                    Error::MetalKernel {
                        message:
                            "J2K Metal classic Tier-1 raw sigprop candidate count exceeds usize"
                                .to_string(),
                    }
                })?,
            );
        stage_stats.tier1_raw_sigprop_new_significant_count_total = stage_stats
            .tier1_raw_sigprop_new_significant_count_total
            .saturating_add(
                usize::try_from(counter.raw_sigprop_new_significant).map_err(|_| {
                    Error::MetalKernel {
                        message:
                            "J2K Metal classic Tier-1 raw sigprop significance count exceeds usize"
                                .to_string(),
                    }
                })?,
            );
        stage_stats.tier1_arithmetic_magref_active_candidate_count_total = stage_stats
            .tier1_arithmetic_magref_active_candidate_count_total
            .saturating_add(
                usize::try_from(counter.arithmetic_magref_active_candidates).map_err(|_| {
                    Error::MetalKernel {
                        message:
                            "J2K Metal classic Tier-1 arithmetic magref candidate count exceeds usize"
                                .to_string(),
                    }
                })?,
            );
        stage_stats.tier1_raw_magref_active_candidate_count_total = stage_stats
            .tier1_raw_magref_active_candidate_count_total
            .saturating_add(
                usize::try_from(counter.raw_magref_active_candidates).map_err(|_| {
                    Error::MetalKernel {
                        message:
                            "J2K Metal classic Tier-1 raw magref candidate count exceeds usize"
                                .to_string(),
                    }
                })?,
            );
        stage_stats.tier1_cleanup_active_candidate_count_total = stage_stats
            .tier1_cleanup_active_candidate_count_total
            .saturating_add(
                usize::try_from(counter.cleanup_active_candidates).map_err(|_| {
                    Error::MetalKernel {
                        message: "J2K Metal classic Tier-1 cleanup candidate count exceeds usize"
                            .to_string(),
                    }
                })?,
            );
        stage_stats.tier1_cleanup_new_significant_count_total = stage_stats
            .tier1_cleanup_new_significant_count_total
            .saturating_add(
                usize::try_from(counter.cleanup_new_significant).map_err(|_| {
                    Error::MetalKernel {
                        message:
                            "J2K Metal classic Tier-1 cleanup significance count exceeds usize"
                                .to_string(),
                    }
                })?,
            );
        stage_stats.tier1_cleanup_rlc_stripe_count_total = stage_stats
            .tier1_cleanup_rlc_stripe_count_total
            .saturating_add(usize::try_from(counter.cleanup_rlc_stripes).map_err(|_| {
                Error::MetalKernel {
                    message: "J2K Metal classic Tier-1 cleanup RLC stripe count exceeds usize"
                        .to_string(),
                }
            })?);
        stage_stats.tier1_cleanup_rlc_zero_stripe_count_total = stage_stats
            .tier1_cleanup_rlc_zero_stripe_count_total
            .saturating_add(
                usize::try_from(counter.cleanup_rlc_zero_stripes).map_err(|_| {
                    Error::MetalKernel {
                        message:
                            "J2K Metal classic Tier-1 cleanup zero-RLC stripe count exceeds usize"
                                .to_string(),
                    }
                })?,
            );
    }
    Ok(())
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn record_classic_tier1_symbol_plan_counters(
    stage_stats: &mut J2kResidentEncodeStageStats,
    readback: &J2kResidentClassicTier1SymbolPlanReadback,
) -> Result<(), Error> {
    let counters = checked_buffer_slice::<J2kClassicTier1SymbolPlanCounters>(
        &readback.buffer,
        readback.count,
        "classic Tier-1 symbol-plan counters",
    )?;
    for counter in counters {
        if counter.code != J2K_ENCODE_STATUS_OK {
            return Err(encode_status_error(
                "classic Tier-1 symbol plan",
                counter.code,
                counter.detail,
            ));
        }
        stage_stats.tier1_symbol_plan_mq_symbol_count_total = stage_stats
            .tier1_symbol_plan_mq_symbol_count_total
            .saturating_add(usize::try_from(counter.mq_symbol_count).map_err(|_| {
                Error::MetalKernel {
                    message: "J2K Metal classic Tier-1 symbol-plan MQ count exceeds usize"
                        .to_string(),
                }
            })?);
        stage_stats.tier1_symbol_plan_raw_bit_count_total = stage_stats
            .tier1_symbol_plan_raw_bit_count_total
            .saturating_add(usize::try_from(counter.raw_bit_count).map_err(|_| {
                Error::MetalKernel {
                    message: "J2K Metal classic Tier-1 symbol-plan raw bit count exceeds usize"
                        .to_string(),
                }
            })?);
        let mq_symbol_count =
            usize::try_from(counter.mq_symbol_count).map_err(|_| Error::MetalKernel {
                message: "J2K Metal classic Tier-1 symbol-plan MQ count exceeds usize".to_string(),
            })?;
        let raw_bit_count =
            usize::try_from(counter.raw_bit_count).map_err(|_| Error::MetalKernel {
                message: "J2K Metal classic Tier-1 symbol-plan raw bit count exceeds usize"
                    .to_string(),
            })?;
        stage_stats.max_tier1_symbol_plan_mq_symbols_per_block = stage_stats
            .max_tier1_symbol_plan_mq_symbols_per_block
            .max(mq_symbol_count);
        stage_stats.max_tier1_symbol_plan_raw_bits_per_block = stage_stats
            .max_tier1_symbol_plan_raw_bits_per_block
            .max(raw_bit_count);
        let mq_packed_bytes = mq_symbol_count
            .saturating_mul(6)
            .saturating_add(7)
            .checked_div(8)
            .unwrap_or(usize::MAX);
        let raw_packed_bytes = raw_bit_count
            .saturating_add(7)
            .checked_div(8)
            .unwrap_or(usize::MAX);
        let packed_token_bytes = mq_packed_bytes.saturating_add(raw_packed_bytes);
        stage_stats.tier1_symbol_plan_packed_token_bytes_total = stage_stats
            .tier1_symbol_plan_packed_token_bytes_total
            .saturating_add(packed_token_bytes);
        stage_stats.max_tier1_symbol_plan_packed_token_bytes_per_block = stage_stats
            .max_tier1_symbol_plan_packed_token_bytes_per_block
            .max(packed_token_bytes);
        stage_stats.tier1_symbol_plan_cleanup_mq_symbol_count_total = stage_stats
            .tier1_symbol_plan_cleanup_mq_symbol_count_total
            .saturating_add(
                usize::try_from(counter.cleanup_mq_symbol_count).map_err(|_| {
                    Error::MetalKernel {
                        message:
                            "J2K Metal classic Tier-1 symbol-plan cleanup MQ count exceeds usize"
                                .to_string(),
                    }
                })?,
            );
        stage_stats.tier1_symbol_plan_sigprop_mq_symbol_count_total = stage_stats
            .tier1_symbol_plan_sigprop_mq_symbol_count_total
            .saturating_add(
                usize::try_from(counter.sigprop_mq_symbol_count).map_err(|_| {
                    Error::MetalKernel {
                        message:
                            "J2K Metal classic Tier-1 symbol-plan sigprop MQ count exceeds usize"
                                .to_string(),
                    }
                })?,
            );
        stage_stats.tier1_symbol_plan_magref_mq_symbol_count_total = stage_stats
            .tier1_symbol_plan_magref_mq_symbol_count_total
            .saturating_add(
                usize::try_from(counter.magref_mq_symbol_count).map_err(|_| {
                    Error::MetalKernel {
                        message:
                            "J2K Metal classic Tier-1 symbol-plan magref MQ count exceeds usize"
                                .to_string(),
                    }
                })?,
            );
        stage_stats.tier1_symbol_plan_raw_sigprop_bit_count_total = stage_stats
            .tier1_symbol_plan_raw_sigprop_bit_count_total
            .saturating_add(usize::try_from(counter.raw_sigprop_bit_count).map_err(|_| {
                Error::MetalKernel {
                    message:
                        "J2K Metal classic Tier-1 symbol-plan raw sigprop bit count exceeds usize"
                            .to_string(),
                }
            })?);
        stage_stats.tier1_symbol_plan_raw_magref_bit_count_total = stage_stats
            .tier1_symbol_plan_raw_magref_bit_count_total
            .saturating_add(usize::try_from(counter.raw_magref_bit_count).map_err(|_| {
                Error::MetalKernel {
                    message:
                        "J2K Metal classic Tier-1 symbol-plan raw magref bit count exceeds usize"
                            .to_string(),
                }
            })?);
        stage_stats.tier1_symbol_plan_cleanup_sign_symbol_count_total = stage_stats
            .tier1_symbol_plan_cleanup_sign_symbol_count_total
            .saturating_add(
                usize::try_from(counter.cleanup_sign_symbol_count).map_err(|_| {
                    Error::MetalKernel {
                        message:
                            "J2K Metal classic Tier-1 symbol-plan cleanup sign count exceeds usize"
                                .to_string(),
                    }
                })?,
            );
        stage_stats.tier1_symbol_plan_sigprop_sign_symbol_count_total = stage_stats
            .tier1_symbol_plan_sigprop_sign_symbol_count_total
            .saturating_add(
                usize::try_from(counter.sigprop_sign_symbol_count).map_err(|_| {
                    Error::MetalKernel {
                        message:
                            "J2K Metal classic Tier-1 symbol-plan sigprop sign count exceeds usize"
                                .to_string(),
                    }
                })?,
            );
        stage_stats.tier1_symbol_plan_mq_symbol_hash_xor ^= usize::try_from(counter.mq_symbol_hash)
            .map_err(|_| Error::MetalKernel {
                message: "J2K Metal classic Tier-1 symbol-plan MQ hash exceeds usize".to_string(),
            })?;
        stage_stats.tier1_symbol_plan_raw_bit_hash_xor ^= usize::try_from(counter.raw_bit_hash)
            .map_err(|_| Error::MetalKernel {
                message: "J2K Metal classic Tier-1 symbol-plan raw hash exceeds usize".to_string(),
            })?;
    }
    Ok(())
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn record_classic_tier1_pass_plan_counters(
    stage_stats: &mut J2kResidentEncodeStageStats,
    readback: &J2kResidentClassicTier1PassPlanReadback,
) -> Result<(), Error> {
    let counters = checked_buffer_slice::<J2kClassicTier1PassPlanCounters>(
        &readback.buffer,
        readback.count,
        "classic Tier-1 pass-plan counters",
    )?;
    for counter in counters {
        if counter.code != J2K_ENCODE_STATUS_OK {
            return Err(encode_status_error(
                "classic Tier-1 pass plan",
                counter.code,
                counter.detail,
            ));
        }
        let mq_symbol_count =
            usize::try_from(counter.mq_symbol_count).map_err(|_| Error::MetalKernel {
                message: "J2K Metal classic Tier-1 pass-plan MQ count exceeds usize".to_string(),
            })?;
        let raw_bit_count =
            usize::try_from(counter.raw_bit_count).map_err(|_| Error::MetalKernel {
                message: "J2K Metal classic Tier-1 pass-plan raw bit count exceeds usize"
                    .to_string(),
            })?;
        stage_stats.tier1_pass_plan_mq_symbol_count_total = stage_stats
            .tier1_pass_plan_mq_symbol_count_total
            .saturating_add(mq_symbol_count);
        stage_stats.tier1_pass_plan_raw_bit_count_total = stage_stats
            .tier1_pass_plan_raw_bit_count_total
            .saturating_add(raw_bit_count);
        stage_stats.tier1_pass_plan_nonempty_mq_pass_count_total = stage_stats
            .tier1_pass_plan_nonempty_mq_pass_count_total
            .saturating_add(usize::try_from(counter.nonempty_mq_passes).map_err(|_| {
                Error::MetalKernel {
                    message:
                        "J2K Metal classic Tier-1 pass-plan nonempty MQ pass count exceeds usize"
                            .to_string(),
                }
            })?);
        stage_stats.tier1_pass_plan_nonempty_raw_pass_count_total = stage_stats
            .tier1_pass_plan_nonempty_raw_pass_count_total
            .saturating_add(usize::try_from(counter.nonempty_raw_passes).map_err(|_| {
                Error::MetalKernel {
                    message:
                        "J2K Metal classic Tier-1 pass-plan nonempty raw pass count exceeds usize"
                            .to_string(),
                }
            })?);
        stage_stats.max_tier1_pass_plan_mq_symbols_per_pass =
            stage_stats.max_tier1_pass_plan_mq_symbols_per_pass.max(
                usize::try_from(counter.max_mq_symbols_per_pass).map_err(|_| {
                    Error::MetalKernel {
                        message:
                            "J2K Metal classic Tier-1 pass-plan max MQ pass count exceeds usize"
                                .to_string(),
                    }
                })?,
            );
        stage_stats.max_tier1_pass_plan_raw_bits_per_pass =
            stage_stats.max_tier1_pass_plan_raw_bits_per_pass.max(
                usize::try_from(counter.max_raw_bits_per_pass).map_err(|_| Error::MetalKernel {
                    message: "J2K Metal classic Tier-1 pass-plan max raw pass count exceeds usize"
                        .to_string(),
                })?,
            );

        let pass_mq_total = counter.mq_symbols_by_pass.iter().try_fold(
            0usize,
            |acc, &value| -> Result<usize, Error> {
                Ok(acc.saturating_add(
                    usize::try_from(value).map_err(|_| Error::MetalKernel {
                        message: "J2K Metal classic Tier-1 pass-plan MQ pass count exceeds usize"
                            .to_string(),
                    })?,
                ))
            },
        )?;
        let pass_raw_total = counter.raw_bits_by_pass.iter().try_fold(
            0usize,
            |acc, &value| -> Result<usize, Error> {
                Ok(acc.saturating_add(usize::try_from(value).map_err(|_| {
                    Error::MetalKernel {
                        message: "J2K Metal classic Tier-1 pass-plan raw pass count exceeds usize"
                            .to_string(),
                    }
                })?))
            },
        )?;
        if pass_mq_total != mq_symbol_count || pass_raw_total != raw_bit_count {
            return Err(Error::MetalKernel {
                message: "J2K Metal classic Tier-1 pass-plan per-pass totals are inconsistent"
                    .to_string(),
            });
        }
    }
    Ok(())
}
