// SPDX-License-Identifier: MIT OR Apache-2.0

use std::time::Duration;

#[cfg(target_os = "macos")]
use crate::compute;

use super::MetalLosslessBufferEncodeOutcome;

/// Optional resident Metal encode stage timings.
///
/// API note: this diagnostic report is constructed by this crate. It is not
/// `#[non_exhaustive]`, but adapter releases may add diagnostic fields as the
/// resident encode path gains more profiling detail.
///
/// Unless a field explicitly says otherwise, timing fields are host-side
/// `Instant` buckets for RCA and should not be read as exact GPU execution
/// elapsed time.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MetalLosslessEncodeStageStats {
    /// Time spent planning the resident encode batch.
    pub plan_duration: Duration,
    /// Time spent preparing and submitting Metal work.
    pub prepare_submit_duration: Duration,
    /// Host-side wall time spent preparing resident encode coefficients.
    pub coefficient_prep_duration: Duration,
    /// Reserved for future finer-grained deinterleave plus RCT profiling.
    ///
    /// Current resident prep timing is reported in `coefficient_prep_duration`.
    pub deinterleave_rct_duration: Duration,
    /// Reserved for future finer-grained forward 5/3 DWT profiling.
    ///
    /// Current resident prep timing is reported in `coefficient_prep_duration`.
    pub dwt53_duration: Duration,
    /// Reserved for future finer-grained coefficient extraction profiling.
    ///
    /// Current resident prep timing is reported in `coefficient_prep_duration`.
    pub coefficient_extract_duration: Duration,
    /// Time spent building HT lookup tables.
    pub ht_table_build_duration: Duration,
    /// Time spent allocating HT output buffers.
    pub ht_buffer_allocation_duration: Duration,
    /// Host-side Metal command encoding time for HT resident command buffers.
    ///
    /// This is the sum of the split command-encode buckets below and is not GPU
    /// kernel execution elapsed time.
    pub ht_command_encode_duration: Duration,
    /// Host-side Metal command encoding time for HT code-block dispatch setup.
    pub ht_block_encode_duration: Duration,
    /// CPU-side setup time for classic Tier-1 batch jobs and buffers.
    pub classic_tier1_setup_duration: Duration,
    /// Host-side Metal command encoding time for classic code-block dispatch setup.
    pub classic_block_encode_duration: Duration,
    /// Host-side CPU time spent packing compact classic Tier-1 tokens.
    ///
    /// This is populated only when
    /// `J2K_METAL_PROFILE_CLASSIC_TIER1_TOKEN_PACK=1` is enabled.
    pub classic_tier1_token_pack_duration: Duration,
    /// CPU-side packet metadata planning time for classic resident batches.
    pub classic_packet_plan_duration: Duration,
    /// CPU-side packet/codestream buffer setup time for classic resident batches.
    pub classic_packet_buffer_setup_duration: Duration,
    /// Host-side time spent committing split classic resident command buffers.
    pub classic_command_buffer_commit_duration: Duration,
    /// Host-side wall time spent harvesting completed resident batch results.
    pub result_harvest_duration: Duration,
    /// Host-side time spent copying shared status buffers into CPU-owned status arrays.
    pub result_status_copy_duration: Duration,
    /// Host-side time spent returning private buffers to the resident buffer pool.
    pub result_private_recycle_duration: Duration,
    /// Host-side time spent returning shared buffers to the resident buffer pool.
    pub result_shared_recycle_duration: Duration,
    /// Host-side time spent validating per-tile status and building codestream handles.
    pub result_codestream_collect_duration: Duration,
    /// Host-side Metal command encoding time for packet block metadata dispatch setup.
    pub packet_block_prep_duration: Duration,
    /// Host-side Metal command encoding time for packet body dispatch setup.
    pub packetization_duration: Duration,
    /// Host-side Metal command encoding time for codestream assembly dispatch setup.
    pub codestream_assembly_duration: Duration,
    /// GPU time spent preparing resident coefficient buffers.
    ///
    /// This includes the resident input deinterleave/RCT, DWT, and coefficient
    /// extraction command buffer when stage profiling is enabled.
    pub coefficient_prep_gpu_duration: Duration,
    /// GPU time spent deinterleaving resident input planes and applying RCT.
    ///
    /// This is populated only when resident coefficient-prep split profiling is enabled.
    pub coefficient_deinterleave_rct_gpu_duration: Duration,
    /// GPU time spent running resident forward DWT 5/3 coefficient prep.
    ///
    /// This is populated only when resident coefficient-prep split profiling is enabled.
    pub coefficient_dwt53_gpu_duration: Duration,
    /// GPU time spent in resident forward DWT 5/3 vertical passes.
    ///
    /// This is populated only when resident coefficient-prep split profiling is enabled.
    pub coefficient_dwt53_vertical_gpu_duration: Duration,
    /// GPU time spent in resident forward DWT 5/3 horizontal passes.
    ///
    /// This is populated only when resident coefficient-prep split profiling is enabled.
    pub coefficient_dwt53_horizontal_gpu_duration: Duration,
    /// GPU time spent extracting resident code-block coefficients.
    ///
    /// This is populated only when resident coefficient-prep split profiling is enabled.
    pub coefficient_extract_gpu_duration: Duration,
    /// GPU time spent copying per-tile coefficient buffers into a batch buffer.
    ///
    /// This is populated only when resident split-command profiling is enabled.
    pub coefficient_copy_gpu_duration: Duration,
    /// Elapsed GPU timestamp window across the resident encode command buffers.
    ///
    /// This is `max(GPUEndTime) - min(GPUStartTime)` for the command buffers
    /// retained by the batch. It is a wall-window companion to summed GPU busy
    /// rows and should not be added to per-stage GPU durations.
    pub gpu_elapsed_wall_duration: Duration,
    /// GPU time spent in the classic Tier-1 code-block encode command.
    ///
    /// This is populated only when classic split-command profiling is enabled.
    pub classic_block_gpu_duration: Duration,
    /// GPU time spent in the profile-only classic Tier-1 density probe.
    ///
    /// This is populated only when classic split-command profiling and
    /// `J2K_METAL_PROFILE_CLASSIC_TIER1_DENSITY=1` are enabled.
    pub classic_tier1_density_gpu_duration: Duration,
    /// GPU time spent in the profile-only classic Tier-1 raw bypass packing probe.
    ///
    /// This is populated only when classic split-command profiling and
    /// `J2K_METAL_PROFILE_CLASSIC_TIER1_RAW_PACK=1` are enabled.
    pub classic_tier1_raw_pack_gpu_duration: Duration,
    /// GPU time spent in the profile-only classic Tier-1 MQ arithmetic packing probe.
    ///
    /// This is populated only when classic split-command profiling and
    /// `J2K_METAL_PROFILE_CLASSIC_TIER1_ARITHMETIC_PACK=1` are enabled.
    pub classic_tier1_arithmetic_pack_gpu_duration: Duration,
    /// GPU time spent in the profile-only classic Tier-1 ordered symbol-plan probe.
    ///
    /// This is populated only when classic split-command profiling and
    /// `J2K_METAL_PROFILE_CLASSIC_TIER1_SYMBOL_PLAN=1` are enabled.
    pub classic_tier1_symbol_plan_gpu_duration: Duration,
    /// GPU time spent in the profile-only classic Tier-1 pass-plan probe.
    ///
    /// This is populated only when classic split-command profiling and
    /// `J2K_METAL_PROFILE_CLASSIC_TIER1_PASS_PLAN=1` are enabled.
    pub classic_tier1_pass_plan_gpu_duration: Duration,
    /// GPU time spent in the profile-only classic Tier-1 compact token-emitter probe.
    ///
    /// This is populated only when classic split-command profiling and
    /// `J2K_METAL_PROFILE_CLASSIC_TIER1_TOKEN_EMIT=1` are enabled.
    pub classic_tier1_token_emit_gpu_duration: Duration,
    /// GPU time spent in the profile-only classic Tier-1 split MQ/raw token-emitter probe.
    ///
    /// This is populated only when classic split-command profiling and
    /// `J2K_METAL_PROFILE_CLASSIC_TIER1_SPLIT_TOKEN_EMIT=1` are enabled.
    pub classic_tier1_split_token_emit_gpu_duration: Duration,
    /// GPU time spent packing compact classic Tier-1 tokens into resident payloads.
    ///
    /// This is populated when the gated classic GPU token-pack route is enabled.
    pub classic_tier1_token_pack_gpu_duration: Duration,
    /// GPU time spent in the HT Tier-1 code-block encode command.
    ///
    /// This is populated only when HT split-command profiling is enabled.
    pub ht_block_gpu_duration: Duration,
    /// GPU time spent preparing packet-block metadata from HT Tier-1 status.
    ///
    /// This is populated only when HT split-command profiling is enabled.
    pub packet_block_prep_gpu_duration: Duration,
    /// GPU time spent in HTJ2K packetization.
    ///
    /// This is populated only when HT split-command profiling is enabled.
    pub packetization_gpu_duration: Duration,
    /// GPU time spent copying packet payload bytes after header packetization.
    ///
    /// This is populated only when HT split-command profiling is enabled.
    pub packet_payload_copy_gpu_duration: Duration,
    /// GPU time spent assembling the HTJ2K codestream buffer.
    ///
    /// This is populated only when HT split-command profiling is enabled.
    pub codestream_assembly_gpu_duration: Duration,
    /// GPU time spent copying packet payload bytes into final codestream buffers.
    ///
    /// This is populated only when HT split-command profiling is enabled.
    pub codestream_payload_copy_gpu_duration: Duration,
    /// Total Tier-1 output capacity, in bytes, across resident code blocks.
    pub tier1_output_capacity_total: usize,
    /// Maximum Tier-1 output capacity, in bytes, for any resident code block.
    pub max_tier1_output_capacity: usize,
    /// Actual Tier-1 output bytes written across resident code blocks.
    pub tier1_output_used_bytes_total: usize,
    /// Maximum actual Tier-1 output bytes written by any resident code block.
    pub max_tier1_output_used_bytes: usize,
    /// Total Tier-1 segment metadata capacity across resident code blocks.
    pub tier1_segment_capacity_total: usize,
    /// Maximum Tier-1 segment metadata capacity for any resident code block.
    pub max_tier1_segment_capacity_per_block: usize,
    /// Actual Tier-1 coding passes emitted across resident code blocks.
    pub tier1_coding_pass_count_total: usize,
    /// Maximum actual Tier-1 coding passes emitted by any resident code block.
    pub max_tier1_coding_passes_per_block: usize,
    /// Estimated classic MQ/arithmetic coding passes across resident code blocks.
    ///
    /// For HTJ2K Tier-1 this remains zero.
    pub tier1_arithmetic_pass_count_total: usize,
    /// Estimated classic raw bypass coding passes across resident code blocks.
    ///
    /// For HTJ2K Tier-1 this remains zero.
    pub tier1_raw_pass_count_total: usize,
    /// Estimated classic cleanup passes across resident code blocks.
    ///
    /// For HTJ2K Tier-1 this remains zero.
    pub tier1_cleanup_pass_count_total: usize,
    /// Estimated classic significance propagation passes across resident code blocks.
    ///
    /// For HTJ2K Tier-1 this remains zero.
    pub tier1_sigprop_pass_count_total: usize,
    /// Estimated classic magnitude refinement passes across resident code blocks.
    ///
    /// For HTJ2K Tier-1 this remains zero.
    pub tier1_magref_pass_count_total: usize,
    /// Estimated classic MQ/arithmetic cleanup passes across resident code blocks.
    ///
    /// For HTJ2K Tier-1 this remains zero.
    pub tier1_arithmetic_cleanup_pass_count_total: usize,
    /// Estimated classic MQ/arithmetic significance propagation passes.
    ///
    /// For HTJ2K Tier-1 this remains zero.
    pub tier1_arithmetic_sigprop_pass_count_total: usize,
    /// Estimated classic MQ/arithmetic magnitude refinement passes.
    ///
    /// For HTJ2K Tier-1 this remains zero.
    pub tier1_arithmetic_magref_pass_count_total: usize,
    /// Estimated classic raw bypass significance propagation passes.
    ///
    /// For HTJ2K Tier-1 this remains zero.
    pub tier1_raw_sigprop_pass_count_total: usize,
    /// Estimated classic raw bypass magnitude refinement passes.
    ///
    /// For HTJ2K Tier-1 this remains zero.
    pub tier1_raw_magref_pass_count_total: usize,
    /// Estimated full coefficient visits made by classic Tier-1 pass scans.
    ///
    /// This is derived from actual emitted pass counts and code-block areas.
    /// For HTJ2K Tier-1 this remains zero.
    pub tier1_full_scan_coeff_visit_count_total: usize,
    /// Estimated full coefficient visits made by MQ/arithmetic pass scans.
    ///
    /// For HTJ2K Tier-1 this remains zero.
    pub tier1_arithmetic_scan_coeff_visit_count_total: usize,
    /// Estimated full coefficient visits made by raw bypass pass scans.
    ///
    /// For HTJ2K Tier-1 this remains zero.
    pub tier1_raw_scan_coeff_visit_count_total: usize,
    /// Estimated full coefficient visits made by cleanup pass scans.
    ///
    /// For HTJ2K Tier-1 this remains zero.
    pub tier1_cleanup_scan_coeff_visit_count_total: usize,
    /// Estimated full coefficient visits made by significance propagation scans.
    ///
    /// For HTJ2K Tier-1 this remains zero.
    pub tier1_sigprop_scan_coeff_visit_count_total: usize,
    /// Estimated full coefficient visits made by magnitude refinement scans.
    ///
    /// For HTJ2K Tier-1 this remains zero.
    pub tier1_magref_scan_coeff_visit_count_total: usize,
    /// Maximum estimated full coefficient scan visits for any classic block.
    ///
    /// For HTJ2K Tier-1 this remains zero.
    pub max_tier1_full_scan_coeff_visits_per_block: usize,
    /// Profile-only count of classic significance propagation candidates.
    ///
    /// This is populated only when classic Tier-1 density profiling is enabled.
    pub tier1_sigprop_active_candidate_count_total: usize,
    /// Profile-only count of coefficients that become significant in sigprop.
    ///
    /// This is populated only when classic Tier-1 density profiling is enabled.
    pub tier1_sigprop_new_significant_count_total: usize,
    /// Profile-only count of classic magnitude refinement candidates.
    ///
    /// This is populated only when classic Tier-1 density profiling is enabled.
    pub tier1_magref_active_candidate_count_total: usize,
    /// Profile-only count of arithmetic-coded significance propagation candidates.
    pub tier1_arithmetic_sigprop_active_candidate_count_total: usize,
    /// Profile-only count of coefficients that become significant in arithmetic sigprop.
    pub tier1_arithmetic_sigprop_new_significant_count_total: usize,
    /// Profile-only count of raw bypass significance propagation candidates.
    pub tier1_raw_sigprop_active_candidate_count_total: usize,
    /// Profile-only count of coefficients that become significant in raw sigprop.
    pub tier1_raw_sigprop_new_significant_count_total: usize,
    /// Profile-only count of arithmetic-coded magnitude refinement candidates.
    pub tier1_arithmetic_magref_active_candidate_count_total: usize,
    /// Profile-only count of raw bypass magnitude refinement candidates.
    pub tier1_raw_magref_active_candidate_count_total: usize,
    /// Profile-only count of cleanup-pass coefficient candidates.
    ///
    /// This excludes coefficients represented only by cleanup RLC stripes.
    pub tier1_cleanup_active_candidate_count_total: usize,
    /// Profile-only count of coefficients that become significant in cleanup.
    ///
    /// This includes significance discovered through cleanup RLC.
    pub tier1_cleanup_new_significant_count_total: usize,
    /// Profile-only count of cleanup stripes encoded by the RLC path.
    pub tier1_cleanup_rlc_stripe_count_total: usize,
    /// Profile-only count of cleanup RLC stripes with no significant coefficient.
    pub tier1_cleanup_rlc_zero_stripe_count_total: usize,
    /// Profile-only exact MQ symbol count from the ordered symbol-plan probe.
    pub tier1_symbol_plan_mq_symbol_count_total: usize,
    /// Profile-only exact raw bypass bit count from the ordered symbol-plan probe.
    pub tier1_symbol_plan_raw_bit_count_total: usize,
    /// Maximum MQ symbols emitted by any block in the ordered symbol-plan probe.
    pub max_tier1_symbol_plan_mq_symbols_per_block: usize,
    /// Maximum raw bypass bits emitted by any block in the ordered symbol-plan probe.
    pub max_tier1_symbol_plan_raw_bits_per_block: usize,
    /// Estimated compact token bytes needed for all blocks in the symbol-plan probe.
    pub tier1_symbol_plan_packed_token_bytes_total: usize,
    /// Maximum estimated compact token bytes needed by any one block.
    pub max_tier1_symbol_plan_packed_token_bytes_per_block: usize,
    /// Profile-only exact cleanup MQ symbol count from the ordered symbol-plan probe.
    pub tier1_symbol_plan_cleanup_mq_symbol_count_total: usize,
    /// Profile-only exact sigprop MQ symbol count from the ordered symbol-plan probe.
    pub tier1_symbol_plan_sigprop_mq_symbol_count_total: usize,
    /// Profile-only exact magref MQ symbol count from the ordered symbol-plan probe.
    pub tier1_symbol_plan_magref_mq_symbol_count_total: usize,
    /// Profile-only exact raw sigprop bit count from the ordered symbol-plan probe.
    pub tier1_symbol_plan_raw_sigprop_bit_count_total: usize,
    /// Profile-only exact raw magref bit count from the ordered symbol-plan probe.
    pub tier1_symbol_plan_raw_magref_bit_count_total: usize,
    /// Profile-only cleanup sign-symbol count from the ordered symbol-plan probe.
    pub tier1_symbol_plan_cleanup_sign_symbol_count_total: usize,
    /// Profile-only sigprop sign-symbol count from the ordered symbol-plan probe.
    pub tier1_symbol_plan_sigprop_sign_symbol_count_total: usize,
    /// XOR of per-block order-sensitive MQ symbol hashes from the symbol-plan probe.
    pub tier1_symbol_plan_mq_symbol_hash_xor: usize,
    /// XOR of per-block order-sensitive raw bit hashes from the symbol-plan probe.
    pub tier1_symbol_plan_raw_bit_hash_xor: usize,
    /// Profile-only MQ symbols counted by coding-pass index.
    pub tier1_pass_plan_mq_symbol_count_total: usize,
    /// Profile-only raw bypass bits counted by coding-pass index.
    pub tier1_pass_plan_raw_bit_count_total: usize,
    /// Count of block-local coding passes that emit at least one MQ symbol.
    pub tier1_pass_plan_nonempty_mq_pass_count_total: usize,
    /// Count of block-local coding passes that emit at least one raw bypass bit.
    pub tier1_pass_plan_nonempty_raw_pass_count_total: usize,
    /// Maximum MQ symbols emitted by any single block-local coding pass.
    pub max_tier1_pass_plan_mq_symbols_per_pass: usize,
    /// Maximum raw bypass bits emitted by any single block-local coding pass.
    pub max_tier1_pass_plan_raw_bits_per_pass: usize,
    /// Exact MQ symbol count from the compact token-emitter probe or gated GPU token-pack route.
    pub tier1_token_emit_mq_symbol_count_total: usize,
    /// Exact raw bypass bit count from the compact token-emitter probe or gated GPU token-pack route.
    pub tier1_token_emit_raw_bit_count_total: usize,
    /// Compact token bytes emitted by the token-emitter probe or gated GPU token-pack route.
    pub tier1_token_emit_token_bytes_total: usize,
    /// Maximum compact token bytes emitted by any one block.
    pub max_tier1_token_emit_token_bytes_per_block: usize,
    /// Segment records emitted by the token-emitter probe or gated GPU token-pack route.
    pub tier1_token_emit_segment_count_total: usize,
    /// Maximum token-emitter segment records for any one block.
    pub max_tier1_token_emit_segments_per_block: usize,
    /// XOR of per-block order-sensitive MQ symbol hashes from token emission.
    pub tier1_token_emit_mq_symbol_hash_xor: usize,
    /// XOR of per-block order-sensitive raw bit hashes from token emission.
    pub tier1_token_emit_raw_bit_hash_xor: usize,
    /// Total bytes produced by packing emitted Tier-1 tokens.
    pub tier1_token_pack_output_bytes_total: usize,
    /// Maximum token-pack output bytes for any one block.
    pub max_tier1_token_pack_output_bytes_per_block: usize,
    /// Resident Tier-1 code blocks that emitted at least one coding pass.
    pub tier1_nonzero_block_count_total: usize,
    /// Resident Tier-1 code blocks that emitted no coding passes.
    pub tier1_zero_block_count_total: usize,
    /// Missing most-significant bitplanes across resident Tier-1 code blocks.
    pub tier1_missing_bitplane_count_total: usize,
    /// Maximum missing most-significant bitplanes for any resident code block.
    pub max_tier1_missing_bitplanes_per_block: usize,
    /// Classic Tier-1 segment records emitted across resident code blocks.
    ///
    /// This remains zero for HTJ2K Tier-1, which does not use classic segment
    /// records.
    pub tier1_segment_count_total: usize,
    /// Maximum classic Tier-1 segment records emitted by any resident code block.
    pub max_tier1_segments_per_block: usize,
    /// Total host-planned packet payload-copy job slots across resident chunks.
    pub packet_payload_copy_job_capacity_total: usize,
    /// Maximum packet payload-copy job slots needed by any tile in the batch.
    pub max_packet_payload_copy_jobs_per_tile: usize,
    /// Actual packet payload-copy jobs emitted by packetization across resident chunks.
    pub packet_payload_copy_job_count_total: usize,
    /// Maximum actual packet payload-copy jobs emitted by any tile in the batch.
    pub max_packet_payload_copy_jobs_used_per_tile: usize,
    /// Actual packet payload-copy bytes emitted by packetization across resident chunks.
    pub packet_payload_copy_bytes_total: usize,
    /// Maximum actual packet payload-copy bytes emitted by any tile in the batch.
    pub max_packet_payload_copy_bytes_per_tile: usize,
    /// Packet payload-copy jobs at or below one copy-kernel stripe.
    pub packet_payload_copy_small_job_count_total: usize,
    /// Packet payload-copy jobs above one stripe and at or below 512 bytes.
    pub packet_payload_copy_medium_job_count_total: usize,
    /// Packet payload-copy jobs above 512 bytes.
    pub packet_payload_copy_large_job_count_total: usize,
    /// Packet payload-copy stripes launched by the copy kernel.
    pub packet_payload_copy_launched_stripe_count_total: usize,
    /// Packet payload-copy stripes that correspond to emitted copy jobs.
    pub packet_payload_copy_active_stripe_count_total: usize,
    /// Total packet output capacity, in bytes, across resident chunks.
    pub packet_output_capacity_total: usize,
    /// Maximum packet output capacity, in bytes, for any tile in the batch.
    pub max_packet_output_capacity: usize,
    /// Actual packet output bytes written by packetization across resident chunks.
    pub packet_output_used_bytes_total: usize,
    /// Maximum actual packet output bytes written by any tile in the batch.
    pub max_packet_output_used_bytes: usize,
    /// Codestream payload-copy bytes, in bytes, across resident chunks.
    pub codestream_payload_copy_bytes_total: usize,
    /// Codestream payload-copy threads launched by the copy kernel.
    pub codestream_payload_copy_launched_thread_count_total: usize,
    /// Estimated codestream payload-copy threads with in-range bytes to copy.
    pub codestream_payload_copy_active_thread_count_total: usize,
    /// Time spent waiting for codestream buffers.
    pub codestream_wait_duration: Duration,
    /// Alias of `codestream_wait_duration` using RCA naming.
    ///
    /// Do not sum this with `codestream_wait_duration` as an independent bucket.
    pub sync_wait_duration: Duration,
    /// Time spent materializing buffer-backed codestream bytes into host bytes.
    ///
    /// Current batch stats paths may leave this at zero. Host byte
    /// materialization timing is surfaced on `MetalLosslessEncodeOutcome` where
    /// applicable; this stage-stats bucket is reserved for stats-bearing
    /// host-output paths.
    pub host_readback_duration: Duration,
    /// Number of resident encode chunks.
    pub chunk_count: usize,
    /// Number of encoded tiles.
    pub tile_count: usize,
    /// Number of encoded code blocks.
    pub code_block_count: usize,
}

/// Combine rule for one stage-stat field in
/// [`MetalLosslessEncodeStageStats::add_assign`]: `dur` and `count` add with
/// saturation, `max` keeps the per-batch maximum, `xor` folds hashes.
macro_rules! stage_stat_combine {
    (dur, $self:ident, $other:ident, $field:ident) => {
        $self.$field = $self.$field.saturating_add($other.$field);
    };
    (count, $self:ident, $other:ident, $field:ident) => {
        $self.$field = $self.$field.saturating_add($other.$field);
    };
    (max, $self:ident, $other:ident, $field:ident) => {
        $self.$field = $self.$field.max($other.$field);
    };
    (xor, $self:ident, $other:ident, $field:ident) => {
        $self.$field ^= $other.$field;
    };
}

/// Contribution of one stage-stat field to
/// [`MetalLosslessEncodeStageStats::has_timings`]: only `dur` fields count.
macro_rules! stage_stat_timing_flag {
    (dur, $any:ident, $self:ident, $field:ident) => {
        $any = $any || $self.$field > Duration::ZERO;
    };
    ($class:ident, $any:ident, $self:ident, $field:ident) => {};
}

/// `From<compute::J2kResidentEncodeStageStats>` rule for one stage-stat
/// field: `resident` fields copy from the compute-layer stats, `local`
/// fields are facade-side only and keep their default.
#[cfg(target_os = "macos")]
macro_rules! stage_stat_from_resident {
    (resident, $out:ident, $stats:ident, $field:ident) => {
        $out.$field = $stats.$field;
    };
    (local, $out:ident, $stats:ident, $field:ident) => {};
}

/// Generate the per-field `MetalLosslessEncodeStageStats` impls from the
/// field table. The destructuring check at the end makes the table
/// exhaustive: adding a struct field without a table entry fails to compile.
macro_rules! j2k_metal_stage_stats_impls {
    ($(($field:ident, $class:ident, $source:ident)),* $(,)?) => {
        impl MetalLosslessEncodeStageStats {
            /// Return whether any non-zero timing was recorded.
            pub fn has_timings(&self) -> bool {
                let mut any = false;
                $(stage_stat_timing_flag!($class, any, self, $field);)*
                any
            }

            /// Accumulate another stage-stats value using saturating duration and counter additions.
            pub fn add_assign(&mut self, other: Self) {
                $(stage_stat_combine!($class, self, other, $field);)*
            }
        }

        #[cfg(target_os = "macos")]
        impl From<compute::J2kResidentEncodeStageStats> for MetalLosslessEncodeStageStats {
            fn from(stats: compute::J2kResidentEncodeStageStats) -> Self {
                let mut out = Self::default();
                $(stage_stat_from_resident!($source, out, stats, $field);)*
                out
            }
        }

        const _: fn(MetalLosslessEncodeStageStats) = |stats| {
            let MetalLosslessEncodeStageStats { $($field: _),* } = stats;
        };
    };
}

j2k_metal_stage_stats_impls! {
    (plan_duration, dur, local),
    (prepare_submit_duration, dur, local),
    (coefficient_prep_duration, dur, resident),
    (deinterleave_rct_duration, dur, resident),
    (dwt53_duration, dur, resident),
    (coefficient_extract_duration, dur, resident),
    (ht_table_build_duration, dur, resident),
    (ht_buffer_allocation_duration, dur, resident),
    (ht_command_encode_duration, dur, resident),
    (ht_block_encode_duration, dur, resident),
    (classic_tier1_setup_duration, dur, resident),
    (classic_block_encode_duration, dur, resident),
    (classic_tier1_token_pack_duration, dur, resident),
    (classic_packet_plan_duration, dur, resident),
    (classic_packet_buffer_setup_duration, dur, resident),
    (classic_command_buffer_commit_duration, dur, resident),
    (result_harvest_duration, dur, resident),
    (result_status_copy_duration, dur, resident),
    (result_private_recycle_duration, dur, resident),
    (result_shared_recycle_duration, dur, resident),
    (result_codestream_collect_duration, dur, resident),
    (packet_block_prep_duration, dur, resident),
    (packetization_duration, dur, resident),
    (codestream_assembly_duration, dur, resident),
    (coefficient_prep_gpu_duration, dur, resident),
    (coefficient_deinterleave_rct_gpu_duration, dur, resident),
    (coefficient_dwt53_gpu_duration, dur, resident),
    (coefficient_dwt53_vertical_gpu_duration, dur, resident),
    (coefficient_dwt53_horizontal_gpu_duration, dur, resident),
    (coefficient_extract_gpu_duration, dur, resident),
    (coefficient_copy_gpu_duration, dur, resident),
    (gpu_elapsed_wall_duration, dur, resident),
    (classic_block_gpu_duration, dur, resident),
    (classic_tier1_density_gpu_duration, dur, resident),
    (classic_tier1_raw_pack_gpu_duration, dur, resident),
    (classic_tier1_arithmetic_pack_gpu_duration, dur, resident),
    (classic_tier1_symbol_plan_gpu_duration, dur, resident),
    (classic_tier1_pass_plan_gpu_duration, dur, resident),
    (classic_tier1_token_emit_gpu_duration, dur, resident),
    (classic_tier1_split_token_emit_gpu_duration, dur, resident),
    (classic_tier1_token_pack_gpu_duration, dur, resident),
    (ht_block_gpu_duration, dur, resident),
    (packet_block_prep_gpu_duration, dur, resident),
    (packetization_gpu_duration, dur, resident),
    (packet_payload_copy_gpu_duration, dur, resident),
    (codestream_assembly_gpu_duration, dur, resident),
    (codestream_payload_copy_gpu_duration, dur, resident),
    (tier1_output_capacity_total, count, resident),
    (max_tier1_output_capacity, max, resident),
    (tier1_output_used_bytes_total, count, resident),
    (max_tier1_output_used_bytes, max, resident),
    (tier1_segment_capacity_total, count, resident),
    (max_tier1_segment_capacity_per_block, max, resident),
    (tier1_coding_pass_count_total, count, resident),
    (max_tier1_coding_passes_per_block, max, resident),
    (tier1_arithmetic_pass_count_total, count, resident),
    (tier1_raw_pass_count_total, count, resident),
    (tier1_cleanup_pass_count_total, count, resident),
    (tier1_sigprop_pass_count_total, count, resident),
    (tier1_magref_pass_count_total, count, resident),
    (tier1_arithmetic_cleanup_pass_count_total, count, resident),
    (tier1_arithmetic_sigprop_pass_count_total, count, resident),
    (tier1_arithmetic_magref_pass_count_total, count, resident),
    (tier1_raw_sigprop_pass_count_total, count, resident),
    (tier1_raw_magref_pass_count_total, count, resident),
    (tier1_full_scan_coeff_visit_count_total, count, resident),
    (tier1_arithmetic_scan_coeff_visit_count_total, count, resident),
    (tier1_raw_scan_coeff_visit_count_total, count, resident),
    (tier1_cleanup_scan_coeff_visit_count_total, count, resident),
    (tier1_sigprop_scan_coeff_visit_count_total, count, resident),
    (tier1_magref_scan_coeff_visit_count_total, count, resident),
    (max_tier1_full_scan_coeff_visits_per_block, max, resident),
    (tier1_sigprop_active_candidate_count_total, count, resident),
    (tier1_sigprop_new_significant_count_total, count, resident),
    (tier1_magref_active_candidate_count_total, count, resident),
    (tier1_arithmetic_sigprop_active_candidate_count_total, count, resident),
    (tier1_arithmetic_sigprop_new_significant_count_total, count, resident),
    (tier1_raw_sigprop_active_candidate_count_total, count, resident),
    (tier1_raw_sigprop_new_significant_count_total, count, resident),
    (tier1_arithmetic_magref_active_candidate_count_total, count, resident),
    (tier1_raw_magref_active_candidate_count_total, count, resident),
    (tier1_cleanup_active_candidate_count_total, count, resident),
    (tier1_cleanup_new_significant_count_total, count, resident),
    (tier1_cleanup_rlc_stripe_count_total, count, resident),
    (tier1_cleanup_rlc_zero_stripe_count_total, count, resident),
    (tier1_symbol_plan_mq_symbol_count_total, count, resident),
    (tier1_symbol_plan_raw_bit_count_total, count, resident),
    (max_tier1_symbol_plan_mq_symbols_per_block, max, resident),
    (max_tier1_symbol_plan_raw_bits_per_block, max, resident),
    (tier1_symbol_plan_packed_token_bytes_total, count, resident),
    (max_tier1_symbol_plan_packed_token_bytes_per_block, max, resident),
    (tier1_symbol_plan_cleanup_mq_symbol_count_total, count, resident),
    (tier1_symbol_plan_sigprop_mq_symbol_count_total, count, resident),
    (tier1_symbol_plan_magref_mq_symbol_count_total, count, resident),
    (tier1_symbol_plan_raw_sigprop_bit_count_total, count, resident),
    (tier1_symbol_plan_raw_magref_bit_count_total, count, resident),
    (tier1_symbol_plan_cleanup_sign_symbol_count_total, count, resident),
    (tier1_symbol_plan_sigprop_sign_symbol_count_total, count, resident),
    (tier1_symbol_plan_mq_symbol_hash_xor, xor, resident),
    (tier1_symbol_plan_raw_bit_hash_xor, xor, resident),
    (tier1_pass_plan_mq_symbol_count_total, count, resident),
    (tier1_pass_plan_raw_bit_count_total, count, resident),
    (tier1_pass_plan_nonempty_mq_pass_count_total, count, resident),
    (tier1_pass_plan_nonempty_raw_pass_count_total, count, resident),
    (max_tier1_pass_plan_mq_symbols_per_pass, max, resident),
    (max_tier1_pass_plan_raw_bits_per_pass, max, resident),
    (tier1_token_emit_mq_symbol_count_total, count, resident),
    (tier1_token_emit_raw_bit_count_total, count, resident),
    (tier1_token_emit_token_bytes_total, count, resident),
    (max_tier1_token_emit_token_bytes_per_block, max, resident),
    (tier1_token_emit_segment_count_total, count, resident),
    (max_tier1_token_emit_segments_per_block, max, resident),
    (tier1_token_emit_mq_symbol_hash_xor, xor, resident),
    (tier1_token_emit_raw_bit_hash_xor, xor, resident),
    (tier1_token_pack_output_bytes_total, count, resident),
    (max_tier1_token_pack_output_bytes_per_block, max, resident),
    (tier1_nonzero_block_count_total, count, resident),
    (tier1_zero_block_count_total, count, resident),
    (tier1_missing_bitplane_count_total, count, resident),
    (max_tier1_missing_bitplanes_per_block, max, resident),
    (tier1_segment_count_total, count, resident),
    (max_tier1_segments_per_block, max, resident),
    (packet_payload_copy_job_capacity_total, count, resident),
    (max_packet_payload_copy_jobs_per_tile, max, resident),
    (packet_payload_copy_job_count_total, count, resident),
    (max_packet_payload_copy_jobs_used_per_tile, max, resident),
    (packet_payload_copy_bytes_total, count, resident),
    (max_packet_payload_copy_bytes_per_tile, max, resident),
    (packet_payload_copy_small_job_count_total, count, resident),
    (packet_payload_copy_medium_job_count_total, count, resident),
    (packet_payload_copy_large_job_count_total, count, resident),
    (packet_payload_copy_launched_stripe_count_total, count, resident),
    (packet_payload_copy_active_stripe_count_total, count, resident),
    (packet_output_capacity_total, count, resident),
    (max_packet_output_capacity, max, resident),
    (packet_output_used_bytes_total, count, resident),
    (max_packet_output_used_bytes, max, resident),
    (codestream_payload_copy_bytes_total, count, resident),
    (codestream_payload_copy_launched_thread_count_total, count, resident),
    (codestream_payload_copy_active_thread_count_total, count, resident),
    (codestream_wait_duration, dur, local),
    (sync_wait_duration, dur, local),
    (host_readback_duration, dur, local),
    (chunk_count, count, local),
    (tile_count, count, local),
    (code_block_count, count, resident),
}

#[cfg(any(target_os = "macos", test))]
pub(super) fn add_resident_prep_duration(
    stats: &mut MetalLosslessEncodeBatchStats,
    duration: Duration,
    profile_stages: bool,
) {
    if !profile_stages {
        return;
    }
    stats.stage_stats.coefficient_prep_duration = stats
        .stage_stats
        .coefficient_prep_duration
        .saturating_add(duration);
}

#[cfg(any(target_os = "macos", test))]
pub(super) fn add_resident_prep_wall_duration(
    stats: &mut MetalLosslessEncodeBatchStats,
    wall_duration: Duration,
    profile_stages: bool,
) {
    add_resident_prep_duration(stats, wall_duration, profile_stages);
}

/// Resolved resident Metal lossless J2K/HTJ2K tile batch encode metrics.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MetalLosslessEncodeBatchStats {
    /// Caller-requested maximum number of in-flight tiles.
    pub configured_inflight_tiles: Option<usize>,
    /// Effective maximum number of in-flight tiles after clamping.
    pub effective_inflight_tiles: usize,
    /// Caller-requested resident encode memory budget in bytes.
    pub configured_memory_budget_bytes: Option<usize>,
    /// Effective resident encode memory budget in bytes.
    pub effective_memory_budget_bytes: usize,
    /// Estimated peak resident memory required per tile.
    pub estimated_peak_bytes_per_tile: usize,
    /// Maximum observed in-flight tiles during the batch.
    pub max_observed_inflight_tiles: usize,
    /// End-to-end wall time for the batch encode.
    pub encode_wall_duration: Duration,
    /// Resident encode stage timing summary.
    pub stage_stats: MetalLosslessEncodeStageStats,
}

/// Resident Metal lossless J2K/HTJ2K tile batch output and batch-level metrics.
pub struct MetalLosslessBufferEncodeBatchOutcome {
    /// Per-tile buffer-backed encode outcomes.
    pub outcomes: Vec<MetalLosslessBufferEncodeOutcome>,
    /// Batch-level resident encode metrics.
    pub stats: MetalLosslessEncodeBatchStats,
}
