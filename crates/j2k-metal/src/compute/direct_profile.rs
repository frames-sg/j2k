// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{
    sync::atomic::{AtomicU64, Ordering as AtomicOrdering},
    time::{Duration, Instant},
};

use j2k_core::PixelFormat;
use j2k_native::{HtCodeBlockDecodeProfile, J2kCodeBlockDecodeProfile};

use crate::profile_env::{decode_profile_label, metal_profile_stages_enabled};

#[cfg(target_os = "macos")]
use super::{completed_command_buffer_gpu_duration, DecodeHybridSplitCommandBuffers};

#[derive(Default)]
pub(super) struct DirectHybridStageTimings {
    pub(super) cpu_tier1: u128,
    pub(super) cpu_tier1_flattened_batches: u128,
    pub(super) cpu_tier1_classic_segment_prep: u128,
    pub(super) cpu_tier1_classic_block_decode: u128,
    pub(super) cpu_tier1_classic_sigprop: u128,
    pub(super) cpu_tier1_classic_magref: u128,
    pub(super) cpu_tier1_classic_cleanup: u128,
    pub(super) cpu_tier1_classic_bypass: u128,
    pub(super) cpu_tier1_classic_output_convert: u128,
    pub(super) cpu_tier1_ht_block_decode: u128,
    pub(super) cpu_tier1_ht_cleanup: u128,
    pub(super) cpu_tier1_ht_mag_sgn: u128,
    pub(super) cpu_tier1_ht_sigma: u128,
    pub(super) cpu_tier1_ht_sigprop: u128,
    pub(super) cpu_tier1_ht_magref: u128,
    pub(super) coefficient_upload: u128,
    pub(super) metal_idwt_encode: u128,
    pub(super) metal_store_encode: u128,
    pub(super) metal_mct_pack_encode: u128,
    pub(super) command_wait: u128,
    pub(super) gpu_command: u128,
    pub(super) metal_idwt_gpu: u128,
    pub(super) metal_idwt_interleave_gpu: u128,
    pub(super) metal_idwt_horizontal_gpu: u128,
    pub(super) metal_idwt_vertical_gpu: u128,
    pub(super) metal_store_gpu: u128,
    pub(super) metal_mct_pack_gpu: u128,
}

#[derive(Default)]
pub(super) struct CpuTier1DecodeSubstageCounters {
    classic_segment_prep: AtomicU64,
    classic_block_decode: AtomicU64,
    classic_sigprop: AtomicU64,
    classic_magref: AtomicU64,
    classic_cleanup: AtomicU64,
    classic_bypass: AtomicU64,
    classic_output_convert: AtomicU64,
    ht_block_decode: AtomicU64,
    ht_cleanup: AtomicU64,
    ht_mag_sgn: AtomicU64,
    ht_sigma: AtomicU64,
    ht_sigprop: AtomicU64,
    ht_magref: AtomicU64,
}

impl CpuTier1DecodeSubstageCounters {
    fn add_counter(counter: &AtomicU64, elapsed_us: u128) {
        counter.fetch_add(
            elapsed_us.min(u128::from(u64::MAX)) as u64,
            AtomicOrdering::Relaxed,
        );
    }

    pub(super) fn record_classic_segment_prep(&self, started: Instant) {
        self.classic_segment_prep
            .fetch_add(elapsed_us_u64(started), AtomicOrdering::Relaxed);
    }

    pub(super) fn record_classic_block_decode(
        &self,
        started: Instant,
        profile: &J2kCodeBlockDecodeProfile,
    ) {
        self.classic_block_decode
            .fetch_add(elapsed_us_u64(started), AtomicOrdering::Relaxed);
        Self::add_counter(&self.classic_sigprop, profile.sigprop_us);
        Self::add_counter(&self.classic_magref, profile.magref_us);
        Self::add_counter(&self.classic_cleanup, profile.cleanup_us);
        Self::add_counter(&self.classic_bypass, profile.bypass_us);
        Self::add_counter(&self.classic_output_convert, profile.output_convert_us);
    }

    pub(super) fn record_ht_block_decode(
        &self,
        started: Instant,
        profile: &HtCodeBlockDecodeProfile,
    ) {
        self.ht_block_decode
            .fetch_add(elapsed_us_u64(started), AtomicOrdering::Relaxed);
        Self::add_counter(&self.ht_cleanup, profile.cleanup_us);
        Self::add_counter(&self.ht_mag_sgn, profile.mag_sgn_us);
        Self::add_counter(&self.ht_sigma, profile.sigma_us);
        Self::add_counter(&self.ht_sigprop, profile.sigprop_us);
        Self::add_counter(&self.ht_magref, profile.magref_us);
    }

    fn load_counter(counter: &AtomicU64) -> u128 {
        u128::from(counter.load(AtomicOrdering::Relaxed))
    }

    pub(super) fn add_to_stage_timings(&self, timings: &mut DirectHybridStageTimings) {
        timings.cpu_tier1_classic_segment_prep = timings
            .cpu_tier1_classic_segment_prep
            .saturating_add(Self::load_counter(&self.classic_segment_prep));
        timings.cpu_tier1_classic_block_decode = timings
            .cpu_tier1_classic_block_decode
            .saturating_add(Self::load_counter(&self.classic_block_decode));
        timings.cpu_tier1_classic_sigprop = timings
            .cpu_tier1_classic_sigprop
            .saturating_add(Self::load_counter(&self.classic_sigprop));
        timings.cpu_tier1_classic_magref = timings
            .cpu_tier1_classic_magref
            .saturating_add(Self::load_counter(&self.classic_magref));
        timings.cpu_tier1_classic_cleanup = timings
            .cpu_tier1_classic_cleanup
            .saturating_add(Self::load_counter(&self.classic_cleanup));
        timings.cpu_tier1_classic_bypass = timings
            .cpu_tier1_classic_bypass
            .saturating_add(Self::load_counter(&self.classic_bypass));
        timings.cpu_tier1_classic_output_convert = timings
            .cpu_tier1_classic_output_convert
            .saturating_add(Self::load_counter(&self.classic_output_convert));
        timings.cpu_tier1_ht_block_decode = timings
            .cpu_tier1_ht_block_decode
            .saturating_add(Self::load_counter(&self.ht_block_decode));
        timings.cpu_tier1_ht_cleanup = timings
            .cpu_tier1_ht_cleanup
            .saturating_add(Self::load_counter(&self.ht_cleanup));
        timings.cpu_tier1_ht_mag_sgn = timings
            .cpu_tier1_ht_mag_sgn
            .saturating_add(Self::load_counter(&self.ht_mag_sgn));
        timings.cpu_tier1_ht_sigma = timings
            .cpu_tier1_ht_sigma
            .saturating_add(Self::load_counter(&self.ht_sigma));
        timings.cpu_tier1_ht_sigprop = timings
            .cpu_tier1_ht_sigprop
            .saturating_add(Self::load_counter(&self.ht_sigprop));
        timings.cpu_tier1_ht_magref = timings
            .cpu_tier1_ht_magref
            .saturating_add(Self::load_counter(&self.ht_magref));
    }
}

pub(super) fn elapsed_us(started: Instant) -> u128 {
    started.elapsed().as_micros()
}

#[cfg(target_os = "macos")]
pub(super) fn record_completed_decode_split_gpu_stages(
    timings: &mut DirectHybridStageTimings,
    command_buffers: &DecodeHybridSplitCommandBuffers,
) {
    let mut gpu_command = Duration::ZERO;
    let mut idwt_gpu = Duration::ZERO;
    if let Some(duration) = completed_command_buffer_gpu_duration(&command_buffers.idwt_interleave)
    {
        timings.metal_idwt_interleave_gpu = timings
            .metal_idwt_interleave_gpu
            .saturating_add(duration.as_micros());
        idwt_gpu = idwt_gpu.saturating_add(duration);
        gpu_command = gpu_command.saturating_add(duration);
    }
    if let Some(duration) = completed_command_buffer_gpu_duration(&command_buffers.idwt_horizontal)
    {
        timings.metal_idwt_horizontal_gpu = timings
            .metal_idwt_horizontal_gpu
            .saturating_add(duration.as_micros());
        idwt_gpu = idwt_gpu.saturating_add(duration);
        gpu_command = gpu_command.saturating_add(duration);
    }
    if let Some(duration) = completed_command_buffer_gpu_duration(&command_buffers.idwt_vertical) {
        timings.metal_idwt_vertical_gpu = timings
            .metal_idwt_vertical_gpu
            .saturating_add(duration.as_micros());
        idwt_gpu = idwt_gpu.saturating_add(duration);
        gpu_command = gpu_command.saturating_add(duration);
    }
    timings.metal_idwt_gpu = timings.metal_idwt_gpu.saturating_add(idwt_gpu.as_micros());
    if let Some(duration) = completed_command_buffer_gpu_duration(&command_buffers.store) {
        timings.metal_store_gpu = timings.metal_store_gpu.saturating_add(duration.as_micros());
        gpu_command = gpu_command.saturating_add(duration);
    }
    if let Some(duration) = completed_command_buffer_gpu_duration(&command_buffers.mct_pack) {
        timings.metal_mct_pack_gpu = timings
            .metal_mct_pack_gpu
            .saturating_add(duration.as_micros());
        gpu_command = gpu_command.saturating_add(duration);
    }
    timings.gpu_command = timings.gpu_command.saturating_add(gpu_command.as_micros());
}

fn elapsed_us_u64(started: Instant) -> u64 {
    elapsed_us(started).min(u128::from(u64::MAX)) as u64
}

pub(super) fn emit_direct_hybrid_stage_timings(
    timings: &DirectHybridStageTimings,
    fmt: PixelFormat,
    batch_count: usize,
) {
    if !metal_profile_stages_enabled() {
        return;
    }

    let fmt_s = format!("{fmt:?}");
    let batch_count_s = batch_count.to_string();
    let label = decode_profile_label();
    for (stage, elapsed_us) in [
        ("cpu_tier1", timings.cpu_tier1),
        (
            "cpu_tier1_flattened_batches",
            timings.cpu_tier1_flattened_batches,
        ),
        (
            "cpu_tier1_classic_segment_prep",
            timings.cpu_tier1_classic_segment_prep,
        ),
        (
            "cpu_tier1_classic_block_decode",
            timings.cpu_tier1_classic_block_decode,
        ),
        (
            "cpu_tier1_classic_sigprop",
            timings.cpu_tier1_classic_sigprop,
        ),
        ("cpu_tier1_classic_magref", timings.cpu_tier1_classic_magref),
        (
            "cpu_tier1_classic_cleanup",
            timings.cpu_tier1_classic_cleanup,
        ),
        ("cpu_tier1_classic_bypass", timings.cpu_tier1_classic_bypass),
        (
            "cpu_tier1_classic_output_convert",
            timings.cpu_tier1_classic_output_convert,
        ),
        (
            "cpu_tier1_ht_block_decode",
            timings.cpu_tier1_ht_block_decode,
        ),
        ("cpu_tier1_ht_cleanup", timings.cpu_tier1_ht_cleanup),
        ("cpu_tier1_ht_mag_sgn", timings.cpu_tier1_ht_mag_sgn),
        ("cpu_tier1_ht_sigma", timings.cpu_tier1_ht_sigma),
        ("cpu_tier1_ht_sigprop", timings.cpu_tier1_ht_sigprop),
        ("cpu_tier1_ht_magref", timings.cpu_tier1_ht_magref),
        ("coefficient_upload", timings.coefficient_upload),
        ("metal_idwt_encode", timings.metal_idwt_encode),
        ("metal_store_encode", timings.metal_store_encode),
        ("metal_mct_pack_encode", timings.metal_mct_pack_encode),
        ("command_wait", timings.command_wait),
        ("gpu_command", timings.gpu_command),
        ("metal_idwt_gpu", timings.metal_idwt_gpu),
        (
            "metal_idwt_interleave_gpu",
            timings.metal_idwt_interleave_gpu,
        ),
        (
            "metal_idwt_horizontal_gpu",
            timings.metal_idwt_horizontal_gpu,
        ),
        ("metal_idwt_vertical_gpu", timings.metal_idwt_vertical_gpu),
        ("metal_store_gpu", timings.metal_store_gpu),
        ("metal_mct_pack_gpu", timings.metal_mct_pack_gpu),
    ] {
        let elapsed_us_s = elapsed_us.to_string();
        let processor = stage_processor(stage);
        let metric = stage_metric(stage);
        let metric_kind = stage_metric_kind(stage);
        let aggregation = stage_aggregation(stage);
        j2k_profile::emit_profile_row_now(
            "j2k",
            "decode",
            "metal_cpu_hybrid",
            &[
                ("pipeline", "decode_hybrid".to_string()),
                ("label", label.clone()),
                ("stage", stage.to_string()),
                ("processor", processor.to_string()),
                ("metric", metric.to_string()),
                ("metric_kind", metric_kind.to_string()),
                ("aggregation", aggregation.to_string()),
                ("fmt", fmt_s.clone()),
                ("batch_count", batch_count_s.clone()),
                ("elapsed_us", elapsed_us_s),
            ],
        );
    }
}

fn stage_processor(stage: &str) -> &'static str {
    match stage {
        "cpu_tier1_flattened_batches" => "scheduler",
        "cpu_tier1"
        | "cpu_tier1_classic_segment_prep"
        | "cpu_tier1_classic_block_decode"
        | "cpu_tier1_classic_sigprop"
        | "cpu_tier1_classic_magref"
        | "cpu_tier1_classic_cleanup"
        | "cpu_tier1_classic_bypass"
        | "cpu_tier1_classic_output_convert"
        | "cpu_tier1_ht_block_decode"
        | "cpu_tier1_ht_cleanup"
        | "cpu_tier1_ht_mag_sgn"
        | "cpu_tier1_ht_sigma"
        | "cpu_tier1_ht_sigprop"
        | "cpu_tier1_ht_magref" => "cpu",
        "coefficient_upload" => "transfer",
        "metal_idwt_encode"
        | "metal_store_encode"
        | "metal_mct_pack_encode"
        | "gpu_command"
        | "metal_idwt_gpu"
        | "metal_idwt_interleave_gpu"
        | "metal_idwt_horizontal_gpu"
        | "metal_idwt_vertical_gpu"
        | "metal_store_gpu"
        | "metal_mct_pack_gpu" => "metal",
        "command_wait" => "wait",
        _ => "hybrid",
    }
}

fn stage_metric(stage: &str) -> &'static str {
    match stage {
        "cpu_tier1_flattened_batches" => "count",
        "cpu_tier1_classic_segment_prep"
        | "cpu_tier1_classic_block_decode"
        | "cpu_tier1_classic_sigprop"
        | "cpu_tier1_classic_magref"
        | "cpu_tier1_classic_cleanup"
        | "cpu_tier1_classic_bypass"
        | "cpu_tier1_classic_output_convert"
        | "cpu_tier1_ht_block_decode"
        | "cpu_tier1_ht_cleanup"
        | "cpu_tier1_ht_mag_sgn"
        | "cpu_tier1_ht_sigma"
        | "cpu_tier1_ht_sigprop"
        | "cpu_tier1_ht_magref" => "cpu_worker_us",
        "gpu_command"
        | "metal_idwt_gpu"
        | "metal_idwt_interleave_gpu"
        | "metal_idwt_horizontal_gpu"
        | "metal_idwt_vertical_gpu"
        | "metal_store_gpu"
        | "metal_mct_pack_gpu" => "gpu_elapsed_us",
        _ => "wall_us",
    }
}

fn stage_metric_kind(stage: &str) -> &'static str {
    match stage_metric(stage) {
        "count" => "counter",
        "cpu_worker_us" => "cpu_worker_sum",
        "gpu_elapsed_us" => "gpu_busy_sum",
        _ => "wall_elapsed",
    }
}

fn stage_aggregation(stage: &str) -> &'static str {
    match stage_metric(stage) {
        "count" | "cpu_worker_us" | "gpu_elapsed_us" => "sum",
        _ => "exclusive",
    }
}
