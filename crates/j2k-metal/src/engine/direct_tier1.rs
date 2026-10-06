// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(target_os = "macos")]
use std::sync::{Arc, OnceLock};

use crate::metal_types::Buffer;
use j2k_core::accelerator::GpuAbi;

use super::abi::{J2kClassicCleanupBatchJob, J2kClassicSegment};
use super::{
    copied_slice_buffer, ClassicTier1Buffers, ClassicTier1Inputs, Error, MetalRuntime,
    PreparedClassicSubBand, PreparedClassicSubBandGroup, PreparedDirectColorPlan,
};

pub(super) const HYBRID_CPU_DECODE_MIN_INPUTS_PER_TASK: usize = 1;

const HYBRID_FLAT_CPU_TIER1_MIN_DIM: u32 = 1024;
const HYBRID_FLAT_CPU_TIER1_MIN_COUNT: usize = 16;
const HYBRID_FLAT_CPU_TIER1_ENV: &str = "J2K_HYBRID_FLAT_CPU_TIER1";

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum DirectTier1Mode {
    Metal,
    CpuUpload,
}

#[cfg(test)]
fn record_direct_tier1_input_buffer_prepare(runtime: &MetalRuntime) {
    super::test_counters::record_direct_tier1_input_buffer_prepare(runtime);
}

#[cfg(not(test))]
fn record_direct_tier1_input_buffer_prepare(_runtime: &MetalRuntime) {}

#[cfg(test)]
fn record_direct_tier1_input_buffer_runtime(runtime: &MetalRuntime) {
    super::test_counters::record_direct_tier1_input_buffer_runtime(runtime);
}

#[cfg(not(test))]
fn record_direct_tier1_input_buffer_runtime(_runtime: &MetalRuntime) {}

fn prepare_direct_tier1_input_buffer<T: GpuAbi>(
    runtime: &MetalRuntime,
    data: &[T],
    mode: DirectTier1Mode,
) -> Result<Buffer, Error> {
    record_direct_tier1_input_buffer_runtime(runtime);
    match mode {
        DirectTier1Mode::Metal => {
            record_direct_tier1_input_buffer_prepare(runtime);
            copied_slice_buffer(&runtime.device, data)
        }
        DirectTier1Mode::CpuUpload => Ok(runtime.tier1_dummy_buffer.clone()),
    }
}

/// Bytes of the buffer `prepare_direct_tier1_input_buffer` creates for `data`.
/// `CpuUpload` binds the runtime's shared placeholder, which no plan owns.
pub(super) fn direct_tier1_input_buffer_bytes<T>(data: &[T], mode: DirectTier1Mode) -> usize {
    match mode {
        DirectTier1Mode::Metal => size_of_val(data).max(1),
        DirectTier1Mode::CpuUpload => 0,
    }
}

impl ClassicTier1Inputs {
    pub(super) fn new(mode: DirectTier1Mode) -> Self {
        Self {
            mode,
            buffers: OnceLock::new(),
        }
    }

    /// Returns the device copies of `coded_data`, `jobs` and `segments`,
    /// building them on the first call.
    pub(super) fn get_or_create(
        &self,
        runtime: &MetalRuntime,
        coded_data: &[u8],
        jobs: &[J2kClassicCleanupBatchJob],
        segments: &[J2kClassicSegment],
    ) -> Result<&ClassicTier1Buffers, Error> {
        if let Some(buffers) = self.buffers.get() {
            return Ok(buffers);
        }
        let buffers = ClassicTier1Buffers {
            coded: prepare_direct_tier1_input_buffer(runtime, coded_data, self.mode)?,
            jobs: prepare_direct_tier1_input_buffer(runtime, jobs, self.mode)?,
            segments: prepare_direct_tier1_input_buffer(runtime, segments, self.mode)?,
        };
        // If another reader filled the lock first, its buffers hold the same
        // bytes and this set is dropped.
        Ok(self.buffers.get_or_init(|| buffers))
    }

    /// Drops buffers built from host vectors that have since been edited.
    pub(super) fn reset(&mut self) {
        self.buffers.take();
    }
}

impl PreparedClassicSubBand {
    pub(super) fn tier1_buffers(
        &self,
        runtime: &MetalRuntime,
    ) -> Result<&ClassicTier1Buffers, Error> {
        self.tier1_inputs
            .get_or_create(runtime, &self.coded_data, &self.jobs, &self.segments)
    }
}

impl PreparedClassicSubBandGroup {
    pub(super) fn tier1_buffers(
        &self,
        runtime: &MetalRuntime,
    ) -> Result<&ClassicTier1Buffers, Error> {
        self.tier1_inputs
            .get_or_create(runtime, &self.coded_data, &self.jobs, &self.segments)
    }
}

pub(super) fn flattened_hybrid_cpu_tier1_enabled() -> bool {
    std::env::var_os(HYBRID_FLAT_CPU_TIER1_ENV).is_some_and(|value| {
        let value = value.to_string_lossy();
        !value.is_empty() && value != "0" && value != "false"
    })
}

pub(super) fn should_flatten_hybrid_cpu_tier1_color_batch(
    plans: &[Arc<PreparedDirectColorPlan>],
) -> bool {
    let Some(first) = plans.first() else {
        return false;
    };
    plans.len() >= HYBRID_FLAT_CPU_TIER1_MIN_COUNT
        && first.dimensions.0.max(first.dimensions.1) >= HYBRID_FLAT_CPU_TIER1_MIN_DIM
        && !plans.iter().all(|plan| Arc::ptr_eq(plan, first))
}

#[cfg(test)]
pub(super) fn record_hybrid_stacked_component_batch(tier1_mode: DirectTier1Mode) {
    if tier1_mode == DirectTier1Mode::CpuUpload {
        super::test_counters::record_hybrid_stacked_component_batch();
    }
}

#[cfg(test)]
pub(super) fn record_stacked_component_batch() {
    super::test_counters::record_stacked_component_batch();
}

#[cfg(not(test))]
pub(super) fn record_stacked_component_batch() {}

#[cfg(not(test))]
pub(super) fn record_hybrid_stacked_component_batch(_tier1_mode: DirectTier1Mode) {}

#[cfg(test)]
pub(super) fn record_hybrid_repeated_output_blit() {
    super::test_counters::record_hybrid_repeated_output_blit();
}

#[cfg(not(test))]
pub(super) fn record_hybrid_repeated_output_blit() {}

#[cfg(test)]
pub(super) fn record_hybrid_cpu_decode_worker_init() {
    super::test_counters::record_hybrid_cpu_decode_worker_init();
}

#[cfg(not(test))]
pub(super) fn record_hybrid_cpu_decode_worker_init() {}

#[cfg(test)]
pub(super) fn record_hybrid_cpu_decode_inputs(count: usize) {
    super::test_counters::record_hybrid_cpu_decode_inputs(count);
}

#[cfg(not(test))]
pub(super) fn record_hybrid_cpu_decode_inputs(_count: usize) {}

#[cfg(test)]
pub(super) fn record_flattened_hybrid_cpu_decode_batch() {
    super::test_counters::record_flattened_hybrid_cpu_decode_batch();
}

#[cfg(not(test))]
pub(super) fn record_flattened_hybrid_cpu_decode_batch() {}
