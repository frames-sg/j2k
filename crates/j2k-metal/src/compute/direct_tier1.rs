// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use j2k_core::accelerator::GpuAbi;
use metal::Buffer;

use super::{borrow_slice_buffer, MetalRuntime, PreparedDirectColorPlan};

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
fn record_direct_tier1_input_buffer_prepare() {
    super::test_counters::record_direct_tier1_input_buffer_prepare();
}

#[cfg(not(test))]
fn record_direct_tier1_input_buffer_prepare() {}

pub(super) fn prepare_direct_tier1_input_buffer<T: GpuAbi>(
    runtime: &MetalRuntime,
    data: &[T],
    mode: DirectTier1Mode,
) -> Buffer {
    match mode {
        DirectTier1Mode::Metal => {
            record_direct_tier1_input_buffer_prepare();
            borrow_slice_buffer(&runtime.device, data)
        }
        DirectTier1Mode::CpuUpload => runtime.tier1_dummy_buffer.clone(),
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
