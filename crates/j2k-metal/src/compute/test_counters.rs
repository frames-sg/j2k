// SPDX-License-Identifier: MIT OR Apache-2.0

use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};

macro_rules! test_atomic_counter {
    ($counter:ident, $reset:ident, $load:ident) => {
        static $counter: AtomicUsize = AtomicUsize::new(0);

        pub(crate) fn $reset() {
            $counter.store(0, Ordering::Relaxed);
        }

        pub(crate) fn $load() -> usize {
            $counter.load(Ordering::Relaxed)
        }
    };
}

test_atomic_counter!(
    HT_BATCH_COEFFICIENT_COPY_BLITS,
    reset_ht_batch_coefficient_copy_blits_for_test,
    ht_batch_coefficient_copy_blits_for_test
);
test_atomic_counter!(
    HYBRID_STACKED_COMPONENT_BATCHES,
    reset_hybrid_stacked_component_batches_for_test,
    hybrid_stacked_component_batches_for_test
);
test_atomic_counter!(
    HYBRID_REPEATED_OUTPUT_BLITS,
    reset_hybrid_repeated_output_blits_for_test,
    hybrid_repeated_output_blits_for_test
);
test_atomic_counter!(
    HYBRID_CPU_DECODE_WORKER_INITS,
    reset_hybrid_cpu_decode_worker_inits_for_test,
    hybrid_cpu_decode_worker_inits_for_test
);
test_atomic_counter!(
    HYBRID_CPU_DECODE_INPUTS,
    reset_hybrid_cpu_decode_inputs_for_test,
    hybrid_cpu_decode_inputs_for_test
);
test_atomic_counter!(
    FLATTENED_HYBRID_CPU_DECODE_BATCHES,
    reset_flattened_hybrid_cpu_decode_batches_for_test,
    flattened_hybrid_cpu_decode_batches_for_test
);

std::thread_local! {
    static RESIDENT_GPU_TIMESTAMP_QUERIES: Cell<usize> = const { Cell::new(0) };
    static RESIDENT_CODESTREAM_COMMAND_BUFFER_WAITS: Cell<usize> = const { Cell::new(0) };
    static DIRECT_TIER1_INPUT_BUFFER_PREPARES: Cell<usize> = const { Cell::new(0) };
    static HYBRID_CPU_DECODE_INPUTS_FOR_THREAD: Cell<usize> = const { Cell::new(0) };
    static LOSSLESS_DEINTERLEAVE_RCT_FUSED_DISPATCHES: Cell<usize> = const { Cell::new(0) };
    static CLASSIC_GPU_TOKEN_PACK_DISPATCHES: Cell<usize> = const { Cell::new(0) };
    static CLASSIC_SPLIT_MQ_BYTE_GPU_TOKEN_PACK_DISPATCHES: Cell<usize> = const { Cell::new(0) };
}

pub(crate) fn reset_resident_gpu_timestamp_queries_for_test() {
    RESIDENT_GPU_TIMESTAMP_QUERIES.with(|queries| queries.set(0));
}

pub(crate) fn resident_gpu_timestamp_queries_for_test() -> usize {
    RESIDENT_GPU_TIMESTAMP_QUERIES.with(Cell::get)
}

pub(crate) fn record_resident_gpu_timestamp_query() {
    RESIDENT_GPU_TIMESTAMP_QUERIES.with(|queries| queries.set(queries.get() + 1));
}

pub(crate) fn reset_resident_codestream_command_buffer_waits_for_test() {
    RESIDENT_CODESTREAM_COMMAND_BUFFER_WAITS.with(|waits| waits.set(0));
}

pub(crate) fn resident_codestream_command_buffer_waits_for_test() -> usize {
    RESIDENT_CODESTREAM_COMMAND_BUFFER_WAITS.with(Cell::get)
}

pub(crate) fn record_resident_codestream_command_buffer_wait() {
    RESIDENT_CODESTREAM_COMMAND_BUFFER_WAITS.with(|waits| waits.set(waits.get() + 1));
}

pub(crate) fn reset_direct_tier1_input_buffer_prepares_for_test() {
    DIRECT_TIER1_INPUT_BUFFER_PREPARES.with(|counter| counter.set(0));
}

pub(crate) fn direct_tier1_input_buffer_prepares_for_test() -> usize {
    DIRECT_TIER1_INPUT_BUFFER_PREPARES.with(Cell::get)
}

pub(crate) fn record_direct_tier1_input_buffer_prepare() {
    DIRECT_TIER1_INPUT_BUFFER_PREPARES.with(|counter| counter.set(counter.get() + 1));
}

pub(crate) fn reset_thread_hybrid_cpu_decode_inputs_for_test() {
    HYBRID_CPU_DECODE_INPUTS_FOR_THREAD.with(|counter| counter.set(0));
}

pub(crate) fn thread_hybrid_cpu_decode_inputs_for_test() -> usize {
    HYBRID_CPU_DECODE_INPUTS_FOR_THREAD.with(Cell::get)
}

pub(crate) fn reset_lossless_deinterleave_rct_fused_dispatches_for_test() {
    LOSSLESS_DEINTERLEAVE_RCT_FUSED_DISPATCHES.with(|dispatches| dispatches.set(0));
}

pub(crate) fn lossless_deinterleave_rct_fused_dispatches_for_test() -> usize {
    LOSSLESS_DEINTERLEAVE_RCT_FUSED_DISPATCHES.with(Cell::get)
}

pub(crate) fn record_lossless_deinterleave_rct_fused_dispatch() {
    LOSSLESS_DEINTERLEAVE_RCT_FUSED_DISPATCHES
        .with(|dispatches| dispatches.set(dispatches.get().saturating_add(1)));
}

pub(crate) fn reset_classic_gpu_token_pack_dispatches_for_test() {
    CLASSIC_GPU_TOKEN_PACK_DISPATCHES.with(|dispatches| dispatches.set(0));
}

pub(crate) fn classic_gpu_token_pack_dispatches_for_test() -> usize {
    CLASSIC_GPU_TOKEN_PACK_DISPATCHES.with(Cell::get)
}

pub(crate) fn record_classic_gpu_token_pack_dispatch() {
    CLASSIC_GPU_TOKEN_PACK_DISPATCHES
        .with(|dispatches| dispatches.set(dispatches.get().saturating_add(1)));
}

pub(crate) fn reset_classic_split_mq_byte_gpu_token_pack_dispatches_for_test() {
    CLASSIC_SPLIT_MQ_BYTE_GPU_TOKEN_PACK_DISPATCHES.with(|dispatches| dispatches.set(0));
}

pub(crate) fn classic_split_mq_byte_gpu_token_pack_dispatches_for_test() -> usize {
    CLASSIC_SPLIT_MQ_BYTE_GPU_TOKEN_PACK_DISPATCHES.with(Cell::get)
}

pub(crate) fn record_classic_split_mq_byte_gpu_token_pack_dispatch() {
    CLASSIC_SPLIT_MQ_BYTE_GPU_TOKEN_PACK_DISPATCHES
        .with(|dispatches| dispatches.set(dispatches.get().saturating_add(1)));
}

pub(crate) fn record_ht_batch_coefficient_copy_blit() {
    HT_BATCH_COEFFICIENT_COPY_BLITS.fetch_add(1, Ordering::Relaxed);
}

pub(crate) fn record_hybrid_stacked_component_batch() {
    HYBRID_STACKED_COMPONENT_BATCHES.fetch_add(1, Ordering::Relaxed);
}

pub(crate) fn record_hybrid_repeated_output_blit() {
    HYBRID_REPEATED_OUTPUT_BLITS.fetch_add(1, Ordering::Relaxed);
}

pub(crate) fn record_hybrid_cpu_decode_worker_init() {
    HYBRID_CPU_DECODE_WORKER_INITS.fetch_add(1, Ordering::Relaxed);
}

pub(crate) fn record_hybrid_cpu_decode_inputs(count: usize) {
    HYBRID_CPU_DECODE_INPUTS.fetch_add(count, Ordering::Relaxed);
    HYBRID_CPU_DECODE_INPUTS_FOR_THREAD
        .with(|counter| counter.set(counter.get().saturating_add(count)));
}

pub(crate) fn record_flattened_hybrid_cpu_decode_batch() {
    FLATTENED_HYBRID_CPU_DECODE_BATCHES.fetch_add(1, Ordering::Relaxed);
}
