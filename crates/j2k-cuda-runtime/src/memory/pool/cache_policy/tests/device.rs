// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::{CudaBufferPool, CudaBufferPoolLimits};
use crate::CudaContext;

fn cuda_runtime_gate() -> bool {
    j2k_test_support::cuda_runtime_gate(module_path!())
}

#[test]
fn first_fit_cache_respects_actual_byte_and_count_high_water() {
    if !cuda_runtime_gate() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let limits = CudaBufferPoolLimits {
        max_cached_bytes: 96,
        max_cached_buffers: 2,
        max_size_buckets: 0,
    };
    let pool = CudaBufferPool::with_limits(context.clone(), limits);
    let clone = pool.clone();
    let buffers = [32usize, 48, 64, 128]
        .into_iter()
        .map(|len| context.allocate(len).expect("device buffer"))
        .collect::<Vec<_>>();
    for buffer in buffers {
        pool.recycle(buffer).expect("completed buffer recycle");
    }

    let diagnostics = pool.diagnostics().expect("pool diagnostics");
    assert_eq!(diagnostics, clone.diagnostics().expect("clone diagnostics"));
    assert_eq!(diagnostics.limits, limits);
    assert_eq!(diagnostics.cached_buffers, 1);
    assert_eq!(diagnostics.cached_bytes, 64);
    assert_eq!(diagnostics.evicted_buffers, 2);
    assert_eq!(diagnostics.rejected_buffers, 1);
    assert!(diagnostics.peak_cached_buffers <= limits.max_cached_buffers);
    assert!(diagnostics.peak_cached_bytes <= limits.max_cached_bytes);
}

#[test]
fn best_fit_cache_evicts_largest_oldest_at_bucket_limit() {
    if !cuda_runtime_gate() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let limits = CudaBufferPoolLimits {
        max_cached_bytes: 1_024,
        max_cached_buffers: 4,
        max_size_buckets: 2,
    };
    let pool = CudaBufferPool::best_fit_with_limits(context.clone(), limits);
    for len in [32usize, 48, 64] {
        pool.recycle(context.allocate(len).expect("device buffer"))
            .expect("completed buffer recycle");
    }

    let diagnostics = pool.diagnostics().expect("pool diagnostics");
    assert_eq!(diagnostics.cached_buffers, 2);
    assert_eq!(diagnostics.cached_bytes, 96);
    assert_eq!(diagnostics.cached_size_buckets, 2);
    assert_eq!(diagnostics.evicted_buffers, 1);
    assert!(diagnostics.peak_cached_size_buckets <= limits.max_size_buckets);
}

#[test]
fn reuse_hold_accounts_oversize_buffer_until_completion() {
    if !cuda_runtime_gate() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let limits = CudaBufferPoolLimits {
        max_cached_bytes: 32,
        max_cached_buffers: 1,
        max_size_buckets: 0,
    };
    let pool = CudaBufferPool::with_limits(context.clone(), limits);
    let hold = pool.defer_reuse().expect("reuse hold");
    pool.recycle(context.allocate(40).expect("device buffer"))
        .expect("deferred recycle");

    let deferred = pool.diagnostics().expect("deferred diagnostics");
    assert_eq!(deferred.cached_buffers, 0);
    assert_eq!(deferred.deferred_buffers, 1);
    assert_eq!(deferred.deferred_bytes, 40);
    assert_eq!(deferred.reuse_holds, 1);

    // The allocation was never submitted to CUDA work, so completion is
    // established without a synchronization operation in this test.
    hold.release().expect("release completed hold");
    let released = pool.diagnostics().expect("released diagnostics");
    assert_eq!(released.deferred_buffers, 0);
    assert_eq!(released.deferred_bytes, 0);
    assert_eq!(released.cached_buffers, 0);
    assert_eq!(released.rejected_buffers, 1);
}
