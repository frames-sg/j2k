// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{fs, path::Path};

use super::super::{assert_pattern_checks, repo_root, PatternCheck};

mod abi;
mod allocation;
mod lifecycle;
mod submit;
mod validation;

#[test]
fn cuda_runtime_execution_and_memory_modules_stay_focused() {
    let root = repo_root();
    assert_runtime_module_line_limits(root);

    let execution = fs::read_to_string(root.join("crates/j2k-cuda-runtime/src/execution.rs"))
        .expect("read CUDA runtime execution facade");
    let execution_memory_ops =
        fs::read_to_string(root.join("crates/j2k-cuda-runtime/src/execution/memory_ops.rs"))
            .expect("read CUDA runtime execution memory operations");
    let memory = fs::read_to_string(root.join("crates/j2k-cuda-runtime/src/memory.rs"))
        .expect("read CUDA runtime memory facade");
    let memory_ranges =
        fs::read_to_string(root.join("crates/j2k-cuda-runtime/src/memory/ranges.rs"))
            .expect("read CUDA runtime checked device ranges");
    assert_pattern_checks(&[
        PatternCheck::new("CUDA execution focused module graph", &execution)
            .required(&[
                "pub(crate) mod completion;",
                "mod events;",
                "mod memory_ops;",
                "mod queued;",
                "pub use queued::{",
            ])
            .forbidden(&[
                "pub(crate) enum CudaSynchronizationOutcome",
                "pub(crate) struct CudaEvent",
                "pub struct CudaQueuedExecution",
                "fn memset_d8(",
                "fn memset_d32(",
            ]),
        PatternCheck::new("CUDA memset ownership and bounds", &execution_memory_ops).required(&[
            "fn validate_memset_target(",
            "fn memset_d8(",
            "cuMemsetD8_v2",
            "fn memset_d32(",
            "cuMemsetD32_v2",
            "with_current_resource_operation",
        ]),
        PatternCheck::new("CUDA memory focused module graph", &memory)
            .required(&[
                "mod pool;",
                "mod pinned_staging;",
                "mod ranges;",
                "pub use self::pool::{",
                "CudaBufferPool, CudaBufferPoolDiagnostics, CudaBufferPoolLimits",
                "CudaBufferPoolReuseGuard",
                "pub(crate) use self::ranges::CheckedDeviceBufferRanges;",
            ])
            .forbidden(&[
                "pub struct CudaBufferPool {",
                "pub(crate) struct CudaBufferPoolReuseGuard",
                "pub struct CudaPooledDeviceBuffer",
                "fn checked_device_ranges_overlap(",
                "fn recycle_pinned_upload_staging(",
            ]),
        PatternCheck::new("CUDA checked device range sweeps", &memory_ranges).required(&[
            "struct CheckedDeviceBufferRanges",
            "fn checked_nonempty_device_range(",
            "try_vec_with_capacity(minimum_count)?",
            "try_vec_reserve(&mut sorted, 1)?",
            "sort_unstable_by_key",
            "fn first_self_overlap(",
            "fn first_cross_overlap(",
            "range_sets_ignore_empty_ranges_and_preserve_original_indices",
            "large_disjoint_self_sweep_avoids_quadratic_pair_scanning",
            "self_sweep_finds_overlap_hidden_after_many_disjoint_ranges",
            "large_disjoint_cross_sweep_avoids_quadratic_pair_scanning",
            "cross_sweep_finds_overlap_hidden_at_end_of_large_sets",
        ]),
    ]);
}

fn assert_runtime_module_line_limits(root: &Path) {
    let modules = [
        ("execution.rs", 340usize),
        ("execution/completion.rs", 170),
        ("execution/events.rs", 310),
        ("execution/events/handles.rs", 175),
        ("execution/memory_ops.rs", 100),
        ("execution/queued.rs", 260),
        ("memory.rs", 650),
        ("memory/pool.rs", 680),
        ("memory/pool/cache_policy.rs", 400),
        ("memory/pool/cache_policy/tests.rs", 125),
        ("memory/pool/cache_policy/tests/device.rs", 150),
        ("memory/pinned_staging.rs", 175),
        ("memory/pinned_staging/operations.rs", 275),
        ("memory/pinned_staging/operations/api.rs", 75),
        ("memory/pinned_staging/operations/checkout.rs", 125),
        ("memory/pinned_staging/operations/checkout/tests.rs", 75),
        ("memory/pinned_staging/operations/gate.rs", 50),
        ("memory/pinned_staging/operations/policy.rs", 75),
        ("memory/pinned_staging/operations/policy/tests.rs", 125),
        ("memory/pinned_staging/operations/recycle.rs", 150),
        ("memory/pinned_staging/pool.rs", 400),
        ("memory/pinned_staging/pool/active.rs", 225),
        ("memory/pinned_staging/pool/active/tests.rs", 100),
        ("memory/pinned_staging/pool/diagnostics.rs", 100),
        ("memory/pinned_staging/pool/tests.rs", 250),
        ("memory/pinned_staging/tests.rs", 125),
        ("memory/pool/readback.rs", 50),
        ("memory/pool/reuse_guard.rs", 150),
        ("memory/pool/size_buckets.rs", 100),
        ("memory/pool/size_buckets/inventory.rs", 50),
        ("memory/ranges.rs", 300),
    ];
    for (relative, max_lines) in modules {
        let path = root.join("crates/j2k-cuda-runtime/src").join(relative);
        let source = fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
        let line_count = source.lines().count();
        assert!(
            line_count < max_lines,
            "crates/j2k-cuda-runtime/src/{relative} has {line_count} lines; split it before reaching {max_lines}"
        );
    }
}

#[test]
fn cuda_host_band_and_diagnostic_owners_remain_move_only() {
    let root = repo_root();
    let transcode_types =
        fs::read_to_string(root.join("crates/j2k-cuda-runtime/src/transcode/types.rs"))
            .expect("read CUDA transcode host-owner types");
    let jpeg_types = fs::read_to_string(root.join("crates/j2k-cuda-runtime/src/jpeg/types.rs"))
        .expect("read CUDA JPEG host-owner types");

    assert_pattern_checks(&[
        PatternCheck::new("CUDA transcode host owners", &transcode_types)
            .required(&[
                "#[derive(Debug, PartialEq, Eq)]\n#[doc(hidden)]\npub struct CudaTranscodeReversible53Bands",
                "#[derive(Debug, PartialEq)]\n#[doc(hidden)]\npub struct CudaTranscodeDwt97Bands",
                "#[derive(Debug, PartialEq, Eq)]\n#[doc(hidden)]\npub struct CudaHtj2k97CodeblockBands",
                "This owner is move-only",
            ]),
        PatternCheck::new("CUDA JPEG diagnostic host owner", &jpeg_types).required(&[
            "#[derive(Debug, Eq, PartialEq)]\npub struct CudaJpegChunkedEntropyReport",
            "The report is move-only",
        ]),
    ]);
}
