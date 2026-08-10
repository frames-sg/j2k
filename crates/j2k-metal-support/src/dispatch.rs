// SPDX-License-Identifier: MIT OR Apache-2.0

use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLComputeCommandEncoder, MTLComputePipelineState, MTLSize};

/// Construct a Metal dispatch size.
#[must_use]
#[expect(
    clippy::cast_possible_truncation,
    reason = "Metal is supported only on 64-bit macOS targets, where NSUInteger and u64 have equal ranges"
)]
pub const fn mtl_size(width: u64, height: u64, depth: u64) -> MTLSize {
    // J2K supports only 64-bit macOS targets, where NSUInteger and u64 have
    // identical ranges.
    MTLSize {
        width: width as usize,
        height: height as usize,
        depth: depth as usize,
    }
}

/// One-dimensional thread-group size with empty SIMD widths clamped to one.
#[must_use]
pub const fn one_d_threads_per_group(simd_width: u64) -> MTLSize {
    mtl_size(if simd_width == 0 { 1 } else { simd_width }, 1, 1)
}

/// Two-dimensional thread-group size preserving SIMD width and filling height.
#[must_use]
pub const fn two_d_threads_per_group(simd_width: u64, max_threads: u64) -> MTLSize {
    let width = if simd_width == 0 { 1 } else { simd_width };
    let max_threads = if max_threads < width {
        width
    } else {
        max_threads
    };
    mtl_size(width, max_threads / width, 1)
}

/// Dispatch a one-dimensional compute workload with one SIMD group per threadgroup.
pub fn dispatch_1d_pipeline(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    width: u64,
) {
    encoder.dispatchThreads_threadsPerThreadgroup(
        mtl_size(width, 1, 1),
        one_d_threads_per_group(pipeline.threadExecutionWidth() as u64),
    );
}

/// Dispatch a single compute thread.
pub fn dispatch_single_thread(encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>) {
    encoder.dispatchThreads_threadsPerThreadgroup(mtl_size(1, 1, 1), mtl_size(1, 1, 1));
}

/// Dispatch a two-dimensional compute workload using the pipeline's SIMD width.
pub fn dispatch_2d_pipeline(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    dims: (u32, u32),
) {
    encoder.dispatchThreads_threadsPerThreadgroup(
        mtl_size(u64::from(dims.0), u64::from(dims.1), 1),
        two_d_threads_per_group(
            pipeline.threadExecutionWidth() as u64,
            pipeline.maxTotalThreadsPerThreadgroup() as u64,
        ),
    );
}

/// Dispatch a three-dimensional compute workload using a 2D threadgroup shape.
pub fn dispatch_3d_pipeline(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    dims: (u32, u32, u32),
) {
    encoder.dispatchThreads_threadsPerThreadgroup(
        mtl_size(u64::from(dims.0), u64::from(dims.1), u64::from(dims.2)),
        two_d_threads_per_group(
            pipeline.threadExecutionWidth() as u64,
            pipeline.maxTotalThreadsPerThreadgroup() as u64,
        ),
    );
}
