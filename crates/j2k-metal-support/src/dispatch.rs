// SPDX-License-Identifier: MIT OR Apache-2.0

use metal::{ComputeCommandEncoderRef, ComputePipelineState, MTLSize};

/// Construct a Metal dispatch size.
#[must_use]
pub const fn mtl_size(width: u64, height: u64, depth: u64) -> MTLSize {
    MTLSize {
        width,
        height,
        depth,
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
    encoder: &ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    width: u64,
) {
    encoder.dispatch_threads(
        mtl_size(width, 1, 1),
        one_d_threads_per_group(pipeline.thread_execution_width()),
    );
}

/// Dispatch a single compute thread.
pub fn dispatch_single_thread(encoder: &ComputeCommandEncoderRef) {
    encoder.dispatch_threads(mtl_size(1, 1, 1), mtl_size(1, 1, 1));
}

/// Dispatch a two-dimensional compute workload using the pipeline's SIMD width.
pub fn dispatch_2d_pipeline(
    encoder: &ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    dims: (u32, u32),
) {
    encoder.dispatch_threads(
        mtl_size(u64::from(dims.0), u64::from(dims.1), 1),
        two_d_threads_per_group(
            pipeline.thread_execution_width(),
            pipeline.max_total_threads_per_threadgroup(),
        ),
    );
}

/// Dispatch a three-dimensional compute workload using a 2D threadgroup shape.
pub fn dispatch_3d_pipeline(
    encoder: &ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    dims: (u32, u32, u32),
) {
    encoder.dispatch_threads(
        mtl_size(u64::from(dims.0), u64::from(dims.1), u64::from(dims.2)),
        two_d_threads_per_group(
            pipeline.thread_execution_width(),
            pipeline.max_total_threads_per_threadgroup(),
        ),
    );
}
