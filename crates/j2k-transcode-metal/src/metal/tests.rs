// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    dwt97_block_value_count, projection_dispatch_sizes, MetalTranscodeError,
    METAL_DCT97_UNSUPPORTED_GRID,
};

#[test]
fn projection_dispatch_sizes_use_16_by_8_threadgroups() {
    let (threads, threadgroup) = projection_dispatch_sizes(5, 6, 7);

    assert_eq!((threads.width, threads.height, threads.depth), (5, 6, 7));
    assert_eq!(
        (threadgroup.width, threadgroup.height, threadgroup.depth),
        (16, 8, 1)
    );
}

#[test]
fn dwt97_block_value_count_rejects_overflow() {
    assert_eq!(dwt97_block_value_count(2), Ok(128));
    assert_eq!(
        dwt97_block_value_count(usize::MAX),
        Err(MetalTranscodeError::UnsupportedJob(
            METAL_DCT97_UNSUPPORTED_GRID
        ))
    );
}
