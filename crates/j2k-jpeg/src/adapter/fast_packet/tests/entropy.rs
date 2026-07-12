// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::adapter::fast_packet::entropy::extract_entropy_segments_with_cap;
use crate::adapter::fast_packet::FastPacketError;
use crate::error::JpegError;

#[test]
fn entropy_extraction_ignores_trailing_bytes_when_reserving() {
    let bytes = [0x11, 0xff, 0x00, 0x22, 0xff, 0xd9, 0xaa, 0xbb];
    let segments = extract_entropy_segments_with_cap(&bytes, None, 7)
        .expect("only the three bytes before EOI count toward the allocation");
    assert_eq!(segments.entropy_bytes, [0x11, 0xff, 0x22]);
    assert_eq!(segments.restart_offsets, [0]);

    let error = extract_entropy_segments_with_cap(&bytes, None, 6)
        .expect_err("entropy plus restart metadata exceeds the aggregate cap");
    assert!(matches!(
        error,
        FastPacketError::Decode(JpegError::MemoryCapExceeded {
            requested: 7,
            cap: 6,
        })
    ));
}

#[test]
fn entropy_extraction_growth_never_crosses_the_configured_cap() {
    const ENTROPY_LEN: usize = 4 * 1024 + 1;
    let mut bytes = vec![0x11; ENTROPY_LEN];
    bytes.extend_from_slice(&[0xff, 0xd9]);

    let allocation_bytes = ENTROPY_LEN + core::mem::size_of::<u32>();
    let segments = extract_entropy_segments_with_cap(&bytes, None, allocation_bytes)
        .expect("the exact logical entropy size fits the cap across a growth boundary");
    assert_eq!(segments.entropy_bytes.len(), ENTROPY_LEN);

    let error = extract_entropy_segments_with_cap(&bytes, None, allocation_bytes - 1)
        .expect_err("the aggregate logical allocation must fail before materialization");
    assert!(matches!(
        error,
        FastPacketError::Decode(JpegError::MemoryCapExceeded { requested, cap })
            if requested == allocation_bytes && cap == allocation_bytes - 1
    ));
}

#[test]
fn entropy_extraction_bounds_entropy_and_restart_offsets_together() {
    let bytes = [0x00, 0xff, 0xd0, 0x00, 0xff, 0xd1, 0x00, 0xff, 0xd9];
    let expected_bytes = 3 + 3 * core::mem::size_of::<u32>();
    let segments = extract_entropy_segments_with_cap(&bytes, Some(1), expected_bytes)
        .expect("the exact entropy and restart-offset allocation fits");
    assert_eq!(segments.entropy_bytes, [0, 0, 0]);
    assert_eq!(segments.restart_offsets, [0, 1, 2]);

    let error = extract_entropy_segments_with_cap(&bytes, Some(1), expected_bytes - 1)
        .expect_err("entropy plus restart offsets exceed the aggregate cap");
    assert!(matches!(
        error,
        FastPacketError::Decode(JpegError::MemoryCapExceeded { requested, cap })
            if requested == expected_bytes && cap == expected_bytes - 1
    ));

    let error = extract_entropy_segments_with_cap(&[0xff, 0xd9], None, 0)
        .expect_err("the mandatory zero restart offset needs one u32 allocation slot");
    assert!(matches!(
        error,
        FastPacketError::Decode(JpegError::MemoryCapExceeded {
            requested: 4,
            cap: 0,
        })
    ));
}
