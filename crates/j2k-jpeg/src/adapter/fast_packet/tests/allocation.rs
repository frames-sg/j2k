// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::adapter::fast_packet::allocation::{
    checked_actual_vec_live_bytes, checked_color_packet_initial_live_bytes,
    checked_color_packet_live_bytes, checked_entropy_live_bytes, checked_gray_packet_live_bytes,
    host_allocation_error,
};
use crate::adapter::fast_packet::{FastPacketError, JpegEntropyCheckpointV1};
use crate::error::JpegError;

#[test]
fn entropy_and_packet_live_byte_boundaries_are_exact() {
    let entropy_len = 101;
    let restart_count = 7;
    let checkpoint_count = 5;
    let initial_live_bytes = 13;
    let terminated_copy_bytes = 7;
    let entropy_bytes = entropy_len + restart_count * core::mem::size_of::<u32>();

    assert_eq!(
        checked_entropy_live_bytes(entropy_len, restart_count, entropy_bytes)
            .expect("E+R exact boundary must fit"),
        entropy_bytes
    );
    let error = checked_entropy_live_bytes(entropy_len, restart_count, entropy_bytes - 1)
        .expect_err("E+R one byte over the cap must fail");
    assert!(matches!(
        error,
        FastPacketError::Decode(JpegError::MemoryCapExceeded { requested, cap })
            if requested == entropy_bytes && cap == entropy_bytes - 1
    ));

    let color_bytes = initial_live_bytes
        + entropy_bytes
        + checkpoint_count * core::mem::size_of::<JpegEntropyCheckpointV1>()
        + terminated_copy_bytes;
    assert_eq!(
        checked_color_packet_live_bytes(
            initial_live_bytes,
            entropy_len,
            restart_count,
            checkpoint_count,
            terminated_copy_bytes,
            color_bytes,
        )
        .expect("retained+E+R+P+T exact boundary must fit"),
        color_bytes
    );
    let error = checked_color_packet_live_bytes(
        initial_live_bytes,
        entropy_len,
        restart_count,
        checkpoint_count,
        terminated_copy_bytes,
        color_bytes - 1,
    )
    .expect_err("retained+E+R+P+T one byte over the cap must fail");
    assert!(matches!(
        error,
        FastPacketError::Decode(JpegError::MemoryCapExceeded { requested, cap })
            if requested == color_bytes && cap == color_bytes - 1
    ));

    let gray_bytes = entropy_bytes + terminated_copy_bytes;
    assert_eq!(
        checked_gray_packet_live_bytes(
            entropy_len,
            restart_count,
            terminated_copy_bytes,
            gray_bytes,
        ),
        Ok(gray_bytes)
    );
}

#[test]
fn cached_packet_external_input_and_decoder_share_one_exact_initial_boundary() {
    let external_input_bytes = 13;
    let retained_decoder_bytes = 17;
    let exact = external_input_bytes + retained_decoder_bytes;

    assert_eq!(
        checked_color_packet_initial_live_bytes(
            external_input_bytes,
            retained_decoder_bytes,
            exact,
        ),
        Ok(exact)
    );
    assert_eq!(
        checked_color_packet_initial_live_bytes(
            external_input_bytes,
            retained_decoder_bytes,
            exact - 1,
        ),
        Err(FastPacketError::Decode(JpegError::MemoryCapExceeded {
            requested: exact,
            cap: exact - 1,
        }))
    );
}

#[test]
fn allocator_returned_packet_capacity_is_postchecked() {
    assert_eq!(
        checked_actual_vec_live_bytes::<u8>(1, 2, 2),
        Err(FastPacketError::Decode(JpegError::MemoryCapExceeded {
            requested: 3,
            cap: 2,
        }))
    );
}

#[test]
fn allocation_overflow_and_reserve_failure_keep_typed_categories() {
    assert_eq!(
        checked_entropy_live_bytes(usize::MAX, 1, usize::MAX),
        Err(FastPacketError::Decode(JpegError::MemoryCapExceeded {
            requested: usize::MAX,
            cap: usize::MAX,
        }))
    );
    assert_eq!(
        host_allocation_error(4096),
        FastPacketError::Decode(JpegError::HostAllocationFailed { bytes: 4096 })
    );
}
