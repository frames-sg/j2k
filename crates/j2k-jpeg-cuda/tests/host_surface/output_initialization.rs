// SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg(feature = "cuda-runtime")]

use j2k_core::PixelFormat;
use j2k_jpeg::DecodeRequest;
use j2k_jpeg_cuda::{Codec, CudaSession};
use j2k_test_support::cuda_jpeg_hardware_decode_gate;

use super::support::{BASELINE_420, OWNED_CUDA_RGB8_MAX_CHANNEL_DELTA};

#[test]
fn caller_owned_pitched_decode_initializes_every_addressable_output_byte() {
    if !cuda_jpeg_hardware_decode_gate(module_path!()) {
        return;
    }

    let dimensions = (16usize, 16usize);
    let row_bytes = dimensions.0 * PixelFormat::Rgb8.bytes_per_pixel();
    let pitch_bytes = row_bytes + 16;
    let output_len = pitch_bytes * (dimensions.1 - 1) + row_bytes;
    let mut session = CudaSession::default();

    // Bootstrap the session context without placing another candidate in the
    // output pool, then seed that pool with a known nonzero allocation.
    let bootstrap = session
        .take_owned_cuda_output_buffer(0)
        .expect("bootstrap CUDA context");
    let sentinel = bootstrap
        .context()
        .upload(&vec![0xa5; output_len])
        .expect("sentinel device output");
    let sentinel_ptr = sentinel.device_ptr();
    session
        .recycle_owned_cuda_output_buffer(sentinel)
        .expect("seed output pool");
    let output = session
        .take_owned_cuda_output_buffer(output_len)
        .expect("caller-owned output");
    assert_eq!(
        output.device_ptr(),
        sentinel_ptr,
        "test must reuse sentinel"
    );

    let stats = Codec::decode_tile_rgb8_into_cuda_buffer_with_session(
        BASELINE_420,
        &output,
        pitch_bytes,
        &mut session,
    )
    .expect("pitched owned CUDA decode");
    assert!(stats.used_owned_cuda_decode());

    let mut downloaded = vec![0u8; output_len];
    output
        .copy_to_host(&mut downloaded)
        .expect("download pitched output");
    let (expected, _) = j2k_jpeg::Decoder::new(BASELINE_420)
        .expect("host decoder")
        .decode_request(DecodeRequest::full(PixelFormat::Rgb8))
        .expect("host decode");
    for row in 0..dimensions.1 {
        let output_start = row * pitch_bytes;
        let expected_start = row * row_bytes;
        let max_delta = downloaded[output_start..output_start + row_bytes]
            .iter()
            .zip(&expected[expected_start..expected_start + row_bytes])
            .map(|(actual, expected)| actual.abs_diff(*expected))
            .max()
            .unwrap_or(0);
        assert!(max_delta <= OWNED_CUDA_RGB8_MAX_CHANNEL_DELTA);
        if row + 1 < dimensions.1 {
            assert!(
                downloaded[output_start + row_bytes..output_start + pitch_bytes]
                    .iter()
                    .all(|&byte| byte == 0),
                "pitched padding row {row} retained uninitialized sentinel bytes"
            );
        }
    }
}
