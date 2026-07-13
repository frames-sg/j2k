// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

#[cfg(target_os = "macos")]
#[test]
#[expect(
    clippy::cast_sign_loss,
    reason = "bounded synthetic pixels are nonnegative"
)]
fn classic_inflight_one_waits_before_submitting_the_next_chunk() {
    if !should_run_metal_runtime() {
        return;
    }

    let first = (0..8 * 8)
        .map(|index| ((index * 7 + 3) & 0xFF) as u8)
        .collect::<Vec<_>>();
    let second = (0..8 * 8)
        .map(|index| ((index * 13 + 5) & 0xFF) as u8)
        .collect::<Vec<_>>();
    let session = crate::MetalBackendSession::system_default().expect("Metal session");
    let first_buffer =
        crate::benchmark_private_buffer_with_bytes(&session, &first).expect("first private input");
    let second_buffer = crate::benchmark_private_buffer_with_bytes(&session, &second)
        .expect("second private input");
    let tiles =
        [&first_buffer, &second_buffer].map(|buffer| super::super::MetalLosslessEncodeTile {
            buffer,
            byte_offset: 0,
            width: 8,
            height: 8,
            pitch_bytes: 8,
            output_width: 8,
            output_height: 8,
            format: PixelFormat::Gray8,
        });
    let options = lossless_options! {
        backend: EncodeBackendPreference::RequireDevice,
        block_coding_mode: J2kBlockCodingMode::Classic,
        validation: J2kEncodeValidation::External,
    };

    compute::reset_resident_codestream_command_buffer_waits_for_test();
    super::super::reset_resident_schedule_counters_for_test();
    let outcome = super::super::encode_lossless_from_padded_metal_buffers_to_metal_batch(
        &tiles,
        &options,
        &session,
        super::super::MetalLosslessEncodeConfig {
            gpu_encode_inflight_tiles: Some(1),
            gpu_encode_memory_budget_bytes: Some(1024 * 1024 * 1024),
        },
    )
    .expect("classic resident batch");

    assert_eq!(outcome.outcomes.len(), 2);
    assert_eq!(
        compute::resident_codestream_command_buffer_waits_for_test(),
        2
    );
    assert_eq!(
        super::super::resident_schedule_counters_for_test(),
        (0, 1, 2)
    );
    for (frame, expected) in outcome.outcomes.iter().zip([first, second]) {
        let codestream = frame
            .encoded
            .codestream_bytes()
            .expect("scheduled Metal codestream bytes are CPU-readable");
        let decoded = Image::new(&codestream, &DecodeSettings::default())
            .expect("scheduled codestream parses")
            .decode_native()
            .expect("scheduled codestream decodes");
        assert_decoded_bytes_match(&decoded.data, &expected);
    }
}

#[cfg(target_os = "macos")]
#[test]
#[expect(
    clippy::cast_sign_loss,
    reason = "bounded synthetic pixels are nonnegative"
)]
fn dropping_submission_does_not_schedule_remaining_resident_chunks() {
    if !should_run_metal_runtime() {
        return;
    }

    let inputs = (0..4)
        .map(|seed| {
            (0..8 * 8)
                .map(|index| ((index * 11 + seed) & 0xFF) as u8)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let session = crate::MetalBackendSession::system_default().expect("Metal session");
    let buffers = inputs
        .iter()
        .map(|bytes| {
            crate::benchmark_private_buffer_with_bytes(&session, bytes)
                .expect("private benchmark input buffer")
        })
        .collect::<Vec<_>>();
    let tiles = buffers
        .iter()
        .map(|buffer| super::super::MetalLosslessEncodeTile {
            buffer,
            byte_offset: 0,
            width: 8,
            height: 8,
            pitch_bytes: 8,
            output_width: 8,
            output_height: 8,
            format: PixelFormat::Gray8,
        })
        .collect::<Vec<_>>();
    let options = lossless_options! {
        backend: EncodeBackendPreference::RequireDevice,
        block_coding_mode: J2kBlockCodingMode::HighThroughput,
        validation: J2kEncodeValidation::External,
    };

    super::super::reset_resident_schedule_counters_for_test();
    let submitted = crate::submit_lossless_batch_to_metal(
        super::super::MetalLosslessEncodeBatchRequest {
            tiles: &tiles,
            staging: super::super::MetalEncodeInputStaging::AlreadyPaddedContiguous,
            config: super::super::MetalLosslessEncodeConfig {
                gpu_encode_inflight_tiles: Some(1),
                gpu_encode_memory_budget_bytes: Some(1024 * 1024 * 1024),
            },
        },
        &options,
        &session,
    )
    .expect("resident submission");
    assert_eq!(
        super::super::resident_schedule_counters_for_test(),
        (1, 1, 1),
        "submit must schedule exactly one active chunk"
    );

    drop(submitted);

    assert_eq!(
        super::super::resident_schedule_counters_for_test(),
        (0, 1, 1),
        "drop must release scheduler ownership without scheduling a suspended chunk"
    );
}
