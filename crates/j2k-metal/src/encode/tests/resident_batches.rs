// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

#[cfg(target_os = "macos")]
#[test]
#[expect(
    clippy::cast_possible_truncation,
    reason = "32x32 batch fixture dimensions and indices fit u8/u32 fields"
)]
fn metal_rgb8_ht_batch_uses_fused_deinterleave_rct_kernel() {
    const WIDTH: usize = 32;
    const HEIGHT: usize = 32;

    if !should_run_metal_runtime() {
        return;
    }

    let first: Vec<u8> = (0..WIDTH * HEIGHT * 3)
        .map(|idx| ((idx * 29 + idx / 7) & 0xFF) as u8)
        .collect();
    let second: Vec<u8> = (0..WIDTH * HEIGHT * 3)
        .map(|idx| 255u8.wrapping_sub(((idx * 13 + idx / 5) & 0xFF) as u8))
        .collect();
    let session = crate::MetalBackendSession::system_default().expect("Metal session");
    let first_buffer = crate::benchmark_private_buffer_with_bytes(&session, &first)
        .expect("private benchmark input buffer");
    let second_buffer = crate::benchmark_private_buffer_with_bytes(&session, &second)
        .expect("private benchmark input buffer");
    let tiles = [
        super::super::MetalLosslessEncodeTile {
            buffer: &first_buffer,
            byte_offset: 0,
            width: WIDTH as u32,
            height: HEIGHT as u32,
            pitch_bytes: WIDTH * 3,
            output_width: WIDTH as u32,
            output_height: HEIGHT as u32,
            format: PixelFormat::Rgb8,
        },
        super::super::MetalLosslessEncodeTile {
            buffer: &second_buffer,
            byte_offset: 0,
            width: WIDTH as u32,
            height: HEIGHT as u32,
            pitch_bytes: WIDTH * 3,
            output_width: WIDTH as u32,
            output_height: HEIGHT as u32,
            format: PixelFormat::Rgb8,
        },
    ];
    let options = lossless_options! {
        backend: EncodeBackendPreference::RequireDevice,
        block_coding_mode: J2kBlockCodingMode::HighThroughput,
        validation: J2kEncodeValidation::External,
    };

    compute::reset_lossless_deinterleave_rct_fused_dispatches_for_test();
    let encoded = super::super::encode_lossless_from_padded_metal_buffers_to_metal_with_report(
        &tiles, &options, &session,
    )
    .expect("Metal RGB8 HTJ2K batch encode");

    assert_eq!(encoded.len(), 2);
    assert!(
        compute::lossless_deinterleave_rct_fused_dispatches_for_test() > 0,
        "RGB8 resident lossless encode should fuse deinterleave and RCT"
    );
    for (frame, expected) in encoded.iter().zip([first, second]) {
        let codestream = frame
            .encoded
            .codestream_bytes()
            .expect("Metal codestream bytes are CPU-readable");
        let decoded = Image::new(&codestream, &DecodeSettings::default())
            .expect("codestream parses")
            .decode_native()
            .expect("codestream decodes");
        assert_eq!(decoded.data, expected);
    }
}

#[cfg(target_os = "macos")]
#[test]
#[expect(
    clippy::cast_sign_loss,
    reason = "bounded synthetic pixel expression is nonnegative"
)]
fn metal_buffer_lossless_batch_encodes_padded_contiguous_inputs() {
    if !should_run_metal_runtime() {
        return;
    }

    let first: Vec<u8> = (0..8 * 8 * 3).map(|i| ((i * 7) & 0xFF) as u8).collect();
    let second: Vec<u8> = (0..8 * 8 * 3)
        .map(|i| ((i * 13 + 5) & 0xFF) as u8)
        .collect();
    let session = crate::MetalBackendSession::system_default().expect("Metal session");
    let first_buffer = session.device().new_buffer_with_data(
        first.as_ptr().cast(),
        first.len() as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let second_buffer = session.device().new_buffer_with_data(
        second.as_ptr().cast(),
        second.len() as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let tiles = [
        super::super::MetalLosslessEncodeTile {
            buffer: &first_buffer,
            byte_offset: 0,
            width: 8,
            height: 8,
            pitch_bytes: 8 * 3,
            output_width: 8,
            output_height: 8,
            format: PixelFormat::Rgb8,
        },
        super::super::MetalLosslessEncodeTile {
            buffer: &second_buffer,
            byte_offset: 0,
            width: 8,
            height: 8,
            pitch_bytes: 8 * 3,
            output_width: 8,
            output_height: 8,
            format: PixelFormat::Rgb8,
        },
    ];

    let encoded = super::super::encode_lossless_from_padded_metal_buffers_with_report(
        &tiles,
        &lossless_options! {
            backend: EncodeBackendPreference::RequireDevice,
        },
        &session,
    )
    .expect("Metal padded buffer batch lossless encode");

    assert_eq!(encoded.len(), 2);
    for (frame, expected) in encoded.iter().zip([first, second]) {
        assert_eq!(frame.encoded.backend, BackendKind::Metal);
        assert!(!frame.input_copy_used);
        let decoded = Image::new(&frame.encoded.codestream, &DecodeSettings::default())
            .expect("codestream parses")
            .decode_native()
            .expect("codestream decodes");
        assert_eq!(decoded.data, expected);
    }
}

#[cfg(target_os = "macos")]
#[test]
#[expect(
    clippy::cast_sign_loss,
    reason = "bounded synthetic pixel expression is nonnegative"
)]
fn metal_padded_private_batch_encode_to_metal_buffers_exposes_per_frame_bytes() {
    if !should_run_metal_runtime() {
        return;
    }

    let first: Vec<u8> = (0..8 * 8 * 3).map(|i| ((i * 17) & 0xFF) as u8).collect();
    let second: Vec<u8> = (0..8 * 8 * 3)
        .map(|i| 255u8.wrapping_sub(((i * 23) & 0xFF) as u8))
        .collect();
    let session = crate::MetalBackendSession::system_default().expect("Metal session");
    let first_buffer = crate::benchmark_private_buffer_with_bytes(&session, &first)
        .expect("private benchmark input buffer");
    let second_buffer = crate::benchmark_private_buffer_with_bytes(&session, &second)
        .expect("private benchmark input buffer");
    let tiles = [
        super::super::MetalLosslessEncodeTile {
            buffer: &first_buffer,
            byte_offset: 0,
            width: 8,
            height: 8,
            pitch_bytes: 8 * 3,
            output_width: 8,
            output_height: 8,
            format: PixelFormat::Rgb8,
        },
        super::super::MetalLosslessEncodeTile {
            buffer: &second_buffer,
            byte_offset: 0,
            width: 8,
            height: 8,
            pitch_bytes: 8 * 3,
            output_width: 8,
            output_height: 8,
            format: PixelFormat::Rgb8,
        },
    ];

    let encoded = super::super::encode_lossless_from_padded_metal_buffers_to_metal_with_report(
        &tiles,
        &lossless_options! {
            backend: EncodeBackendPreference::RequireDevice,
        },
        &session,
    )
    .expect("Metal padded buffer batch lossless encode to Metal buffers");

    assert_eq!(encoded.len(), 2);
    assert_eq!(
        encoded[0].encoded.codestream_buffer.as_ptr(),
        encoded[1].encoded.codestream_buffer.as_ptr(),
        "classic J2K resident batch encode should assemble codestreams into one shared batch buffer"
    );
    assert_eq!(encoded[0].encoded.byte_offset, 0);
    assert!(
        encoded[1].encoded.byte_offset > 0,
        "second classic J2K batch codestream should be a nonzero slice into the shared batch buffer"
    );
    for (frame, expected) in encoded.iter().zip([first, second]) {
        assert!(!frame.input_copy_used);
        assert!(frame.resident.coefficient_prep_used);
        assert!(frame.resident.packetization_used);
        assert!(frame.resident.codestream_assembly_used);
        let codestream = frame
            .encoded
            .codestream_bytes()
            .expect("Metal codestream bytes are CPU-readable");
        let decoded = Image::new(&codestream, &DecodeSettings::default())
            .expect("codestream parses")
            .decode_native()
            .expect("codestream decodes");
        assert_eq!(decoded.data, expected);
    }
}

#[cfg(target_os = "macos")]
#[test]
#[expect(
    clippy::cast_sign_loss,
    reason = "bounded synthetic pixel expression is nonnegative"
)]
fn metal_padded_private_batch_dwt_encode_to_metal_buffers_round_trips() {
    if !should_run_metal_runtime() {
        return;
    }

    let first: Vec<u8> = (0..128 * 128 * 3)
        .map(|i| ((i * 17 + i / 3) & 0xFF) as u8)
        .collect();
    let second: Vec<u8> = (0..128 * 128 * 3)
        .map(|i| 255u8.wrapping_sub(((i * 23 + i / 5) & 0xFF) as u8))
        .collect();
    let session = crate::MetalBackendSession::system_default().expect("Metal session");
    let first_buffer = crate::benchmark_private_buffer_with_bytes(&session, &first)
        .expect("private benchmark input buffer");
    let second_buffer = crate::benchmark_private_buffer_with_bytes(&session, &second)
        .expect("private benchmark input buffer");
    let tiles = [
        super::super::MetalLosslessEncodeTile {
            buffer: &first_buffer,
            byte_offset: 0,
            width: 128,
            height: 128,
            pitch_bytes: 128 * 3,
            output_width: 128,
            output_height: 128,
            format: PixelFormat::Rgb8,
        },
        super::super::MetalLosslessEncodeTile {
            buffer: &second_buffer,
            byte_offset: 0,
            width: 128,
            height: 128,
            pitch_bytes: 128 * 3,
            output_width: 128,
            output_height: 128,
            format: PixelFormat::Rgb8,
        },
    ];

    let encoded = super::super::encode_lossless_from_padded_metal_buffers_to_metal_with_report(
        &tiles,
        &lossless_options! {
            backend: EncodeBackendPreference::RequireDevice,
            validation: J2kEncodeValidation::External,
        },
        &session,
    )
    .expect("Metal padded DWT buffer batch lossless encode to Metal buffers");

    assert_eq!(encoded.len(), 2);
    for (frame, expected) in encoded.iter().zip([first, second]) {
        assert!(!frame.input_copy_used);
        assert!(frame.resident.coefficient_prep_used);
        assert!(frame.resident.packetization_used);
        assert!(frame.resident.codestream_assembly_used);
        let codestream = frame
            .encoded
            .codestream_bytes()
            .expect("Metal codestream bytes are CPU-readable");
        let decoded = Image::new(&codestream, &DecodeSettings::default())
            .expect("codestream parses")
            .decode_native()
            .expect("codestream decodes");
        assert_decoded_bytes_match(&decoded.data, &expected);
    }
}

#[cfg(target_os = "macos")]
#[test]
#[expect(
    clippy::cast_sign_loss,
    reason = "bounded synthetic pixel expression is nonnegative"
)]
fn metal_edge_private_batch_encode_to_metal_buffers_stays_resident() {
    if !should_run_metal_runtime() {
        return;
    }

    let first: Vec<u8> = (0..7 * 5 * 3).map(|i| ((i * 17) & 0xFF) as u8).collect();
    let second: Vec<u8> = (0..6 * 8 * 3)
        .map(|i| 255u8.wrapping_sub(((i * 19) & 0xFF) as u8))
        .collect();
    let session = crate::MetalBackendSession::system_default().expect("Metal session");
    let first_buffer = crate::benchmark_private_buffer_with_bytes(&session, &first)
        .expect("private benchmark input buffer");
    let second_buffer = crate::benchmark_private_buffer_with_bytes(&session, &second)
        .expect("private benchmark input buffer");
    compute::reset_ht_batch_coefficient_copy_blits_for_test();
    let tiles = [
        super::super::MetalLosslessEncodeTile {
            buffer: &first_buffer,
            byte_offset: 0,
            width: 7,
            height: 5,
            pitch_bytes: 7 * 3,
            output_width: 8,
            output_height: 8,
            format: PixelFormat::Rgb8,
        },
        super::super::MetalLosslessEncodeTile {
            buffer: &second_buffer,
            byte_offset: 0,
            width: 6,
            height: 8,
            pitch_bytes: 6 * 3,
            output_width: 8,
            output_height: 8,
            format: PixelFormat::Rgb8,
        },
    ];

    let encoded = super::super::encode_lossless_from_metal_buffers_to_metal_with_report(
        &tiles,
        &lossless_options! {
            backend: EncodeBackendPreference::RequireDevice,
        },
        &session,
    )
    .expect("Metal edge buffer batch lossless encode to Metal buffers");

    assert_eq!(encoded.len(), 2);
    for frame in &encoded {
        assert!(!frame.input_copy_used);
        assert!(frame.resident.coefficient_prep_used);
        assert!(frame.resident.packetization_used);
        assert!(frame.resident.codestream_assembly_used);
    }

    for (frame, (expected, width, height)) in encoded
        .iter()
        .zip([(first, 7usize, 5usize), (second, 6usize, 8usize)])
    {
        let codestream = frame
            .encoded
            .codestream_bytes()
            .expect("Metal codestream bytes are CPU-readable");
        let decoded = Image::new(&codestream, &DecodeSettings::default())
            .expect("codestream parses")
            .decode_native()
            .expect("codestream decodes");
        for y in 0..8usize {
            for x in 0..8usize {
                let dst = (y * 8 + x) * 3;
                if x < width && y < height {
                    let src = (y * width + x) * 3;
                    assert_eq!(&decoded.data[dst..dst + 3], &expected[src..src + 3]);
                } else {
                    assert_eq!(&decoded.data[dst..dst + 3], &[0, 0, 0]);
                }
            }
        }
    }
}

#[cfg(target_os = "macos")]
#[test]
#[expect(
    clippy::cast_sign_loss,
    reason = "bounded synthetic pixel expression is nonnegative"
)]
fn metal_ht_private_batch_encode_to_metal_buffers_stays_resident() {
    if !should_run_metal_runtime() {
        return;
    }

    let first: Vec<u8> = (0..8 * 8).map(|i| ((i * 11) & 0xFF) as u8).collect();
    let second: Vec<u8> = (0..8 * 8)
        .map(|i| 255u8.wrapping_sub(((i * 13) & 0xFF) as u8))
        .collect();
    let session = crate::MetalBackendSession::system_default().expect("Metal session");
    let first_buffer = crate::benchmark_private_buffer_with_bytes(&session, &first)
        .expect("private benchmark input buffer");
    let second_buffer = crate::benchmark_private_buffer_with_bytes(&session, &second)
        .expect("private benchmark input buffer");
    let tiles = [
        super::super::MetalLosslessEncodeTile {
            buffer: &first_buffer,
            byte_offset: 0,
            width: 8,
            height: 8,
            pitch_bytes: 8,
            output_width: 8,
            output_height: 8,
            format: PixelFormat::Gray8,
        },
        super::super::MetalLosslessEncodeTile {
            buffer: &second_buffer,
            byte_offset: 0,
            width: 8,
            height: 8,
            pitch_bytes: 8,
            output_width: 8,
            output_height: 8,
            format: PixelFormat::Gray8,
        },
    ];

    compute::reset_resident_gpu_timestamp_queries_for_test();
    let encoded = super::super::encode_lossless_from_padded_metal_buffers_to_metal_with_report(
        &tiles,
        &lossless_options! {
            backend: EncodeBackendPreference::RequireDevice,
            block_coding_mode: J2kBlockCodingMode::HighThroughput,
        },
        &session,
    )
    .expect("Metal HTJ2K batch lossless encode to Metal buffers");

    assert_eq!(encoded.len(), 2);
    assert_eq!(
        compute::ht_batch_coefficient_copy_blits_for_test(),
        0,
        "HTJ2K resident batch prep should write directly into the batch coefficient buffer"
    );
    assert_eq!(
        compute::resident_gpu_timestamp_queries_for_test(),
        7,
        "HTJ2K resident batch should query each unique retained command buffer timestamp once"
    );
    assert_eq!(
        encoded[0].encoded.codestream_buffer.as_ptr(),
        encoded[1].encoded.codestream_buffer.as_ptr(),
        "HTJ2K resident batch encode should assemble codestreams into one shared batch buffer"
    );
    assert_eq!(encoded[0].encoded.byte_offset, 0);
    assert!(
        encoded[1].encoded.byte_offset > 0,
        "second HTJ2K batch codestream should be a nonzero slice into the shared batch buffer"
    );
    for (frame, expected) in encoded.iter().zip([first, second]) {
        assert!(!frame.input_copy_used);
        assert!(frame.resident.coefficient_prep_used);
        assert!(frame.resident.packetization_used);
        assert!(frame.resident.codestream_assembly_used);
        let codestream = frame
            .encoded
            .codestream_bytes()
            .expect("Metal codestream bytes are CPU-readable");
        assert!(codestream.windows(2).any(|window| window == [0xFF, 0x50]));
        let decoded = Image::new(&codestream, &DecodeSettings::default())
            .expect("codestream parses")
            .decode_native()
            .expect("codestream decodes");
        assert_eq!(decoded.data, expected);
    }
}

#[cfg(target_os = "macos")]
#[test]
#[expect(
    clippy::cast_possible_truncation,
    reason = "bounded batch fixture dimensions and indices fit u8/u32 fields"
)]
fn metal_ht_private_batch_encode_reuses_private_arenas_between_batches() {
    const WIDTH: usize = 37;
    const HEIGHT: usize = 41;

    if !should_run_metal_runtime() {
        return;
    }

    let first: Vec<u8> = (0..WIDTH * HEIGHT)
        .map(|i| ((i * 7 + 3) & 0xFF) as u8)
        .collect();
    let second: Vec<u8> = (0..WIDTH * HEIGHT)
        .map(|i| 255u8.wrapping_sub(((i * 5 + 11) & 0xFF) as u8))
        .collect();
    let device = metal::Device::system_default().expect("Metal device");
    let session = crate::MetalBackendSession::new(device.clone());
    let first_buffer = crate::benchmark_private_buffer_with_bytes(&session, &first)
        .expect("private benchmark input buffer");
    let second_buffer = crate::benchmark_private_buffer_with_bytes(&session, &second)
        .expect("private benchmark input buffer");
    let tiles = [
        super::super::MetalLosslessEncodeTile {
            buffer: &first_buffer,
            byte_offset: 0,
            width: WIDTH as u32,
            height: HEIGHT as u32,
            pitch_bytes: WIDTH,
            output_width: WIDTH as u32,
            output_height: HEIGHT as u32,
            format: PixelFormat::Gray8,
        },
        super::super::MetalLosslessEncodeTile {
            buffer: &second_buffer,
            byte_offset: 0,
            width: WIDTH as u32,
            height: HEIGHT as u32,
            pitch_bytes: WIDTH,
            output_width: WIDTH as u32,
            output_height: HEIGHT as u32,
            format: PixelFormat::Gray8,
        },
    ];
    let options = lossless_options! {
        backend: EncodeBackendPreference::RequireDevice,
        block_coding_mode: J2kBlockCodingMode::HighThroughput,
        validation: J2kEncodeValidation::External,
    };

    compute::with_isolated_runtime_for_device_for_test(&device, || {
        compute::reset_private_buffer_pool_misses_for_test();
        super::super::encode_lossless_from_padded_metal_buffers_to_metal_with_report(
            &tiles, &options, &session,
        )?;
        let first_misses = compute::private_buffer_pool_misses_for_test();
        assert!(
            first_misses > 0,
            "first unique HTJ2K batch should populate reusable private arenas"
        );

        compute::reset_private_buffer_pool_misses_for_test();
        let encoded = super::super::encode_lossless_from_padded_metal_buffers_to_metal_with_report(
            &tiles, &options, &session,
        )?;

        assert_eq!(
            compute::private_buffer_pool_misses_for_test(),
            0,
            "second same-shape HTJ2K batch should reuse private arenas"
        );
        assert_eq!(encoded.len(), 2);
        Ok(())
    })
    .expect("isolated HTJ2K Metal runtime");
}
