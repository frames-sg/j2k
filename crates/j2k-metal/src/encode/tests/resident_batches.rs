// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(target_os = "macos")]
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

    let _profile = crate::profile_env::force_metal_profile_stage_mode_for_test(
        j2k_profile::ProfileStageMode::Disabled,
    );
    compute::reset_metal_command_buffers_for_test();
    compute::reset_lossless_deinterleave_rct_fused_dispatches_for_test();
    let encoded = super::super::encode_lossless_from_padded_metal_buffers_to_metal_with_report(
        &tiles, &options, &session,
    )
    .expect("Metal RGB8 HTJ2K batch encode");

    assert_eq!(encoded.len(), 2);
    assert_eq!(
        compute::metal_command_buffers_for_test(),
        7,
        "resident HT batch retains the measured separate-stage submission route",
    );
    assert!(
        compute::lossless_deinterleave_rct_fused_dispatches_for_test() > 0,
        "RGB8 resident lossless encode should fuse deinterleave and RCT"
    );
    let profiled = {
        let _profile = crate::profile_env::force_metal_profile_stages_for_test(true);
        super::super::encode_lossless_from_padded_metal_buffers_to_metal_with_report(
            &tiles, &options, &session,
        )
        .expect("profiled Metal HTJ2K batch encode")
    };
    for (plain, profiled) in encoded.iter().zip(&profiled) {
        assert_eq!(
            plain.encoded.codestream_bytes().expect("plain codestream"),
            profiled
                .encoded
                .codestream_bytes()
                .expect("profiled codestream"),
            "profiling must preserve the codestream",
        );
    }
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
    let first_buffer =
        j2k_metal_support::checked_shared_buffer_with_slice(session.device(), &first)
            .expect("upload first test tile");
    let second_buffer =
        j2k_metal_support::checked_shared_buffer_with_slice(session.device(), &second)
            .expect("upload second test tile");
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
        objc2::rc::Retained::as_ptr(&encoded[0].encoded.codestream_buffer),
        objc2::rc::Retained::as_ptr(&encoded[1].encoded.codestream_buffer),
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
fn metal_padded_private_batch_dwt_encode_to_metal_buffers_round_trips() {
    if !should_run_metal_runtime() {
        return;
    }

    let session = crate::MetalBackendSession::system_default().expect("Metal session");
    for (width, height) in [(128u32, 128u32), (65, 35)] {
        let first: Vec<u8> = (0..width * height * 3)
            .map(|i| ((i * 17 + i / 3) & 0xFF) as u8)
            .collect();
        let second: Vec<u8> = (0..width * height * 3)
            .map(|i| 255u8.wrapping_sub(((i * 23 + i / 5) & 0xFF) as u8))
            .collect();
        let first_buffer = crate::benchmark_private_buffer_with_bytes(&session, &first)
            .expect("private benchmark input buffer");
        let second_buffer = crate::benchmark_private_buffer_with_bytes(&session, &second)
            .expect("private benchmark input buffer");
        let tiles = [
            super::super::MetalLosslessEncodeTile {
                buffer: &first_buffer,
                byte_offset: 0,
                width,
                height,
                pitch_bytes: width as usize * 3,
                output_width: width,
                output_height: height,
                format: PixelFormat::Rgb8,
            },
            super::super::MetalLosslessEncodeTile {
                buffer: &second_buffer,
                byte_offset: 0,
                width,
                height,
                pitch_bytes: width as usize * 3,
                output_width: width,
                output_height: height,
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
    let _profile = crate::profile_env::force_metal_profile_stages_for_test(false);
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
        objc2::rc::Retained::as_ptr(&encoded[0].encoded.codestream_buffer),
        objc2::rc::Retained::as_ptr(&encoded[1].encoded.codestream_buffer),
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
    let device = j2k_metal_support::system_default_device().expect("Metal device");
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

#[cfg(target_os = "macos")]
#[test]
fn perf_shape_ht_private_batch_has_no_warm_pool_misses_or_evictions() {
    const WIDTH: usize = 512;
    const HEIGHT: usize = 512;
    const BATCH_SIZE: usize = 16;
    const EXPECTED_PRIVATE_BUFFER_TAKES: usize = 103;

    if !should_run_metal_runtime() {
        return;
    }

    let pixels: Vec<u8> = (0..WIDTH * HEIGHT * 3)
        .map(|index| u8::try_from((index * 29 + index / 7) & 0xff).expect("masked pixel"))
        .collect();
    let device = j2k_metal_support::system_default_device().expect("Metal device");
    let session = crate::MetalBackendSession::new(device.clone());
    let input = crate::benchmark_private_buffer_with_bytes(&session, &pixels)
        .expect("private benchmark input buffer");
    let width_u32 = u32::try_from(WIDTH).expect("test width fits u32");
    let height_u32 = u32::try_from(HEIGHT).expect("test height fits u32");
    let tiles: Vec<_> = (0..BATCH_SIZE)
        .map(|_| super::super::MetalLosslessEncodeTile {
            buffer: &input,
            byte_offset: 0,
            width: width_u32,
            height: height_u32,
            pitch_bytes: WIDTH * 3,
            output_width: width_u32,
            output_height: height_u32,
            format: PixelFormat::Rgb8,
        })
        .collect();
    let options = lossless_options! {
        backend: EncodeBackendPreference::RequireDevice,
        block_coding_mode: J2kBlockCodingMode::HighThroughput,
        validation: J2kEncodeValidation::External,
    };

    compute::with_isolated_runtime_for_device_for_test(&device, || {
        super::super::encode_lossless_from_padded_metal_buffers_to_metal_with_report(
            &tiles, &options, &session,
        )?;
        compute::reset_private_buffer_pool_misses_for_test();
        compute::reset_private_buffer_pool_take_probes_for_test();
        let encoded = super::super::encode_lossless_from_padded_metal_buffers_to_metal_with_report(
            &tiles, &options, &session,
        )?;
        let diagnostics = session.buffer_pool_diagnostics()?;

        assert_eq!(compute::private_buffer_pool_misses_for_test(), 0);
        assert!(
            compute::private_buffer_pool_take_probes_for_test()
                <= EXPECTED_PRIVATE_BUFFER_TAKES * 7,
            "warm resident buffer acquisition should remain near-linear"
        );
        assert_eq!(diagnostics.private.evictions, 0);
        assert_eq!(encoded.len(), BATCH_SIZE);
        Ok(())
    })
    .expect("isolated 16x512 RGB8 HTJ2K resident batch");
}

/// The CAP marker's `Bp` field (Part 15 Table A.4 magnitude bound), or `None`
/// when the codestream has no CAP marker.
#[cfg(target_os = "macos")]
fn cap_magnitude_bound_field(codestream: &[u8]) -> Option<u16> {
    let cap = codestream
        .windows(2)
        .position(|window| window == [0xFF, 0x50])?;
    // Marker, Lcap, Pcap (4 bytes), then the Part 15 capability word.
    let word = codestream.get(cap + 8..cap + 10)?;
    Some(u16::from_be_bytes([word[0], word[1]]) & 0x1F)
}

#[cfg(target_os = "macos")]
const CAP_FIXTURE_SIZE: u32 = 128;
#[cfg(target_os = "macos")]
const CAP_FIXTURE_LEVELS: u8 = 2;

/// Two noisy RGB8 frames and the CPU HTJ2K lossless codestream of each.
/// Noise keeps the RCT chroma and its 5/3 detail bands above eight magnitude
/// bits, so the measured CAP bound exceeds `bit_depth - 1`.
#[cfg(target_os = "macos")]
fn cap_fixture() -> (Vec<Vec<u8>>, Vec<Vec<u8>>) {
    const SIZE: u32 = CAP_FIXTURE_SIZE;
    let frames: Vec<Vec<u8>> = [0x2545_f491_u32, 0x9e37_79b9]
        .into_iter()
        .map(|mut state| {
            (0..SIZE * SIZE * 3)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 17;
                    state ^= state << 5;
                    state.to_le_bytes()[1]
                })
                .collect()
        })
        .collect();
    // The CPU encoder with the resident route's fixed choices: 32x32 code
    // blocks, the RCT, and two guard bits for three components.
    let cpu_options = EncodeOptions {
        num_decomposition_levels: CAP_FIXTURE_LEVELS,
        reversible: true,
        code_block_width_exp: 3,
        code_block_height_exp: 3,
        guard_bits: 2,
        use_mct: true,
        ..EncodeOptions::default()
    };
    let expected: Vec<Vec<u8>> = frames
        .iter()
        .map(|pixels| {
            j2k_native::encode_htj2k(pixels, SIZE, SIZE, 3, 8, false, &cpu_options)
                .expect("CPU HTJ2K lossless encode")
        })
        .collect();
    for codestream in &expected {
        assert!(
            cap_magnitude_bound_field(codestream).is_some_and(|bp| bp >= 1),
            "fixture must need more than eight magnitude bits"
        );
    }
    (frames, expected)
}

#[cfg(target_os = "macos")]
fn cap_fixture_options() -> J2kLosslessEncodeOptions {
    lossless_options! {
        backend: EncodeBackendPreference::RequireDevice,
        block_coding_mode: J2kBlockCodingMode::HighThroughput,
        max_decomposition_levels: Some(CAP_FIXTURE_LEVELS),
        validation: J2kEncodeValidation::External,
    }
}

#[cfg(target_os = "macos")]
fn cap_fixture_tile(
    buffer: &crate::metal_types::Buffer,
) -> super::super::MetalLosslessEncodeTile<'_> {
    super::super::MetalLosslessEncodeTile {
        buffer,
        byte_offset: 0,
        width: CAP_FIXTURE_SIZE,
        height: CAP_FIXTURE_SIZE,
        pitch_bytes: CAP_FIXTURE_SIZE as usize * 3,
        output_width: CAP_FIXTURE_SIZE,
        output_height: CAP_FIXTURE_SIZE,
        format: PixelFormat::Rgb8,
    }
}

#[cfg(target_os = "macos")]
fn assert_codestream_matches_cpu(actual: &[u8], expected: &[u8], label: &str) {
    assert_eq!(
        cap_magnitude_bound_field(actual),
        cap_magnitude_bound_field(expected),
        "{label}: CAP magnitude bound"
    );
    let first_difference = actual
        .iter()
        .zip(expected)
        .position(|(actual, expected)| actual != expected);
    assert!(
        actual == expected,
        "{label}: resident codestream differs from the CPU \
         (lengths {} / {}, first difference at byte {first_difference:?})",
        actual.len(),
        expected.len(),
    );
}

/// The batched resident route measures each tile's CAP bound in its batched
/// codestream assembly.
#[cfg(target_os = "macos")]
#[test]
fn metal_resident_rgb8_ht_lossless_codestreams_match_cpu_bytes_including_cap() {
    if !should_run_metal_runtime() {
        return;
    }

    let (frames, expected) = cap_fixture();
    let session = crate::MetalBackendSession::system_default().expect("Metal session");
    let buffers: Vec<_> = frames
        .iter()
        .map(|pixels| {
            crate::benchmark_private_buffer_with_bytes(&session, pixels)
                .expect("private benchmark input buffer")
        })
        .collect();
    let options = cap_fixture_options();
    for count in [1, 2] {
        let tiles: Vec<_> = buffers[..count].iter().map(cap_fixture_tile).collect();
        let encoded = super::super::encode_lossless_from_padded_metal_buffers_to_metal_with_report(
            &tiles, &options, &session,
        )
        .expect("resident Metal HTJ2K lossless encode");
        assert_eq!(encoded.len(), count);
        for (index, (frame, expected)) in encoded.iter().zip(&expected).enumerate() {
            assert!(frame.resident.codestream_assembly_used);
            let actual = frame
                .encoded
                .codestream_bytes()
                .expect("Metal codestream bytes are CPU-readable");
            assert_codestream_matches_cpu(
                &actual,
                expected,
                &format!("batch of {count}, tile {index}"),
            );
        }
    }
}

/// The single-tile resident route measures the CAP bound in its own
/// codestream assembly kernel.
#[cfg(target_os = "macos")]
#[test]
fn metal_single_tile_resident_rgb8_ht_lossless_codestream_matches_cpu_bytes_including_cap() {
    if !should_run_metal_runtime() {
        return;
    }

    let (frames, expected) = cap_fixture();
    let session = crate::MetalBackendSession::system_default().expect("Metal session");
    for (index, (pixels, expected)) in frames.iter().zip(&expected).enumerate() {
        let buffer = crate::benchmark_private_buffer_with_bytes(&session, pixels)
            .expect("private benchmark input buffer");
        let outcome = super::super::try_encode_lossless_tile_device_resident_with_report(
            cap_fixture_tile(&buffer),
            cap_fixture_options(),
            &session,
            super::super::MetalEncodeInputStaging::AlreadyPaddedContiguous,
        )
        .expect("single-tile resident Metal HTJ2K lossless encode")
        .expect("the tile qualifies for the single-tile resident route");
        assert!(outcome.resident.codestream_assembly_used);
        assert_codestream_matches_cpu(
            &outcome.encoded.codestream,
            expected,
            &format!("single tile {index}"),
        );
    }
}
