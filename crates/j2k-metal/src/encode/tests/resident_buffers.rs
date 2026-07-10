// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

#[cfg(target_os = "macos")]
#[test]
#[expect(
    clippy::cast_sign_loss,
    reason = "bounded synthetic pixel expression is nonnegative"
)]
fn metal_buffer_lossless_encode_pads_edge_tile_on_device() {
    if !should_run_metal_runtime() {
        return;
    }

    let pixels: Vec<u8> = (0..7 * 5 * 3).map(|i| ((i * 19) & 0xFF) as u8).collect();
    let device = metal::Device::system_default().expect("Metal device");
    let session = crate::MetalBackendSession::new(device);
    let buffer = session.device().new_buffer_with_data(
        pixels.as_ptr().cast(),
        pixels.len() as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let encoded = super::super::encode_lossless_from_metal_buffer(
        super::super::MetalLosslessEncodeTile {
            buffer: &buffer,
            byte_offset: 0,
            width: 7,
            height: 5,
            pitch_bytes: 7 * 3,
            output_width: 8,
            output_height: 8,
            format: PixelFormat::Rgb8,
        },
        &lossless_options! {
            backend: EncodeBackendPreference::RequireDevice,
        },
        &session,
    )
    .expect("Metal buffer lossless encode");

    assert_eq!(encoded.backend, BackendKind::Metal);
    let decoded = Image::new(&encoded.codestream, &DecodeSettings::default())
        .expect("codestream parses")
        .decode_native()
        .expect("codestream decodes");
    assert_eq!(decoded.width, 8);
    assert_eq!(decoded.height, 8);
    for y in 0..8usize {
        for x in 0..8usize {
            let dst = (y * 8 + x) * 3;
            if x < 7 && y < 5 {
                let src = (y * 7 + x) * 3;
                assert_eq!(&decoded.data[dst..dst + 3], &pixels[src..src + 3]);
            } else {
                assert_eq!(&decoded.data[dst..dst + 3], &[0, 0, 0]);
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
fn submitted_metal_buffer_lossless_encode_wait_round_trips() {
    if !should_run_metal_runtime() {
        return;
    }

    let pixels: Vec<u8> = (0..7 * 5 * 3).map(|i| ((i * 19) & 0xFF) as u8).collect();
    let session = crate::MetalBackendSession::system_default().expect("Metal session");
    let buffer = session.device().new_buffer_with_data(
        pixels.as_ptr().cast(),
        pixels.len() as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let submitted = super::super::submit_lossless_from_metal_buffer(
        super::super::MetalLosslessEncodeTile {
            buffer: &buffer,
            byte_offset: 0,
            width: 7,
            height: 5,
            pitch_bytes: 7 * 3,
            output_width: 8,
            output_height: 8,
            format: PixelFormat::Rgb8,
        },
        &lossless_options! {
            backend: EncodeBackendPreference::RequireDevice,
        },
        &session,
    )
    .expect("submit Metal buffer lossless encode");
    let encoded = submitted.wait().expect("wait Metal buffer lossless encode");

    assert_eq!(encoded.backend, BackendKind::Metal);
    let decoded = Image::new(&encoded.codestream, &DecodeSettings::default())
        .expect("codestream parses")
        .decode_native()
        .expect("codestream decodes");
    assert_eq!(decoded.width, 8);
    assert_eq!(decoded.height, 8);
    for y in 0..8usize {
        for x in 0..8usize {
            let dst = (y * 8 + x) * 3;
            if x < 7 && y < 5 {
                let src = (y * 7 + x) * 3;
                assert_eq!(&decoded.data[dst..dst + 3], &pixels[src..src + 3]);
            } else {
                assert_eq!(&decoded.data[dst..dst + 3], &[0, 0, 0]);
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
fn metal_buffer_lossless_encode_accepts_padded_contiguous_input_without_copy() {
    if !should_run_metal_runtime() {
        return;
    }

    let pixels: Vec<u8> = (0..8 * 8 * 3).map(|i| ((i * 31) & 0xFF) as u8).collect();
    let session = crate::MetalBackendSession::system_default().expect("Metal session");
    let buffer = session.device().new_buffer_with_data(
        pixels.as_ptr().cast(),
        pixels.len() as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let encoded = super::super::encode_lossless_from_padded_metal_buffer_with_report(
        super::super::MetalLosslessEncodeTile {
            buffer: &buffer,
            byte_offset: 0,
            width: 8,
            height: 8,
            pitch_bytes: 8 * 3,
            output_width: 8,
            output_height: 8,
            format: PixelFormat::Rgb8,
        },
        &lossless_options! {
            backend: EncodeBackendPreference::RequireDevice,
        },
        &session,
    )
    .expect("Metal padded buffer lossless encode");

    assert_eq!(encoded.encoded.backend, BackendKind::Metal);
    assert!(!encoded.input_copy_used);
    assert_eq!(encoded.input_copy_duration, std::time::Duration::ZERO);
    let decoded = Image::new(&encoded.encoded.codestream, &DecodeSettings::default())
        .expect("codestream parses")
        .decode_native()
        .expect("codestream decodes");
    assert_eq!(decoded.width, 8);
    assert_eq!(decoded.height, 8);
    assert_eq!(decoded.data, pixels);
}

#[cfg(target_os = "macos")]
#[test]
#[expect(
    clippy::cast_sign_loss,
    reason = "bounded synthetic pixel expression is nonnegative"
)]
fn metal_padded_private_rgb8_encode_uses_resident_coefficient_prep() {
    if !should_run_metal_runtime() {
        return;
    }

    let pixels: Vec<u8> = (0..8 * 8 * 3).map(|i| ((i * 31) & 0xFF) as u8).collect();
    let session = crate::MetalBackendSession::system_default().expect("Metal session");
    let buffer = crate::benchmark_private_buffer_with_bytes(&session, &pixels)
        .expect("private benchmark input buffer");

    let encoded = super::super::encode_lossless_from_padded_metal_buffer_with_report(
        super::super::MetalLosslessEncodeTile {
            buffer: &buffer,
            byte_offset: 0,
            width: 8,
            height: 8,
            pitch_bytes: 8 * 3,
            output_width: 8,
            output_height: 8,
            format: PixelFormat::Rgb8,
        },
        &lossless_options! {
            backend: EncodeBackendPreference::RequireDevice,
        },
        &session,
    )
    .expect("Metal private padded buffer lossless encode");

    assert_eq!(encoded.encoded.backend, BackendKind::Metal);
    assert!(!encoded.input_copy_used);
    assert!(encoded.resident.coefficient_prep_used);
    assert!(encoded.resident.packetization_used);
    assert!(encoded.resident.codestream_assembly_used);
    let decoded = Image::new(&encoded.encoded.codestream, &DecodeSettings::default())
        .expect("codestream parses")
        .decode_native()
        .expect("codestream decodes");
    assert_decoded_bytes_match(&decoded.data, &pixels);
}

#[cfg(target_os = "macos")]
#[test]
#[expect(
    clippy::cast_sign_loss,
    reason = "bounded synthetic pixel expression is nonnegative"
)]
fn metal_padded_private_rgb8_encode_to_metal_buffer_exposes_finished_bytes() {
    if !should_run_metal_runtime() {
        return;
    }

    let pixels: Vec<u8> = (0..8 * 8 * 3).map(|i| ((i * 37) & 0xFF) as u8).collect();
    let session = crate::MetalBackendSession::system_default().expect("Metal session");
    let buffer = crate::benchmark_private_buffer_with_bytes(&session, &pixels)
        .expect("private benchmark input buffer");

    let encoded = super::super::encode_lossless_from_padded_metal_buffer_to_metal_with_report(
        super::super::MetalLosslessEncodeTile {
            buffer: &buffer,
            byte_offset: 0,
            width: 8,
            height: 8,
            pitch_bytes: 8 * 3,
            output_width: 8,
            output_height: 8,
            format: PixelFormat::Rgb8,
        },
        &lossless_options! {
            backend: EncodeBackendPreference::RequireDevice,
        },
        &session,
    )
    .expect("Metal private padded buffer lossless encode to Metal buffer");

    assert!(!encoded.input_copy_used);
    assert!(encoded.resident.coefficient_prep_used);
    assert!(encoded.resident.packetization_used);
    assert!(encoded.resident.codestream_assembly_used);
    if let Some(duration) = encoded.gpu_duration {
        assert!(duration > Duration::ZERO);
    }
    assert_eq!(encoded.encoded.byte_offset, 0);
    assert!(encoded.encoded.byte_len > 0);
    assert!(encoded.encoded.capacity >= encoded.encoded.byte_len);
    let codestream = encoded
        .encoded
        .codestream_bytes()
        .expect("Metal codestream bytes are CPU-readable");
    assert!(codestream.starts_with(&[0xFF, 0x4F]));
    let decoded = Image::new(&codestream, &DecodeSettings::default())
        .expect("codestream parses")
        .decode_native()
        .expect("codestream decodes");
    assert_eq!(decoded.data, pixels);
}

#[cfg(target_os = "macos")]
#[test]
#[expect(
    clippy::cast_sign_loss,
    reason = "bounded synthetic pixel expression is nonnegative"
)]
fn metal_edge_private_rgb8_encode_to_metal_buffer_pads_and_stays_resident() {
    if !should_run_metal_runtime() {
        return;
    }

    let pixels: Vec<u8> = (0..7 * 5 * 3).map(|i| ((i * 41) & 0xFF) as u8).collect();
    let session = crate::MetalBackendSession::system_default().expect("Metal session");
    let buffer = crate::benchmark_private_buffer_with_bytes(&session, &pixels)
        .expect("private benchmark input buffer");

    let encoded = super::super::encode_lossless_from_metal_buffer_to_metal_with_report(
        super::super::MetalLosslessEncodeTile {
            buffer: &buffer,
            byte_offset: 0,
            width: 7,
            height: 5,
            pitch_bytes: 7 * 3,
            output_width: 8,
            output_height: 8,
            format: PixelFormat::Rgb8,
        },
        &lossless_options! {
            backend: EncodeBackendPreference::RequireDevice,
        },
        &session,
    )
    .expect("Metal private edge buffer lossless encode to Metal buffer");

    assert!(!encoded.input_copy_used);
    assert!(encoded.resident.coefficient_prep_used);
    assert!(encoded.resident.packetization_used);
    assert!(encoded.resident.codestream_assembly_used);
    let codestream = encoded
        .encoded
        .codestream_bytes()
        .expect("Metal codestream bytes are CPU-readable");
    assert!(codestream.starts_with(&[0xFF, 0x4F]));
    let decoded = Image::new(&codestream, &DecodeSettings::default())
        .expect("codestream parses")
        .decode_native()
        .expect("codestream decodes");
    assert_eq!(decoded.width, 8);
    assert_eq!(decoded.height, 8);
    for y in 0..8usize {
        for x in 0..8usize {
            let dst = (y * 8 + x) * 3;
            if x < 7 && y < 5 {
                let src = (y * 7 + x) * 3;
                assert_eq!(&decoded.data[dst..dst + 3], &pixels[src..src + 3]);
            } else {
                assert_eq!(&decoded.data[dst..dst + 3], &[0, 0, 0]);
            }
        }
    }
}

#[cfg(target_os = "macos")]
#[test]
fn submitted_private_padded_rgb8_encode_snapshots_before_wait() {
    if !should_run_metal_runtime() {
        return;
    }

    let pixels: Vec<u8> = (0..8 * 8 * 3)
        .map(|i| u8::try_from((i * 31) & 0xFF).expect("masked pixel fits u8"))
        .collect();
    let replacement = vec![0u8; pixels.len()];
    let session = crate::MetalBackendSession::system_default().expect("Metal session");
    let buffer = crate::benchmark_private_buffer_with_bytes(&session, &pixels)
        .expect("private benchmark input buffer");

    let submitted = super::super::submit_lossless_from_padded_metal_buffer(
        super::super::MetalLosslessEncodeTile {
            buffer: &buffer,
            byte_offset: 0,
            width: 8,
            height: 8,
            pitch_bytes: 8 * 3,
            output_width: 8,
            output_height: 8,
            format: PixelFormat::Rgb8,
        },
        &lossless_options! {
            backend: EncodeBackendPreference::RequireDevice,
        },
        &session,
    )
    .expect("submit Metal private padded RGB8 encode");
    crate::benchmark_overwrite_private_buffer_with_bytes(&session, &buffer, &replacement)
        .expect("overwrite private benchmark input buffer");

    let encoded = submitted.wait().expect("wait submitted encode");
    let decoded = Image::new(&encoded.codestream, &DecodeSettings::default())
        .expect("codestream parses")
        .decode_native()
        .expect("codestream decodes");
    assert_eq!(decoded.data, pixels);
}

#[cfg(target_os = "macos")]
#[test]
fn metal_padded_private_gray8_dwt_encode_uses_resident_coefficient_prep() {
    if !should_run_metal_runtime() {
        return;
    }

    let mut pixels = Vec::with_capacity(128 * 128);
    for y in 0..128u32 {
        for x in 0..128u32 {
            pixels.push(((x * 7 + y * 11 + (x ^ y)) & 0xFF) as u8);
        }
    }
    let session = crate::MetalBackendSession::system_default().expect("Metal session");
    let buffer = crate::benchmark_private_buffer_with_bytes(&session, &pixels)
        .expect("private benchmark input buffer");

    let encoded = super::super::encode_lossless_from_padded_metal_buffer_with_report(
        super::super::MetalLosslessEncodeTile {
            buffer: &buffer,
            byte_offset: 0,
            width: 128,
            height: 128,
            pitch_bytes: 128,
            output_width: 128,
            output_height: 128,
            format: PixelFormat::Gray8,
        },
        &lossless_options! {
            backend: EncodeBackendPreference::RequireDevice,
        },
        &session,
    )
    .expect("Metal private padded DWT buffer lossless encode");

    assert_eq!(encoded.encoded.backend, BackendKind::Metal);
    assert!(!encoded.input_copy_used);
    assert!(encoded.resident.coefficient_prep_used);
    assert!(encoded.resident.packetization_used);
    assert!(encoded.resident.codestream_assembly_used);
    let decoded = Image::new(&encoded.encoded.codestream, &DecodeSettings::default())
        .expect("codestream parses")
        .decode_native()
        .expect("codestream decodes");
    assert_decoded_bytes_match(&decoded.data, &pixels);
}

#[cfg(target_os = "macos")]
#[test]
fn metal_padded_private_rgb8_dwt_encode_uses_resident_coefficient_prep() {
    if !should_run_metal_runtime() {
        return;
    }

    let mut pixels = Vec::with_capacity(128 * 128 * 3);
    for y in 0..128u32 {
        for x in 0..128u32 {
            pixels.push(((x * 3 + y * 5) & 0xFF) as u8);
            pixels.push(((x * 7 + y * 11) & 0xFF) as u8);
            pixels.push(((x * 13 + y * 17) & 0xFF) as u8);
        }
    }
    let session = crate::MetalBackendSession::system_default().expect("Metal session");
    let buffer = crate::benchmark_private_buffer_with_bytes(&session, &pixels)
        .expect("private benchmark input buffer");

    let encoded = super::super::encode_lossless_from_padded_metal_buffer_with_report(
        super::super::MetalLosslessEncodeTile {
            buffer: &buffer,
            byte_offset: 0,
            width: 128,
            height: 128,
            pitch_bytes: 128 * 3,
            output_width: 128,
            output_height: 128,
            format: PixelFormat::Rgb8,
        },
        &lossless_options! {
            backend: EncodeBackendPreference::RequireDevice,
        },
        &session,
    )
    .expect("Metal private padded RGB8 DWT buffer lossless encode");

    assert_eq!(encoded.encoded.backend, BackendKind::Metal);
    assert!(encoded.resident.coefficient_prep_used);
    assert!(encoded.resident.packetization_used);
    assert!(encoded.resident.codestream_assembly_used);
    let decoded = Image::new(&encoded.encoded.codestream, &DecodeSettings::default())
        .expect("codestream parses")
        .decode_native()
        .expect("codestream decodes");
    assert_decoded_bytes_match(&decoded.data, &pixels);
}

#[cfg(target_os = "macos")]
#[test]
fn metal_padded_private_gray8_dwt_resident_codestream_decodes_natively() {
    if !should_run_metal_runtime() {
        return;
    }

    let mut pixels = Vec::with_capacity(128 * 128);
    for y in 0..128u32 {
        for x in 0..128u32 {
            pixels.push(((x * 7 + y * 11 + (x ^ y)) & 0xFF) as u8);
        }
    }
    let session = crate::MetalBackendSession::system_default().expect("Metal session");
    let buffer = crate::benchmark_private_buffer_with_bytes(&session, &pixels)
        .expect("private benchmark input buffer");

    let encoded = super::super::encode_lossless_from_padded_metal_buffer_with_report(
        super::super::MetalLosslessEncodeTile {
            buffer: &buffer,
            byte_offset: 0,
            width: 128,
            height: 128,
            pitch_bytes: 128,
            output_width: 128,
            output_height: 128,
            format: PixelFormat::Gray8,
        },
        &lossless_options! {
            backend: EncodeBackendPreference::RequireDevice,
            validation: J2kEncodeValidation::External,
        },
        &session,
    )
    .expect("Metal private padded DWT buffer lossless encode");

    let decoded = Image::new(&encoded.encoded.codestream, &DecodeSettings::default())
        .expect("codestream parses")
        .decode_native()
        .expect("codestream decodes");
    assert_decoded_bytes_match(&decoded.data, &pixels);
}

#[cfg(target_os = "macos")]
#[test]
fn metal_padded_private_rgb8_dwt_resident_codestream_decodes_natively() {
    if !should_run_metal_runtime() {
        return;
    }

    let mut pixels = Vec::with_capacity(128 * 128 * 3);
    for y in 0..128u32 {
        for x in 0..128u32 {
            pixels.push(((x * 3 + y * 5) & 0xFF) as u8);
            pixels.push(((x * 7 + y * 11) & 0xFF) as u8);
            pixels.push(((x * 13 + y * 17) & 0xFF) as u8);
        }
    }
    let session = crate::MetalBackendSession::system_default().expect("Metal session");
    let buffer = crate::benchmark_private_buffer_with_bytes(&session, &pixels)
        .expect("private benchmark input buffer");

    let encoded = super::super::encode_lossless_from_padded_metal_buffer_with_report(
        super::super::MetalLosslessEncodeTile {
            buffer: &buffer,
            byte_offset: 0,
            width: 128,
            height: 128,
            pitch_bytes: 128 * 3,
            output_width: 128,
            output_height: 128,
            format: PixelFormat::Rgb8,
        },
        &lossless_options! {
            backend: EncodeBackendPreference::RequireDevice,
            validation: J2kEncodeValidation::External,
        },
        &session,
    )
    .expect("Metal private padded RGB8 DWT buffer lossless encode");

    let decoded = Image::new(&encoded.encoded.codestream, &DecodeSettings::default())
        .expect("codestream parses")
        .decode_native()
        .expect("codestream decodes");
    assert_decoded_bytes_match(&decoded.data, &pixels);
}

#[cfg(target_os = "macos")]
#[test]
fn metal_padded_private_gray8_rpcl_encode_uses_resident_coefficient_prep() {
    if !should_run_metal_runtime() {
        return;
    }

    let mut pixels = Vec::with_capacity(128 * 128);
    for y in 0..128u32 {
        for x in 0..128u32 {
            pixels.push(((x * 5 + y * 9 + (x ^ y)) & 0xFF) as u8);
        }
    }
    let session = crate::MetalBackendSession::system_default().expect("Metal session");
    let buffer = crate::benchmark_private_buffer_with_bytes(&session, &pixels)
        .expect("private benchmark input buffer");

    let encoded = super::super::encode_lossless_from_padded_metal_buffer_with_report(
        super::super::MetalLosslessEncodeTile {
            buffer: &buffer,
            byte_offset: 0,
            width: 128,
            height: 128,
            pitch_bytes: 128,
            output_width: 128,
            output_height: 128,
            format: PixelFormat::Gray8,
        },
        &lossless_options! {
            backend: EncodeBackendPreference::RequireDevice,
            progression: J2kProgressionOrder::Rpcl,
        },
        &session,
    )
    .expect("Metal private padded RPCL buffer lossless encode");

    assert_eq!(encoded.encoded.backend, BackendKind::Metal);
    assert!(encoded.resident.coefficient_prep_used);
    assert!(encoded.resident.packetization_used);
    assert!(encoded.resident.codestream_assembly_used);
    let decoded = Image::new(&encoded.encoded.codestream, &DecodeSettings::default())
        .expect("codestream parses")
        .decode_native()
        .expect("codestream decodes");
    assert_eq!(decoded.data, pixels);
}

#[cfg(target_os = "macos")]
#[test]
fn metal_padded_private_gray16_encode_uses_resident_coefficient_prep() {
    if !should_run_metal_runtime() {
        return;
    }

    let mut pixels = Vec::with_capacity(8 * 8 * 2);
    for idx in 0..64u16 {
        let value = idx.wrapping_mul(997).wrapping_add(123);
        pixels.extend_from_slice(&value.to_le_bytes());
    }
    let session = crate::MetalBackendSession::system_default().expect("Metal session");
    let buffer = crate::benchmark_private_buffer_with_bytes(&session, &pixels)
        .expect("private benchmark input buffer");

    let encoded = super::super::encode_lossless_from_padded_metal_buffer_with_report(
        super::super::MetalLosslessEncodeTile {
            buffer: &buffer,
            byte_offset: 0,
            width: 8,
            height: 8,
            pitch_bytes: 8 * 2,
            output_width: 8,
            output_height: 8,
            format: PixelFormat::Gray16,
        },
        &lossless_options! {
            backend: EncodeBackendPreference::RequireDevice,
        },
        &session,
    )
    .expect("Metal private padded Gray16 buffer lossless encode");

    assert_eq!(encoded.encoded.backend, BackendKind::Metal);
    assert!(!encoded.input_copy_used);
    assert!(encoded.resident.coefficient_prep_used);
    assert!(encoded.resident.packetization_used);
    assert!(encoded.resident.codestream_assembly_used);
    let decoded = Image::new(&encoded.encoded.codestream, &DecodeSettings::default())
        .expect("codestream parses")
        .decode_native()
        .expect("codestream decodes");
    assert_eq!(decoded.data, pixels);
}

#[cfg(target_os = "macos")]
#[test]
#[expect(
    clippy::cast_sign_loss,
    reason = "bounded synthetic pixel expression is nonnegative"
)]
fn metal_padded_private_ht_encode_to_metal_buffer_stays_resident() {
    if !should_run_metal_runtime() {
        return;
    }

    let pixels: Vec<u8> = (0..8 * 8).map(|i| ((i * 31) & 0xFF) as u8).collect();
    let session = crate::MetalBackendSession::system_default().expect("Metal session");
    let buffer = crate::benchmark_private_buffer_with_bytes(&session, &pixels)
        .expect("private benchmark input buffer");

    let encoded = super::super::encode_lossless_from_padded_metal_buffer_to_metal_with_report(
        super::super::MetalLosslessEncodeTile {
            buffer: &buffer,
            byte_offset: 0,
            width: 8,
            height: 8,
            pitch_bytes: 8,
            output_width: 8,
            output_height: 8,
            format: PixelFormat::Gray8,
        },
        &lossless_options! {
            backend: EncodeBackendPreference::RequireDevice,
            block_coding_mode: J2kBlockCodingMode::HighThroughput,
        },
        &session,
    )
    .expect("Metal private padded HTJ2K buffer lossless encode");

    assert!(!encoded.input_copy_used);
    assert!(encoded.resident.coefficient_prep_used);
    assert!(encoded.resident.packetization_used);
    assert!(encoded.resident.codestream_assembly_used);
    let codestream = encoded
        .encoded
        .codestream_bytes()
        .expect("Metal codestream bytes are CPU-readable");
    assert!(codestream.windows(2).any(|window| window == [0xFF, 0x50]));
    let cod_marker = codestream
        .windows(2)
        .position(|window| window == [0xFF, 0x52])
        .expect("COD marker");
    assert_eq!(codestream[cod_marker + 12], 0x40);
    let decoded = Image::new(&codestream, &DecodeSettings::default())
        .expect("codestream parses")
        .decode_native()
        .expect("codestream decodes");
    assert_eq!(decoded.data, pixels);
}

#[cfg(target_os = "macos")]
#[test]
#[expect(
    clippy::cast_sign_loss,
    reason = "bounded masked fixture expression is nonnegative"
)]
fn metal_padded_private_rgb8_ht_rpcl_512_encode_preserves_three_dwt_levels_and_stays_resident() {
    if !should_run_metal_runtime() {
        return;
    }

    let pixels: Vec<u8> = (0..512 * 512 * 3)
        .map(|idx| ((idx * 47 + idx / 17) & 0xFF) as u8)
        .collect();
    let session = crate::MetalBackendSession::system_default().expect("Metal session");
    let buffer = crate::benchmark_private_buffer_with_bytes(&session, &pixels)
        .expect("private benchmark input buffer");

    let encoded = super::super::encode_lossless_from_padded_metal_buffer_to_metal_with_report(
        super::super::MetalLosslessEncodeTile {
            buffer: &buffer,
            byte_offset: 0,
            width: 512,
            height: 512,
            pitch_bytes: 512 * 3,
            output_width: 512,
            output_height: 512,
            format: PixelFormat::Rgb8,
        },
        &lossless_options! {
            backend: EncodeBackendPreference::RequireDevice,
            block_coding_mode: J2kBlockCodingMode::HighThroughput,
            progression: J2kProgressionOrder::Rpcl,
        },
        &session,
    )
    .expect("Metal private padded HTJ2K RPCL 512 buffer lossless encode");

    assert!(!encoded.input_copy_used);
    assert!(encoded.resident.coefficient_prep_used);
    assert!(encoded.resident.packetization_used);
    assert!(encoded.resident.codestream_assembly_used);
    let codestream = encoded
        .encoded
        .codestream_bytes()
        .expect("Metal codestream bytes are CPU-readable");
    let cod_marker = codestream
        .windows(2)
        .position(|window| window == [0xFF, 0x52])
        .expect("COD marker");
    assert_eq!(codestream[cod_marker + 5], 0x02);
    assert_eq!(codestream[cod_marker + 9], 3);
    assert_eq!(codestream[cod_marker + 12], 0x40);
    let decoded = Image::new(&codestream, &DecodeSettings::default())
        .expect("codestream parses")
        .decode_native()
        .expect("codestream decodes");
    assert_eq!(decoded.data, pixels);
}
