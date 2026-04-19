// SPDX-License-Identifier: Apache-2.0

use dicom_toolkit_jpeg2000::{encode, DecodeSettings, EncodeOptions, Image};
use slidecodec_core::{BufferError, CodecError, Downscale, PixelFormat, Rect};
use slidecodec_j2k::{J2kDecoder, J2kError};

fn encode_codestream(
    pixels: &[u8],
    width: u32,
    height: u32,
    components: u8,
    bit_depth: u8,
    reversible: bool,
) -> Vec<u8> {
    let options = EncodeOptions {
        reversible,
        num_decomposition_levels: 1,
        ..EncodeOptions::default()
    };
    encode(pixels, width, height, components, bit_depth, false, &options).expect("encode")
}

fn wrap_codestream_jp2(
    codestream: &[u8],
    width: u32,
    height: u32,
    components: u16,
    bit_depth: u8,
    colorspace_enum: u32,
) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&[0, 0, 0, 12, b'j', b'P', b' ', b' ', 0x0D, 0x0A, 0x87, 0x0A]);
    bytes.extend_from_slice(&[
        0, 0, 0, 20, b'f', b't', b'y', b'p', b'j', b'p', b'2', b' ', 0, 0, 0, 0, b'j', b'p', b'2',
        b' ',
    ]);

    let bpc = bit_depth.saturating_sub(1);
    bytes.extend_from_slice(&[
        0,
        0,
        0,
        45,
        b'j',
        b'p',
        b'2',
        b'h',
        0,
        0,
        0,
        22,
        b'i',
        b'h',
        b'd',
        b'r',
    ]);
    bytes.extend_from_slice(&height.to_be_bytes());
    bytes.extend_from_slice(&width.to_be_bytes());
    bytes.extend_from_slice(&components.to_be_bytes());
    bytes.extend_from_slice(&[bpc, 7, 0, 0]);
    bytes.extend_from_slice(&[0, 0, 0, 15, b'c', b'o', b'l', b'r', 1, 0, 0]);
    bytes.extend_from_slice(&colorspace_enum.to_be_bytes());

    let len = (8 + codestream.len()) as u32;
    bytes.extend_from_slice(&len.to_be_bytes());
    bytes.extend_from_slice(b"jp2c");
    bytes.extend_from_slice(codestream);
    bytes
}

fn backend_decode_u8(bytes: &[u8]) -> Vec<u8> {
    Image::new(bytes, &DecodeSettings::default())
        .expect("backend image")
        .decode()
        .expect("backend decode")
}

#[test]
fn decode_rgb8_codestream_roundtrips_reversible_pixels() {
    let pixels = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120];
    let codestream = encode_codestream(&pixels, 2, 2, 3, 8, true);
    let expected = backend_decode_u8(&codestream);
    let mut decoder = J2kDecoder::new(&codestream).expect("decoder");
    let mut out = [0_u8; 12];
    let outcome = decoder
        .decode_into(&mut out, 2 * 3, PixelFormat::Rgb8)
        .expect("decode");
    assert_eq!(outcome.decoded, slidecodec_core::Rect::full((2, 2)));
    assert_eq!(out, expected.as_slice());
}

#[test]
fn decode_rgba8_fills_opaque_alpha_for_rgb_source() {
    let pixels = [1, 2, 3, 4, 5, 6];
    let codestream = encode_codestream(&pixels, 2, 1, 3, 8, true);
    let mut decoder = J2kDecoder::new(&codestream).expect("decoder");
    let mut out = [0_u8; 8];
    decoder
        .decode_into(&mut out, 2 * 4, PixelFormat::Rgba8)
        .expect("decode");
    assert_eq!(out, [1, 2, 3, 255, 4, 5, 6, 255]);
}

#[test]
fn decode_gray8_jp2_roundtrips_reversible_pixels() {
    let pixels = [3, 9, 27, 81];
    let codestream = encode_codestream(&pixels, 2, 2, 1, 8, true);
    let jp2 = wrap_codestream_jp2(&codestream, 2, 2, 1, 8, 17);
    let mut decoder = J2kDecoder::new(&jp2).expect("decoder");
    let mut out = [0_u8; 4];
    decoder
        .decode_into(&mut out, 2, PixelFormat::Gray8)
        .expect("decode");
    assert_eq!(out, pixels);
}

#[test]
fn decode_gray16_roundtrips_native_samples() {
    let samples = [0_u16, 1024, 2048, 4095];
    let pixels: Vec<u8> = samples.into_iter().flat_map(u16::to_le_bytes).collect();
    let codestream = encode_codestream(&pixels, 2, 2, 1, 12, true);
    let mut decoder = J2kDecoder::new(&codestream).expect("decoder");
    let mut out = [0_u8; 8];
    decoder
        .decode_into(&mut out, 2 * 2, PixelFormat::Gray16)
        .expect("decode");
    assert_eq!(out, pixels.as_slice());
}

#[test]
fn decode_gray16_widens_8bit_samples_to_full_u16_range() {
    let pixels = [0_u8, 64, 128, 255];
    let codestream = encode_codestream(&pixels, 2, 2, 1, 8, true);
    let mut decoder = J2kDecoder::new(&codestream).expect("decoder");
    let mut out = [0_u8; 8];
    decoder
        .decode_into(&mut out, 2 * 2, PixelFormat::Gray16)
        .expect("decode");
    let expected: Vec<u8> = [0_u16, 16448, 32896, 65535]
        .into_iter()
        .flat_map(u16::to_le_bytes)
        .collect();
    assert_eq!(out, expected.as_slice());
}

#[test]
fn decode_rgb16_roundtrips_native_samples() {
    let samples = [0_u16, 1, 2, 1024, 2048, 3072];
    let pixels: Vec<u8> = samples.into_iter().flat_map(u16::to_le_bytes).collect();
    let codestream = encode_codestream(&pixels, 2, 1, 3, 12, true);
    let mut decoder = J2kDecoder::new(&codestream).expect("decoder");
    let mut out = [0_u8; 12];
    decoder
        .decode_into(&mut out, 2 * 3 * 2, PixelFormat::Rgb16)
        .expect("decode");
    assert_eq!(out, pixels.as_slice());
}

#[test]
fn decode_region_into_is_not_implemented_yet() {
    let pixels = [10, 20, 30, 40];
    let codestream = encode_codestream(&pixels, 2, 2, 1, 8, true);
    let mut decoder = J2kDecoder::new(&codestream).expect("decoder");
    let mut pool = slidecodec_j2k::J2kScratchPool::new();
    let mut out = [0_u8; 4];
    let err = decoder
        .decode_region_into(&mut pool, &mut out, 2, PixelFormat::Gray8, Rect { x: 0, y: 0, w: 1, h: 1 })
        .unwrap_err();
    assert!(err.is_not_implemented());
}

#[test]
fn decode_scaled_into_is_not_implemented_yet() {
    let pixels = [10, 20, 30, 40];
    let codestream = encode_codestream(&pixels, 2, 2, 1, 8, true);
    let mut decoder = J2kDecoder::new(&codestream).expect("decoder");
    let mut pool = slidecodec_j2k::J2kScratchPool::new();
    let mut out = [0_u8; 4];
    let err = decoder
        .decode_scaled_into(&mut pool, &mut out, 2, PixelFormat::Gray8, Downscale::Half)
        .unwrap_err();
    assert!(err.is_not_implemented());
}

#[test]
fn decode_rejects_unsupported_rgba16_output() {
    let pixels = [1, 2, 3, 4, 5, 6];
    let codestream = encode_codestream(&pixels, 2, 1, 3, 8, true);
    let mut decoder = J2kDecoder::new(&codestream).expect("decoder");
    let mut out = [0_u8; 16];
    let err = decoder
        .decode_into(&mut out, 2 * 4 * 2, PixelFormat::Rgba16)
        .unwrap_err();
    assert!(matches!(err, J2kError::Unsupported(_)));
}

#[test]
fn decode_rejects_small_output_buffer() {
    let pixels = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120];
    let codestream = encode_codestream(&pixels, 2, 2, 3, 8, true);
    let mut decoder = J2kDecoder::new(&codestream).expect("decoder");
    let mut out = [0_u8; 11];
    let err = decoder
        .decode_into(&mut out, 6, PixelFormat::Rgb8)
        .unwrap_err();
    assert!(matches!(
        err,
        J2kError::Buffer(BufferError::OutputTooSmall { .. })
    ));
}

#[test]
fn decode_rejects_too_small_stride() {
    let pixels = [10, 20, 30, 40, 50, 60];
    let codestream = encode_codestream(&pixels, 2, 1, 3, 8, true);
    let mut decoder = J2kDecoder::new(&codestream).expect("decoder");
    let mut out = [0_u8; 6];
    let err = decoder
        .decode_into(&mut out, 5, PixelFormat::Rgb8)
        .unwrap_err();
    assert!(matches!(
        err,
        J2kError::Buffer(BufferError::StrideTooSmall { .. })
    ));
}
