// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

const SENTINEL: u8 = 0xA5;

/// A shared buffer holding one image at `byte_offset` with `row_padding`
/// extra bytes per row, every byte preset to [`SENTINEL`].
struct PaddedDestination {
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    layout: MetalImageLayout,
    row_bytes: usize,
}

impl PaddedDestination {
    fn new(
        device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
        dimensions: (u32, u32),
        fmt: PixelFormat,
        byte_offset: usize,
        row_padding: usize,
    ) -> Self {
        let row_bytes = usize::try_from(dimensions.0).unwrap() * fmt.bytes_per_pixel();
        let pitch = row_bytes + row_padding;
        let len = byte_offset + pitch * usize::try_from(dimensions.1).unwrap() + 8;
        let buffer = j2k_metal_support::checked_shared_buffer_for_len::<u8>(device, len)
            .expect("destination buffer");
        // SAFETY: the buffer is fresh and not yet visible to any GPU work.
        unsafe { j2k_metal_support::checked_buffer_write(&buffer, 0, &vec![SENTINEL; len]) }
            .expect("preset destination bytes");
        let layout =
            MetalImageLayout::new(byte_offset, dimensions, pitch, fmt).expect("destination layout");
        Self {
            buffer,
            layout,
            row_bytes,
        }
    }

    fn destination(&self) -> MetalImageDestination {
        // SAFETY: each test decodes synchronously and reads the buffer only
        // after `decode_into_metal` has returned, so no access overlaps the write.
        unsafe { MetalImageDestination::from_exclusive_buffer(self.buffer.clone(), self.layout) }
            .expect("destination")
    }

    fn bytes(&self) -> Vec<u8> {
        // SAFETY: called only after the decode that wrote the buffer finished.
        unsafe {
            j2k_metal_support::checked_buffer_read_vec::<u8>(&self.buffer, 0, self.buffer.length())
        }
        .expect("read destination")
    }

    /// The image rows packed tightly, and every byte outside them.
    fn split(&self) -> (Vec<u8>, Vec<u8>) {
        let bytes = self.bytes();
        let pitch = self.layout.pitch_bytes();
        let start = self.layout.byte_offset();
        let rows = usize::try_from(self.layout.dimensions().1).unwrap();
        let mut image = Vec::new();
        let mut outside = bytes[..start].to_vec();
        for row in 0..rows {
            let row_start = start + row * pitch;
            image.extend_from_slice(&bytes[row_start..row_start + self.row_bytes]);
            outside.extend_from_slice(&bytes[row_start + self.row_bytes..row_start + pitch]);
        }
        outside.extend_from_slice(&bytes[start + rows * pitch..]);
        (image, outside)
    }

    fn poison_image(&self) {
        let pitch = self.layout.pitch_bytes();
        for row in 0..usize::try_from(self.layout.dimensions().1).unwrap() {
            let offset = self.layout.byte_offset() + row * pitch;
            // SAFETY: no decode is in flight between test steps.
            unsafe {
                j2k_metal_support::checked_buffer_write(
                    &self.buffer,
                    offset,
                    &vec![!SENTINEL; self.row_bytes],
                )
            }
            .expect("poison image rows");
        }
    }
}

fn surface_bytes(codestream: &[u8], fmt: PixelFormat) -> (Vec<u8>, (u32, u32)) {
    let mut decoder = J2kDecoder::new(codestream).expect("parse fixture");
    let surface = decoder
        .decode_request_to_device(MetalDecodeRequest::full(fmt, BackendRequest::Metal))
        .expect("surface decode");
    let dimensions = surface.dimensions();
    (
        surface.as_bytes().expect("surface bytes").into_owned(),
        dimensions,
    )
}

/// Decode into a tight destination and a padded, offset one, twice each
/// through one decoder, and require the surface decode's bytes every time
/// with nothing written outside the image.
fn assert_decode_into_matches_surface(codestream: &[u8], fmt: PixelFormat, label: &str) {
    let (expected, dimensions) = surface_bytes(codestream, fmt);
    let session = MetalBackendSession::system_default().expect("Metal session");
    for (byte_offset, row_padding) in [(0, 0), (16, 12)] {
        let target =
            PaddedDestination::new(session.device(), dimensions, fmt, byte_offset, row_padding);
        let mut decoder = J2kDecoder::new(codestream).expect("parse fixture");
        for pass in 0..2 {
            target.poison_image();
            decoder
                .decode_into_metal(&target.destination())
                .unwrap_or_else(|error| panic!("{label} decode_into_metal pass {pass}: {error}"));
            let (image, outside) = target.split();
            assert_eq!(
                image, expected,
                "{label} offset={byte_offset} padding={row_padding} pass={pass}"
            );
            assert!(
                outside.iter().all(|&byte| byte == SENTINEL),
                "{label} wrote outside the destination image"
            );
        }
    }

    let target = PaddedDestination::new(session.device(), dimensions, fmt, 4, 8);
    J2kDecoder::new(codestream)
        .expect("parse fixture")
        .decode_into_metal_with_session(&target.destination(), &session)
        .unwrap_or_else(|error| panic!("{label} decode_into_metal_with_session: {error}"));
    assert_eq!(target.split().0, expected, "{label} session decode");
}

#[test]
fn decode_into_matches_surface_decode_for_gray_outputs() {
    if !should_run_metal_runtime() {
        return;
    }
    let gray8 = fixture_ht_gray8_offset_sized(37, 29, 11);
    assert_decode_into_matches_surface(&gray8, PixelFormat::Gray8, "HT gray8 -> Gray8");
    assert_decode_into_matches_surface(&gray8, PixelFormat::Gray16, "HT gray8 -> Gray16");

    let gray12 = fixture_ht_gray12_offset(301);
    assert_decode_into_matches_surface(&gray12, PixelFormat::Gray16, "HT gray12 -> Gray16");

    let (signed, _) = fixture_ht_signed_gray12(-40);
    assert_decode_into_matches_surface(&signed, PixelFormat::Gray16, "HT signed gray12 -> Gray16");
    assert_decode_into_matches_surface(&signed, PixelFormat::Gray8, "HT signed gray12 -> Gray8");

    let mut pixels = Vec::new();
    for index in 0..(61_u32 * 43) {
        let sample =
            i16::try_from(i32::try_from((index * 977) % 60_000).unwrap() - 30_000).unwrap();
        pixels.extend_from_slice(&sample.to_le_bytes());
    }
    let signed16 = encode_htj2k(&pixels, 61, 43, 1, 16, true, &EncodeOptions::default())
        .expect("encode signed 16-bit gray");
    assert_decode_into_matches_surface(
        &signed16,
        PixelFormat::Gray16,
        "HT signed gray16 -> Gray16",
    );
}

#[test]
fn decode_into_matches_surface_decode_for_color_outputs() {
    if !should_run_metal_runtime() {
        return;
    }
    let mut pixels = Vec::new();
    for index in 0..(33_u32 * 21) {
        pixels.extend(
            index.to_le_bytes()[..3]
                .iter()
                .map(|byte| byte.wrapping_mul(37)),
        );
    }
    let rgb8 = encode_htj2k(&pixels, 33, 21, 3, 8, false, &EncodeOptions::default())
        .expect("encode 8-bit RGB");
    assert_decode_into_matches_surface(&rgb8, PixelFormat::Rgb8, "HT RGB 8-bit -> Rgb8");

    let rgb16 = fixture_ht_rgb_u16_sized(19, 23, 300);
    assert_decode_into_matches_surface(&rgb16, PixelFormat::Rgb16, "HT RGB U16 -> Rgb16");
}

#[test]
fn decode_into_rejects_mismatched_destinations_without_writing() {
    if !should_run_metal_runtime() {
        return;
    }
    let session = MetalBackendSession::system_default().expect("Metal session");
    let gray8 = fixture_ht_gray8_offset_sized(37, 29, 11);
    let rgb8 = fixture_ht_rgb_u8_sized(33, 21, 7);
    for (codestream, dimensions, fmt, label) in [
        (&gray8, (36, 29), PixelFormat::Gray8, "narrow"),
        (&gray8, (37, 30), PixelFormat::Gray8, "tall"),
        (&gray8, (37, 29), PixelFormat::Rgb8, "gray image into RGB"),
        (&rgb8, (33, 21), PixelFormat::Gray16, "RGB image into gray"),
        (
            &rgb8,
            (33, 21),
            PixelFormat::Rgb8,
            "6-bit RGB, rescaled by surface decodes",
        ),
        (&rgb8, (33, 21), PixelFormat::Rgba8, "RGB image into RGBA"),
        (&gray8, (37, 29), PixelFormat::Rgba16, "unsupported format"),
    ] {
        let target = PaddedDestination::new(session.device(), dimensions, fmt, 8, 4);
        let result = J2kDecoder::new(codestream)
            .expect("parse fixture")
            .decode_into_metal(&target.destination());
        assert!(result.is_err(), "{label} must be rejected");
        assert!(
            target.bytes().iter().all(|&byte| byte == SENTINEL),
            "{label} wrote into the rejected destination"
        );
    }
}
