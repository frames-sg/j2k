// SPDX-License-Identifier: MIT OR Apache-2.0

//! JP2 files whose codestream has more components than the declared color
//! space and no channel definition box. ISO/IEC 15444-1 I.5.3.6 leaves the
//! extra components unspecified, so the declared color space stands and the
//! extras are left out of decoded output. Kakadu wrote such files from RGBA
//! TIFFs: an `sRGB` `colr` box over four components and no `cdef` box.

use j2k::{
    recode_j2k_to_htj2k_lossless, DecodeSettings, J2kDecoder, J2kRowDecodeOptions, J2kScratchPool,
    J2kToHtj2kMode, J2kToHtj2kOptions, PixelFormat, Rect, RowSink,
};
use j2k_native::{encode, ColorSpace, EncodeOptions, Image};

const WIDTH: u32 = 24;
const HEIGHT: u32 = 20;
const SRGB: u32 = 16;
const GREYSCALE: u32 = 17;

/// Interleaved 8-bit samples for `components` components.
fn pixels(components: u16) -> Vec<u8> {
    (0..WIDTH * HEIGHT * u32::from(components))
        .map(|i| u8::try_from((i * 37 + i / 7) % 256).expect("byte"))
        .collect()
}

fn codestream(components: u16) -> Vec<u8> {
    encode(
        &pixels(components),
        WIDTH,
        HEIGHT,
        components,
        8,
        false,
        &EncodeOptions {
            reversible: true,
            num_decomposition_levels: 2,
            ..EncodeOptions::default()
        },
    )
    .expect("encode fixture")
}

fn jp2_box(kind: [u8; 4], payload: &[u8]) -> Vec<u8> {
    let len = u32::try_from(payload.len() + 8).expect("box length");
    let mut out = len.to_be_bytes().to_vec();
    out.extend_from_slice(&kind);
    out.extend_from_slice(payload);
    out
}

/// A JP2 with an enumerated `colr` box and no `cdef` box, whatever the
/// codestream's component count.
fn jp2_without_cdef(components: u16, enumerated: u32) -> Vec<u8> {
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&HEIGHT.to_be_bytes());
    ihdr.extend_from_slice(&WIDTH.to_be_bytes());
    ihdr.extend_from_slice(&components.to_be_bytes());
    ihdr.extend_from_slice(&[7, 7, 0, 0]);
    let mut colr = vec![1, 0, 0];
    colr.extend_from_slice(&enumerated.to_be_bytes());
    let mut header = jp2_box(*b"ihdr", &ihdr);
    header.extend(jp2_box(*b"colr", &colr));

    let mut out = jp2_box(*b"jP  ", &[0x0D, 0x0A, 0x87, 0x0A]);
    out.extend(jp2_box(*b"ftyp", b"jp2 \0\0\0\0jp2 "));
    out.extend(jp2_box(*b"jp2h", &header));
    out.extend(jp2_box(*b"jp2c", &codestream(components)));
    out
}

/// The first `kept` of every `components` interleaved samples.
fn leading_channels(samples: &[u8], components: usize, kept: usize) -> Vec<u8> {
    samples
        .chunks_exact(components)
        .flat_map(|pixel| pixel[..kept].iter().copied())
        .collect()
}

fn with_opaque_alpha(rgb: &[u8]) -> Vec<u8> {
    rgb.as_chunks::<3>()
        .0
        .iter()
        .flat_map(|pixel| [pixel[0], pixel[1], pixel[2], u8::MAX])
        .collect()
}

fn region_into(
    decoder: &mut J2kDecoder<'_>,
    fmt: PixelFormat,
    roi: Rect,
) -> Result<Vec<u8>, j2k::J2kError> {
    let stride = roi.w as usize * fmt.bytes_per_pixel();
    let mut out = vec![0_u8; stride * roi.h as usize];
    decoder.decode_region_into(&mut J2kScratchPool::new(), &mut out, stride, fmt, roi)?;
    Ok(out)
}

fn whole_into(decoder: &mut J2kDecoder<'_>, fmt: PixelFormat) -> Vec<u8> {
    let stride = WIDTH as usize * fmt.bytes_per_pixel();
    let mut out = vec![0_u8; stride * HEIGHT as usize];
    decoder
        .decode_into(&mut out, stride, fmt)
        .expect("whole-image decode");
    out
}

const FULL: Rect = Rect {
    x: 0,
    y: 0,
    w: WIDTH,
    h: HEIGHT,
};

#[test]
fn srgb_with_an_extra_component_decodes_as_rgb() {
    let bytes = jp2_without_cdef(4, SRGB);
    let rgb = leading_channels(&pixels(4), 4, 3);
    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");

    assert_eq!(whole_into(&mut decoder, PixelFormat::Rgb8), rgb);
    assert_eq!(
        region_into(&mut decoder, PixelFormat::Rgb8, FULL).expect("Rgb8 region"),
        rgb
    );
    // The extra component is not alpha: RGBA output is opaque.
    assert_eq!(
        whole_into(&mut decoder, PixelFormat::Rgba8),
        with_opaque_alpha(&rgb)
    );
    assert_eq!(
        region_into(&mut decoder, PixelFormat::Rgba8, FULL).expect("Rgba8 region"),
        with_opaque_alpha(&rgb)
    );

    let roi = Rect {
        x: 5,
        y: 3,
        w: 11,
        h: 9,
    };
    let expected: Vec<u8> = (roi.y..roi.y + roi.h)
        .flat_map(|y| {
            let start = (y * WIDTH + roi.x) as usize * 3;
            rgb[start..start + roi.w as usize * 3].to_vec()
        })
        .collect();
    assert_eq!(
        region_into(&mut decoder, PixelFormat::Rgb8, roi).expect("Rgb8 sub-region"),
        expected
    );
}

#[test]
fn srgb_with_an_extra_component_decodes_scaled_and_sixteen_bit_regions() {
    let bytes = jp2_without_cdef(4, SRGB);
    let rgb = leading_channels(&pixels(4), 4, 3);
    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");

    for levels in 1..=2_u8 {
        let (w, h) = (WIDTH.div_ceil(1 << levels), HEIGHT.div_ceil(1 << levels));
        let stride = w as usize * 3;
        let mut out = vec![0_u8; stride * h as usize];
        let outcome = decoder
            .decode_region_scaled_pow2_into(
                &mut J2kScratchPool::new(),
                &mut out,
                stride,
                PixelFormat::Rgb8,
                FULL,
                levels,
            )
            .expect("scaled Rgb8 region");
        assert_eq!((outcome.decoded.w, outcome.decoded.h), (w, h));
    }

    let stride = WIDTH as usize * PixelFormat::Rgb16.bytes_per_pixel();
    let mut out = vec![0_u8; stride * HEIGHT as usize];
    decoder
        .decode_into(&mut out, stride, PixelFormat::Rgb16)
        .expect("Rgb16 decode");
    let samples: Vec<u16> = out
        .as_chunks::<2>()
        .0
        .iter()
        .map(|sample| u16::from_ne_bytes(*sample))
        .collect();
    let widened: Vec<u16> = rgb.iter().map(|&sample| u16::from(sample) * 257).collect();
    assert_eq!(samples, widened);
}

#[test]
fn greyscale_with_an_extra_component_decodes_as_gray() {
    let bytes = jp2_without_cdef(2, GREYSCALE);
    let gray = leading_channels(&pixels(2), 2, 1);
    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");

    assert_eq!(whole_into(&mut decoder, PixelFormat::Gray8), gray);
    assert_eq!(
        region_into(&mut decoder, PixelFormat::Gray8, FULL).expect("Gray8 region"),
        gray
    );
}

#[test]
fn native_output_omits_the_extra_components() {
    for (components, enumerated, kept) in [(4, SRGB, 3), (5, SRGB, 3), (2, GREYSCALE, 1)] {
        let bytes = jp2_without_cdef(components, enumerated);
        let image = Image::new(&bytes, &j2k_native::DecodeSettings::default()).expect("image");
        assert_eq!(
            image.ignored_components(),
            usize::from(components) - kept,
            "{components} components"
        );
        assert!(!image.has_alpha());
        let expected_space = if kept == 1 {
            ColorSpace::Gray
        } else {
            ColorSpace::RGB
        };
        assert_eq!(
            core::mem::discriminant(image.color_space()),
            core::mem::discriminant(&expected_space)
        );
        assert_eq!(
            image
                .decode_native_components()
                .expect("planes")
                .planes()
                .len(),
            kept
        );
        assert_eq!(
            usize::from(image.decode_native().expect("raw bitmap").num_components),
            kept
        );
        // Streamed native planes cover up to four components.
        let streamed = image.decode_native_region_components_rows_with_context(
            (0, 0, WIDTH, HEIGHT),
            &mut j2k_native::DecoderContext::default(),
        );
        if components <= 4 {
            assert_eq!(streamed.expect("streamed planes").planes().len(), kept);
        }
    }
}

#[test]
fn lenient_decoding_still_infers_alpha_for_one_extra_component() {
    let bytes = jp2_without_cdef(4, SRGB);
    let mut decoder =
        J2kDecoder::new_with_settings(&bytes, DecodeSettings::lenient()).expect("decoder");
    assert_eq!(whole_into(&mut decoder, PixelFormat::Rgba8), pixels(4));
}

#[test]
fn raw_codestreams_keep_their_component_count_interpretation() {
    let bytes = codestream(4);
    let image = Image::new(&bytes, &j2k_native::DecodeSettings::default()).expect("image");
    assert_eq!(image.ignored_components(), 0);
    assert!(matches!(image.color_space(), ColorSpace::CMYK));
}

struct Rows<T> {
    samples: Vec<T>,
    row_len: usize,
    next_y: u32,
}

impl<T> Rows<T> {
    fn new(channels: usize) -> Self {
        Self {
            samples: Vec::new(),
            row_len: WIDTH as usize * channels,
            next_y: 0,
        }
    }
}

impl<T: j2k_core::Sample> RowSink<T> for Rows<T> {
    type Error = core::convert::Infallible;

    fn write_row(&mut self, y: u32, row: &[T]) -> Result<(), Self::Error> {
        assert_eq!(y, self.next_y);
        assert_eq!(row.len(), self.row_len);
        self.samples.extend_from_slice(row);
        self.next_y += 1;
        Ok(())
    }
}

#[test]
fn bounded_u8_rows_omit_unspecified_components() {
    for (components, enumerated, kept) in [(2, GREYSCALE, 1), (5, SRGB, 3), (4, SRGB, 3)] {
        let bytes = jp2_without_cdef(components, enumerated);
        let expected = leading_channels(&pixels(components), usize::from(components), kept);
        for stripe_rows in [1, 7, HEIGHT] {
            let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
            let mut rows = Rows::<u8>::new(kept);
            decoder
                .decode_rows_u8_bounded(&mut rows, J2kRowDecodeOptions::new(stripe_rows))
                .expect("bounded u8 rows");
            assert_eq!(rows.next_y, HEIGHT);
            assert_eq!(rows.samples, expected, "{components} components");
            assert_eq!(decoder.info().components, components);
        }
    }
}

#[test]
fn bounded_u16_rows_omit_unspecified_components() {
    for (components, enumerated, kept) in [(2, GREYSCALE, 1), (5, SRGB, 3), (4, SRGB, 3)] {
        let bytes = jp2_without_cdef(components, enumerated);
        let expected: Vec<u16> =
            leading_channels(&pixels(components), usize::from(components), kept)
                .into_iter()
                .map(|sample| u16::from(sample) * 257)
                .collect();
        for stripe_rows in [1, 7, HEIGHT] {
            let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
            let mut rows = Rows::<u16>::new(kept);
            decoder
                .decode_rows_u16_bounded(&mut rows, J2kRowDecodeOptions::new(stripe_rows))
                .expect("bounded u16 rows");
            assert_eq!(rows.next_y, HEIGHT);
            assert_eq!(rows.samples, expected, "{components} components");
            assert_eq!(decoder.info().components, components);
        }
    }
}

#[test]
fn recode_preserves_unspecified_component_samples() {
    let bytes = jp2_without_cdef(4, SRGB);
    let recoded = recode_j2k_to_htj2k_lossless(&bytes, J2kToHtj2kOptions::default())
        .expect("recode all four components");
    assert_eq!(recoded.report.mode, J2kToHtj2kMode::PixelPreserving);
    assert_eq!(recoded.report.components, 4);
    // Decode the raw output codestream so display-channel filtering cannot
    // hide the fourth component from this comparison.
    let image = Image::new(&recoded.bytes, &j2k_native::DecodeSettings::default())
        .expect("recoded codestream");
    let bitmap = image.decode_native().expect("all recoded samples");
    assert_eq!(bitmap.num_components, 4);
    assert_eq!(bitmap.data, pixels(4));
}
