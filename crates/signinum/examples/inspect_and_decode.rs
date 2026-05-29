// SPDX-License-Identifier: Apache-2.0

//! Generate small JPEG and JPEG 2000 fixtures, then inspect and decode them
//! through the facade crate.
//!
//! Run with:
//! `cargo run -p signinum --example inspect_and_decode`

use signinum::{
    j2k::{encode_j2k_lossless, J2kDecoder, J2kLosslessEncodeOptions, J2kLosslessSamples},
    jpeg::{
        encode_jpeg_baseline, Decoder as JpegDecoder, JpegBackend, JpegEncodeOptions, JpegSamples,
        JpegSubsampling,
    },
    PixelFormat,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let width = 8;
    let height = 8;
    let gray: Vec<u8> = (0..width * height)
        .map(|value| u8::try_from(value).expect("8x8 fixture values fit in u8"))
        .collect();

    let jpeg = encode_jpeg_baseline(
        JpegSamples::Gray8 {
            data: &gray,
            width,
            height,
        },
        JpegEncodeOptions::new(90, JpegSubsampling::Gray, None, JpegBackend::Auto),
    )?;
    let jpeg_info = JpegDecoder::inspect(&jpeg.data)?;
    let jpeg_stride = jpeg_info.dimensions.0 as usize * PixelFormat::Gray8.bytes_per_pixel();
    let mut jpeg_decoded = vec![0_u8; jpeg_stride * jpeg_info.dimensions.1 as usize];
    JpegDecoder::new(&jpeg.data)?.decode_into(
        &mut jpeg_decoded,
        jpeg_stride,
        PixelFormat::Gray8,
    )?;

    let j2k_samples = J2kLosslessSamples::new(&gray, width, height, 1, 8, false)?;
    let j2k = encode_j2k_lossless(j2k_samples, &J2kLosslessEncodeOptions::default())?;
    let j2k_info = J2kDecoder::inspect(&j2k.codestream)?;
    let j2k_stride = j2k_info.dimensions.0 as usize * PixelFormat::Gray8.bytes_per_pixel();
    let mut j2k_decoded = vec![0_u8; j2k_stride * j2k_info.dimensions.1 as usize];
    J2kDecoder::new(&j2k.codestream)?.decode_into(
        &mut j2k_decoded,
        j2k_stride,
        PixelFormat::Gray8,
    )?;

    assert_eq!(jpeg_info.dimensions, (width, height));
    assert_eq!(j2k_info.dimensions, (width, height));
    assert_eq!(j2k_decoded, gray);

    println!(
        "JPEG {}x{} -> {} bytes; J2K {}x{} -> {} bytes",
        jpeg_info.dimensions.0,
        jpeg_info.dimensions.1,
        jpeg.data.len(),
        j2k_info.dimensions.0,
        j2k_info.dimensions.1,
        j2k.codestream.len()
    );
    Ok(())
}
