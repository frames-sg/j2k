// SPDX-License-Identifier: MIT OR Apache-2.0

//! Encode decoded samples as `opj_decompress` 2.5 writes each format.
//!
//! Sample conversions follow the tool's `imagetobmp`, `imagetopng`,
//! `imagetopnm` and `imagetotif`; the `image` crate writes the containers.
//! Where that crate cannot express the tool's container, the samples stay the
//! same and the container differs: gray PNG shorter than 8 bits is written as
//! 8-bit (scaled as the tool scales other short precisions), TIFF stores
//! 1-16-bit samples in 8- or 16-bit fields instead of packing them, and RGB
//! PNM deeper than 8 bits is written as PAM (`P7`) instead of `P6`.

use std::io::Cursor;

use image::{
    codecs::{
        bmp::BmpEncoder,
        png::PngEncoder,
        pnm::{
            ArbitraryHeader, ArbitraryTuplType, GraymapHeader, PixmapHeader, PnmEncoder, PnmHeader,
            SampleEncoding,
        },
        tiff::TiffEncoder,
    },
    ExtendedColorType, ImageEncoder,
};

use super::{args::OutputFormat, decode::DecodedImage};

/// The bytes of `image` in `format`.
pub(crate) fn encode(image: &DecodedImage, format: OutputFormat) -> Result<Vec<u8>, String> {
    match format {
        OutputFormat::Bmp => bmp(image),
        OutputFormat::Png => png(image),
        OutputFormat::Tiff => tiff(image),
        OutputFormat::Pnm { gray } => pnm(image, gray),
    }
}

/// `force-rgb`: replicate gray into red, green and blue, as
/// `opj_decompress` converts a gray image; colour images are unchanged.
pub(crate) fn force_rgb(image: DecodedImage) -> DecodedImage {
    if image.channels != 1 {
        return image;
    }
    let samples = image
        .samples
        .iter()
        .flat_map(|&sample| [sample; 3])
        .collect();
    DecodedImage {
        channels: 3,
        samples,
        ..image
    }
}

fn signed_offset(precision: u8, signed: bool) -> i32 {
    if signed {
        1 << (precision - 1)
    } else {
        0
    }
}

/// `clip_component`: clamp samples to `precision` bits, reading unsigned
/// samples as unsigned 32-bit values as `opj_decompress` does.
fn clip(samples: &[i32], precision: u8, signed: bool) -> Vec<i32> {
    let max = (1_i64 << precision) - 1;
    samples
        .iter()
        .map(|&sample| {
            if signed {
                let high = max / 2;
                i32::try_from(i64::from(sample).clamp(-high - 1, high)).unwrap_or(sample)
            } else if i64::from(sample) > max || sample < 0 {
                i32::try_from(max).unwrap_or(i32::MAX)
            } else {
                sample
            }
        })
        .collect()
}

/// `scale_component`: rescale to `to` bits, multiplying up with truncating
/// division or shifting down.
fn scale(samples: &[i32], from: u8, to: u8, signed: bool) -> Vec<i32> {
    samples
        .iter()
        .map(|&sample| {
            let value = i64::from(sample);
            let scaled = if from > to {
                value >> (from - to)
            } else if signed {
                value * (1_i64 << (to - 1)) / (1_i64 << (from - 1))
            } else {
                value * ((1_i64 << to) - 1) / ((1_i64 << from) - 1)
            };
            i32::try_from(scaled).unwrap_or(sample)
        })
        .collect()
}

fn to_u8(sample: i32) -> u8 {
    u8::try_from(sample.clamp(0, 255)).unwrap_or(u8::MAX)
}

fn to_u16(sample: i32) -> u16 {
    u16::try_from(sample.clamp(0, 65_535)).unwrap_or(u16::MAX)
}

fn color_type(channels: usize, sixteen: bool) -> Result<ExtendedColorType, String> {
    Ok(match (channels, sixteen) {
        (1, false) => ExtendedColorType::L8,
        (1, true) => ExtendedColorType::L16,
        (2, false) => ExtendedColorType::La8,
        (2, true) => ExtendedColorType::La16,
        (3, false) => ExtendedColorType::Rgb8,
        (3, true) => ExtendedColorType::Rgb16,
        (4, false) => ExtendedColorType::Rgba8,
        (4, true) => ExtendedColorType::Rgba16,
        _ => return Err(format!("{channels}-channel output is not supported")),
    })
}

fn native_bytes(samples: &[u16]) -> Vec<u8> {
    samples
        .iter()
        .flat_map(|sample| sample.to_ne_bytes())
        .collect()
}

/// `imagetobmp`: 24-bit BGR from the first three channels, or 8-bit gray
/// with a gray palette from the first; deeper samples are rounded to 8 bits.
fn bmp(image: &DecodedImage) -> Result<Vec<u8>, String> {
    let precision = image.precision;
    if precision < 8 {
        return Err(format!(
            "imagetobmp: Unsupported precision: {precision}; BMP output needs 8 or more bits"
        ));
    }
    let offset = signed_offset(precision, image.signed);
    let eight = |sample: i32| {
        let mut value = sample + offset;
        if precision > 8 {
            let adjust = i32::from(precision - 8);
            value = (value >> adjust) + ((value >> (adjust - 1)) % 2);
        }
        to_u8(value)
    };
    let kept = if image.channels >= 3 { 3 } else { 1 };
    let pixels = image
        .samples
        .chunks_exact(image.channels)
        .flat_map(|pixel| pixel[..kept].iter().map(|&sample| eight(sample)))
        .collect::<Vec<_>>();
    let mut output = Vec::new();
    BmpEncoder::new(&mut output)
        .encode(&pixels, image.width, image.height, color_type(kept, false)?)
        .map_err(|error| format!("imagetobmp: {error}"))?;
    Ok(output)
}

/// `imagetopng`: 8- and 16-bit samples as they are, 9-15 bits scaled to 16,
/// shorter colour samples scaled to 8; signed samples are offset to unsigned.
fn png(image: &DecodedImage) -> Result<Vec<u8>, String> {
    let channels = image.channels;
    let mut precision = image.precision;
    let mut samples = clip(&image.samples, precision, image.signed);
    let target = match precision {
        9..=15 => 16,
        1..=7 if channels > 1 => 8,
        5 | 6 => 8,
        3 | 7 => precision + 1,
        _ => precision,
    };
    // Gray PNG shorter than 8 bits is scaled to 8: the encoder writes only 8-
    // and 16-bit PNG.
    let target = if target < 8 { 8 } else { target };
    if target != precision {
        samples = scale(&samples, precision, target, image.signed);
        precision = target;
    }
    if precision != 8 && precision != 16 {
        return Err(format!(
            "imagetopng: can not create PNG with bit depth {precision}"
        ));
    }
    let offset = signed_offset(precision, image.signed);
    let bytes = if precision == 16 {
        native_bytes(
            &samples
                .iter()
                .map(|&sample| to_u16(sample + offset))
                .collect::<Vec<_>>(),
        )
    } else {
        samples
            .iter()
            .map(|&sample| to_u8(sample + offset))
            .collect()
    };
    let mut output = Vec::new();
    PngEncoder::new(&mut output)
        .write_image(
            &bytes,
            image.width,
            image.height,
            color_type(channels, precision == 16)?,
        )
        .map_err(|error| format!("imagetopng: {error}"))?;
    Ok(output)
}

/// `imagetotif`: samples of up to 16 bits, signed ones offset to unsigned,
/// minimum-is-black gray or RGB(A).
fn tiff(image: &DecodedImage) -> Result<Vec<u8>, String> {
    let precision = image.precision;
    if precision > 16 {
        return Err(format!(
            "imagetotif: Bits={precision}, Only 1 to 16 bits implemented"
        ));
    }
    let offset = signed_offset(precision, image.signed);
    let samples = clip(&image.samples, precision, image.signed);
    let sixteen = precision > 8;
    let bytes = if sixteen {
        native_bytes(
            &samples
                .iter()
                .map(|&sample| to_u16(sample + offset))
                .collect::<Vec<_>>(),
        )
    } else {
        samples
            .iter()
            .map(|&sample| to_u8(sample + offset))
            .collect()
    };
    let mut output = Vec::new();
    TiffEncoder::new(Cursor::new(&mut output))
        .write_image(
            &bytes,
            image.width,
            image.height,
            color_type(image.channels, sixteen)?,
        )
        .map_err(|error| format!("imagetotif: {error}"))?;
    Ok(output)
}

/// `imagetopnm`: `P5` gray (the first component for a `.pgm` name), `P6`
/// RGB and `P7` with alpha, with maximum value `2^precision - 1` and the
/// samples unscaled.
fn pnm(image: &DecodedImage, want_gray: bool) -> Result<Vec<u8>, String> {
    let precision = image.precision;
    if precision > 16 {
        return Err(format!(
            "imagetopnm: precision {precision} is larger than 16: refused"
        ));
    }
    let (width, height) = (image.width, image.height);
    let maximum = (1_u32 << precision) - 1;
    let offset = signed_offset(precision, image.signed);
    let sixteen = precision > 8;
    let channels = if want_gray { 1 } else { image.channels };
    let samples = image
        .samples
        .chunks_exact(image.channels)
        .flat_map(|pixel| pixel[..channels].iter().copied());
    // OpenJPEG offsets signed samples except in its 8-bit multi-component path.
    let offset = if channels > 1 && !sixteen { 0 } else { offset };
    let header: PnmHeader = match channels {
        1 => GraymapHeader {
            encoding: SampleEncoding::Binary,
            height,
            width,
            maxwhite: maximum,
        }
        .into(),
        3 if !sixteen => PixmapHeader {
            encoding: SampleEncoding::Binary,
            height,
            width,
            maxval: maximum,
        }
        .into(),
        _ => ArbitraryHeader {
            height,
            width,
            depth: u32::try_from(channels).unwrap_or(u32::MAX),
            maxval: maximum,
            tupltype: Some(match channels {
                2 => ArbitraryTuplType::GrayscaleAlpha,
                3 => ArbitraryTuplType::RGB,
                _ => ArbitraryTuplType::RGBAlpha,
            }),
        }
        .into(),
    };
    let mut output = Vec::new();
    let mut encoder = PnmEncoder::new(&mut output).with_header(header);
    let color = color_type(channels, sixteen)?;
    let result = if sixteen {
        let samples = samples
            .map(|sample| to_u16(sample + offset))
            .collect::<Vec<_>>();
        encoder.encode(samples.as_slice(), width, height, color)
    } else {
        let samples = samples
            .map(|sample| to_u8(sample + offset))
            .collect::<Vec<_>>();
        encoder.encode(samples.as_slice(), width, height, color)
    };
    result.map_err(|error| format!("imagetopnm: {error}"))?;
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::{clip, encode, force_rgb, scale};
    use crate::opj::{args::OutputFormat, decode::DecodedImage};
    use image::ImageDecoder;
    use std::io::Cursor;

    fn image(channels: usize, precision: u8, signed: bool, samples: Vec<i32>) -> DecodedImage {
        DecodedImage {
            width: u32::try_from(samples.len() / channels).expect("width"),
            height: 1,
            channels,
            precision,
            signed,
            samples,
        }
    }

    fn decoded_bytes(format: image::ImageFormat, bytes: &[u8]) -> (image::ColorType, Vec<u8>) {
        let decoded = image::load_from_memory_with_format(bytes, format).expect("read output");
        (decoded.color(), decoded.as_bytes().to_vec())
    }

    /// The palette indices of an 8-bit gray BMP, through its gray palette.
    fn bmp_gray(bytes: &[u8]) -> Vec<u8> {
        image::load_from_memory_with_format(bytes, image::ImageFormat::Bmp)
            .expect("read BMP")
            .to_luma8()
            .into_raw()
    }

    #[test]
    fn bmp_rounds_deep_samples_and_drops_alpha_as_openjpeg_does() {
        // 12-bit: (v >> 4) + bit 3, clamped.
        let gray = image(1, 12, false, vec![0, 7, 8, 4095, 2047]);
        let bytes = encode(&gray, OutputFormat::Bmp).expect("bmp");
        assert_eq!(bmp_gray(&bytes), vec![0, 0, 1, 255, 128]);
        let rgba = image(4, 8, false, vec![1, 2, 3, 4, 5, 6, 7, 8]);
        let (color, pixels) = decoded_bytes(
            image::ImageFormat::Bmp,
            &encode(&rgba, OutputFormat::Bmp).expect("bmp"),
        );
        assert_eq!(color, image::ColorType::Rgb8);
        assert_eq!(pixels, vec![1, 2, 3, 5, 6, 7]);
        let signed = image(1, 8, true, vec![-128, 0, 127]);
        assert_eq!(
            bmp_gray(&encode(&signed, OutputFormat::Bmp).expect("bmp")),
            vec![0, 128, 255]
        );
        let shallow = image(1, 4, false, vec![1]);
        assert!(encode(&shallow, OutputFormat::Bmp).is_err());
    }

    #[test]
    fn png_scales_like_openjpeg() {
        let twelve = image(1, 12, false, vec![0, 1, 4095]);
        let (color, bytes) = decoded_bytes(
            image::ImageFormat::Png,
            &encode(&twelve, OutputFormat::Png).expect("png"),
        );
        assert_eq!(color, image::ColorType::L16);
        let values = bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&pair| u16::from_ne_bytes(pair))
            .collect::<Vec<_>>();
        assert_eq!(values, vec![0, 16, 65535]);
        assert_eq!(scale(&[-2048, 2047], 12, 16, true), vec![-32768, 32752]);
        assert_eq!(scale(&[3], 2, 8, false), vec![255]);
        assert_eq!(clip(&[-1, 300], 8, false), vec![255, 255]);
        assert_eq!(clip(&[-200, 300], 8, true), vec![-128, 127]);
    }

    #[test]
    fn pnm_keeps_raw_samples_with_the_precision_maximum() {
        let gray = image(1, 12, false, vec![0, 4095]);
        let bytes = encode(&gray, OutputFormat::Pnm { gray: false }).expect("pnm");
        assert!(bytes.starts_with(b"P5\n"), "{:?}", &bytes[..12]);
        assert!(String::from_utf8_lossy(&bytes).contains("4095"));
        assert_eq!(&bytes[bytes.len() - 4..], &[0, 0, 0x0F, 0xFF]);
        let rgb = image(3, 8, false, vec![1, 2, 3]);
        let bytes = encode(&rgb, OutputFormat::Pnm { gray: true }).expect("pgm");
        assert!(bytes.starts_with(b"P5\n"));
        assert_eq!(bytes.last(), Some(&1));
        let bytes = encode(&rgb, OutputFormat::Pnm { gray: false }).expect("ppm");
        assert!(bytes.starts_with(b"P6\n"));
        assert!(bytes.ends_with(&[1, 2, 3]));
        let rgba = image(4, 16, false, vec![1, 2, 3, 4]);
        let bytes = encode(&rgba, OutputFormat::Pnm { gray: false }).expect("pam");
        assert!(bytes.starts_with(b"P7\n"));
        assert!(String::from_utf8_lossy(&bytes).contains("RGB_ALPHA"));
    }

    #[test]
    fn tiff_keeps_raw_samples() {
        let deep = image(3, 12, false, vec![1, 2, 4095]);
        let bytes = encode(&deep, OutputFormat::Tiff).expect("tiff");
        let decoder =
            image::codecs::tiff::TiffDecoder::new(Cursor::new(bytes.as_slice())).expect("tiff");
        assert_eq!(decoder.color_type(), image::ColorType::Rgb16);
        let (_, pixels) = decoded_bytes(image::ImageFormat::Tiff, &bytes);
        let values = pixels
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&pair| u16::from_ne_bytes(pair))
            .collect::<Vec<_>>();
        assert_eq!(values, vec![1, 2, 4095]);
    }

    #[test]
    fn force_rgb_replicates_gray_only() {
        let gray = force_rgb(image(1, 8, false, vec![7, 9]));
        assert_eq!((gray.channels, gray.samples), (3, vec![7, 7, 7, 9, 9, 9]));
        let rgb = force_rgb(image(3, 8, false, vec![1, 2, 3]));
        assert_eq!(rgb.samples, vec![1, 2, 3]);
    }
}
