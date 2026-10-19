// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use crate::{DecodeRequest, Decoder};
use j2k_core::PixelFormat;

fn options(subsampling: JpegSubsampling, quality: u8) -> JpegEncodeOptions {
    JpegEncodeOptions {
        quality,
        subsampling,
        restart_interval: None,
        backend: JpegBackend::Cpu,
    }
}

/// A smooth 12-bit field spanning most of the range: gradients whose slope
/// differs per channel, so chroma varies across the image.
fn field(width: u32, height: u32, channels: usize) -> Vec<u16> {
    let mut data = Vec::new();
    for y in 0..height {
        for x in 0..width {
            for channel in 0..channels {
                let channel = u32::try_from(channel).expect("channel");
                let value = 100 + x * (1000 + 500 * channel) / width + y * 1500 / height;
                data.push(u16::try_from(value).expect("12-bit"));
            }
        }
    }
    data
}

fn decode16(bytes: &[u8], fmt: PixelFormat) -> Vec<u16> {
    let (pixels, _) = Decoder::new(bytes)
        .expect("decoder")
        .decode_request(DecodeRequest::full(fmt))
        .expect("decode");
    pixels
        .as_chunks::<2>()
        .0
        .iter()
        .map(|&pair| u16::from_ne_bytes(pair))
        .collect()
}

fn max_error(a: &[u16], b: &[u16]) -> u16 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| x.abs_diff(y))
        .max()
        .unwrap_or(0)
}

#[test]
fn grayscale_round_trips_within_quantization_error() {
    let (width, height) = (45, 29);
    let data = field(width, height, 1);
    let encoded = encode_jpeg_extended12(
        JpegSamples12::Gray {
            data: &data,
            width,
            height,
        },
        options(JpegSubsampling::Gray, 100),
    )
    .expect("encode");
    // SOF1, 12-bit precision.
    let sof = encoded
        .data
        .windows(2)
        .position(|w| w == [0xFF, 0xC1])
        .expect("SOF1");
    assert_eq!(encoded.data[sof + 4], 12);
    let decoded = decode16(&encoded.data, PixelFormat::Gray16);
    assert!(
        max_error(&decoded, &data) <= 3,
        "max error {}",
        max_error(&decoded, &data)
    );
}

#[test]
fn color_round_trips_at_every_subsampling() {
    let (width, height) = (37, 21);
    let data = field(width, height, 3);
    for subsampling in [
        JpegSubsampling::Ybr444,
        JpegSubsampling::Ybr422,
        JpegSubsampling::Ybr420,
    ] {
        for restart_interval in [None, Some(2)] {
            let mut options = options(subsampling, 95);
            options.restart_interval = restart_interval;
            let encoded = encode_jpeg_extended12(
                JpegSamples12::Rgb {
                    data: &data,
                    width,
                    height,
                },
                options,
            )
            .expect("encode");
            let decoded = decode16(&encoded.data, PixelFormat::Rgb16);
            let error = max_error(&decoded, &data);
            assert!(
                error <= 48,
                "{subsampling:?} {restart_interval:?}: max error {error}"
            );
            let markers = encoded
                .data
                .windows(2)
                .filter(|w| w[0] == 0xFF && (0xD0..=0xD7).contains(&w[1]))
                .count();
            assert_eq!(markers > 0, restart_interval.is_some());
        }
    }
}

#[test]
fn invalid_inputs_are_rejected() {
    let data = [0_u16, 4096, 0, 0];
    let samples = JpegSamples12::Gray {
        data: &data,
        width: 2,
        height: 2,
    };
    assert!(matches!(
        encode_jpeg_extended12(samples, options(JpegSubsampling::Gray, 90)),
        Err(JpegEncodeError::SampleOutOfRange {
            index: 1,
            value: 4096,
            precision: 12
        })
    ));
    assert!(matches!(
        encode_jpeg_extended12(samples, options(JpegSubsampling::Ybr420, 90)),
        Err(JpegEncodeError::IncompatibleSubsampling { .. })
    ));
    let mut cuda = options(JpegSubsampling::Gray, 90);
    cuda.backend = JpegBackend::Cuda;
    assert!(matches!(
        encode_jpeg_extended12(samples, cuda),
        Err(JpegEncodeError::UnsupportedBackend { .. })
    ));
    assert!(matches!(
        encode_jpeg_extended12(
            JpegSamples12::Rgb {
                data: &data,
                width: 2,
                height: 2
            },
            options(JpegSubsampling::Ybr444, 90)
        ),
        Err(JpegEncodeError::SampleLength {
            expected: 12,
            actual: 4
        })
    ));
}

#[test]
fn black_and_white_extremes_stay_in_range() {
    let (width, height) = (16, 16);
    let data: Vec<u16> = (0..width * height)
        .map(|index| if (index / 3) % 2 == 0 { 0 } else { 4095 })
        .collect();
    let encoded = encode_jpeg_extended12(
        JpegSamples12::Gray {
            data: &data,
            width,
            height,
        },
        options(JpegSubsampling::Gray, 100),
    )
    .expect("encode");
    let decoded = decode16(&encoded.data, PixelFormat::Gray16);
    assert!(max_error(&decoded, &data) <= 8);
}
