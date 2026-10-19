// SPDX-License-Identifier: MIT OR Apache-2.0

//! Small CPU RGB codec boundary for reproducible research sweeps.

use pyo3::{prelude::*, types::PyBytes};

use crate::{owned_items, value_error};

/// Encode interleaved sRGB uint8 pixels. JPEG uses 4:4:4; wavelet codecs use ICT.
#[pyfunction]
#[pyo3(signature = (data, width, height, *, codec="htj2k", quality=None, target_bpp=None, levels=5, profile=None))]
#[expect(
    clippy::too_many_arguments,
    reason = "Python exposes these controls as keyword arguments"
)]
pub(crate) fn encode_rgb<'py>(
    py: Python<'py>,
    data: &Bound<'_, PyBytes>,
    width: u32,
    height: u32,
    codec: &str,
    quality: Option<u8>,
    target_bpp: Option<f64>,
    levels: u8,
    profile: Option<Vec<Vec<f32>>>,
) -> PyResult<Bound<'py, PyBytes>> {
    let pixels = data.as_bytes();
    if !matches!(codec, "jpeg" | "j2k" | "htj2k") {
        return Err(value_error("codec must be jpeg, j2k, or htj2k"));
    }
    if levels > 16
        || quality.is_some_and(|q| !(1..=100).contains(&q))
        || target_bpp.is_some_and(|x| !x.is_finite() || x <= 0.0)
    {
        return Err(value_error("invalid levels, quality, or target_bpp"));
    }
    if quality.is_some() && target_bpp.is_some() {
        return Err(value_error("quality and target_bpp are mutually exclusive"));
    }
    let encoded = py.detach(|| {
        if codec == "jpeg" {
            if profile.is_some() || target_bpp.is_some() {
                return Err(value_error(
                    "JPEG accepts quality only; rate-match its quality sweep externally",
                ));
            }
            return j2k_jpeg::encode_jpeg_baseline(
                j2k_jpeg::JpegSamples::Rgb8 {
                    data: pixels,
                    width,
                    height,
                },
                j2k_jpeg::JpegEncodeOptions {
                    quality: quality.unwrap_or(90),
                    subsampling: j2k_jpeg::JpegSubsampling::Ybr444,
                    backend: j2k_jpeg::JpegBackend::Cpu,
                    ..j2k_jpeg::JpegEncodeOptions::default()
                },
            )
            .map(|encoded| encoded.data)
            .map_err(value_error);
        }
        if codec == "j2k" && quality.is_some() {
            return Err(value_error(
                "quality is only supported for JPEG and HTJ2K Qfactor",
            ));
        }
        let samples =
            j2k::J2kLossySamples::new(pixels, width, height, 3, 8, false).map_err(value_error)?;
        let options = j2k::J2kLossyEncodeOptions::default()
            .with_cpu_only_backend()
            .with_max_decomposition_levels(Some(levels))
            .with_block_coding_mode(if codec == "htj2k" {
                j2k::J2kBlockCodingMode::HighThroughput
            } else {
                j2k::J2kBlockCodingMode::Classic
            })
            .with_rate_target(target_bpp.map(j2k::J2kRateTarget::BitsPerPixel))
            .with_qfactor(quality)
            .with_quantization_profile(profile);
        j2k::encode_j2k_lossy(samples, &options)
            .map(|encoded| encoded.codestream)
            .map_err(value_error)
    })?;
    Ok(PyBytes::new(py, &encoded))
}

/// Decode one encoded image to `(width, height, interleaved RGB uint8 bytes)`.
#[pyfunction]
pub(crate) fn decode_rgb<'py>(
    py: Python<'py>,
    data: &Bound<'_, PyBytes>,
) -> PyResult<(u32, u32, Bound<'py, PyBytes>)> {
    let encoded = data.as_bytes();
    let (width, height, pixels) = py.detach(|| {
        if encoded.starts_with(&[0xff, 0xd8]) {
            let decoder = j2k_jpeg::Decoder::new(encoded).map_err(value_error)?;
            if decoder.info().bit_depth != 8 {
                return Err(value_error("decode_rgb requires 8-bit JPEG"));
            }
            let (width, height) = decoder.info().dimensions;
            let (pixels, _) = decoder
                .decode_request(j2k_jpeg::DecodeRequest::full(j2k_jpeg::PixelFormat::Rgb8))
                .map_err(value_error)?;
            Ok((width, height, pixels))
        } else {
            let mut decoder = j2k::J2kDecoder::new(encoded).map_err(value_error)?;
            let info = decoder.info();
            if info.bit_depth != 8 || info.components != 3 {
                return Err(value_error(
                    "decode_rgb requires three 8-bit JPEG 2000 components",
                ));
            }
            let (width, height) = info.dimensions;
            let length = (width as usize)
                .checked_mul(height as usize)
                .and_then(|x| x.checked_mul(3))
                .ok_or_else(|| value_error("RGB geometry overflow"))?;
            if length > j2k_native::DEFAULT_MAX_CODEC_BYTES {
                return Err(value_error("RGB output exceeds the codec byte limit"));
            }
            let mut pixels = owned_items((0..length).map(|_| 0_u8))?;
            decoder
                .decode_into(&mut pixels, width as usize * 3, j2k::PixelFormat::Rgb8)
                .map_err(value_error)?;
            Ok((width, height, pixels))
        }
    })?;
    Ok((width, height, PyBytes::new(py, &pixels)))
}
