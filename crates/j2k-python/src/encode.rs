// SPDX-License-Identifier: MIT OR Apache-2.0

//! CPU encoding: general JPEG 2000 / HTJ2K samples, and a small RGB codec
//! boundary for reproducible research sweeps. Keyword arguments translate into
//! `j2k` encode options; the codec validates and applies every coding rule.

use pyo3::{
    prelude::*,
    types::{PyBytes, PyDict},
};

use crate::{owned_items, value_error};

/// One target, or cumulative quality-layer targets.
#[derive(FromPyObject)]
pub(crate) enum Targets<T> {
    One(T),
    Layers(Vec<T>),
}

impl<T> Targets<T> {
    fn into_targets(
        self,
        unit: impl Fn(T) -> j2k::J2kRateTarget,
    ) -> PyResult<Vec<j2k::J2kRateTarget>> {
        match self {
            Self::One(target) => owned_items(std::iter::once(unit(target))),
            Self::Layers(targets) => owned_items(targets.into_iter().map(unit)),
        }
    }
}

/// The coding parameters `encode` accepts, as the codec names them.
struct Coding {
    block_mode: j2k::J2kBlockCodingMode,
    progression: j2k::J2kProgressionOrder,
    markers: Vec<j2k::J2kMarkerSegment>,
}

impl Coding {
    fn parse(codec: &str, progression: &str, markers: &[String]) -> PyResult<Self> {
        Ok(Self {
            block_mode: match codec {
                "j2k" => j2k::J2kBlockCodingMode::Classic,
                "htj2k" => j2k::J2kBlockCodingMode::HighThroughput,
                _ => return Err(value_error("codec must be j2k or htj2k")),
            },
            progression: match progression {
                "lrcp" => j2k::J2kProgressionOrder::Lrcp,
                "rlcp" => j2k::J2kProgressionOrder::Rlcp,
                "rpcl" => j2k::J2kProgressionOrder::Rpcl,
                "pcrl" => j2k::J2kProgressionOrder::Pcrl,
                "cprl" => j2k::J2kProgressionOrder::Cprl,
                _ => {
                    return Err(value_error(
                        "progression must be lrcp, rlcp, rpcl, pcrl, or cprl",
                    ))
                }
            },
            markers: markers
                .iter()
                .map(|marker| match marker.as_str() {
                    "sop" => Ok(j2k::J2kMarkerSegment::Sop),
                    "eph" => Ok(j2k::J2kMarkerSegment::Eph),
                    "tlm" => Ok(j2k::J2kMarkerSegment::Tlm),
                    "plt" => Ok(j2k::J2kMarkerSegment::Plt),
                    "plm" => Ok(j2k::J2kMarkerSegment::Plm),
                    "ppm" => Ok(j2k::J2kMarkerSegment::Ppm),
                    "ppt" => Ok(j2k::J2kMarkerSegment::Ppt),
                    _ => Err(value_error(
                        "markers must be among sop, eph, tlm, plt, plm, ppm, and ppt",
                    )),
                })
                .collect::<PyResult<_>>()?,
        })
    }
}

/// What the codec measured on an encoded codestream.
struct Report {
    codestream_bytes: u64,
    bits_per_pixel: f64,
    psnr_db: Option<f64>,
    quality_layers: u16,
}

/// Encode interleaved samples: lossless by default, or lossy with optional
/// rate or quality targets. Samples of up to 8 bits take one byte; deeper
/// samples take two, little-endian.
#[pyfunction]
#[pyo3(signature = (
    data, width, height, *, components=3, bit_depth=8, signed=false, codec="htj2k",
    mode="lossless", levels=None, progression="lrcp", tile_size=None, tile_part_packets=None,
    markers=Vec::new(), quality_layers=None, color_transform="auto", qfactor=None,
    target_bpp=None, target_bytes=None, target_psnr=None, file=false, icc_profile=None,
    return_report=false,
))]
#[expect(
    clippy::too_many_arguments,
    reason = "Python exposes these controls as keyword arguments"
)]
pub(crate) fn encode<'py>(
    py: Python<'py>,
    data: &Bound<'_, PyBytes>,
    width: u32,
    height: u32,
    components: u16,
    bit_depth: u8,
    signed: bool,
    codec: &str,
    mode: &str,
    levels: Option<u8>,
    progression: &str,
    tile_size: Option<(u32, u32)>,
    tile_part_packets: Option<u16>,
    markers: Vec<String>,
    quality_layers: Option<u8>,
    color_transform: &str,
    qfactor: Option<u8>,
    target_bpp: Option<Targets<f64>>,
    target_bytes: Option<Targets<u64>>,
    target_psnr: Option<Targets<f64>>,
    file: bool,
    icc_profile: Option<&[u8]>,
    return_report: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let coding = Coding::parse(codec, progression, &markers)?;
    let lossless = match mode {
        "lossless" => true,
        "lossy" => false,
        _ => return Err(value_error("mode must be lossless or lossy")),
    };
    // "auto" applies the mode's colour transform to three components.
    let color_transform = match color_transform {
        "auto" => true,
        "none" => false,
        _ => return Err(value_error("color_transform must be auto or none")),
    };
    let targets = match (target_bpp, target_bytes, target_psnr) {
        (None, None, None) => Vec::new(),
        (Some(bpp), None, None) => bpp.into_targets(j2k::J2kRateTarget::BitsPerPixel)?,
        (None, Some(bytes), None) => bytes.into_targets(j2k::J2kRateTarget::Bytes)?,
        (None, None, Some(psnr)) => psnr.into_targets(j2k::J2kRateTarget::PsnrDb)?,
        _ => {
            return Err(value_error(
                "target_bpp, target_bytes, and target_psnr are mutually exclusive",
            ))
        }
    };
    if icc_profile.is_some() && !file {
        return Err(value_error("icc_profile needs file=True"));
    }
    let samples = data.as_bytes();
    let (bytes, report) = py.detach(|| {
        let (codestream, report) = if lossless {
            if qfactor.is_some() || !targets.is_empty() {
                return Err(value_error("lossless encoding takes no qfactor or targets"));
            }
            let input = j2k::J2kLosslessSamples::new(
                samples, width, height, components, bit_depth, signed,
            )
            .map_err(value_error)?;
            let layers = quality_layers.unwrap_or(1);
            let options = j2k::J2kLosslessEncodeOptions::default()
                .with_cpu_only_backend()
                .with_block_coding_mode(coding.block_mode)
                .with_progression(coding.progression)
                .with_max_decomposition_levels(levels)
                .with_tile_size(tile_size)
                .with_tile_part_packet_limit(tile_part_packets)
                .with_quality_layers(layers)
                .with_marker_segments(&coding.markers)
                .with_reversible_transform(if color_transform {
                    j2k::ReversibleTransform::Rct53
                } else {
                    j2k::ReversibleTransform::None53
                });
            let codestream = j2k::encode_j2k_lossless(input, &options)
                .map_err(value_error)?
                .codestream;
            let length = codestream.len() as u64;
            #[expect(
                clippy::cast_precision_loss,
                reason = "a rate is a ratio of byte and pixel counts far below 2^53"
            )]
            let bits_per_pixel = length as f64 * 8.0 / (f64::from(width) * f64::from(height));
            let report = Report {
                codestream_bytes: length,
                bits_per_pixel,
                psnr_db: None,
                quality_layers: u16::from(layers),
            };
            (codestream, report)
        } else {
            if quality_layers.is_some() || !color_transform {
                return Err(value_error(
                    "lossy layers come from list targets, and lossy coding keeps its colour transform",
                ));
            }
            let input =
                j2k::J2kLossySamples::new(samples, width, height, components, bit_depth, signed)
                    .map_err(value_error)?;
            let options = j2k::J2kLossyEncodeOptions::default()
                .with_cpu_only_backend()
                .with_block_coding_mode(coding.block_mode)
                .with_progression(coding.progression)
                .with_max_decomposition_levels(levels)
                .with_tile_size(tile_size)
                .with_tile_part_packet_limit(tile_part_packets)
                .with_marker_segments(coding.markers)
                .with_qfactor(qfactor);
            let options = match targets.as_slice() {
                [] => options,
                [target] => options.with_rate_target(Some(*target)),
                layers => options.with_quality_layers(owned_items(
                    layers.iter().copied().map(j2k::J2kQualityLayer::new),
                )?),
            };
            let encoded = j2k::encode_j2k_lossy(input, &options).map_err(value_error)?;
            let report = Report {
                codestream_bytes: encoded.report.actual_bytes,
                bits_per_pixel: encoded.report.actual_bits_per_pixel,
                psnr_db: encoded.report.psnr_db,
                quality_layers: encoded.report.quality_layers,
            };
            (encoded.codestream, report)
        };
        if !file {
            return Ok((codestream, report));
        }
        let wrap = if coding.block_mode == j2k::J2kBlockCodingMode::HighThroughput {
            j2k::J2kFileWrapOptions::jph()
        } else {
            j2k::J2kFileWrapOptions::jp2()
        };
        let wrap = match icc_profile {
            Some(profile) => wrap.with_color(j2k::J2kFileColorSpec::IccProfile(profile)),
            None => wrap,
        };
        let bytes = j2k::wrap_j2k_codestream(&codestream, wrap).map_err(value_error)?;
        Ok((bytes, report))
    })?;
    let bytes = PyBytes::new(py, &bytes).into_any();
    if !return_report {
        return Ok(bytes);
    }
    let summary = PyDict::new(py);
    summary.set_item("codestream_bytes", report.codestream_bytes)?;
    summary.set_item("bits_per_pixel", report.bits_per_pixel)?;
    summary.set_item("psnr_db", report.psnr_db)?;
    summary.set_item("quality_layers", report.quality_layers)?;
    Ok((bytes, summary).into_pyobject(py)?.into_any())
}

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
    data: &Bound<'_, PyAny>,
) -> PyResult<(u32, u32, Bound<'py, PyBytes>)> {
    let input = crate::input::EncodedInput::extract(data)?;
    let encoded = input.as_bytes();
    let (width, height, pixels) = py.detach(|| {
        if crate::is_jpeg(encoded) {
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
