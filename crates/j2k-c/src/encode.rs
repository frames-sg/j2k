// SPDX-License-Identifier: MIT OR Apache-2.0

//! Encoding of interleaved samples: lossless, or lossy to rate and quality
//! targets. This module only translates ABI fields into `j2k` encode options;
//! the codec validates and applies every coding rule.

#![forbid(unsafe_code)]

use crate::{
    abi::{
        J2kCEncodeOptions, J2kCEncodeParams, J2kCEncodeReport, J2K_ENCODE_FILE, J2K_ENCODE_FLAGS,
        J2K_ENCODE_HTJ2K, J2K_ENCODE_SIGNED, J2K_LEVELS_AUTO, J2K_MARKERS, J2K_MARKER_EPH,
        J2K_MARKER_PLM, J2K_MARKER_PLT, J2K_MARKER_PPM, J2K_MARKER_PPT, J2K_MARKER_SOP,
        J2K_MARKER_TLM, J2K_MODE_LOSSLESS, J2K_MODE_LOSSY, J2K_PARAMS_FLAGS,
        J2K_PARAMS_NO_COLOR_TRANSFORM, J2K_PROGRESSION_CPRL, J2K_PROGRESSION_LRCP,
        J2K_PROGRESSION_PCRL, J2K_PROGRESSION_RLCP, J2K_PROGRESSION_RPCL, J2K_RATE_BITS_PER_PIXEL,
        J2K_RATE_BYTES, J2K_RATE_NONE, J2K_RATE_PSNR_DB,
    },
    error::{Error, Result},
};
use j2k::{
    encode_j2k_lossless, encode_j2k_lossy, wrap_j2k_codestream, J2kBlockCodingMode,
    J2kFileColorSpec, J2kFileWrapOptions, J2kLosslessEncodeOptions, J2kLosslessSamples,
    J2kLossyEncodeOptions, J2kLossySamples, J2kMarkerSegment, J2kProgressionOrder, J2kQualityLayer,
    J2kRateTarget, ReversibleTransform,
};

/// `j2k_encode_params` with its arrays borrowed as slices.
#[derive(Default)]
pub(crate) struct EncodeParams<'a> {
    pub(crate) fields: J2kCEncodeParams,
    pub(crate) targets: &'a [f64],
    pub(crate) icc_profile: Option<&'a [u8]>,
}

/// Encoded bytes and what the codec measured on their codestream.
pub(crate) struct Encoded {
    pub(crate) bytes: Vec<u8>,
    pub(crate) report: J2kCEncodeReport,
}

/// The sample layout `options` describes.
struct Geometry {
    width: u32,
    height: u32,
    components: u16,
    bit_depth: u8,
    signed: bool,
}

impl Geometry {
    fn from_options(options: &J2kCEncodeOptions) -> Result<Self> {
        if options.flags & !J2K_ENCODE_FLAGS != 0 || options.reserved != 0 {
            return Err(Error::argument(
                "unknown encode flags, or nonzero reserved field",
            ));
        }
        Ok(Self {
            width: options.width,
            height: options.height,
            components: u16::try_from(options.components)
                .map_err(|_| Error::argument("component count exceeds 65535"))?,
            bit_depth: u8::try_from(options.bit_depth)
                .ok()
                .filter(|bits| (1..=16).contains(bits))
                .ok_or_else(|| Error::argument("bit depth must be between 1 and 16"))?,
            signed: options.flags & J2K_ENCODE_SIGNED != 0,
        })
    }

    fn bits_per_pixel(&self, bytes: usize) -> f64 {
        // Exact for any codestream below 2^53 bytes and image below 2^53 pixels.
        #[expect(
            clippy::cast_precision_loss,
            reason = "a rate is a ratio of byte and pixel counts far below 2^53"
        )]
        let (bits, pixels) = (
            bytes as f64 * 8.0,
            f64::from(self.width) * f64::from(self.height),
        );
        bits / pixels
    }
}

/// The options both coding modes share, translated from ABI fields.
struct Coding {
    block_mode: J2kBlockCodingMode,
    progression: J2kProgressionOrder,
    levels: Option<u8>,
    tile_size: Option<(u32, u32)>,
    tile_part_packets: Option<u16>,
    markers: Vec<J2kMarkerSegment>,
}

impl Coding {
    fn from_fields(options: &J2kCEncodeOptions, params: &J2kCEncodeParams) -> Result<Self> {
        if params.flags & !J2K_PARAMS_FLAGS != 0 || params.markers & !J2K_MARKERS != 0 {
            return Err(Error::argument("unknown encode parameter or marker flags"));
        }
        Ok(Self {
            block_mode: if options.flags & J2K_ENCODE_HTJ2K != 0 {
                J2kBlockCodingMode::HighThroughput
            } else {
                J2kBlockCodingMode::Classic
            },
            progression: progression(params.progression)?,
            levels: if params.levels == J2K_LEVELS_AUTO {
                None
            } else {
                Some(
                    u8::try_from(params.levels)
                        .map_err(|_| Error::argument("decomposition levels exceed 255"))?,
                )
            },
            tile_size: match (params.tile_width, params.tile_height) {
                (0, 0) => None,
                size => Some(size),
            },
            tile_part_packets: match params.tile_part_packets {
                0 => None,
                packets => Some(
                    u16::try_from(packets)
                        .map_err(|_| Error::argument("tile-part packet limit exceeds 65535"))?,
                ),
            },
            markers: [
                (J2K_MARKER_SOP, J2kMarkerSegment::Sop),
                (J2K_MARKER_EPH, J2kMarkerSegment::Eph),
                (J2K_MARKER_TLM, J2kMarkerSegment::Tlm),
                (J2K_MARKER_PLT, J2kMarkerSegment::Plt),
                (J2K_MARKER_PLM, J2kMarkerSegment::Plm),
                (J2K_MARKER_PPM, J2kMarkerSegment::Ppm),
                (J2K_MARKER_PPT, J2kMarkerSegment::Ppt),
            ]
            .into_iter()
            .filter(|&(bit, _)| params.markers & bit != 0)
            .map(|(_, marker)| marker)
            .collect(),
        })
    }
}

fn progression(value: u32) -> Result<J2kProgressionOrder> {
    Ok(match value {
        J2K_PROGRESSION_LRCP => J2kProgressionOrder::Lrcp,
        J2K_PROGRESSION_RLCP => J2kProgressionOrder::Rlcp,
        J2K_PROGRESSION_RPCL => J2kProgressionOrder::Rpcl,
        J2K_PROGRESSION_PCRL => J2kProgressionOrder::Pcrl,
        J2K_PROGRESSION_CPRL => J2kProgressionOrder::Cprl,
        _ => return Err(Error::argument("unknown progression order")),
    })
}

/// The lossy targets `rate` and `targets` describe, in codec units.
fn rate_targets(rate: u32, targets: &[f64]) -> Result<Vec<J2kRateTarget>> {
    // The largest byte count an f64 represents exactly.
    const MAX_EXACT_BYTES: f64 = 9_007_199_254_740_992.0;
    if (rate == J2K_RATE_NONE) != targets.is_empty() {
        return Err(Error::argument(
            "lossy targets need a rate unit, and a rate unit needs targets",
        ));
    }
    targets
        .iter()
        .map(|&target| match rate {
            J2K_RATE_BITS_PER_PIXEL => Ok(J2kRateTarget::BitsPerPixel(target)),
            J2K_RATE_PSNR_DB => Ok(J2kRateTarget::PsnrDb(target)),
            J2K_RATE_BYTES
                if target.fract() == 0.0 && (1.0..=MAX_EXACT_BYTES).contains(&target) =>
            {
                #[expect(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "a positive whole number no larger than 2^53 converts exactly"
                )]
                Ok(J2kRateTarget::Bytes(target as u64))
            }
            J2K_RATE_BYTES => Err(Error::argument(
                "byte targets must be whole numbers from 1 to 2^53",
            )),
            _ => Err(Error::argument("unknown rate unit")),
        })
        .collect()
}

/// Encode `samples` as `options` and `params` describe: a raw codestream, or a
/// JP2/JPH file with `J2K_ENCODE_FILE`. The CPU encoder decodes its own output
/// before returning it: lossless output must reproduce `samples` exactly.
pub(crate) fn encode(
    samples: &[u8],
    options: &J2kCEncodeOptions,
    params: &EncodeParams<'_>,
) -> Result<Encoded> {
    let geometry = Geometry::from_options(options)?;
    let coding = Coding::from_fields(options, &params.fields)?;
    let file = options.flags & J2K_ENCODE_FILE != 0;
    if params.icc_profile.is_some() && !file {
        return Err(Error::argument("an ICC profile needs J2K_ENCODE_FILE"));
    }
    let (codestream, report) = match params.fields.mode {
        J2K_MODE_LOSSLESS => encode_lossless(samples, &geometry, &coding, params)?,
        J2K_MODE_LOSSY => encode_lossy(samples, &geometry, coding, params)?,
        _ => return Err(Error::argument("unknown encode mode")),
    };
    if !file {
        return Ok(Encoded {
            bytes: codestream,
            report,
        });
    }
    let wrap = if options.flags & J2K_ENCODE_HTJ2K != 0 {
        J2kFileWrapOptions::jph()
    } else {
        J2kFileWrapOptions::jp2()
    };
    let wrap = match params.icc_profile {
        Some(profile) => wrap.with_color(J2kFileColorSpec::IccProfile(profile)),
        None => wrap,
    };
    Ok(Encoded {
        bytes: wrap_j2k_codestream(&codestream, wrap)?,
        report,
    })
}

fn encode_lossless(
    samples: &[u8],
    geometry: &Geometry,
    coding: &Coding,
    params: &EncodeParams<'_>,
) -> Result<(Vec<u8>, J2kCEncodeReport)> {
    let fields = &params.fields;
    if fields.qfactor != 0 || fields.rate != J2K_RATE_NONE || !params.targets.is_empty() {
        return Err(Error::argument(
            "lossless encoding takes no qfactor, rate unit, or targets",
        ));
    }
    let quality_layers = match fields.quality_layers {
        0 => 1,
        layers => {
            u8::try_from(layers).map_err(|_| Error::argument("quality layer count exceeds 255"))?
        }
    };
    let input = J2kLosslessSamples::new(
        samples,
        geometry.width,
        geometry.height,
        geometry.components,
        geometry.bit_depth,
        geometry.signed,
    )?;
    let settings = J2kLosslessEncodeOptions::default()
        .with_cpu_only_backend()
        .with_block_coding_mode(coding.block_mode)
        .with_progression(coding.progression)
        .with_max_decomposition_levels(coding.levels)
        .with_tile_size(coding.tile_size)
        .with_tile_part_packet_limit(coding.tile_part_packets)
        .with_quality_layers(quality_layers)
        .with_marker_segments(&coding.markers)
        .with_reversible_transform(if fields.flags & J2K_PARAMS_NO_COLOR_TRANSFORM != 0 {
            ReversibleTransform::None53
        } else {
            ReversibleTransform::Rct53
        });
    let codestream = encode_j2k_lossless(input, &settings)?.codestream;
    let report = J2kCEncodeReport {
        codestream_bytes: codestream.len() as u64,
        bits_per_pixel: geometry.bits_per_pixel(codestream.len()),
        quality_layers: u32::from(quality_layers),
        ..Default::default()
    };
    Ok((codestream, report))
}

fn encode_lossy(
    samples: &[u8],
    geometry: &Geometry,
    coding: Coding,
    params: &EncodeParams<'_>,
) -> Result<(Vec<u8>, J2kCEncodeReport)> {
    let fields = &params.fields;
    if fields.quality_layers != 0 || fields.flags & J2K_PARAMS_NO_COLOR_TRANSFORM != 0 {
        return Err(Error::argument(
            "lossy layers come from targets, and lossy coding keeps its colour transform",
        ));
    }
    let qfactor = match fields.qfactor {
        0 => None,
        quality => Some(u8::try_from(quality).map_err(|_| Error::argument("qfactor exceeds 255"))?),
    };
    let targets = rate_targets(fields.rate, params.targets)?;
    let input = J2kLossySamples::new(
        samples,
        geometry.width,
        geometry.height,
        geometry.components,
        geometry.bit_depth,
        geometry.signed,
    )?;
    let settings = J2kLossyEncodeOptions::default()
        .with_cpu_only_backend()
        .with_block_coding_mode(coding.block_mode)
        .with_progression(coding.progression)
        .with_max_decomposition_levels(coding.levels)
        .with_tile_size(coding.tile_size)
        .with_tile_part_packet_limit(coding.tile_part_packets)
        .with_marker_segments(coding.markers)
        .with_qfactor(qfactor);
    let settings = match targets.as_slice() {
        [] => settings,
        [target] => settings.with_rate_target(Some(*target)),
        layers => {
            settings.with_quality_layers(layers.iter().copied().map(J2kQualityLayer::new).collect())
        }
    };
    let encoded = encode_j2k_lossy(input, &settings)?;
    let report = J2kCEncodeReport {
        codestream_bytes: encoded.report.actual_bytes,
        bits_per_pixel: encoded.report.actual_bits_per_pixel,
        psnr_db: encoded.report.psnr_db.unwrap_or(0.0),
        quality_layers: u32::from(encoded.report.quality_layers),
        ..Default::default()
    };
    Ok((encoded.codestream, report))
}
