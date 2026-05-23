// SPDX-License-Identifier: Apache-2.0

//! JPEG 2000-family coefficient-domain recoding APIs.
//!
//! The direct 5/3 path decodes classic JPEG 2000 Tier-1 code-blocks into
//! reversible wavelet coefficients and re-encodes those coefficients with
//! HTJ2K block coding. It does not run inverse DWT, forward DWT, or a
//! pixel-domain lossless encode. The output is coefficient-preserving for the
//! supported reversible 5/3 profile, not byte-preserving unless passthrough is
//! reported.

use alloc::vec::Vec;

use signinum_core::{
    Colorspace, CompressedPayloadKind, CompressedTransferSyntax, PassthroughRequirements,
    Unsupported,
};
use signinum_j2k_native::{DecodeSettings, EncodeOptions, EncodeProgressionOrder, Image};

use crate::{
    encode::{J2kEncodeValidation, J2kProgressionOrder},
    parse::{parse_image_info, ParsedImageInfo},
    J2kError, J2kView,
};

/// Options for classic JPEG 2000 reversible 5/3 to HTJ2K lossless recoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct J2kToHtj2kOptions {
    /// Requested output payload shape.
    ///
    /// DICOM encapsulated WSI frames use raw JPEG 2000-family codestreams, so
    /// the default is [`CompressedPayloadKind::Jpeg2000Codestream`]. JP2 output
    /// is not produced by the coefficient-domain recoder yet.
    pub output_payload_kind: CompressedPayloadKind,
    /// Output packet progression order.
    pub progression: J2kProgressionOrder,
    /// Optional decoded-pixel validation of the produced codestream.
    pub validation: J2kEncodeValidation,
}

impl Default for J2kToHtj2kOptions {
    fn default() -> Self {
        Self {
            output_payload_kind: CompressedPayloadKind::Jpeg2000Codestream,
            progression: J2kProgressionOrder::Lrcp,
            validation: J2kEncodeValidation::CpuRoundTrip,
        }
    }
}

/// Recode path used for a J2K/JP2 to HTJ2K request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum J2kToHtj2kMode {
    /// Input bytes already matched the requested HTJ2K transfer syntax and
    /// payload kind, so bytes were copied unchanged.
    Passthrough,
    /// Classic reversible 5/3 code-blocks were entropy-decoded to quantized
    /// wavelet coefficients and re-encoded with HT block coding.
    CoefficientPreserving,
    /// Reserved for an explicit decode-pixels/re-encode fallback.
    PixelPreserving,
}

/// Metadata describing a J2K/JP2 to HTJ2K recode.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct J2kToHtj2kReport {
    pub mode: J2kToHtj2kMode,
    pub input_transfer_syntax: CompressedTransferSyntax,
    pub output_transfer_syntax: CompressedTransferSyntax,
    pub input_payload_kind: CompressedPayloadKind,
    pub output_payload_kind: CompressedPayloadKind,
    pub width: u32,
    pub height: u32,
    pub components: u8,
    pub bit_depth: u8,
}

/// HTJ2K codestream bytes and recode metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReencodedHtj2k {
    pub bytes: Vec<u8>,
    pub report: J2kToHtj2kReport,
}

/// Recode a classic JPEG 2000 reversible 5/3 J2K/JP2 input to lossless HTJ2K.
///
/// This is a JPEG 2000-family coefficient-domain recode. For supported classic
/// lossless 5/3 sources it preserves decoded quantized wavelet coefficients and
/// changes only the block coding and packetized codestream representation. It
/// is not a DCT JPEG transcode and does not claim byte preservation except when
/// [`J2kToHtj2kMode::Passthrough`] is reported.
pub fn recode_j2k_to_htj2k_lossless(
    bytes: &[u8],
    options: J2kToHtj2kOptions,
) -> Result<ReencodedHtj2k, J2kError> {
    let view = J2kView::parse(bytes)?;
    let parsed = parse_image_info(bytes)?;
    let info = view.info().clone();
    let output_transfer_syntax = CompressedTransferSyntax::HtJpeg2000Lossless;

    if let Some(candidate) = view.passthrough_candidate() {
        let requirements =
            PassthroughRequirements::new(output_transfer_syntax, options.output_payload_kind);
        if let Ok(copy) = candidate.copy_bytes_if_eligible(&requirements) {
            return Ok(ReencodedHtj2k {
                bytes: copy.to_vec(),
                report: J2kToHtj2kReport {
                    mode: J2kToHtj2kMode::Passthrough,
                    input_transfer_syntax: candidate.transfer_syntax(),
                    output_transfer_syntax,
                    input_payload_kind: candidate.payload_kind(),
                    output_payload_kind: options.output_payload_kind,
                    width: info.dimensions.0,
                    height: info.dimensions.1,
                    components: info.components,
                    bit_depth: info.bit_depth,
                },
            });
        }
    }

    validate_recode_request(&parsed, options)?;

    let source = Image::new(bytes, &DecodeSettings::default())
        .map_err(|err| map_native_decode_error(err, "source JPEG 2000 parse failed"))?;
    let coefficients = source
        .decode_reversible_53_coefficients()
        .map_err(|err| map_native_decode_error(err, "source coefficient extraction failed"))?;

    let encode_options = native_encode_options(options, &coefficients);
    let codestream = signinum_j2k_native::encode_precomputed_htj2k_53_with_mct(
        &coefficients.image,
        &encode_options,
        coefficients.use_mct,
    )
    .map_err(|err| J2kError::Backend(format!("HTJ2K coefficient recode failed: {err}")))?;

    if options.validation == J2kEncodeValidation::CpuRoundTrip {
        validate_recode_roundtrip(bytes, &codestream)?;
    }

    Ok(ReencodedHtj2k {
        bytes: codestream,
        report: J2kToHtj2kReport {
            mode: J2kToHtj2kMode::CoefficientPreserving,
            input_transfer_syntax: parsed.transfer_syntax,
            output_transfer_syntax,
            input_payload_kind: parsed.payload_kind,
            output_payload_kind: options.output_payload_kind,
            width: parsed.info.dimensions.0,
            height: parsed.info.dimensions.1,
            components: parsed.info.components,
            bit_depth: parsed.info.bit_depth,
        },
    })
}

fn validate_recode_request(
    parsed: &ParsedImageInfo,
    options: J2kToHtj2kOptions,
) -> Result<(), J2kError> {
    if options.output_payload_kind != CompressedPayloadKind::Jpeg2000Codestream {
        return Err(Unsupported {
            what: "coefficient-domain J2K to HTJ2K recode currently emits only raw codestreams",
        }
        .into());
    }
    if parsed.transfer_syntax != CompressedTransferSyntax::Jpeg2000Lossless {
        return Err(Unsupported {
            what: "coefficient-domain lossless recode currently supports only classic lossless J2K",
        }
        .into());
    }
    if !matches!(parsed.info.components, 1 | 3) {
        return Err(Unsupported {
            what: "coefficient-domain lossless recode supports only grayscale or RGB component counts",
        }
        .into());
    }
    if !matches!(parsed.info.bit_depth, 8 | 16) {
        return Err(Unsupported {
            what: "coefficient-domain lossless recode supports only 8-bit or 16-bit sources",
        }
        .into());
    }
    if !matches!(
        parsed.info.colorspace,
        Colorspace::Grayscale
            | Colorspace::SGray
            | Colorspace::Rgb
            | Colorspace::SRgb
            | Colorspace::Rct
    ) {
        return Err(Unsupported {
            what: "coefficient-domain lossless recode supports only Gray/RGB/RCT colorspaces",
        }
        .into());
    }
    if parsed.components.iter().any(|component| component.signed) {
        return Err(Unsupported {
            what: "signed JPEG 2000 sources are not supported for coefficient-domain recode yet",
        }
        .into());
    }
    if parsed
        .components
        .iter()
        .any(|component| component.bit_depth != parsed.info.bit_depth)
    {
        return Err(Unsupported {
            what: "mixed component bit depths are not supported for coefficient-domain recode",
        }
        .into());
    }
    if parsed
        .components
        .iter()
        .any(|component| component.x_rsiz != 1 || component.y_rsiz != 1)
    {
        return Err(Unsupported {
            what: "component subsampling is not supported for coefficient-domain recode yet",
        }
        .into());
    }
    Ok(())
}

fn native_encode_options(
    options: J2kToHtj2kOptions,
    coefficients: &signinum_j2k_native::Reversible53CoefficientImage,
) -> EncodeOptions {
    EncodeOptions {
        reversible: true,
        use_ht_block_coding: true,
        use_mct: coefficients.use_mct,
        code_block_width_exp: coefficients.code_block_width_exp,
        code_block_height_exp: coefficients.code_block_height_exp,
        guard_bits: coefficients.guard_bits,
        progression_order: native_progression(options.progression),
        write_tlm: options.progression == J2kProgressionOrder::Rpcl,
        validate_high_throughput_codestream: false,
        ..EncodeOptions::default()
    }
}

fn native_progression(progression: J2kProgressionOrder) -> EncodeProgressionOrder {
    match progression {
        J2kProgressionOrder::Lrcp => EncodeProgressionOrder::Lrcp,
        J2kProgressionOrder::Rpcl => EncodeProgressionOrder::Rpcl,
    }
}

fn validate_recode_roundtrip(source: &[u8], encoded: &[u8]) -> Result<(), J2kError> {
    let source = Image::new(source, &DecodeSettings::default())
        .map_err(|err| map_native_decode_error(err, "source JPEG 2000 validation parse failed"))?
        .decode_native()
        .map_err(|err| map_native_decode_error(err, "source JPEG 2000 validation decode failed"))?;
    let encoded = Image::new(encoded, &DecodeSettings::default())
        .map_err(|err| map_native_decode_error(err, "HTJ2K validation parse failed"))?
        .decode_native()
        .map_err(|err| map_native_decode_error(err, "HTJ2K validation decode failed"))?;

    if source.width != encoded.width
        || source.height != encoded.height
        || source.bit_depth != encoded.bit_depth
        || source.num_components != encoded.num_components
        || source.data != encoded.data
    {
        return Err(J2kError::Backend(
            "HTJ2K coefficient recode failed pixel validation".to_string(),
        ));
    }
    Ok(())
}

fn map_native_decode_error(
    err: signinum_j2k_native::DecodeError,
    context: &'static str,
) -> J2kError {
    match err {
        signinum_j2k_native::DecodeError::Decoding(
            signinum_j2k_native::DecodingError::UnsupportedFeature(what),
        ) => J2kError::Unsupported(Unsupported { what }),
        _ => J2kError::Backend(format!("{context}: {err}")),
    }
}
