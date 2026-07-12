// SPDX-License-Identifier: MIT OR Apache-2.0

//! JPEG 2000-family coefficient-domain recoding APIs.
//!
//! The direct 5/3 path decodes classic JPEG 2000 Tier-1 code-blocks into
//! reversible wavelet coefficients and re-encodes those coefficients with
//! HTJ2K block coding. It does not run inverse DWT, forward DWT, or a
//! pixel-domain lossless encode. The output is coefficient-preserving for the
//! supported reversible 5/3 profile, not byte-preserving unless passthrough is
//! reported.

use alloc::vec::Vec;

use j2k_core::{
    CompressedPayloadKind, CompressedTransferSyntax, PassthroughCandidate, PassthroughRequirements,
    Unsupported,
};

use crate::{
    encode::{J2kEncodeValidation, J2kProgressionOrder},
    parse::{parse_image_info, ParsedImageInfo},
    J2kError,
};

mod allocation;
mod coefficient;
mod component_grid;
mod output;
mod pixel;
mod validation;

/// Options for classic JPEG 2000 reversible 5/3 to HTJ2K lossless recoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct J2kToHtj2kOptions {
    /// Requested output payload shape.
    ///
    /// DICOM encapsulated WSI frames use raw JPEG 2000-family codestreams, so
    /// the default is [`CompressedPayloadKind::Jpeg2000Codestream`]. Use
    /// [`CompressedPayloadKind::JphFile`] when an HTJ2K still-image file wrapper
    /// is required.
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

impl J2kToHtj2kOptions {
    /// Create J2K/JP2 to HTJ2K recode options.
    pub const fn new(
        output_payload_kind: CompressedPayloadKind,
        progression: J2kProgressionOrder,
        validation: J2kEncodeValidation,
    ) -> Self {
        Self {
            output_payload_kind,
            progression,
            validation,
        }
    }
}

/// Recode path used for a J2K/JP2 to HTJ2K request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum J2kToHtj2kMode {
    /// Input bytes already matched the requested HTJ2K transfer syntax and
    /// payload kind, so bytes were copied unchanged.
    Passthrough,
    /// The source HTJ2K codestream already matched the requested transfer
    /// syntax and was copied unchanged into a new file wrapper.
    CodestreamPreserving,
    /// Classic reversible 5/3 code-blocks were entropy-decoded to quantized
    /// wavelet coefficients and re-encoded with HT block coding.
    CoefficientPreserving,
    /// Pixels were decoded at native bit depth and re-encoded losslessly with
    /// HT block coding because the coefficient-domain path could not represent
    /// the source profile.
    PixelPreserving,
}

/// Metadata describing a J2K/JP2 to HTJ2K recode.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct J2kToHtj2kReport {
    /// Recode path used for this output.
    pub mode: J2kToHtj2kMode,
    /// Classified input transfer syntax.
    pub input_transfer_syntax: CompressedTransferSyntax,
    /// Output transfer syntax.
    pub output_transfer_syntax: CompressedTransferSyntax,
    /// Classified input payload/container kind.
    pub input_payload_kind: CompressedPayloadKind,
    /// Output payload/container kind.
    pub output_payload_kind: CompressedPayloadKind,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Component count.
    pub components: u16,
    /// Significant bits per component.
    pub bit_depth: u8,
}

/// HTJ2K codestream bytes and recode metadata.
#[derive(Debug, PartialEq, Eq)]
pub struct ReencodedHtj2k {
    /// Encoded HTJ2K bytes.
    pub bytes: Vec<u8>,
    /// Recode metadata and selected path.
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
    let parsed = parse_image_info(bytes)?;
    let output_transfer_syntax = CompressedTransferSyntax::HtJpeg2000Lossless;

    let candidate = PassthroughCandidate::new(
        bytes,
        parsed.transfer_syntax,
        parsed.payload_kind,
        parsed.info.clone(),
    );
    let requirements =
        PassthroughRequirements::new(output_transfer_syntax, options.output_payload_kind);
    if let Ok(copy) = candidate.copy_bytes_if_eligible(&requirements) {
        let output =
            allocation::copy_bytes(copy, parsed.allocated_bytes()?, "HTJ2K passthrough output")?;
        return Ok(ReencodedHtj2k {
            bytes: output,
            report: recode_report(
                &parsed,
                J2kToHtj2kMode::Passthrough,
                output_transfer_syntax,
                options.output_payload_kind,
                parsed.info.components,
                parsed.info.bit_depth,
            ),
        });
    }

    validate_output_payload_kind(options.output_payload_kind)?;

    let validation_mode = options.validation;
    let result = if parsed.transfer_syntax == output_transfer_syntax
        && parsed.payload_kind == CompressedPayloadKind::Jpeg2000Codestream
        && options.output_payload_kind == CompressedPayloadKind::JphFile
    {
        let output = output::wrap_borrowed_jph(bytes, &parsed, true)?;
        ReencodedHtj2k {
            bytes: output,
            report: recode_report(
                &parsed,
                J2kToHtj2kMode::CodestreamPreserving,
                output_transfer_syntax,
                options.output_payload_kind,
                parsed.info.components,
                parsed.info.bit_depth,
            ),
        }
    } else if !coefficient::supports(&parsed) {
        pixel::recode(bytes, &parsed, options, output_transfer_syntax)?
    } else {
        coefficient::recode(bytes, &parsed, options, output_transfer_syntax)?
    };

    drop(parsed);
    if validation_mode == J2kEncodeValidation::CpuRoundTrip {
        validation::roundtrip(
            bytes,
            &result.bytes,
            recode_validation_context(result.report.mode),
        )?;
    }
    Ok(result)
}

const fn recode_validation_context(mode: J2kToHtj2kMode) -> &'static str {
    match mode {
        J2kToHtj2kMode::Passthrough => "HTJ2K passthrough",
        J2kToHtj2kMode::CodestreamPreserving => "HTJ2K codestream-preserving wrap",
        J2kToHtj2kMode::CoefficientPreserving => "HTJ2K coefficient recode",
        J2kToHtj2kMode::PixelPreserving => "HTJ2K pixel-preserving recode",
    }
}

fn validate_output_payload_kind(payload_kind: CompressedPayloadKind) -> Result<(), J2kError> {
    match payload_kind {
        CompressedPayloadKind::Jpeg2000Codestream | CompressedPayloadKind::JphFile => Ok(()),
        CompressedPayloadKind::Jp2File => Err(Unsupported {
            what: "HTJ2K file output uses JPH, not JP2",
        }
        .into()),
        _ => Err(Unsupported {
            what: "J2K to HTJ2K recode output must be a raw HTJ2K codestream or JPH file",
        }
        .into()),
    }
}

fn recode_report(
    parsed: &ParsedImageInfo,
    mode: J2kToHtj2kMode,
    output_transfer_syntax: CompressedTransferSyntax,
    output_payload_kind: CompressedPayloadKind,
    components: u16,
    bit_depth: u8,
) -> J2kToHtj2kReport {
    J2kToHtj2kReport {
        mode,
        input_transfer_syntax: parsed.transfer_syntax,
        output_transfer_syntax,
        input_payload_kind: parsed.payload_kind,
        output_payload_kind,
        width: parsed.info.dimensions.0,
        height: parsed.info.dimensions.1,
        components,
        bit_depth,
    }
}

fn map_native_decode_error(err: j2k_native::DecodeError, context: &'static str) -> J2kError {
    match err {
        j2k_native::DecodeError::Decoding(j2k_native::DecodingError::UnsupportedFeature(what)) => {
            J2kError::Unsupported(Unsupported { what })
        }
        _ => J2kError::from_native_decode_error_with_context(err, context),
    }
}

fn native_decode_error_is_unsupported(err: &j2k_native::DecodeError) -> bool {
    matches!(
        err,
        j2k_native::DecodeError::Decoding(j2k_native::DecodingError::UnsupportedFeature(_))
    )
}
