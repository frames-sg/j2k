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
    Colorspace, CompressedPayloadKind, CompressedTransferSyntax, PassthroughRequirements,
    Unsupported,
};
use j2k_native::{DecodeSettings, EncodeOptions, Image};

use crate::{
    encode::{
        encode_j2k_lossless, encode_j2k_lossless_typed_components, native_progression_order,
        J2kBlockCodingMode, J2kEncodeValidation, J2kLosslessEncodeOptions, J2kLosslessSamples,
        J2kLosslessTypedComponentPlane, J2kLosslessTypedComponentSamples, J2kProgressionOrder,
        ReversibleTransform,
    },
    parse::{parse_image_info, ParsedImageInfo},
    wrap::{wrap_j2k_codestream, J2kFileBoxMetadata, J2kFileColorSpec, J2kFileWrapOptions},
    J2kError, J2kView,
};

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
#[derive(Debug, Clone, PartialEq, Eq)]
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

    validate_output_payload_kind(options.output_payload_kind)?;

    if parsed.transfer_syntax == output_transfer_syntax
        && parsed.payload_kind == CompressedPayloadKind::Jpeg2000Codestream
        && options.output_payload_kind == CompressedPayloadKind::JphFile
    {
        let output =
            finalize_recode_output(bytes.to_vec(), options.output_payload_kind, &parsed, true)?;
        if options.validation == J2kEncodeValidation::CpuRoundTrip {
            validate_recode_roundtrip(bytes, &output, "HTJ2K codestream-preserving wrap")?;
        }
        return Ok(ReencodedHtj2k {
            bytes: output,
            report: J2kToHtj2kReport {
                mode: J2kToHtj2kMode::CodestreamPreserving,
                input_transfer_syntax: parsed.transfer_syntax,
                output_transfer_syntax,
                input_payload_kind: parsed.payload_kind,
                output_payload_kind: options.output_payload_kind,
                width: parsed.info.dimensions.0,
                height: parsed.info.dimensions.1,
                components: parsed.info.components,
                bit_depth: parsed.info.bit_depth,
            },
        });
    }

    if !supports_coefficient_domain_recode(&parsed) {
        return pixel_preserving_recode(bytes, &parsed, options, output_transfer_syntax);
    }

    let source = Image::new(bytes, &DecodeSettings::default())
        .map_err(|err| map_native_decode_error(err, "source JPEG 2000 parse failed"))?;
    let coefficients = match source.decode_reversible_53_coefficients() {
        Ok(coefficients) => coefficients,
        Err(err) if native_decode_error_is_unsupported(&err) => {
            return pixel_preserving_recode(bytes, &parsed, options, output_transfer_syntax);
        }
        Err(err) => {
            return Err(map_native_decode_error(
                err,
                "source coefficient extraction failed",
            ));
        }
    };

    let encode_options = native_encode_options(options, &coefficients);
    let codestream = j2k_native::encode_precomputed_htj2k_53_with_mct(
        &coefficients.image,
        &encode_options,
        coefficients.use_mct,
    )
    .map_err(|err| J2kError::Backend(format!("HTJ2K coefficient recode failed: {err}")))?;

    let output = finalize_recode_output(codestream, options.output_payload_kind, &parsed, true)?;

    if options.validation == J2kEncodeValidation::CpuRoundTrip {
        validate_recode_roundtrip(bytes, &output, "HTJ2K coefficient recode")?;
    }

    Ok(ReencodedHtj2k {
        bytes: output,
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

fn supports_coefficient_domain_recode(parsed: &ParsedImageInfo) -> bool {
    if parsed.transfer_syntax != CompressedTransferSyntax::Jpeg2000Lossless {
        return false;
    }
    if parsed.file_metadata.as_ref().is_some_and(|metadata| {
        metadata.palette.is_some() || !metadata.component_mappings.is_empty()
    }) {
        return false;
    }
    if !matches!(parsed.info.components, 1 | 3) {
        return false;
    }
    if !matches!(parsed.info.bit_depth, 8 | 16) {
        return false;
    }
    if !matches!(
        parsed.info.colorspace,
        Colorspace::Grayscale
            | Colorspace::SGray
            | Colorspace::Rgb
            | Colorspace::SRgb
            | Colorspace::Rct
    ) {
        return false;
    }
    if parsed.components.iter().any(|component| component.signed) {
        return false;
    }
    if parsed
        .components
        .iter()
        .any(|component| component.bit_depth != parsed.info.bit_depth)
    {
        return false;
    }
    true
}

fn pixel_preserving_recode(
    bytes: &[u8],
    parsed: &ParsedImageInfo,
    options: J2kToHtj2kOptions,
    output_transfer_syntax: CompressedTransferSyntax,
) -> Result<ReencodedHtj2k, J2kError> {
    let uses_resolved_pixels = pixel_fallback_uses_resolved_pixels(parsed);
    if uses_resolved_pixels {
        return pixel_preserving_recode_packed(bytes, parsed, options, output_transfer_syntax);
    }
    pixel_preserving_recode_components(bytes, parsed, options, output_transfer_syntax)
}

fn pixel_fallback_uses_resolved_pixels(parsed: &ParsedImageInfo) -> bool {
    parsed.file_metadata.as_ref().is_some_and(|metadata| {
        metadata.palette.is_some() || !metadata.component_mappings.is_empty()
    })
}

fn pixel_preserving_recode_packed(
    bytes: &[u8],
    parsed: &ParsedImageInfo,
    options: J2kToHtj2kOptions,
    output_transfer_syntax: CompressedTransferSyntax,
) -> Result<ReencodedHtj2k, J2kError> {
    let source = Image::new(bytes, &DecodeSettings::default())
        .map_err(|err| map_native_decode_error(err, "source JPEG 2000 parse failed"))?;
    let decoded = source
        .decode_native()
        .map_err(|err| map_native_decode_error(err, "source JPEG 2000 pixel fallback failed"))?;
    let signed = decoded.component_signed.iter().all(|signed| *signed);
    let samples = J2kLosslessSamples::new(
        &decoded.data,
        decoded.width,
        decoded.height,
        decoded.num_components,
        decoded.bit_depth,
        signed,
    )?;
    let encode_options = J2kLosslessEncodeOptions {
        block_coding_mode: J2kBlockCodingMode::HighThroughput,
        progression: options.progression,
        max_decomposition_levels: high_bit_recode_decomposition_limit(parsed),
        reversible_transform: ReversibleTransform::None53,
        validation: J2kEncodeValidation::External,
        ..J2kLosslessEncodeOptions::default()
    };
    let encoded = encode_j2k_lossless(samples, &encode_options)?;
    finish_pixel_preserving_recode(encoded, bytes, parsed, options, output_transfer_syntax)
}

fn pixel_preserving_recode_components(
    bytes: &[u8],
    parsed: &ParsedImageInfo,
    options: J2kToHtj2kOptions,
    output_transfer_syntax: CompressedTransferSyntax,
) -> Result<ReencodedHtj2k, J2kError> {
    let source = Image::new(bytes, &DecodeSettings::default())
        .map_err(|err| map_native_decode_error(err, "source JPEG 2000 parse failed"))?;
    let components = source.decode_native_components().map_err(|err| {
        map_native_decode_error(err, "source JPEG 2000 component pixel fallback failed")
    })?;
    let encode_options = J2kLosslessEncodeOptions {
        block_coding_mode: J2kBlockCodingMode::HighThroughput,
        progression: options.progression,
        max_decomposition_levels: high_bit_recode_decomposition_limit(parsed),
        reversible_transform: ReversibleTransform::None53,
        validation: J2kEncodeValidation::External,
        ..J2kLosslessEncodeOptions::default()
    };
    let component_grid_data = components
        .planes()
        .iter()
        .enumerate()
        .map(|(index, plane)| {
            component_grid_plane_data(
                plane.data(),
                plane.dimensions(),
                components.dimensions(),
                plane.sampling(),
                plane.bit_depth(),
                index,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let planes = components
        .planes()
        .iter()
        .enumerate()
        .map(|(index, plane)| J2kLosslessTypedComponentPlane {
            data: component_grid_data[index]
                .as_deref()
                .unwrap_or_else(|| plane.data()),
            x_rsiz: plane.sampling().0,
            y_rsiz: plane.sampling().1,
            bit_depth: plane.bit_depth(),
            signed: plane.signed(),
        })
        .collect::<Vec<_>>();
    let samples = J2kLosslessTypedComponentSamples::new(
        &planes,
        components.dimensions().0,
        components.dimensions().1,
    )?;
    let encoded = encode_j2k_lossless_typed_components(samples, &encode_options)?;
    finish_pixel_preserving_recode(encoded, bytes, parsed, options, output_transfer_syntax)
}

fn component_grid_plane_data(
    data: &[u8],
    plane_dimensions: (u32, u32),
    reference_dimensions: (u32, u32),
    sampling: (u8, u8),
    bit_depth: u8,
    plane_index: usize,
) -> Result<Option<Vec<u8>>, J2kError> {
    let (x_rsiz, y_rsiz) = sampling;
    if x_rsiz == 0 || y_rsiz == 0 {
        return Err(J2kError::InvalidSamples {
            what: format!("component plane {plane_index} sampling factors must be non-zero"),
        });
    }
    let bytes_per_sample = recode_bytes_per_sample(bit_depth)?;
    let component_width = reference_dimensions.0.div_ceil(u32::from(x_rsiz));
    let component_height = reference_dimensions.1.div_ceil(u32::from(y_rsiz));
    let expected_len = checked_plane_bytes(
        component_width,
        component_height,
        bytes_per_sample,
        plane_index,
    )?;
    if data.len() == expected_len {
        return Ok(None);
    }

    let expanded_len = checked_plane_bytes(
        reference_dimensions.0,
        reference_dimensions.1,
        bytes_per_sample,
        plane_index,
    )?;
    if plane_dimensions != reference_dimensions || data.len() != expanded_len {
        return Err(J2kError::InvalidSamples {
            what: format!(
                "component plane {plane_index} data length mismatch: expected {expected_len} component-grid bytes or {expanded_len} expanded bytes, got {}",
                data.len()
            ),
        });
    }

    let mut compacted = Vec::with_capacity(expected_len);
    let source_width = reference_dimensions.0 as usize;
    for component_y in 0..component_height as usize {
        let source_y =
            component_y
                .checked_mul(usize::from(y_rsiz))
                .ok_or(J2kError::DimensionOverflow {
                    width: reference_dimensions.0,
                    height: reference_dimensions.1,
                })?;
        for component_x in 0..component_width as usize {
            let source_x = component_x.checked_mul(usize::from(x_rsiz)).ok_or(
                J2kError::DimensionOverflow {
                    width: reference_dimensions.0,
                    height: reference_dimensions.1,
                },
            )?;
            let start = source_y
                .checked_mul(source_width)
                .and_then(|row| row.checked_add(source_x))
                .and_then(|sample| sample.checked_mul(bytes_per_sample))
                .ok_or(J2kError::DimensionOverflow {
                    width: reference_dimensions.0,
                    height: reference_dimensions.1,
                })?;
            compacted.extend_from_slice(&data[start..start + bytes_per_sample]);
        }
    }
    Ok(Some(compacted))
}

fn checked_plane_bytes(
    width: u32,
    height: u32,
    bytes_per_sample: usize,
    plane_index: usize,
) -> Result<usize, J2kError> {
    (width as usize)
        .checked_mul(height as usize)
        .and_then(|samples| samples.checked_mul(bytes_per_sample))
        .ok_or_else(|| J2kError::InvalidSamples {
            what: format!("component plane {plane_index} dimensions overflow"),
        })
}

fn recode_bytes_per_sample(bit_depth: u8) -> Result<usize, J2kError> {
    match bit_depth {
        1..=8 => Ok(1),
        9..=16 => Ok(2),
        17..=24 => Ok(3),
        25..=32 => Ok(4),
        33..=38 => Ok(5),
        _ => Err(J2kError::Unsupported(Unsupported {
            what: "JPEG 2000 component planes support 1-38 bits per sample",
        })),
    }
}

fn finish_pixel_preserving_recode(
    encoded: crate::encode::EncodedJ2k,
    source: &[u8],
    parsed: &ParsedImageInfo,
    options: J2kToHtj2kOptions,
    output_transfer_syntax: CompressedTransferSyntax,
) -> Result<ReencodedHtj2k, J2kError> {
    let output = finalize_recode_output(
        encoded.codestream,
        options.output_payload_kind,
        parsed,
        false,
    )?;

    if options.validation == J2kEncodeValidation::CpuRoundTrip {
        validate_recode_roundtrip(source, &output, "HTJ2K pixel-preserving recode")?;
    }

    Ok(ReencodedHtj2k {
        bytes: output,
        report: J2kToHtj2kReport {
            mode: J2kToHtj2kMode::PixelPreserving,
            input_transfer_syntax: parsed.transfer_syntax,
            output_transfer_syntax,
            input_payload_kind: parsed.payload_kind,
            output_payload_kind: options.output_payload_kind,
            width: parsed.info.dimensions.0,
            height: parsed.info.dimensions.1,
            components: encoded.components,
            bit_depth: encoded.bit_depth,
        },
    })
}

fn high_bit_recode_decomposition_limit(parsed: &ParsedImageInfo) -> Option<u8> {
    parsed
        .components
        .iter()
        .any(|component| component.bit_depth > 24)
        .then_some(0)
}

fn finalize_recode_output(
    codestream: Vec<u8>,
    payload_kind: CompressedPayloadKind,
    parsed: &ParsedImageInfo,
    preserve_file_metadata: bool,
) -> Result<Vec<u8>, J2kError> {
    match payload_kind {
        CompressedPayloadKind::Jpeg2000Codestream => Ok(codestream),
        CompressedPayloadKind::JphFile => {
            wrap_recode_jph(&codestream, parsed, preserve_file_metadata)
        }
        _ => Err(Unsupported {
            what: "J2K to HTJ2K recode output must be a raw HTJ2K codestream or JPH file",
        }
        .into()),
    }
}

fn wrap_recode_jph(
    codestream: &[u8],
    parsed: &ParsedImageInfo,
    preserve_file_metadata: bool,
) -> Result<Vec<u8>, J2kError> {
    let Some(metadata) = parsed.file_metadata.as_ref() else {
        return wrap_j2k_codestream(codestream, J2kFileWrapOptions::jph());
    };
    if !preserve_file_metadata
        && (metadata.palette.is_some() || !metadata.component_mappings.is_empty())
    {
        return wrap_j2k_codestream(codestream, J2kFileWrapOptions::jph());
    }
    let color_specs = metadata
        .color_specs
        .iter()
        .filter_map(J2kFileColorSpec::from_inspected)
        .collect::<Vec<_>>();
    let mut options = if color_specs.is_empty() {
        J2kFileColorSpec::from_file_metadata(metadata)
            .map_or_else(J2kFileWrapOptions::jph, |color| {
                J2kFileWrapOptions::jph().with_color(color)
            })
    } else {
        J2kFileWrapOptions::jph().with_color_specs(&color_specs)
    };
    if preserve_file_metadata {
        options = options.with_metadata(J2kFileBoxMetadata::from_file_metadata(metadata));
    }
    wrap_j2k_codestream(codestream, options)
}

fn native_encode_options(
    options: J2kToHtj2kOptions,
    coefficients: &j2k_native::Reversible53CoefficientImage,
) -> EncodeOptions {
    EncodeOptions {
        reversible: true,
        use_ht_block_coding: true,
        use_mct: coefficients.use_mct,
        code_block_width_exp: coefficients.code_block_width_exp,
        code_block_height_exp: coefficients.code_block_height_exp,
        guard_bits: coefficients.guard_bits,
        progression_order: native_progression_order(options.progression),
        write_tlm: options.progression == J2kProgressionOrder::Rpcl,
        validate_high_throughput_codestream: false,
        ..EncodeOptions::default()
    }
}

fn validate_recode_roundtrip(
    source: &[u8],
    encoded: &[u8],
    context: &'static str,
) -> Result<(), J2kError> {
    let source_image = Image::new(source, &DecodeSettings::default())
        .map_err(|err| map_native_decode_error(err, "source JPEG 2000 validation parse failed"))?;
    let encoded_image = Image::new(encoded, &DecodeSettings::default())
        .map_err(|err| map_native_decode_error(err, "HTJ2K validation parse failed"))?;

    if let (Ok(source), Ok(encoded)) = (source_image.decode_native(), encoded_image.decode_native())
    {
        if source.width != encoded.width
            || source.height != encoded.height
            || source.bit_depth != encoded.bit_depth
            || source.num_components != encoded.num_components
            || source.data != encoded.data
        {
            return Err(J2kError::Backend(format!(
                "{context} failed pixel validation"
            )));
        }
        return Ok(());
    }

    let source = source_image.decode_native_components().map_err(|err| {
        map_native_decode_error(err, "source JPEG 2000 component validation decode failed")
    })?;
    let encoded = encoded_image
        .decode_native_components()
        .map_err(|err| map_native_decode_error(err, "HTJ2K component validation decode failed"))?;
    if source.dimensions() != encoded.dimensions()
        || source.planes().len() != encoded.planes().len()
    {
        return Err(J2kError::Backend(format!(
            "{context} failed component validation"
        )));
    }
    for (source_plane, encoded_plane) in source.planes().iter().zip(encoded.planes()) {
        if source_plane.dimensions() != encoded_plane.dimensions()
            || source_plane.sampling() != encoded_plane.sampling()
            || source_plane.bit_depth() != encoded_plane.bit_depth()
            || source_plane.signed() != encoded_plane.signed()
            || source_plane.data() != encoded_plane.data()
        {
            return Err(J2kError::Backend(format!(
                "{context} failed component validation"
            )));
        }
    }
    Ok(())
}

fn map_native_decode_error(err: j2k_native::DecodeError, context: &'static str) -> J2kError {
    match err {
        j2k_native::DecodeError::Decoding(j2k_native::DecodingError::UnsupportedFeature(what)) => {
            J2kError::Unsupported(Unsupported { what })
        }
        _ => J2kError::Backend(format!("{context}: {err}")),
    }
}

fn native_decode_error_is_unsupported(err: &j2k_native::DecodeError) -> bool {
    matches!(
        err,
        j2k_native::DecodeError::Decoding(j2k_native::DecodingError::UnsupportedFeature(_))
    )
}
