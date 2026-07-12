// SPDX-License-Identifier: MIT OR Apache-2.0

//! Coefficient-preserving classic JPEG 2000 to HTJ2K route.

use super::{
    map_native_decode_error, native_decode_error_is_unsupported, output, pixel, recode_report,
    J2kToHtj2kMode, J2kToHtj2kOptions, ReencodedHtj2k,
};
use crate::{
    encode::{native_progression_order, J2kProgressionOrder},
    parse::ParsedImageInfo,
    J2kError,
};
use j2k_core::{Colorspace, CompressedTransferSyntax};
use j2k_native::{DecodeSettings, EncodeOptions, Image};

pub(super) fn supports(parsed: &ParsedImageInfo) -> bool {
    if parsed.transfer_syntax != CompressedTransferSyntax::Jpeg2000Lossless {
        return false;
    }
    if parsed.file_metadata.as_ref().is_some_and(|metadata| {
        metadata.palette.is_some() || !metadata.component_mappings.is_empty()
    }) {
        return false;
    }
    if !matches!(parsed.info.components, 1 | 3) || !matches!(parsed.info.bit_depth, 8 | 16) {
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
    !parsed
        .components
        .iter()
        .any(|component| component.signed || component.bit_depth != parsed.info.bit_depth)
}

pub(super) fn recode(
    bytes: &[u8],
    parsed: &ParsedImageInfo,
    options: J2kToHtj2kOptions,
    output_transfer_syntax: CompressedTransferSyntax,
) -> Result<ReencodedHtj2k, J2kError> {
    let source = Image::new(bytes, &DecodeSettings::default())
        .map_err(|err| map_native_decode_error(err, "source JPEG 2000 parse failed"))?;
    let coefficients = match source.decode_reversible_53_coefficients() {
        Ok(coefficients) => coefficients,
        Err(err) if native_decode_error_is_unsupported(&err) => {
            return pixel::recode(bytes, parsed, options, output_transfer_syntax);
        }
        Err(err) => {
            return Err(map_native_decode_error(
                err,
                "source coefficient extraction failed",
            ));
        }
    };
    drop(source);

    let encode_options = native_encode_options(options, &coefficients);
    let codestream = coefficients
        .encode_htj2k(&encode_options)
        .map_err(map_coefficient_encode_error)?;
    drop(coefficients);

    let output = output::finalize_owned(codestream, options.output_payload_kind, parsed, true)?;
    Ok(ReencodedHtj2k {
        bytes: output,
        report: recode_report(
            parsed,
            J2kToHtj2kMode::CoefficientPreserving,
            output_transfer_syntax,
            options.output_payload_kind,
            parsed.info.components,
            parsed.info.bit_depth,
        ),
    })
}

fn map_coefficient_encode_error(source: j2k_native::EncodeError) -> J2kError {
    J2kError::from_native_encode_error_with_context(
        source,
        "native HTJ2K coefficient recode failed",
    )
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

#[cfg(test)]
mod tests;
