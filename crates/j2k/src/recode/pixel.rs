// SPDX-License-Identifier: MIT OR Apache-2.0

//! Pixel-preserving HTJ2K fallback routes and temporary owner accounting.

use super::{
    allocation, component_grid, map_native_decode_error, output, recode_report, J2kToHtj2kMode,
    J2kToHtj2kOptions, ReencodedHtj2k,
};
use crate::{
    encode::{
        encode_j2k_lossless_typed_components, J2kBlockCodingMode, J2kEncodeValidation,
        J2kLosslessEncodeOptions, J2kLosslessTypedComponentPlane, J2kLosslessTypedComponentSamples,
        ReversibleTransform,
    },
    parse::ParsedImageInfo,
    J2kError,
};
use j2k_core::{BufferError, CompressedTransferSyntax};
use j2k_native::{DecodeSettings, Image};

pub(super) fn recode(
    bytes: &[u8],
    parsed: &ParsedImageInfo,
    options: J2kToHtj2kOptions,
    output_transfer_syntax: CompressedTransferSyntax,
) -> Result<ReencodedHtj2k, J2kError> {
    recode_components(bytes, parsed, options, output_transfer_syntax)
}

fn recode_components(
    bytes: &[u8],
    parsed: &ParsedImageInfo,
    options: J2kToHtj2kOptions,
    output_transfer_syntax: CompressedTransferSyntax,
) -> Result<ReencodedHtj2k, J2kError> {
    let encode_options = encode_options(parsed, options);
    let resolve_to_reference_grid = parsed.file_metadata.as_ref().is_some_and(|metadata| {
        metadata.palette.is_some() || !metadata.component_mappings.is_empty()
    });
    let encoded = {
        let components = {
            let source = Image::new(bytes, &DecodeSettings::default())
                .map_err(|err| map_native_decode_error(err, "source JPEG 2000 parse failed"))?;
            source.decode_native_components().map_err(|err| {
                map_native_decode_error(err, "source JPEG 2000 component pixel fallback failed")
            })?
        };
        let component_bytes =
            components
                .allocated_bytes()
                .ok_or(J2kError::Buffer(BufferError::SizeOverflow {
                    what: "HTJ2K component fallback decoded owners",
                }))?;
        let retained_bytes = allocation::checked_add_owned_bytes(
            parsed.allocated_bytes()?,
            component_bytes,
            "HTJ2K component fallback retained owners",
        )?;
        let mut budget = allocation::RecodeAllocationBudget::from_live_bytes(retained_bytes)?;
        let mut component_grid_data = budget.try_vec(
            components.planes().len(),
            "HTJ2K component-grid owner metadata",
        )?;
        for (index, plane) in components.planes().iter().enumerate() {
            let data = if resolve_to_reference_grid {
                component_grid::resolved_plane_data(
                    plane.data(),
                    plane.dimensions(),
                    components.dimensions(),
                    plane.sampling(),
                    plane.bit_depth(),
                    index,
                    &mut budget,
                )?
            } else {
                component_grid::plane_data(
                    plane.data(),
                    plane.dimensions(),
                    components.dimensions(),
                    plane.sampling(),
                    plane.bit_depth(),
                    index,
                    &mut budget,
                )?
            };
            component_grid_data.push(data);
        }
        let mut planes = budget.try_vec(
            components.planes().len(),
            "HTJ2K typed component descriptors",
        )?;
        for (index, plane) in components.planes().iter().enumerate() {
            let sampling = if resolve_to_reference_grid {
                (1, 1)
            } else {
                plane.sampling()
            };
            planes.push(J2kLosslessTypedComponentPlane {
                data: component_grid_data[index]
                    .as_deref()
                    .unwrap_or_else(|| plane.data()),
                x_rsiz: sampling.0,
                y_rsiz: sampling.1,
                bit_depth: plane.bit_depth(),
                signed: plane.signed(),
            });
        }
        let samples = J2kLosslessTypedComponentSamples::new(
            &planes,
            components.dimensions().0,
            components.dimensions().1,
        )?;
        let encoded = encode_j2k_lossless_typed_components(samples, &encode_options)?;
        budget.include_bytes(
            encoded.codestream.capacity(),
            "HTJ2K component fallback encoded output",
        )?;
        encoded
    };
    finish(encoded, parsed, options, output_transfer_syntax)
}

fn finish(
    encoded: crate::encode::EncodedJ2k,
    parsed: &ParsedImageInfo,
    options: J2kToHtj2kOptions,
    output_transfer_syntax: CompressedTransferSyntax,
) -> Result<ReencodedHtj2k, J2kError> {
    let components = encoded.components;
    let bit_depth = encoded.bit_depth;
    let output = output::finalize_owned(
        encoded.codestream,
        options.output_payload_kind,
        parsed,
        false,
    )?;
    Ok(ReencodedHtj2k {
        bytes: output,
        report: recode_report(
            parsed,
            J2kToHtj2kMode::PixelPreserving,
            output_transfer_syntax,
            options.output_payload_kind,
            components,
            bit_depth,
        ),
    })
}

fn encode_options(
    parsed: &ParsedImageInfo,
    options: J2kToHtj2kOptions,
) -> J2kLosslessEncodeOptions {
    J2kLosslessEncodeOptions {
        block_coding_mode: J2kBlockCodingMode::HighThroughput,
        progression: options.progression,
        max_decomposition_levels: parsed
            .components
            .iter()
            .any(|component| component.bit_depth > 24)
            .then_some(0),
        reversible_transform: ReversibleTransform::None53,
        validation: J2kEncodeValidation::External,
        ..J2kLosslessEncodeOptions::default()
    }
}
