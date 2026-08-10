// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_compare::openjpeg::OpenJpegDecodedImage;

use j2k_native::{inspect_htj2k_capabilities, Htj2kCapabilityMode};

use crate::{EncoderBlockCoding, EncoderMarker, EncoderMode};

use super::super::input::GeneratedInput;
use crate::encoder::EncoderCase;

pub(super) fn validate_metadata(
    case: &EncoderCase,
    expected: &GeneratedInput,
    actual: &OpenJpegDecodedImage,
    reference_name: &str,
) -> Result<(), String> {
    if actual.dimensions != (case.width, case.height) {
        return Err(format!(
            "{reference_name} dimensions are {:?}, expected {}x{}",
            actual.dimensions, case.width, case.height,
        ));
    }
    if actual.components.len() != expected.components.len() {
        return Err(format!(
            "{reference_name} returned {} components, expected {}",
            actual.components.len(),
            expected.components.len()
        ));
    }
    for (index, (expected, actual)) in expected
        .components
        .iter()
        .zip(&actual.components)
        .enumerate()
    {
        let expected_dimensions = (expected.dimensions[0], expected.dimensions[1]);
        let expected_sampling = (
            u32::from(expected.sampling[0]),
            u32::from(expected.sampling[1]),
        );
        if actual.dimensions != expected_dimensions
            || actual.sampling != expected_sampling
            || actual.bit_depth != expected.bit_depth
            || actual.signed != expected.signed
            || actual.samples.len() != expected.samples.len()
        {
            return Err(format!(
                "{reference_name} component {index} metadata differs from the encoder input"
            ));
        }
    }
    Ok(())
}

pub(super) fn validate_markers(case: &EncoderCase, codestream: &[u8]) -> Result<(), String> {
    for marker in [
        EncoderMarker::Soc,
        EncoderMarker::Siz,
        EncoderMarker::Cod,
        EncoderMarker::Qcd,
        EncoderMarker::Sot,
        EncoderMarker::Sod,
        EncoderMarker::Eoc,
    ]
    .into_iter()
    .chain(case.markers.iter().copied())
    {
        if !contains_marker(codestream, marker) {
            return Err(format!(
                "encoded codestream is missing requested {marker:?} marker"
            ));
        }
    }
    let capabilities = inspect_htj2k_capabilities(codestream)
        .map_err(|error| format!("inspect encoded CAP/CPF semantics: {error}"))?;
    match (case.block_coding, capabilities) {
        (EncoderBlockCoding::Classic, None) => Ok(()),
        (EncoderBlockCoding::Classic, Some(_)) => {
            Err("classic encoder case unexpectedly advertises Part 15".to_string())
        }
        (EncoderBlockCoding::HighThroughput, None) => {
            Err("HT encoder case does not advertise Pcap15".to_string())
        }
        (EncoderBlockCoding::HighThroughput, Some(capabilities)) => {
            if capabilities.mode() != Htj2kCapabilityMode::HtOnly
                || !capabilities.default_ht_block_coding()
                || capabilities.default_mixed_block_coding()
            {
                return Err(
                    "HT encoder CAP mode and default COD style are inconsistent".to_string()
                );
            }
            if capabilities.roi() != case.roi.is_some()
                || capabilities.ht_irreversible() != (case.mode == EncoderMode::Lossy)
            {
                return Err("HT encoder CAP flags do not match the encoded case".to_string());
            }
            // Source precision is not a BMAGB lower bound: irreversible
            // quantization can reduce the encoded cleanup magnitude. The
            // encoder derives BMAGB before discarding its coefficient owners.
            Ok(())
        }
    }
}

fn contains_marker(codestream: &[u8], marker: EncoderMarker) -> bool {
    let code = match marker {
        EncoderMarker::Soc => 0x4F,
        EncoderMarker::Cap => 0x50,
        EncoderMarker::Prf => 0x56,
        EncoderMarker::Cpf => 0x59,
        EncoderMarker::Sot => 0x90,
        EncoderMarker::Sod => 0x93,
        EncoderMarker::Eoc => 0xD9,
        EncoderMarker::Siz => 0x51,
        EncoderMarker::Cod => 0x52,
        EncoderMarker::Coc => 0x53,
        EncoderMarker::Rgn => 0x5E,
        EncoderMarker::Qcd => 0x5C,
        EncoderMarker::Qcc => 0x5D,
        EncoderMarker::Poc => 0x5F,
        EncoderMarker::Tlm => 0x55,
        EncoderMarker::Plm => 0x57,
        EncoderMarker::Plt => 0x58,
        EncoderMarker::Ppm => 0x60,
        EncoderMarker::Ppt => 0x61,
        EncoderMarker::Sop => 0x91,
        EncoderMarker::Eph => 0x92,
        EncoderMarker::Crg => 0x63,
        EncoderMarker::Com => 0x64,
    };
    codestream.windows(2).any(|bytes| bytes == [0xFF, code])
}
