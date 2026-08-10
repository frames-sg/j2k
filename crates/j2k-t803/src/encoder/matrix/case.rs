// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;

use super::{validation, EncoderMatrixError};
use crate::encoder::{
    EncoderBlockCoding, EncoderCase, EncoderInputKind, EncoderIut, EncoderMarker, EncoderMode,
    EncoderOperation, EncoderPayload, EncoderRateTarget, EncoderReferenceDecoder,
};

pub(super) fn validate_case(case: &EncoderCase) -> Result<(), EncoderMatrixError> {
    if case.iuts.is_empty()
        || !case.iuts.windows(2).all(|pair| pair[0] < pair[1])
        || case.width == 0
        || case.height == 0
        || !(1..=16_384).contains(&case.components)
        || !(1..=38).contains(&case.bit_depth)
        || case.decomposition_levels > 32
        || case.lossless_quality_layers == 0
        || case.markers.iter().collect::<BTreeSet<_>>().len() != case.markers.len()
    {
        return validation(format!("{} has invalid basic parameters", case.id));
    }
    if case
        .tile_size
        .is_some_and(|[width, height]| width == 0 || height == 0)
        || case.tile_part_packet_limit == Some(0)
        || case
            .precinct_exponents
            .iter()
            .any(|[width, height]| *width > 15 || *height > 15)
    {
        return validation(format!("{} has invalid tiling or precinct data", case.id));
    }
    if let Some(roi) = case.roi {
        let fits = roi.component < case.components
            && roi.width > 0
            && roi.height > 0
            && roi.shift > 0
            && roi
                .x
                .checked_add(roi.width)
                .is_some_and(|x1| x1 <= case.width)
            && roi
                .y
                .checked_add(roi.height)
                .is_some_and(|y1| y1 <= case.height);
        if !fits || !case.markers.contains(&EncoderMarker::Rgn) {
            return validation(format!("{} has invalid ROI data", case.id));
        }
    } else if case.markers.contains(&EncoderMarker::Rgn) {
        return validation(format!("{} requests RGN without an ROI", case.id));
    }
    validate_operation(case)?;
    validate_coding(case)?;
    validate_reference_decoder(case)?;
    validate_input_surface(case)?;
    validate_mode(case)
}

fn validate_reference_decoder(case: &EncoderCase) -> Result<(), EncoderMatrixError> {
    let ht_roi = case.block_coding == EncoderBlockCoding::HighThroughput && case.roi.is_some();
    match (case.reference_decoder, ht_roi) {
        (EncoderReferenceDecoder::OpenHtj2k, true) | (EncoderReferenceDecoder::OpenJpeg, false) => {
            Ok(())
        }
        (EncoderReferenceDecoder::OpenHtj2k, false) => validation(format!(
            "{} selects OpenHTJ2K without an HT+RGN capability requirement",
            case.id
        )),
        (EncoderReferenceDecoder::OpenJpeg, true) => validation(format!(
            "{} selects OpenJPEG for unsupported HT+RGN decoding",
            case.id
        )),
    }
}

fn validate_operation(case: &EncoderCase) -> Result<(), EncoderMatrixError> {
    match case.operation {
        EncoderOperation::Encode => {
            if case.source_payload.is_some() {
                return validation(format!(
                    "{} declares a source payload for an encode case",
                    case.id
                ));
            }
        }
        EncoderOperation::Recode => {
            let supported_source = matches!(
                case.source_payload,
                Some(EncoderPayload::Codestream | EncoderPayload::Jp2)
            );
            if case.iuts != [EncoderIut::Cpu]
                || case.mode != EncoderMode::Lossless
                || case.block_coding != EncoderBlockCoding::HighThroughput
                || !supported_source
                || case.input != EncoderInputKind::Interleaved
                || !matches!(case.components, 1 | 3)
                || !matches!(case.bit_depth, 8 | 16)
                || case.signed
                || case.roi.is_some()
                || case.lossless_quality_layers != 1
                || case.pairwise_scope.is_some()
            {
                return validation(format!(
                    "{} is outside the coefficient-preserving recode matrix surface",
                    case.id
                ));
            }
        }
    }
    Ok(())
}

fn validate_coding(case: &EncoderCase) -> Result<(), EncoderMatrixError> {
    match (case.block_coding, case.payload) {
        (EncoderBlockCoding::Classic, EncoderPayload::Codestream) => {
            if case
                .markers
                .iter()
                .any(|marker| matches!(marker, EncoderMarker::Cap | EncoderMarker::Cpf))
            {
                return validation(format!(
                    "{} requests Part 15 markers for classic block coding",
                    case.id
                ));
            }
        }
        (EncoderBlockCoding::Classic, EncoderPayload::Jp2 | EncoderPayload::Jph) => {
            return validation(format!(
                "{} requests an unsupported classic output wrapper",
                case.id
            ));
        }
        (EncoderBlockCoding::HighThroughput, EncoderPayload::Codestream | EncoderPayload::Jph) => {
            if !case.markers.contains(&EncoderMarker::Cap) {
                return validation(format!("{} does not require the HTJ2K CAP marker", case.id));
            }
        }
        (EncoderBlockCoding::HighThroughput, EncoderPayload::Jp2) => {
            return validation(format!(
                "{} requests a JP2 wrapper for high-throughput block coding",
                case.id
            ));
        }
    }
    Ok(())
}

fn validate_input_surface(case: &EncoderCase) -> Result<(), EncoderMatrixError> {
    match case.input {
        EncoderInputKind::Interleaved => {
            if !case.sampling.is_empty()
                || !case.component_bit_depths.is_empty()
                || !case.component_signedness.is_empty()
            {
                return validation(format!("{} mixes interleaved and planar metadata", case.id));
            }
        }
        EncoderInputKind::ComponentPlanes => {
            validate_cpu_planar_case(case)?;
            if !case.component_bit_depths.is_empty() || !case.component_signedness.is_empty() {
                return validation(format!(
                    "{} has typed metadata on homogeneous planes",
                    case.id
                ));
            }
        }
        EncoderInputKind::TypedComponentPlanes => {
            validate_cpu_planar_case(case)?;
            if case.component_bit_depths.len() != usize::from(case.components)
                || case.component_signedness.len() != usize::from(case.components)
                || case
                    .component_bit_depths
                    .iter()
                    .any(|depth| !(1..=38).contains(depth))
            {
                return validation(format!("{} has invalid typed component metadata", case.id));
            }
        }
    }
    Ok(())
}

fn validate_mode(case: &EncoderCase) -> Result<(), EncoderMatrixError> {
    match case.mode {
        EncoderMode::Lossless => {
            if case.lossy_rate_target.is_some()
                || !case.lossy_quality_layers.is_empty()
                || case.minimum_psnr_db.is_some()
                || case.maximum_rate_overshoot_percent.is_some()
            {
                return validation(format!("{} has lossy targets on a lossless case", case.id));
            }
        }
        EncoderMode::Lossy => {
            if case.input != EncoderInputKind::Interleaved || case.roi.is_some() {
                return validation(format!(
                    "{} uses an unsupported lossy matrix surface",
                    case.id
                ));
            }
            for target in case
                .lossy_rate_target
                .iter()
                .chain(case.lossy_quality_layers.iter())
            {
                validate_rate_target(*target, &case.id)?;
            }
            if case
                .minimum_psnr_db
                .is_none_or(|value| !value.is_finite() || value <= 0.0)
            {
                return validation(format!("{} has no finite minimum PSNR gate", case.id));
            }
            let has_rate_gate = case
                .lossy_rate_target
                .iter()
                .chain(case.lossy_quality_layers.last())
                .any(|target| {
                    matches!(
                        target,
                        EncoderRateTarget::BitsPerPixel(_) | EncoderRateTarget::Bytes(_)
                    )
                });
            if has_rate_gate
                != case
                    .maximum_rate_overshoot_percent
                    .is_some_and(|value| value.is_finite() && (0.0..=100.0).contains(&value))
            {
                return validation(format!("{} has an invalid rate overshoot gate", case.id));
            }
        }
    }
    Ok(())
}

fn validate_cpu_planar_case(case: &EncoderCase) -> Result<(), EncoderMatrixError> {
    if case.mode != EncoderMode::Lossless
        || case.iuts != [EncoderIut::Cpu]
        || case.sampling.len() != usize::from(case.components)
        || case
            .sampling
            .iter()
            .any(|[x_rsiz, y_rsiz]| *x_rsiz == 0 || *y_rsiz == 0)
    {
        return validation(format!("{} has invalid planar surface metadata", case.id));
    }
    Ok(())
}

fn validate_rate_target(target: EncoderRateTarget, id: &str) -> Result<(), EncoderMatrixError> {
    let valid = match target {
        EncoderRateTarget::BitsPerPixel(value) | EncoderRateTarget::PsnrDb(value) => {
            value.is_finite() && value > 0.0
        }
        EncoderRateTarget::Bytes(value) => value > 0,
    };
    if valid {
        Ok(())
    } else {
        validation(format!("{id} has an invalid lossy rate target"))
    }
}

pub(super) fn validate_boundaries(cases: &[EncoderCase]) -> Result<(), EncoderMatrixError> {
    for bit_depth in [1, 8, 12, 16, 31] {
        require(
            cases.iter().any(|case| case.bit_depth == bit_depth),
            "bit-depth boundary",
        )?;
    }
    for components in [1, 2, 3, 4, 5] {
        require(
            cases.iter().any(|case| case.components == components),
            "component-count boundary",
        )?;
    }
    for level in [0, 1, 2, 3, 5] {
        require(
            cases.iter().any(|case| case.decomposition_levels == level),
            "decomposition-level boundary",
        )?;
    }
    require(
        cases.iter().any(|case| (case.width, case.height) == (1, 1)),
        "singleton geometry",
    )?;
    require(cases.iter().any(|case| case.tile_size.is_some()), "tiling")?;
    require(
        cases
            .iter()
            .any(|case| case.tile_part_packet_limit.is_some()),
        "tile parts",
    )?;
    require(
        cases.iter().any(|case| !case.precinct_exponents.is_empty()),
        "precincts",
    )?;
    require(cases.iter().any(|case| case.roi.is_some()), "ROI maxshift")?;
    require(
        cases
            .iter()
            .any(|case| case.input == EncoderInputKind::ComponentPlanes),
        "component sampling",
    )?;
    require(
        cases
            .iter()
            .any(|case| case.input == EncoderInputKind::TypedComponentPlanes),
        "mixed typed components",
    )?;
    for target_kind in ["bits-per-pixel", "bytes", "psnr-db"] {
        require(
            cases.iter().any(|case| {
                case.lossy_rate_target
                    .iter()
                    .chain(case.lossy_quality_layers.iter())
                    .any(|target| rate_kind(*target) == target_kind)
            }),
            "lossy rate-target variant",
        )?;
    }
    require(
        cases.iter().any(|case| case.lossless_quality_layers > 1)
            && cases.iter().any(|case| case.lossy_quality_layers.len() > 1),
        "lossless and lossy quality layers",
    )?;
    Ok(())
}

fn require(present: bool, what: &str) -> Result<(), EncoderMatrixError> {
    if present {
        Ok(())
    } else {
        validation(format!("matrix does not cover {what}"))
    }
}

fn rate_kind(target: EncoderRateTarget) -> &'static str {
    match target {
        EncoderRateTarget::BitsPerPixel(_) => "bits-per-pixel",
        EncoderRateTarget::Bytes(_) => "bytes",
        EncoderRateTarget::PsnrDb(_) => "psnr-db",
    }
}
