// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k::{
    EncodeBackendPreference, J2kBlockCodingMode, J2kEncodeValidation, J2kLosslessEncodeOptions,
    J2kLossyEncodeOptions, J2kMarkerSegment, J2kProgressionOrder, J2kQualityLayer, J2kRateTarget,
    ReversibleTransform,
};

use crate::encoder::{EncoderBlockCoding, EncoderCase, EncoderInputKind, EncoderRateTarget};

pub(super) fn lossless_options(
    case: &EncoderCase,
    backend: EncodeBackendPreference,
) -> J2kLosslessEncodeOptions {
    let mut options = J2kLosslessEncodeOptions::default();
    options.backend = backend;
    options.block_coding_mode = block_coding(case.block_coding);
    options.progression = progression(case.progression);
    options.max_decomposition_levels = Some(case.decomposition_levels);
    options.tile_size = case.tile_size.map(|[width, height]| (width, height));
    options.tile_part_packet_limit = case.tile_part_packet_limit;
    options.quality_layers = case.lossless_quality_layers;
    options.write_tlm = case.markers.contains(&crate::EncoderMarker::Tlm);
    options.write_plt = case.markers.contains(&crate::EncoderMarker::Plt);
    options.write_plm = case.markers.contains(&crate::EncoderMarker::Plm);
    options.write_ppm = case.markers.contains(&crate::EncoderMarker::Ppm);
    options.write_ppt = case.markers.contains(&crate::EncoderMarker::Ppt);
    options.write_sop = case.markers.contains(&crate::EncoderMarker::Sop);
    options.write_eph = case.markers.contains(&crate::EncoderMarker::Eph);
    options.reversible_transform =
        if case.input == EncoderInputKind::Interleaved && matches!(case.components, 3 | 4) {
            ReversibleTransform::Rct53
        } else {
            ReversibleTransform::None53
        };
    options.validation = J2kEncodeValidation::External;
    options
}

pub(super) fn lossy_options(
    case: &EncoderCase,
    backend: EncodeBackendPreference,
) -> J2kLossyEncodeOptions {
    let mut options = J2kLossyEncodeOptions::default();
    options.backend = backend;
    options.block_coding_mode = block_coding(case.block_coding);
    options.progression = progression(case.progression);
    options.max_decomposition_levels = Some(case.decomposition_levels);
    options.rate_target = case.lossy_rate_target.map(rate_target);
    options.quality_layers = case
        .lossy_quality_layers
        .iter()
        .copied()
        .map(rate_target)
        .map(J2kQualityLayer::new)
        .collect();
    options.tile_size = case.tile_size.map(|[width, height]| (width, height));
    options.tile_part_packet_limit = case.tile_part_packet_limit;
    options.precinct_exponents = case
        .precinct_exponents
        .iter()
        .map(|[width, height]| (*width, *height))
        .collect();
    options.marker_segments = marker_segments(case);
    options.validation = J2kEncodeValidation::External;
    options
}

const fn block_coding(value: EncoderBlockCoding) -> J2kBlockCodingMode {
    match value {
        EncoderBlockCoding::Classic => J2kBlockCodingMode::Classic,
        EncoderBlockCoding::HighThroughput => J2kBlockCodingMode::HighThroughput,
    }
}

pub(super) fn progression(value: crate::EncoderProgression) -> J2kProgressionOrder {
    match value {
        crate::EncoderProgression::Lrcp => J2kProgressionOrder::Lrcp,
        crate::EncoderProgression::Rlcp => J2kProgressionOrder::Rlcp,
        crate::EncoderProgression::Rpcl => J2kProgressionOrder::Rpcl,
        crate::EncoderProgression::Pcrl => J2kProgressionOrder::Pcrl,
        crate::EncoderProgression::Cprl => J2kProgressionOrder::Cprl,
    }
}

fn rate_target(value: EncoderRateTarget) -> J2kRateTarget {
    match value {
        EncoderRateTarget::BitsPerPixel(value) => J2kRateTarget::BitsPerPixel(value),
        EncoderRateTarget::Bytes(value) => J2kRateTarget::Bytes(value),
        EncoderRateTarget::PsnrDb(value) => J2kRateTarget::PsnrDb(value),
    }
}

fn marker_segments(case: &EncoderCase) -> Vec<J2kMarkerSegment> {
    case.markers
        .iter()
        .filter_map(|marker| match marker {
            crate::EncoderMarker::Tlm => Some(J2kMarkerSegment::Tlm),
            crate::EncoderMarker::Plm => Some(J2kMarkerSegment::Plm),
            crate::EncoderMarker::Plt => Some(J2kMarkerSegment::Plt),
            crate::EncoderMarker::Ppm => Some(J2kMarkerSegment::Ppm),
            crate::EncoderMarker::Ppt => Some(J2kMarkerSegment::Ppt),
            crate::EncoderMarker::Sop => Some(J2kMarkerSegment::Sop),
            crate::EncoderMarker::Eph => Some(J2kMarkerSegment::Eph),
            _ => None,
        })
        .collect()
}
