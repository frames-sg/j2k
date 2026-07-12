// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec::Vec;

use super::super::DecodeSettings;
use super::auxiliary::{com_marker, plm_marker, ppm_marker, tlm_marker};
use super::markers;
use super::size::size_marker;
use super::validation::{skipped_levels_to_reach_target, validate};
use super::{
    coc_marker, cod_marker, poc_marker, qcc_marker, qcd_marker, rgn_marker, skip_marker_segment,
    CodingStyleComponent, ComponentSizeInfo, Header, QuantizationInfo,
};
use crate::error::{bail, DecodingError, MarkerError, Result, ValidationError};
use crate::reader::BitReader;

mod allocation;
mod components;

use allocation::{
    try_extend_progression_changes, try_flatten_packet_lengths, try_flatten_ppm_packets,
    try_none_vec, HeaderMarkerBudget,
};
use components::build_component_infos;

#[expect(
    clippy::similar_names,
    reason = "paired axis, subband, and marker names follow JPEG 2000 specification notation"
)]
#[expect(
    clippy::too_many_lines,
    reason = "the ordered JPEG 2000 state machine stays cohesive to preserve marker, packet, pass, and sample order"
)]
pub(crate) fn read_header<'a>(
    reader: &mut BitReader<'a>,
    settings: &DecodeSettings,
    retained_baseline_bytes: usize,
) -> Result<Header<'a>> {
    if reader.read_marker()? != markers::SIZ {
        bail!(MarkerError::Expected("SIZ"));
    }

    let mut marker_budget = HeaderMarkerBudget::with_retained_baseline(retained_baseline_bytes)?;
    let mut size_data = size_marker(reader, marker_budget.remaining_bytes())?;
    marker_budget.account_capacity::<ComponentSizeInfo>(size_data.component_sizes.capacity())?;

    let mut cod = None;
    let mut qcd = None;

    let num_components = u16::try_from(size_data.component_sizes.len())
        .map_err(|_| ValidationError::TooManyChannels)?;
    let mut cod_components: Vec<Option<CodingStyleComponent>> =
        try_none_vec(usize::from(num_components), &mut marker_budget)?;
    let mut qcd_components: Vec<Option<QuantizationInfo>> =
        try_none_vec(usize::from(num_components), &mut marker_budget)?;
    let mut rgn_components: Vec<Option<u8>> =
        try_none_vec(usize::from(num_components), &mut marker_budget)?;
    let mut progression_changes = Vec::new();
    let mut plm_markers = Vec::new();
    let mut ppm_markers = Vec::new();
    loop {
        match reader.peek_marker().ok_or(MarkerError::Invalid)? {
            markers::SOT => break,
            markers::CAP | markers::CPF => {
                reader.read_marker()?;
                skip_marker_segment(reader).ok_or(MarkerError::ParseFailure("CAP/CPF"))?;
            }
            markers::COD => {
                reader.read_marker()?;
                let replacement = cod_marker(reader)?;
                let old_count = cod
                    .as_ref()
                    .map_or(0, |current: &super::CodingStyleDefault| {
                        current
                            .component_parameters
                            .parameters
                            .precinct_exponents
                            .capacity()
                    });
                let new_count = replacement
                    .component_parameters
                    .parameters
                    .precinct_exponents
                    .capacity();
                marker_budget.account_capacity::<(u8, u8)>(new_count)?;
                cod = Some(replacement);
                marker_budget.release_capacity::<(u8, u8)>(old_count)?;
            }
            markers::COC => {
                reader.read_marker()?;
                let (component_index, coc) = coc_marker(reader, num_components)?;
                let slot = cod_components
                    .get_mut(component_index as usize)
                    .ok_or(MarkerError::ParseFailure("COC"))?;
                let old_count = slot.as_ref().map_or(0, |current| {
                    current.parameters.precinct_exponents.capacity()
                });
                marker_budget
                    .account_capacity::<(u8, u8)>(coc.parameters.precinct_exponents.capacity())?;
                *slot = Some(coc);
                marker_budget.release_capacity::<(u8, u8)>(old_count)?;
            }
            markers::QCD => {
                reader.read_marker()?;
                let replacement = qcd_marker(reader)?;
                let old_count = qcd.as_ref().map_or(0, |current: &super::QuantizationInfo| {
                    current.step_sizes.capacity()
                });
                marker_budget
                    .account_capacity::<super::StepSize>(replacement.step_sizes.capacity())?;
                qcd = Some(replacement);
                marker_budget.release_capacity::<super::StepSize>(old_count)?;
            }
            markers::QCC => {
                reader.read_marker()?;
                let (component_index, qcc) = qcc_marker(reader, num_components)?;
                let slot = qcd_components
                    .get_mut(component_index as usize)
                    .ok_or(MarkerError::ParseFailure("QCC"))?;
                let old_count = slot
                    .as_ref()
                    .map_or(0, |current| current.step_sizes.capacity());
                marker_budget.account_capacity::<super::StepSize>(qcc.step_sizes.capacity())?;
                *slot = Some(qcc);
                marker_budget.release_capacity::<super::StepSize>(old_count)?;
            }
            markers::POC => {
                reader.read_marker()?;
                let num_layers = cod
                    .as_ref()
                    .ok_or(MarkerError::ParseFailure("POC"))?
                    .num_layers;
                let changes = poc_marker(
                    reader,
                    num_components,
                    num_layers,
                    marker_budget.remaining_bytes() / 2,
                )?;
                try_extend_progression_changes(
                    &mut progression_changes,
                    changes,
                    &mut marker_budget,
                )?;
            }
            markers::RGN => {
                reader.read_marker()?;
                let rgn =
                    rgn_marker(reader, num_components).ok_or(MarkerError::ParseFailure("RGN"))?;
                if rgn.style != 0 {
                    bail!(DecodingError::UnsupportedFeature("explicit ROI coding"));
                }
                *rgn_components
                    .get_mut(rgn.component_index as usize)
                    .ok_or(MarkerError::ParseFailure("RGN"))? = Some(rgn.shift);
            }
            markers::TLM => {
                reader.read_marker()?;
                tlm_marker(reader).ok_or(MarkerError::ParseFailure("TLM"))?;
            }
            markers::PLM => {
                reader.read_marker()?;
                marker_budget.try_reserve_next(&mut plm_markers)?;
                let marker = plm_marker(reader, marker_budget.remaining_bytes())?;
                marker_budget.account_capacity::<u32>(marker.packet_lengths.capacity())?;
                plm_markers.push(marker);
            }
            markers::COM => {
                reader.read_marker()?;
                com_marker(reader).ok_or(MarkerError::ParseFailure("COM"))?;
            }
            markers::PPM => {
                reader.read_marker()?;
                marker_budget.try_reserve_next(&mut ppm_markers)?;
                let marker = ppm_marker(reader, marker_budget.remaining_bytes())?;
                marker_budget
                    .account_capacity::<super::PpmPacket<'_>>(marker.packets.capacity())?;
                ppm_markers.push(marker);
            }
            markers::CRG => {
                reader.read_marker()?;
                skip_marker_segment(reader);
            }
            (0x30..=0x3F) => {
                // "All markers with the marker code between 0xFF30 and 0xFF3F
                // have no marker segment parameters. They shall be skipped by
                // the decoder."
                reader.read_marker()?;
                // skip_marker_segment(reader);
            }
            _ => {
                bail!(MarkerError::Unsupported);
            }
        }
    }

    let cod = cod.ok_or(MarkerError::Missing("COD"))?;
    let qcd = qcd.ok_or(MarkerError::Missing("QCD"))?;

    let component_infos = build_component_infos(
        &size_data.component_sizes,
        &cod_components,
        &qcd_components,
        &rgn_components,
        &cod,
        &qcd,
        &mut marker_budget,
    )?;

    // Components can have different number of resolution levels. In that case, we
    // can only skip as many resolution levels as the component with the smallest
    // number of resolution levels.
    let min_num_resolution_levels = component_infos
        .iter()
        .map(super::model::ComponentInfo::num_resolution_levels)
        .min()
        .ok_or(ValidationError::InvalidComponentMetadata)?;
    let skipped_resolution_levels =
        if let Some((target_width, target_height)) = settings.target_resolution {
            if target_width == 0 || target_height == 0 {
                bail!(ValidationError::InvalidDimensions);
            }
            let width_log =
                skipped_levels_to_reach_target(size_data.checked_image_width()?, target_width);
            let height_log =
                skipped_levels_to_reach_target(size_data.checked_image_height()?, target_height);

            width_log.min(height_log)
        } else {
            0
        }
        .min(min_num_resolution_levels - 1);

    // If the user defined a maximum resolution level that is lower than the
    // maximum available one, the final image needs to be shrunk further.
    let resolution_shrink_factor = 1u32
        .checked_shl(u32::from(skipped_resolution_levels))
        .ok_or(ValidationError::InvalidDimensions)?;
    size_data.x_resolution_shrink_factor = size_data
        .x_resolution_shrink_factor
        .checked_mul(resolution_shrink_factor)
        .ok_or(ValidationError::InvalidDimensions)?;
    size_data.y_resolution_shrink_factor = size_data
        .y_resolution_shrink_factor
        .checked_mul(resolution_shrink_factor)
        .ok_or(ValidationError::InvalidDimensions)?;
    size_data.checked_image_width()?;
    size_data.checked_image_height()?;

    ppm_markers.sort_by_key(|ppm_marker| ppm_marker.sequence_idx);
    plm_markers.sort_by_key(|plm_marker| plm_marker.sequence_idx);

    let header = Header {
        size_data,
        global_coding_style: cod,
        component_infos,
        progression_changes,
        plm_packet_lengths: try_flatten_packet_lengths(plm_markers, &mut marker_budget)?,
        ppm_packets: try_flatten_ppm_packets(ppm_markers, &mut marker_budget)?,
        skipped_resolution_levels,
        strict: settings.strict,
    };

    validate(&header)?;

    Ok(header)
}
