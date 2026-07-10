// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec;
use alloc::vec::Vec;

use super::super::DecodeSettings;
use super::auxiliary::{com_marker, plm_marker, ppm_marker, tlm_marker};
use super::markers;
use super::size::size_marker;
use super::validation::{skipped_levels_to_reach_target, validate};
use super::{
    coc_marker, cod_marker, poc_marker, qcc_marker, qcd_marker, rgn_marker, skip_marker_segment,
    ComponentInfo, Header,
};
use crate::error::{bail, DecodingError, MarkerError, Result, ValidationError};
use crate::reader::BitReader;

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
) -> Result<Header<'a>> {
    if reader.read_marker()? != markers::SIZ {
        bail!(MarkerError::Expected("SIZ"));
    }

    let mut size_data = size_marker(reader)?;

    let mut cod = None;
    let mut qcd = None;

    let num_components = u16::try_from(size_data.component_sizes.len())
        .map_err(|_| ValidationError::TooManyChannels)?;
    let mut cod_components = vec![None; usize::from(num_components)];
    let mut qcd_components = vec![None; usize::from(num_components)];
    let mut rgn_components = vec![None; usize::from(num_components)];
    let mut progression_changes = vec![];
    let mut plm_markers = vec![];
    let mut ppm_markers = vec![];

    loop {
        match reader.peek_marker().ok_or(MarkerError::Invalid)? {
            markers::SOT => break,
            markers::CAP | markers::CPF => {
                reader.read_marker()?;
                skip_marker_segment(reader).ok_or(MarkerError::ParseFailure("CAP/CPF"))?;
            }
            markers::COD => {
                reader.read_marker()?;
                cod = Some(cod_marker(reader).ok_or(MarkerError::ParseFailure("COD"))?);
            }
            markers::COC => {
                reader.read_marker()?;
                let (component_index, coc) =
                    coc_marker(reader, num_components).ok_or(MarkerError::ParseFailure("COC"))?;
                *cod_components
                    .get_mut(component_index as usize)
                    .ok_or(MarkerError::ParseFailure("COC"))? = Some(coc);
            }
            markers::QCD => {
                reader.read_marker()?;
                qcd = Some(qcd_marker(reader).ok_or(MarkerError::ParseFailure("QCD"))?);
            }
            markers::QCC => {
                reader.read_marker()?;
                let (component_index, qcc) =
                    qcc_marker(reader, num_components).ok_or(MarkerError::ParseFailure("QCC"))?;
                *qcd_components
                    .get_mut(component_index as usize)
                    .ok_or(MarkerError::ParseFailure("QCC"))? = Some(qcc);
            }
            markers::POC => {
                reader.read_marker()?;
                let num_layers = cod
                    .as_ref()
                    .ok_or(MarkerError::ParseFailure("POC"))?
                    .num_layers;
                progression_changes.extend(
                    poc_marker(reader, num_components, num_layers)
                        .ok_or(MarkerError::ParseFailure("POC"))?,
                );
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
                plm_markers.push(plm_marker(reader).ok_or(MarkerError::ParseFailure("PLM"))?);
            }
            markers::COM => {
                reader.read_marker()?;
                com_marker(reader).ok_or(MarkerError::ParseFailure("COM"))?;
            }
            markers::PPM => {
                reader.read_marker()?;
                ppm_markers.push(ppm_marker(reader).ok_or(MarkerError::ParseFailure("PPM"))?);
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

    let component_infos: Vec<ComponentInfo> = size_data
        .component_sizes
        .iter()
        .enumerate()
        .map(|(idx, csi)| ComponentInfo {
            size_info: *csi,
            coding_style: cod_components[idx]
                .clone()
                .map(|mut c| {
                    c.flags.raw |= cod.component_parameters.flags.raw;

                    c
                })
                .unwrap_or(cod.component_parameters.clone()),
            quantization_info: qcd_components[idx].clone().unwrap_or(qcd.clone()),
            roi_shift: rgn_components[idx].unwrap_or(0),
        })
        .collect();

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
        global_coding_style: cod.clone(),
        component_infos,
        progression_changes,
        plm_packet_lengths: plm_markers
            .into_iter()
            .flat_map(|marker| marker.packet_lengths)
            .collect(),
        ppm_packets: ppm_markers
            .into_iter()
            .flat_map(|i| i.packets)
            .filter_map(|p| if p.data.is_empty() { None } else { Some(p) })
            .collect(),
        skipped_resolution_levels,
        strict: settings.strict,
    };

    validate(&header)?;

    Ok(header)
}
