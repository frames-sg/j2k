// SPDX-License-Identifier: MIT OR Apache-2.0

use super::progression::read_component_index;
use super::{PpmMarkerData, RgnMarkerData};
use crate::error::{MarkerError, Result};
use crate::reader::BitReader;

mod packet_lengths;

#[cfg(test)]
pub(crate) use packet_lengths::decode_packet_lengths;
pub(super) use packet_lengths::plm_marker;
pub(crate) use packet_lengths::plt_marker;

/// COM Marker (A.9.2).
pub(super) fn com_marker(reader: &mut BitReader<'_>) -> Option<()> {
    skip_marker_segment(reader)
}

/// TLM marker (A.7.1).
pub(super) fn tlm_marker(reader: &mut BitReader<'_>) -> Option<()> {
    skip_marker_segment(reader)
}

/// PPM marker (A.7.4).
pub(super) fn ppm_marker<'a>(reader: &mut BitReader<'a>) -> Result<PpmMarkerData<'a>> {
    let segment_len = reader
        .read_u16()
        .and_then(|length| length.checked_sub(2))
        .ok_or(MarkerError::ParseFailure("PPM"))? as usize;
    let ppm_data = reader
        .read_bytes(segment_len)
        .ok_or(MarkerError::ParseFailure("PPM"))?;
    let sequence_idx = ppm_data
        .first()
        .copied()
        .ok_or(MarkerError::ParseFailure("PPM"))?;

    Ok(PpmMarkerData {
        sequence_idx,
        data: &ppm_data[1..],
    })
}

/// RGN marker (A.6.3).
pub(crate) fn rgn_marker(reader: &mut BitReader<'_>, csiz: u16) -> Option<RgnMarkerData> {
    let length = reader.read_u16()?;
    let component_index_bytes = if csiz < 257 { 1 } else { 2 };
    if length != 4 + component_index_bytes {
        return None;
    }

    let component_index = read_component_index(reader, csiz)?;
    if component_index >= csiz {
        return None;
    }

    let style = reader.read_byte()?;
    let shift = reader.read_byte()?;

    Some(RgnMarkerData {
        component_index,
        style,
        shift,
    })
}

/// CRG marker (A.6.2).
pub(crate) fn crg_marker(reader: &mut BitReader<'_>, csiz: u16) -> Option<()> {
    let length = reader.read_u16()?;
    let payload_length = usize::from(csiz).checked_mul(4)?;
    if usize::from(length) != payload_length.checked_add(2)? {
        return None;
    }
    reader.skip_bytes(payload_length)
}

pub(crate) fn skip_marker_segment(reader: &mut BitReader<'_>) -> Option<()> {
    marker_segment_payload(reader).map(|_| ())
}

pub(crate) fn marker_segment_payload<'a>(reader: &mut BitReader<'a>) -> Option<&'a [u8]> {
    let length = reader.read_u16()?.checked_sub(2)?;
    reader.read_bytes(length as usize)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ppm_retains_payload_for_cross_marker_accumulation() {
        let data = [0, 7, 3, 0, 0, 1, 0];
        let mut reader = BitReader::new(&data);

        let marker = ppm_marker(&mut reader).expect("valid PPM marker segment");

        assert_eq!(marker.sequence_idx, 3);
        assert_eq!(marker.data, [0, 0, 1, 0]);
        assert_eq!(reader.offset(), data.len());
    }
}
