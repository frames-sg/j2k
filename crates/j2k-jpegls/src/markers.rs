// SPDX-License-Identifier: MIT OR Apache-2.0

//! Marker segment parsing and writing (T.87 Annex C, T.81 Annex B).

use crate::error::{JpegLsError, UnsupportedFeature};
use crate::info::{Info, InterleaveMode, ScanInfo};
use crate::params::{CodingParameters, PresetCodingParameters};

pub(crate) const SOI: u8 = 0xD8;
pub(crate) const EOI: u8 = 0xD9;
pub(crate) const SOS: u8 = 0xDA;
pub(crate) const DRI: u8 = 0xDD;
pub(crate) const SOF55: u8 = 0xF7;
pub(crate) const LSE: u8 = 0xF8;
pub(crate) const RST0: u8 = 0xD0;
const DNL: u8 = 0xDC;
const COM: u8 = 0xFE;
const APP8: u8 = 0xE8;

/// LSE parameter IDs (T.87 C.2.4.1).
const LSE_PRESET_PARAMETERS: u8 = 1;
const LSE_MAPPING_TABLE: u8 = 2;
const LSE_MAPPING_TABLE_CONTINUATION: u8 = 3;
const LSE_OVERSIZE_DIMENSIONS: u8 = 4;

/// Frame header (SOF55).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Frame {
    pub(crate) width: u16,
    pub(crate) height: u16,
    pub(crate) bit_depth: u8,
    pub(crate) component_ids: Vec<u8>,
}

/// A scan header and where its entropy-coded data starts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Scan {
    pub(crate) components: Vec<usize>,
    pub(crate) near: u8,
    pub(crate) interleave: InterleaveMode,
    pub(crate) restart_interval: u32,
    pub(crate) params: CodingParameters,
    /// Offset of the first entropy-coded byte.
    pub(crate) data_offset: usize,
    /// Offset of the marker that ends the scan's data.
    pub(crate) end_offset: usize,
}

/// The validated marker structure of a JPEG-LS stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Stream {
    pub(crate) frame: Frame,
    pub(crate) scans: Vec<Scan>,
    pub(crate) first_preset: Option<PresetCodingParameters>,
}

impl Stream {
    pub(crate) fn info(&self) -> Info {
        let scans: Vec<ScanInfo> = self
            .scans
            .iter()
            .map(|scan| ScanInfo {
                components: scan.components.clone(),
                near: scan.near,
                interleave: scan.interleave,
                restart_interval: (scan.restart_interval != 0).then_some(scan.restart_interval),
                parameters: scan.params.preset(),
            })
            .collect();
        let first = scans.first();
        Info {
            dimensions: (u32::from(self.frame.width), u32::from(self.frame.height)),
            components: u8::try_from(self.frame.component_ids.len()).unwrap_or(u8::MAX),
            component_ids: self.frame.component_ids.clone(),
            bit_depth: self.frame.bit_depth,
            near: first.map_or(0, |scan| scan.near),
            interleave: first.map_or(InterleaveMode::None, |scan| scan.interleave),
            restart_interval: first.and_then(|scan| scan.restart_interval),
            preset: self.first_preset,
            scans,
        }
    }
}

/// Byte cursor over the stream with bounds-checked reads.
struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl Cursor<'_> {
    fn byte(&mut self) -> Result<u8, JpegLsError> {
        let byte = *self
            .data
            .get(self.pos)
            .ok_or(JpegLsError::Truncated { offset: self.pos })?;
        self.pos += 1;
        Ok(byte)
    }

    fn u16(&mut self) -> Result<u16, JpegLsError> {
        Ok(u16::from_be_bytes([self.byte()?, self.byte()?]))
    }

    /// Read a marker, skipping fill bytes. Returns the code and the offset of
    /// its first `FF`.
    fn marker(&mut self) -> Result<(u8, usize), JpegLsError> {
        let offset = self.pos;
        let first = self.byte()?;
        if first != 0xFF {
            return Err(JpegLsError::UnexpectedMarker {
                marker: first,
                offset,
            });
        }
        let mut code = self.byte()?;
        while code == 0xFF {
            code = self.byte()?;
        }
        Ok((code, offset))
    }

    /// Read a segment length and return the payload, advancing past it.
    fn segment(&mut self, marker: u8, offset: usize) -> Result<&[u8], JpegLsError> {
        let length = usize::from(self.u16()?);
        if length < 2 {
            return Err(invalid(marker, offset, "segment length below 2"));
        }
        let start = self.pos;
        let end = start + length - 2;
        let payload = self.data.get(start..end).ok_or(JpegLsError::Truncated {
            offset: self.data.len(),
        })?;
        self.pos = end;
        Ok(payload)
    }
}

const fn invalid(marker: u8, offset: usize, reason: &'static str) -> JpegLsError {
    JpegLsError::InvalidSegment {
        marker,
        offset,
        reason,
    }
}

fn be16(payload: &[u8], at: usize) -> u16 {
    u16::from_be_bytes([
        payload.get(at).copied().unwrap_or(0),
        payload.get(at + 1).copied().unwrap_or(0),
    ])
}

/// Header state while walking the stream.
#[derive(Default)]
struct ParseState {
    frame: Option<Frame>,
    preset: Option<PresetCodingParameters>,
    first_preset: Option<PresetCodingParameters>,
    restart_interval: u32,
    scans: Vec<Scan>,
    coded: Vec<bool>,
}

/// Parse and validate the marker structure of a JPEG-LS stream, skipping
/// over entropy-coded data.
pub(crate) fn parse(data: &[u8]) -> Result<Stream, JpegLsError> {
    if data.get(..2) != Some(&[0xFF, SOI]) {
        return Err(JpegLsError::MissingSoi);
    }
    let mut cursor = Cursor { data, pos: 2 };
    let mut state = ParseState::default();
    loop {
        let (code, offset) = cursor.marker()?;
        match code {
            EOI => return finish(state),
            SOF55 => {
                let payload = cursor.segment(code, offset)?;
                if state.frame.is_some() {
                    return Err(JpegLsError::UnexpectedMarker {
                        marker: code,
                        offset,
                    });
                }
                let frame = parse_frame(payload, offset)?;
                state.coded = core::iter::repeat_n(false, frame.component_ids.len()).collect();
                state.frame = Some(frame);
            }
            0xC0..=0xCF if !matches!(code, 0xC4 | 0xC8 | 0xCC) => {
                return Err(UnsupportedFeature::NonJpegLsFrame { marker: code }.into());
            }
            LSE => {
                let payload = cursor.segment(code, offset)?;
                let preset = parse_lse(payload, offset)?;
                if state.scans.is_empty() {
                    state.first_preset = Some(preset);
                }
                state.preset = Some(preset);
            }
            DRI => {
                let payload = cursor.segment(code, offset)?;
                state.restart_interval = parse_dri(payload, offset)?;
            }
            SOS => {
                let payload = cursor.segment(code, offset)?;
                let mut scan = parse_scan(payload, offset, &mut state)?;
                scan.data_offset = cursor.pos;
                scan.end_offset = skip_entropy_data(data, cursor.pos);
                cursor.pos = scan.end_offset;
                state.scans.push(scan);
            }
            APP8 => {
                let payload = cursor.segment(code, offset)?;
                check_hp_color_transform(payload)?;
            }
            0xE0..=0xEF | COM => {
                cursor.segment(code, offset)?;
            }
            DNL => return Err(UnsupportedFeature::DeferredHeight.into()),
            _ => {
                return Err(JpegLsError::UnexpectedMarker {
                    marker: code,
                    offset,
                })
            }
        }
    }
}

fn finish(state: ParseState) -> Result<Stream, JpegLsError> {
    let frame = state
        .frame
        .ok_or(JpegLsError::MissingMarker { marker: "SOF55" })?;
    if state.scans.is_empty() {
        return Err(JpegLsError::MissingMarker { marker: "SOS" });
    }
    if let Some(index) = state.coded.iter().position(|&coded| !coded) {
        return Err(JpegLsError::MissingComponent {
            component_id: frame.component_ids.get(index).copied().unwrap_or(0),
        });
    }
    Ok(Stream {
        frame,
        scans: state.scans,
        first_preset: state.first_preset,
    })
}

fn parse_frame(payload: &[u8], offset: usize) -> Result<Frame, JpegLsError> {
    let invalid = |reason| invalid(SOF55, offset, reason);
    if payload.len() < 6 {
        return Err(invalid("frame header shorter than 6 bytes"));
    }
    let bit_depth = payload[0];
    let height = be16(payload, 1);
    let width = be16(payload, 3);
    let count = usize::from(payload[5]);
    if !(2..=16).contains(&bit_depth) {
        return Err(invalid("bits per sample must be 2 to 16"));
    }
    if width == 0 {
        return Err(invalid("zero width"));
    }
    if height == 0 {
        return Err(UnsupportedFeature::DeferredHeight.into());
    }
    if count == 0 {
        return Err(invalid("zero components"));
    }
    if payload.len() != 6 + 3 * count {
        return Err(invalid("length does not match the component count"));
    }
    let mut component_ids = Vec::new();
    for spec in payload[6..].as_chunks::<3>().0 {
        let &[id, sampling, _] = spec;
        if component_ids.contains(&id) {
            return Err(invalid("duplicate component identifier"));
        }
        if sampling != 0x11 {
            return Err(UnsupportedFeature::SamplingFactors {
                component_id: id,
                horizontal: sampling >> 4,
                vertical: sampling & 0x0F,
            }
            .into());
        }
        component_ids.push(id);
    }
    Ok(Frame {
        width,
        height,
        bit_depth,
        component_ids,
    })
}

fn parse_lse(payload: &[u8], offset: usize) -> Result<PresetCodingParameters, JpegLsError> {
    match payload.first().copied() {
        Some(LSE_PRESET_PARAMETERS) => {
            if payload.len() != 11 {
                return Err(invalid(LSE, offset, "preset parameters must be 11 bytes"));
            }
            Ok(PresetCodingParameters {
                max_value: be16(payload, 1),
                t1: be16(payload, 3),
                t2: be16(payload, 5),
                t3: be16(payload, 7),
                reset: be16(payload, 9),
            })
        }
        Some(LSE_MAPPING_TABLE | LSE_MAPPING_TABLE_CONTINUATION) => {
            Err(UnsupportedFeature::MappingTable.into())
        }
        Some(LSE_OVERSIZE_DIMENSIONS) => Err(UnsupportedFeature::OversizeDimensions.into()),
        _ => Err(invalid(LSE, offset, "unknown LSE parameter ID")),
    }
}

fn parse_dri(payload: &[u8], offset: usize) -> Result<u32, JpegLsError> {
    // T.87 C.2.5 allows a 2-, 3-, or 4-byte restart interval.
    if !(2..=4).contains(&payload.len()) {
        return Err(invalid(
            DRI,
            offset,
            "restart interval must be 2 to 4 bytes",
        ));
    }
    Ok(payload
        .iter()
        .fold(0_u32, |value, &byte| (value << 8) | u32::from(byte)))
}

fn parse_scan(payload: &[u8], offset: usize, state: &mut ParseState) -> Result<Scan, JpegLsError> {
    let invalid = |reason| invalid(SOS, offset, reason);
    let frame = state
        .frame
        .as_ref()
        .ok_or(JpegLsError::MissingMarker { marker: "SOF55" })?;
    let count = usize::from(
        *payload
            .first()
            .ok_or_else(|| invalid("empty scan header"))?,
    );
    if payload.len() != 4 + 2 * count {
        return Err(invalid("length does not match the component count"));
    }
    let near = payload[1 + 2 * count];
    let interleave = InterleaveMode::from_code(payload[2 + 2 * count])
        .ok_or_else(|| invalid("interleave mode must be 0, 1, or 2"))?;
    let transform = payload[3 + 2 * count];
    if count == 0 {
        return Err(invalid("zero components"));
    }
    if count > 1 && interleave == InterleaveMode::None {
        return Err(invalid("interleave mode 0 codes one component per scan"));
    }
    if count > crate::context::MAX_INTERLEAVED_COMPONENTS {
        return Err(invalid("a scan codes at most 4 components"));
    }
    if transform >> 4 != 0 {
        return Err(invalid("Ah must be zero"));
    }
    if transform != 0 {
        return Err(UnsupportedFeature::PointTransform(transform).into());
    }
    let mut components = Vec::new();
    for spec in payload[1..=2 * count].as_chunks::<2>().0 {
        let &[id, table] = spec;
        let index = frame
            .component_ids
            .iter()
            .position(|&frame_id| frame_id == id)
            .ok_or_else(|| invalid("component not in the frame"))?;
        if table != 0 {
            return Err(UnsupportedFeature::MappingTable.into());
        }
        let coded = state
            .coded
            .get_mut(index)
            .ok_or_else(|| invalid("component not in the frame"))?;
        if *coded {
            return Err(invalid("component coded by an earlier scan"));
        }
        *coded = true;
        components.push(index);
    }
    let params =
        CodingParameters::resolve(state.preset, frame.bit_depth, near).map_err(|error| {
            JpegLsError::InvalidPresetParameters {
                reason: error.reason(),
            }
        })?;
    Ok(Scan {
        components,
        near,
        interleave,
        restart_interval: state.restart_interval,
        params,
        data_offset: 0,
        end_offset: 0,
    })
}

/// Reject an HP color transform (APP8 `mrfx`) other than none.
fn check_hp_color_transform(payload: &[u8]) -> Result<(), JpegLsError> {
    if payload.len() == 5 && payload.starts_with(b"mrfx") && payload[4] != 0 {
        return Err(UnsupportedFeature::ColorTransform(payload[4]).into());
    }
    Ok(())
}

/// Offset of the next marker at or after `from` that is not a restart marker,
/// or the end of the data.
fn skip_entropy_data(data: &[u8], from: usize) -> usize {
    let mut pos = from;
    loop {
        let end = next_marker(data, pos);
        match data.get(end + 1) {
            Some(code) if (RST0..RST0 + 8).contains(code) => pos = end + 2,
            _ => return end,
        }
    }
}

/// Offset of the first `FF` at or after `from` followed by a byte with its
/// high bit set, or the end of the data. In entropy-coded data, `FF` is
/// always followed by a byte below `0x80`.
pub(crate) fn next_marker(data: &[u8], from: usize) -> usize {
    let mut pos = from;
    while let Some(offset) = data
        .get(pos..)
        .and_then(|rest| rest.iter().position(|&b| b == 0xFF))
    {
        let ff = pos + offset;
        match data.get(ff + 1) {
            Some(&next) if next >= 0x80 => return ff,
            Some(_) => pos = ff + 1,
            None => return data.len(),
        }
    }
    data.len()
}

/// Write the SOI and SOF55 marker segments.
pub(crate) fn write_frame(
    out: &mut Vec<u8>,
    width: u16,
    height: u16,
    bit_depth: u8,
    components: u8,
) {
    out.extend_from_slice(&[0xFF, SOI, 0xFF, SOF55]);
    out.extend_from_slice(&(8 + 3 * u16::from(components)).to_be_bytes());
    out.push(bit_depth);
    out.extend_from_slice(&height.to_be_bytes());
    out.extend_from_slice(&width.to_be_bytes());
    out.push(components);
    for id in 1..=components {
        out.extend_from_slice(&[id, 0x11, 0]);
    }
}

/// Write an LSE segment with preset coding parameters.
pub(crate) fn write_preset(out: &mut Vec<u8>, preset: PresetCodingParameters) {
    out.extend_from_slice(&[0xFF, LSE, 0, 13, LSE_PRESET_PARAMETERS]);
    for value in [
        preset.max_value,
        preset.t1,
        preset.t2,
        preset.t3,
        preset.reset,
    ] {
        out.extend_from_slice(&value.to_be_bytes());
    }
}

/// Write a DRI segment with a 2-byte restart interval.
pub(crate) fn write_restart_interval(out: &mut Vec<u8>, lines: u16) {
    out.extend_from_slice(&[0xFF, DRI, 0, 4]);
    out.extend_from_slice(&lines.to_be_bytes());
}

/// Write an SOS segment. `component_ids` are frame component identifiers.
pub(crate) fn write_scan(
    out: &mut Vec<u8>,
    component_ids: &[u8],
    near: u8,
    interleave: InterleaveMode,
) {
    let count = u8::try_from(component_ids.len()).unwrap_or(u8::MAX);
    out.extend_from_slice(&[0xFF, SOS]);
    out.extend_from_slice(&(6 + 2 * u16::from(count)).to_be_bytes());
    out.push(count);
    for &id in component_ids {
        out.extend_from_slice(&[id, 0]);
    }
    out.extend_from_slice(&[near, interleave.code(), 0]);
}

pub(crate) fn write_end(out: &mut Vec<u8>) {
    out.extend_from_slice(&[0xFF, EOI]);
}

#[cfg(test)]
mod tests {
    use super::{next_marker, parse};
    use crate::error::{JpegLsError, UnsupportedFeature};

    /// SOI, SOF55 (8-bit, 2x1, one component), SOS, two data bytes, EOI.
    fn minimal() -> Vec<u8> {
        Vec::from([
            0xFF, 0xD8, 0xFF, 0xF7, 0, 11, 8, 0, 1, 0, 2, 1, 1, 0x11, 0, 0xFF, 0xDA, 0, 8, 1, 1, 0,
            0, 0, 0, 0x00, 0x00, 0xFF, 0xD9,
        ])
    }

    #[test]
    fn parses_a_minimal_stream() {
        let stream = parse(&minimal()).expect("valid stream");
        assert_eq!((stream.frame.width, stream.frame.height), (2, 1));
        assert_eq!(stream.scans.len(), 1);
        assert_eq!(stream.scans[0].data_offset, 25);
        assert_eq!(stream.scans[0].end_offset, 27);
    }

    #[test]
    fn marker_search_skips_stuffed_ff_bytes() {
        assert_eq!(next_marker(&[0x12, 0xFF, 0x7F, 0xFF, 0xD9], 0), 3);
        assert_eq!(next_marker(&[0x12, 0xFF], 0), 2);
    }

    #[test]
    fn rejects_unsupported_and_malformed_headers() {
        let mut sampled = minimal();
        sampled[13] = 0x21;
        assert!(matches!(
            parse(&sampled),
            Err(JpegLsError::Unsupported(
                UnsupportedFeature::SamplingFactors { .. }
            ))
        ));
        let mut baseline = minimal();
        baseline[3] = 0xC0;
        assert_eq!(
            parse(&baseline),
            Err(JpegLsError::Unsupported(
                UnsupportedFeature::NonJpegLsFrame { marker: 0xC0 }
            ))
        );
        let mut transformed = minimal();
        transformed[24] = 1;
        assert_eq!(
            parse(&transformed),
            Err(JpegLsError::Unsupported(
                UnsupportedFeature::PointTransform(1)
            ))
        );
        let no_eoi = &minimal()[..27];
        assert!(matches!(parse(no_eoi), Err(JpegLsError::Truncated { .. })));
    }
}
