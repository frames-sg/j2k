// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{DicomRleDecodeReport, DicomRleLayout, HEADER_LEN, MAX_SEGMENTS};
use crate::{
    error::{malformed_input, truncated_input_at},
    packbits::Runs,
    TileCodecError,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Conformance {
    Strict,
    Lenient,
}

/// Segment offsets from a validated RLE header.
struct Header {
    offsets: [usize; MAX_SEGMENTS],
    count: usize,
    nonzero_unused_offsets: bool,
}

impl Header {
    /// Byte range of segment `index` within the frame.
    fn segment_range(&self, index: usize, frame_len: usize) -> (usize, usize) {
        let start = self.offsets[index];
        let end = if index + 1 < self.count {
            self.offsets[index + 1]
        } else {
            frame_len
        };
        (start, end)
    }
}

pub(super) fn decode_frame(
    input: &[u8],
    layout: DicomRleLayout,
    out: &mut [u8],
    conformance: Conformance,
) -> Result<DicomRleDecodeReport, TileCodecError> {
    if out.len() != layout.frame_len() {
        return Err(TileCodecError::LengthMismatch {
            what: "DICOM RLE output buffer",
            expected: layout.frame_len(),
            actual: out.len(),
        });
    }
    let header = parse_header(input, layout.segment_count())?;
    if header.nonzero_unused_offsets && conformance == Conformance::Strict {
        return Err(malformed_input(
            "DICOM RLE header",
            "unused segment offsets are not zero",
        ));
    }

    let mut report = DicomRleDecodeReport {
        overlong_segments: 0,
        nonzero_unused_offsets: header.nonzero_unused_offsets,
    };
    let pixels = layout.pixels();
    for index in 0..header.count {
        // `parse_header` checked that the ranges lie inside `input`; an empty
        // segment could only end early, which is an error.
        let (start, end) = header.segment_range(index, input.len());
        let segment = Segment {
            data: input.get(start..end).unwrap_or_default(),
            frame_offset: start,
            is_last: index + 1 == header.count,
        };
        let (first, stride) = layout.segment_lane(index);
        let mut slots = out.iter_mut().skip(first).step_by(stride).take(pixels);
        if segment.decode(&mut slots, conformance)? == SegmentEnd::Overlong {
            report.overlong_segments |= 1 << index;
        }
    }
    Ok(report)
}

/// Validate the header against the frame length and the layout's segment
/// count. Unused offsets are reported, not rejected, so that the caller
/// decides by conformance.
fn parse_header(input: &[u8], expected_segments: usize) -> Result<Header, TileCodecError> {
    let Some(header) = input.first_chunk::<HEADER_LEN>() else {
        return Err(truncated_input_at(input.len(), "DICOM RLE header"));
    };
    let (words, _) = header.as_chunks::<4>();
    let mut fields = words.iter().map(|word| u32::from_le_bytes(*word));
    let count = fields.next().unwrap_or_default();
    let count = usize::try_from(count)
        .ok()
        .filter(|count| (1..=MAX_SEGMENTS).contains(count))
        .ok_or_else(|| {
            malformed_input(
                "DICOM RLE header",
                format!("segment count {count} is not between 1 and {MAX_SEGMENTS}"),
            )
        })?;
    if count != expected_segments {
        return Err(malformed_input(
            "DICOM RLE header",
            format!("frame has {count} segments; the layout needs {expected_segments}"),
        ));
    }

    let mut offsets = [0_usize; MAX_SEGMENTS];
    let mut previous = None;
    for (slot, field) in offsets.iter_mut().zip(fields.by_ref()).take(count) {
        // An offset that does not fit in usize is past the end of the frame.
        let offset = usize::try_from(field).unwrap_or(usize::MAX);
        match previous {
            None if offset != HEADER_LEN => {
                return Err(malformed_input(
                    "DICOM RLE header",
                    format!("first segment offset {offset} is not {HEADER_LEN}"),
                ));
            }
            Some(previous) if offset <= previous => {
                return Err(malformed_input(
                    "DICOM RLE header",
                    format!("segment offset {offset} does not follow {previous}"),
                ));
            }
            _ => {}
        }
        *slot = offset;
        previous = Some(offset);
    }
    // Offsets increase, so checking the last one bounds them all. Every
    // segment needs at least one byte.
    if previous.is_none_or(|last| last >= input.len()) {
        return Err(truncated_input_at(input.len(), "DICOM RLE segment offsets"));
    }
    let nonzero_unused_offsets = fields.any(|field| field != 0);
    Ok(Header {
        offsets,
        count,
        nonzero_unused_offsets,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SegmentEnd {
    /// The data ended once the rows were complete, apart from padding.
    Exact,
    /// Lenient decoding discarded data after the rows were complete.
    Overlong,
}

struct Segment<'a> {
    data: &'a [u8],
    /// Offset of `data` within the frame, for error positions.
    frame_offset: usize,
    is_last: bool,
}

impl Segment<'_> {
    /// Decode the segment into `slots`, one byte per pixel.
    fn decode<'s, I>(
        &self,
        slots: &mut I,
        conformance: Conformance,
    ) -> Result<SegmentEnd, TileCodecError>
    where
        I: ExactSizeIterator<Item = &'s mut u8>,
    {
        let mut runs = Runs::new(self.data);
        while slots.len() > 0 {
            let run = match runs.next() {
                Some(Ok(run)) => run,
                Some(Err(truncated))
                    if conformance == Conformance::Lenient
                        && truncated.available.len() >= slots.len() =>
                {
                    // A literal run cut off by the end of the segment that
                    // still completes the rows.
                    for (&byte, slot) in truncated.available.iter().zip(slots) {
                        *slot = byte;
                    }
                    return Ok(SegmentEnd::Overlong);
                }
                Some(Err(_)) | None => return Err(self.ended_early()),
            };
            if run.len() > slots.len() {
                if conformance == Conformance::Strict {
                    return Err(malformed_input(
                        "DICOM RLE segment",
                        "a run continues past the end of the segment's rows",
                    ));
                }
                run.write_to(slots);
                return Ok(SegmentEnd::Overlong);
            }
            run.write_to(slots);
        }
        if is_padding(runs.remaining()) {
            Ok(SegmentEnd::Exact)
        } else if conformance == Conformance::Lenient {
            Ok(SegmentEnd::Overlong)
        } else {
            Err(malformed_input(
                "DICOM RLE segment",
                "data continues after the segment's rows are complete",
            ))
        }
    }

    /// The segment ran out of data before its rows were complete. For the
    /// last segment that means the frame is truncated; an earlier segment
    /// is bounded by the next offset, so it is malformed.
    fn ended_early(&self) -> TileCodecError {
        if self.is_last {
            truncated_input_at(self.frame_offset + self.data.len(), "DICOM RLE segment")
        } else {
            malformed_input(
                "DICOM RLE segment",
                format!(
                    "segment at offset {} ends before its rows are complete",
                    self.frame_offset
                ),
            )
        }
    }
}

/// Bytes allowed after a segment's rows: -128 no-op codes, then at most one
/// zero byte that pads the segment to an even length.
fn is_padding(rest: &[u8]) -> bool {
    let rest = rest.strip_suffix(&[0]).unwrap_or(rest);
    rest.iter().all(|&byte| byte == 0x80)
}

#[cfg(test)]
mod tests {
    use super::is_padding;

    #[test]
    fn padding_is_no_ops_then_one_zero_byte() {
        for padding in [&[][..], &[0], &[0x80], &[0x80, 0x80, 0]] {
            assert!(is_padding(padding), "{padding:?}");
        }
        for not_padding in [&[0, 0][..], &[0, 0x80], &[1], &[0xFF, 0]] {
            assert!(!is_padding(not_padding), "{not_padding:?}");
        }
    }
}
