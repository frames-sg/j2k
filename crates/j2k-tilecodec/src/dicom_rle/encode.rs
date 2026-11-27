// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{DicomRleLayout, HEADER_LEN, MAX_SEGMENTS};
use crate::TileCodecError;
use j2k_core::BufferError;

/// Longest literal or replicate run.
const MAX_RUN: usize = 128;

pub(super) fn encode_frame(
    samples: &[u8],
    layout: DicomRleLayout,
    out: &mut Vec<u8>,
) -> Result<(), TileCodecError> {
    if samples.len() != layout.frame_len() {
        return Err(TileCodecError::LengthMismatch {
            what: "DICOM RLE input samples",
            expected: layout.frame_len(),
            actual: samples.len(),
        });
    }
    let columns = usize::from(layout.columns());
    let rows = usize::from(layout.rows());
    // A run of one byte costs two bytes and longer runs cost less, so a row
    // never encodes to more than twice its length.
    let max_row_len = 2 * columns;

    out.clear();
    try_reserve(out, HEADER_LEN)?;
    out.extend_from_slice(&[0; HEADER_LEN]);
    let mut row = Vec::new();
    row.try_reserve_exact(columns)
        .map_err(|_| allocation_failed(columns, "DICOM RLE row scratch"))?;

    let mut header = [0_u32; 1 + MAX_SEGMENTS];
    header[0] = u32::from(layout.segment_count);
    for segment in 0..layout.segment_count() {
        let start = out.len();
        header[1 + segment] = u32::try_from(start).map_err(|_| {
            TileCodecError::Buffer(BufferError::SizeOverflow {
                what: "DICOM RLE segment offset",
            })
        })?;
        let (first, stride) = layout.segment_lane(segment);
        let mut lane = samples.iter().skip(first).step_by(stride);
        for _ in 0..rows {
            row.clear();
            row.extend(lane.by_ref().take(columns));
            try_reserve(out, max_row_len)?;
            encode_row(&row, out);
        }
        if (out.len() - start) % 2 == 1 {
            try_reserve(out, 1)?;
            out.push(0);
        }
    }

    let header_bytes = header.iter().flat_map(|field| field.to_le_bytes());
    for (slot, byte) in out.iter_mut().zip(header_bytes) {
        *slot = byte;
    }
    Ok(())
}

fn try_reserve(out: &mut Vec<u8>, additional: usize) -> Result<(), TileCodecError> {
    out.try_reserve(additional)
        .map_err(|_| allocation_failed(out.len().saturating_add(additional), "DICOM RLE frame"))
}

const fn allocation_failed(bytes: usize, what: &'static str) -> TileCodecError {
    TileCodecError::Buffer(BufferError::HostAllocationFailed { bytes, what })
}

/// Encode one row. Every group of two or more equal bytes becomes replicate
/// runs, and the single bytes between them literal runs. A group of
/// 128·k + 1 bytes ends with a one-byte literal run.
fn encode_row(row: &[u8], out: &mut Vec<u8>) {
    let mut literal_start = 0;
    let mut position = 0;
    for group in row.chunk_by(|a, b| a == b) {
        if let [byte, ..] = *group {
            if group.len() > 1 {
                push_literal(&row[literal_start..position], out);
                push_replicate(byte, group.len(), out);
                literal_start = position + group.len();
            }
        }
        position += group.len();
    }
    push_literal(&row[literal_start..], out);
}

fn push_literal(bytes: &[u8], out: &mut Vec<u8>) {
    for chunk in bytes.chunks(MAX_RUN) {
        out.push(literal_header(chunk.len()));
        out.extend_from_slice(chunk);
    }
}

fn push_replicate(byte: u8, count: usize, out: &mut Vec<u8>) {
    for _ in 0..count / MAX_RUN {
        out.extend_from_slice(&[replicate_header(MAX_RUN), byte]);
    }
    match count % MAX_RUN {
        0 => {}
        1 => out.extend_from_slice(&[literal_header(1), byte]),
        rest => out.extend_from_slice(&[replicate_header(rest), byte]),
    }
}

/// Header of a literal run of `len` bytes: `len - 1`.
#[expect(
    clippy::cast_possible_truncation,
    reason = "literal runs hold 1 to 128 bytes, so the header is 0 to 127"
)]
const fn literal_header(len: usize) -> u8 {
    (len - 1) as u8
}

/// Header of a replicate run of `count` bytes: `1 - count` as a signed byte.
#[expect(
    clippy::cast_possible_truncation,
    reason = "replicate runs hold 2 to 128 bytes, so the header is 129 to 255"
)]
const fn replicate_header(count: usize) -> u8 {
    (257 - count) as u8
}

#[cfg(test)]
mod tests {
    use super::encode_row;

    fn encoded(row: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        encode_row(row, &mut out);
        out
    }

    #[test]
    fn rows_split_into_literal_and_replicate_runs() {
        assert_eq!(encoded(&[]), [0_u8; 0]);
        assert_eq!(encoded(&[5]), [0, 5]);
        assert_eq!(encoded(&[1, 2, 2, 3]), [0, 1, 255, 2, 0, 3]);
        assert_eq!(encoded(&[4, 4, 4, 4]), [253, 4]);
    }

    #[test]
    fn long_groups_split_at_128_bytes() {
        let run = |count| std::iter::repeat_n(7_u8, count).collect::<Vec<_>>();
        assert_eq!(encoded(&run(128)), [129, 7]);
        assert_eq!(encoded(&run(129)), [129, 7, 0, 7]);
        assert_eq!(encoded(&run(130)), [129, 7, 255, 7]);

        let literal = (0..=129_u8).collect::<Vec<_>>();
        let out = encoded(&literal);
        assert_eq!(out.len(), 132);
        assert_eq!((out[0], out[129], out[130], out[131]), (127, 1, 128, 129));
    }
}
