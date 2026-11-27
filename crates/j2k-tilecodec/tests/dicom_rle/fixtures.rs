// SPDX-License-Identifier: MIT OR Apache-2.0

//! The DICOM RLE fixture matrix that `corpus/conformance/rle_matrix.py`
//! writes: real frames from the RLE test files bundled with pydicom 2.4.4
//! and synthetic frames encoded by pydicom, with reference samples and
//! pydicom's encoding of every reference. See `fixtures/dicom_rle/README.md`.

use j2k_tilecodec::{DicomRleLayout, DicomRlePlanarConfiguration};

const CASES_TSV: &str = include_str!("../fixtures/dicom_rle/cases.tsv");
const DATA: &[u8] = include_bytes!("../fixtures/dicom_rle/data.bin");

/// One RLE frame with its reference samples.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Case {
    pub(crate) name: &'static str,
    /// `<file>#<frame>` for frames from pydicom's test files, `synthetic`
    /// for frames pydicom encoded.
    pub(crate) source: &'static str,
    pub(crate) rows: u16,
    pub(crate) columns: u16,
    pub(crate) samples: u16,
    pub(crate) bits: u16,
    /// The RLE frame as its encoder wrote it.
    pub(crate) rle: &'static [u8],
    /// Native little-endian samples, color-by-pixel.
    reference: &'static [u8],
    /// pydicom's encoding of the reference.
    pub(crate) pydicom_rle: &'static [u8],
}

impl Case {
    pub(crate) fn layout(&self, planar: DicomRlePlanarConfiguration) -> DicomRleLayout {
        DicomRleLayout::new(self.rows, self.columns, self.samples, self.bits, planar)
            .unwrap_or_else(|error| panic!("{}: layout: {error}", self.name))
    }

    /// The reference samples arranged in `planar` layout.
    pub(crate) fn reference(&self, planar: DicomRlePlanarConfiguration) -> Vec<u8> {
        match planar {
            DicomRlePlanarConfiguration::Interleaved => Vec::from(self.reference),
            DicomRlePlanarConfiguration::Planar => {
                let bytes = usize::from(self.bits / 8);
                let samples = usize::from(self.samples);
                let pixel_len = bytes * samples;
                (0..samples)
                    .flat_map(|sample| {
                        self.reference
                            .chunks_exact(pixel_len)
                            .flat_map(move |pixel| &pixel[sample * bytes..(sample + 1) * bytes])
                    })
                    .copied()
                    .collect()
            }
        }
    }
}

fn slice(field: &str) -> &'static [u8] {
    let (offset, len) = field
        .split_once(':')
        .expect("DICOM RLE fixture slices are offset:length");
    let offset: usize = offset.parse().expect("DICOM RLE fixture offset");
    let len: usize = len.parse().expect("DICOM RLE fixture length");
    &DATA[offset..offset + len]
}

/// Largest reference that Miri runs decode, to keep its run time
/// reasonable: every synthetic and `rtdose_rle` frame.
const MIRI_MAX_REFERENCE_LEN: usize = 2048;

/// Every case in the committed matrix, in generation order. Under Miri, only
/// the cases whose references are at most [`MIRI_MAX_REFERENCE_LEN`] bytes.
pub(crate) fn cases() -> Vec<Case> {
    CASES_TSV
        .lines()
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(|line| {
            let fields: Vec<&'static str> = line.split('\t').collect();
            assert_eq!(fields.len(), 9, "DICOM RLE fixture row: {line}");
            let case = Case {
                name: fields[0],
                source: fields[1],
                rows: fields[2].parse().expect("rows"),
                columns: fields[3].parse().expect("columns"),
                samples: fields[4].parse().expect("samples"),
                bits: fields[5].parse().expect("bits"),
                rle: slice(fields[6]),
                reference: slice(fields[7]),
                pydicom_rle: slice(fields[8]),
            };
            let expected = usize::from(case.rows)
                * usize::from(case.columns)
                * usize::from(case.samples)
                * usize::from(case.bits / 8);
            assert_eq!(case.reference.len(), expected, "{}: reference", case.name);
            case
        })
        .filter(|case| !cfg!(miri) || case.reference.len() <= MIRI_MAX_REFERENCE_LEN)
        .collect()
}

/// Test-only reading of an RLE header: the segment byte ranges of `frame`.
pub(crate) fn segment_ranges(frame: &[u8]) -> Vec<std::ops::Range<usize>> {
    let word = |index: usize| {
        let bytes = frame[index * 4..index * 4 + 4]
            .try_into()
            .expect("four header bytes");
        usize::try_from(u32::from_le_bytes(bytes)).expect("offset fits usize")
    };
    let count = word(0);
    (0..count)
        .map(|segment| {
            let end = if segment + 1 < count {
                word(segment + 2)
            } else {
                frame.len()
            };
            word(segment + 1)..end
        })
        .collect()
}

/// An RLE frame with the given segments, offsets in order and unused
/// offsets zero.
pub(crate) fn frame(segments: &[&[u8]]) -> Vec<u8> {
    let mut words = [0_u32; 16];
    words[0] = u32::try_from(segments.len()).expect("segment count");
    let mut offset = 64;
    for (slot, segment) in words[1..].iter_mut().zip(segments) {
        *slot = u32::try_from(offset).expect("offset");
        offset += segment.len();
    }
    frame_with_header(words, segments)
}

/// An RLE frame with an arbitrary header.
pub(crate) fn frame_with_header(words: [u32; 16], segments: &[&[u8]]) -> Vec<u8> {
    words
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .chain(segments.iter().flat_map(|segment| segment.iter().copied()))
        .collect()
}
