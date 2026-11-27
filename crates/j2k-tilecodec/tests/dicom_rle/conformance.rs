// SPDX-License-Identifier: MIT OR Apache-2.0

//! Decoding real and synthetic RLE frames, and encoding them.

use crate::fixtures::{cases, frame, segment_ranges};
use j2k_tilecodec::{DicomRleCodec, DicomRleLayout, DicomRlePlanarConfiguration};

const LAYOUTS: [DicomRlePlanarConfiguration; 2] = [
    DicomRlePlanarConfiguration::Interleaved,
    DicomRlePlanarConfiguration::Planar,
];

fn decode(input: &[u8], layout: DicomRleLayout) -> Vec<u8> {
    let mut out = std::iter::repeat_n(0xA5_u8, layout.frame_len()).collect::<Vec<_>>();
    DicomRleCodec::decode_frame(input, layout, &mut out).unwrap_or_else(|error| panic!("{error}"));
    out
}

fn encode(samples: &[u8], layout: DicomRleLayout) -> Vec<u8> {
    let mut out = Vec::new();
    DicomRleCodec::encode_frame(samples, layout, &mut out)
        .unwrap_or_else(|error| panic!("{error}"));
    out
}

#[test]
#[cfg_attr(miri, ignore = "Miri runs only the small fixtures")]
fn fixture_matrix_covers_the_supported_formats() {
    let cases = cases();
    assert_eq!(cases.len(), 33);
    let real = cases
        .iter()
        .filter(|case| case.source != "synthetic")
        .count();
    assert_eq!(real, 19, "frames from pydicom's RLE test files");

    let mut segment_counts = cases
        .iter()
        .map(|case| case.samples * case.bits / 8)
        .collect::<Vec<_>>();
    segment_counts.sort_unstable();
    segment_counts.dedup();
    assert_eq!(segment_counts, [1, 2, 3, 4, 6, 10, 12, 15]);
    for bits in [8, 16, 32] {
        assert!(cases.iter().any(|case| case.bits == bits), "{bits} bits");
    }
}

#[test]
fn every_fixture_decodes_to_its_reference_in_both_layouts() {
    for case in cases() {
        for planar in LAYOUTS {
            let layout = case.layout(planar);
            let expected = case.reference(planar);
            assert_eq!(
                decode(case.rle, layout),
                expected,
                "{} {planar:?}",
                case.name
            );
            assert_eq!(
                decode(case.pydicom_rle, layout),
                expected,
                "{} {planar:?}: pydicom encoding",
                case.name
            );
        }
    }
}

#[test]
fn lenient_decoding_reports_every_fixture_as_conformant() {
    for case in cases() {
        let layout = case.layout(DicomRlePlanarConfiguration::Interleaved);
        let mut out = std::iter::repeat_n(0_u8, layout.frame_len()).collect::<Vec<_>>();
        let report = DicomRleCodec::decode_frame_lenient(case.rle, layout, &mut out)
            .unwrap_or_else(|error| panic!("{}: {error}", case.name));
        assert!(report.is_conformant(), "{}: {report:?}", case.name);
        assert_eq!(
            out,
            case.reference(layout.planar_configuration()),
            "{}",
            case.name
        );
    }
}

#[test]
fn encoder_output_matches_pydicom_byte_for_byte() {
    for case in cases() {
        for planar in LAYOUTS {
            let layout = case.layout(planar);
            let encoded = encode(&case.reference(planar), layout);
            assert!(
                encoded == case.pydicom_rle,
                "{} {planar:?}: encoder output differs from pydicom's",
                case.name
            );
        }
    }
}

#[test]
#[cfg_attr(miri, ignore = "Miri runs only the small fixtures")]
fn real_frames_differ_from_pydicom_only_in_run_merging_and_padding() {
    // MR_small_RLE and the rtdose_rle frames differ from pydicom's encoding
    // of the same samples: their encoder (GDCM, for rtdose_rle) merges a
    // two-byte repeat between literal runs into the literal, as PS3.5 Annex
    // G.3.1 suggests, and the rtdose_rle frames leave odd-length segments
    // unpadded. Both forms are valid and decode to the reference (checked
    // above), and both keep runs within rows.
    let mut differing = Vec::new();
    let mut odd_original_segments = 0;
    for case in cases()
        .into_iter()
        .filter(|case| case.source != "synthetic")
    {
        for (frame, encoder) in [(case.rle, "original"), (case.pydicom_rle, "pydicom")] {
            for range in segment_ranges(frame) {
                let context = format!("{} {encoder} segment at {}", case.name, range.start);
                assert_runs_stay_within_rows(&frame[range], case.rows, case.columns, &context);
            }
        }
        if case.rle == case.pydicom_rle {
            continue;
        }
        differing.push(case.name);
        let original = segment_ranges(case.rle);
        let pydicom = segment_ranges(case.pydicom_rle);
        assert_eq!(original.len(), pydicom.len(), "{}", case.name);
        for (original, pydicom) in original.iter().zip(&pydicom) {
            assert!(original.len() <= pydicom.len(), "{}", case.name);
            assert_eq!(pydicom.len() % 2, 0, "{}: pydicom pads segments", case.name);
            odd_original_segments += original.len() % 2;
        }
    }
    assert_eq!(differing.len(), 16, "{differing:?}");
    assert!(differing
        .iter()
        .all(|name| name.starts_with("MR_small_RLE") || name.starts_with("rtdose_rle")));
    assert!(
        odd_original_segments > 0,
        "rtdose_rle leaves odd segments unpadded"
    );
}

#[test]
fn encoded_segments_are_even_and_runs_stay_within_rows() {
    for case in cases() {
        let layout = case.layout(DicomRlePlanarConfiguration::Interleaved);
        let encoded = encode(
            &case.reference(DicomRlePlanarConfiguration::Interleaved),
            layout,
        );
        let header = &encoded[..64];
        let unused = &header[4 * (1 + layout.segment_count())..];
        assert!(
            unused.iter().all(|&byte| byte == 0),
            "{}: unused offsets",
            case.name
        );

        for (index, range) in segment_ranges(&encoded).into_iter().enumerate() {
            let context = format!("{} segment {index}", case.name);
            assert_eq!(range.len() % 2, 0, "{context}: odd length");
            assert_runs_stay_within_rows(&encoded[range], case.rows, case.columns, &context);
        }
    }
}

/// Check that a segment decodes to exactly `rows` × `columns` bytes and that
/// every row boundary is also a run boundary.
fn assert_runs_stay_within_rows(segment: &[u8], rows: u16, columns: u16, context: &str) {
    let columns = usize::from(columns);
    let total = usize::from(rows) * columns;
    let run_ends = run_end_positions(segment);
    assert_eq!(run_ends.last(), Some(&total), "{context}");
    for boundary in (columns..total).step_by(columns) {
        assert!(
            run_ends.binary_search(&boundary).is_ok(),
            "{context}: a run crosses the row boundary at byte {boundary}"
        );
    }
}

/// Decoded positions at which each run of a segment ends, read with a
/// separate test-only `PackBits` header walk. A trailing zero pad byte is
/// skipped.
fn run_end_positions(segment: &[u8]) -> Vec<usize> {
    let mut ends = Vec::new();
    let (mut position, mut index) = (0_usize, 0_usize);
    while let Some(&header) = segment.get(index) {
        let header = i8::from_le_bytes([header]);
        match header {
            0..=127 if index + 1 == segment.len() && header == 0 => break,
            0..=127 => {
                let len = usize::try_from(header).expect("non-negative") + 1;
                position += len;
                index += 1 + len;
            }
            -127..=-1 => {
                position += usize::try_from(1 - i16::from(header)).expect("positive");
                index += 2;
            }
            -128 => index += 1,
        }
        ends.push(position);
    }
    ends
}

#[test]
fn segments_go_by_sample_then_most_significant_byte_first() {
    // Two 16-bit samples per pixel, two pixels. Segment k holds byte k of
    // the big-endian composite pixel code (PS3.5 Annex G.2).
    let rle = frame(&[
        &[1, 0x12, 0x13],
        &[1, 0x34, 0x35],
        &[1, 0x56, 0x57],
        &[1, 0x78, 0x79],
    ]);
    let interleaved =
        DicomRleLayout::new(1, 2, 2, 16, DicomRlePlanarConfiguration::Interleaved).expect("layout");
    assert_eq!(
        decode(&rle, interleaved),
        [0x34, 0x12, 0x78, 0x56, 0x35, 0x13, 0x79, 0x57]
    );
    let planar =
        DicomRleLayout::new(1, 2, 2, 16, DicomRlePlanarConfiguration::Planar).expect("layout");
    assert_eq!(
        decode(&rle, planar),
        [0x34, 0x12, 0x35, 0x13, 0x78, 0x56, 0x79, 0x57]
    );
}

#[test]
fn layout_reports_its_geometry() {
    let layout =
        DicomRleLayout::new(3, 5, 3, 32, DicomRlePlanarConfiguration::Planar).expect("layout");
    assert_eq!(
        (
            layout.rows(),
            layout.columns(),
            layout.samples_per_pixel(),
            layout.bits_allocated(),
            layout.planar_configuration(),
            layout.segment_count(),
            layout.frame_len(),
        ),
        (3, 5, 3, 32, DicomRlePlanarConfiguration::Planar, 12, 180)
    );
}

#[test]
fn encoder_reuses_and_replaces_the_output_buffer() {
    let layout =
        DicomRleLayout::new(1, 4, 1, 8, DicomRlePlanarConfiguration::Interleaved).expect("layout");
    let mut out = (0..200_u8).collect::<Vec<_>>();
    DicomRleCodec::encode_frame(&[9, 9, 9, 9], layout, &mut out).expect("encode");
    assert_eq!(out, frame(&[&[253, 9]]));
}
