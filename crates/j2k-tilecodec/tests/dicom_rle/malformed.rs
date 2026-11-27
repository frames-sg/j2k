// SPDX-License-Identifier: MIT OR Apache-2.0

//! Malformed and truncated frames, invalid layouts and mismatched buffers.

use crate::fixtures::{cases, frame, frame_with_header};
use j2k_core::{BufferError, CodecError};
use j2k_tilecodec::{
    DicomRleCodec, DicomRleDecodeReport, DicomRleLayout, DicomRlePlanarConfiguration,
    TileCodecError,
};

const INTERLEAVED: DicomRlePlanarConfiguration = DicomRlePlanarConfiguration::Interleaved;

fn layout(rows: u16, columns: u16, samples: u16, bits: u16) -> DicomRleLayout {
    DicomRleLayout::new(rows, columns, samples, bits, INTERLEAVED).expect("layout")
}

fn zeroed(len: usize) -> Vec<u8> {
    std::iter::repeat_n(0_u8, len).collect()
}

fn strict(input: &[u8], layout: DicomRleLayout) -> Result<Vec<u8>, TileCodecError> {
    let mut out = zeroed(layout.frame_len());
    DicomRleCodec::decode_frame(input, layout, &mut out).map(|()| out)
}

fn lenient(
    input: &[u8],
    layout: DicomRleLayout,
) -> Result<(Vec<u8>, DicomRleDecodeReport), TileCodecError> {
    let mut out = zeroed(layout.frame_len());
    DicomRleCodec::decode_frame_lenient(input, layout, &mut out).map(|report| (out, report))
}

fn assert_malformed(result: Result<impl std::fmt::Debug, TileCodecError>, context: &str) {
    match result {
        Err(TileCodecError::Malformed { .. }) => {}
        other => panic!("{context}: expected a malformed-input error, got {other:?}"),
    }
}

fn assert_truncated(result: Result<impl std::fmt::Debug, TileCodecError>, context: &str) {
    match result {
        Err(error) if error.is_truncated() => {}
        other => panic!("{context}: expected a truncated-input error, got {other:?}"),
    }
}

#[test]
#[cfg_attr(miri, ignore = "decodes every prefix of every fixture")]
fn every_truncated_prefix_is_an_error_unless_it_drops_only_padding() {
    // A prefix that cuts only padding or no-op codes from the last segment
    // still decodes exactly; any other cut is a truncated-input error, in
    // both modes.
    let mut padding_only_cuts = 0;
    for case in cases() {
        let layout = case.layout(INTERLEAVED);
        let reference = case.reference(INTERLEAVED);
        let encodings = if case.rle == case.pydicom_rle {
            &[case.rle][..]
        } else {
            &[case.rle, case.pydicom_rle][..]
        };
        for full in encodings {
            for cut in 0..full.len() {
                let prefix = &full[..cut];
                match strict(prefix, layout) {
                    Ok(out) => {
                        assert!(out == reference, "{} cut {cut}", case.name);
                        padding_only_cuts += 1;
                    }
                    Err(error) => {
                        assert!(error.is_truncated(), "{} cut {cut}: {error:?}", case.name);
                    }
                }
                match lenient(prefix, layout) {
                    Ok((out, _)) => {
                        assert!(out == reference, "{} cut {cut}: lenient", case.name);
                    }
                    Err(error) => {
                        assert!(error.is_truncated(), "{} cut {cut}: {error:?}", case.name);
                    }
                }
            }
        }
    }
    assert!(padding_only_cuts > 0, "no fixture ends with a padding byte");
}

#[test]
fn header_segment_count_must_be_1_to_15_and_match_the_layout() {
    let one = layout(1, 2, 1, 8);
    for count in [0_u32, 16, 255, u32::MAX] {
        let mut words = [0_u32; 16];
        words[0] = count;
        words[1] = 64;
        let input = frame_with_header(words, &[&[1, 7, 7]]);
        assert_malformed(strict(&input, one), &format!("count {count}"));
        assert_malformed(lenient(&input, one), &format!("count {count} lenient"));
    }

    // Two well-formed segments for a layout that needs one, and the reverse.
    let two_segments = frame(&[&[1, 7, 7], &[1, 8, 8]]);
    assert_malformed(strict(&two_segments, one), "two segments, one expected");
    let one_segment = frame(&[&[1, 7, 7, 1, 8, 8]]);
    assert_malformed(
        strict(&one_segment, layout(1, 2, 1, 16)),
        "one segment, two expected",
    );
    assert_malformed(
        strict(&one_segment, layout(1, 2, 2, 8)),
        "one segment, two samples",
    );
}

#[test]
fn header_offsets_must_start_at_64_and_increase() {
    let two = layout(1, 2, 1, 16);
    let body: &[&[u8]] = &[&[1, 7, 7], &[1, 8, 8]];
    let header = |first: u32, second: u32| {
        let mut words = [0_u32; 16];
        words[..3].copy_from_slice(&[2, first, second]);
        words
    };
    for (first, second) in [(0, 67), (60, 67), (68, 71), (64, 64), (64, 63), (64, 0)] {
        let input = frame_with_header(header(first, second), body);
        let context = format!("offsets {first}, {second}");
        assert_malformed(strict(&input, two), &context);
        assert_malformed(lenient(&input, two), &context);
    }
    assert_eq!(
        strict(&frame_with_header(header(64, 67), body), two).expect("valid offsets"),
        [8, 7, 8, 7]
    );
}

#[test]
fn offsets_at_or_past_the_end_of_the_frame_are_truncation() {
    let two = layout(1, 2, 1, 16);
    for second in [70_u32, 71, 1000, u32::MAX] {
        let mut words = [0_u32; 16];
        words[..3].copy_from_slice(&[2, 64, second]);
        let input = frame_with_header(words, &[&[1, 7, 7], &[1, 8, 8]]);
        let context = format!("second offset {second}");
        assert_truncated(strict(&input, two), &context);
        assert_truncated(lenient(&input, two), &context);
    }
    // A header with no segment data at all.
    let mut words = [0_u32; 16];
    words[..2].copy_from_slice(&[1, 64]);
    assert_truncated(
        strict(&frame_with_header(words, &[]), layout(1, 2, 1, 8)),
        "no data",
    );
    assert_truncated(strict(&[], layout(1, 2, 1, 8)), "empty input");
}

#[test]
fn nonzero_unused_offsets_fail_strict_and_are_reported_when_lenient() {
    let one = layout(1, 2, 1, 8);
    let mut input = frame(&[&[1, 7, 7]]);
    input[60] = 1;
    assert_malformed(strict(&input, one), "unused offset");
    let (out, report) = lenient(&input, one).expect("lenient");
    assert_eq!(out, [7, 7]);
    assert!(report.has_nonzero_unused_offsets());
    assert_eq!(report.overlong_segments(), 0);
    assert!(!report.is_conformant());
}

#[test]
fn runs_past_the_end_of_the_rows_fail_strict_and_are_reported_when_lenient() {
    let one = layout(2, 2, 1, 8);
    for (segment, expected) in [
        // A replicate run of five bytes for four.
        (&[252_u8, 3][..], [3, 3, 3, 3]),
        // A literal run of five bytes for four.
        (&[4, 1, 2, 3, 4, 5], [1, 2, 3, 4]),
        // A complete segment followed by another run.
        (&[255, 6, 255, 7, 0, 9], [6, 6, 7, 7]),
        // Trailing bytes that are not padding.
        (&[255, 6, 255, 7, 0, 0], [6, 6, 7, 7]),
        (&[255, 6, 255, 7, 0x81], [6, 6, 7, 7]),
        // A literal run cut off by the end of the segment after it has
        // filled the rows.
        (&[255, 6, 5, 7, 7], [6, 6, 7, 7]),
    ] {
        let context = format!("segment {segment:?}");
        let input = frame(&[segment]);
        assert_malformed_or_truncated(strict(&input, one), &context);
        let (out, report) =
            lenient(&input, one).unwrap_or_else(|error| panic!("{context}: {error}"));
        assert_eq!(out, expected, "{context}");
        assert_eq!(report.overlong_segments(), 1, "{context}");
        assert!(!report.has_nonzero_unused_offsets(), "{context}");
    }

    // The mask names the segments that needed it.
    let two = layout(1, 2, 1, 16);
    let input = frame(&[&[1, 1, 2], &[253, 9]]);
    let (out, report) = lenient(&input, two).expect("lenient");
    assert_eq!(out, [9, 1, 9, 2]);
    assert_eq!(report.overlong_segments(), 0b10);
}

fn assert_malformed_or_truncated(result: Result<Vec<u8>, TileCodecError>, context: &str) {
    match result {
        Err(TileCodecError::Malformed { .. }) => {}
        Err(error) if error.is_truncated() => {}
        other => panic!("{context}: expected an error, got {other:?}"),
    }
}

#[test]
fn padding_and_no_ops_after_the_rows_are_accepted() {
    let one = layout(1, 4, 1, 8);
    for segment in [
        &[253_u8, 5][..],
        &[253, 5, 0],
        &[253, 5, 0x80],
        &[253, 5, 0x80, 0x80, 0],
        &[0x80, 254, 5, 0x80, 0, 5],
    ] {
        let input = frame(&[segment]);
        assert_eq!(
            strict(&input, one).expect("strict"),
            [5, 5, 5, 5],
            "{segment:?}"
        );
        let (_, report) = lenient(&input, one).expect("lenient");
        assert!(report.is_conformant(), "{segment:?}");
    }
}

#[test]
fn segments_that_end_early_are_errors_in_both_modes() {
    let two = layout(1, 4, 1, 16);
    // The first segment holds three of four bytes: malformed, because the
    // next offset bounds it.
    let short_first = frame(&[&[2, 1, 2, 3], &[3, 1, 2, 3, 4]]);
    assert_malformed(strict(&short_first, two), "short first segment");
    assert_malformed(lenient(&short_first, two), "short first segment, lenient");
    // The last segment ends early: the frame is truncated.
    let short_last = frame(&[&[3, 1, 2, 3, 4], &[2, 1, 2, 3]]);
    assert_truncated(strict(&short_last, two), "short last segment");
    assert_truncated(lenient(&short_last, two), "short last segment, lenient");
    // A literal run cut short before the rows are full.
    let cut_literal = frame(&[&[3, 1, 2, 3, 4], &[3, 1, 2]]);
    assert_truncated(lenient(&cut_literal, two), "cut literal");
    // A replicate header with no byte to repeat.
    let cut_replicate = frame(&[&[3, 1, 2, 3, 4], &[1, 1, 2, 255]]);
    assert_truncated(lenient(&cut_replicate, two), "cut replicate");
}

#[test]
fn layouts_reject_unsupported_formats() {
    let unsupported = |result: Result<DicomRleLayout, TileCodecError>, context: &str| match result {
        Err(error @ TileCodecError::Unsupported(_)) => assert!(error.is_unsupported()),
        other => panic!("{context}: expected unsupported, got {other:?}"),
    };
    for bits in [0_u16, 1, 4, 12, 24, 64] {
        unsupported(
            DicomRleLayout::new(4, 4, 1, bits, INTERLEAVED),
            &format!("{bits} bits allocated"),
        );
    }
    unsupported(DicomRleLayout::new(0, 4, 1, 8, INTERLEAVED), "zero rows");
    unsupported(DicomRleLayout::new(4, 0, 1, 8, INTERLEAVED), "zero columns");
    unsupported(DicomRleLayout::new(4, 4, 0, 8, INTERLEAVED), "zero samples");
    for (samples, bits) in [(16_u16, 8_u16), (8, 16), (4, 32), (u16::MAX, 32)] {
        unsupported(
            DicomRleLayout::new(4, 4, samples, bits, INTERLEAVED),
            &format!("{samples} samples of {bits} bits"),
        );
    }
    for (samples, bits, segments) in [(15_u16, 8_u16, 15), (7, 16, 14), (3, 32, 12)] {
        let layout = DicomRleLayout::new(4, 4, samples, bits, INTERLEAVED).expect("layout");
        assert_eq!(layout.segment_count(), segments);
    }
    let largest = DicomRleLayout::new(u16::MAX, u16::MAX, 3, 32, INTERLEAVED);
    if let Ok(largest) = largest {
        let frame_len = u64::try_from(largest.frame_len()).expect("u64");
        assert_eq!(frame_len, 65_535_u64 * 65_535 * 12);
    } else {
        assert!(matches!(
            largest,
            Err(TileCodecError::Buffer(BufferError::SizeOverflow { .. }))
        ));
    }
}

#[test]
fn buffers_must_match_the_layout_exactly() {
    let one = layout(2, 2, 1, 8);
    let input = frame(&[&[253, 4]]);
    for len in [0, 3, 5, 64] {
        let mut out = zeroed(len);
        for result in [
            DicomRleCodec::decode_frame(&input, one, &mut out)
                .map(|()| DicomRleDecodeReport::default()),
            DicomRleCodec::decode_frame_lenient(&input, one, &mut out),
        ] {
            match result {
                Err(
                    error @ TileCodecError::LengthMismatch {
                        expected: 4,
                        actual,
                        ..
                    },
                ) if actual == len => assert!(error.is_buffer_error()),
                other => panic!("output of {len} bytes: {other:?}"),
            }
        }
        let mut encoded = Vec::new();
        match DicomRleCodec::encode_frame(&zeroed(len), one, &mut encoded) {
            Err(TileCodecError::LengthMismatch {
                expected: 4,
                actual,
                ..
            }) if actual == len => {}
            other => panic!("input of {len} bytes: {other:?}"),
        }
    }

    // A layout far larger than the input is rejected before anything is
    // decoded or allocated.
    let huge = layout(u16::MAX, u16::MAX, 1, 8);
    let mut out = zeroed(4);
    assert!(matches!(
        DicomRleCodec::decode_frame(&input, huge, &mut out),
        Err(TileCodecError::LengthMismatch { .. })
    ));
}

#[test]
#[cfg_attr(miri, ignore = "decodes many corrupted frames")]
fn corrupted_fixtures_never_panic_and_strict_success_implies_lenient_agreement() {
    for case in cases().into_iter().filter(|case| case.rle.len() <= 700) {
        let layout = case.layout(INTERLEAVED);
        let mut corrupted = Vec::from(case.rle);
        for index in 0..corrupted.len() {
            for flip in [0x01_u8, 0x80, 0xFF] {
                corrupted[index] ^= flip;
                let strict_result = strict(&corrupted, layout);
                let lenient_result = lenient(&corrupted, layout);
                if let Ok(out) = strict_result {
                    let (lenient_out, report) = lenient_result
                        .unwrap_or_else(|error| panic!("{} {index}: {error}", case.name));
                    assert_eq!(out, lenient_out, "{} byte {index}", case.name);
                    assert!(report.is_conformant(), "{} byte {index}", case.name);
                }
                corrupted[index] ^= flip;
            }
        }
    }
}
