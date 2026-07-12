// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use alloc::{vec, vec::Vec};

fn dqt_definition(id: u8, value: u8) -> Vec<u8> {
    let mut definition = vec![id];
    definition.extend(core::iter::repeat_n(value, 64));
    definition
}

fn dqt_16_definition(id: u8, value: u16) -> Vec<u8> {
    let mut definition = vec![0x10 | id];
    for _ in 0..64 {
        definition.extend_from_slice(&value.to_be_bytes());
    }
    definition
}

fn dht_definition(class: u8, id: u8, symbol: u8) -> Vec<u8> {
    let mut definition = vec![(class << 4) | id, 1];
    definition.extend(core::iter::repeat_n(0, 15));
    definition.push(symbol);
    definition
}

fn marker_segment(marker: u8, payload: &[u8]) -> Vec<u8> {
    let mut segment = vec![0xff, marker];
    let length = u16::try_from(payload.len() + 2).expect("test marker length");
    segment.extend_from_slice(&length.to_be_bytes());
    segment.extend_from_slice(payload);
    segment
}

fn table_stream(segments: &[Vec<u8>]) -> Vec<u8> {
    let mut stream = vec![0xff, 0xd8];
    for segment in segments {
        stream.extend_from_slice(segment);
    }
    stream.extend_from_slice(&[0xff, 0xd9]);
    stream
}

fn normalized(input: &[u8], policy: DuplicateTablePolicy) -> Result<Vec<u8>, JpegError> {
    let mut output = Vec::new();
    for_each_normalized_segment(input, policy, |segment| segment.append_to(&mut output))?;
    Ok(output)
}

#[test]
fn multi_table_dqt_conflict_uses_the_later_identifier() {
    let mut combined = dqt_definition(0, 1);
    combined.extend_from_slice(&dqt_16_definition(1, 2));
    let first = marker_segment(0xdb, &combined);
    let second = marker_segment(0xdb, &dqt_16_definition(1, 3));
    let expected_offset = 2 + first.len() + 4;
    let stream = table_stream(&[first, second]);

    assert!(matches!(
        normalized(&stream, DuplicateTablePolicy::RejectConflicting),
        Err(JpegError::ConflictingDuplicateTable {
            table: TableKind::Quant,
            id: 1,
            offset,
        }) if offset == expected_offset
    ));
}

#[test]
fn multi_table_dht_distinguishes_dc_and_ac_definitions() {
    let mut combined = dht_definition(0, 0, 0x11);
    combined.extend_from_slice(&dht_definition(1, 0, 0x22));
    let first = marker_segment(0xc4, &combined);
    let second = marker_segment(0xc4, &dht_definition(1, 0, 0x33));
    let expected_offset = 2 + first.len() + 4;
    let stream = table_stream(&[first, second]);

    assert!(matches!(
        normalized(&stream, DuplicateTablePolicy::RejectConflicting),
        Err(JpegError::ConflictingDuplicateTable {
            table: TableKind::HuffmanAc,
            id: 0,
            offset,
        }) if offset == expected_offset
    ));
}

#[test]
fn duplicate_policies_preserve_or_coalesce_identical_definitions() {
    let segment = marker_segment(0xdb, &dqt_definition(0, 7));
    let stream = table_stream(&[segment.clone(), segment.clone()]);

    let mut preserved = segment.clone();
    preserved.extend_from_slice(&segment);
    assert_eq!(
        normalized(&stream, DuplicateTablePolicy::RejectConflicting).unwrap(),
        preserved
    );
    assert_eq!(
        normalized(&stream, DuplicateTablePolicy::AllowIdentical).unwrap(),
        segment
    );
}

#[test]
fn allow_identical_rebuilds_partially_deduplicated_multi_table_markers() {
    let quant0 = dqt_definition(0, 1);
    let quant1 = dqt_definition(1, 2);
    let first = marker_segment(0xdb, &quant0);
    let mut repeated_and_new = quant0.clone();
    repeated_and_new.extend_from_slice(&quant1);
    let second = marker_segment(0xdb, &repeated_and_new);
    let stream = table_stream(&[first.clone(), second]);

    let mut expected = first;
    expected.extend_from_slice(&marker_segment(0xdb, &quant1));
    assert_eq!(
        normalized(&stream, DuplicateTablePolicy::AllowIdentical).unwrap(),
        expected
    );

    let dc0 = dht_definition(0, 0, 0x11);
    let ac0 = dht_definition(1, 0, 0x22);
    let first = marker_segment(0xc4, &dc0);
    let mut repeated_and_new = dc0.clone();
    repeated_and_new.extend_from_slice(&ac0);
    let second = marker_segment(0xc4, &repeated_and_new);
    let stream = table_stream(&[first.clone(), second]);

    let mut expected = first;
    expected.extend_from_slice(&marker_segment(0xc4, &ac0));
    assert_eq!(
        normalized(&stream, DuplicateTablePolicy::AllowIdentical).unwrap(),
        expected
    );
}

#[test]
fn malformed_or_truncated_dqt_definitions_fail_closed() {
    let empty = table_stream(&[marker_segment(0xdb, &[])]);
    assert!(matches!(
        normalized(&empty, DuplicateTablePolicy::RejectConflicting),
        Err(JpegError::InvalidSegmentLength { marker: 0xdb, .. })
    ));

    let invalid_id = table_stream(&[marker_segment(0xdb, &dqt_definition(4, 1))]);
    assert!(matches!(
        normalized(&invalid_id, DuplicateTablePolicy::RejectConflicting),
        Err(JpegError::InvalidSegmentLength { marker: 0xdb, .. })
    ));

    let mut invalid_precision = dqt_definition(0, 1);
    invalid_precision[0] = 0x20;
    let invalid_precision = table_stream(&[marker_segment(0xdb, &invalid_precision)]);
    assert!(matches!(
        normalized(&invalid_precision, DuplicateTablePolicy::RejectConflicting),
        Err(JpegError::UnsupportedBitDepth { depth: 2 })
    ));

    let mut truncated_later = dqt_definition(0, 1);
    truncated_later.extend_from_slice(&[1, 1, 1]);
    let truncated = table_stream(&[marker_segment(0xdb, &truncated_later)]);
    assert!(matches!(
        normalized(&truncated, DuplicateTablePolicy::RejectConflicting),
        Err(JpegError::Truncated { .. })
    ));
}

#[test]
fn malformed_or_truncated_dht_definitions_fail_closed() {
    let invalid_class = table_stream(&[marker_segment(0xc4, &dht_definition(2, 0, 1))]);
    assert!(matches!(
        normalized(&invalid_class, DuplicateTablePolicy::RejectConflicting),
        Err(JpegError::InvalidSegmentLength { marker: 0xc4, .. })
    ));

    let invalid_id = table_stream(&[marker_segment(0xc4, &dht_definition(0, 4, 1))]);
    assert!(matches!(
        normalized(&invalid_id, DuplicateTablePolicy::RejectConflicting),
        Err(JpegError::InvalidSegmentLength { marker: 0xc4, .. })
    ));

    let mut truncated_later = dht_definition(0, 0, 1);
    truncated_later.extend_from_slice(&[0x10, 1, 0, 0]);
    let truncated = table_stream(&[marker_segment(0xc4, &truncated_later)]);
    assert!(matches!(
        normalized(&truncated, DuplicateTablePolicy::RejectConflicting),
        Err(JpegError::Truncated { .. })
    ));
}

#[test]
fn table_stream_without_duplicates_preserves_marker_bytes() {
    let mut quant = dqt_definition(0, 1);
    quant.extend_from_slice(&dqt_definition(1, 2));
    let mut huffman = dht_definition(0, 0, 0x11);
    huffman.extend_from_slice(&dht_definition(1, 0, 0x22));
    let segments = [
        marker_segment(0xe1, &[1, 2, 3]),
        marker_segment(0xdb, &quant),
        marker_segment(0xc4, &huffman),
        marker_segment(0xdd, &[0, 8]),
    ];
    let stream = table_stream(&segments);
    let expected = segments.concat();

    for policy in [
        DuplicateTablePolicy::AllowIdentical,
        DuplicateTablePolicy::RejectConflicting,
    ] {
        assert_eq!(normalized(&stream, policy).unwrap(), expected);
    }
}
