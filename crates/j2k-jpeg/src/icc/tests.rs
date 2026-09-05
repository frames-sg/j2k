// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

fn jpeg() -> Vec<u8> {
    j2k_test_support::baseline_444_8x8_jpeg()
}

#[test]
fn multi_chunk_profile_round_trips() {
    let profile = (0..MAX_ICC_CHUNK_DATA_LEN + 17)
        .map(|index| u8::try_from(index % 251).expect("ICC fixture samples are below 251"))
        .collect::<Vec<_>>();
    let encoded = insert_icc_profile(&jpeg(), &profile).expect("insert ICC profile");

    assert_eq!(
        extract_icc_profile(&encoded).expect("extract ICC profile"),
        Some(profile)
    );
}

#[test]
fn set_replaces_icc_chunks_and_preserves_unrelated_app2() {
    let mut source = jpeg();
    let unrelated = [0xff, 0xe2, 0x00, 0x08, b'V', b'E', b'N', b'D', b'O', b'R'];
    source.splice(2..2, unrelated);
    let source = insert_icc_profile(&source, &[1, 2, 3, 4]).expect("insert original profile");

    let replaced = set_icc_profile(&source, &[8, 13, 21]).expect("replace profile");

    assert_eq!(
        extract_icc_profile(&replaced).expect("extract replacement"),
        Some(vec![8, 13, 21])
    );
    assert!(replaced
        .windows(unrelated.len())
        .any(|window| window == unrelated));
}

#[test]
fn extraction_rejects_incomplete_chunk_sequence() {
    let profile = vec![7; MAX_ICC_CHUNK_DATA_LEN + 1];
    let mut encoded = insert_icc_profile(&jpeg(), &profile).expect("insert two chunks");
    let first_signature = encoded
        .windows(ICC_SIGNATURE.len())
        .position(|window| window == ICC_SIGNATURE)
        .expect("ICC signature");
    encoded[first_signature + ICC_SIGNATURE.len() + 1] = 3;
    let second_signature = encoded[first_signature + ICC_SIGNATURE.len()..]
        .windows(ICC_SIGNATURE.len())
        .position(|window| window == ICC_SIGNATURE)
        .map(|offset| offset + first_signature + ICC_SIGNATURE.len())
        .expect("second ICC signature");
    encoded[second_signature + ICC_SIGNATURE.len() + 1] = 3;

    assert!(matches!(
        extract_icc_profile(&encoded),
        Err(IccProfileError::MissingChunk {
            sequence: 3,
            count: 3
        })
    ));
}
