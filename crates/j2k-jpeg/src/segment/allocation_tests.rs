// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec;

use super::{
    checked_abbreviated_output_len, table_normalization::for_each_normalized_segment,
    table_normalization::NormalizedSegment, DuplicateTablePolicy, JpegError,
};

#[test]
fn abbreviated_output_length_has_an_exact_shared_cap_boundary() {
    let cap = j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES;
    assert_eq!(
        checked_abbreviated_output_len(0, cap - 4).expect("exact output boundary"),
        cap
    );
    assert!(matches!(
        checked_abbreviated_output_len(0, cap - 3),
        Err(JpegError::MemoryCapExceeded { requested, cap: limit })
            if requested > limit && limit == cap
    ));
}

#[test]
fn normalized_segments_stay_borrowed_and_identical_tables_are_deduplicated() {
    let mut table = vec![0xff, 0xdb, 0x00, 67, 0x00];
    table.extend(core::iter::repeat_n(1, 64));
    let mut tables = vec![0xff, 0xd8];
    tables.extend_from_slice(&table);
    tables.extend_from_slice(&table);
    tables.extend_from_slice(&[0xff, 0xd9]);
    let mut accepted = 0usize;
    for_each_normalized_segment(&tables, DuplicateTablePolicy::AllowIdentical, |segment| {
        accepted += 1;
        let NormalizedSegment::Borrowed(bytes) = segment else {
            panic!("single retained table remains borrowed");
        };
        let table_start = tables.as_ptr() as usize;
        let segment_start = bytes.as_ptr() as usize;
        assert!((table_start..table_start + tables.len()).contains(&segment_start));
        Ok(())
    })
    .expect("borrowed normalized tables");
    assert_eq!(accepted, 1);
}
