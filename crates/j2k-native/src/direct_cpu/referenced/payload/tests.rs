// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use crate::J2kCodestreamRange;

#[test]
fn cursor_concatenates_fragmented_refinement_records_in_order() {
    let input = [0x10, 0x11, 0xff, 0xff, 0x20, 0xff, 0xff, 0x21, 0x22];
    let payloads = [
        HtCodeBlockPayloadRanges {
            cleanup: J2kCodestreamRange {
                offset: 0,
                length: 2,
            },
            refinement: Some(J2kCodestreamRange {
                offset: 4,
                length: 1,
            }),
        },
        HtCodeBlockPayloadRanges {
            cleanup: J2kCodestreamRange {
                offset: 7,
                length: 0,
            },
            refinement: Some(J2kCodestreamRange {
                offset: 7,
                length: 2,
            }),
        },
    ];
    let mut combined = Vec::new();
    let mut cursor = ReferencedPayloadCursor::new(&input, &payloads);

    assert_eq!(
        cursor.next_data(2, 3, &mut combined).unwrap(),
        [0x10, 0x11, 0x20, 0x21, 0x22]
    );
    cursor.ensure_exhausted().unwrap();
}
