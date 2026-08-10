// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use j2k_native::{HtCodeBlockPayloadRanges, J2kCodestreamRange};

fn range(offset: usize, length: usize) -> J2kCodestreamRange {
    J2kCodestreamRange { offset, length }
}

#[test]
fn ht_payload_span_consumes_all_refinement_continuation_records() {
    let payloads = [
        HtCodeBlockPayloadRanges {
            cleanup: range(0, 2),
            refinement: Some(range(3, 1)),
        },
        HtCodeBlockPayloadRanges {
            cleanup: range(5, 0),
            refinement: Some(range(5, 2)),
        },
        HtCodeBlockPayloadRanges {
            cleanup: range(7, 1),
            refinement: None,
        },
    ];
    let mut cursor = 0;

    let fragmented = next_ht_payload_record_span(&payloads, &mut cursor, 2, 3).unwrap();
    let cleanup_only = next_ht_payload_record_span(&payloads, &mut cursor, 1, 0).unwrap();

    assert_eq!(fragmented.first_record, 0);
    assert_eq!(fragmented.record_count, 2);
    assert_eq!(cleanup_only.first_record, 2);
    assert_eq!(cleanup_only.record_count, 1);
    assert_eq!(cursor, payloads.len());
}

#[test]
fn malformed_ht_payload_continuation_does_not_advance_the_cursor() {
    let payloads = [
        HtCodeBlockPayloadRanges {
            cleanup: range(0, 2),
            refinement: Some(range(3, 1)),
        },
        HtCodeBlockPayloadRanges {
            cleanup: range(4, 1),
            refinement: Some(range(5, 2)),
        },
    ];
    let mut cursor = 0;

    assert!(next_ht_payload_record_span(&payloads, &mut cursor, 2, 3).is_err());
    assert_eq!(cursor, 0);
}
