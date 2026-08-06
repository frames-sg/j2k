// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec;

use super::*;

#[test]
fn tile_part_data_continues_across_marker_segments() {
    let first = [0, 0, 0, 3, 0xaa];
    let second = [0xbb, 0xcc, 0, 0, 0, 1, 0xdd];
    let markers = vec![
        PpmMarkerData {
            sequence_idx: 0,
            data: &first,
        },
        PpmMarkerData {
            sequence_idx: 1,
            data: &second,
        },
    ];
    let mut budget = HeaderMarkerBudget::default();
    budget
        .account_capacity::<PpmMarkerData<'_>>(markers.capacity())
        .expect("marker allocation fits");

    let packets = try_flatten_ppm_packets(markers, &mut budget).expect("valid continued PPM");

    assert_eq!(packets.len(), 3);
    assert_eq!(packets[0].data, [0xaa]);
    assert!(!packets[0].ends_tile_part);
    assert_eq!(packets[1].data, [0xbb, 0xcc]);
    assert!(packets[1].ends_tile_part);
    assert_eq!(packets[2].data, [0xdd]);
    assert!(packets[2].ends_tile_part);
}

#[test]
fn marker_sequence_must_be_contiguous() {
    let payload = [0, 0, 0, 1, 0xaa];
    let markers = vec![PpmMarkerData {
        sequence_idx: 1,
        data: &payload,
    }];
    let mut budget = HeaderMarkerBudget::default();
    budget
        .account_capacity::<PpmMarkerData<'_>>(markers.capacity())
        .expect("marker allocation fits");

    assert!(try_flatten_ppm_packets(markers, &mut budget).is_err());
}
