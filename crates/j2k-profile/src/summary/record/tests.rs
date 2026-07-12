// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::string::String;
use core::mem::size_of;

use super::{numeric_retained_bytes, NumericSum};

#[test]
fn numeric_retained_bytes_counts_outer_and_nested_allocator_capacities() {
    let mut key = String::new();
    key.try_reserve_exact(7).expect("test key capacity");
    key.push_str("time");
    let values = [NumericSum { key, sum: 11 }];
    let outer_capacity = 3;

    assert_eq!(
        numeric_retained_bytes(&values, outer_capacity).expect("bounded retained bytes"),
        outer_capacity * size_of::<NumericSum>() + values[0].key.capacity()
    );
    assert_eq!(values[0].sum, 11);
}
