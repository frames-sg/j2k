// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::string::String;
use core::mem::size_of;

use super::{numeric_retained_bytes, NumericSum};
use crate::{ProfileField, ProfileSummary};

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

#[test]
fn existing_row_accepts_new_numeric_fields_transactionally_and_sorts_them() {
    let mut summary = ProfileSummary::new([]).expect("empty aggregate summary");
    summary
        .record_u128("jpeg", "decode", "cpu", &[("elapsed_us", 3)])
        .expect("initial numeric row");

    summary
        .record_u128("jpeg", "decode", "cpu", &[("elapsed_us", 5), ("bytes", 10)])
        .expect("existing row grows by one metric");

    assert_eq!(
        summary.format_rows().expect("format grown row"),
        ["j2k_profile_summary codec=jpeg op=decode path=cpu count=2 bytes_sum=10 elapsed_us_sum=8 elapsed_us_avg=4"]
    );

    let fields = [
        ProfileField::label("route", "scalar").expect("label field"),
        ProfileField::metric_with_summary("attempts", 7, false).expect("unsummarized metric"),
        ProfileField::metric("output_bytes", 11).expect("summarized metric"),
    ];
    summary
        .record_fields("jpeg", "encode", "cpu", &fields)
        .expect("typed fields row");

    assert_eq!(
        summary.format_rows().expect("format typed row")[1],
        "j2k_profile_summary codec=jpeg op=encode path=cpu count=1 output_bytes_sum=11"
    );
}
