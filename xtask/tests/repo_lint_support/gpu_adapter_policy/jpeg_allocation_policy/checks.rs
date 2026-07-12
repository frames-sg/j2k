// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::super::rust_function_policy::FunctionCalls;
use super::JpegAllocationSources;

mod adapter_checks;
mod checkpoint_checks;
mod packet_checks;
mod structure_checks;

fn calls(source_name: &str, source: &str, function_name: &str) -> FunctionCalls {
    FunctionCalls::parse(source_name, source, function_name)
}

fn assert_ordered(label: &str, source: &str, patterns: &[&str]) {
    let mut remainder = source;
    for pattern in patterns {
        let position = remainder
            .find(pattern)
            .unwrap_or_else(|| panic!("{label} is missing ordered pattern {pattern:?}"));
        remainder = &remainder[position + pattern.len()..];
    }
}

pub(super) fn assert_policy(sources: &JpegAllocationSources) {
    assert!(
        include_str!("checks/adapter_checks.rs").lines().count() < 75
            && include_str!("checks/checkpoint_checks.rs").lines().count() < 110
            && include_str!("checks/packet_checks.rs").lines().count() < 130
            && include_str!("checks/structure_checks.rs").lines().count() < 140,
        "JPEG allocation policy leaves must stay focused"
    );
    structure_checks::assert_policy(sources);
    checkpoint_checks::assert_policy(sources);
    packet_checks::assert_policy(sources);
    adapter_checks::assert_policy(sources);
}
