// SPDX-License-Identifier: MIT OR Apache-2.0

//! Repeated native decode must retain, account, and reset component owners.

use std::fs;

use super::rust_function_policy::FunctionCalls;
use super::{assert_pattern_checks, repo_root, PatternCheck};

fn read(relative: &str) -> String {
    fs::read_to_string(repo_root().join(relative))
        .unwrap_or_else(|error| panic!("read {relative}: {error}"))
}

#[test]
fn decoder_context_reuse_is_nonvacuous_and_capacity_accounted() {
    let decode = read("crates/j2k-native/src/j2c/decode.rs");
    let reuse = read("crates/j2k-native/src/j2c/decode/reuse.rs");
    let image = read("crates/j2k-native/src/image.rs");
    let native = read("crates/j2k-native/src/image/output_api.rs");
    let tests = read("crates/j2k-native/src/tests.rs");

    FunctionCalls::parse("native decode", &decode, "decode").assert_ordered(
        "retained component accounting before parse and reset",
        &["prepare_reused_decode_baseline", "tile::parse", "reset"],
    );

    assert_pattern_checks(&[
        PatternCheck::new("native retained-channel baseline", &reuse)
            .required(&[
                "retained_channel_bytes: usize",
                "release_tile_scratch_allocations",
                "storage.release_all_allocations",
                "checked_combined_context_bytes",
                "retained_channel_bytes_with_cap",
                "ContextCapacityBudget::from_live_bytes",
                "reset_channel_data",
                "try_reset_zeros",
                "try_resize_decode_elements",
                "include_capacity_overage",
            ])
            .forbidden(&["self.channel_data.clear()"]),
        PatternCheck::new("packed output preserves context owners", &image).forbidden(&[
            "decoder_context.tile_decode_context.channel_data.clear()",
            "mem::take(&mut decoder_context.tile_decode_context.channel_data)",
        ]),
        PatternCheck::new("native output preserves context owners", &native).forbidden(&[
            "decoder_context.tile_decode_context.channel_data.clear()",
            "mem::take(&mut decoder_context.tile_decode_context.channel_data)",
        ]),
        PatternCheck::new("nonvacuous context-reuse regressions", &tests).required(&[
            "decoder_context_reuses_component_owners_across_packed_and_component_outputs",
            "exact_integer_decoder_context_reuses_and_resets_i64_samples",
            "assert_eq!(context.tile_decode_context.channel_data.len(), 3)",
            "assert_eq!(channel_capacity_snapshot(&context), owners)",
        ]),
    ]);
}
