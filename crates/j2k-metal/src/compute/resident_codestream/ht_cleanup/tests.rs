// SPDX-License-Identifier: MIT OR Apache-2.0

fn function_body<'a>(source: &'a str, name: &str) -> &'a str {
    source
        .split_once(name)
        .unwrap_or_else(|| panic!("missing {name}"))
        .1
        .split("\n#[cfg(")
        .next()
        .expect("HT cleanup function body")
}

#[test]
fn zero_fill_is_barriered_before_single_and_batched_ht_cleanup() {
    let source = include_str!("../ht_cleanup.rs");
    for name in ["fn dispatch_ht_cleanup(", "fn dispatch_ht_cleanup_batched("] {
        let body = function_body(source, name);
        let zero_fill = body
            .find("dispatch_zero_u32_buffer_in_encoder")
            .unwrap_or_else(|| panic!("missing zero-fill dispatch in {name}"));
        let barrier = body
            .find("encoder.memory_barrier_with_resources(&[decoded]);")
            .unwrap_or_else(|| panic!("missing decoded-buffer barrier in {name}"));
        let cleanup = body
            .find("encoder.set_compute_pipeline_state")
            .unwrap_or_else(|| panic!("missing cleanup pipeline in {name}"));
        assert!(zero_fill < barrier && barrier < cleanup, "{name}");
    }
}
