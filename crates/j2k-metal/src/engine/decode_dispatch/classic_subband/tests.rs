// SPDX-License-Identifier: MIT OR Apache-2.0

fn function_body<'a>(source: &'a str, name: &str) -> &'a str {
    source
        .split_once(name)
        .unwrap_or_else(|| panic!("missing {name}"))
        .1
        .split("\n#[cfg(")
        .next()
        .expect("classic sub-band function body")
}

#[test]
fn prepared_classic_zero_fill_is_barriered_before_cleanup_dispatch() {
    let source = include_str!("../classic_subband.rs");
    for name in [
        "fn encode_prepared_classic_sub_band_to_buffer_in_encoder",
        "fn encode_prepared_classic_sub_band_group_to_buffer_in_encoder",
    ] {
        let body = function_body(source, name);
        let zero_fill = body
            .find("dispatch_zero_u32_buffer_in_encoder")
            .unwrap_or_else(|| panic!("missing zero-fill dispatch in {name}"));
        let barrier = body
            .find("encoder.memory_barrier_with_resources(&[output]);")
            .unwrap_or_else(|| panic!("missing zero-fill resource barrier in {name}"));
        let cleanup = body
            .find("dispatch_classic_cleanup_batched_in_encoder(")
            .unwrap_or_else(|| panic!("missing cleanup dispatch in {name}"));
        assert!(zero_fill < barrier && barrier < cleanup, "{name}");
    }
}
